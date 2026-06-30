"""Feature-engineering pipeline orchestration.

Purpose
-------
Compose the feature-engineering stages into one reproducible, config-driven
flow per dataset. All selectors are FIT ON TRAINING DATA ONLY and reused to
transform validation/test (no leakage):

    load processed -> variance -> correlation -> statistical selection
        -> optional PCA -> save transformed datasets, reports, artefacts

This is the single orchestration surface; scripts call into it rather than
re-implementing the sequence. Inputs are the Phase 2 processed parquet files;
processed raw data is never modified.

Outputs (namespaced per dataset):
    <features_out_dir>/<id>/{train,validation,test}.parquet
    <features_out_dir>/<id>/{feature_report,feature_metadata,
                             selected_features,removed_features}.json
    <artifacts_dir>/<id>/{feature_selector,pca}.joblib
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.features import reports
from src.features.correlation import fit_correlation_filter
from src.features.dimensionality import apply_pca, fit_pca
from src.features.metadata import FeatureMetadata, select_feature_columns, split_xy
from src.features.selection import FeatureSelector, select_features
from src.features.variance import fit_variance_threshold
from src.utils.config import load_dataset_config
from src.utils.io import read_parquet, save_artifact, write_json, write_parquet
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

_SPLIT_FILES = ("train", "validation", "test")


@dataclass
class FeaturePipelineResult:
    """Summary of a completed feature-engineering run for one dataset.

    Attributes
    ----------
    dataset_id:
        Processed dataset id.
    output_paths:
        Mapping of artefact name -> written path.
    n_original:
        Original feature count.
    n_retained:
        Retained feature (or PCA component) count.
    pca_enabled:
        Whether PCA was applied.
    """

    dataset_id: str
    output_paths: dict[str, Path]
    n_original: int
    n_retained: int
    pca_enabled: bool


def _resolve_seed(config: Mapping[str, Any], features_config: Mapping[str, Any]) -> int:
    """Resolve the seed: ``features.random_seed`` else ``project.seed``."""
    if features_config.get("random_seed") is not None:
        return int(features_config["random_seed"])
    return int(config.get("project", {}).get("seed", 42))


def _load_split(proc_dir: Path, name: str) -> "Any":
    """Read one processed split parquet file."""
    path = proc_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed split not found: {path}. Run preprocessing (Phase 2) first."
        )
    return read_parquet(path)


def _write_frame(frame: "Any", path: Path, io_config: Mapping[str, Any]) -> Path:
    """Persist a transformed frame in the configured format (parquet | csv)."""
    fmt = str(io_config.get("format", "parquet"))
    if fmt == "csv":
        target = path.with_suffix(".csv")
        ensure_dir(target.parent)
        frame.to_csv(target, index=False)
        return target
    compression = str(io_config.get("compression", "snappy"))
    return write_parquet(frame, path.with_suffix(".parquet"), compression=compression)


def _attach_label(features: "Any", y: "Any", label_column: str) -> "Any":
    """Reattach the (index-aligned) label column without duplicating features."""
    result = features.copy(deep=False)
    result[label_column] = y
    return result


def run_feature_pipeline(
    dataset_id: str,
    config: Mapping[str, Any],
    paths: Any,
) -> FeaturePipelineResult:
    """Run the full feature-engineering pipeline for a single dataset.

    Parameters
    ----------
    dataset_id:
        Registered dataset identifier.
    config:
        Effective merged configuration.
    paths:
        Resolved :class:`src.utils.paths.Paths`.

    Returns
    -------
    FeaturePipelineResult

    Raises
    ------
    ValueError
        If the dataset has no configured label column (unsupervised data is not
        yet supported by statistical selection).
    FileNotFoundError
        If processed splits are missing.
    """
    logger.info("Starting feature-engineering pipeline for dataset '%s'", dataset_id)
    features_cfg = config["features"]
    seed = _resolve_seed(config, features_cfg)

    dataset_config = load_dataset_config(dataset_id)
    label_column = dataset_config.get("label_column")
    if not label_column:
        raise ValueError(
            f"[{dataset_id}] no label column configured; supervised feature "
            f"selection requires a target."
        )

    # 1. Load processed splits.
    proc_dir = Path(paths.processed_out_dir) / dataset_id
    train = _load_split(proc_dir, "train")
    val = _load_split(proc_dir, "validation")
    test = _load_split(proc_dir, "test")

    x_train, y_train = split_xy(train, label_column)
    x_val, _ = split_xy(val, label_column)
    x_test, _ = split_xy(test, label_column)

    # Exclude provenance/metadata and any non-numeric columns so that variance,
    # correlation and statistical selection only ever receive numeric features
    # (root cause of "could not convert string to float: 'train'").
    extra_exclude = features_cfg.get("exclude_columns", []) or []
    x_train, excluded = select_feature_columns(x_train, extra_exclude)
    if excluded["provenance"]:
        logger.warning(
            "[%s] excluding provenance/metadata columns from features: %s",
            dataset_id, excluded["provenance"],
        )
    if excluded["non_numeric"]:
        logger.warning(
            "[%s] excluding non-numeric feature columns: %s",
            dataset_id, excluded["non_numeric"],
        )
    # Apply the identical column reduction to validation/test (no leakage).
    feature_columns = list(x_train.columns)
    x_val = x_val[feature_columns]
    x_test = x_test[feature_columns]
    original_features = [str(c) for c in feature_columns]

    # 2. Variance filtering (fit on train).
    variance_res = None
    current = x_train
    if features_cfg.get("variance", {}).get("enabled", True):
        variance_res = fit_variance_threshold(
            current, float(features_cfg["variance"].get("threshold", 0.0))
        )
        current = current[variance_res.kept]

    # 3. Correlation filtering (fit on train).
    correlation_res = None
    if features_cfg.get("correlation", {}).get("enabled", True):
        corr_cfg = features_cfg["correlation"]
        correlation_res = fit_correlation_filter(
            current,
            float(corr_cfg.get("threshold", 0.95)),
            str(corr_cfg.get("method", "pearson")),
        )
        current = current[correlation_res.kept]

    # 4. Statistical / model-based selection (fit on train).
    selection_res = None
    selected_columns = [str(c) for c in current.columns]
    if features_cfg.get("selection", {}).get("enabled", True):
        selection_res = select_features(current, y_train, features_cfg, seed)
        selected_columns = selection_res.selected

    selector = FeatureSelector(
        selected_features=selected_columns,
        method=selection_res.method if selection_res else "none",
    )

    # Apply the single selector to every split (reproduces all column filtering).
    x_train_sel = selector.transform(x_train)
    x_val_sel = selector.transform(x_val)
    x_test_sel = selector.transform(x_test)

    # 5. Optional PCA (fit on train).
    pca = fit_pca(x_train_sel, features_cfg.get("dimensionality", {}).get("pca", {}), seed)
    if pca is not None:
        x_train_final = apply_pca(pca, x_train_sel)
        x_val_final = apply_pca(pca, x_val_sel)
        x_test_final = apply_pca(pca, x_test_sel)
        retained_features = list(pca.component_names)
    else:
        x_train_final, x_val_final, x_test_final = x_train_sel, x_val_sel, x_test_sel
        retained_features = list(selected_columns)

    # 6. Persist everything.
    output_paths = _persist(
        dataset_id=dataset_id,
        paths=paths,
        features_cfg=features_cfg,
        label_column=label_column,
        original_features=original_features,
        retained_features=retained_features,
        excluded_columns=excluded,
        variance_res=variance_res,
        correlation_res=correlation_res,
        selection_res=selection_res,
        pca=pca,
        selector=selector,
        frames={
            "train": _attach_label(x_train_final, y_train, label_column),
            "validation": _attach_label(x_val_final, val[label_column], label_column),
            "test": _attach_label(x_test_final, test[label_column], label_column),
        },
    )

    logger.info(
        "Feature engineering complete for '%s': %d -> %d feature(s)%s.",
        dataset_id, len(original_features), len(retained_features),
        " (PCA)" if pca else "",
    )
    return FeaturePipelineResult(
        dataset_id=dataset_id,
        output_paths=output_paths,
        n_original=len(original_features),
        n_retained=len(retained_features),
        pca_enabled=pca is not None,
    )


def _persist(**kw: Any) -> dict[str, Path]:
    """Write transformed datasets, reports and artefacts; return their paths."""
    dataset_id: str = kw["dataset_id"]
    paths = kw["paths"]
    features_cfg: Mapping[str, Any] = kw["features_cfg"]

    feat_dir = ensure_dir(Path(paths.features_out_dir) / dataset_id)
    art_dir = ensure_dir(Path(paths.artifacts_dir) / dataset_id)
    io_cfg = features_cfg.get("io", {})

    out: dict[str, Path] = {}

    # Transformed datasets.
    for name, frame in kw["frames"].items():
        out[name] = _write_frame(frame, feat_dir / name, io_cfg)

    # Reports.
    if features_cfg.get("reports", {}).get("enabled", True):
        metadata = FeatureMetadata(
            dataset_id=dataset_id,
            label_column=kw["label_column"],
            original_features=kw["original_features"],
            retained_features=kw["retained_features"],
            pca_components=kw["pca"].n_components if kw["pca"] else None,
        )
        out["feature_report"] = write_json(
            reports.build_feature_report(
                dataset_id,
                kw["original_features"],
                kw["variance_res"],
                kw["correlation_res"],
                kw["selection_res"],
                kw["pca"],
                kw["retained_features"],
                excluded_columns=kw.get("excluded_columns"),
            ),
            feat_dir / "feature_report.json",
        )
        out["feature_metadata"] = write_json(
            reports.build_feature_metadata(metadata), feat_dir / "feature_metadata.json"
        )
        out["selected_features"] = write_json(
            reports.build_selected_features(kw["selection_res"], kw["retained_features"]),
            feat_dir / "selected_features.json",
        )
        out["removed_features"] = write_json(
            reports.build_removed_features(kw["variance_res"], kw["correlation_res"]),
            feat_dir / "removed_features.json",
        )

    # Artefacts.
    if features_cfg.get("artifacts", {}).get("save", True):
        out["feature_selector"] = save_artifact(
            kw["selector"], art_dir / "feature_selector.joblib"
        )
        if kw["pca"] is not None:
            out["pca"] = save_artifact(kw["pca"], art_dir / "pca.joblib")

    return out


def run_all_features(config: Mapping[str, Any], paths: Any) -> dict[str, FeaturePipelineResult]:
    """Run feature engineering for every dataset in ``data.active_datasets``.

    Datasets without processed splits (e.g. SNMP) are skipped with a warning
    rather than aborting the batch.

    Returns
    -------
    dict
        dataset_id -> :class:`FeaturePipelineResult` (only successful datasets).
    """
    results: dict[str, FeaturePipelineResult] = {}
    for dataset_id in config.get("data", {}).get("active_datasets", []):
        try:
            results[dataset_id] = run_feature_pipeline(dataset_id, config, paths)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.warning("[%s] feature engineering skipped: %s", dataset_id, exc)
    return results

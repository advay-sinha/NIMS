"""End-to-end data preprocessing pipeline orchestration.

Purpose
-------
Compose the Phase 2 preprocessing stages into one reproducible flow per
dataset, honouring the documented order so that encoders and scalers never see
validation/test data (no leakage):

    load_raw -> validate -> clean -> SPLIT -> fit(encode, scale) on train
             -> apply to val/test -> persist splits, artefacts, reports, manifest

This is the single orchestration surface; scripts call into it rather than
re-implementing the sequence (CLAUDE.md > Repository Principles: no duplicated
logic). Raw data is never mutated.

Memory (Phase 2.1)
------------------
The pipeline is engineered for multi-million-row datasets on 8 GB machines:
stages avoid full-frame duplication (Copy-on-Write shallow copies, column-level
replacement), release intermediates with optional garbage collection between
stages, downcast numeric outputs, and record a per-stage memory profile. These
affect resource usage only — never the preprocessing logic or report formats.

Outputs (namespaced per dataset to avoid overwriting between datasets):
    <preprocessing_dir>/<id>/{cleaning,encoding,scaling,split}_report.json
    <preprocessing_dir>/<id>/preprocessing_manifest.json
    <preprocessing_dir>/<id>/memory_profile.json
    <processed_out_dir>/<id>/{train,validation,test}.parquet
    <artifacts_dir>/<id>/{encoder,scaler,label_encoder}.joblib
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.data.base import DatasetSplit
from src.data.cleaning import clean_dataset
from src.data.encoding import (
    EncodingReport,
    FittedLabelEncoder,
    apply_encoder,
    apply_label_encoder,
    fit_encoder,
    fit_label_encoder,
)
from src.data.fingerprint import build_fingerprint
from src.data.registry import get_loader_cls
from src.data.scaling import ScalingReport, apply_scaler, fit_scaler
from src.data.splitting import build_split_report, train_val_test_split
from src.data.validation import build_report
from src.utils.config import load_dataset_config
from src.utils.io import save_artifact, write_json, write_parquet
from src.utils.memory import MemoryReport, collect_garbage
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run for one dataset.

    Attributes
    ----------
    dataset_id:
        Processed dataset id.
    split:
        The produced :class:`DatasetSplit`.
    output_paths:
        Mapping of artefact name -> written path.
    validation_passed:
        Whether validation produced no errors.
    """

    dataset_id: str
    split: DatasetSplit | None
    output_paths: dict[str, Path]
    validation_passed: bool


def _resolve_seed(config: Mapping[str, Any], split_config: Mapping[str, Any]) -> int:
    """Resolve the split seed: ``data.split.seed`` else ``project.seed``."""
    if split_config.get("seed") is not None:
        return int(split_config["seed"])
    return int(config.get("project", {}).get("seed", 42))


def _numeric_columns(frame: "Any") -> list[str]:
    """Return numeric column names of ``frame`` (continuous features)."""
    import numpy as np

    return list(frame.select_dtypes(include=[np.number]).columns)


def downcast_numeric(frame: "Any", enabled: bool = True) -> tuple["Any", dict[str, str]]:
    """Downcast numeric columns to the smallest safe dtype.

    ``float64 -> float32`` and ``int64 -> int32/int16/...`` where values fit.
    Boolean and non-numeric (categorical) columns are never touched. Uses a
    Copy-on-Write shallow copy so only changed column blocks are reallocated.

    Parameters
    ----------
    frame:
        Input DataFrame.
    enabled:
        When ``False``, returns ``frame`` unchanged with an empty dtype map.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, str]]
        The (possibly downcast) frame and a ``{column: new_dtype}`` map of the
        columns whose dtype changed.
    """
    if not enabled:
        return frame, {}
    import pandas as pd

    result = frame.copy(deep=False)
    changed: dict[str, str] = {}
    for col in frame.columns:
        dtype = frame[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            continue
        if pd.api.types.is_float_dtype(dtype):
            new = pd.to_numeric(frame[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dtype):
            new = pd.to_numeric(frame[col], downcast="integer")
        else:
            continue
        if new.dtype != dtype:
            result[col] = new
            changed[str(col)] = str(new.dtype)
    return result, changed


def _assemble(features: "Any", y_encoded: "Any | None", label_column: str | None) -> "Any":
    """Combine scaled features with the encoded label into one DataFrame.

    Uses a Copy-on-Write shallow copy plus a single column assignment rather
    than ``concat`` so the (large) feature column blocks are never duplicated.
    """
    if y_encoded is None or label_column is None:
        return features
    result = features.copy(deep=False)
    result[label_column] = y_encoded  # positional ndarray assign; CoW adds 1 block
    return result


def _write_frame(frame: "Any", path: Path, io_config: Mapping[str, Any]) -> Path:
    """Persist a processed frame in the configured format (parquet | csv)."""
    fmt = str(io_config.get("processed_format", "parquet"))
    if fmt == "csv":
        target = path.with_suffix(".csv")
        ensure_dir(target.parent)
        frame.to_csv(target, index=False)
        return target
    compression = str(io_config.get("compression", "snappy"))
    return write_parquet(frame, path.with_suffix(".parquet"), compression=compression)


def run_pipeline(
    dataset_id: str,
    config: Mapping[str, Any],
    paths: Any,
) -> PipelineResult:
    """Run the full preprocessing pipeline for a single dataset.

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
    PipelineResult

    Raises
    ------
    RuntimeError
        If validation fails with errors (fail-fast; nothing is persisted).
    """
    logger.info("Starting preprocessing pipeline for dataset '%s'", dataset_id)
    data_cfg = config["data"]
    opt_cfg = data_cfg.get("optimization", {})
    collect = bool(opt_cfg.get("collect_garbage", True))
    mem = MemoryReport(enabled=bool(opt_cfg.get("profile_memory", True)))

    dataset_config = load_dataset_config(dataset_id)
    loader = get_loader_cls(dataset_id)(dataset_config, paths)
    raw = loader.load_raw()

    # 1. Validate; fail fast on schema errors before doing any work.
    report = build_report(raw.frame, dataset_config, label_column=raw.label_column)
    if not report.schema_passed:
        raise RuntimeError(
            f"[{dataset_id}] validation failed with errors; refusing to preprocess."
        )
    label_column = raw.label_column

    # 2. Clean the full frame (keeps features/label row-aligned).
    with mem.stage("cleaning"):
        cleaned, cleaning_report = clean_dataset(raw.frame, data_cfg["cleaning"])
    if label_column is not None and label_column not in cleaned.columns:
        raise RuntimeError(
            f"[{dataset_id}] label column '{label_column}' was removed during cleaning."
        )
    collect_garbage(collect)

    # 3. Separate features / target.
    if label_column is not None:
        y = cleaned[label_column]
        x = cleaned.drop(columns=[label_column])
    else:
        y, x = None, cleaned
    has_label = y is not None

    categorical = [c for c in raw.categorical_columns if c in x.columns]
    numeric_cols = _numeric_columns(x)

    # 4. Split BEFORE fitting any transform (no leakage).
    seed = _resolve_seed(config, data_cfg["split"])
    with mem.stage("splitting"):
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
            x, y, data_cfg["split"], seed
        )
        split_report = build_split_report(y_train, y_val, y_test, data_cfg["split"], seed)
    del cleaned, x, y  # release the full-size frames; splits are self-contained
    collect_garbage(collect)

    # 5. Encode (fit on train only).
    with mem.stage("encoding"):
        fitted_encoder = fit_encoder(x_train, categorical, data_cfg["encoding"])
        x_train_e = apply_encoder(fitted_encoder, x_train)
        x_val_e = apply_encoder(fitted_encoder, x_val)
        x_test_e = apply_encoder(fitted_encoder, x_test)

        label_encoder: FittedLabelEncoder | None = None
        y_train_e, y_val_e, y_test_e = y_train, y_val, y_test
        if has_label and data_cfg["encoding"].get("encode_labels", True):
            label_encoder = fit_label_encoder(y_train, column=label_column)
            y_train_e = apply_label_encoder(label_encoder, y_train)
            y_val_e = apply_label_encoder(label_encoder, y_val)
            y_test_e = apply_label_encoder(label_encoder, y_test)
    # Free pre-encode feature frames when encoding produced distinct objects.
    if x_train_e is not x_train:
        del x_train, x_val, x_test
    collect_garbage(collect)

    encoding_report = EncodingReport(
        strategy=str(data_cfg["encoding"].get("categorical_strategy", "onehot")),
        encoded_columns=list(categorical),
        n_input_categorical=len(categorical),
        n_output_features=int(x_train_e.shape[1]),
        label_column=label_column,
        label_classes=list(label_encoder.classes) if label_encoder else [],
        elapsed_seconds=_stage_elapsed(mem, "encoding"),
    )

    # 6. Scale (fit on train only); scale original continuous columns only.
    with mem.stage("scaling"):
        fitted_scaler = fit_scaler(x_train_e, numeric_cols, data_cfg["scaling"])
        x_train_s = apply_scaler(fitted_scaler, x_train_e)
        x_val_s = apply_scaler(fitted_scaler, x_val_e)
        x_test_s = apply_scaler(fitted_scaler, x_test_e)
    if x_train_s is not x_train_e:
        del x_train_e, x_val_e, x_test_e
    collect_garbage(collect)

    scaling_report = ScalingReport(
        strategy=fitted_scaler.strategy,
        scaled_columns=list(fitted_scaler.columns),
        n_scaled_features=len(fitted_scaler.columns),
        elapsed_seconds=_stage_elapsed(mem, "scaling"),
    )

    split = DatasetSplit(
        x_train=x_train_s, y_train=y_train_e,
        x_val=x_val_s, y_val=y_val_e,
        x_test=x_test_s, y_test=y_test_e,
    )

    # 7. Persist everything (downcasting numeric outputs along the way).
    with mem.stage("serialization"):
        output_paths, downcast_dtypes = _persist(
            dataset_id=dataset_id,
            dataset_config=dataset_config,
            paths=paths,
            data_cfg=data_cfg,
            opt_cfg=opt_cfg,
            label_column=label_column,
            loader=loader,
            report=report,
            split=split,
            seed=seed,
            cleaning_report=cleaning_report,
            encoding_report=encoding_report,
            scaling_report=scaling_report,
            split_report=split_report,
            fitted_encoder=fitted_encoder,
            fitted_scaler=fitted_scaler,
            label_encoder=label_encoder,
        )
    collect_garbage(collect)

    # 8. Persist the memory profile (now that every stage is recorded).
    if mem.enabled:
        prep_dir = ensure_dir(Path(paths.preprocessing_dir) / dataset_id)
        profile = mem.to_dict()
        profile["dataset_id"] = dataset_id
        profile["downcast_dtypes"] = downcast_dtypes
        output_paths["memory_profile"] = write_json(
            profile, prep_dir / "memory_profile.json"
        )
        logger.info("[%s] peak RAM %.1f MB across pipeline.", dataset_id,
                    profile["peak_mb"])

    logger.info("Preprocessing complete for '%s' (%d artefacts).",
                dataset_id, len(output_paths))
    return PipelineResult(
        dataset_id=dataset_id,
        split=split,
        output_paths=output_paths,
        validation_passed=report.schema_passed,
    )


def _stage_elapsed(mem: MemoryReport, stage: str) -> float:
    """Return the recorded elapsed seconds for a profiled stage (0 if absent)."""
    for entry in mem.stages:
        if entry.get("stage") == stage:
            return round(float(entry.get("elapsed_seconds", 0.0)), 6)
    return 0.0


def _persist(**kw: Any) -> tuple[dict[str, Path], dict[str, str]]:
    """Write processed splits, artefacts, reports and the manifest.

    Returns the mapping of artefact name -> written path and the dtype map of
    columns downcast on the processed outputs.
    """
    dataset_id: str = kw["dataset_id"]
    paths = kw["paths"]
    data_cfg: Mapping[str, Any] = kw["data_cfg"]
    opt_cfg: Mapping[str, Any] = kw["opt_cfg"]
    label_column: str | None = kw["label_column"]
    split: DatasetSplit = kw["split"]

    prep_dir = ensure_dir(Path(paths.preprocessing_dir) / dataset_id)
    proc_dir = ensure_dir(Path(paths.processed_out_dir) / dataset_id)
    art_dir = ensure_dir(Path(paths.artifacts_dir) / dataset_id)
    io_cfg = data_cfg.get("io", {})
    downcast_enabled = bool(opt_cfg.get("downcast_numeric", True))

    out: dict[str, Path] = {}
    downcast_dtypes: dict[str, str] = {}

    # Processed splits (downcast numeric dtypes immediately before writing).
    frames = {
        "train": _assemble(split.x_train, split.y_train, label_column),
        "validation": _assemble(split.x_val, split.y_val, label_column),
        "test": _assemble(split.x_test, split.y_test, label_column),
    }
    for name, frame in frames.items():
        frame, dtypes = downcast_numeric(frame, downcast_enabled)
        downcast_dtypes.update(dtypes)
        out[name] = _write_frame(frame, proc_dir / name, io_cfg)
        del frame

    # Fitted artefacts.
    out["encoder"] = save_artifact(kw["fitted_encoder"], art_dir / "encoder.joblib")
    out["scaler"] = save_artifact(kw["fitted_scaler"], art_dir / "scaler.joblib")
    if kw["label_encoder"] is not None:
        out["label_encoder"] = save_artifact(
            kw["label_encoder"], art_dir / "label_encoder.joblib"
        )

    # Stage reports.
    out["cleaning_report"] = write_json(
        kw["cleaning_report"].to_dict(), prep_dir / "cleaning_report.json"
    )
    out["encoding_report"] = write_json(
        kw["encoding_report"].to_dict(), prep_dir / "encoding_report.json"
    )
    out["scaling_report"] = write_json(
        kw["scaling_report"].to_dict(), prep_dir / "scaling_report.json"
    )
    out["split_report"] = write_json(
        kw["split_report"].to_dict(), prep_dir / "split_report.json"
    )

    # Manifest (records provenance + output locations + optimization metadata).
    manifest = _build_manifest(out=out, downcast_dtypes=downcast_dtypes, **kw)
    out["manifest"] = write_json(manifest, prep_dir / "preprocessing_manifest.json")
    return out, downcast_dtypes


def _build_manifest(
    out: dict[str, Path], downcast_dtypes: dict[str, str], **kw: Any
) -> dict[str, Any]:
    """Assemble the preprocessing manifest dict."""
    data_cfg: Mapping[str, Any] = kw["data_cfg"]
    opt_cfg: Mapping[str, Any] = kw["opt_cfg"]
    report = kw["report"]
    loader = kw["loader"]
    fitted_encoder = kw["fitted_encoder"]
    fitted_scaler = kw["fitted_scaler"]
    label_encoder = kw["label_encoder"]
    split_cfg = data_cfg["split"]

    fingerprint = build_fingerprint(
        kw["dataset_config"],
        loader.raw_dir(),
        n_rows=report.n_rows,
        n_features=report.n_features,
    )

    return {
        "dataset_id": kw["dataset_id"],
        "dataset_name": kw["dataset_config"].get("name"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(kw["seed"]),
        "fingerprint": fingerprint,
        "preprocessing_config": {
            "cleaning": dict(data_cfg.get("cleaning", {})),
            "encoding": dict(data_cfg.get("encoding", {})),
            "scaling": dict(data_cfg.get("scaling", {})),
            "split": dict(split_cfg),
            "io": dict(data_cfg.get("io", {})),
            "optimization": dict(opt_cfg),
        },
        "split_ratios": {
            "train": float(split_cfg["train_size"]),
            "validation": float(split_cfg["val_size"]),
            "test": float(split_cfg["test_size"]),
        },
        "encoder": {
            "strategy": fitted_encoder.strategy,
            "columns": list(fitted_encoder.columns),
            "n_output_features": len(fitted_encoder.feature_names_out),
            "path": str(out.get("encoder")),
        },
        "scaler": {
            "strategy": fitted_scaler.strategy,
            "columns": list(fitted_scaler.columns),
            "path": str(out.get("scaler")),
        },
        "label_encoder": (
            {
                "column": label_encoder.column,
                "classes": list(label_encoder.classes),
                "path": str(out.get("label_encoder")),
            }
            if label_encoder is not None
            else None
        ),
        "optimization": {
            "downcast_numeric": bool(opt_cfg.get("downcast_numeric", True)),
            "collect_garbage": bool(opt_cfg.get("collect_garbage", True)),
            "profile_memory": bool(opt_cfg.get("profile_memory", True)),
            "downcast_dtypes": downcast_dtypes,
        },
        "outputs": {name: str(path) for name, path in out.items()},
    }


def run_all(config: Mapping[str, Any], paths: Any) -> dict[str, PipelineResult]:
    """Run the pipeline for every dataset in ``data.active_datasets``.

    Loaders that are not yet implemented (``NotImplementedError``) are skipped
    with a warning rather than aborting the batch.

    Parameters
    ----------
    config:
        Effective merged configuration.
    paths:
        Resolved paths.

    Returns
    -------
    dict
        dataset_id -> :class:`PipelineResult` (only successfully run datasets).
    """
    results: dict[str, PipelineResult] = {}
    for dataset_id in config.get("data", {}).get("active_datasets", []):
        try:
            results[dataset_id] = run_pipeline(dataset_id, config, paths)
        except NotImplementedError:
            logger.warning("[%s] loader not implemented yet — skipping.", dataset_id)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
            logger.error("[%s] preprocessing failed: %s", dataset_id, exc)
    return results

"""Engine A training orchestration.

Purpose
-------
Compose model training and evaluation into one reproducible flow per
(dataset, model): load the feature-engineered splits, fit on training data,
evaluate on validation/test, and persist the model, metrics and an experiment
manifest under a unique, never-overwritten directory. This is the single
orchestration surface; scripts call into it.

Human-in-the-Loop
-----------------
This module only *defines* the training flow. Actual long-running training is
launched by the human via ``scripts/train_model.py`` (CLAUDE.md > HITL).

Outputs (per run):
    <experiments_dir>/<dataset>/<model>/<run_id>/{model.joblib,metrics.json,manifest.json}
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.explainability.runner import maybe_explain_after_training
from src.features.metadata import select_feature_columns, split_xy
from src.models.registry import build_model
from src.training.experiment import build_manifest, create_experiment
from src.training.experiment_index import append_index_row
from src.training.feature_audit import audit_features, load_expected_features
from src.training.metrics import classification_metrics
from src.utils.config import load_dataset_config
from src.utils.hardware import log_hardware_summary
from src.utils.io import read_parquet, write_json
from src.utils.timer import Timer

logger = logging.getLogger(__name__)

_OPTIONAL_SPLITS = ("validation", "test")


@dataclass
class TrainingResult:
    """Summary of one completed training run.

    Attributes
    ----------
    experiment_id:
        Unique run id.
    dataset_id, model_name:
        What was trained.
    output_dir:
        Directory containing the run's artefacts.
    metrics:
        Per-split metric dictionaries.
    output_paths:
        Mapping of artefact name -> path.
    """

    experiment_id: str
    dataset_id: str
    model_name: str
    output_dir: Path
    metrics: dict[str, Any]
    output_paths: dict[str, Path]


def _load_xy(frame: "Any", label_column: str) -> tuple["Any", "Any"]:
    """Split a feature frame into numeric-only features and the target."""
    x, y = split_xy(frame, label_column)
    x_numeric, excluded = select_feature_columns(x)
    if excluded["provenance"] or excluded["non_numeric"]:
        logger.warning(
            "Dropped non-feature columns before training: provenance=%s non_numeric=%s",
            excluded["provenance"], excluded["non_numeric"],
        )
    return x_numeric, y


def _load_splits(
    feat_dir: Path, label_column: str
) -> dict[str, tuple["Any", "Any"]]:
    """Load train (+ optional validation/test) feature splits as ``(X, y)``."""
    train_path = feat_dir / "train.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Feature-engineered train split not found: {train_path}. "
            f"Run feature engineering (Phase 3) first."
        )
    x_train, y_train = _load_xy(read_parquet(train_path), label_column)
    feature_columns = list(x_train.columns)
    splits: dict[str, tuple[Any, Any]] = {"train": (x_train, y_train)}

    for name in _OPTIONAL_SPLITS:
        path = feat_dir / f"{name}.parquet"
        if path.is_file():
            x, y = _load_xy(read_parquet(path), label_column)
            # Align to the training feature columns (defensive; should match).
            splits[name] = (x[feature_columns], y)
    return splits


def train_model(
    dataset_id: str,
    model_name: str,
    config: Mapping[str, Any],
    paths: Any,
) -> TrainingResult:
    """Train and evaluate one model on one dataset; persist the experiment.

    Parameters
    ----------
    dataset_id:
        Registered dataset identifier.
    model_name:
        Registered model id (``models.<name>`` must exist in config).
    config:
        Effective merged configuration.
    paths:
        Resolved :class:`src.utils.paths.Paths`.

    Returns
    -------
    TrainingResult

    Raises
    ------
    KeyError
        If the model has no configuration block.
    FileNotFoundError
        If feature-engineered data is missing.
    """
    training_cfg = config["training"]
    models_cfg = config.get("models", {})
    if model_name not in models_cfg:
        raise KeyError(f"No configuration for model '{model_name}' under 'models'.")

    seed = training_cfg.get("random_seed")
    seed = int(seed) if seed is not None else int(config.get("project", {}).get("seed", 42))
    use_gpu = bool(training_cfg.get("use_gpu", True))
    average = str(training_cfg.get("evaluation", {}).get("average", "weighted"))

    dataset_config = load_dataset_config(dataset_id)
    label_column = dataset_config.get("label_column")
    if not label_column:
        raise ValueError(f"[{dataset_id}] no label column configured for training.")

    logger.info("Training '%s' on dataset '%s' (seed=%d).", model_name, dataset_id, seed)
    hardware = log_hardware_summary()

    feat_dir = Path(paths.features_out_dir) / dataset_id
    splits = _load_splits(feat_dir, label_column)
    x_train, y_train = splits["train"]
    x_val, y_val = splits.get("validation", (None, None))

    # Diagnostics + defensive guards: the training matrix must be the full
    # feature-engineered split, never a sampled/debug subset.
    feature_names = list(x_train.columns)
    logger.info(
        "[%s/%s] X_train.shape=%s y_train.shape=%s | %d selected feature(s)",
        dataset_id, model_name, tuple(x_train.shape), tuple(y_train.shape),
        len(feature_names),
    )
    logger.debug("[%s/%s] selected features: %s", dataset_id, model_name, feature_names)

    min_train_rows = int(training_cfg.get("min_train_rows", 1000))
    assert x_train.shape[1] > 0, (
        f"[{dataset_id}/{model_name}] no feature columns to train on "
        f"(shape={x_train.shape})."
    )
    assert x_train.shape[0] > min_train_rows, (
        f"[{dataset_id}/{model_name}] training matrix has only {x_train.shape[0]} "
        f"rows (expected > {min_train_rows}); the full feature-engineered split "
        f"was not loaded. Check the data path for an accidental subset."
    )

    # Pre-fit feature audit: columns, order, dtypes, missing values, duplicates
    # verified against the feature-engineering artefacts for every split.
    audit_features(splits, load_expected_features(feat_dir), dataset_id, model_name)

    model = build_model(model_name, models_cfg[model_name], use_gpu, seed)

    with Timer(f"train::{model_name}") as train_timer:
        model.fit(x_train, y_train, x_val, y_val)

    metrics, timings = _evaluate(model, splits, average)
    timings["train_seconds"] = round(train_timer.elapsed, 4)

    result = _persist_experiment(
        dataset_id=dataset_id,
        model_name=model_name,
        model=model,
        config=config,
        paths=paths,
        hardware=hardware,
        metrics=metrics,
        timings=timings,
        seed=seed,
    )
    # Post-training explainability (configuration-gated; never fails the run).
    maybe_explain_after_training(result, model, splits, config, paths)
    return result


def _evaluate(
    model: Any,
    splits: Mapping[str, tuple["Any", "Any"]],
    average: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the fitted model on every available split."""
    metrics: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    # Use the model's complete fitted class list so multiclass ROC-AUC covers
    # every class, not only those present in a given split.
    classes = model.classes_
    for split_name, (x, y) in splits.items():
        with Timer(f"predict::{split_name}") as inf_timer:
            y_pred = model.predict(x)
        proba = model.predict_proba(x) if model.is_supervised else None
        metrics[split_name] = classification_metrics(y, y_pred, proba, average, classes=classes)
        timings[f"predict_{split_name}_seconds"] = round(inf_timer.elapsed, 4)
        if not model.is_supervised and metrics[split_name]["n_classes"] > 2:
            logger.warning(
                "[%s] %s is an anomaly detector; supervised metrics on a "
                "%d-class target are indicative only.",
                split_name, model.name, metrics[split_name]["n_classes"],
            )
    return metrics, timings


def _persist_experiment(**kw: Any) -> TrainingResult:
    """Save the model, metrics and manifest under a unique experiment dir."""
    dataset_id: str = kw["dataset_id"]
    model_name: str = kw["model_name"]
    model = kw["model"]
    config: Mapping[str, Any] = kw["config"]
    paths = kw["paths"]

    experiment = create_experiment(dataset_id, model_name, Path(paths.experiments_dir))
    out_dir = experiment.output_dir

    model_path = model.save(out_dir / "model.joblib")
    model_size = int(os.path.getsize(model_path))
    kw["timings"]["model_size_bytes"] = model_size

    metrics_path = write_json(kw["metrics"], out_dir / "metrics.json")

    config_snapshot = {
        "training": dict(config.get("training", {})),
        "model": dict(config.get("models", {}).get(model_name, {})),
    }
    artefacts = {"model": str(model_path), "metrics": str(metrics_path)}
    manifest = build_manifest(
        experiment=experiment,
        model_description=model.describe(),
        config_snapshot=config_snapshot,
        hardware=kw["hardware"],
        metrics=kw["metrics"],
        timings=kw["timings"],
        artefacts=artefacts,
        seed=kw["seed"],
    )
    manifest_path = write_json(manifest, out_dir / "manifest.json")
    try:
        append_index_row(manifest, Path(paths.experiments_dir))
    except OSError as exc:
        # A locked index (e.g. the CSV open in Excel) must not fail a run whose
        # artefacts are already persisted; the index is rebuildable from the
        # manifests via scripts/build_experiment_index.py.
        logger.error(
            "Experiment index update failed (%s); run 'python -m "
            "scripts.build_experiment_index' to backfill.", exc,
        )

    logger.info(
        "[%s/%s] model saved (%.1f KB); experiment %s complete.",
        dataset_id, model_name, model_size / 1024, experiment.experiment_id,
    )
    return TrainingResult(
        experiment_id=experiment.experiment_id,
        dataset_id=dataset_id,
        model_name=model_name,
        output_dir=out_dir,
        metrics=kw["metrics"],
        output_paths={
            "model": model_path,
            "metrics": metrics_path,
            "manifest": manifest_path,
        },
    )

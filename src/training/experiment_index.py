"""Experiment index — one searchable CSV row per training run.

Purpose
-------
Manifests are complete but one-per-directory; comparing dozens of runs means
opening dozens of JSON files. This module maintains
``<experiments_dir>/experiment_index.csv`` with one summary row per run so
history is greppable/sortable. Rows are always *derived from manifests* (the
single source of truth): the trainer appends a row after each run, and
:func:`rebuild_index` regenerates the whole file from disk, which both
backfills old runs and repairs a diverged index.

Outputs
-------
CSV columns: ``timestamp, dataset, model, run_id, train_seconds, best_epoch,
accuracy, f1, roc_auc, hardware, key_hyperparameters``. Metric columns use the
test split, falling back to validation then train. ``best_epoch`` is the
number of epochs actually trained (deep models only; empty for classical).

Examples
--------
>>> from src.training.experiment_index import rebuild_index  # doctest: +SKIP
>>> rebuild_index(Path("outputs/experiments"))               # doctest: +SKIP

Limitations
-----------
``key_hyperparameters`` is a compact human-readable summary (scalar params
only), not a round-trippable encoding; the manifest remains authoritative.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

INDEX_FILENAME = "experiment_index.csv"

INDEX_COLUMNS = [
    "timestamp",
    "dataset",
    "model",
    "run_id",
    "train_seconds",
    "best_epoch",
    "accuracy",
    "f1",
    "roc_auc",
    "hardware",
    "key_hyperparameters",
]

# Split preference for the summary metric columns.
_METRIC_SPLITS = ("test", "validation", "train")


def _summary_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the preferred split's metric dict (test > validation > train)."""
    for split in _METRIC_SPLITS:
        if split in metrics:
            return metrics[split]
    return {}


def _key_hyperparameters(manifest: Mapping[str, Any]) -> str:
    """Compact ``k=v; ...`` summary of the configured scalar hyperparameters.

    Deep-learning models carry a nested ``training`` block; its headline
    values (batch size, learning rate, epochs) are lifted into the summary.
    """
    params = dict(
        manifest.get("config_snapshot", {}).get("model", {}).get("params", {})
    )
    training = params.pop("training", None)
    flat: dict[str, Any] = {
        k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))
    }
    if isinstance(training, Mapping):
        for key in ("batch_size", "learning_rate", "epochs"):
            if key in training:
                flat[key] = training[key]
    return "; ".join(f"{k}={flat[k]}" for k in sorted(flat))


def index_row_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive one index row from a run manifest.

    Parameters
    ----------
    manifest:
        A parsed ``manifest.json`` as written by the trainer.

    Returns
    -------
    dict
        Keys exactly matching :data:`INDEX_COLUMNS`.
    """
    model = manifest.get("model", {}) or {}
    fitted = model.get("fitted_params") or {}
    metrics = _summary_metrics(manifest.get("metrics", {}) or {})
    device = model.get("device") or (manifest.get("hardware", {}) or {}).get("device")

    def _metric(name: str) -> Any:
        value = metrics.get(name)
        return round(value, 6) if isinstance(value, float) else value

    return {
        "timestamp": manifest.get("created_at"),
        "dataset": manifest.get("dataset_id"),
        "model": manifest.get("model_name"),
        "run_id": manifest.get("experiment_id"),
        "train_seconds": (manifest.get("timings", {}) or {}).get("train_seconds"),
        "best_epoch": fitted.get("epochs_trained"),
        "accuracy": _metric("accuracy"),
        "f1": _metric("f1"),
        "roc_auc": _metric("roc_auc"),
        "hardware": device,
        "key_hyperparameters": _key_hyperparameters(manifest),
    }


def append_index_row(manifest: Mapping[str, Any], experiments_root: Path) -> Path:
    """Append one run's row to the experiment index (creating it if needed).

    Parameters
    ----------
    manifest:
        The run manifest to summarise.
    experiments_root:
        Base experiments directory (``paths.experiments_dir``).

    Returns
    -------
    Path
        The index CSV path.
    """
    index_path = Path(experiments_root) / INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.is_file()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(index_row_from_manifest(manifest))
    logger.info("Experiment index updated: %s", index_path)
    return index_path


def rebuild_index(experiments_root: Path) -> Path:
    """Regenerate the full index from every manifest on disk.

    Used to backfill runs recorded before the index existed and to repair the
    file if it ever diverges from the manifests. Rows are ordered by
    ``created_at``.

    Parameters
    ----------
    experiments_root:
        Base experiments directory (``paths.experiments_dir``).

    Returns
    -------
    Path
        The index CSV path.
    """
    from src.utils.io import read_json

    root = Path(experiments_root)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/*/*/manifest.json")):
        try:
            rows.append(index_row_from_manifest(read_json(manifest_path)))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable manifest %s: %s", manifest_path, exc)
    rows.sort(key=lambda row: str(row.get("timestamp") or ""))

    index_path = root / INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Experiment index rebuilt: %s (%d run(s)).", index_path, len(rows))
    return index_path

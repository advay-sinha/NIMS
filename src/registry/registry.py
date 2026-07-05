"""Registry entry construction and best-candidate selection.

Purpose
-------
Turn completed experiment manifests into validated registry entries and
select the best candidate per dataset by a configurable metric. Pure logic —
file layout and persistence live in :mod:`src.registry.artifacts`.

Inputs
------
Experiment manifest paths plus the artefact roots (for optional
explainability / error-analysis / visualization / feature references).

Outputs
-------
Registry entry dicts and the best-per-dataset mapping.

Limitations
-----------
Metrics are read from the manifests as persisted — never recomputed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

STATUSES = ("candidate", "staging", "production", "archived")

# Scalar metrics kept per split; matrices/reports stay in the run directory.
_SCALAR_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


class RegistryError(RuntimeError):
    """Raised when registration, promotion or resolution cannot proceed."""


def entry_from_manifest(
    manifest_path: Path,
    roots: Mapping[str, Path],
    *,
    source: str = "benchmark",
    tags: Sequence[str] = (),
    status: str = "candidate",
    registered_at: str | None = None,
) -> dict[str, Any]:
    """Build one validated registry entry from an experiment manifest.

    Parameters
    ----------
    manifest_path:
        The run's ``manifest.json``.
    roots:
        Artefact roots for optional references — keys ``explainability``,
        ``error_analysis``, ``visualizations``, ``features``,
        ``preprocessing_artifacts`` (all optional).
    source:
        How the entry was registered (``manual`` | ``benchmark``); overridden
        to ``optimization`` when the manifest carries optimization provenance.
    tags:
        Free-form labels.
    status:
        Initial lifecycle status (one of :data:`STATUSES`).
    registered_at:
        Preserved registration timestamp (rebuilds); defaults to now.

    Returns
    -------
    dict
        The registry entry.

    Raises
    ------
    RegistryError
        When the manifest or model artefact is missing, or the manifest
        lacks dataset/model identity or metrics.
    """
    from src.utils.io import read_json

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise RegistryError(f"Manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)

    experiment_id = manifest.get("experiment_id")
    dataset = manifest.get("dataset_id")
    model_type = manifest.get("model_name")
    metrics = manifest.get("metrics")
    if not dataset:
        raise RegistryError(f"{manifest_path}: manifest has no dataset_id.")
    if not model_type:
        raise RegistryError(f"{manifest_path}: manifest has no model_name.")
    if not metrics:
        raise RegistryError(f"{manifest_path}: manifest has no metrics.")
    if status not in STATUSES:
        raise RegistryError(f"Unknown status '{status}' (expected {STATUSES}).")

    model_path = Path(manifest.get("artefacts", {}).get("model", ""))
    if not model_path.is_file():
        # The canonical location is next to the manifest; check it before
        # failing (artefact paths in old manifests are absolute).
        fallback = manifest_path.parent / "model.joblib"
        if fallback.is_file():
            model_path = fallback
        else:
            raise RegistryError(
                f"{experiment_id}: model artefact not found at {model_path}."
            )

    scalar_metrics = {
        split: {
            key: values.get(key)
            for key in _SCALAR_METRICS
            if values.get(key) is not None
        }
        for split, values in metrics.items()
        if isinstance(values, dict)
    }

    provenance = manifest.get("provenance") or {}
    if provenance.get("source") == "optimization":
        source = "optimization"

    artifacts: dict[str, str] = {
        "model": str(model_path),
        "manifest": str(manifest_path),
    }
    for name, subdir in (
        ("explainability", experiment_id),
        ("error_analysis", experiment_id),
        ("visualizations", experiment_id),
        ("features", dataset),
        ("preprocessing_artifacts", dataset),
    ):
        root = roots.get(name)
        if root is not None and (Path(root) / subdir).is_dir():
            artifacts[name] = str(Path(root) / subdir)

    return {
        "registry_id": f"reg-{experiment_id}",
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model_type": model_type,
        "model_artifact_path": str(model_path),
        "manifest_path": str(manifest_path),
        "metrics": scalar_metrics,
        "test_f1": scalar_metrics.get("test", {}).get("f1"),
        "test_accuracy": scalar_metrics.get("test", {}).get("accuracy"),
        "validation_f1": scalar_metrics.get("validation", {}).get("f1"),
        "created_at": manifest.get("created_at"),
        "registered_at": registered_at
        or datetime.now(timezone.utc).isoformat(),
        "source": source,
        "optimization_study_id": provenance.get("study_id"),
        "tags": list(tags),
        "status": status,
        "artifacts": artifacts,
    }


def metric_value(entry: Mapping[str, Any], selection_metric: str) -> float | None:
    """Read a ``<split>_<metric>`` value (e.g. ``test_f1``) from an entry."""
    split, _, key = selection_metric.partition("_")
    return (entry.get("metrics", {}).get(split) or {}).get(key)


def select_best_per_dataset(
    entries: Sequence[Mapping[str, Any]],
    *,
    selection_metric: str = "test_f1",
    higher_is_better: bool = True,
) -> dict[str, dict[str, Any]]:
    """Pick the best non-archived entry per dataset by the configured metric.

    Parameters
    ----------
    entries:
        Registry entries.
    selection_metric:
        ``<split>_<metric>`` name, e.g. ``test_f1``.
    higher_is_better:
        Direction of comparison.

    Returns
    -------
    dict
        ``{dataset: {experiment_id, model_type, metric, value}}`` — datasets
        whose entries all lack the metric are omitted.
    """
    best: dict[str, dict[str, Any]] = {}
    sign = 1.0 if higher_is_better else -1.0
    for entry in entries:
        if entry.get("status") == "archived":
            continue
        value = metric_value(entry, selection_metric)
        if value is None:
            continue
        current = best.get(entry["dataset"])
        if current is None or sign * value > sign * current["value"] or (
            value == current["value"]
            and entry["experiment_id"] < current["experiment_id"]
        ):
            best[entry["dataset"]] = {
                "experiment_id": entry["experiment_id"],
                "model_type": entry["model_type"],
                "metric": selection_metric,
                "value": value,
            }
    return dict(sorted(best.items()))


def filter_registrable(
    entries: Sequence[dict[str, Any]],
    *,
    require_test_metrics: bool = True,
    allow_optimized_models: bool = True,
) -> list[dict[str, Any]]:
    """Apply the configured registration policy to candidate entries."""
    kept = []
    for entry in entries:
        if require_test_metrics and entry.get("test_f1") is None:
            logger.warning(
                "Skipping %s: no test metrics recorded.", entry["experiment_id"]
            )
            continue
        if not allow_optimized_models and entry.get("source") == "optimization":
            logger.info(
                "Skipping %s: optimized models are excluded by configuration.",
                entry["experiment_id"],
            )
            continue
        kept.append(entry)
    return kept

"""Model resolution for inference and reporting.

Purpose
-------
Resolve "which model serves dataset X?" from the registry files. This is the
lookup surface a future inference service (FastAPI) will call — it returns
paths and metadata only; loading the model is the caller's job.

Inputs
------
The registry directory, a dataset id and a stage.

Outputs
-------
A resolution dict (see :func:`resolve_model`).

Limitations
-----------
No inference is performed here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.registry.artifacts import load_production, load_registry
from src.registry.registry import RegistryError

logger = logging.getLogger(__name__)

_STAGES = ("production", "best")


def resolve_model(
    dataset: str,
    stage: str = "production",
    *,
    registry_dir: Path,
) -> dict[str, Any]:
    """Resolve the model serving a dataset at a given stage.

    Parameters
    ----------
    dataset:
        Dataset identifier.
    stage:
        ``production`` (explicit assignment from ``production.json``) or
        ``best`` (automatic best candidate from ``best_per_dataset.json``,
        resolved through the registry entries).
    registry_dir:
        Registry directory containing the built registry files.

    Returns
    -------
    dict
        ``experiment_id``, ``model_type``, ``model_artifact_path``,
        ``manifest_path``, ``metrics``, ``status``, ``stage`` and the
        optional ``artifacts`` references (features, preprocessing
        artefacts, explainability, ...).

    Raises
    ------
    RegistryError
        For unknown stages, missing assignments, or an assignment that no
        longer matches a registered entry.
    """
    if stage not in _STAGES:
        raise RegistryError(f"Unknown stage '{stage}' (expected {_STAGES}).")

    document = load_registry(registry_dir)
    entries = {e["experiment_id"]: e for e in document.get("entries", [])}
    if not entries:
        raise RegistryError(
            f"Registry at {registry_dir} is empty; run "
            f"'python -m scripts.build_model_registry' first."
        )

    if stage == "production":
        assignment = load_production(registry_dir).get(dataset)
        if assignment is None:
            raise RegistryError(
                f"No production model assigned for dataset '{dataset}'; "
                f"promote one with scripts/promote_model.py."
            )
        experiment_id = assignment["experiment_id"]
    else:
        from src.registry.artifacts import BEST_FILE
        from src.utils.io import read_json

        best_path = Path(registry_dir) / BEST_FILE
        best = read_json(best_path) if best_path.is_file() else {}
        pick = best.get(dataset)
        if pick is None:
            raise RegistryError(
                f"No best candidate recorded for dataset '{dataset}'."
            )
        experiment_id = pick["experiment_id"]

    entry = entries.get(experiment_id)
    if entry is None:
        raise RegistryError(
            f"Assigned experiment '{experiment_id}' for '{dataset}' is not in "
            f"the registry; rebuild it (scripts/build_model_registry.py)."
        )

    resolution = {
        "experiment_id": entry["experiment_id"],
        "dataset": entry["dataset"],
        "model_type": entry["model_type"],
        "model_artifact_path": entry["model_artifact_path"],
        "manifest_path": entry["manifest_path"],
        "metrics": entry["metrics"],
        "status": entry["status"],
        "stage": stage,
        "artifacts": entry.get("artifacts", {}),
    }
    logger.info("Resolved %s/%s -> %s", dataset, stage, entry["experiment_id"])
    return resolution

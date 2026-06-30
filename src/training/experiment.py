"""Experiment bookkeeping.

Purpose
-------
Give every training run a unique id and an isolated output directory so no
experiment is ever overwritten (CLAUDE.md > Experiment Rules). Records the
configuration snapshot, hardware and metrics alongside the saved model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """A single training run's identity and output location.

    Attributes
    ----------
    experiment_id:
        Unique id ``<dataset>_<model>_<UTC timestamp>``.
    dataset_id:
        Dataset trained on.
    model_name:
        Model id.
    output_dir:
        Isolated directory for this run's artefacts.
    created_at:
        UTC ISO-8601 creation timestamp.
    """

    experiment_id: str
    dataset_id: str
    model_name: str
    output_dir: Path
    created_at: str


def create_experiment(
    dataset_id: str,
    model_name: str,
    experiments_root: Path,
) -> Experiment:
    """Create a unique experiment with an isolated output directory.

    The directory is ``<experiments_root>/<dataset>/<model>/<run_id>``; a
    numeric suffix is appended if the timestamped id already exists, so runs are
    never overwritten.

    Parameters
    ----------
    dataset_id:
        Dataset identifier.
    model_name:
        Model identifier.
    experiments_root:
        Base experiments directory (``paths.experiments_dir``).

    Returns
    -------
    Experiment
    """
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S")
    base_id = f"{dataset_id}_{model_name}_{stamp}"

    parent = Path(experiments_root) / dataset_id / model_name
    candidate = parent / base_id
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base_id}_{suffix}"
        suffix += 1

    ensure_dir(candidate)
    experiment = Experiment(
        experiment_id=candidate.name,
        dataset_id=dataset_id,
        model_name=model_name,
        output_dir=candidate,
        created_at=now.isoformat(),
    )
    logger.info("Created experiment '%s' -> %s", experiment.experiment_id, candidate)
    return experiment


def build_manifest(
    experiment: Experiment,
    model_description: dict[str, Any],
    config_snapshot: dict[str, Any],
    hardware: dict[str, Any],
    metrics: dict[str, Any],
    timings: dict[str, Any],
    artefacts: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    """Assemble the experiment manifest dict."""
    return {
        "experiment_id": experiment.experiment_id,
        "dataset_id": experiment.dataset_id,
        "model_name": experiment.model_name,
        "created_at": experiment.created_at,
        "seed": seed,
        "model": model_description,
        "config_snapshot": config_snapshot,
        "hardware": hardware,
        "timings": timings,
        "metrics": metrics,
        "artefacts": artefacts,
    }

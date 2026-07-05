"""Optimization artefact persistence.

Purpose
-------
Write one study's results in a fixed layout::

    outputs/optimization/<study_id>/
        metadata.json            # identity, metric, trial counts, best value
        trials.csv               # one row per trial with expanded params
        best_params.json         # config-shaped params of the best trial
        best_trial.json          # best trial number/value/raw params
        optimization_summary.md  # human-readable summary

Inputs
------
The finished Optuna study plus study identity.

Outputs
-------
Mapping of artefact name -> written path.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_BASE_COLUMNS = ["trial_number", "state", "value", "duration_seconds"]


def trials_rows(study: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Flatten a study's trials into CSV columns and rows.

    Parameters
    ----------
    study:
        Finished Optuna study.

    Returns
    -------
    tuple
        ``(columns, rows)`` — base columns plus one column per parameter
        name seen across trials (sorted, ``param_`` prefixed).
    """
    param_names = sorted({name for t in study.trials for name in t.params})
    columns = _BASE_COLUMNS + [f"param_{name}" for name in param_names]
    rows = []
    for trial in study.trials:
        duration = None
        if trial.datetime_start and trial.datetime_complete:
            duration = round(
                (trial.datetime_complete - trial.datetime_start).total_seconds(), 3
            )
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "duration_seconds": duration,
        }
        for name in param_names:
            row[f"param_{name}"] = trial.params.get(name)
        rows.append(row)
    return columns, rows


def write_optimization_artifacts(
    study: Any,
    *,
    study_id: str,
    dataset_id: str,
    model_name: str,
    metric: str,
    direction: str,
    n_trials_requested: int,
    seed: int,
    best_params: Mapping[str, Any],
    output_root: Path,
    summary_md: str,
) -> dict[str, Path]:
    """Persist all artefacts for one finished study.

    Parameters
    ----------
    study:
        Finished Optuna study with at least one completed trial.
    study_id, dataset_id, model_name, metric, direction, seed:
        Study identity recorded into ``metadata.json``.
    n_trials_requested:
        Trials asked for (completed count may be lower on failures/timeout).
    best_params:
        Config-shaped parameters of the best trial (as produced by the
        search space, ready to merge into the model config).
    output_root:
        Optimization output root (``paths.optimization_dir``).
    summary_md:
        Rendered Markdown summary to persist alongside the data files.

    Returns
    -------
    dict[str, Path]
        Artefact name -> written path.
    """
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(output_root) / study_id)
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]

    metadata = {
        "study_id": study_id,
        "dataset": dataset_id,
        "model_type": model_name,
        "metric": metric,
        "direction": direction,
        "n_trials_requested": n_trials_requested,
        "n_trials_completed": len(completed),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_value": study.best_value,
    }
    paths = {"metadata": write_json(metadata, out_dir / "metadata.json")}

    columns, rows = trials_rows(study)
    trials_path = out_dir / "trials.csv"
    with trials_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    paths["trials"] = trials_path

    paths["best_params"] = write_json(dict(best_params), out_dir / "best_params.json")

    best = study.best_trial
    paths["best_trial"] = write_json(
        {
            "trial_number": best.number,
            "value": best.value,
            "state": best.state.name,
            "params": dict(best.params),
        },
        out_dir / "best_trial.json",
    )

    summary_path = out_dir / "optimization_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    paths["summary"] = summary_path

    logger.info(
        "Optimization artefacts written to %s (%d/%d trials completed, best "
        "%s=%.6f).", out_dir, len(completed), n_trials_requested, metric,
        study.best_value,
    )
    return paths

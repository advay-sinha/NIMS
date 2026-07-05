"""Optimization reporting helper.

Purpose
-------
Summarise one study as a Markdown block for logs, the persisted
``optimization_summary.md`` and future report integration — without touching
:mod:`src.training.reporting`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def optimization_summary(
    study: Any,
    *,
    study_id: str,
    metric: str,
    best_params: Mapping[str, Any],
    artifact_dir: Path | str,
    final_experiment_id: str | None = None,
) -> str:
    """Render a Markdown summary of one finished study.

    Parameters
    ----------
    study:
        Finished Optuna study.
    study_id, metric:
        Study identity.
    best_params:
        Config-shaped best parameters.
    artifact_dir:
        Where the study artefacts live.
    final_experiment_id:
        Experiment id of the final best-params training run, when performed.

    Returns
    -------
    str
        Markdown block.
    """
    states: dict[str, int] = {}
    for trial in study.trials:
        states[trial.state.name] = states.get(trial.state.name, 0) + 1

    lines = [
        f"# Optimization Summary — {study_id}",
        "",
        f"- Best {metric}: {study.best_value:.6f} "
        f"(trial {study.best_trial.number})",
        "- Trials: " + ", ".join(
            f"{count} {state.lower()}" for state, count in sorted(states.items())
        ),
        f"- Artefacts: `{artifact_dir}`",
    ]
    if final_experiment_id is not None:
        lines.append(f"- Final model experiment: `{final_experiment_id}`")
    lines.extend(
        [
            "",
            "## Best parameters",
            "",
            "```json",
            json.dumps(dict(best_params), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)

"""Registry reporting helper.

Purpose
-------
Summarise the registry state as a Markdown block for logs and reports —
production assignments, best candidates, metric values and gaps — without
touching :mod:`src.training.reporting`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def registry_summary(
    registry: Mapping[str, Any],
    best: Mapping[str, Any],
    production: Mapping[str, Any],
    registry_dir: Path | str,
) -> str:
    """Render a Markdown summary of the registry state.

    Parameters
    ----------
    registry:
        The ``registry.json`` document.
    best:
        The ``best_per_dataset.json`` mapping.
    production:
        The ``production.json`` mapping.
    registry_dir:
        Where the registry files live.

    Returns
    -------
    str
        Markdown block: per-dataset production/best models with metric
        values, missing production assignments, entry count and location.
    """
    entries = registry.get("entries", [])
    metric = registry.get("selection_metric", "test_f1")
    datasets = sorted({e["dataset"] for e in entries})

    lines = [
        "## Model Registry",
        "",
        f"- Registered experiments: {len(entries)}",
        f"- Selection metric: {metric}",
        f"- Location: `{registry_dir}`",
        "",
        f"| Dataset | Production model | Best candidate | Best {metric} |",
        "|---------|------------------|----------------|--------------:|",
    ]
    for dataset in datasets:
        assignment = production.get(dataset)
        prod_cell = (
            f"{assignment['model_type']} (`{assignment['experiment_id']}`)"
            if assignment else "—"
        )
        pick = best.get(dataset)
        best_cell = (
            f"{pick['model_type']} (`{pick['experiment_id']}`)" if pick else "—"
        )
        value_cell = f"{pick['value']:.4f}" if pick else "—"
        lines.append(f"| {dataset} | {prod_cell} | {best_cell} | {value_cell} |")
    lines.append("")

    missing = [d for d in datasets if d not in production]
    if missing:
        lines.extend(
            [f"Missing production assignments: {', '.join(missing)}", ""]
        )
    return "\n".join(lines)

"""Visualization reporting helper.

Purpose
-------
Summarise one visualization run as a Markdown block for logs and reports —
the integration point for the reporting layer, without touching
:mod:`src.training.reporting`.

Inputs
------
The result mapping returned by
:func:`src.visualization.runner.generate_visualizations`.

Outputs
-------
A Markdown string.
"""

from __future__ import annotations

from typing import Any, Mapping


def visualization_summary(result: Mapping[str, Any]) -> str:
    """Render a Markdown summary of one visualization run.

    Parameters
    ----------
    result:
        ``{"metadata": Path, "plots": {name: Path}, "skipped": {name: reason}}``
        as returned by the runner.

    Returns
    -------
    str
        Markdown block: generated plots, skipped plots with reasons, and the
        artefact location.
    """
    plots: Mapping[str, Any] = result.get("plots", {})
    skipped: Mapping[str, str] = result.get("skipped", {})
    lines = ["## Visualizations", ""]
    if plots:
        lines.append(f"Generated ({len(plots)}):")
        lines.extend(f"- `{path.name}`" for _name, path in sorted(plots.items()))
        lines.append("")
    if skipped:
        lines.append(f"Skipped ({len(skipped)}):")
        lines.extend(
            f"- {name}: {reason}" for name, reason in sorted(skipped.items())
        )
        lines.append("")
    metadata = result.get("metadata")
    if metadata is not None:
        lines.extend([f"Artefacts: `{metadata.parent}`", ""])
    return "\n".join(lines)

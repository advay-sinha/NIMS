"""Error-analysis reporting helper.

Purpose
-------
Summarise one error analysis as a Markdown block for logs and reports. This
is the integration point for the reporting layer — it consumes the analysis
result without touching :mod:`src.training.reporting`.

Inputs
------
An :class:`src.error_analysis.analyzer.ErrorAnalysisResult` and its artefact
directory.

Outputs
-------
A Markdown string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_TOP_HARDEST = 5


def error_analysis_summary(result: Any, artifact_dir: Path | str) -> str:
    """Render a Markdown summary of one error analysis.

    Parameters
    ----------
    result:
        :class:`ErrorAnalysisResult` to summarise.
    artifact_dir:
        Where the artefacts were written (referenced in the summary).

    Returns
    -------
    str
        Markdown block: headline metrics, top hardest classes and the
        artefact location.
    """
    summary = result.summary
    lines = [
        "## Error Analysis",
        "",
        f"- Accuracy: {summary['accuracy']:.4f}",
        f"- Macro F1: {summary['macro_f1']:.4f}",
        f"- Weighted F1: {summary['weighted_f1']:.4f}",
        f"- Misclassified: {summary['n_misclassified']:,} of "
        f"{summary['n_samples']:,} samples",
        f"- Artefacts: `{artifact_dir}`",
        "",
    ]
    hardest = result.hardest_classes.head(_TOP_HARDEST)
    if len(hardest) > 0:
        lines.extend(
            [
                f"Top {len(hardest)} hardest classes (lowest F1):",
                "",
                "| Rank | Class | Support | Precision | Recall | F1 |",
                "|-----:|-------|--------:|----------:|-------:|---:|",
            ]
        )
        for row in hardest.itertuples(index=False):
            lines.append(
                f"| {row.rank} | {row.class_label} | {row.support} "
                f"| {row.precision:.4f} | {row.recall:.4f} "
                f"| {row.f1_score:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)

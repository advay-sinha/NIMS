"""Dataset audit report rendering.

Purpose
-------
Render a human-readable Markdown audit (Phase 1A deliverable) from the
:class:`src.data.validation.DataReport` of each dataset plus its fingerprint.
The audit summarises data quality and derives recommended preprocessing
actions, risks and notes for future model training.

This module performs NO computation on raw data — it consumes already-computed
reports and only formats them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from src.data.validation import DataReport

logger = logging.getLogger(__name__)

# Imbalance ratio above which class imbalance is called out as a training risk.
_IMBALANCE_FLAG: float = 10.0


def _recommended_actions(stats: Mapping[str, Any]) -> list[str]:
    """Derive preprocessing recommendations from a statistics dict."""
    actions: list[str] = []
    if stats.get("infinite_values"):
        actions.append(
            "Replace ±inf values with NaN before imputation "
            f"(columns: {', '.join(sorted(stats['infinite_values']))})."
        )
    if stats.get("total_missing", 0) > 0:
        actions.append(
            "Impute missing values (median for numeric, most-frequent for "
            "categorical); fit imputers on the training split only."
        )
    if stats.get("duplicate_rows", 0) > 0:
        actions.append(
            f"Drop {stats['duplicate_rows']} exact duplicate rows to avoid "
            "train/test leakage and inflated metrics."
        )
    ratio = stats.get("class_imbalance", {}).get("imbalance_ratio")
    if ratio is not None and ratio >= _IMBALANCE_FLAG:
        actions.append(
            "Mitigate class imbalance (class weights, focal loss, or "
            "resampling) — never oversample before splitting."
        )
    if stats.get("categorical_summary"):
        actions.append(
            "Encode categorical features with an encoder fit on the training "
            "split only (handle_unknown='ignore')."
        )
    if stats.get("numerical_summary"):
        actions.append(
            "Scale numeric features (standard/robust) with a scaler fit on the "
            "training split only."
        )
    if not actions:
        actions.append("No immediate preprocessing issues detected.")
    return actions


def _risks(report: DataReport) -> list[str]:
    """Derive risks from a report's validation issues and statistics."""
    risks: list[str] = []
    for issue in report.validation_issues:
        if issue["severity"] in {"error", "warning"}:
            risks.append(f"[{issue['severity']}] {issue['message']}")
    ratio = report.statistics.get("class_imbalance", {}).get("imbalance_ratio")
    if ratio is not None and ratio >= _IMBALANCE_FLAG:
        risks.append(
            "Accuracy will be misleading under this imbalance — rely on "
            "precision/recall/F1 and ROC-AUC."
        )
    if not risks:
        risks.append("No significant risks detected from automated checks.")
    return risks


def _training_notes(report: DataReport) -> list[str]:
    """Notes for future model training (general + dataset-specific)."""
    notes = [
        "Use a stratified train/validation/test split keyed on the label.",
        "Fit all encoders/scalers on the training split only; reuse for "
        "validation, test and inference.",
        "Report precision, recall, F1, ROC-AUC and false-positive rate — not "
        "accuracy alone.",
    ]
    imbalance = report.statistics.get("class_imbalance", {})
    if imbalance.get("n_classes", 0) > 2:
        notes.append(
            f"Target has {imbalance['n_classes']} classes; decide early between "
            "multi-class and collapsed binary (attack vs normal) framing."
        )
    return notes


def _format_distribution(distribution: Mapping[str, int], limit: int = 10) -> str:
    """Render a label distribution as a compact, truncated list."""
    if not distribution:
        return "_unsupervised / no label column_"
    items = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    shown = items[:limit]
    lines = [f"- `{label}`: {count:,}" for label, count in shown]
    if len(items) > limit:
        lines.append(f"- _… {len(items) - limit} more class(es)_")
    return "\n".join(lines)


def _dataset_section(report: DataReport, fingerprint: Mapping[str, Any]) -> str:
    """Render the Markdown section for a single dataset."""
    s = report.statistics
    imbalance = s.get("class_imbalance", {})
    ratio = imbalance.get("imbalance_ratio")
    ratio_str = f"{ratio:,.1f}:1" if ratio is not None else "n/a"
    infinite = s.get("infinite_values", {})

    lines = [
        f"## {report.name or report.dataset_id} (`{report.dataset_id}`)",
        "",
        f"- **Engine:** {report.engine or 'n/a'}",
        f"- **Schema valid:** {'✓' if report.schema_passed else '✗'}",
        f"- **SHA256:** `{fingerprint.get('sha256', '')[:16]}…`",
        f"- **Schema version:** {fingerprint.get('schema_version', 'n/a')}",
        "",
        "### Dataset overview",
        f"- Rows: **{s.get('n_rows', 0):,}**",
        f"- Features: **{s.get('n_features', 0)}**",
        f"- Label column: `{report.label_column}`" if report.label_column else
        "- Label column: _none (unsupervised)_",
        "",
        "### Duplicate analysis",
        f"- Exact duplicate rows: **{s.get('duplicate_rows', 0):,}**",
        "",
        "### Missing value analysis",
        f"- Total missing cells: **{s.get('total_missing', 0):,}** across "
        f"{len(s.get('missing_values', {}))} column(s)",
        "",
        "### Infinite values",
        (
            f"- Columns with ±inf: **{len(infinite)}** "
            f"({', '.join(sorted(infinite))})"
            if infinite
            else "- None detected."
        ),
        "",
        "### Class imbalance",
        f"- Classes: **{imbalance.get('n_classes', 0)}**",
        f"- Majority/minority ratio: **{ratio_str}**",
        f"- Distribution:\n{_format_distribution(s.get('label_distribution', {}))}",
        "",
        "### Memory usage",
        f"- In-memory footprint: **{s.get('memory_usage_mb', 0):,.2f} MB**",
        "",
        "### Recommended preprocessing actions",
        *[f"- {a}" for a in _recommended_actions(s)],
        "",
        "### Risks",
        *[f"- {r}" for r in _risks(report)],
        "",
        "### Notes for future model training",
        *[f"- {n}" for n in _training_notes(report)],
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def render_audit_markdown(
    reports: Mapping[str, DataReport],
    fingerprints: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render the full multi-dataset audit as a Markdown document.

    Parameters
    ----------
    reports:
        Mapping of dataset id -> :class:`DataReport`.
    fingerprints:
        Mapping of dataset id -> fingerprint dict.

    Returns
    -------
    str
        The Markdown document.
    """
    generated = datetime.now(timezone.utc).isoformat()
    header = [
        "# NetSentinel — Dataset Audit Report",
        "",
        f"_Generated: {generated}_",
        "",
        f"Datasets audited: **{len(reports)}**",
        "",
        "> Phase 1A audit. No preprocessing, feature engineering or training "
        "has been performed — this report informs those later stages.",
        "",
        "---",
        "",
    ]
    sections = [
        _dataset_section(reports[ds_id], fingerprints.get(ds_id, {}))
        for ds_id in reports
    ]
    return "\n".join(header) + "\n".join(sections)

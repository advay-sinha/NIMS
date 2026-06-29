"""Dataset validation and report orchestration.

Purpose
-------
Validate a raw dataset and orchestrate assembly of its data report. Statistical
profiling is delegated to :mod:`src.data.statistics`; this module owns only:

    - schema validation (expected columns present / unexpected absent),
    - datatype validation (configured roles match observed dtypes),
    - required-column validation (label + categorical columns),
    - rule validation (quality thresholds: duplicates, missing, infinities,
      severe class imbalance, all-null columns),
    - orchestration + :class:`ValidationReport` / :class:`DataReport` generation.

Checks are defensive: problems surface as structured issues rather than being
silently coerced (CLAUDE.md > Error Handling). This module is READ-ONLY — it
never mutates the frame and never preprocesses, encodes, scales or splits.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.data import statistics

logger = logging.getLogger(__name__)

# Quality-rule thresholds (named to avoid magic numbers).
SEVERE_IMBALANCE_RATIO: float = 10.0
"""Imbalance ratio (majority/minority) at or above which a warning is raised."""


@dataclass
class ValidationIssue:
    """A single validation finding.

    Attributes
    ----------
    severity:
        ``"error"`` | ``"warning"`` | ``"info"``.
    code:
        Short machine-readable code, e.g. ``"missing_label_column"``.
    message:
        Human-readable description.
    """

    severity: str
    code: str
    message: str


@dataclass
class ValidationReport:
    """Aggregated validation result for one dataset."""

    dataset_id: str
    passed: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str) -> None:
        """Record an issue; any ``"error"`` flips :attr:`passed` to ``False``."""
        self.issues.append(ValidationIssue(severity, code, message))
        if severity == "error":
            self.passed = False

    def as_list(self) -> list[dict[str, str]]:
        """Return issues as a list of plain dicts (JSON-friendly)."""
        return [asdict(issue) for issue in self.issues]


@dataclass
class DataReport:
    """Self-describing report: dataset identity + statistics + validation."""

    dataset_id: str
    name: str | None
    engine: str | None
    label_column: str | None
    statistics: dict[str, Any]
    schema_passed: bool
    validation_issues: list[dict[str, str]]

    # -- convenience accessors (read-through to statistics) ----------------- #
    @property
    def n_rows(self) -> int:
        """Row count (from statistics)."""
        return int(self.statistics.get("n_rows", 0))

    @property
    def n_features(self) -> int:
        """Feature/column count (from statistics)."""
        return int(self.statistics.get("n_features", 0))

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "engine": self.engine,
            "label_column": self.label_column,
            "statistics": self.statistics,
            "schema_passed": self.schema_passed,
            "validation_issues": self.validation_issues,
        }


# --------------------------------------------------------------------------- #
# Validation rules                                                            #
# --------------------------------------------------------------------------- #
def validate_schema(
    frame: "Any",
    dataset_config: Mapping[str, Any],
    report: ValidationReport,
) -> ValidationReport:
    """Validate the declared ``columns`` schema (warnings for absences)."""
    present = set(map(str, frame.columns))
    expected = dataset_config.get("columns")
    if expected:
        for col in sorted(set(expected) - present):
            report.add("warning", "missing_column", f"expected column absent: {col}")
    return report


def validate_required_columns(
    frame: "Any",
    dataset_config: Mapping[str, Any],
    report: ValidationReport,
) -> ValidationReport:
    """Validate that required columns (label + categoricals) are present.

    A missing label column is an ``error``; missing configured categorical
    columns are ``warning``.
    """
    present = set(map(str, frame.columns))

    label_column = dataset_config.get("label_column")
    if label_column is not None and label_column not in present:
        report.add(
            "error",
            "missing_label_column",
            f"configured label column '{label_column}' not found",
        )

    for col in dataset_config.get("categorical_columns", []) or []:
        if col not in present:
            report.add(
                "warning",
                "missing_categorical_column",
                f"configured categorical column absent: {col}",
            )
    return report


def validate_datatypes(
    frame: "Any",
    dataset_config: Mapping[str, Any],
    report: ValidationReport,
) -> ValidationReport:
    """Validate observed dtypes against configured roles.

    Flags configured categorical columns that arrived as numeric dtype (often a
    sign of silent label-like encoding) as a ``warning``.
    """
    import numpy as np

    numeric_cols = set(frame.select_dtypes(include=[np.number]).columns)
    for col in dataset_config.get("categorical_columns", []) or []:
        if col in numeric_cols:
            report.add(
                "warning",
                "categorical_stored_as_numeric",
                f"categorical column '{col}' has a numeric dtype",
            )
    return report


def validate_rules(
    stats: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    report: ValidationReport,
) -> ValidationReport:
    """Validate quality rules using precomputed statistics.

    Uses the statistics dict (no recomputation) to raise warnings for
    duplicate rows, missing values, infinite values and severe class imbalance.

    Parameters
    ----------
    stats:
        Output of :func:`src.data.statistics.dataset_statistics`.
    dataset_config:
        The ``dataset`` config block.
    report:
        Report to append findings to.
    """
    if stats.get("duplicate_rows", 0) > 0:
        report.add(
            "warning",
            "duplicate_rows",
            f"{stats['duplicate_rows']} duplicate rows present",
        )

    if stats.get("total_missing", 0) > 0:
        report.add(
            "warning",
            "missing_values",
            f"{stats['total_missing']} missing values across "
            f"{len(stats.get('missing_values', {}))} columns",
        )

    infinite = stats.get("infinite_values", {})
    if infinite:
        report.add(
            "warning",
            "infinite_values",
            f"infinite values in {len(infinite)} column(s): "
            f"{', '.join(sorted(infinite))}",
        )

    ratio = stats.get("class_imbalance", {}).get("imbalance_ratio")
    if ratio is not None and ratio >= SEVERE_IMBALANCE_RATIO:
        report.add(
            "warning",
            "severe_class_imbalance",
            f"class imbalance ratio {ratio:.1f} >= {SEVERE_IMBALANCE_RATIO}",
        )

    # All-null columns carry no signal and break downstream stages.
    for col, n_missing in stats.get("missing_values", {}).items():
        if n_missing == stats.get("n_rows"):
            report.add("error", "all_null_column", f"column '{col}' is entirely null")

    return report


# --------------------------------------------------------------------------- #
# Orchestration / persistence                                                 #
# --------------------------------------------------------------------------- #
def validate_dataset(
    frame: "Any",
    dataset_config: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> ValidationReport:
    """Run every validation against ``frame`` and return the report.

    Parameters
    ----------
    frame:
        Raw DataFrame.
    dataset_config:
        The ``dataset`` config block.
    stats:
        Precomputed statistics (for rule validation, no recomputation).

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport(dataset_id=dataset_config.get("id", "unknown"))
    validate_schema(frame, dataset_config, report)
    validate_required_columns(frame, dataset_config, report)
    validate_datatypes(frame, dataset_config, report)
    validate_rules(stats, dataset_config, report)
    return report


def build_report(
    frame: "Any",
    dataset_config: Mapping[str, Any],
    label_column: str | None = None,
) -> DataReport:
    """Profile and validate a raw dataset into a :class:`DataReport`.

    Profiling is delegated to :mod:`src.data.statistics`; validation runs
    against the same computed statistics.

    Parameters
    ----------
    frame:
        Raw DataFrame (already ingested by a loader).
    dataset_config:
        The ``dataset`` config block.
    label_column:
        Label column override. Defaults to ``dataset_config["label_column"]``.

    Returns
    -------
    DataReport
    """
    label = (
        label_column if label_column is not None else dataset_config.get("label_column")
    )
    categorical = tuple(dataset_config.get("categorical_columns", []) or [])

    stats = statistics.dataset_statistics(frame, label, categorical)
    validation = validate_dataset(frame, dataset_config, stats)

    report = DataReport(
        dataset_id=dataset_config.get("id", "unknown"),
        name=dataset_config.get("name"),
        engine=dataset_config.get("engine"),
        label_column=label,
        statistics=stats,
        schema_passed=validation.passed,
        validation_issues=validation.as_list(),
    )
    logger.info(
        "[%s] report: rows=%s features=%s missing=%s duplicates=%s passed=%s",
        report.dataset_id,
        report.n_rows,
        report.n_features,
        stats["total_missing"],
        stats["duplicate_rows"],
        report.schema_passed,
    )
    return report


def save_report(report: DataReport, path: str | Path) -> Path:
    """Persist a :class:`DataReport` to JSON (parent dirs created)."""
    from src.utils.io import write_json

    return write_json(report.to_dict(), path)

"""Telemetry schema validation.

Purpose
-------
Verify a loaded telemetry frame against its configured schema before any
preprocessing: required columns, timestamp parseability, numeric payload,
duplicates, missing values, impossible negatives/bounds, counter
monotonicity and per-device/interface coverage. Violations are collected as
issues (``error`` blocks the pipeline; ``warning`` is informational), never
raised mid-check — the caller decides.

Outputs
-------
A :class:`ValidationReport` (persisted as JSON + Markdown by
:mod:`src.network_health.artifacts`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.network_health.schema import TelemetrySchema

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Outcome of one telemetry validation run."""

    dataset_id: str
    n_rows: int = 0
    n_devices: int = 0
    n_interfaces: int = 0
    issues: list[dict[str, str]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: str, check: str, message: str) -> None:
        """Record one issue."""
        self.issues.append(
            {"severity": severity, "check": check, "message": message}
        )

    @property
    def passed(self) -> bool:
        """True when no error-severity issue was found."""
        return not any(i["severity"] == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "dataset_id": self.dataset_id,
            "passed": self.passed,
            "n_rows": self.n_rows,
            "n_devices": self.n_devices,
            "n_interfaces": self.n_interfaces,
            "n_errors": sum(1 for i in self.issues if i["severity"] == "error"),
            "n_warnings": sum(1 for i in self.issues if i["severity"] == "warning"),
            "issues": self.issues,
            "coverage": self.coverage,
        }


def validate_telemetry(
    frame: "Any", schema: TelemetrySchema, dataset_id: str
) -> ValidationReport:
    """Run every schema check on a telemetry frame.

    Parameters
    ----------
    frame:
        Loaded telemetry rows.
    schema:
        The dataset's configured schema.
    dataset_id:
        Identity recorded in the report.

    Returns
    -------
    ValidationReport
    """
    import pandas as pd

    report = ValidationReport(dataset_id=dataset_id, n_rows=int(len(frame)))

    missing_required = [
        c for c in schema.required_columns if c not in frame.columns
    ]
    if missing_required:
        report.add(
            "error", "required_columns",
            f"Missing required column(s): {', '.join(missing_required)}",
        )
        return report  # remaining checks need the columns

    # Timestamp parseability.
    timestamps = pd.to_datetime(
        frame[schema.timestamp_column], errors="coerce", format="mixed"
    )
    n_bad = int(timestamps.isna().sum())
    if n_bad:
        report.add(
            "error", "timestamp",
            f"{n_bad} row(s) have unparseable '{schema.timestamp_column}' values.",
        )

    # Numeric telemetry payload.
    for column in schema.numeric_columns:
        if column not in frame.columns:
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        n_non_numeric = int(coerced.isna().sum() - frame[column].isna().sum())
        if n_non_numeric:
            report.add(
                "error", "numeric",
                f"Column '{column}' has {n_non_numeric} non-numeric value(s).",
            )

    # Duplicates: one reading per (timestamp, device, interface).
    key = [schema.timestamp_column, *schema.series_columns]
    n_duplicates = int(frame.duplicated(subset=key).sum())
    if n_duplicates:
        report.add(
            "warning", "duplicates",
            f"{n_duplicates} duplicate reading(s) for {key} (removed in "
            f"preprocessing).",
        )

    # Missing values.
    missing = frame[list(schema.required_columns)].isna().sum()
    for column, count in missing[missing > 0].items():
        report.add(
            "warning", "missing_values",
            f"Column '{column}' has {int(count)} missing value(s).",
        )

    # Impossible negatives / bound violations.
    for column in schema.non_negative_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        n_negative = int((values < 0).sum())
        if n_negative:
            report.add(
                "error", "negative_values",
                f"Column '{column}' has {n_negative} negative value(s).",
            )
    for column, (low, high) in schema.bounded_columns.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        n_out = int(((values < low) | (values > high)).sum())
        if n_out:
            report.add(
                "warning", "bounds",
                f"Column '{column}' has {n_out} value(s) outside "
                f"[{low}, {high}] (clipped in preprocessing).",
            )

    # Counter monotonicity per series (decreases = resets/wraps).
    if not timestamps.isna().any():
        ordered = frame.assign(**{schema.timestamp_column: timestamps}).sort_values(
            [*schema.series_columns, schema.timestamp_column]
        )
        for column in schema.counter_columns:
            if column not in ordered.columns:
                continue
            deltas = ordered.groupby(schema.series_columns, sort=False)[column].diff()
            n_decreases = int((deltas < 0).sum())
            if n_decreases:
                report.add(
                    "warning", "counter_monotonicity",
                    f"Counter '{column}' decreases {n_decreases} time(s) "
                    f"(resets/wraps; deltas are clipped in preprocessing).",
                )

    # Per-device / per-interface coverage.
    report.n_devices = int(frame[schema.device_column].nunique())
    report.n_interfaces = int(
        frame.groupby(schema.device_column)[schema.interface_column]
        .nunique().sum()
    )
    rows_per_series = frame.groupby(schema.series_columns).size()
    report.coverage = {
        "rows_per_series_min": int(rows_per_series.min()),
        "rows_per_series_max": int(rows_per_series.max()),
        "rows_per_series_mean": float(rows_per_series.mean()),
        "n_series": int(len(rows_per_series)),
    }

    logger.info(
        "Validation for '%s': %s (%d error(s), %d warning(s)).",
        dataset_id, "PASSED" if report.passed else "FAILED",
        sum(1 for i in report.issues if i["severity"] == "error"),
        sum(1 for i in report.issues if i["severity"] == "warning"),
    )
    return report

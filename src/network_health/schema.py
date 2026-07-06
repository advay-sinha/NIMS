"""Telemetry schema definition.

Purpose
-------
One typed object describing a network-health telemetry dataset — column
roles, required columns and value bounds — built from configuration so no
column name is ever hardcoded in pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TelemetrySchema:
    """Column roles and constraints of one telemetry dataset.

    Attributes
    ----------
    timestamp_column, device_column, interface_column:
        Identity columns of each reading.
    label_column:
        Optional anomaly label column (``None`` for unlabeled telemetry).
    required_columns:
        Columns every file must provide.
    counter_columns:
        Monotonic SNMP counters (delta/rate source).
    gauge_columns:
        Instantaneous metrics.
    status_columns:
        Operational/administrative status columns.
    bounded_columns:
        ``{column: (min, max)}`` value ranges.
    """

    timestamp_column: str = "timestamp"
    device_column: str = "device_id"
    interface_column: str = "interface_id"
    label_column: str | None = None
    required_columns: tuple[str, ...] = ()
    counter_columns: tuple[str, ...] = ()
    gauge_columns: tuple[str, ...] = ()
    status_columns: tuple[str, ...] = ()
    bounded_columns: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, schema_config: Mapping[str, Any]) -> "TelemetrySchema":
        """Build a schema from the ``network_health.schema`` config block."""
        bounds = {
            str(column): (float(low), float(high))
            for column, (low, high) in dict(
                schema_config.get("bounded_columns") or {}
            ).items()
        }
        return cls(
            timestamp_column=str(schema_config.get("timestamp_column", "timestamp")),
            device_column=str(schema_config.get("device_column", "device_id")),
            interface_column=str(
                schema_config.get("interface_column", "interface_id")
            ),
            label_column=schema_config.get("label_column"),
            required_columns=tuple(schema_config.get("required_columns") or ()),
            counter_columns=tuple(schema_config.get("counter_columns") or ()),
            gauge_columns=tuple(schema_config.get("gauge_columns") or ()),
            status_columns=tuple(schema_config.get("status_columns") or ()),
            bounded_columns=bounds,
        )

    @property
    def series_columns(self) -> list[str]:
        """Grouping key of one telemetry series (device + interface)."""
        return [self.device_column, self.interface_column]

    @property
    def numeric_columns(self) -> list[str]:
        """Counter and gauge columns (the numeric telemetry payload)."""
        return list(self.counter_columns) + list(self.gauge_columns)

    @property
    def non_negative_columns(self) -> list[str]:
        """Columns where negative values are physically impossible."""
        return self.numeric_columns

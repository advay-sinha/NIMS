"""Unified streaming event model.

A single typed event schema that normalises signals from every engine so the
monitoring state and the dashboard can treat them uniformly. Pure value objects
— no IO, no execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

# --- event types ------------------------------------------------------------
CYBER_ALERT = "cyber_alert"
HEALTH_ANOMALY = "health_anomaly"
CONFIG_FINDING = "config_finding"
TOPOLOGY_WARNING = "topology_warning"
REMEDIATION_PLAN = "remediation_plan"
CORRELATION_INCIDENT = "correlation_incident"
SAFETY_STATUS = "safety_status"
SYSTEM_STATUS = "system_status"

EVENT_TYPES: tuple[str, ...] = (
    CYBER_ALERT, HEALTH_ANOMALY, CONFIG_FINDING, TOPOLOGY_WARNING,
    REMEDIATION_PLAN, CORRELATION_INCIDENT, SAFETY_STATUS, SYSTEM_STATUS)

# --- source engines ---------------------------------------------------------
ENGINE_A = "engine_a"
ENGINE_B = "engine_b"
ENGINE_C = "engine_c"
CORRELATION = "correlation"
SYSTEM = "system"

SEVERITY_ORDER: dict[str, int] = {
    "critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _digest(*parts: Any) -> str:
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def event_id(event_type: str, source_engine: str, entity_id: str,
             title: str, ref: str = "") -> str:
    """Deterministic event id from the event's natural key."""
    return f"EVT-{_digest(event_type, source_engine, entity_id, title, ref)}"


@dataclass(frozen=True)
class StreamEvent:
    """One normalised monitoring event replayed from a persisted artefact."""

    event_id: str
    event_type: str
    source_engine: str
    severity: str
    title: str
    summary: str
    entity_type: str
    entity_id: str
    timestamp: Optional[str] = None      # source time if known
    emitted_at: Optional[str] = None     # replay/emit time (set by the replayer)
    seq: Optional[int] = None            # monotonic order (set by the replayer)
    device_id: Optional[str] = None
    interface_id: Optional[str] = None
    vlan_id: Optional[str] = None
    incident_id: Optional[str] = None
    signal_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def with_emission(self, seq: int, emitted_at: str) -> "StreamEvent":
        """Return a copy stamped with a replay sequence and emit time."""
        from dataclasses import replace
        return replace(self, seq=seq, emitted_at=emitted_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_engine": self.source_engine,
            "severity": self.severity,
            "title": self.title,
            "summary": self.summary,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "timestamp": self.timestamp,
            "emitted_at": self.emitted_at,
            "seq": self.seq,
            "device_id": self.device_id,
            "interface_id": self.interface_id,
            "vlan_id": self.vlan_id,
            "incident_id": self.incident_id,
            "signal_id": self.signal_id,
            "payload": self.payload,
        }

    def to_row(self) -> dict[str, Any]:
        """Flat mapping for compact display (payload dropped)."""
        row = self.to_dict()
        row.pop("payload", None)
        return row


def event_from_dict(data: dict[str, Any]) -> StreamEvent:
    """Rebuild a :class:`StreamEvent` from a serialised mapping (log replay)."""
    known = {f for f in StreamEvent.__dataclass_fields__}
    return StreamEvent(**{k: v for k, v in data.items() if k in known})

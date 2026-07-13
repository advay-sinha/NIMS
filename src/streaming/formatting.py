"""Display helpers for the streaming layer (pure Python, no Streamlit).

Reshapes events/state into table rows and provides stable labels and the
offline banner text. No IO, mutates nothing.
"""

from __future__ import annotations

from typing import Any

from src.streaming import models as m

EVENT_TYPE_LABEL: dict[str, str] = {
    m.CYBER_ALERT: "Cyber alert",
    m.HEALTH_ANOMALY: "Health anomaly",
    m.CONFIG_FINDING: "Config finding",
    m.TOPOLOGY_WARNING: "Topology warning",
    m.REMEDIATION_PLAN: "Remediation plan (dry-run)",
    m.CORRELATION_INCIDENT: "Correlated incident",
    m.SAFETY_STATUS: "Safety status",
    m.SYSTEM_STATUS: "System status",
}

STREAM_SAFETY_BANNER = (
    "Live demo replay of persisted artefacts — offline only. No device access, "
    "no packet capture, no command execution.")


def event_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact table rows for a list of event dicts (newest first)."""
    rows = []
    for e in events:
        rows.append({
            "seq": e.get("seq"),
            "type": EVENT_TYPE_LABEL.get(e.get("event_type"), e.get("event_type")),
            "severity": e.get("severity"),
            "engine": e.get("source_engine"),
            "device": e.get("device_id") or "-",
            "title": e.get("title"),
            "emitted_at": e.get("emitted_at"),
        })
    return rows


def sort_by_severity(items: list[dict[str, Any]], key: str = "severity"
                     ) -> list[dict[str, Any]]:
    """Order items most-severe first (stable)."""
    return sorted(items, key=lambda i: m.SEVERITY_ORDER.get(
        str(i.get(key, "info")).lower(), 9))

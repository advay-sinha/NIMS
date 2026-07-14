"""In-memory monitoring state maintained from the replayed event stream.

Applies events one at a time and keeps a compact, JSON-serialisable snapshot
(counts, active incidents, active devices, recent events, safety posture) that
the dashboard can poll. No IO here — persistence lives in ``artifacts.py``.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Optional

from src.streaming import models as m
from src.streaming.models import StreamEvent

_RECENT_MAX = 50


@dataclass
class MonitoringState:
    """Live-updating monitoring state (updated by :meth:`apply`)."""

    started_at: Optional[str] = None
    last_event_at: Optional[str] = None
    total_events: int = 0
    events_by_type: Counter = field(default_factory=Counter)
    events_by_severity: Counter = field(default_factory=Counter)
    events_by_engine: Counter = field(default_factory=Counter)
    active_incidents: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_devices: set[str] = field(default_factory=set)
    recent: deque = field(default_factory=lambda: deque(maxlen=_RECENT_MAX))
    # --- Phase 13 syslog tracking (backward-compatible additions) ---
    syslog_event_count: int = 0
    syslog_findings_count: int = 0
    syslog_categories: Counter = field(default_factory=Counter)
    clock_reliability_status: str = "reliable"
    incidents_with_syslog_evidence: int = 0

    def apply(self, event: StreamEvent) -> None:
        """Fold one event into the state."""
        self.total_events += 1
        self.started_at = self.started_at or event.emitted_at or event.timestamp
        self.last_event_at = event.emitted_at or event.timestamp or self.last_event_at
        self.events_by_type[event.event_type] += 1
        self.events_by_severity[event.severity] += 1
        self.events_by_engine[event.source_engine] += 1

        if event.source_engine == m.SYSLOG:
            self._apply_syslog(event)
        if event.event_type == m.CORRELATION_INCIDENT and event.incident_id:
            self.active_incidents[event.incident_id] = {
                "incident_id": event.incident_id, "severity": event.severity,
                "title": event.title, "device_id": event.device_id,
                "emitted_at": event.emitted_at}
            if int((event.payload or {}).get("syslog_signal_count", 0) or 0) > 0:
                self.incidents_with_syslog_evidence += 1
        if event.device_id:
            self.active_devices.add(event.device_id)
        self.recent.append(event.to_row())

    def _apply_syslog(self, event: StreamEvent) -> None:
        """Track syslog-origin event categories and clock reliability."""
        self.syslog_event_count += 1
        category = str((event.payload or {}).get("syslog_category", "other"))
        self.syslog_categories[category] += 1
        if event.event_type in (m.CONFIG_FINDING, m.TOPOLOGY_WARNING,
                                m.CYBER_ALERT):
            self.syslog_findings_count += 1
        if category in ("clock_integrity",) or event.event_type == m.SAFETY_STATUS:
            if "clock" in event.title.lower():
                self.clock_reliability_status = "degraded"

    # ------------------------------------------------------------ snapshots

    def _critical_count(self) -> int:
        return sum(1 for i in self.active_incidents.values()
                   if str(i.get("severity", "")).lower() in ("critical", "high"))

    def snapshot(self) -> dict[str, Any]:
        """Compact, JSON-serialisable current-state view for the dashboard."""
        return {
            "started_at": self.started_at,
            "last_event_at": self.last_event_at,
            "total_events": self.total_events,
            "events_by_type": dict(self.events_by_type),
            "events_by_severity": dict(self.events_by_severity),
            "events_by_engine": dict(self.events_by_engine),
            "active_incident_count": len(self.active_incidents),
            "critical_incident_count": self._critical_count(),
            "active_incidents": sorted(
                self.active_incidents.values(),
                key=lambda i: m.SEVERITY_ORDER.get(
                    str(i.get("severity", "info")).lower(), 9)),
            "active_device_count": len(self.active_devices),
            "active_devices": sorted(self.active_devices),
            "recent_events": list(self.recent)[-20:][::-1],
            "syslog": self._syslog_snapshot(),
            "safety": {
                "offline_only": True,
                "no_device_access": True,
                "no_packet_capture": True,
                "no_command_execution": True,
                "mode": "demo_replay",
            },
        }

    def _syslog_snapshot(self) -> dict[str, Any]:
        """Compact syslog-activity view for the dashboard."""
        cats = self.syslog_categories
        return {
            "syslog_event_count": self.syslog_event_count,
            "syslog_findings_count": self.syslog_findings_count,
            "top_syslog_categories": dict(cats.most_common(8)),
            "clock_reliability_status": self.clock_reliability_status,
            "management_auth_activity": cats.get("snmp_auth", 0)
            + cats.get("management_access", 0),
            "port_instability_count": cats.get("port_instability", 0),
            "loop_redundancy_candidates": cats.get("loop_redundancy", 0),
            "duplicate_ip_count": cats.get("duplicate_ip", 0),
            "poe_fault_count": cats.get("poe", 0),
            "incidents_with_syslog_evidence": self.incidents_with_syslog_evidence,
        }

    def summary(self) -> dict[str, Any]:
        """A smaller roll-up (counts + safety) for ``stream_summary.json``."""
        return {
            "total_events": self.total_events,
            "events_by_type": dict(self.events_by_type),
            "events_by_severity": dict(self.events_by_severity),
            "events_by_engine": dict(self.events_by_engine),
            "active_incident_count": len(self.active_incidents),
            "critical_incident_count": self._critical_count(),
            "active_device_count": len(self.active_devices),
            "syslog_event_count": self.syslog_event_count,
            "syslog_findings_count": self.syslog_findings_count,
            "clock_reliability_status": self.clock_reliability_status,
            "started_at": self.started_at,
            "last_event_at": self.last_event_at,
            "safety_note": ("Offline demo replay — no device access, no packet "
                            "capture, no command execution."),
        }

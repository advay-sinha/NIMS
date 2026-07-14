"""Streaming source: replay a syslog-ingestion run as normalised events.

Reads the artefacts written by ``scripts.ingest_switch_syslog`` (via the
read-only dashboard loader) and yields :class:`~src.streaming.models.StreamEvent`
objects for the monitoring state / dashboard:

    * ``cyber_alert``       — syslog security findings (SNMP / ARP / duplicate-IP)
    * ``config_finding``    — syslog interface / PoE / device findings
    * ``topology_warning``  — syslog MAC-flap / ERPS findings
    * ``health_anomaly``    — Engine B weak-label positive windows
    * ``safety_status``     — clock-integrity / offline-posture status
    * ``system_status``     — parser summary heartbeat

All events carry ``source_engine = "syslog"`` and a ``syslog_category`` in their
payload so :class:`~src.streaming.state.MonitoringState` can track syslog
activity distinctly. This source is disabled by default
(``streaming.yaml`` ``sources.syslog``); like every source it only reads local
files — it never contacts a device, polls SNMP, captures packets or executes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.dashboard import loader as dash
from src.streaming import models as m
from src.streaming.models import StreamEvent, event_id

logger = logging.getLogger(__name__)

# Finding category -> (stream event type, coarse syslog category).
_SECURITY_CATEGORIES = {"security"}
_TOPOLOGY_CATEGORIES = {"loop", "topology"}


def _opt(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _syslog_category(finding: dict[str, Any]) -> str:
    """Coarse category label for state aggregation, from a finding's rule_id."""
    return {
        "SYS-SNMP-AUTHFAIL": "snmp_auth",
        "SYS-PORT-FLAP": "port_instability",
        "SYS-MAC-FLAP": "loop_redundancy",
        "SYS-ERPS-CHURN": "loop_redundancy",
        "SYS-ARP-FAST": "duplicate_ip",
        "SYS-POE-FAULT": "poe",
        "SYS-TELNET": "management_access",
        "SYS-DEVICE": "device_health",
    }.get(str(finding.get("rule_id", "")), "other")


def _finding_event(f: dict[str, Any], run_id: str, ts: Optional[str]
                   ) -> StreamEvent:
    """Map one syslog finding to its stream event by category (engine=syslog)."""
    category = str(f.get("category", ""))
    device, interface = _opt(f.get("device")), _opt(f.get("interface"))
    title = str(f.get("title", f.get("rule_id", "syslog finding")))
    if category in _SECURITY_CATEGORIES:
        event_type = m.CYBER_ALERT
    elif category in _TOPOLOGY_CATEGORIES:
        event_type = m.TOPOLOGY_WARNING
    else:
        event_type = m.CONFIG_FINDING
    payload = dict(f)
    payload["syslog_category"] = _syslog_category(f)
    return StreamEvent(
        event_id=event_id(event_type, m.SYSLOG, str(f.get("finding_id", "")), title),
        event_type=event_type, source_engine=m.SYSLOG,
        severity=str(f.get("severity", "info")), title=title,
        summary=str(f.get("evidence") or f.get("recommendation") or ""),
        entity_type="interface" if interface else "device",
        entity_id=interface or device or run_id,
        timestamp=_opt(f.get("last_seen")) or ts,
        device_id=device, interface_id=interface, vlan_id=_opt(f.get("vlan")),
        payload=payload)


def events_from_syslog(syslog_dir, run_id: str) -> list[StreamEvent]:
    """Load a syslog run and emit its findings, weak-label anomalies and status."""
    data = dash.load_syslog_run(syslog_dir, run_id)
    if not data.get("available"):
        return []

    summary = data.get("summary", {})
    ts = (summary.get("time_range") or {}).get("last")
    events: list[StreamEvent] = []

    # findings -> config_finding / cyber_alert / topology_warning
    for f in data.get("findings", []):
        events.append(_finding_event(f, run_id, ts))

    # weak labels -> health_anomaly (one aggregate event per firing label)
    positives = (data.get("weak_label_summary", {}) or {}).get(
        "positive_windows", {}) or {}
    for label, count in positives.items():
        if not count:
            continue
        severity = "high" if count >= 20 else "medium" if count >= 5 else "low"
        events.append(StreamEvent(
            event_id=event_id(m.HEALTH_ANOMALY, m.SYSLOG, run_id, label),
            event_type=m.HEALTH_ANOMALY, source_engine=m.SYSLOG,
            severity=severity, title=f"Syslog weak label: {label}",
            summary=f"{count} window(s) crossed the {label} threshold "
                    "(heuristic, not a verified incident).",
            entity_type="dataset", entity_id=run_id, timestamp=_opt(ts),
            payload={"label": label, "windows": count,
                     "syslog_category": "health_anomaly"}))

    # clock integrity -> safety_status (chronology reliability)
    clock_unreliable = int(summary.get("clock_unreliable_events", 0) or 0)
    if clock_unreliable > 0:
        events.append(StreamEvent(
            event_id=event_id(m.SAFETY_STATUS, m.SYSLOG, run_id, "clock"),
            event_type=m.SAFETY_STATUS, source_engine=m.SYSLOG, severity="low",
            title="Syslog clock integrity notice",
            summary=(f"{clock_unreliable} unreliable-clock event(s); event "
                     "ordering and time-based correlation may be approximate."),
            entity_type="system", entity_id=run_id, timestamp=_opt(ts),
            payload={"clock_unreliable_events": clock_unreliable,
                     "syslog_category": "clock_integrity"}))

    # parser summary -> system_status
    events.append(StreamEvent(
        event_id=event_id(m.SYSTEM_STATUS, m.SYSLOG, run_id, "syslog"),
        event_type=m.SYSTEM_STATUS, source_engine=m.SYSLOG, severity="info",
        title=f"Industrial syslog ingested ({run_id})",
        summary=f"{summary.get('parsed_events', 0)} events parsed from "
                f"{len(summary.get('input_files', []))} file(s); "
                f"hosts: {', '.join(summary.get('hosts', [])) or 'n/a'}.",
        entity_type="system", entity_id=run_id, timestamp=_opt(ts),
        payload={"parser_summary": summary, "syslog_category": "status"}))

    logger.info("Collected %d syslog event(s) from run %s.", len(events), run_id)
    return events

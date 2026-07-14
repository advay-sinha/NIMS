"""Event sources — turn persisted artefacts into normalised stream events.

Every source reads local files only (via the Streamlit-free dashboard loader)
and yields :class:`~src.streaming.models.StreamEvent` objects. No source ever
contacts a device, polls SNMP, captures packets or runs a pipeline — they replay
what other phases already wrote to disk.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.dashboard import loader as dash
from src.streaming import models as m
from src.streaming.models import StreamEvent, event_id

logger = logging.getLogger(__name__)


def _cfg(config: dict[str, Any], dotted: str, default: Any) -> Any:
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _opt(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ------------------------------------------------------------- per-engine


def events_from_correlation(correlation_dir, correlation_id: str
                            ) -> list[StreamEvent]:
    """Correlation incidents -> ``correlation_incident`` events."""
    data = dash.load_correlation(correlation_dir, correlation_id)
    if not data.get("available"):
        return []
    ts = (data.get("summary") or {}).get("timestamp")
    events: list[StreamEvent] = []
    for inc in data.get("incidents", []):
        devices = inc.get("affected_devices", []) or []
        title = str(inc.get("title", "incident"))
        iid = str(inc.get("incident_id", ""))
        events.append(StreamEvent(
            event_id=event_id(m.CORRELATION_INCIDENT, m.CORRELATION, iid, title),
            event_type=m.CORRELATION_INCIDENT, source_engine=m.CORRELATION,
            severity=str(inc.get("severity", "info")), title=title,
            summary=str(inc.get("root_cause_hypothesis", "")),
            entity_type="incident", entity_id=iid, timestamp=_opt(ts),
            device_id=_opt(devices[0]) if devices else None,
            incident_id=iid, payload=inc))
    return events


def events_from_engine_c(network_config_dir, snapshot_id: str) -> list[StreamEvent]:
    """Engine C findings / topology warnings / remediation -> events."""
    data = dash.load_engine_c_dashboard(network_config_dir, snapshot_id)
    if not data.get("available"):
        return []
    views = data.get("views", {})
    meta = views.get("export_metadata") or {}
    ts = _opt(meta.get("generated_at"))
    events: list[StreamEvent] = []

    for f in (views.get("findings_view") or {}).get("findings", []):
        device, interface = _opt(f.get("device")), _opt(f.get("interface"))
        title = str(f.get("title", f.get("rule_id", "finding")))
        events.append(StreamEvent(
            event_id=event_id(m.CONFIG_FINDING, m.ENGINE_C,
                              str(f.get("finding_id", "")), title),
            event_type=m.CONFIG_FINDING, source_engine=m.ENGINE_C,
            severity=str(f.get("severity", "info")), title=title,
            summary=str(f.get("evidence") or f.get("recommendation") or ""),
            entity_type="interface" if interface else "device",
            entity_id=interface or device or snapshot_id, timestamp=ts,
            device_id=device, interface_id=interface, vlan_id=_opt(f.get("vlan")),
            payload=f))

    for w in (views.get("topology_view") or {}).get("warnings", []):
        device, interface = _opt(w.get("device")), _opt(w.get("interface"))
        sev = {"warning": "medium", "info": "info"}.get(
            str(w.get("severity", "info")), str(w.get("severity", "info")))
        events.append(StreamEvent(
            event_id=event_id(m.TOPOLOGY_WARNING, m.ENGINE_C,
                              str(w.get("warning_id", "")), str(w.get("message"))),
            event_type=m.TOPOLOGY_WARNING, source_engine=m.ENGINE_C,
            severity=sev, title=str(w.get("message", "topology warning")),
            summary=str(w.get("evidence") or ""), entity_type="interface",
            entity_id=interface or device or snapshot_id, timestamp=ts,
            device_id=device, interface_id=interface, payload=w))

    for a in (views.get("remediation_view") or {}).get("command_actions", []):
        device, interface = _opt(a.get("device")), _opt(a.get("interface"))
        title = str(a.get("title", "remediation plan"))
        events.append(StreamEvent(
            event_id=event_id(m.REMEDIATION_PLAN, m.ENGINE_C,
                              str(a.get("action_id", "")), title),
            event_type=m.REMEDIATION_PLAN, source_engine=m.ENGINE_C,
            severity=str(a.get("risk_level", "medium")),
            title=f"Dry-run plan: {title}",
            summary="Dry-run only; requires explicit human confirmation.",
            entity_type="interface" if interface else "device",
            entity_id=interface or device or snapshot_id, timestamp=ts,
            device_id=device, interface_id=interface, payload=a))
    return events


def events_from_engine_b(network_health_dir) -> list[StreamEvent]:
    """Engine B anomaly summaries -> ``health_anomaly`` events."""
    data = dash.load_engine_b(network_health_dir)
    if not data.get("available"):
        return []
    events: list[StreamEvent] = []
    for ds in data.get("datasets", []):
        rate = float(ds.get("anomaly_rate") or 0.0)
        if rate <= 0:
            continue
        severity = "high" if rate >= 0.2 else "medium" if rate >= 0.05 else "low"
        name = str(ds.get("dataset"))
        events.append(StreamEvent(
            event_id=event_id(m.HEALTH_ANOMALY, m.ENGINE_B, name,
                              str(ds.get("experiment_id"))),
            event_type=m.HEALTH_ANOMALY, source_engine=m.ENGINE_B,
            severity=severity,
            title=f"Network-health anomalies ({name})",
            summary=f"{rate:.1%} of test rows flagged anomalous (aggregate).",
            entity_type="dataset", entity_id=name, payload=ds))
    return events


def events_from_engine_a(registry_dir, reports_dir, error_analysis_dir,
                         visualizations_dir, experiments_dir) -> list[StreamEvent]:
    """Engine A production models -> aggregate ``cyber_alert`` events."""
    data = dash.load_engine_a(registry_dir, reports_dir, error_analysis_dir,
                              visualizations_dir, experiments_dir)
    if not data.get("available"):
        return []
    events: list[StreamEvent] = []
    for model in data.get("models", []):
        name = str(model.get("dataset"))
        events.append(StreamEvent(
            event_id=event_id(m.CYBER_ALERT, m.ENGINE_A, name,
                              str(model.get("experiment_id"))),
            event_type=m.CYBER_ALERT, source_engine=m.ENGINE_A, severity="info",
            title=f"Intrusion model active ({name})",
            summary=f"Model '{model.get('model_type')}' promoted; "
                    "aggregate coverage indicator (no live per-flow alert).",
            entity_type="dataset", entity_id=name, payload=model))
    return events


def system_status_event(summary: str, severity: str = "info") -> StreamEvent:
    """A ``system_status`` heartbeat/marker event."""
    return StreamEvent(
        event_id=event_id(m.SYSTEM_STATUS, m.SYSTEM, "stream", summary),
        event_type=m.SYSTEM_STATUS, source_engine=m.SYSTEM, severity=severity,
        title="Streaming demo status", summary=summary, entity_type="system",
        entity_id="stream", timestamp=datetime.now(timezone.utc).isoformat())


def safety_status_event() -> StreamEvent:
    """A ``safety_status`` event restating the offline/no-execution posture."""
    summary = ("Offline replay only — no device access, no packet capture, no "
               "command execution. Remediation remains dry-run and human-confirmed.")
    return StreamEvent(
        event_id=event_id(m.SAFETY_STATUS, m.SYSTEM, "safety", summary),
        event_type=m.SAFETY_STATUS, source_engine=m.SYSTEM, severity="info",
        title="Safety status", summary=summary, entity_type="system",
        entity_id="safety", timestamp=datetime.now(timezone.utc).isoformat())


# ------------------------------------------------------------- orchestration


def collect_events(config: dict[str, Any], dirs: dict[str, Any]
                   ) -> list[StreamEvent]:
    """Collect events from every enabled source (per ``streaming.yaml``)."""
    events: list[StreamEvent] = [system_status_event("demo replay started"),
                                 safety_status_event()]

    if _cfg(config, "sources.correlation.enabled", True):
        cid = _cfg(config, "sources.correlation.default_correlation_id",
                   "sample_correlation")
        events += events_from_correlation(dirs["correlation_dir"], cid)
    if _cfg(config, "sources.engine_c.enabled", True):
        sid = _cfg(config, "sources.engine_c.default_snapshot_id",
                   "sample_remediation")
        events += events_from_engine_c(dirs["network_config_dir"], sid)
    if _cfg(config, "sources.engine_b.enabled", True):
        events += events_from_engine_b(dirs["network_health_dir"])
    if _cfg(config, "sources.engine_a.enabled", True):
        events += events_from_engine_a(
            dirs["registry_dir"], dirs["reports_dir"], dirs["error_analysis_dir"],
            dirs["visualizations_dir"], dirs["experiments_dir"])
    if _cfg(config, "sources.syslog.enabled", False):
        run_id = _cfg(config, "sources.syslog.default_run_id", None)
        syslog_dir = dirs.get("syslog_ingestion_dir")
        if run_id and syslog_dir is not None:
            from src.streaming.syslog_source import events_from_syslog
            events += events_from_syslog(syslog_dir, run_id)
    logger.info("Collected %d event(s) from offline artefacts.", len(events))
    return events

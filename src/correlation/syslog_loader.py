"""Read-only loader turning persisted syslog artefacts into correlation signals.

Consumes the Phase-12 ingestion outputs under
``outputs/syslog_ingestion/<run_id>/`` (parser summary + Engine C findings, and
optionally the parsed events for generic fallbacks) and normalises them into
:class:`~src.correlation.models.Signal` objects tagged ``engine="syslog"``.

Strictly read-only: it never re-runs ingestion, never re-parses raw logs, never
opens a socket, never contacts a device, and tolerates every missing optional
artefact (returning ``available=False`` with actionable guidance when a run does
not exist).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.correlation.loader import cfg
from src.correlation.models import SYSLOG, LoadResult, Signal, signal_id
from src.correlation.signal_normalization import (
    SYSLOG_CLOCK_UNRELIABLE,
    SYSLOG_GENERIC_WARNING,
    resolve_confidence,
    source_type_for_finding,
)

logger = logging.getLogger(__name__)

_MAC_RE = re.compile(r"([0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\.[0-9a-fA-F]{4})")
# Source types that summarise many host-level events rather than one entity.
_AGGREGATE_SOURCE_TYPES = {
    "SYSLOG_SNMP_AUTH_ACTIVITY", "SYSLOG_HA_STATE_CHANGE",
    SYSLOG_GENERIC_WARNING, SYSLOG_CLOCK_UNRELIABLE,
}

MISSING_RUN_GUIDANCE = (
    "Syslog ingestion artefacts not found. Run:\n"
    "python -m scripts.ingest_switch_syslog --input-dir <dir> --run-id <id>")


# --------------------------------------------------------------- discovery
def resolve_run_id(syslog_dir: str | Path, run: Optional[str]) -> Optional[str]:
    """Resolve a run id, supporting ``latest`` and ``None`` (newest valid run)."""
    root = Path(syslog_dir)
    if run and run != "latest":
        return run if (root / run / "parser_summary.json").is_file() else None
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir()
            if p.is_dir() and (p / "parser_summary.json").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime).name


def _read_json(path: Path, warnings: list[str], label: str) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.append(f"could not read syslog {label} ({path.name}): {exc}")
        return None


# --------------------------------------------------------------- contract
def load_syslog_artifacts(syslog_dir: str | Path, run: Optional[str]
                          ) -> dict[str, Any]:
    """Load a syslog run's persisted artefacts into the read-only contract dict.

    Returns ``available`` plus run id, source path, counts, findings, events and
    metadata. Missing optional artefacts are tolerated and warned about.
    """
    warnings: list[str] = []
    root = Path(syslog_dir)
    run_id = resolve_run_id(root, run)
    if run_id is None:
        return {"available": False, "run_id": None, "source_path": None,
                "event_count": 0, "parsed_count": 0, "generic_count": 0,
                "findings": [], "events": [], "metadata": {},
                "warnings": [MISSING_RUN_GUIDANCE]}

    run_dir = root / run_id
    summary = _read_json(run_dir / "parser_summary.json", warnings, "summary") or {}
    findings = _read_json(run_dir / "engine_c" / "syslog_findings.json",
                          warnings, "findings") or []
    events = _read_json(run_dir / "parsed_events.json", warnings, "events") or []
    status = summary.get("parse_status", {}) if isinstance(summary, dict) else {}

    return {
        "available": True,
        "run_id": run_id,
        "source_path": str(run_dir),
        "event_count": int(summary.get("parsed_events", len(events)) or 0),
        "parsed_count": int(status.get("parsed", 0) or 0),
        "generic_count": int(status.get("generic", 0) or 0),
        "findings": findings if isinstance(findings, list) else [],
        "events": events if isinstance(events, list) else [],
        "metadata": summary if isinstance(summary, dict) else {},
        "warnings": warnings,
    }


# --------------------------------------------------------------- signals
def load_syslog_signals(
    syslog_dir: str | Path, run: Optional[str], config: dict[str, Any]
) -> tuple[LoadResult, dict[str, Any]]:
    """Load normalised syslog signals plus a syslog metrics roll-up.

    ``config`` is the correlation config; the ``syslog`` block controls generic
    inclusion, caps and clock handling. Returns ``(LoadResult, meta)`` where
    ``meta`` feeds the correlation summary's syslog fields.
    """
    data = load_syslog_artifacts(syslog_dir, run)
    result = LoadResult(engine=SYSLOG, source=data.get("run_id"))
    result.warnings.extend(data.get("warnings", []))

    meta = {
        "syslog_source": data.get("run_id"),
        "syslog_signals_loaded": 0,
        "syslog_events_represented": int(
            (data.get("metadata") or {}).get("weighted_events",
                                             data.get("event_count", 0)) or 0),
        "syslog_findings_loaded": len(data.get("findings", [])),
        "generic_syslog_count": int(data.get("generic_count", 0)),
        "clock_unreliable_count": int(
            (data.get("metadata") or {}).get("clock_unreliable_events", 0) or 0),
    }
    if not data.get("available"):
        return result, meta

    syslog_cfg = cfg(config, "syslog", {}) or {}
    run_id = data["run_id"]

    for finding in data["findings"]:
        result.signals.append(_finding_signal(finding, run_id, syslog_cfg))

    if meta["clock_unreliable_count"] > 0 and bool(
            cfg(config, "rules.CLOCK_INTEGRITY_RISK.enabled", True)):
        result.signals.append(_clock_signal(data, run_id, syslog_cfg))

    if bool(syslog_cfg.get("include_generic", True)):
        result.signals.extend(_generic_signals(data, run_id, syslog_cfg))

    meta["syslog_signals_loaded"] = len(result.signals)
    logger.info("Syslog run '%s': %d signal(s) loaded (offline, read-only).",
                run_id, len(result.signals))
    return result, meta


def _finding_signal(finding: dict[str, Any], run_id: str,
                    syslog_cfg: dict[str, Any]) -> Signal:
    source_type = source_type_for_finding(finding)
    device = _opt(finding.get("device"))
    interface = _opt(finding.get("interface"))
    vlan = _opt(finding.get("vlan"))
    details = finding.get("details") or {}
    title = str(finding.get("title", finding.get("rule_id", "syslog finding")))

    aggregate = source_type in _AGGREGATE_SOURCE_TYPES
    entity_confident = device is not None and (
        aggregate or interface is not None)
    confidence, band, _notes = resolve_confidence(
        source_type, clock_unreliable=False, entity_confident=entity_confident,
        config=syslog_cfg)

    mac = _opt(details.get("mac"))
    if mac is None:
        m = _MAC_RE.search(title)
        mac = m.group(1) if m else None
    src_ip = _opt(details.get("ip_address"))

    src = f"syslog_ingestion/{run_id}/engine_c/syslog_findings.json"
    tags = tuple(str(t) for t in finding.get("tags") or ()) + (source_type, "syslog")
    category = str(finding.get("category", "syslog"))
    return Signal(
        signal_id=signal_id(SYSLOG, src, source_type, device, interface, title),
        engine=SYSLOG, source_artifact=src, category=category,
        source_type=source_type, severity=str(finding.get("severity", "info")),
        confidence=confidence, confidence_label=band, title=title,
        description=str(finding.get("evidence") or finding.get("recommendation") or ""),
        raw_reference=str(finding.get("finding_id", "")),
        timestamp=_opt(finding.get("last_seen")),
        time_start=_opt(finding.get("first_seen")),
        time_end=_opt(finding.get("last_seen")),
        device=device, interface=interface, vlan=vlan, src_ip=src_ip, mac=mac,
        event_count=int(finding.get("event_count", 1) or 1),
        clock_unreliable=False, entity_confident=entity_confident,
        aggregate=aggregate, tags=tags)


def _clock_signal(data: dict[str, Any], run_id: str,
                  syslog_cfg: dict[str, Any]) -> Signal:
    meta = data.get("metadata") or {}
    count = int(meta.get("clock_unreliable_events", 0) or 0)
    hosts = meta.get("hosts") or []
    device = _opt(hosts[0]) if hosts else None
    confidence, band, _notes = resolve_confidence(
        SYSLOG_CLOCK_UNRELIABLE, clock_unreliable=True, config=syslog_cfg)
    src = f"syslog_ingestion/{run_id}/parser_summary.json"
    title = "Unreliable device clock(s) observed in syslog"
    return Signal(
        signal_id=signal_id(SYSLOG, src, SYSLOG_CLOCK_UNRELIABLE, device, None, title),
        engine=SYSLOG, source_artifact=src, category="clock",
        source_type=SYSLOG_CLOCK_UNRELIABLE, severity="low", confidence=confidence,
        confidence_label=band, title=title,
        description=(f"{count} boot-clock/unreliable-timestamp event(s) observed; "
                     "event ordering and cross-source time correlation may be "
                     "approximate for the affected devices."),
        raw_reference=str(count), device=device, event_count=count,
        clock_unreliable=True, entity_confident=device is not None,
        aggregate=True, tags=("clock_unreliable", SYSLOG_CLOCK_UNRELIABLE, "syslog"))


def _generic_signals(data: dict[str, Any], run_id: str,
                     syslog_cfg: dict[str, Any]) -> list[Signal]:
    """Aggregate generic (unmatched-mnemonic) events into capped, low signals."""
    cap = int(syslog_cfg.get("max_generic_signals", 25))
    severity = str(syslog_cfg.get("generic_signal_severity", "info"))
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for event in data.get("events", []):
        if event.get("parse_status") != "generic":
            continue
        host = _opt(event.get("hostname")) or "unknown"
        code = str(event.get("code") or event.get("facility") or "GENERIC")
        key = (host, code)
        grp = groups.setdefault(key, {"count": 0, "first": None, "last": None})
        grp["count"] += max(1, int(event.get("duplicate_count", 1) or 1))
        ts = _opt(event.get("timestamp"))
        if ts:
            grp["first"] = min(grp["first"], ts) if grp["first"] else ts
            grp["last"] = max(grp["last"], ts) if grp["last"] else ts

    ordered = sorted(groups.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    signals: list[Signal] = []
    for (host, code), grp in ordered[:cap]:
        confidence, band, _ = resolve_confidence(
            SYSLOG_GENERIC_WARNING, config=syslog_cfg)
        src = f"syslog_ingestion/{run_id}/parsed_events.json"
        title = f"Unclassified syslog activity: {code} on {host}"
        signals.append(Signal(
            signal_id=signal_id(SYSLOG, src, SYSLOG_GENERIC_WARNING, host, code, title),
            engine=SYSLOG, source_artifact=src, category="generic",
            source_type=SYSLOG_GENERIC_WARNING, severity=severity,
            confidence=confidence, confidence_label=band, title=title,
            description=(f"{grp['count']} generic (unclassified) {code} event(s) "
                         f"on {host}; reported at low confidence."),
            raw_reference=code, device=host, time_start=grp["first"],
            time_end=grp["last"], event_count=grp["count"], entity_confident=False,
            aggregate=True, tags=("generic", SYSLOG_GENERIC_WARNING, "syslog")))
    return signals


def _opt(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

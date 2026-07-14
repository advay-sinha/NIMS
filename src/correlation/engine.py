"""Correlation orchestration and deterministic scoring.

Runs the enabled cross-engine rules over the loaded signals, applies a single-
engine high-risk fallback for anything left uncorrelated, then scores each
grouping into a :class:`~src.correlation.models.CorrelatedIncident` with a
deterministic severity and confidence. No ML, no randomness, no IO.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.correlation.loader import cfg
from src.correlation.models import (
    ENGINE_A,
    ENGINE_B,
    ENGINE_C,
    SYSLOG,
    SEVERITY_LEVELS,
    SEVERITY_ORDER,
    CorrelatedIncident,
    CorrelationSummary,
    IncidentEvidence,
    Signal,
    incident_id,
)
from src.correlation.rules import (
    CROSS_ENGINE_RULES,
    SINGLE_ENGINE_HIGH_RISK,
    SINGLE_SYSLOG_HIGH_RISK,
    IncidentDraft,
    SignalIndex,
    rule_single_engine_high_risk,
    rule_single_syslog_high_risk,
)
from src.correlation.signal_normalization import match_entities

logger = logging.getLogger(__name__)

SAFETY_NOTE = (
    "Correlation is offline and artefact-driven; no packets were captured, no "
    "SNMP was polled, no device was contacted and no command was executed. "
    "Remediation remains human-confirmed and dry-run only."
)

_ENGINE_LABEL = {ENGINE_A: "Engine A (cyber)", ENGINE_B: "Engine B (health)",
                 ENGINE_C: "Engine C (config)", SYSLOG: "Syslog (industrial)"}
_EVIDENCE_ENGINE_ORDER = (ENGINE_A, ENGINE_B, ENGINE_C, SYSLOG)


@dataclass
class CorrelationResult:
    """Everything a correlation run produces."""

    correlation_id: str
    signals: list[Signal]
    incidents: list[CorrelatedIncident]
    summary: CorrelationSummary


def correlate(
    signals: list[Signal],
    config: dict[str, Any],
    correlation_id: str,
    sources: dict[str, Optional[str]],
    syslog_meta: Optional[dict[str, Any]] = None,
) -> CorrelationResult:
    """Correlate ``signals`` into incidents and roll up a summary.

    ``syslog_meta`` (optional) carries the syslog loader's roll-up counts so the
    summary can report syslog coverage; when absent the run behaves exactly as
    before (backward compatible).
    """
    index = SignalIndex.build(signals)
    incidents: dict[str, CorrelatedIncident] = {}
    covered: set[str] = set()

    for rule_id, rule_fn in CROSS_ENGINE_RULES.items():
        if not _rule_enabled(config, rule_id):
            continue
        for draft in rule_fn(index, config):
            incident = _score(draft, config)
            incidents.setdefault(incident.incident_id, incident)
            covered.update(incident.signals)

    if _rule_enabled(config, SINGLE_ENGINE_HIGH_RISK):
        for draft in rule_single_engine_high_risk(index, config, covered):
            incident = _score(draft, config)
            incidents.setdefault(incident.incident_id, incident)
            covered.update(incident.signals)

    if _rule_enabled(config, SINGLE_SYSLOG_HIGH_RISK):
        for draft in rule_single_syslog_high_risk(index, config, covered):
            incident = _score(draft, config)
            incidents.setdefault(incident.incident_id, incident)
            covered.update(incident.signals)

    ordered = sorted(
        incidents.values(),
        key=lambda i: (SEVERITY_ORDER.get(i.severity, 9), -i.confidence,
                       i.incident_id))
    summary = _summarise(correlation_id, signals, ordered, sources, syslog_meta)
    logger.info("Correlation '%s': %d signal(s) -> %d incident(s) "
                "(offline; no commands executed).", correlation_id,
                len(signals), len(ordered))
    return CorrelationResult(correlation_id, signals, ordered, summary)


# --------------------------------------------------------------------- scoring


def _rule_enabled(config: dict[str, Any], rule_id: str) -> bool:
    return bool(cfg(config, f"rules.{rule_id}.enabled", True))


def _score(draft: IncidentDraft, config: dict[str, Any]) -> CorrelatedIncident:
    members = draft.signals
    engines = tuple(sorted({s.engine for s in members}))
    severity, sev_factors = _severity(draft, members, engines, config)
    confidence, conf_factors = _confidence(draft, members, engines, config)

    devices = _distinct(s.device for s in members)
    interfaces = _distinct(s.interface for s in members)
    vlans = _distinct(s.vlan for s in members)
    ips = _distinct([*(s.src_ip for s in members), *(s.dst_ip for s in members)])
    signal_ids = tuple(s.signal_id for s in members)
    syslog_ids = tuple(s.signal_id for s in members if s.engine == SYSLOG)

    quality_notes = _evidence_quality_notes(members, draft)
    return CorrelatedIncident(
        incident_id=incident_id(draft.rule_id, signal_ids),
        rule_id=draft.rule_id, title=draft.title, severity=severity,
        confidence=confidence, engines=engines, signals=signal_ids,
        evidence=_evidence(members),
        recommended_actions=tuple(draft.recommendations),
        root_cause_hypothesis=draft.root_cause_hypothesis,
        affected_devices=devices, affected_interfaces=interfaces,
        related_vlans=vlans, related_ips=ips,
        safety_notes=draft.safety_notes or (SAFETY_NOTE,), tags=draft.tags,
        aggregate_only=all(s.aggregate for s in members),
        scoring_factors=tuple(sev_factors + conf_factors),
        syslog_signal_count=len(syslog_ids), syslog_signal_ids=syslog_ids,
        time_reliability=_time_reliability(members),
        entity_match_confidence=_entity_match_confidence(members, config),
        evidence_quality_notes=quality_notes)


def _severity(draft: IncidentDraft, members: list[Signal],
              engines: tuple[str, ...], config: dict[str, Any]
              ) -> tuple[str, list[str]]:
    """Most-severe member, escalated by rule boost / multi-engine alignment."""
    base_index = min((SEVERITY_ORDER.get(s.severity, 4) for s in members),
                     default=4)
    factors = [f"max_signal_severity={SEVERITY_LEVELS[base_index]}"]
    boost = 0
    if len(engines) >= 2:
        rule_boost = int(cfg(config, f"rules.{draft.rule_id}.severity_boost", 0))
        if rule_boost:
            boost += rule_boost
            factors.append(f"multi_engine_boost=+{rule_boost}")
    if _cross_engine_interface(members):
        boost += 1
        factors.append("interface_alignment=+1")
    final_index = max(0, base_index - boost)
    if draft.severity_cap:
        cap_index = SEVERITY_ORDER.get(draft.severity_cap, 4)
        if final_index < cap_index:       # would be more severe than the cap
            final_index = cap_index
            factors.append(f"severity_cap={draft.severity_cap}")
    return SEVERITY_LEVELS[final_index], factors


def _confidence(draft: IncidentDraft, members: list[Signal],
                engines: tuple[str, ...], config: dict[str, Any]
                ) -> tuple[float, list[str]]:
    score = float(draft.base_confidence)
    factors = [f"base={score:.2f}"]

    multi = float(cfg(config, "scoring.multi_engine_bonus", 0.2))
    if len(engines) > 1 and multi:
        add = multi * (len(engines) - 1)
        score += add
        factors.append(f"multi_engine(+{add:.2f})")

    same_if = float(cfg(config, "scoring.same_interface_bonus", 0.25))
    if same_if and _cross_engine_interface(members):
        score += same_if
        factors.append(f"same_interface(+{same_if:.2f})")

    same_dev = float(cfg(config, "scoring.same_device_bonus", 0.15))
    if same_dev and _cross_engine_device(members):
        score += same_dev
        factors.append(f"same_device(+{same_dev:.2f})")

    penalty = float(cfg(config, "scoring.aggregate_signal_confidence_penalty", 0.2))
    if penalty and any(s.aggregate for s in members):
        score -= penalty
        factors.append(f"aggregate_penalty(-{penalty:.2f})")

    clamped = max(0.0, min(1.0, score))
    return clamped, factors


def _cross_engine_interface(members: list[Signal]) -> bool:
    """True if one non-null interface appears under two different engines."""
    seen: dict[str, set[str]] = {}
    for s in members:
        if s.interface:
            seen.setdefault(s.interface, set()).add(s.engine)
    return any(len(engs) > 1 for engs in seen.values())


def _cross_engine_device(members: list[Signal]) -> bool:
    seen: dict[str, set[str]] = {}
    for s in members:
        if s.device:
            seen.setdefault(s.device, set()).add(s.engine)
    return any(len(engs) > 1 for engs in seen.values())


def _evidence(members: list[Signal]) -> tuple[IncidentEvidence, ...]:
    """One evidence bundle per source engine (Engine A/B/C + Syslog)."""
    by_engine: dict[str, list[Signal]] = {}
    for s in members:
        by_engine.setdefault(s.engine, []).append(s)
    bundles: list[IncidentEvidence] = []
    for engine in _EVIDENCE_ENGINE_ORDER:
        sigs = by_engine.get(engine)
        if not sigs:
            continue
        label = _ENGINE_LABEL.get(engine, engine)
        titles = "; ".join(sorted({s.title for s in sigs}))
        bundles.append(IncidentEvidence(
            engine=engine,
            summary=f"{label}: {len(sigs)} signal(s) — {titles}",
            signal_ids=tuple(s.signal_id for s in sigs),
            source_artifacts=tuple(sorted({s.source_artifact for s in sigs}))))
    return tuple(bundles)


def _time_reliability(members: list[Signal]) -> str:
    """reliable / approximate / unreliable based on member clock reliability."""
    flags = [bool(s.clock_unreliable) for s in members]
    if flags and all(flags):
        return "unreliable"
    if any(flags):
        return "approximate"
    return "reliable"


def _entity_match_confidence(members: list[Signal], config: dict[str, Any]) -> str:
    """How confidently the member signals refer to the same entity."""
    if len(members) < 2:
        return "n/a"
    require_exact = bool(cfg(config,
                             "entity_matching.require_exact_device_match", True))
    normalize = bool(cfg(config, "entity_matching.normalize_interfaces", True))
    ranks = {"exact": 0, "normalized": 1, "uncertain": 2, "none": 3}
    worst = "exact"
    compared = False
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            if a.engine == b.engine:
                continue
            compared = True
            match = match_entities(a.device, a.interface, b.device, b.interface,
                                   require_exact_device=require_exact,
                                   normalize=normalize)
            if ranks[match] > ranks[worst]:
                worst = match
    return worst if compared else "n/a"


def _evidence_quality_notes(members: list[Signal],
                            draft: IncidentDraft) -> tuple[str, ...]:
    """Cautious notes about the evidence backing an incident."""
    notes: list[str] = []
    if any(s.clock_unreliable for s in members):
        notes.append("Some timestamps are unreliable; chronology is approximate.")
    if any(s.aggregate for s in members):
        notes.append("Includes aggregated/summary evidence (down-weighted).")
    if any((s.confidence_label or "") == "low" for s in members):
        notes.append("Some evidence is low-confidence (generic or incomplete).")
    if any(not s.entity_confident for s in members):
        notes.append("Some device/interface identities are incomplete.")
    for alt in getattr(draft, "alternatives", ()) or ():
        notes.append(f"Alternative explanation: {alt}")
    return tuple(dict.fromkeys(notes))


def _distinct(values) -> tuple[str, ...]:
    out: list[str] = []
    for v in values:
        if v and v not in out:
            out.append(str(v))
    return tuple(out)


# --------------------------------------------------------------------- summary


def _summarise(correlation_id: str, signals: list[Signal],
               incidents: list[CorrelatedIncident],
               sources: dict[str, Optional[str]],
               syslog_meta: Optional[dict[str, Any]] = None) -> CorrelationSummary:
    by_engine = Counter(s.engine for s in signals)
    meta = syslog_meta or {}
    # Keep the historical {A,B,C} shape for non-syslog runs; only surface the
    # syslog bucket when syslog is actually in play (backward compatibility).
    engines_reported = [ENGINE_A, ENGINE_B, ENGINE_C]
    if syslog_meta is not None or by_engine.get(SYSLOG, 0):
        engines_reported.append(SYSLOG)
    return CorrelationSummary(
        correlation_id=correlation_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        engine_a_source=sources.get(ENGINE_A),
        engine_b_source=sources.get(ENGINE_B),
        engine_c_source=sources.get(ENGINE_C),
        syslog_source=meta.get("syslog_source") or sources.get(SYSLOG),
        total_signals=len(signals),
        signals_by_engine={e: by_engine.get(e, 0) for e in engines_reported},
        total_incidents=len(incidents),
        incidents_by_severity=dict(Counter(i.severity for i in incidents)),
        incidents_by_rule=dict(Counter(i.rule_id for i in incidents)),
        multi_engine_incident_count=sum(1 for i in incidents if i.multi_engine),
        aggregate_signal_count=sum(1 for s in signals if s.aggregate),
        syslog_signals_loaded=int(meta.get("syslog_signals_loaded",
                                           by_engine.get(SYSLOG, 0))),
        syslog_events_represented=int(meta.get("syslog_events_represented", 0)),
        syslog_findings_loaded=int(meta.get("syslog_findings_loaded", 0)),
        generic_syslog_count=int(meta.get("generic_syslog_count", 0)),
        clock_unreliable_count=int(meta.get("clock_unreliable_count", 0)),
        incidents_with_syslog_evidence=sum(
            1 for i in incidents if i.syslog_signal_count > 0),
        safety_note=SAFETY_NOTE)

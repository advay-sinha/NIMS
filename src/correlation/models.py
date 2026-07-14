"""Correlation-engine data models.

Typed, frozen dataclasses for the correlation pipeline. Signals are the
normalised, engine-agnostic unit ingested from each engine's artefacts;
:class:`CorrelatedIncident` is the unified, operator-facing grouping produced by
the deterministic correlation rules.

Nothing here performs IO or executes anything — these are pure value objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

# Discrete severity ordering shared by signals and incidents (0 = most severe).
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}
SEVERITY_LEVELS: tuple[str, ...] = ("critical", "high", "medium", "low", "info")

# Engine identifiers (stable strings persisted into artefacts).
ENGINE_A = "engine_a"
ENGINE_B = "engine_b"
ENGINE_C = "engine_c"
# Phase 13: persisted industrial-syslog evidence is a first-class source engine.
# It is deliberately NOT mislabelled as Engine B/C — syslog evidence has its own
# certainty characteristics (clock reliability, dedup weight, generic fallbacks).
SYSLOG = "syslog"


def _digest(*parts: Any) -> str:
    """Stable short sha1 digest of the given parts (deterministic ids)."""
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def signal_id(engine: str, source_artifact: str, category: str,
              device: Optional[str], interface: Optional[str],
              title: str) -> str:
    """Deterministic signal id from its natural key."""
    return f"SIG-{_digest(engine, source_artifact, category, device, interface, title)}"


def incident_id(rule_id: str, signal_ids: tuple[str, ...]) -> str:
    """Deterministic incident id from the rule and its member signals.

    The signal ids are sorted so the id is independent of match ordering.
    """
    return f"INC-{_digest(rule_id, *sorted(signal_ids))}"


@dataclass(frozen=True)
class Signal:
    """A single normalised observation from one engine.

    ``confidence`` is a 0-1 float. ``aggregate`` marks signals that summarise a
    dataset/model rather than a specific observed event (Engine A/B currently
    expose only aggregate artefacts); aggregate signals are penalised during
    incident confidence scoring so they never masquerade as precise alerts.
    """

    signal_id: str
    engine: str
    source_artifact: str
    category: str
    severity: str
    confidence: float
    title: str
    description: str
    raw_reference: str
    timestamp: Optional[str] = None
    device: Optional[str] = None
    interface: Optional[str] = None
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    vlan: Optional[str] = None
    aggregate: bool = False
    tags: tuple[str, ...] = ()
    # --- Phase 13 optional fields (backward-compatible; default = previous
    # behaviour). Populated for syslog-sourced signals; harmless for A/B/C.
    source_type: Optional[str] = None       # e.g. SYSLOG_PORT_FLAP
    mac: Optional[str] = None
    time_start: Optional[str] = None        # start of the aggregated time range
    time_end: Optional[str] = None          # end of the aggregated time range
    event_count: int = 1                    # evidence weight (dedup/aggregation)
    confidence_label: Optional[str] = None  # high/medium/low quality band
    clock_unreliable: bool = False          # timestamps not precisely ordered
    entity_confident: bool = True           # device/interface identity is certain

    def to_row(self) -> dict[str, Any]:
        """Flat serialisable mapping (used for JSON and CSV export)."""
        return {
            "signal_id": self.signal_id,
            "engine": self.engine,
            "source_artifact": self.source_artifact,
            "category": self.category,
            "source_type": self.source_type,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "confidence_label": self.confidence_label,
            "title": self.title,
            "description": self.description,
            "raw_reference": self.raw_reference,
            "timestamp": self.timestamp,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "device": self.device,
            "interface": self.interface,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "vlan": self.vlan,
            "mac": self.mac,
            "event_count": self.event_count,
            "clock_unreliable": self.clock_unreliable,
            "entity_confident": self.entity_confident,
            "aggregate": self.aggregate,
            "tags": ";".join(self.tags),
        }


@dataclass(frozen=True)
class IncidentEvidence:
    """A corroborating evidence bundle behind one incident."""

    engine: str
    summary: str
    signal_ids: tuple[str, ...] = ()
    source_artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "summary": self.summary,
            "signal_ids": list(self.signal_ids),
            "source_artifacts": list(self.source_artifacts),
        }


@dataclass(frozen=True)
class IncidentRecommendation:
    """A human-owned, non-executed recommended action."""

    title: str
    detail: str
    owner: str                       # network / security / ops
    requires_confirmation: bool = True
    safety_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "detail": self.detail,
            "owner": self.owner,
            "requires_confirmation": self.requires_confirmation,
            "safety_note": self.safety_note,
        }


@dataclass(frozen=True)
class CorrelatedIncident:
    """A unified, operator-facing incident grouping one or more signals."""

    incident_id: str
    rule_id: str
    title: str
    severity: str                    # critical/high/medium/low/info
    confidence: float                # 0-1
    engines: tuple[str, ...]
    signals: tuple[str, ...]         # member signal ids
    evidence: tuple[IncidentEvidence, ...]
    recommended_actions: tuple[IncidentRecommendation, ...]
    root_cause_hypothesis: str
    affected_devices: tuple[str, ...] = ()
    affected_interfaces: tuple[str, ...] = ()
    related_vlans: tuple[str, ...] = ()
    related_ips: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    status: str = "open"
    aggregate_only: bool = False
    scoring_factors: tuple[str, ...] = ()
    # --- Phase 13 optional fields (backward-compatible).
    syslog_signal_count: int = 0
    syslog_signal_ids: tuple[str, ...] = ()
    time_reliability: str = "reliable"       # reliable/approximate/unreliable
    entity_match_confidence: str = "exact"   # exact/normalized/uncertain/n/a
    evidence_quality_notes: tuple[str, ...] = ()

    @property
    def multi_engine(self) -> bool:
        """True when signals from more than one engine corroborate the incident."""
        return len(self.engines) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "engines": list(self.engines),
            "multi_engine": self.multi_engine,
            "aggregate_only": self.aggregate_only,
            "affected_devices": list(self.affected_devices),
            "affected_interfaces": list(self.affected_interfaces),
            "related_vlans": list(self.related_vlans),
            "related_ips": list(self.related_ips),
            "signals": list(self.signals),
            "evidence": [e.to_dict() for e in self.evidence],
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "recommended_actions": [a.to_dict() for a in self.recommended_actions],
            "safety_notes": list(self.safety_notes),
            "tags": list(self.tags),
            "scoring_factors": list(self.scoring_factors),
            "syslog_signal_count": self.syslog_signal_count,
            "syslog_signal_ids": list(self.syslog_signal_ids),
            "time_reliability": self.time_reliability,
            "entity_match_confidence": self.entity_match_confidence,
            "evidence_quality_notes": list(self.evidence_quality_notes),
        }

    def to_row(self) -> dict[str, Any]:
        """Flat mapping for CSV export (nested fields collapsed to strings)."""
        return {
            "incident_id": self.incident_id,
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "engines": ";".join(self.engines),
            "multi_engine": self.multi_engine,
            "aggregate_only": self.aggregate_only,
            "affected_devices": ";".join(self.affected_devices),
            "affected_interfaces": ";".join(self.affected_interfaces),
            "related_vlans": ";".join(self.related_vlans),
            "related_ips": ";".join(self.related_ips),
            "signal_count": len(self.signals),
            "syslog_signal_count": self.syslog_signal_count,
            "time_reliability": self.time_reliability,
            "entity_match_confidence": self.entity_match_confidence,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "tags": ";".join(self.tags),
        }


@dataclass(frozen=True)
class CorrelationSummary:
    """The serialisable roll-up written to ``correlation_summary.json``."""

    correlation_id: str
    timestamp: str
    engine_a_source: Optional[str]
    engine_b_source: Optional[str]
    engine_c_source: Optional[str]
    total_signals: int
    signals_by_engine: dict[str, int]
    total_incidents: int
    incidents_by_severity: dict[str, int]
    incidents_by_rule: dict[str, int]
    multi_engine_incident_count: int
    aggregate_signal_count: int
    safety_note: str
    # --- Phase 13 optional syslog roll-up (backward-compatible defaults).
    syslog_source: Optional[str] = None
    syslog_signals_loaded: int = 0
    syslog_events_represented: int = 0
    syslog_findings_loaded: int = 0
    generic_syslog_count: int = 0
    clock_unreliable_count: int = 0
    incidents_with_syslog_evidence: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "engine_a_source": self.engine_a_source,
            "engine_b_source": self.engine_b_source,
            "engine_c_source": self.engine_c_source,
            "syslog_source": self.syslog_source,
            "total_signals": self.total_signals,
            "signals_by_engine": self.signals_by_engine,
            "total_incidents": self.total_incidents,
            "incidents_by_severity": self.incidents_by_severity,
            "incidents_by_rule": self.incidents_by_rule,
            "multi_engine_incident_count": self.multi_engine_incident_count,
            "aggregate_signal_count": self.aggregate_signal_count,
            "syslog_signals_loaded": self.syslog_signals_loaded,
            "syslog_events_represented": self.syslog_events_represented,
            "syslog_findings_loaded": self.syslog_findings_loaded,
            "generic_syslog_count": self.generic_syslog_count,
            "clock_unreliable_count": self.clock_unreliable_count,
            "incidents_with_syslog_evidence": self.incidents_with_syslog_evidence,
            "safety_note": self.safety_note,
        }


@dataclass
class LoadResult:
    """Signals loaded from one engine plus any non-fatal warnings."""

    engine: str
    source: Optional[str]
    signals: list[Signal] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

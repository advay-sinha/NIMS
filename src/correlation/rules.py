"""Deterministic, configurable correlation rules.

Each rule inspects an indexed set of :class:`~src.correlation.models.Signal`
objects and emits zero or more :class:`IncidentDraft` groupings. Rules are pure
and deterministic — no IO, no randomness, no ML — so the same signals always
produce the same incidents. Severity/confidence scoring is applied later by the
engine; a rule only decides *what groups together* and supplies the explanation
and human-owned recommendations.

Rules are enabled/disabled and tuned through ``configs/correlation.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.correlation.loader import cfg
from src.correlation.models import (
    ENGINE_A,
    ENGINE_B,
    ENGINE_C,
    IncidentRecommendation,
    Signal,
)

# Rule identifiers (must match keys under ``rules:`` in the config).
DOS_SATURATION = "DOS_SATURATION"
LINK_DEGRADATION = "LINK_DEGRADATION"
CONFIG_EXPOSURE = "CONFIG_EXPOSURE"
VLAN_POLICY_RISK = "VLAN_POLICY_RISK"
SINGLE_ENGINE_HIGH_RISK = "SINGLE_ENGINE_HIGH_RISK"

_CONFIRM_NOTE = ("Human-confirmed and dry-run only; Engine C never executes a "
                 "command against a device.")


@dataclass
class IncidentDraft:
    """A pre-scoring incident grouping produced by a rule."""

    rule_id: str
    title: str
    signals: list[Signal]
    root_cause_hypothesis: str
    recommendations: list[IncidentRecommendation] = field(default_factory=list)
    tags: tuple[str, ...] = ()
    safety_notes: tuple[str, ...] = ()
    base_confidence: float = 0.4


@dataclass
class SignalIndex:
    """Signals bucketed by engine for cheap rule access."""

    engine_a: list[Signal]
    engine_b: list[Signal]
    engine_c: list[Signal]

    @classmethod
    def build(cls, signals: list[Signal]) -> "SignalIndex":
        buckets: dict[str, list[Signal]] = {ENGINE_A: [], ENGINE_B: [], ENGINE_C: []}
        for sig in signals:
            buckets.setdefault(sig.engine, []).append(sig)
        return cls(buckets[ENGINE_A], buckets[ENGINE_B], buckets[ENGINE_C])


# --------------------------------------------------------------- classification


def _has_tag(signal: Signal, wanted: set[str]) -> bool:
    return bool(wanted.intersection(t.lower() for t in signal.tags))


def _dos_signals(index: SignalIndex, config: dict[str, Any]) -> list[Signal]:
    dos_families = {str(x).lower() for x in
                    cfg(config, "engine_a.dos_families", ["dos", "ddos", "flood"])}
    return [s for s in index.engine_a if _has_tag(s, dos_families)]


def _attack_signals(index: SignalIndex) -> list[Signal]:
    return [s for s in index.engine_a if _has_tag(s, {"attack"})]


def _health_signals(index: SignalIndex) -> list[Signal]:
    return [s for s in index.engine_b if _has_tag(s, {"anomaly", "degradation"})]


def _engine_c_by(index: SignalIndex, config: dict[str, Any], key: str,
                 default: list[str]) -> list[Signal]:
    """Engine C signals whose rule_id/category matches a configured set."""
    wanted = {str(x).lower() for x in cfg(config, f"engine_c.{key}", default)}
    out: list[Signal] = []
    for sig in index.engine_c:
        labels = {t.lower() for t in sig.tags} | {sig.category.lower()}
        if wanted.intersection(labels):
            out.append(sig)
    return out


# --------------------------------------------------------------------- rules


def rule_dos_saturation(index: SignalIndex, config: dict[str, Any]
                        ) -> list[IncidentDraft]:
    """A. Attack activity + network saturation (+ optional interface warning)."""
    attack = _dos_signals(index, config) or _attack_signals(index)
    health = _health_signals(index)
    if not (attack and health):
        return []
    saturation = _engine_c_by(index, config, "saturation_categories",
                              ["performance", "topology", "availability",
                               "interface"])
    members = list(attack) + list(health) + list(saturation)
    engines = {s.engine for s in members}
    min_engines = int(cfg(config, "rules.DOS_SATURATION.min_engines", 2))
    if len(engines) < min_engines:
        return []
    rec = IncidentRecommendation(
        title="Investigate possible attack-induced saturation",
        detail="Correlate the intrusion indicator with the network-health "
               "anomaly; inspect utilisation/discards on the implicated "
               "interfaces before any change.",
        owner="security", safety_note=_CONFIRM_NOTE)
    return [IncidentDraft(
        rule_id=DOS_SATURATION,
        title="Possible attack-induced network saturation",
        signals=members,
        root_cause_hypothesis=(
            "Intrusion-detection coverage for attack traffic coincides with a "
            "network-health anomaly, which is consistent with (but does not "
            "prove) attack-induced saturation such as a flood or DDoS."),
        recommendations=[rec], tags=("attack", "saturation", "multi-engine"),
        safety_notes=(_CONFIRM_NOTE,), base_confidence=0.45)]


def rule_link_degradation(index: SignalIndex, config: dict[str, Any]
                          ) -> list[IncidentDraft]:
    """B. Health degradation + physical/config finding, no strong cyber signal."""
    if _attack_signals(index):
        return []                        # DoS rule owns the attack-present case
    health = _health_signals(index)
    degraded = _engine_c_by(index, config, "degradation_categories",
                            ["port", "stp", "performance", "availability",
                             "loop risk", "loop_risk", "topology"])
    if not (health and degraded):
        return []
    drafts: list[IncidentDraft] = []
    for device, sigs in _group_by_device(degraded).items():
        members = list(health) + sigs
        rec = IncidentRecommendation(
            title="Inspect physical/config link degradation",
            detail="Check cabling, error/discard counters and STP/port state on "
                   "the affected interfaces; plan any change as an Engine C "
                   "dry-run.",
            owner="network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=LINK_DEGRADATION,
            title=f"Possible link degradation on {device}",
            signals=members,
            root_cause_hypothesis=(
                "A network-health anomaly aligns with configuration/port "
                "findings and no intrusion signal, which points to a physical "
                "or configuration-driven link problem rather than an attack."),
            recommendations=[rec], tags=("degradation", "physical", "config"),
            safety_notes=(_CONFIRM_NOTE,), base_confidence=0.4))
    return drafts


def rule_config_exposure(index: SignalIndex, config: dict[str, Any]
                         ) -> list[IncidentDraft]:
    """C. Exposed/misconfigured segment (+ optional suspicious cyber signal)."""
    exposure = _engine_c_by(index, config, "exposure_rules",
                            ["UNUSED_PORT_ADMIN_UP", "TRUNK_WITHOUT_NEIGHBOR",
                             "TRUNK_UNAUTHORIZED_VLAN", "STP_DISABLED",
                             "STP_MISSING"])
    if not exposure:
        return []
    attack = _attack_signals(index)
    drafts: list[IncidentDraft] = []
    for device, sigs in _group_by_device(exposure).items():
        members = sigs + list(attack)
        cyber = " with concurrent intrusion-detection exposure" if attack else ""
        rec = IncidentRecommendation(
            title="Review exposed/misconfigured segment",
            detail="Confirm whether the port/VLAN/STP state is intended; plan a "
                   "dry-run shutdown or VLAN correction if not.",
            owner="security" if attack else "network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=CONFIG_EXPOSURE,
            title=f"Possible exposed or misconfigured segment on {device}",
            signals=members,
            root_cause_hypothesis=(
                "Configuration findings indicate an exposed or misconfigured "
                f"segment{cyber}; this is a candidate attack surface that "
                "should be reviewed."),
            recommendations=[rec], tags=("exposure", "config", "security"),
            safety_notes=(_CONFIRM_NOTE,), base_confidence=0.4))
    return drafts


def rule_vlan_policy_risk(index: SignalIndex, config: dict[str, Any]
                          ) -> list[IncidentDraft]:
    """D. VLAN policy issue combined with an attack/suspicious aggregate."""
    attack = _attack_signals(index)
    if not attack:
        return []
    vlan_issues = _engine_c_by(index, config, "vlan_rules",
                               ["TRUNK_UNAUTHORIZED_VLAN",
                                "TRUNK_MISSING_REQUIRED_VLAN",
                                "WRONG_ACCESS_VLAN", "WRONG_NATIVE_VLAN"])
    if not vlan_issues:
        return []
    drafts: list[IncidentDraft] = []
    for device, sigs in _group_by_device(vlan_issues).items():
        members = sigs + list(attack)
        rec = IncidentRecommendation(
            title="Review VLAN policy against cyber exposure",
            detail="Validate the trunk/access VLAN configuration against policy; "
                   "plan a dry-run VLAN correction if the segment is exposed.",
            owner="security", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=VLAN_POLICY_RISK,
            title=f"VLAN policy risk with cyber exposure on {device}",
            signals=members,
            root_cause_hypothesis=(
                "A VLAN configuration issue coincides with intrusion-detection "
                "coverage of attack traffic, so the misconfigured VLAN may "
                "widen the exposed segment."),
            recommendations=[rec], tags=("vlan", "policy", "security"),
            safety_notes=(_CONFIRM_NOTE,), base_confidence=0.42))
    return drafts


def rule_single_engine_high_risk(
    index: SignalIndex, config: dict[str, Any], covered: set[str]
) -> list[IncidentDraft]:
    """E. Fallback: important single-engine signals not already correlated."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    min_sev = str(cfg(config, "rules.SINGLE_ENGINE_HIGH_RISK.min_severity", "high"))
    ceiling = order.get(min_sev, 1)
    drafts: list[IncidentDraft] = []
    for sig in index.engine_a + index.engine_b + index.engine_c:
        if sig.signal_id in covered:
            continue
        if order.get(sig.severity, 4) > ceiling:
            continue
        rec = IncidentRecommendation(
            title="Review single-engine high-risk item",
            detail="No cross-engine correlation was found; review this item on "
                   "its own and confirm before any change.",
            owner="network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=SINGLE_ENGINE_HIGH_RISK,
            title=f"High-risk {sig.engine} item: {sig.title}",
            signals=[sig],
            root_cause_hypothesis=(
                "Single-engine finding with no corroborating signal from the "
                "other engines; reported at reduced confidence."),
            recommendations=[rec], tags=("single-engine",) + sig.tags[:2],
            safety_notes=(_CONFIRM_NOTE,), base_confidence=0.3))
    return drafts


# --------------------------------------------------------------------- helpers


def _group_by_device(signals: list[Signal]) -> dict[str, list[Signal]]:
    grouped: dict[str, list[Signal]] = {}
    for sig in signals:
        grouped.setdefault(sig.device or "unknown", []).append(sig)
    return grouped


# Correlation rules that combine engines (run before the single-engine fallback).
CROSS_ENGINE_RULES: dict[str, Callable[[SignalIndex, dict[str, Any]],
                                       list[IncidentDraft]]] = {
    DOS_SATURATION: rule_dos_saturation,
    LINK_DEGRADATION: rule_link_degradation,
    CONFIG_EXPOSURE: rule_config_exposure,
    VLAN_POLICY_RISK: rule_vlan_policy_risk,
}

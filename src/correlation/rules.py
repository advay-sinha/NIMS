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
from typing import Any, Callable, Optional

from src.correlation.loader import cfg
from src.correlation.models import (
    ENGINE_A,
    ENGINE_B,
    ENGINE_C,
    SYSLOG,
    IncidentRecommendation,
    Signal,
)
from src.correlation.signal_normalization import (
    SYSLOG_DUPLICATE_IP,
    SYSLOG_ERPS_CHURN,
    SYSLOG_MAC_FLAP,
    SYSLOG_MANAGEMENT_ACCESS,
    SYSLOG_POE_FAULT,
    SYSLOG_PORT_FLAP,
    SYSLOG_SNMP_AUTH_ACTIVITY,
)

# Rule identifiers (must match keys under ``rules:`` in the config).
DOS_SATURATION = "DOS_SATURATION"
LINK_DEGRADATION = "LINK_DEGRADATION"
CONFIG_EXPOSURE = "CONFIG_EXPOSURE"
VLAN_POLICY_RISK = "VLAN_POLICY_RISK"
SINGLE_ENGINE_HIGH_RISK = "SINGLE_ENGINE_HIGH_RISK"
# Phase 13 syslog-aware rules.
SYSLOG_SNMP_AUTH_CAMPAIGN = "SYSLOG_SNMP_AUTH_CAMPAIGN"
PORT_INSTABILITY = "PORT_INSTABILITY"
LOOP_OR_REDUNDANCY_INSTABILITY = "LOOP_OR_REDUNDANCY_INSTABILITY"
DUPLICATE_IP_CONFLICT = "DUPLICATE_IP_CONFLICT"
POE_ENDPOINT_FAILURE = "POE_ENDPOINT_FAILURE"
MANAGEMENT_ACCESS_EXPOSURE = "MANAGEMENT_ACCESS_EXPOSURE"
CLOCK_INTEGRITY_RISK = "CLOCK_INTEGRITY_RISK"
SINGLE_SYSLOG_HIGH_RISK = "SINGLE_SYSLOG_HIGH_RISK"

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
    # Optional alternative explanations shown to operators (Phase 13).
    alternatives: tuple[str, ...] = ()
    # Optional least-severe ceiling (e.g. cap at "high" unless multi-source).
    severity_cap: Optional[str] = None


@dataclass
class SignalIndex:
    """Signals bucketed by engine/source for cheap rule access."""

    engine_a: list[Signal]
    engine_b: list[Signal]
    engine_c: list[Signal]
    syslog: list[Signal] = field(default_factory=list)

    @classmethod
    def build(cls, signals: list[Signal]) -> "SignalIndex":
        buckets: dict[str, list[Signal]] = {
            ENGINE_A: [], ENGINE_B: [], ENGINE_C: [], SYSLOG: []}
        for sig in signals:
            buckets.setdefault(sig.engine, []).append(sig)
        return cls(buckets[ENGINE_A], buckets[ENGINE_B], buckets[ENGINE_C],
                   buckets[SYSLOG])

    def all_signals(self) -> list[Signal]:
        return self.engine_a + self.engine_b + self.engine_c + self.syslog


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


# ----------------------------------------------------------- syslog rules


_OPERATOR_ONLY = ("Investigation-oriented only — no device command is generated "
                  "and no remediation is applied.")


def _by_source_type(signals: list[Signal], *types: str) -> list[Signal]:
    wanted = set(types)
    return [s for s in signals if s.source_type in wanted]


def _same_device(signals: list[Signal], device: str) -> list[Signal]:
    key = (device or "").strip().lower()
    return [s for s in signals if (s.device or "").strip().lower() == key]


def _weight(signals: list[Signal]) -> int:
    return sum(max(1, int(s.event_count)) for s in signals)


def _engine_c_same_interface(index: SignalIndex, device: str, interface: str,
                             config: dict[str, Any]) -> list[Signal]:
    """Engine C signals on the same (normalised) device/interface."""
    from src.correlation.signal_normalization import match_entities
    require_exact = bool(cfg(config,
                             "entity_matching.require_exact_device_match", True))
    normalize = bool(cfg(config, "entity_matching.normalize_interfaces", True))
    out: list[Signal] = []
    for sig in index.engine_c:
        if not sig.interface:
            continue
        match = match_entities(device, interface, sig.device, sig.interface,
                               require_exact_device=require_exact, normalize=normalize)
        if match in ("exact", "normalized"):
            out.append(sig)
    return out


def rule_syslog_snmp_auth_campaign(index: SignalIndex, config: dict[str, Any]
                                   ) -> list[IncidentDraft]:
    """A. Repeated SNMP auth/community failures — suspected mgmt-plane campaign."""
    snmp = _by_source_type(index.syslog, SYSLOG_SNMP_AUTH_ACTIVITY)
    if not snmp:
        return []
    minimum = int(cfg(config, "rules.SYSLOG_SNMP_AUTH_CAMPAIGN.minimum_attempts", 10))
    attack = _attack_signals(index)
    drafts: list[IncidentDraft] = []
    for device, sigs in _group_by_device(snmp).items():
        if _weight(sigs) < minimum:
            continue
        members = list(sigs) + list(attack)
        rec = IncidentRecommendation(
            title="Investigate SNMP authentication activity",
            detail="Identify the source host and confirm whether it belongs to an "
                   "authorized monitoring system; review community/ACL "
                   "configuration and related firewall/network logs.",
            owner="security", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=SYSLOG_SNMP_AUTH_CAMPAIGN,
            title=f"Suspected SNMP authentication campaign on {device}",
            signals=members,
            root_cause_hypothesis=(
                "Repeated SNMP authorization/community failures are consistent "
                "with management-plane reconnaissance or brute-force, but do not "
                "confirm compromise; verification against the source host is "
                "required."),
            recommendations=[rec],
            tags=("syslog", "snmp", "security", "management-plane"),
            safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.5,
            alternatives=("An authorized monitoring system misconfigured with the "
                          "wrong community string could produce the same pattern.",)))
    return drafts


def rule_port_instability(index: SignalIndex, config: dict[str, Any]
                          ) -> list[IncidentDraft]:
    """B. Repeated port up/down transitions, optionally + health/config."""
    ports = _by_source_type(index.syslog, SYSLOG_PORT_FLAP)
    if not ports:
        return []
    minimum = int(cfg(config, "rules.PORT_INSTABILITY.minimum_transitions", 4))
    health = _health_signals(index)
    drafts: list[IncidentDraft] = []
    for sig in ports:
        if _weight([sig]) < minimum:
            continue
        device, iface = sig.device or "unknown", sig.interface or ""
        corr_health = _same_device(health, device)
        corr_cfg = _engine_c_same_interface(index, device, iface, config) if iface \
            else []
        members = [sig] + corr_health + corr_cfg
        rec = IncidentRecommendation(
            title="Inspect link stability",
            detail="Inspect the cable/SFP/connector, check endpoint power, review "
                   "interface counters and confirm whether planned maintenance "
                   "explains the transitions.",
            owner="network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=PORT_INSTABILITY,
            title=f"Port instability on {iface or device}",
            signals=members,
            root_cause_hypothesis=(
                "Repeated interface up/down transitions are consistent with a "
                "physical/link problem or an intermittent endpoint/power issue "
                "rather than an attack."),
            recommendations=[rec], tags=("syslog", "port", "instability"),
            safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.5,
            alternatives=("Planned maintenance or a rebooting endpoint could "
                          "explain intermittent transitions.",)))
    return drafts


def rule_loop_or_redundancy(index: SignalIndex, config: dict[str, Any]
                            ) -> list[IncidentDraft]:
    """C. MAC flaps + ERPS churn + topology warnings — possible loop/redundancy."""
    mac = _by_source_type(index.syslog, SYSLOG_MAC_FLAP)
    erps = _by_source_type(index.syslog, SYSLOG_ERPS_CHURN)
    if not (mac or erps):
        return []
    minimum = int(cfg(
        config, "rules.LOOP_OR_REDUNDANCY_INSTABILITY.minimum_mac_flaps", 3))
    topo = [s for s in index.engine_c if s.category == "topology"
            or "topology" in {t.lower() for t in s.tags}]
    drafts: list[IncidentDraft] = []
    devices = {s.device for s in mac + erps if s.device}
    for device in sorted(devices):
        dev_mac = _same_device(mac, device)
        dev_erps = _same_device(erps, device)
        dev_topo = _same_device(topo, device)
        if not dev_mac and not dev_erps:
            continue
        if dev_mac and _weight(dev_mac) < minimum and not dev_erps:
            continue
        members = dev_mac + dev_erps + dev_topo
        sources = sum(1 for grp in (dev_mac, dev_erps, dev_topo) if grp)
        require_multi = bool(cfg(
            config,
            "rules.LOOP_OR_REDUNDANCY_INSTABILITY.require_multi_source_for_critical",
            True))
        cap = "high" if (require_multi and sources < 2) else None
        rec = IncidentRecommendation(
            title="Inspect ring/STP and MAC learning",
            detail="Inspect ring/STP state and recent topology changes, identify "
                   "interfaces learning the same MAC, and verify whether "
                   "redundancy transitions are expected.",
            owner="network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=LOOP_OR_REDUNDANCY_INSTABILITY,
            title=f"Possible switching loop or redundancy instability on {device}",
            signals=members,
            root_cause_hypothesis=(
                "MAC movement together with ring/topology activity is a candidate "
                "switching loop or redundancy instability inferred from combined "
                "evidence; it is not a confirmed loop unless an explicit loop "
                "message is present."),
            recommendations=[rec], tags=("syslog", "loop", "redundancy", "topology")
            + (("multi-source",) if sources > 1 else ()),
            safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.5,
            severity_cap=cap,
            alternatives=("Expected ERPS/STP protection switching during "
                          "maintenance could resemble instability.",)))
    return drafts


def rule_duplicate_ip(index: SignalIndex, config: dict[str, Any]
                      ) -> list[IncidentDraft]:
    """D. Duplicate-IP / ARP instability — probable IP conflict."""
    dup = _by_source_type(index.syslog, SYSLOG_DUPLICATE_IP)
    if not dup:
        return []
    mac = _by_source_type(index.syslog, SYSLOG_MAC_FLAP)
    drafts: list[IncidentDraft] = []
    for device, sigs in _group_by_device(dup).items():
        members = list(sigs) + _same_device(mac, device)
        rec = IncidentRecommendation(
            title="Resolve possible IP conflict",
            detail="Identify hosts advertising the same address, inspect ARP/MAC "
                   "tables and review DHCP/static assignments.",
            owner="network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=DUPLICATE_IP_CONFLICT,
            title=f"Probable IP conflict observed on {device}",
            signals=members,
            root_cause_hypothesis=(
                "A duplicate-IP / rapid ARP-change message is consistent with an "
                "IP address conflict requiring host/network investigation."),
            recommendations=[rec], tags=("syslog", "duplicate_ip", "arp"),
            safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.55,
            severity_cap="high",   # spec: duplicate-IP is medium/high, not critical
            alternatives=("A planned host migration or HA failover can briefly "
                          "move an address between MACs.",)))
    return drafts


def rule_poe_endpoint_failure(index: SignalIndex, config: dict[str, Any]
                              ) -> list[IncidentDraft]:
    """E. PoE fault (+ interface transition / PoE config finding)."""
    poe = _by_source_type(index.syslog, SYSLOG_POE_FAULT)
    if not poe:
        return []
    ports = _by_source_type(index.syslog, SYSLOG_PORT_FLAP)
    drafts: list[IncidentDraft] = []
    for sig in poe:
        device, iface = sig.device or "unknown", sig.interface or ""
        same_iface = [p for p in ports if (p.interface or "") == iface] if iface else []
        corr_cfg = _engine_c_same_interface(index, device, iface, config) if iface \
            else []
        members = [sig] + same_iface + corr_cfg
        rec = IncidentRecommendation(
            title="Inspect PoE delivery and endpoint",
            detail="Inspect the power budget and per-port state, verify endpoint "
                   "power requirements, and inspect cabling and the powered device.",
            owner="network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=POE_ENDPOINT_FAILURE,
            title=f"Likely PoE/endpoint issue on {iface or device}",
            signals=members,
            root_cause_hypothesis=(
                "A PoE fault, optionally alongside interface transitions, is "
                "consistent with a powered-endpoint, cabling or PoE-delivery "
                "issue."),
            recommendations=[rec], tags=("syslog", "poe", "endpoint"),
            safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.5))
    return drafts


def rule_management_access_exposure(index: SignalIndex, config: dict[str, Any]
                                    ) -> list[IncidentDraft]:
    """F. Telnet/WEB management access (+ SNMP / exposure / cyber)."""
    mgmt = _by_source_type(index.syslog, SYSLOG_MANAGEMENT_ACCESS)
    if not mgmt:
        return []
    snmp = _by_source_type(index.syslog, SYSLOG_SNMP_AUTH_ACTIVITY)
    exposure = _engine_c_by(index, config, "exposure_rules",
                            ["UNUSED_PORT_ADMIN_UP", "STP_DISABLED"])
    attack = _attack_signals(index)
    drafts: list[IncidentDraft] = []
    for device, sigs in _group_by_device(mgmt).items():
        members = list(sigs) + _same_device(snmp, device) \
            + _same_device(exposure, device) + list(attack)
        rec = IncidentRecommendation(
            title="Review management-plane exposure",
            detail="Confirm whether the management access is expected and from an "
                   "authorized source; prefer SSH over telnet and restrict "
                   "management by ACL.",
            owner="security", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=MANAGEMENT_ACCESS_EXPOSURE,
            title=f"Management-plane access activity on {device}",
            signals=members,
            root_cause_hypothesis=(
                "Management access events (telnet/web), optionally with SNMP "
                "activity or configuration exposure, indicate management-plane "
                "exposure; unauthorized access is not assumed without explicit "
                "evidence."),
            recommendations=[rec], tags=("syslog", "management", "security"),
            safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.45))
    return drafts


def rule_clock_integrity(index: SignalIndex, config: dict[str, Any]
                         ) -> list[IncidentDraft]:
    """G. Unreliable-clock evidence degrades time correlation."""
    clock = [s for s in index.syslog if "clock_unreliable" in s.tags]
    if not clock:
        return []
    rec = IncidentRecommendation(
        title="Restore reliable device time",
        detail="Verify NTP configuration and reachability; until clocks are "
               "reliable, treat event ordering and cross-source time correlation "
               "as approximate.",
        owner="ops", safety_note=_CONFIRM_NOTE)
    return [IncidentDraft(
        rule_id=CLOCK_INTEGRITY_RISK,
        title="Monitoring clock integrity risk",
        signals=list(clock),
        root_cause_hypothesis=(
            "Boot-clock / unreliable-timestamp events mean event ordering and "
            "time-based cross-source correlation may be degraded for the affected "
            "devices."),
        recommendations=[rec], tags=("syslog", "clock", "integrity"),
        safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.4)]


def rule_single_syslog_high_risk(
    index: SignalIndex, config: dict[str, Any], covered: set[str]
) -> list[IncidentDraft]:
    """H. Fallback: isolated high/critical syslog signals not otherwise used."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    ceiling = order.get(
        str(cfg(config, "rules.SINGLE_SYSLOG_HIGH_RISK.min_severity", "high")), 1)
    drafts: list[IncidentDraft] = []
    for sig in index.syslog:
        if sig.signal_id in covered or order.get(sig.severity, 4) > ceiling:
            continue
        rec = IncidentRecommendation(
            title="Review isolated high-risk syslog item",
            detail="No cross-source correlation was found; review this syslog "
                   "item on its own and confirm before any change.",
            owner="network", safety_note=_CONFIRM_NOTE)
        drafts.append(IncidentDraft(
            rule_id=SINGLE_SYSLOG_HIGH_RISK,
            title=f"High-risk syslog item: {sig.title}",
            signals=[sig],
            root_cause_hypothesis=(
                "Single high-risk syslog signal with no corroborating evidence "
                "from other sources; reported at reduced confidence."),
            recommendations=[rec], tags=("syslog", "single-source") + sig.tags[:2],
            safety_notes=(_CONFIRM_NOTE, _OPERATOR_ONLY), base_confidence=0.35))
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
    # Phase 13 syslog-aware rules (share the same pipeline; YAML-gated).
    SYSLOG_SNMP_AUTH_CAMPAIGN: rule_syslog_snmp_auth_campaign,
    PORT_INSTABILITY: rule_port_instability,
    LOOP_OR_REDUNDANCY_INSTABILITY: rule_loop_or_redundancy,
    DUPLICATE_IP_CONFLICT: rule_duplicate_ip,
    POE_ENDPOINT_FAILURE: rule_poe_endpoint_failure,
    MANAGEMENT_ACCESS_EXPOSURE: rule_management_access_exposure,
    CLOCK_INTEGRITY_RISK: rule_clock_integrity,
}

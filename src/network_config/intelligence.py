"""Engine C Phase 7 — configuration-intelligence synthesis.

Purpose
-------
Consolidate the already-persisted Engine C artefacts (inventory, topology,
findings, remediation plan, dry-run audit, and an optional snapshot diff) into
a single operator-facing summary: a deterministic risk score per finding,
conservative root-cause hypotheses, and prioritised operator action items.

This module reads artefacts only — it never recomputes inventory/topology/
findings, never mutates an artefact, never contacts a device and never executes
a command. All reasoning is deterministic and evidence-based; root-cause wording
is deliberately cautious (``possible`` / ``candidate`` / ``likely``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.network_config.findings import SEVERITY_ORDER

logger = logging.getLogger(__name__)

SAFETY_NOTE = (
    "Offline analysis only; no commands were executed. Remediation plans are "
    "dry-run and require explicit human confirmation. Dry-run validation is "
    "not live-device verification."
)

# --- deterministic risk-scoring weights (explainable, no ML) ----------------
_SEVERITY_WEIGHT = {"critical": 100, "high": 75, "medium": 50, "low": 25,
                    "info": 10}
_CONFIDENCE_FACTOR = {"high": 1.0, "medium": 0.85, "low": 0.7}
_CATEGORY_BUMP = {"security": 10, "loop risk": 10, "loop_risk": 10, "stp": 5,
                  "topology": 3}
_TOPOLOGY_RELEVANCE_BUMP = 5
_COMMAND_REMEDIATION_BUMP = 5
_VERIFICATION_DELTA = {"failed": 15, "unknown": 5, "passed": -25,
                       "not_applicable": 0}
# risk_score -> risk_level thresholds (inclusive lower bounds).
_RISK_LEVELS = (("critical", 85), ("high", 65), ("medium", 40), ("low", 20),
                ("info", 0))


# ------------------------------------------------------------------- models


@dataclass(frozen=True)
class RiskSummary:
    """A deterministic, explainable risk score for one finding."""

    risk_score: int                     # 0-100
    risk_level: str                     # critical/high/medium/low/info
    contributing_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceBundle:
    """The corroborating evidence behind a hypothesis or action item."""

    findings: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RootCauseHypothesis:
    """A cautious, evidence-based root-cause hypothesis (never asserted fact)."""

    hypothesis_id: str
    title: str
    severity: str
    confidence: str                     # possible/candidate/likely
    affected_devices: tuple[str, ...]
    affected_interfaces: tuple[str, ...]
    related_findings: tuple[str, ...]
    evidence: tuple[str, ...]
    explanation: str
    recommended_next_steps: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorActionItem:
    """A prioritised, human-owned next step (never auto-executed)."""

    action_item_id: str
    priority: str                       # P1/P2/P3
    title: str
    device: Optional[str]
    interface: Optional[str]
    source_finding_id: Optional[str]
    source_action_id: Optional[str]
    action_type: str                    # investigate/config-plan/verify/monitor
    safety_status: str
    recommended_owner: str              # network/security/ops
    next_step: str


@dataclass(frozen=True)
class ConfigIntelligenceSummary:
    """The serialisable roll-up written to config_intelligence_summary.json."""

    snapshot_id: str
    timestamp: str
    total_interfaces: int
    total_vlans: int
    total_topology_edges: int
    total_findings: int
    findings_by_severity: dict[str, int]
    findings_by_category: dict[str, int]
    total_remediation_actions: int
    command_actions: int
    investigation_actions: int
    blocked_actions: int
    dry_run_available: bool
    diff_available: bool
    verification_passed: int
    verification_failed: int
    verification_unknown: int
    top_risks: tuple[dict[str, Any], ...]
    root_cause_hypotheses_count: int
    operator_action_items_count: int
    safety_note: str = SAFETY_NOTE


@dataclass(frozen=True)
class ConfigIntelligence:
    """Everything the report needs: summary + ranked risks + reasoning."""

    summary: ConfigIntelligenceSummary
    ranked_findings: tuple[tuple[dict[str, Any], RiskSummary], ...]
    hypotheses: tuple[RootCauseHypothesis, ...]
    action_items: tuple[OperatorActionItem, ...]
    artifacts: "SnapshotArtifacts"
    diff: Optional["DiffArtifacts"] = None


# ----------------------------------------------------------------- loaders


@dataclass
class SnapshotArtifacts:
    """Loaded per-snapshot artefacts (inventory required, rest optional)."""

    snapshot_id: str
    directory: str
    inventory: dict[str, Any]
    topology: Optional[dict[str, Any]] = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    rule_summary: Optional[dict[str, Any]] = None
    remediation_plan: Optional[dict[str, Any]] = None
    remediation_summary: Optional[dict[str, Any]] = None
    dry_run_execution: Optional[dict[str, Any]] = None
    execution_summary: Optional[dict[str, Any]] = None
    batfish: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DiffArtifacts:
    """Loaded diff artefacts for a diff-aware report."""

    diff_id: str
    snapshot_diff: Optional[dict[str, Any]] = None
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    diff_summary: Optional[dict[str, Any]] = None
    warnings: list[str] = field(default_factory=list)


def _read(path: Path, warnings: list[str], label: str) -> Any:
    if not path.is_file():
        warnings.append(f"optional {label} artefact missing: {path.name}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warnings.append(f"could not parse {label} artefact {path.name}: {exc}")
        return None


def load_snapshot_artifacts(
    directory: str | Path, snapshot_id: str | None = None
) -> SnapshotArtifacts:
    """Load all snapshot artefacts. ``inventory.json`` is required."""
    root = Path(directory)
    inv_path = root / "inventory.json"
    if not inv_path.is_file():
        raise FileNotFoundError(f"Required inventory.json not found in {root}")
    try:
        inventory = json.loads(inv_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid inventory.json in {root}: {exc}") from exc
    if not isinstance(inventory, dict) or "devices" not in inventory:
        raise ValueError(f"inventory.json in {root} is missing 'devices'")

    warnings: list[str] = []
    findings = _read(root / "findings.json", warnings, "findings")
    return SnapshotArtifacts(
        snapshot_id=snapshot_id or str(inventory.get("snapshot_id", root.name)),
        directory=str(root),
        inventory=inventory,
        topology=_as_dict(_read(root / "topology.json", warnings, "topology")),
        findings=findings if isinstance(findings, list) else [],
        rule_summary=_as_dict(_read(root / "rule_summary.json", warnings,
                                    "rule_summary")),
        remediation_plan=_as_dict(_read(root / "remediation_plan.json", warnings,
                                        "remediation_plan")),
        remediation_summary=_as_dict(_read(root / "remediation_summary.json",
                                           warnings, "remediation_summary")),
        dry_run_execution=_as_dict(_read(root / "dry_run_execution.json",
                                         warnings, "dry_run_execution")),
        execution_summary=_as_dict(_read(root / "execution_summary.json",
                                         warnings, "execution_summary")),
        batfish=_as_dict(_read(root / "batfish" / "batfish_summary.json",
                               warnings, "batfish")),
        warnings=warnings,
    )


def load_diff_artifacts(directory: str | Path, diff_id: str) -> DiffArtifacts:
    """Load optional diff artefacts (all fields tolerated missing)."""
    root = Path(directory)
    warnings: list[str] = []
    verifications = _read(root / "verification_results.json", warnings,
                          "verification_results")
    return DiffArtifacts(
        diff_id=diff_id,
        snapshot_diff=_as_dict(_read(root / "snapshot_diff.json", warnings,
                                     "snapshot_diff")),
        verification_results=verifications if isinstance(verifications, list)
        else [],
        diff_summary=_as_dict(_read(root / "diff_summary.json", warnings,
                                    "diff_summary")),
        warnings=warnings,
    )


def _as_dict(value: Any) -> Optional[dict[str, Any]]:
    return value if isinstance(value, dict) else None


# ------------------------------------------------------------ risk scoring


def risk_level_for(score: int) -> str:
    """Map a 0-100 risk score to a discrete level (deterministic thresholds)."""
    for level, lower in _RISK_LEVELS:
        if score >= lower:
            return level
    return "info"


def score_finding(
    finding: dict[str, Any],
    *,
    topology_relevant: bool = False,
    has_command_remediation: bool = False,
    verification_status: Optional[str] = None,
) -> RiskSummary:
    """Return a deterministic :class:`RiskSummary` for one finding."""
    severity = str(finding.get("severity", "info"))
    confidence = str(finding.get("confidence", "medium"))
    category = str(finding.get("category", ""))

    base = _SEVERITY_WEIGHT.get(severity, 10)
    factor = _CONFIDENCE_FACTOR.get(confidence, 0.85)
    score = base * factor
    factors = [f"severity={severity}({base})",
               f"confidence={confidence}(x{factor})"]

    bump = _CATEGORY_BUMP.get(category, 0)
    if bump:
        score += bump
        factors.append(f"category={category}(+{bump})")
    if topology_relevant:
        score += _TOPOLOGY_RELEVANCE_BUMP
        factors.append(f"topology-relevant(+{_TOPOLOGY_RELEVANCE_BUMP})")
    if has_command_remediation:
        score += _COMMAND_REMEDIATION_BUMP
        factors.append(f"config-change-needed(+{_COMMAND_REMEDIATION_BUMP})")
    if verification_status in _VERIFICATION_DELTA:
        delta = _VERIFICATION_DELTA[verification_status]
        if delta:
            score += delta
            factors.append(f"verification={verification_status}({delta:+d})")

    clamped = max(0, min(100, int(round(score))))
    return RiskSummary(risk_score=clamped, risk_level=risk_level_for(clamped),
                       contributing_factors=tuple(factors))


# --------------------------------------------------------- index utilities


def _iter_inventory(inventory: dict[str, Any]):
    for device in inventory.get("devices") or []:
        yield str((device.get("device") or {}).get("device_id", "unknown")), device


def _topology_interfaces(topology: Optional[dict[str, Any]]) -> set[tuple[str, str]]:
    """(device, interface) pairs that appear on a topology edge."""
    out: set[tuple[str, str]] = set()
    if not topology:
        return out
    for edge in topology.get("edges") or []:
        out.add((str(edge.get("local_device")), str(edge.get("local_interface"))))
        out.add((str(edge.get("remote_device")),
                 str(edge.get("remote_interface"))))
    return out


def _command_bearing_finding_ids(plan: Optional[dict[str, Any]]) -> set[str]:
    if not plan:
        return set()
    return {
        str(a.get("finding_id"))
        for a in plan.get("actions") or []
        if a.get("commands") and str(a.get("status")) == "planned"
    }


def _verification_by_finding(diff: Optional[DiffArtifacts]) -> dict[str, str]:
    if not diff:
        return {}
    return {
        str(v.get("finding_id")): str(v.get("status"))
        for v in diff.verification_results
        if v.get("finding_id")
    }


# ----------------------------------------------------- root-cause hypotheses


def _hid(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"RCH-{digest[:8]}"


def _findings_by_rule(findings: list[dict[str, Any]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in findings:
        if str(f.get("status", "open")) == "open":
            out.setdefault(str(f.get("rule_id")), []).append(f)
    return out


def _confidence_for(evidence_count: int) -> str:
    """Cautious wording keyed on how many signals corroborate the hypothesis."""
    if evidence_count >= 3:
        return "likely"
    if evidence_count == 2:
        return "candidate"
    return "possible"


def generate_hypotheses(
    artifacts: SnapshotArtifacts, diff: Optional[DiffArtifacts] = None
) -> list[RootCauseHypothesis]:
    """Generate conservative root-cause hypotheses from finding combinations."""
    by_rule = _findings_by_rule(artifacts.findings)
    stp_ifaces = {(str(d), str(i.get("interface")))
                  for d, dev in _iter_inventory(artifacts.inventory)
                  for i in dev.get("stp_states") or []}
    hypotheses: list[RootCauseHypothesis] = []

    hypotheses += _hyp_loop(by_rule)
    hypotheses += _hyp_stale_uplink(by_rule, stp_ifaces)
    hypotheses += _hyp_unused_exposed(by_rule, artifacts.inventory)
    hypotheses += _hyp_vlan_not_remediated(by_rule, diff)
    hypotheses += _hyp_unauthorized_vlan(diff)
    return hypotheses


def _dev_iface(findings: list[dict]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    devices = tuple(sorted({str(f.get("device")) for f in findings
                            if f.get("device")}))
    ifaces = tuple(sorted({str(f.get("interface")) for f in findings
                           if f.get("interface")}))
    return devices, ifaces


def _hyp_loop(by_rule) -> list[RootCauseHypothesis]:
    stp = by_rule.get("STP_BLOCKING_ACCESS_PORT", [])
    mac = by_rule.get("MAC_ON_MULTIPLE_INTERFACES", [])
    if not (stp and mac):
        return []
    related = stp + mac
    devices, ifaces = _dev_iface(related)
    evidence = (f"{len(stp)} STP-blocking access-port finding(s)",
                f"{len(mac)} MAC-on-multiple-interfaces finding(s)")
    return [RootCauseHypothesis(
        hypothesis_id=_hid("loop", devices, ifaces),
        title="Possible layer-2 loop or unmanaged switch",
        severity="high", confidence=_confidence_for(len(related)),
        affected_devices=devices, affected_interfaces=ifaces,
        related_findings=tuple(f.get("finding_id") for f in related),
        evidence=evidence,
        explanation="An access port in STP blocking together with the same MAC "
                    "seen on multiple interfaces is consistent with a bridging "
                    "loop or an unmanaged switch cabled into an access port.",
        recommended_next_steps=(
            "Inspect cabling/patch-panel mapping for the affected ports.",
            "Confirm the MAC movement is a loop, flap or duplicate before any "
            "change."),
        tags=("loop", "stp", "l2"))]


def _hyp_stale_uplink(by_rule, stp_ifaces) -> list[RootCauseHypothesis]:
    trunk_no_nbr = by_rule.get("TRUNK_WITHOUT_NEIGHBOR", [])
    hits = [f for f in trunk_no_nbr
            if (str(f.get("device")), str(f.get("interface"))) not in stp_ifaces]
    if not hits:
        return []
    devices, ifaces = _dev_iface(hits)
    return [RootCauseHypothesis(
        hypothesis_id=_hid("stale_uplink", devices, ifaces),
        title="Possible undocumented uplink or stale trunk",
        severity="medium", confidence=_confidence_for(1 + len(hits)),
        affected_devices=devices, affected_interfaces=ifaces,
        related_findings=tuple(f.get("finding_id") for f in hits),
        evidence=(f"{len(hits)} trunk(s) with no LLDP/CDP neighbour",
                  "no spanning-tree data present for the trunk"),
        explanation="A trunk with no discovered neighbour and no spanning-tree "
                    "state may be an undocumented uplink or a stale trunk "
                    "configuration left on an unused port.",
        recommended_next_steps=(
            "Verify LLDP/CDP is enabled and re-capture neighbour tables.",
            "Confirm the trunk is intended; otherwise plan to convert or "
            "disable it."),
        tags=("topology", "trunk"))]


def _hyp_unused_exposed(by_rule, inventory) -> list[RootCauseHypothesis]:
    unused = by_rule.get("UNUSED_PORT_ADMIN_UP", [])
    if not unused:
        return []
    macs = {(str(d), str(m.get("interface")))
            for d, dev in _iter_inventory(inventory)
            for m in dev.get("mac_entries") or []}
    nbrs = {(str(d), str(n.get("local_interface")))
            for d, dev in _iter_inventory(inventory)
            for n in dev.get("neighbors") or []}
    exposed = [f for f in unused
               if (str(f.get("device")), str(f.get("interface"))) not in macs
               and (str(f.get("device")), str(f.get("interface"))) not in nbrs]
    if not exposed:
        return []
    devices, ifaces = _dev_iface(exposed)
    return [RootCauseHypothesis(
        hypothesis_id=_hid("unused_exposed", devices, ifaces),
        title="Unused exposed port should be disabled",
        severity="medium", confidence="likely",
        affected_devices=devices, affected_interfaces=ifaces,
        related_findings=tuple(f.get("finding_id") for f in exposed),
        evidence=(f"{len(exposed)} unused admin-up port(s)",
                  "no learned MACs on the port(s)",
                  "no LLDP/CDP neighbour on the port(s)"),
        explanation="An administratively-up port with no MACs and no neighbour "
                    "is an unused but exposed access point into the network and "
                    "is a candidate for shutdown.",
        recommended_next_steps=(
            "Confirm the port is genuinely unused.",
            "Plan a dry-run shutdown (requires human confirmation)."),
        tags=("security", "port"))]


def _hyp_vlan_not_remediated(by_rule, diff) -> list[RootCauseHypothesis]:
    missing = by_rule.get("TRUNK_MISSING_REQUIRED_VLAN", [])
    if not (missing and diff):
        return []
    ver = _verification_by_finding(diff)
    unresolved = [f for f in missing
                  if ver.get(str(f.get("finding_id"))) in ("failed", "unknown")]
    if not unresolved:
        return []
    devices, ifaces = _dev_iface(unresolved)
    return [RootCauseHypothesis(
        hypothesis_id=_hid("vlan_not_remediated", devices, ifaces),
        title="VLAN propagation issue not remediated",
        severity="high", confidence=_confidence_for(1 + len(unresolved)),
        affected_devices=devices, affected_interfaces=ifaces,
        related_findings=tuple(f.get("finding_id") for f in unresolved),
        evidence=("required VLAN missing from trunk allowed list",
                  "verification did not confirm the VLAN after the diff"),
        explanation="A required VLAN is still absent from the trunk after the "
                    "compared change, so VLAN propagation for the affected "
                    "segment may remain broken.",
        recommended_next_steps=(
            "Re-capture 'show interfaces trunk' after applying the plan.",
            "Confirm the VLAN exists end-to-end along the path."),
        tags=("vlan", "trunk", "diff"))]


def _hyp_unauthorized_vlan(diff) -> list[RootCauseHypothesis]:
    if not diff or not diff.snapshot_diff:
        return []
    added = [r for r in diff.snapshot_diff.get("records") or []
             if r.get("category") == "finding" and r.get("change_type") == "added"
             and r.get("field") == "TRUNK_UNAUTHORIZED_VLAN"]
    if not added:
        return []
    devices = tuple(sorted({str(r.get("device")) for r in added
                            if r.get("device")}))
    ifaces = tuple(sorted({str(r.get("interface")) for r in added
                           if r.get("interface")}))
    return [RootCauseHypothesis(
        hypothesis_id=_hid("unauthorized_vlan", devices, ifaces),
        title="Unauthorized VLAN exposed on trunk after change",
        severity="high", confidence="candidate",
        affected_devices=devices, affected_interfaces=ifaces,
        related_findings=(),
        evidence=("a TRUNK_UNAUTHORIZED_VLAN finding is new in the after "
                  "snapshot",),
        explanation="An unauthorized VLAN appears on a trunk only in the after "
                    "snapshot, so a recent change may have introduced or "
                    "exposed a VLAN policy violation.",
        recommended_next_steps=(
            "Review the trunk allowed-VLAN change against policy.",
            "Plan removal of the unauthorized VLAN (requires confirmation)."),
        tags=("vlan", "security", "diff"))]


# ----------------------------------------------------- operator action items


_SEVERITY_PRIORITY = {"critical": "P1", "high": "P1", "medium": "P2",
                      "low": "P3", "info": "P3"}
_SECURITY_CATEGORIES = {"security", "loop risk", "loop_risk"}


def _aid(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"AI-{digest[:8]}"


def _owner_for(category: str) -> str:
    return "security" if category in _SECURITY_CATEGORIES else "network"


def generate_action_items(
    artifacts: SnapshotArtifacts, diff: Optional[DiffArtifacts] = None
) -> list[OperatorActionItem]:
    """Prioritised operator action items from remediation actions + verifications."""
    items: list[OperatorActionItem] = []
    findings_by_id = {str(f.get("finding_id")): f for f in artifacts.findings}

    plan = artifacts.remediation_plan
    if plan:
        for action in plan.get("actions") or []:
            item = _item_for_action(action, findings_by_id)
            if item:
                items.append(item)
    else:
        for finding in artifacts.findings:
            if str(finding.get("status", "open")) != "open":
                continue
            if finding.get("severity") in ("critical", "high"):
                items.append(_item_for_finding(finding))

    if diff:
        for verification in diff.verification_results:
            items.append(_item_for_verification(verification))

    order = {"P1": 0, "P2": 1, "P3": 2}
    items.sort(key=lambda i: (order.get(i.priority, 9), i.action_type,
                              i.device or "", i.interface or ""))
    return items


def _item_for_action(action, findings_by_id) -> Optional[OperatorActionItem]:
    status = str(action.get("status"))
    severity = str(action.get("severity", "medium"))
    device, interface = action.get("device"), action.get("interface")
    base = dict(device=device, interface=interface,
                source_finding_id=action.get("finding_id"),
                source_action_id=action.get("action_id"))
    if status == "planned" and action.get("commands"):
        return OperatorActionItem(
            action_item_id=_aid("action", action.get("action_id")),
            priority=_SEVERITY_PRIORITY.get(severity, "P2"),
            title=f"Review dry-run plan: {action.get('title')}",
            action_type="config-plan",
            safety_status="requires human confirmation (dry-run only)",
            recommended_owner="network",
            next_step="Review the proposed commands and rollback, then apply "
                      "only after explicit confirmation.",
            **base)
    if action.get("action_type") == "investigation":
        category = str((findings_by_id.get(str(action.get("finding_id")))
                        or {}).get("category", ""))
        return OperatorActionItem(
            action_item_id=_aid("action", action.get("action_id")),
            priority=_SEVERITY_PRIORITY.get(severity, "P2"),
            title=f"Investigate: {action.get('title')}",
            action_type="investigate", safety_status="read-only investigation",
            recommended_owner=_owner_for(category),
            next_step="Follow the investigation steps; no configuration change "
                      "is proposed yet.", **base)
    if status in ("blocked", "skipped"):
        return OperatorActionItem(
            action_item_id=_aid("action", action.get("action_id")),
            priority="P3",
            title=f"Manual review ({status}): {action.get('title')}",
            action_type="investigate",
            safety_status="no automated remediation available",
            recommended_owner="network",
            next_step=f"No safe template applied ({action.get('reason') or status}); "
                      "review manually.", **base)
    return None


def _item_for_finding(finding) -> OperatorActionItem:
    category = str(finding.get("category", ""))
    return OperatorActionItem(
        action_item_id=_aid("finding", finding.get("finding_id")),
        priority=_SEVERITY_PRIORITY.get(str(finding.get("severity")), "P2"),
        title=f"Address finding: {finding.get('title')}",
        device=finding.get("device"), interface=finding.get("interface"),
        source_finding_id=finding.get("finding_id"), source_action_id=None,
        action_type="investigate", safety_status="read-only investigation",
        recommended_owner=_owner_for(category),
        next_step="Review the finding evidence and plan a safe remediation.")


def _item_for_verification(verification) -> OperatorActionItem:
    status = str(verification.get("status"))
    spec = {
        "failed": ("P1", "verify", "Verification failed; re-apply the plan and "
                   "re-capture state.", "network"),
        "unknown": ("P2", "verify", "Verification inconclusive; manually check "
                    "the after state.", "network"),
        "passed": ("P3", "monitor", "Verification passed; monitor and close the "
                   "item.", "ops"),
        "not_applicable": ("P3", "monitor", "No config change was planned; "
                           "monitor only.", "ops"),
    }.get(status, ("P2", "verify", "Review verification result.", "network"))
    priority, action_type, next_step, owner = spec
    return OperatorActionItem(
        action_item_id=_aid("verification", verification.get("verification_id")),
        priority=priority,
        title=f"{status.replace('_', ' ').title()} verification: "
              f"{verification.get('rule_id')}",
        device=verification.get("device"), interface=verification.get("interface"),
        source_finding_id=verification.get("finding_id"),
        source_action_id=verification.get("action_id"),
        action_type=action_type, safety_status="offline verification only",
        recommended_owner=owner, next_step=next_step)


# --------------------------------------------------------------- orchestrator


def build_intelligence(
    artifacts: SnapshotArtifacts, diff: Optional[DiffArtifacts] = None
) -> ConfigIntelligence:
    """Assemble the full configuration-intelligence object (deterministic)."""
    topo_ifaces = _topology_interfaces(artifacts.topology)
    cmd_finding_ids = _command_bearing_finding_ids(artifacts.remediation_plan)
    verifications = _verification_by_finding(diff)

    ranked: list[tuple[dict[str, Any], RiskSummary]] = []
    for finding in artifacts.findings:
        if str(finding.get("status", "open")) != "open":
            continue
        key = (str(finding.get("device")), str(finding.get("interface")))
        risk = score_finding(
            finding,
            topology_relevant=key in topo_ifaces,
            has_command_remediation=str(finding.get("finding_id")) in cmd_finding_ids,
            verification_status=verifications.get(str(finding.get("finding_id"))),
        )
        ranked.append((finding, risk))
    ranked.sort(key=lambda pair: (-pair[1].risk_score,
                                  SEVERITY_ORDER.get(pair[0].get("severity"), 9)))

    hypotheses = generate_hypotheses(artifacts, diff)
    action_items = generate_action_items(artifacts, diff)
    summary = _build_summary(artifacts, diff, ranked, hypotheses, action_items)
    return ConfigIntelligence(
        summary=summary, ranked_findings=tuple(ranked),
        hypotheses=tuple(hypotheses), action_items=tuple(action_items),
        artifacts=artifacts, diff=diff)


def _build_summary(artifacts, diff, ranked, hypotheses, action_items
                   ) -> ConfigIntelligenceSummary:
    inv = artifacts.inventory
    total_interfaces = sum(len(d.get("interfaces") or [])
                           for _, d in _iter_inventory(inv))
    total_vlans = sum(len(d.get("vlans") or []) for _, d in _iter_inventory(inv))
    topo = artifacts.topology or {}
    rule_sum = artifacts.rule_summary or {}
    rem_sum = artifacts.remediation_summary or {}
    diff_sum = (diff.diff_summary if diff else None) or {}

    top_risks = tuple(
        {"finding_id": f.get("finding_id"), "rule_id": f.get("rule_id"),
         "title": f.get("title"), "device": f.get("device"),
         "interface": f.get("interface"), "risk_score": r.risk_score,
         "risk_level": r.risk_level}
        for f, r in ranked[:5])

    open_findings = [f for f in artifacts.findings
                     if str(f.get("status", "open")) == "open"]
    return ConfigIntelligenceSummary(
        snapshot_id=artifacts.snapshot_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_interfaces=total_interfaces,
        total_vlans=total_vlans,
        total_topology_edges=len(topo.get("edges") or []),
        total_findings=len(open_findings),
        findings_by_severity=dict(rule_sum.get("findings_by_severity") or
                                  _count(open_findings, "severity")),
        findings_by_category=dict(rule_sum.get("findings_by_category") or
                                  _count(open_findings, "category")),
        total_remediation_actions=int(rem_sum.get("total_actions", 0)),
        command_actions=int(rem_sum.get("command_actions", 0)),
        investigation_actions=int(rem_sum.get("investigation_actions", 0)),
        blocked_actions=int(rem_sum.get("blocked_actions", 0)),
        dry_run_available=artifacts.execution_summary is not None,
        diff_available=diff is not None,
        verification_passed=int(diff_sum.get("verification_passed", 0)),
        verification_failed=int(diff_sum.get("verification_failed", 0)),
        verification_unknown=int(diff_sum.get("verification_unknown", 0)),
        top_risks=top_risks,
        root_cause_hypotheses_count=len(hypotheses),
        operator_action_items_count=len(action_items))


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row.get(key))] = out.get(str(row.get(key)), 0) + 1
    return out

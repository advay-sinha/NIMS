"""Engine C Phase 3 — YAML-driven configuration rule engine.

Purpose
-------
Evaluate a parsed :class:`~src.network_config.models.NetworkInventory` (and,
when available, a :class:`~src.network_config.topology.NetworkTopology`) against
a set of YAML-defined rules and emit structured
:class:`~src.network_config.findings.Finding` objects. Detection only — no
remediation is executed here.

Design
------
Each rule is a small pure function ``(RuleContext) -> list[dict]`` returning
finding drafts; the engine attaches rule metadata (severity/category/confidence
from YAML or the rule default), a deterministic id and suppression status.
Thresholds, expected VLANs and keywords all come from configuration — never
hardcoded here.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from src.network_config.findings import Finding, SEVERITY_ORDER, make_finding_id
from src.network_config.models import NetworkInventory
from src.network_config.topology import (
    NetworkTopology,
    _norm_device,
    _norm_iface,
)

logger = logging.getLogger(__name__)


@dataclass
class RuleContext:
    """Inputs available to a rule function."""

    inventory: NetworkInventory
    topology: Optional[NetworkTopology]
    config: Mapping[str, Any]                 # this rule's YAML block


@dataclass(frozen=True)
class RuleSpec:
    """Static metadata for a registered rule."""

    rule_id: str
    func: Callable[[RuleContext], list[dict]]
    category: str
    default_severity: str
    title: str
    requires_topology: bool = False
    default_confidence: str = "medium"
    source: str = "inventory"


# ------------------------------------------------------------- rule helpers


def _devices(ctx: RuleContext):
    return ctx.inventory.devices


def _trunk_names(snap: Any) -> list[str]:
    names = {t.interface for t in snap.trunks}
    names |= {i.name for i in snap.interfaces if i.mode == "trunk"}
    return sorted(names)


def _access_names(snap: Any) -> list[str]:
    return sorted(i.name for i in snap.interfaces if i.mode == "access")


def _str_set(values: Any) -> set[str]:
    return {str(v) for v in (values or [])}


# ------------------------------------------------------------------- rules


def _rule_access_disallowed_vlan(ctx: RuleContext) -> list[dict]:
    allowed = _str_set(ctx.config.get("allowed_vlans"))
    if not allowed:
        return []
    out: list[dict] = []
    for snap in _devices(ctx):
        for iface in snap.interfaces:
            if iface.mode == "access" and iface.vlan and str(iface.vlan) not in allowed:
                out.append({
                    "device": snap.device.device_id, "interface": iface.name,
                    "vlan": str(iface.vlan),
                    "evidence": (f"access VLAN {iface.vlan} is not in the "
                                 f"allowed set {sorted(allowed, key=_vlan_key)}"),
                    "recommendation": ("Reassign the port to an approved VLAN "
                                       "or update the allowed-VLAN policy."),
                })
    return out


def _rule_trunk_missing_required_vlan(ctx: RuleContext) -> list[dict]:
    required = _str_set(ctx.config.get("required_vlans"))
    if not required:
        return []
    out: list[dict] = []
    for snap in _devices(ctx):
        for trunk in snap.trunks:
            missing = required - _str_set(trunk.allowed_vlans)
            if missing:
                ordered = sorted(missing, key=_vlan_key)
                out.append({
                    "device": snap.device.device_id,
                    "interface": trunk.interface,
                    "evidence": (f"trunk is missing required VLAN(s): "
                                 f"{ordered}"),
                    "recommendation": ("Add the required VLAN(s) to the trunk "
                                       "allowed list."),
                    "details": {"missing_vlans": ordered},
                })
    return out


def _rule_trunk_unauthorized_vlan(ctx: RuleContext) -> list[dict]:
    authorized = _str_set(ctx.config.get("authorized_vlans"))
    if not authorized:
        return []
    out: list[dict] = []
    for snap in _devices(ctx):
        for trunk in snap.trunks:
            extra = _str_set(trunk.allowed_vlans) - authorized
            if extra:
                ordered = sorted(extra, key=_vlan_key)
                out.append({
                    "device": snap.device.device_id,
                    "interface": trunk.interface,
                    "evidence": (f"trunk allows unauthorized VLAN(s): "
                                 f"{ordered}"),
                    "recommendation": ("Prune unauthorized VLANs from the trunk "
                                       "allowed list."),
                    "details": {"unauthorized_vlans": ordered},
                })
    return out


def _rule_native_vlan_mismatch(ctx: RuleContext) -> list[dict]:
    expected = ctx.config.get("expected_native_vlan")
    if expected is None:
        return []
    out: list[dict] = []
    for snap in _devices(ctx):
        for trunk in snap.trunks:
            if trunk.native_vlan is not None and str(trunk.native_vlan) != str(
                expected
            ):
                out.append({
                    "device": snap.device.device_id,
                    "interface": trunk.interface, "vlan": str(trunk.native_vlan),
                    "evidence": (f"candidate native VLAN mismatch: trunk native "
                                 f"{trunk.native_vlan}, expected {expected}"),
                    "recommendation": ("Confirm the intended native VLAN for "
                                       "this trunk."),
                })
    return out


def _rule_access_too_many_macs(ctx: RuleContext) -> list[dict]:
    threshold = int(ctx.config.get("threshold", 5))
    out: list[dict] = []
    for snap in _devices(ctx):
        access = set(_access_names(snap))
        macs: dict[str, set[str]] = defaultdict(set)
        for entry in snap.mac_entries:
            if entry.interface:
                macs[entry.interface].add(entry.mac_address)
        for iface in sorted(access):
            count = len(macs.get(iface, set()))
            if count > threshold:
                out.append({
                    "device": snap.device.device_id, "interface": iface,
                    "evidence": (f"{count} MAC address(es) learned on an access "
                                 f"port (> threshold {threshold})"),
                    "recommendation": ("Investigate for an unmanaged switch/hub "
                                       "or misconfiguration downstream."),
                })
    return out


def _rule_poe_disabled_expected(ctx: RuleContext) -> list[dict]:
    keywords = [str(k).lower() for k in ctx.config.get("expected_poe_keywords") or []]
    if not keywords:
        return []
    out: list[dict] = []
    for snap in _devices(ctx):
        for iface in snap.interfaces:
            desc = (iface.description or "").lower()
            if not desc or not any(k in desc for k in keywords):
                continue
            if iface.poe_enabled is not True or (
                iface.poe_state and iface.poe_state != "on"
            ):
                state = iface.poe_state or ("disabled"
                                            if iface.poe_enabled is False
                                            else "unknown")
                out.append({
                    "device": snap.device.device_id, "interface": iface.name,
                    "evidence": (f"description '{iface.description}' suggests a "
                                 f"powered device but PoE is {state}"),
                    "recommendation": ("Enable PoE on this port if it powers an "
                                       "AP/phone/camera."),
                })
    return out


def _rule_unused_port_admin_up(ctx: RuleContext) -> list[dict]:
    out: list[dict] = []
    for snap in _devices(ctx):
        for iface in snap.interfaces:
            # notconnect = administratively enabled but no link (unused-but-up).
            if (iface.status or "").lower() == "notconnect":
                out.append({
                    "device": snap.device.device_id, "interface": iface.name,
                    "evidence": (f"port is operationally down (status "
                                 f"'{iface.status}') but administratively "
                                 "enabled"),
                    "recommendation": ("Shut down unused ports or add a "
                                       "description if the port is in use."),
                })
    return out


def _rule_stp_blocking_access_port(ctx: RuleContext) -> list[dict]:
    out: list[dict] = []
    for snap in _devices(ctx):
        access = set(_access_names(snap))
        for state in snap.stp_states:
            if state.state == "blocking" and state.interface in access:
                out.append({
                    "device": snap.device.device_id,
                    "interface": state.interface, "vlan": state.vlan,
                    "evidence": (f"VLAN {state.vlan} role {state.role} state "
                                 "blocking on an access port"),
                    "recommendation": ("Investigate a possible loop or "
                                       "misconfiguration; access ports should "
                                       "not block."),
                })
    return out


def _rule_mac_multiple_interfaces(ctx: RuleContext) -> list[dict]:
    out: list[dict] = []
    for snap in _devices(ctx):
        iface_per_mac: dict[str, set[str]] = defaultdict(set)
        for entry in snap.mac_entries:
            if entry.interface:
                iface_per_mac[entry.mac_address].add(entry.interface)
        for mac in sorted(iface_per_mac):
            ifaces = iface_per_mac[mac]
            if len(ifaces) > 1:
                out.append({
                    "device": snap.device.device_id,
                    "evidence": (f"candidate loop/flap: MAC {mac} learned on "
                                 f"multiple interfaces {sorted(ifaces)}"),
                    "recommendation": ("Check for a loop, port flapping or "
                                       "duplicate learning."),
                })
    return out


def _rule_trunk_without_neighbor(ctx: RuleContext) -> list[dict]:
    topology = ctx.topology
    assert topology is not None  # guarded by requires_topology in the engine
    edge_ports = {
        (_norm_device(e.local_device), _norm_iface(e.local_interface))
        for e in topology.edges
    }
    out: list[dict] = []
    for snap in _devices(ctx):
        device = _norm_device(snap.device.device_id)
        for iface in _trunk_names(snap):
            if (device, _norm_iface(iface)) not in edge_ports:
                out.append({
                    "device": snap.device.device_id, "interface": iface,
                    "evidence": ("trunk port has no discovered LLDP/CDP "
                                 "neighbour (candidate unmanaged neighbour or "
                                 "misconfiguration)"),
                    "recommendation": ("Verify neighbour discovery or the "
                                       "expected connectivity of this trunk."),
                })
    return out


def _vlan_key(value: str) -> tuple[int, str]:
    """Sort VLAN id strings numerically when possible."""
    return (0, f"{int(value):06d}") if str(value).isdigit() else (1, str(value))


# --------------------------------------------------------------- registry


RULES: dict[str, RuleSpec] = {
    spec.rule_id: spec
    for spec in (
        RuleSpec("ACCESS_PORT_DISALLOWED_VLAN", _rule_access_disallowed_vlan,
                 "vlan", "high", "Access port on disallowed VLAN",
                 default_confidence="high"),
        RuleSpec("TRUNK_MISSING_REQUIRED_VLAN", _rule_trunk_missing_required_vlan,
                 "vlan", "high", "Trunk missing required VLAN",
                 default_confidence="high"),
        RuleSpec("TRUNK_UNAUTHORIZED_VLAN", _rule_trunk_unauthorized_vlan,
                 "vlan", "medium", "Trunk allows unauthorized VLAN",
                 default_confidence="high"),
        RuleSpec("NATIVE_VLAN_MISMATCH", _rule_native_vlan_mismatch,
                 "vlan", "medium", "Native VLAN mismatch candidate",
                 default_confidence="medium"),
        RuleSpec("ACCESS_PORT_TOO_MANY_MACS", _rule_access_too_many_macs,
                 "port", "medium", "Access port has too many MAC addresses",
                 default_confidence="medium"),
        RuleSpec("POE_DISABLED_EXPECTED", _rule_poe_disabled_expected,
                 "poe", "high", "PoE disabled on port expected to power a device",
                 default_confidence="medium"),
        RuleSpec("UNUSED_PORT_ADMIN_UP", _rule_unused_port_admin_up,
                 "port", "low", "Unused port administratively enabled",
                 default_confidence="low"),
        RuleSpec("STP_BLOCKING_ACCESS_PORT", _rule_stp_blocking_access_port,
                 "stp", "medium", "STP blocking on access port",
                 default_confidence="high"),
        RuleSpec("MAC_ON_MULTIPLE_INTERFACES", _rule_mac_multiple_interfaces,
                 "stp", "medium", "MAC learned on multiple interfaces",
                 default_confidence="low"),
        RuleSpec("TRUNK_WITHOUT_NEIGHBOR", _rule_trunk_without_neighbor,
                 "topology", "medium", "Trunk without discovered neighbor",
                 requires_topology=True, default_confidence="medium",
                 source="topology"),
    )
}


# --------------------------------------------------------------- the engine


class RuleEngine:
    """Applies enabled YAML rules to inventory + topology, emitting findings."""

    def __init__(self, rules_config: Mapping[str, Any]):
        self.rules_config = dict(rules_config or {})

    def evaluate(
        self,
        inventory: NetworkInventory,
        topology: Optional[NetworkTopology] = None,
    ) -> tuple[list[Finding], dict[str, Any]]:
        """Return ``(findings, rule_summary)`` for one snapshot."""
        global_enabled = bool(
            self.rules_config.get("global", {}).get("enabled", True)
        )
        rule_cfgs = dict(self.rules_config.get("rules") or {})
        suppression = dict(self.rules_config.get("suppression") or {})

        enabled: list[str] = []
        disabled: list[str] = []
        evaluated: list[str] = []
        skipped: list[str] = []
        findings: list[Finding] = []
        suppressed_count = 0

        for rule_id in sorted(RULES):
            spec = RULES[rule_id]
            rcfg = dict(rule_cfgs.get(rule_id) or {})
            active = (
                global_enabled and rule_id in rule_cfgs
                and rcfg.get("enabled", True)
            )
            if not active:
                disabled.append(rule_id)
                continue
            enabled.append(rule_id)
            if spec.requires_topology and topology is None:
                skipped.append(rule_id)
                logger.warning(
                    "Rule %s requires topology but none is available; skipped.",
                    rule_id,
                )
                continue
            evaluated.append(rule_id)
            for draft, finding in self._build(spec, rcfg, inventory, topology,
                                              suppression):
                if finding.status == "suppressed":
                    suppressed_count += 1
                findings.append(finding)

        findings.sort(key=lambda f: (f.severity_rank, f.rule_id,
                                     f.device or "", f.interface or "",
                                     f.vlan or ""))
        summary = self._summary(inventory.snapshot_id, findings, evaluated,
                                enabled, disabled, skipped, suppressed_count)
        logger.info(
            "Rule engine '%s': %d finding(s) from %d evaluated rule(s) "
            "(%d suppressed).",
            inventory.snapshot_id, summary["total_findings"], len(evaluated),
            suppressed_count,
        )
        return findings, summary

    def _build(self, spec, rcfg, inventory, topology, suppression):
        ctx = RuleContext(inventory=inventory, topology=topology, config=rcfg)
        severity = str(rcfg.get("severity", spec.default_severity))
        category = str(rcfg.get("category", spec.category))
        confidence = str(rcfg.get("confidence", spec.default_confidence))
        tags = tuple(str(t) for t in (rcfg.get("tags") or ()))
        for draft in spec.func(ctx):
            device = draft.get("device")
            interface = draft.get("interface")
            vlan = draft.get("vlan")
            status = "suppressed" if _is_suppressed(
                suppression, spec.rule_id, device, interface, tags
            ) else "open"
            finding = Finding(
                finding_id=make_finding_id(spec.rule_id, device, interface, vlan),
                rule_id=spec.rule_id,
                title=str(draft.get("title") or spec.title),
                severity=severity, category=category, device=device,
                interface=interface, vlan=vlan, status=status,
                evidence=draft.get("evidence"),
                recommendation=draft.get("recommendation"),
                confidence=confidence, source=spec.source, tags=tags,
                details=dict(draft.get("details") or {}),
            )
            yield draft, finding

    @staticmethod
    def _summary(snapshot_id, findings, evaluated, enabled, disabled, skipped,
                 suppressed_count) -> dict[str, Any]:
        open_findings = [f for f in findings if f.status == "open"]
        by_severity = Counter(f.severity for f in open_findings)
        by_category = Counter(f.category for f in open_findings)
        return {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_findings": len(open_findings),
            "findings_by_severity": {
                sev: by_severity[sev]
                for sev in SEVERITY_ORDER if by_severity[sev]
            },
            "findings_by_category": dict(
                sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "rules_evaluated": evaluated,
            "rules_enabled": enabled,
            "rules_disabled": disabled,
            "rules_skipped": skipped,
            "suppressed_count": suppressed_count,
        }


def _is_suppressed(
    suppression: Mapping[str, Any],
    rule_id: str,
    device: Optional[str],
    interface: Optional[str],
    tags: tuple[str, ...],
) -> bool:
    """Match a finding against the configured suppression items."""
    if not suppression.get("enabled", True):
        return False
    for item in suppression.get("items") or []:
        if "rule_id" in item and item["rule_id"] != rule_id:
            continue
        if "device" in item and item["device"] != device:
            continue
        if "interface" in item and item["interface"] != interface:
            continue
        if "tag" in item and item["tag"] not in tags:
            continue
        return True
    return False


def load_rules_config(path: str | Path) -> dict[str, Any]:
    """Load a standalone rules YAML file (``configs/network_rules.yaml``)."""
    import yaml

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Rules config not found: {resolved}")
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def run_rules(
    inventory: NetworkInventory,
    topology: Optional[NetworkTopology],
    rules_config: Mapping[str, Any],
) -> tuple[list[Finding], dict[str, Any]]:
    """Convenience wrapper: evaluate a rules config in one call."""
    return RuleEngine(rules_config).evaluate(inventory, topology)

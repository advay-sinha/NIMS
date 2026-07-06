"""Engine C Phase 4 — safe remediation plan generation.

Purpose
-------
Turn Phase 3 :class:`~src.network_config.findings.Finding` objects into a
structured :class:`RemediationPlan`. This module generates **plans only** — it
never connects to a device and never executes a command. Every config-changing
action is dry-run only, requires explicit human confirmation and carries a
rollback plus a verification step; findings without a safe template become
investigation-only or blocked actions.

Design
------
One config-driven template per rule id (:data:`TEMPLATES`) builds the candidate
commands (or read-only investigation steps). The generator attaches safety
checks, deterministic ids and status, and refuses to emit a command-bearing
action that lacks a rollback or verification (it is blocked instead).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from src.network_config.findings import SEVERITY_ORDER, Finding
from src.network_config.safety import (
    DO_NOT_EXECUTE,
    SafetyCheck,
    build_safety_checks,
    validate_command_action,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- models


@dataclass(frozen=True)
class VerificationStep:
    """A read-only check to confirm an action had the intended effect."""

    command: str
    expected_result: str


@dataclass(frozen=True)
class RollbackAction:
    """The commands that undo a remediation action."""

    commands: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class RemediationAction:
    """A single planned (never executed) remediation for one finding."""

    action_id: str
    finding_id: str
    rule_id: str
    title: str
    severity: str
    action_type: str                     # e.g. vlan_trunk_add / port_shutdown
    device: Optional[str] = None
    interface: Optional[str] = None
    vlan: Optional[str] = None
    commands: tuple[str, ...] = ()
    rollback: Optional[RollbackAction] = None
    verification_steps: tuple[VerificationStep, ...] = ()
    safety_checks: tuple[SafetyCheck, ...] = ()
    requires_confirmation: bool = True
    dry_run_only: bool = True
    status: str = "planned"              # planned / skipped / blocked
    reason: Optional[str] = None
    risk_level: str = "medium"
    source: str = "rule_engine"
    tags: tuple[str, ...] = ()

    @property
    def rollback_commands(self) -> tuple[str, ...]:
        """Flat rollback command list (empty when there is no rollback)."""
        return self.rollback.commands if self.rollback else ()

    @property
    def is_command_bearing(self) -> bool:
        return bool(self.commands)


@dataclass(frozen=True)
class RemediationPlan:
    """A full remediation plan for one snapshot (dry-run, do-not-execute)."""

    snapshot_id: str
    generated_at: str
    actions: tuple[RemediationAction, ...] = ()
    dry_run_only: bool = True
    requires_confirmation: bool = True
    do_not_execute: bool = DO_NOT_EXECUTE


# --------------------------------------------------------------- templates
# Each template returns a dict describing the candidate change (command kind)
# or the read-only checks (investigation kind). ``<iface>`` placeholders are
# substituted with the finding's interface by the generator.


@dataclass(frozen=True)
class TemplateSpec:
    """Static metadata + builder for one rule's remediation template."""

    builder: Callable[[Finding], dict[str, Any]]
    kind: str                            # "command" or "investigation"
    default_risk: str
    action_type: str


def _iface(finding: Finding) -> str:
    return finding.interface or ""


def _t_trunk_add(finding: Finding) -> dict[str, Any]:
    vlans = [str(v) for v in finding.details.get("missing_vlans", [])]
    iface = _iface(finding)
    if not vlans:  # nothing concrete to change -> let the engine block it
        return {"commands": [], "rollback": [], "verification": []}
    return {
        "commands": [f"interface {iface}",
                     *[f"switchport trunk allowed vlan add {v}" for v in vlans]],
        "rollback": [f"interface {iface}",
                     *[f"switchport trunk allowed vlan remove {v}"
                       for v in vlans]],
        "verification": [("show interfaces trunk",
                          f"VLAN(s) {vlans} allowed on {iface}")],
    }


def _t_trunk_remove(finding: Finding) -> dict[str, Any]:
    vlans = [str(v) for v in finding.details.get("unauthorized_vlans", [])]
    iface = _iface(finding)
    if not vlans:  # nothing concrete to change -> let the engine block it
        return {"commands": [], "rollback": [], "verification": []}
    return {
        "commands": [f"interface {iface}",
                     *[f"switchport trunk allowed vlan remove {v}"
                       for v in vlans]],
        "rollback": [f"interface {iface}",
                     *[f"switchport trunk allowed vlan add {v}" for v in vlans]],
        "verification": [("show interfaces trunk",
                          f"VLAN(s) {vlans} no longer allowed on {iface}")],
    }


def _t_port_shutdown(finding: Finding) -> dict[str, Any]:
    iface = _iface(finding)
    return {
        "commands": [f"interface {iface}", "shutdown"],
        "rollback": [f"interface {iface}", "no shutdown"],
        "verification": [("show interface status",
                          f"{iface} shows disabled/administratively down")],
    }


def _t_poe_enable(finding: Finding) -> dict[str, Any]:
    iface = _iface(finding)
    return {
        "commands": [f"interface {iface}", "power inline auto"],
        "rollback": [f"interface {iface}", "power inline never"],
        "verification": [("show power inline",
                          f"{iface} PoE admin state is auto")],
    }


def _t_poe_disable(finding: Finding) -> dict[str, Any]:
    iface = _iface(finding)
    return {
        "commands": [f"interface {iface}", "power inline never"],
        "rollback": [f"interface {iface}", "power inline auto"],
        "verification": [("show power inline",
                          f"{iface} PoE admin state is never")],
    }


def _inv_steps(*steps: tuple[str, str]) -> dict[str, Any]:
    return {"investigation": list(steps)}


def _t_inv_stp_block(finding: Finding) -> dict[str, Any]:
    iface = _iface(finding)
    return _inv_steps(
        (f"show spanning-tree interface {iface} detail",
         "confirm the port role/state and expected topology"),
        ("(manual) inspect cabling and patch-panel mapping",
         "identify any unexpected uplink into an access port"),
    )


def _t_inv_mac_multi(finding: Finding) -> dict[str, Any]:
    return _inv_steps(
        ("show mac address-table",
         "locate the interfaces learning the duplicated MAC"),
        ("(manual) check for a loop or unmanaged switch",
         "confirm whether the MAC movement is a loop, flap or duplicate"),
    )


def _t_inv_trunk_no_neighbor(finding: Finding) -> dict[str, Any]:
    iface = _iface(finding)
    return _inv_steps(
        ("show lldp neighbors", "verify LLDP is enabled and a neighbour appears"),
        ("show cdp neighbors", "verify CDP as an alternative discovery source"),
        (f"(manual) inspect cabling for {iface}",
         "confirm the expected connectivity of the trunk"),
    )


def _t_inv_too_many_macs(finding: Finding) -> dict[str, Any]:
    iface = _iface(finding)
    return _inv_steps(
        (f"show mac address-table interface {iface}",
         "enumerate the MAC addresses on the access port"),
        ("(manual) check for an unmanaged switch or hub downstream",
         "confirm whether the extra MACs are expected"),
    )


def _t_inv_access_vlan(finding: Finding) -> dict[str, Any]:
    return _inv_steps(
        ("show interface status", "verify the intended access VLAN for the port"),
        ("(manual) confirm the VLAN assignment policy",
         "decide the correct VLAN before any change"),
    )


def _t_inv_native_vlan(finding: Finding) -> dict[str, Any]:
    return _inv_steps(
        ("show interfaces trunk", "verify the trunk native VLAN"),
        ("(manual) confirm the intended native VLAN",
         "align both trunk ends before any change"),
    )


TEMPLATES: dict[str, TemplateSpec] = {
    "TRUNK_MISSING_REQUIRED_VLAN": TemplateSpec(
        _t_trunk_add, "command", "medium", "vlan_trunk_add"),
    "TRUNK_UNAUTHORIZED_VLAN": TemplateSpec(
        _t_trunk_remove, "command", "high", "vlan_trunk_remove"),
    "UNUSED_PORT_ADMIN_UP": TemplateSpec(
        _t_port_shutdown, "command", "medium", "port_shutdown"),
    "POE_DISABLED_EXPECTED": TemplateSpec(
        _t_poe_enable, "command", "medium", "poe_enable"),
    "POE_ENABLED_UNEXPECTED": TemplateSpec(
        _t_poe_disable, "command", "medium", "poe_disable"),
    "STP_BLOCKING_ACCESS_PORT": TemplateSpec(
        _t_inv_stp_block, "investigation", "medium", "investigation"),
    "MAC_ON_MULTIPLE_INTERFACES": TemplateSpec(
        _t_inv_mac_multi, "investigation", "medium", "investigation"),
    "TRUNK_WITHOUT_NEIGHBOR": TemplateSpec(
        _t_inv_trunk_no_neighbor, "investigation", "low", "investigation"),
    "ACCESS_PORT_TOO_MANY_MACS": TemplateSpec(
        _t_inv_too_many_macs, "investigation", "medium", "investigation"),
    "ACCESS_PORT_DISALLOWED_VLAN": TemplateSpec(
        _t_inv_access_vlan, "investigation", "medium", "investigation"),
    "NATIVE_VLAN_MISMATCH": TemplateSpec(
        _t_inv_native_vlan, "investigation", "low", "investigation"),
}


# -------------------------------------------------------------- generator


class RemediationGenerator:
    """Builds a dry-run remediation plan from findings (no execution)."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config or {})
        self.global_cfg = dict(self.config.get("global") or {})
        self.templates_cfg = dict(self.config.get("templates") or {})

    def generate(
        self, findings: Sequence[Finding], snapshot_id: str
    ) -> tuple[RemediationPlan, dict[str, Any]]:
        """Return ``(plan, summary)`` for the open findings."""
        open_findings = [f for f in findings if f.status == "open"]
        actions = [self._action_for(f) for f in open_findings]
        actions.sort(key=lambda a: (
            SEVERITY_ORDER.get(a.severity, 9), a.rule_id, a.device or "",
            a.interface or "",
        ))
        plan = RemediationPlan(
            snapshot_id=snapshot_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            actions=tuple(actions),
            dry_run_only=bool(self.global_cfg.get("dry_run_only", True)),
            requires_confirmation=bool(
                self.global_cfg.get("require_confirmation", True)
            ),
        )
        summary = _summarise(snapshot_id, len(open_findings), plan)
        logger.info(
            "Remediation plan '%s': %d action(s) (%d command, %d "
            "investigation, %d blocked) — dry-run only, no commands executed.",
            snapshot_id, summary["total_actions"], summary["command_actions"],
            summary["investigation_actions"], summary["blocked_actions"],
        )
        return plan, summary

    def _action_for(self, finding: Finding) -> RemediationAction:
        base = dict(
            action_id=f"ACT-{finding.finding_id}",
            finding_id=finding.finding_id, rule_id=finding.rule_id,
            title=finding.title, severity=finding.severity,
            device=finding.device, interface=finding.interface,
            vlan=finding.vlan, source="rule_engine", tags=finding.tags,
        )
        spec = TEMPLATES.get(finding.rule_id)
        tcfg = self.templates_cfg.get(finding.rule_id)
        if spec is None:
            return RemediationAction(
                **base, action_type="blocked", status="blocked",
                risk_level="low", requires_confirmation=False,
                reason="no remediation template available for this rule",
            )
        if tcfg is not None and not tcfg.get("enabled", True):
            return RemediationAction(
                **base, action_type=spec.action_type, status="skipped",
                risk_level=str((tcfg or {}).get("risk_level", spec.default_risk)),
                requires_confirmation=False,
                reason="remediation template disabled in configuration",
            )
        tcfg = dict(tcfg or {})
        risk = str(tcfg.get("risk_level", spec.default_risk))
        mode = tcfg.get("mode")
        parts = spec.builder(finding)
        # Investigation kind, or a command template forced to investigation.
        if spec.kind == "investigation" or mode == "investigation_only":
            steps = parts.get("investigation") or []
            return RemediationAction(
                **base, action_type="investigation", status="planned",
                risk_level=risk if risk in ("low", "medium", "high") else "low",
                requires_confirmation=False,
                verification_steps=tuple(
                    VerificationStep(cmd, exp) for cmd, exp in steps
                ),
                reason="investigation-only; no configuration change proposed",
            )
        return self._command_action(base, spec, parts, risk)

    def _command_action(self, base, spec, parts, risk) -> RemediationAction:
        commands = list(parts.get("commands") or [])
        rollback = list(parts.get("rollback") or [])
        verification = [VerificationStep(cmd, exp)
                        for cmd, exp in parts.get("verification") or []]
        ok, reason = validate_command_action(
            commands=commands, rollback_commands=rollback,
            verification_steps=verification, global_cfg=self.global_cfg,
        )
        if not ok:
            return RemediationAction(
                **base, action_type=spec.action_type, status="blocked",
                risk_level=risk, requires_confirmation=False, reason=reason,
            )
        safety = build_safety_checks(
            commands=commands, rollback_commands=rollback,
            verification_steps=verification, risk_level=risk,
            global_cfg=self.global_cfg,
        )
        return RemediationAction(
            **base, action_type=spec.action_type, status="planned",
            commands=tuple(commands),
            rollback=RollbackAction(commands=tuple(rollback),
                                    note="undoes the proposed change"),
            verification_steps=tuple(verification), safety_checks=safety,
            requires_confirmation=bool(
                self.global_cfg.get("require_confirmation", True)
            ),
            dry_run_only=bool(self.global_cfg.get("dry_run_only", True)),
            risk_level=risk,
        )


def _summarise(snapshot_id, total_findings, plan) -> dict[str, Any]:
    actions = plan.actions
    by_risk = Counter(a.risk_level for a in actions if a.status == "planned")
    by_status = Counter(a.status for a in actions)
    return {
        "snapshot_id": snapshot_id,
        "timestamp": plan.generated_at,
        "total_findings": total_findings,
        "total_actions": len(actions),
        "command_actions": sum(1 for a in actions if a.is_command_bearing),
        "investigation_actions": sum(
            1 for a in actions if a.action_type == "investigation"
        ),
        "blocked_actions": sum(1 for a in actions if a.status == "blocked"),
        "actions_by_risk": dict(by_risk),
        "actions_by_status": dict(by_status),
        "dry_run_only": plan.dry_run_only,
        "requires_confirmation": plan.requires_confirmation,
        "do_not_execute": plan.do_not_execute,
    }


def load_remediation_config(path: str | Path) -> dict[str, Any]:
    """Load the standalone remediation YAML (``configs/remediation.yaml``)."""
    import yaml

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Remediation config not found: {resolved}")
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def generate_remediation(
    findings: Sequence[Finding], snapshot_id: str, config: Mapping[str, Any]
) -> tuple[RemediationPlan, dict[str, Any]]:
    """Convenience wrapper around :class:`RemediationGenerator`."""
    return RemediationGenerator(config).generate(findings, snapshot_id)

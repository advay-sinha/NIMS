"""Engine C Phase 6 — offline remediation verification.

Purpose
-------
Given a *before* snapshot's ``remediation_plan.json`` and an *after* snapshot,
decide — from saved artefacts only — whether each planned remediation goal now
appears satisfied. This is evidence-based and deliberately conservative: when
the after snapshot lacks the data to judge, the result is ``unknown``, never a
false ``failed``. Nothing here contacts a device or executes a command.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.network_config.diff import (
    SnapshotData,
    finding_key,
    interface_map,
    open_findings,
    poe_map,
    trunk_map,
)

logger = logging.getLogger(__name__)

_POE_ENABLED_TOKENS = {"on", "auto", "static", "enabled", "true"}
# States that show a port was actually shut down. "notconnect" is NOT here:
# it is the unused-admin-up state the finding flags, so it must not count as
# remediated (that would be false confidence — CLAUDE.md/Phase 6: unknown or
# failed beats a false pass).
_PORT_DOWN_TOKENS = {"disabled", "administratively down", "admin down",
                     "shutdown"}


@dataclass(frozen=True)
class VerificationResult:
    """Whether one planned remediation goal appears met in the after snapshot."""

    verification_id: str
    action_id: Optional[str]
    finding_id: Optional[str]
    rule_id: Optional[str]
    device: Optional[str]
    interface: Optional[str]
    expected_outcome: str
    observed_outcome: str
    status: str              # passed / failed / unknown / not_applicable
    evidence: Optional[str] = None
    recommendation: Optional[str] = None


def _ver_id(action_id: str) -> str:
    return f"VER-{hashlib.sha1(str(action_id).encode()).hexdigest()[:8]}"


class RemediationVerifier:
    """Verify a before-plan's actions against an after snapshot (offline)."""

    def __init__(self, before: SnapshotData, after: SnapshotData):
        self.before = before
        self.after = after
        self.after_interfaces = interface_map(after.inventory)
        self.after_trunks = trunk_map(after.inventory)
        self.after_poe = poe_map(after.inventory)
        self.after_findings_available = after.findings is not None
        self.after_open_keys = set(open_findings(after.findings))
        self.before_findings_by_id = {
            f.get("finding_id"): f for f in (before.findings or [])
        }

    def verify(self) -> list[VerificationResult]:
        plan = self.before.remediation
        if not plan:
            return []
        results = []
        for action in plan.get("actions") or []:
            if str(action.get("status")) != "planned":
                continue        # blocked/skipped actions carry no goal to verify
            results.append(self._verify(action))
        return results

    # -- dispatch -------------------------------------------------------------

    def _verify(self, action: dict[str, Any]) -> VerificationResult:
        rule_id = str(action.get("rule_id", ""))
        handler = {
            "UNUSED_PORT_ADMIN_UP": self._verify_unused_port,
            "TRUNK_MISSING_REQUIRED_VLAN": self._verify_trunk_add,
            "TRUNK_UNAUTHORIZED_VLAN": self._verify_trunk_remove,
            "POE_DISABLED_EXPECTED": self._verify_poe_enable,
        }.get(rule_id)
        if action.get("action_type") == "investigation":
            return self._verify_investigation(action)
        if handler is None:
            return self._result(action, "goal for this rule cannot be verified "
                                 "from saved artefacts", "no verifier available",
                                 "unknown", recommendation="Verify manually.")
        return handler(action)

    # -- per-rule verifiers ---------------------------------------------------

    def _verify_unused_port(self, action):
        device, interface = action.get("device"), action.get("interface")
        expected = "port no longer flagged unused/admin-up (shut down)"
        if self._finding_resolved(action):
            return self._result(action, expected,
                                "finding no longer present in after snapshot",
                                "passed", evidence="unused-port finding resolved")
        iface = self.after_interfaces.get((str(device), str(interface)))
        if iface is not None:
            status = str(iface.get("status", "")).lower()
            if any(tok in status for tok in _PORT_DOWN_TOKENS):
                return self._result(action, expected,
                                    f"interface status is '{status}'", "passed",
                                    evidence="interface now down/disabled")
            if self.after_findings_available:
                return self._result(action, expected,
                                    f"interface still '{status}', finding persists",
                                    "failed",
                                    recommendation="Apply the planned shutdown.")
        return self._result(action, expected,
                            "insufficient after-state data for this interface",
                            "unknown", recommendation="Re-capture the port state.")

    def _verify_trunk_add(self, action):
        required = self._vlans_from(action, "missing_vlans", "add")
        expected = f"required VLAN(s) {sorted(required)} present on trunk allowed list"
        trunk = self.after_trunks.get((str(action.get("device")),
                                       str(action.get("interface"))))
        if trunk is None or not required:
            return self._result(action, expected,
                                "no after-state trunk data to compare", "unknown",
                                recommendation="Re-capture 'show interfaces trunk'.")
        allowed = {str(v) for v in (trunk.get("allowed_vlans") or [])}
        missing = sorted(required - allowed)
        if not missing:
            return self._result(action, expected,
                                f"allowed VLANs now include {sorted(required)}",
                                "passed", evidence="required VLAN(s) present")
        return self._result(action, expected,
                            f"still missing VLAN(s) {missing}", "failed",
                            recommendation="Add the missing VLAN(s) to the trunk.")

    def _verify_trunk_remove(self, action):
        unauthorized = self._vlans_from(action, "unauthorized_vlans", "remove")
        expected = f"unauthorized VLAN(s) {sorted(unauthorized)} removed from trunk"
        trunk = self.after_trunks.get((str(action.get("device")),
                                       str(action.get("interface"))))
        if trunk is None or not unauthorized:
            return self._result(action, expected,
                                "no after-state trunk data to compare", "unknown",
                                recommendation="Re-capture 'show interfaces trunk'.")
        allowed = {str(v) for v in (trunk.get("allowed_vlans") or [])}
        still = sorted(unauthorized & allowed)
        if not still:
            return self._result(action, expected,
                                "unauthorized VLAN(s) no longer allowed", "passed",
                                evidence="unauthorized VLAN(s) removed")
        return self._result(action, expected,
                            f"VLAN(s) {still} still allowed", "failed",
                            recommendation="Remove the unauthorized VLAN(s).")

    def _verify_poe_enable(self, action):
        device, interface = str(action.get("device")), str(action.get("interface"))
        expected = "PoE administratively enabled (auto/static/on)"
        poe = self.after_poe.get((device, interface))
        iface = self.after_interfaces.get((device, interface))
        observed_tokens = []
        if poe is not None:
            observed_tokens += [str(poe.get("admin_state", "")).lower(),
                                str(poe.get("oper_state", "")).lower()]
        if iface is not None:
            observed_tokens.append(str(iface.get("poe_state", "")).lower())
            if iface.get("poe_enabled") is True:
                observed_tokens.append("enabled")
        observed_tokens = [t for t in observed_tokens if t]
        if not observed_tokens:
            return self._result(action, expected,
                                "no after-state PoE data to compare", "unknown",
                                recommendation="Re-capture 'show power inline'.")
        if any(tok in _POE_ENABLED_TOKENS for tok in observed_tokens):
            return self._result(action, expected,
                                f"PoE state is {observed_tokens}", "passed",
                                evidence="PoE now enabled")
        return self._result(action, expected,
                            f"PoE state is {observed_tokens}", "failed",
                            recommendation="Enable PoE on the port.")

    def _verify_investigation(self, action):
        expected = "related finding cleared after investigation"
        if self.after_findings_available and self._finding_resolved(action):
            return self._result(action, expected,
                                "related finding no longer present", "passed",
                                evidence="investigation finding resolved")
        return self._result(action, expected,
                            "investigation-only; no config change was planned",
                            "not_applicable",
                            recommendation="Review investigation steps manually.")

    # -- helpers --------------------------------------------------------------

    def _finding_resolved(self, action) -> bool:
        """True when the action's originating finding is gone in the after set."""
        if not self.after_findings_available:
            return False
        finding = self.before_findings_by_id.get(action.get("finding_id"))
        if finding is None:
            return False
        return finding_key(finding) not in self.after_open_keys

    def _vlans_from(self, action, detail_key, verb) -> set[str]:
        """VLANs to check: from the before finding's details, else the commands."""
        finding = self.before_findings_by_id.get(action.get("finding_id"))
        if finding:
            vlans = (finding.get("details") or {}).get(detail_key)
            if vlans:
                return {str(v) for v in vlans}
        found = set()
        for command in action.get("commands") or []:
            tokens = str(command).split()
            if verb in tokens and tokens and tokens[-1].isdigit():
                found.add(tokens[-1])
        return found

    def _result(self, action, expected, observed, status, evidence=None,
                recommendation=None) -> VerificationResult:
        return VerificationResult(
            verification_id=_ver_id(action.get("action_id", "")),
            action_id=action.get("action_id"),
            finding_id=action.get("finding_id"),
            rule_id=action.get("rule_id"),
            device=action.get("device"), interface=action.get("interface"),
            expected_outcome=expected, observed_outcome=observed, status=status,
            evidence=evidence, recommendation=recommendation)


def verify_remediation(
    before: SnapshotData, after: SnapshotData
) -> list[VerificationResult]:
    """Convenience wrapper around :class:`RemediationVerifier`."""
    return RemediationVerifier(before, after).verify()

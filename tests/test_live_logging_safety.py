"""Tests for adapter safety guards + no-mutation source assertions."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.live_logging.adapters import safety
from src.live_logging.adapters.errors import SafetyViolation

ADAPTER_DIR = Path("src/live_logging/adapters")


@pytest.mark.parametrize("flag", list(safety.FORBIDDEN_FLAGS))
def test_forbidden_flags_rejected(flag):
    with pytest.raises(SafetyViolation):
        safety.assert_read_only({flag: True})


def test_read_only_required_for_live():
    with pytest.raises(SafetyViolation):
        safety.assert_read_only({"read_only": False}, require_read_only=True)
    safety.assert_read_only({"read_only": True}, require_read_only=True)  # ok


@pytest.mark.parametrize(
    "command",
    ["configure terminal", "write memory", "no shutdown",
     "copy running-config startup-config", "snmpset x", "vlan 20"],
)
def test_mutating_commands_rejected(command):
    with pytest.raises(SafetyViolation):
        safety.assert_command_allowed(command, ["show running-config", command])


def test_non_allowlisted_command_rejected():
    with pytest.raises(SafetyViolation):
        safety.assert_command_allowed("show version", ["show running-config"])


def test_allowlisted_show_command_ok():
    safety.assert_command_allowed("show running-config", ["show running-config"])


def test_no_mutation_apis_in_adapter_sources():
    """No live adapter may reference SNMP SET or config-write transports."""
    banned = ("setCmd", "snmpset", "SnmpSet")
    for path in ADAPTER_DIR.glob("*.py"):
        if path.name == "safety.py":
            continue  # safety.py legitimately names forbidden tokens as a denylist
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token} found in {path.name}"


def test_config_adapter_uses_reject_host_key_policy():
    text = (ADAPTER_DIR / "hirschmann_config.py").read_text(encoding="utf-8")
    assert "RejectPolicy" in text  # host-key verification, never AutoAdd
    assert "AutoAddPolicy" not in text

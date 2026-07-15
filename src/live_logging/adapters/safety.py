"""Static + runtime safety guards for live adapters.

Enforces the read-only boundary (spec section 13). Any configuration that would
permit SNMP SET, configuration mode, write commands, remediation, or arbitrary
targets/OIDs/commands is rejected before an adapter can run. A fixed denylist of
device-mutating command tokens is used to validate any allowlisted retrieval
command.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.live_logging.adapters.errors import SafetyViolation

# Config flags that must never be true (rejecting configuration when set).
FORBIDDEN_FLAGS: tuple[str, ...] = (
    "allow_set",
    "allow_config_mode",
    "allow_write_commands",
    "allow_remediation",
    "allow_arbitrary_targets",
    "allow_arbitrary_oids",
    "allow_arbitrary_commands",
)

# Device-mutating / dangerous command tokens forbidden in any allowlisted command.
FORBIDDEN_COMMAND_TOKENS: tuple[str, ...] = (
    "snmpset",
    "configure terminal",
    "config term",
    "conf t",
    "write memory",
    "write mem",
    "copy running-config startup-config",
    "copy run start",
    "shutdown",
    "no shutdown",
    "vlan ",
    "switchport",
    "poe ",
    "power inline",
    "spanning-tree",
    "reload",
    "erase",
    "delete",
    "format",
    "rm ",
    "reboot",
)


def assert_read_only(cfg: Mapping[str, Any], *, require_read_only: bool = False) -> None:
    """Raise :class:`SafetyViolation` if the config breaches the boundary.

    Parameters
    ----------
    cfg:
        The source configuration block.
    require_read_only:
        When True (live mode), also require ``read_only`` to be explicitly true.
    """
    for flag in FORBIDDEN_FLAGS:
        if bool(cfg.get(flag, False)):
            raise SafetyViolation(f"forbidden flag '{flag}' is enabled")
    if require_read_only and cfg.get("read_only", True) is not True:
        raise SafetyViolation("read_only must be true for live mode")


def assert_command_allowed(command: str, allowlist: Iterable[str]) -> None:
    """Validate a retrieval command against the denylist and an allowlist.

    Raises
    ------
    SafetyViolation
        If the command contains a mutating token or is not in ``allowlist``.
    """
    lowered = command.strip().lower()
    for token in FORBIDDEN_COMMAND_TOKENS:
        if token in lowered:
            raise SafetyViolation(f"command contains forbidden token '{token.strip()}'")
    allowed = {c.strip().lower() for c in allowlist}
    if lowered not in allowed:
        raise SafetyViolation(f"command '{command}' is not in the approved allowlist")


def assert_commands_allowlist(allowlist: Iterable[str]) -> None:
    """Validate that every configured allowlisted command is itself safe."""
    for command in allowlist:
        lowered = command.strip().lower()
        for token in FORBIDDEN_COMMAND_TOKENS:
            if token in lowered:
                raise SafetyViolation(
                    f"configured command '{command}' contains forbidden token "
                    f"'{token.strip()}'"
                )

"""Engine C Phase 4 — remediation safety primitives.

Purpose
-------
Encode the non-negotiable safety posture of remediation *planning*: plans are
dry-run only, require explicit human confirmation, and every config-changing
action must carry a rollback and a verification step. This module builds the
per-action :class:`SafetyCheck` list and validates that a command-bearing
action is safe to *plan* (never to execute — nothing here runs a command).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Hard invariant surfaced in plan metadata and every artefact.
DO_NOT_EXECUTE = True

VALID_RISK_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class SafetyCheck:
    """One named safety precondition and whether the plan satisfies it."""

    name: str
    satisfied: bool
    detail: str


def build_safety_checks(
    *,
    commands: Sequence[str],
    rollback_commands: Sequence[str],
    verification_steps: Sequence[Any],
    risk_level: str,
    global_cfg: Mapping[str, Any],
) -> tuple[SafetyCheck, ...]:
    """Build the safety-check list for a command-bearing action."""
    return (
        SafetyCheck(
            "requires_confirmation",
            bool(global_cfg.get("require_confirmation", True)),
            "human confirmation is required before any application",
        ),
        SafetyCheck(
            "dry_run_only",
            bool(global_cfg.get("dry_run_only", True)),
            "plan is dry-run only; no command is executed",
        ),
        SafetyCheck(
            "rollback_present",
            len(rollback_commands) > 0,
            "a rollback command set exists for every change",
        ),
        SafetyCheck(
            "verification_present",
            len(verification_steps) > 0,
            "at least one verification step is defined",
        ),
        SafetyCheck(
            "risk_level_assigned",
            risk_level in VALID_RISK_LEVELS,
            f"risk level is one of {VALID_RISK_LEVELS}",
        ),
    )


def validate_command_action(
    *,
    commands: Sequence[str],
    rollback_commands: Sequence[str],
    verification_steps: Sequence[Any],
    global_cfg: Mapping[str, Any],
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for a would-be command-bearing action.

    A command action is only safe to *plan* when it has commands, a rollback
    (if required) and a verification step (if required). Otherwise it must be
    blocked rather than emitted as an actionable change.
    """
    if not commands:
        return False, "no candidate commands could be generated"
    if global_cfg.get("require_rollback", True) and not rollback_commands:
        return False, "rollback commands are required but none were generated"
    if global_cfg.get("require_verification", True) and not verification_steps:
        return False, "verification steps are required but none were generated"
    return True, ""

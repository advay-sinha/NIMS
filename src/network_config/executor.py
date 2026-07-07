"""Engine C Phase 5 — dry-run action executor (validation only).

Purpose
-------
Read a Phase 4 ``remediation_plan.json`` and produce a **dry-run** execution
result: every planned action is safety-validated and recorded as *what would be
done*, but nothing is ever executed. This module never contacts a device, never
opens a network connection and never imports a device-control or SSH library.
Every record has ``executed = False`` and ``would_execute = False``.

Design
------
:class:`DryRunExecutor` walks the plan's actions, applies the Phase 5 safety
rules (:func:`_validate`) and emits one :class:`ActionExecutionRecord` per
action. Upstream ``blocked``/``skipped`` states are preserved; command-bearing
actions missing a rollback, verification, confirmation or the dry-run flag are
demoted to ``blocked``. The result is a pure data object — persistence and
audit logging live in sibling modules.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.network_config.safety import DO_NOT_EXECUTE

logger = logging.getLogger(__name__)

# Hard, non-negotiable invariants for this phase. Nothing here ever flips them.
EXECUTION_MODE = "dry_run"
EXECUTED = False
WOULD_EXECUTE = False

_VALID_STATUSES = ("validated", "skipped", "blocked", "failed")


# ------------------------------------------------------------------- models


@dataclass(frozen=True)
class ActionExecutionRecord:
    """The dry-run outcome for one planned remediation action."""

    execution_id: str
    action_id: str
    finding_id: Optional[str]
    rule_id: Optional[str]
    device: Optional[str]
    interface: Optional[str]
    action_type: str
    requested_commands: tuple[str, ...] = ()
    rollback_commands: tuple[str, ...] = ()
    verification_steps: tuple[dict[str, str], ...] = ()
    safety_checks: tuple[dict[str, Any], ...] = ()
    status: str = "validated"
    reason: Optional[str] = None
    risk_level: str = "medium"
    requires_confirmation: bool = True
    dry_run_only: bool = True
    timestamp: str = ""
    execution_mode: str = EXECUTION_MODE
    executed: bool = EXECUTED
    would_execute: bool = WOULD_EXECUTE


@dataclass(frozen=True)
class DryRunExecutionResult:
    """The full dry-run execution over one remediation plan (do-not-execute)."""

    snapshot_id: str
    generated_at: str
    operator: str
    records: tuple[ActionExecutionRecord, ...] = ()
    source: str = "dry_run_executor"
    execution_mode: str = EXECUTION_MODE
    executed: bool = EXECUTED
    would_execute: bool = WOULD_EXECUTE
    do_not_execute: bool = DO_NOT_EXECUTE


@dataclass(frozen=True)
class ExecutionSummary:
    """Roll-up of a dry-run execution (surfaced in artefacts and the report)."""

    snapshot_id: str
    timestamp: str
    operator: str
    total_actions: int
    validated_actions: int
    blocked_actions: int
    skipped_actions: int
    failed_actions: int
    execution_mode: str = EXECUTION_MODE
    executed: bool = EXECUTED
    would_execute: bool = WOULD_EXECUTE
    commands_executed: int = 0
    do_not_execute: bool = DO_NOT_EXECUTE


# ------------------------------------------------------------- validation


def _rollback_commands(action: Mapping[str, Any]) -> list[str]:
    """Flatten the plan action's rollback commands (dict, list or absent)."""
    rollback = action.get("rollback")
    if isinstance(rollback, Mapping):
        return [str(c) for c in (rollback.get("commands") or [])]
    if isinstance(rollback, (list, tuple)):
        return [str(c) for c in rollback]
    return []


def _validate(action: Mapping[str, Any], safety: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(status, reason)`` for one plan action under dry-run rules.

    A command-bearing planned action is only ``validated`` when it is dry-run
    only, requires confirmation and carries both a rollback and a verification
    step; otherwise it is ``blocked``. Investigation actions must not carry
    config-changing commands. Upstream ``blocked``/``skipped`` states are kept.
    """
    plan_status = str(action.get("status", ""))
    if plan_status == "blocked":
        return "blocked", action.get("reason") or "blocked in remediation plan"
    if plan_status == "skipped":
        return "skipped", action.get("reason") or "skipped in remediation plan"
    if plan_status != "planned":
        return "blocked", f"unexpected plan status '{plan_status}'"

    commands = [str(c) for c in (action.get("commands") or [])]
    action_type = str(action.get("action_type", ""))
    dry_run_only = bool(action.get("dry_run_only", False))
    requires_confirmation = bool(action.get("requires_confirmation", False))

    if safety.get("block_if_not_dry_run", True) and not dry_run_only:
        return "blocked", "action is not marked dry_run_only"

    if action_type == "investigation":
        if (safety.get("block_config_commands_in_investigation_actions", True)
                and commands):
            return "blocked", "investigation action contains config commands"
        return "validated", "investigation-only; read-only steps validated"

    if commands:
        if not requires_confirmation:
            return "blocked", "command action does not require confirmation"
        if safety.get("block_if_missing_rollback", True) and not _rollback_commands(
            action
        ):
            return "blocked", "command action is missing rollback commands"
        if (safety.get("block_if_missing_verification", True)
                and not (action.get("verification_steps") or [])):
            return "blocked", "command action is missing verification steps"
        return "validated", "command action validated for dry-run (not executed)"

    return "validated", "no-op planned action validated"


def _execution_id(snapshot_id: str, action_id: str) -> str:
    """Deterministic per-action execution id (stable across re-runs)."""
    digest = hashlib.sha1(f"{snapshot_id}|{action_id}".encode()).hexdigest()
    return f"EXE-{digest[:8]}"


# -------------------------------------------------------------- executor


class DryRunExecutor:
    """Validate a remediation plan in dry-run mode (never executes)."""

    def __init__(self, config: Mapping[str, Any], operator: str | None = None):
        self.config = dict(config or {})
        self.global_cfg = dict(self.config.get("global") or {})
        self.safety_cfg = dict(self.config.get("safety") or {})
        self.operator = operator or str(
            self.global_cfg.get("default_operator", "offline_dry_run")
        )

    def execute(
        self, plan_payload: Mapping[str, Any], snapshot_id: str | None = None
    ) -> DryRunExecutionResult:
        """Return a :class:`DryRunExecutionResult` for the plan's actions."""
        snapshot_id = snapshot_id or str(plan_payload.get("snapshot_id", "snapshot"))
        timestamp = datetime.now(timezone.utc).isoformat()
        actions = list(plan_payload.get("actions") or [])
        records = tuple(
            self._record(action, snapshot_id, timestamp) for action in actions
        )
        result = DryRunExecutionResult(
            snapshot_id=snapshot_id,
            generated_at=timestamp,
            operator=self.operator,
            records=records,
        )
        by_status = Counter(r.status for r in records)
        logger.info(
            "Dry-run execution '%s' by %s: %d action(s) "
            "(%d validated, %d blocked, %d skipped) — no commands executed.",
            snapshot_id, self.operator, len(records),
            by_status.get("validated", 0), by_status.get("blocked", 0),
            by_status.get("skipped", 0),
        )
        return result

    def _record(
        self, action: Mapping[str, Any], snapshot_id: str, timestamp: str
    ) -> ActionExecutionRecord:
        action_id = str(action.get("action_id", ""))
        status, reason = _validate(action, self.safety_cfg)
        steps = tuple(
            {"command": str(s.get("command", "")),
             "expected_result": str(s.get("expected_result", ""))}
            for s in (action.get("verification_steps") or [])
        )
        return ActionExecutionRecord(
            execution_id=_execution_id(snapshot_id, action_id),
            action_id=action_id,
            finding_id=action.get("finding_id"),
            rule_id=action.get("rule_id"),
            device=action.get("device"),
            interface=action.get("interface"),
            action_type=str(action.get("action_type", "")),
            requested_commands=tuple(str(c) for c in (action.get("commands") or [])),
            rollback_commands=tuple(_rollback_commands(action)),
            verification_steps=steps,
            safety_checks=tuple(action.get("safety_checks") or []),
            status=status,
            reason=reason,
            risk_level=str(action.get("risk_level", "medium")),
            requires_confirmation=bool(action.get("requires_confirmation", False)),
            dry_run_only=bool(action.get("dry_run_only", False)),
            timestamp=timestamp,
        )


def summarise_execution(result: DryRunExecutionResult) -> ExecutionSummary:
    """Roll a :class:`DryRunExecutionResult` up into an :class:`ExecutionSummary`."""
    by_status = Counter(r.status for r in result.records)
    return ExecutionSummary(
        snapshot_id=result.snapshot_id,
        timestamp=result.generated_at,
        operator=result.operator,
        total_actions=len(result.records),
        validated_actions=by_status.get("validated", 0),
        blocked_actions=by_status.get("blocked", 0),
        skipped_actions=by_status.get("skipped", 0),
        failed_actions=by_status.get("failed", 0),
    )


# ------------------------------------------------------------------- io


def load_executor_config(path: str | Path) -> dict[str, Any]:
    """Load the dry-run executor YAML (``configs/network_action_executor.yaml``)."""
    import yaml

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Executor config not found: {resolved}")
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def load_remediation_plan(path: str | Path) -> dict[str, Any]:
    """Load a Phase 4 ``remediation_plan.json`` payload for dry-run validation."""
    import json

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Remediation plan not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid remediation plan JSON: {resolved} ({exc})") from exc
    if not isinstance(payload, dict) or "actions" not in payload:
        raise ValueError(f"Remediation plan is missing 'actions': {resolved}")
    return payload

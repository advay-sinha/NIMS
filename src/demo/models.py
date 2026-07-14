"""Typed models for the offline demo orchestrator.

Pure value objects: a resolved demo configuration, an approved local command,
one planned stage and its execution result. No IO, no execution — building these
runs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# --- stage statuses ---------------------------------------------------------
PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"
SKIPPED = "skipped"
REUSED_EXISTING = "reused_existing"

TERMINAL_OK = (SUCCESS, SKIPPED, REUSED_EXISTING)

# --- stage kinds ------------------------------------------------------------
KIND_COMMAND = "command"     # runs one or more allowlisted subprocess commands
KIND_INTERNAL = "internal"   # runs a pure-Python readiness/reporting function


@dataclass(frozen=True)
class DemoCommand:
    """One approved local Python module invocation (never a raw shell string)."""

    module: str
    args: tuple[str, ...] = ()
    note: str = ""

    @property
    def argv(self) -> list[str]:
        """The argument array executed (``python -m <module> <args...>``)."""
        return ["python", "-m", self.module, *self.args]

    @property
    def display(self) -> str:
        """The exact command a user would type (for dry-run / reports)."""
        return " ".join(self.argv)

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "args": list(self.args),
                "note": self.note, "display": self.display}


@dataclass
class StageResult:
    """The outcome of one demo stage."""

    name: str
    title: str
    status: str = PENDING
    kind: str = KIND_COMMAND
    required: bool = True
    reused: bool = False
    commands: list[DemoCommand] = field(default_factory=list)
    exit_codes: list[int] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in TERMINAL_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "title": self.title, "status": self.status,
            "kind": self.kind, "required": self.required, "reused": self.reused,
            "commands": [c.to_dict() for c in self.commands],
            "exit_codes": self.exit_codes, "artifacts": self.artifacts,
            "warnings": self.warnings, "failure_reason": self.failure_reason,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "details": self.details,
        }


@dataclass
class DemoConfig:
    """Resolved demo configuration (CLI overrides merged over configs/demo.yaml)."""

    engine_c_input_dir: str
    engine_c_snapshot: str
    engine_a_datasets: tuple[str, ...]
    engine_a_model: str
    engine_b_dataset: str
    engine_b_model: str
    syslog_run: str
    correlation_id: str
    refresh_engine_c: bool = True
    conditional_training: bool = True
    skip_training: bool = False
    force_train_engine_a: bool = False
    force_train_engine_b: bool = False
    require_syslog: bool = False
    reuse_assessment: bool = False
    continue_on_error: bool = False
    launch_dashboard: bool = False
    dry_run: bool = False
    syslog_fallback_fixture: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_c_input_dir": self.engine_c_input_dir,
            "engine_c_snapshot": self.engine_c_snapshot,
            "engine_a_datasets": list(self.engine_a_datasets),
            "engine_a_model": self.engine_a_model,
            "engine_b_dataset": self.engine_b_dataset,
            "engine_b_model": self.engine_b_model,
            "syslog_run": self.syslog_run,
            "correlation_id": self.correlation_id,
            "refresh_engine_c": self.refresh_engine_c,
            "conditional_training": self.conditional_training,
            "skip_training": self.skip_training,
            "force_train_engine_a": self.force_train_engine_a,
            "force_train_engine_b": self.force_train_engine_b,
            "require_syslog": self.require_syslog,
            "reuse_assessment": self.reuse_assessment,
            "continue_on_error": self.continue_on_error,
            "launch_dashboard": self.launch_dashboard,
            "dry_run": self.dry_run,
            "syslog_fallback_fixture": self.syslog_fallback_fixture,
        }

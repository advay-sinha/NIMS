"""Executor for the offline demo plan (allowlisted commands only).

The runner may execute **only** the approved local Python module entry points in
:data:`ALLOWLIST`, always as an argument array (never ``shell=True``). Anything
else raises. After each stage it re-checks the relevant artefacts to decide the
final stage status (success / reused / skipped / failed). In dry-run mode it
executes nothing — it only records the exact commands.

Nothing here contacts a device, opens a socket, captures packets, executes
remediation or deletes a source artefact.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.demo import readiness
from src.demo.models import (
    FAILED,
    KIND_INTERNAL,
    REUSED_EXISTING,
    RUNNING,
    SKIPPED,
    SUCCESS,
    DemoCommand,
    DemoConfig,
    StageResult,
)

logger = logging.getLogger(__name__)

# The ONLY modules the orchestrator is permitted to execute.
ALLOWLIST: frozenset[str] = frozenset({
    "scripts.validate_datasets", "scripts.run_audit", "scripts.run_preprocessing",
    "scripts.run_feature_engineering", "scripts.train_model",
    "scripts.generate_validation_report", "scripts.build_experiment_index",
    "scripts.build_model_registry", "scripts.promote_model",
    "scripts.prepare_network_health_dataset", "scripts.validate_network_health",
    "scripts.run_network_health_preprocessing", "scripts.train_network_health_model",
    "scripts.analyze_network_config", "scripts.dry_run_network_actions",
    "scripts.generate_network_config_report",
    "scripts.export_network_config_dashboard", "scripts.ingest_switch_syslog",
    "scripts.run_correlation", "scripts.run_streaming_demo",
    "scripts.validate_engine_c_safety", "scripts.run_dashboard",
})

_COMMAND_TIMEOUT_SECONDS = 3600


class DisallowedCommandError(RuntimeError):
    """Raised when a command outside the allowlist is attempted."""


def assert_allowed(command: DemoCommand) -> None:
    """Reject any command whose module is not on the allowlist."""
    if command.module not in ALLOWLIST:
        raise DisallowedCommandError(
            f"Command '{command.module}' is not on the approved allowlist.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DemoRunner:
    """Executes a demo plan stage by stage with per-stage artefact rechecks."""

    def __init__(self, config: DemoConfig, paths, project_root) -> None:
        self.config = config
        self.paths = paths
        self.project_root = project_root

    # ------------------------------------------------------------ execution
    def run(self, stages: list[StageResult]) -> dict[str, Any]:
        """Run every stage in order; return a readiness roll-up.

        Required-stage failure halts the run (unless ``--continue-on-error`` and
        the stage is optional). Dry-run executes nothing.
        """
        dashboard_result: dict[str, Any] = {}
        halted = False
        for stage in stages:
            if halted:
                stage.status = SKIPPED
                stage.warnings.append("Skipped: a prior required stage failed.")
                continue
            self._run_stage(stage)
            if stage.name == "frontend_readiness":
                dashboard_result = stage.details.get("dashboard", {})
            if not stage.ok and stage.required:
                if not (self.config.continue_on_error and not stage.required):
                    halted = True

        return {
            "all_required_ok": all(s.ok for s in stages if s.required),
            "dashboard": dashboard_result,
            "stages": stages,
        }

    def _run_stage(self, stage: StageResult) -> None:
        stage.started_at = _now()
        start = time.perf_counter()
        try:
            if stage.status == REUSED_EXISTING:
                pass  # decided at plan time; still recheck below for safety
            elif stage.kind == KIND_INTERNAL:
                self._run_internal(stage)
            else:
                self._run_commands(stage)
            if stage.status != FAILED:
                self._recheck(stage)
        except DisallowedCommandError as exc:
            stage.status = FAILED
            stage.failure_reason = str(exc)
        finally:
            stage.ended_at = _now()
            stage.elapsed_seconds = time.perf_counter() - start

    def _run_commands(self, stage: StageResult) -> None:
        if not stage.commands:
            return
        stage.status = RUNNING
        for command in stage.commands:
            assert_allowed(command)          # enforced even in dry-run
            if self.config.dry_run:
                continue
            code, tail = self._exec(command)
            stage.exit_codes.append(code)
            if code != 0:
                stage.status = FAILED
                stage.failure_reason = (
                    f"`{command.display}` exited {code}. {tail}")
                return
        if not self.config.dry_run:
            stage.status = SUCCESS

    def _exec(self, command: DemoCommand) -> tuple[int, str]:
        """Run one allowlisted module as an argv array (no shell)."""
        argv = [sys.executable, "-m", command.module, *command.args]
        logger.info("[demo] running: %s", command.display)
        try:
            proc = subprocess.run(  # noqa: S603 - argv array, allowlisted module
                argv, cwd=str(self.project_root), capture_output=True,
                text=True, timeout=_COMMAND_TIMEOUT_SECONDS, shell=False)
        except subprocess.TimeoutExpired:
            return 124, "command timed out"
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return proc.returncode, " | ".join(tail)

    # ------------------------------------------------------------ internal
    def _run_internal(self, stage: StageResult) -> None:
        stage.status = RUNNING  # concrete status set by _recheck

    # ------------------------------------------------------------ rechecks
    def _recheck(self, stage: StageResult) -> None:
        recheck = stage.details.get("recheck")
        if self.config.dry_run and stage.kind != KIND_INTERNAL:
            # Nothing was executed; leave command stages as planned.
            if stage.status not in (REUSED_EXISTING,):
                stage.status = SKIPPED
                stage.details["dry_run"] = True
            return
        handler = getattr(self, f"_recheck_{recheck}", None)
        if handler is not None:
            handler(stage)

    def _mark(self, stage: StageResult, ready: bool, missing: Any = None) -> None:
        if ready:
            if stage.status != REUSED_EXISTING:
                stage.status = SUCCESS
        else:
            stage.status = FAILED
            stage.failure_reason = stage.failure_reason or (
                f"Required artefacts still missing: {missing}")

    def _recheck_env(self, stage: StageResult) -> None:
        from pathlib import Path
        input_dir = Path(self.config.engine_c_input_dir)
        if not input_dir.is_absolute():
            input_dir = Path(self.project_root) / input_dir
        ok = input_dir.is_dir()
        stage.details["engine_c_input_dir_exists"] = ok
        if not ok:
            stage.warnings.append(
                f"Engine C sample input dir not found: {input_dir}")
        self._mark(stage, ok, missing=str(input_dir))

    def _recheck_engine_a(self, stage: StageResult) -> None:
        ready = readiness.engine_a_ready(self.paths.registry_dir,
                                         self.config.engine_a_datasets)
        stage.details["readiness"] = ready
        stage.artifacts.append(str(self.paths.registry_dir / "production.json"))
        if ready["ready"] and not stage.reused and stage.status != FAILED \
                and not stage.commands:
            stage.status = REUSED_EXISTING
        self._mark(stage, ready["ready"], missing=ready["missing_datasets"])

    def _recheck_engine_b(self, stage: StageResult) -> None:
        ready = readiness.engine_b_ready(self.paths.network_health_dir,
                                         self.config.engine_b_dataset)
        stage.details["readiness"] = ready
        self._mark(stage, ready["ready"], missing=self.config.engine_b_dataset)

    def _recheck_engine_c(self, stage: StageResult) -> None:
        # Assessment stage: core artefacts only (dashboard views come later).
        ready = readiness.engine_c_ready(self.paths.network_config_dir,
                                         self.config.engine_c_snapshot,
                                         include_dashboard=False)
        stage.details["readiness"] = ready
        stage.artifacts.append(ready["path"])
        self._mark(stage, ready["ready"], missing=ready["missing"])

    def _recheck_engine_c_dashboard(self, stage: StageResult) -> None:
        ready = readiness.engine_c_ready(self.paths.network_config_dir,
                                         self.config.engine_c_snapshot,
                                         include_dashboard=True)
        stage.details["readiness"] = ready
        stage.artifacts.append(ready["path"])
        self._mark(stage, ready["ready"], missing=ready["missing"])

    def _recheck_syslog(self, stage: StageResult) -> None:
        ready = readiness.syslog_ready(self.paths.outputs_dir,
                                       self.config.syslog_run)
        stage.details["readiness"] = ready
        if ready["ready"]:
            stage.status = REUSED_EXISTING if not stage.commands else SUCCESS
            stage.details["run_id"] = ready["run_id"]
        elif self.config.require_syslog:
            stage.status = FAILED
            stage.failure_reason = ("--require-syslog set but no syslog run is "
                                    "available. Ingest first with "
                                    "scripts.ingest_switch_syslog.")
        else:
            stage.status = SKIPPED

    def _recheck_correlation(self, stage: StageResult) -> None:
        ready = readiness.correlation_ready(self.paths.correlation_dir,
                                            self.config.correlation_id)
        stage.details["readiness"] = ready
        stage.artifacts.append(ready["path"])
        self._mark(stage, ready["ready"], missing=ready["missing"])

    def _recheck_streaming(self, stage: StageResult) -> None:
        ready = readiness.streaming_ready(
            self.paths.outputs_dir / "streaming")
        stage.details["readiness"] = ready
        self._mark(stage, ready["ready"], missing=ready["missing"])

    def _recheck_dashboard(self, stage: StageResult) -> None:
        result = readiness.dashboard_readiness(
            self.paths, self.config.engine_c_snapshot,
            self.config.correlation_id, self.config.syslog_run)
        stage.details["dashboard"] = result
        if not result["safety_banner_ok"]:
            stage.warnings.append("Safety banner check did not confirm offline / "
                                  "no-command-execution posture.")
        self._mark(stage, result["ready"],
                   missing=result["missing_required_sections"])

    def _recheck_report(self, stage: StageResult) -> None:
        stage.status = SUCCESS  # the report itself is written by artifacts.py

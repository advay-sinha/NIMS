"""Preflight readiness assessment for live adapters.

Produces a non-destructive readiness report per source: adapter presence,
dependency availability, mode, enabled/live state, required env vars present
(never their values), bind-port availability, target inventory presence, path
writability, safety status and the exact remaining setup steps. No state-changing
or destructive test is performed.
"""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.live_logging.adapters import MODE_LIVE
from src.live_logging.adapters.base import AdapterContext, LiveAdapter
from src.live_logging.adapters.registry import build_adapter, build_context, resolve_source, SPEC_SOURCES
from src.live_logging.models import utc_now_iso

# Overall readiness statuses (spec section 7).
READY = "READY"
NOT_READY = "NOT_READY"
DISABLED = "DISABLED"
BLOCKED_BY_SAFETY = "BLOCKED_BY_SAFETY"
MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
MISSING_CONFIGURATION = "MISSING_CONFIGURATION"
MISSING_CREDENTIALS = "MISSING_CREDENTIALS"


@dataclass
class ReadinessReport:
    """Structured readiness for one source (safe to serialise — no secrets)."""

    source: str
    friendly_name: str
    engine_target: str
    status: str
    adapter_exists: bool = True
    dependency: str | None = None
    dependency_ok: bool = True
    mode: str = "offline"
    enabled: bool = False
    live_enabled: bool = False
    read_only: bool = True
    required_env_vars: list[str] = field(default_factory=list)
    env_present: dict[str, bool] = field(default_factory=dict)
    bind_port: int | None = None
    bind_port_available: bool | None = None
    targets_configured: bool | None = None
    checkpoint_writable: bool = True
    output_writable: bool = True
    source_allowlist: list[str] = field(default_factory=list)
    safety_problems: list[str] = field(default_factory=list)
    remaining_steps: list[str] = field(default_factory=list)
    can_run_once_live: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _port_available(host: str, port: int) -> bool:
    # No SO_REUSEADDR here: this is an availability probe, so an exclusive bind
    # gives an honest "is the port actually free" answer.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def assess(adapter: LiveAdapter, context: AdapterContext) -> ReadinessReport:
    """Assess a single adapter's readiness (non-destructive)."""
    dep = adapter.check_dependencies()
    safety_problems = [p for p in adapter.validate_configuration()]
    env_vars = adapter.required_env_vars()
    env_present = {name: bool(os.environ.get(name)) for name in env_vars}
    report = ReadinessReport(
        source=adapter.name,
        friendly_name=adapter.friendly_name,
        engine_target=adapter.engine_target,
        status=NOT_READY,
        dependency=adapter.dependency,
        dependency_ok=dep.ok,
        mode=adapter.mode,
        enabled=adapter.enabled,
        live_enabled=(adapter.mode == MODE_LIVE and adapter.enabled),
        read_only=adapter.read_only,
        required_env_vars=env_vars,
        env_present=env_present,
        source_allowlist=list(adapter.cfg.get("allowed_sources") or []),
        checkpoint_writable=_writable(Path(context.checkpoint_dir or (context.output_dir / "checkpoints"))),
        output_writable=_writable(Path(context.output_dir)),
    )

    # Bind-port availability for receivers.
    if "bind_port" in adapter.cfg:
        report.bind_port = int(adapter.cfg.get("bind_port"))
        report.bind_port_available = _port_available(
            str(adapter.cfg.get("bind_host", "127.0.0.1")), report.bind_port
        )
    # SNMP target inventory presence.
    if adapter.name == "hirschmann_snmp":
        tf = adapter.cfg.get("targets_file")
        report.targets_configured = bool(tf and os.path.isfile(tf))

    _finalize(report, adapter, safety_problems, env_present)
    return report


def _finalize(
    report: ReadinessReport,
    adapter: LiveAdapter,
    safety_problems: list[str],
    env_present: dict[str, bool],
) -> None:
    steps: list[str] = []
    # Precedence: disabled < safety < dependency < config < creds < ready.
    if not adapter.enabled or adapter.mode != MODE_LIVE:
        report.status = DISABLED
        if not adapter.enabled:
            steps.append(f"Set {adapter.name}.enabled: true after approval")
        if adapter.mode != MODE_LIVE:
            steps.append(f"Set {adapter.name}.mode: live for a live run")
        report.remaining_steps = steps + _shared_steps(report)
        report.safety_problems = safety_problems
        return

    if safety_problems:
        report.status = BLOCKED_BY_SAFETY
        report.safety_problems = safety_problems
        report.remaining_steps = [f"Resolve safety issue: {p}" for p in safety_problems]
        return

    if not report.dependency_ok:
        report.status = MISSING_DEPENDENCY
        report.remaining_steps = [f"Install the '{adapter.dependency}' package"]
        return

    missing_env = [name for name, present in env_present.items() if not present]
    missing_cfg = _missing_config(report, adapter)
    if missing_cfg:
        report.status = MISSING_CONFIGURATION
        report.remaining_steps = missing_cfg
        return
    if missing_env:
        report.status = MISSING_CREDENTIALS
        report.remaining_steps = [f"Set environment variable {name}" for name in missing_env]
        return

    report.status = READY
    report.can_run_once_live = True
    report.remaining_steps = []


def _missing_config(report: ReadinessReport, adapter: LiveAdapter) -> list[str]:
    steps: list[str] = []
    if report.bind_port_available is False:
        steps.append(f"Free UDP port {report.bind_port} (currently in use)")
    if report.targets_configured is False:
        steps.append("Provide configs/hirschmann_targets.local.yaml with target inventory")
    if not report.output_writable:
        steps.append("Ensure the output directory is writable")
    if not report.checkpoint_writable:
        steps.append("Ensure the checkpoint directory is writable")
    return steps


def _shared_steps(report: ReadinessReport) -> list[str]:
    steps: list[str] = []
    if report.dependency and not report.dependency_ok:
        steps.append(f"Install the '{report.dependency}' package")
    missing_env = [name for name, present in report.env_present.items() if not present]
    steps += [f"Set environment variable {name}" for name in missing_env]
    return steps


def assess_source(
    source: str,
    live_cfg: Mapping[str, Any],
    sophos_cfg: Mapping[str, Any],
    hirschmann_cfg: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> ReadinessReport:
    """Build the adapter for a source and assess its readiness."""
    source = resolve_source(source)
    context = build_context(live_cfg, sophos_cfg, hirschmann_cfg, output_dir=output_dir)
    adapter = build_adapter(source, sophos_cfg, hirschmann_cfg, context)
    return assess(adapter, context)


def assess_all(
    live_cfg: Mapping[str, Any],
    sophos_cfg: Mapping[str, Any],
    hirschmann_cfg: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> list[ReadinessReport]:
    """Assess every known source."""
    return [
        assess_source(src, live_cfg, sophos_cfg, hirschmann_cfg, output_dir)
        for src in SPEC_SOURCES
    ]


def write_readiness(reports: list[ReadinessReport], output_dir: str | Path) -> Path:
    """Persist readiness.json for the backend API to serve."""
    import json

    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "readiness.json"
    payload = {
        "generated_at": utc_now_iso(),
        "sources": [r.to_dict() for r in reports],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path

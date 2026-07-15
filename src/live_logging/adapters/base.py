"""Common live-adapter contract and shared run/persist machinery.

Every adapter subclasses :class:`LiveAdapter` and implements the mode-specific
collectors (``_collect_offline`` / ``_collect_mock`` / ``_collect_live``) plus a
little metadata. The base class provides the shared surface required by the spec
(validate_configuration, check_dependencies, test_connection, run_once, start,
stop, health, checkpoint, status) and reuses the existing normalizer / redaction
/ event-store / checkpoint layers so no downstream logic is duplicated.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.live_logging import normalizer, redaction
from src.live_logging.adapters import MODE_LIVE, MODE_MOCK, MODE_OFFLINE, MODES
from src.live_logging.adapters import safety
from src.live_logging.adapters.errors import ConfigurationError, SafetyViolation
from src.live_logging.checkpoint import CheckpointManager
from src.live_logging.event_store import EventStore
from src.live_logging.models import ENGINE_UNKNOWN, utc_now_iso

logger = logging.getLogger(__name__)

# Collector return type: (source records, parse-error samples).
CollectResult = tuple[list[dict[str, Any]], list[str]]


@dataclass
class AdapterContext:
    """Shared runtime context passed to every adapter."""

    output_dir: Path
    routing: dict[str, str] = field(default_factory=dict)
    secret_env_vars: list[str] = field(default_factory=list)
    redact_secrets: bool = True
    checkpoint_dir: Path | None = None
    dry_run: bool = False
    # Test/mock-injected payload (interpreted per adapter).
    mock: Any = None

    def store(self) -> EventStore:
        return EventStore(self.output_dir)

    def checkpoints(self) -> CheckpointManager:
        return CheckpointManager(self.checkpoint_dir or (Path(self.output_dir) / "checkpoints"))


@dataclass
class AdapterResult:
    """Outcome of a non-destructive test (test_connection/check_dependencies)."""

    ok: bool
    detail: str | None = None


@dataclass
class AdapterRunResult:
    """Outcome of a single run_once."""

    source: str
    engine_target: str
    mode: str
    events: int = 0
    raw_events: int = 0
    parse_errors: list[str] = field(default_factory=list)
    persisted: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveAdapter:
    """Base class for all live ingestion adapters."""

    # Subclass metadata.
    name: str = ""                 # readiness id (spec source name)
    source_key: str = ""           # routing key (matches normalizer routing map)
    engine_target: str = ENGINE_UNKNOWN
    friendly_name: str = ""
    dependency: str | None = None  # pip module required for live; None if none

    def __init__(self, cfg: dict[str, Any], context: AdapterContext) -> None:
        self.cfg = dict(cfg or {})
        self.ctx = context
        self.mode = str(self.cfg.get("mode", MODE_OFFLINE)).lower()
        self.enabled = bool(self.cfg.get("enabled", False))
        self.read_only = self.cfg.get("read_only", True) is True

    # ---- to override --------------------------------------------------------

    def _collect_offline(self) -> CollectResult:
        raise NotImplementedError

    def _collect_mock(self) -> CollectResult:
        raise NotImplementedError

    def _collect_live(self) -> CollectResult:
        raise NotImplementedError(f"{self.name} live mode is not implemented")

    def _test_connection_live(self) -> None:
        """Non-destructive live reachability check (override per adapter)."""
        raise NotImplementedError

    def _validate_extra(self) -> list[str]:
        """Adapter-specific configuration problems (override as needed)."""
        return []

    def required_env_vars(self) -> list[str]:
        """Env var names required for live mode (override; never their values)."""
        return []

    # ---- shared surface -----------------------------------------------------

    def check_dependencies(self) -> AdapterResult:
        """Report whether the live dependency is importable (never imports it)."""
        if not self.dependency:
            return AdapterResult(True, "no external dependency required")
        present = importlib.util.find_spec(self.dependency) is not None
        return AdapterResult(present, None if present else f"missing dependency: {self.dependency}")

    def validate_configuration(self) -> list[str]:
        """Return a list of configuration problems (empty when valid)."""
        problems: list[str] = []
        if self.mode not in MODES:
            problems.append(f"mode must be one of {MODES}, got {self.mode!r}")
        try:
            safety.assert_read_only(self.cfg, require_read_only=(self.mode == MODE_LIVE))
        except SafetyViolation as exc:
            problems.append(str(exc))
        problems.extend(self._validate_extra())
        return problems

    def collect(self) -> CollectResult:
        """Collect source records for the current mode (enforces safety on live)."""
        if self.mode == MODE_LIVE:
            safety.assert_read_only(self.cfg, require_read_only=True)
            if not self.enabled:
                raise ConfigurationError(f"{self.name} live mode requires enabled=true")
            dep = self.check_dependencies()
            if not dep.ok:
                raise ConfigurationError(dep.detail or "missing dependency")
            return self._collect_live()
        if self.mode == MODE_MOCK:
            return self._collect_mock()
        return self._collect_offline()

    def run_once(self) -> AdapterRunResult:
        """Collect once, redact, normalize, persist and checkpoint."""
        try:
            records, errors = self.collect()
        except Exception as exc:  # noqa: BLE001 — isolate the source
            return AdapterRunResult(self.source_key, self.engine_target, self.mode,
                                    error=_safe_error(exc), persisted=False)
        if self.ctx.redact_secrets:
            records = [redaction.redact(r, self.ctx.secret_env_vars) for r in records]
        events, raws = normalizer.build_batch(records, self.ctx.routing)

        if self.ctx.dry_run:
            return AdapterRunResult(self.source_key, self.engine_target, self.mode,
                                    events=len(events), raw_events=len(raws),
                                    parse_errors=errors, persisted=False)

        store = self.ctx.store()
        store.append_normalized(events)
        store.append_raw(raws)
        self.ctx.checkpoints().save(
            self.source_key,
            {"last_run": utc_now_iso(), "event_count": len(events), "mode": self.mode},
        )
        return AdapterRunResult(self.source_key, self.engine_target, self.mode,
                                events=len(events), raw_events=len(raws), parse_errors=errors)

    def test_connection(self) -> AdapterResult:
        """Non-destructive connectivity check; offline/mock always succeed."""
        if self.mode != MODE_LIVE:
            return AdapterResult(True, f"{self.mode} mode — no live connection attempted")
        dep = self.check_dependencies()
        if not dep.ok:
            return dep
        try:
            self._test_connection_live()
            return AdapterResult(True, "connection ok")
        except Exception as exc:  # noqa: BLE001
            return AdapterResult(False, _safe_error(exc))

    # start/stop exist for the interface; persistent listeners are out of scope
    # for this phase (run-once is the first live-validation path).
    def start(self) -> AdapterRunResult:
        """Alias for run_once (persistent collection is not enabled this phase)."""
        return self.run_once()

    def stop(self) -> None:
        """No persistent resource is held; nothing to stop."""
        return None

    def checkpoint(self) -> dict[str, Any]:
        """Return the persisted checkpoint cursor for this source."""
        return self.ctx.checkpoints().load(self.source_key).to_dict()

    def health(self) -> dict[str, Any]:
        """Lightweight health snapshot for status reporting."""
        return {
            "source": self.source_key,
            "name": self.name,
            "engine_target": self.engine_target,
            "mode": self.mode,
            "enabled": self.enabled,
            "read_only": self.read_only,
            "dependency_ok": self.check_dependencies().ok,
        }

    def status(self) -> dict[str, Any]:
        """Full status: health + configuration validity."""
        return {**self.health(), "problems": self.validate_configuration()}


def _safe_error(exc: Exception) -> str:
    """Redact any secret-looking content from an exception message."""
    return redaction.redact_text(f"{type(exc).__name__}: {exc}")

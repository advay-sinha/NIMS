"""Ingestion orchestration (one-shot, offline-first) (Phase 9).

Purpose
-------
Runs the enabled ingestion sources once, in a bounded retry loop, isolating each
source so one failure never crashes the others (spec Phase 9 > Failure and Retry
Strategy). For each source it: parses → redacts → normalizes → persists
(append-only JSONL) → checkpoints, then records a :class:`SourceStatus`. The
aggregate :class:`IngestionStatus` is returned (and written by
:mod:`reports`).

Only OFFLINE sample sources are wired here. Live clients are injected (mockable)
and disabled by default; nothing in this module opens a socket, polls a device
or executes a command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from src.live_logging import (
    config_diff,
    hirschmann_config,
    hirschmann_snmp,
    hirschmann_traps,
    normalizer,
    redaction,
    sophos_api,
    sophos_syslog,
)
from src.live_logging.checkpoint import CheckpointManager
from src.live_logging.event_store import EventStore
from src.live_logging.models import (
    STATUS_DISABLED,
    STATUS_FAILED,
    STATUS_OK,
    IngestionStatus,
    SourceStatus,
    utc_now_iso,
)
from src.live_logging.routing import DEFAULT_ROUTING, route

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "outputs/live_logging"

# Per-source parser callables return (records, parse_error_samples).
ParseResult = tuple[list[dict[str, Any]], list[str]]


class LiveLogger:
    """Offline-first ingestion orchestrator."""

    def __init__(
        self,
        live_cfg: Mapping[str, Any],
        sophos_cfg: Mapping[str, Any],
        hirschmann_cfg: Mapping[str, Any],
        output_dir: str | Path | None = None,
        sophos_fetcher: Optional[Callable[[], list[Mapping[str, Any]]]] = None,
    ) -> None:
        self.live_cfg = dict(live_cfg or {})
        self.sophos_cfg = dict(sophos_cfg or {})
        self.hirschmann_cfg = dict(hirschmann_cfg or {})
        self.output_dir = Path(output_dir or self.live_cfg.get("output_dir") or DEFAULT_OUTPUT_DIR)
        self.routing = {**DEFAULT_ROUTING, **dict(self.live_cfg.get("routing") or {})}
        self.retry = dict(self.live_cfg.get("retry") or {})
        self.redact_secrets = bool(self.live_cfg.get("redact_secrets", True))
        self.secret_env_vars = self._collect_secret_env_vars()
        self._sophos_fetcher = sophos_fetcher

        checkpoint_dir = (
            self.live_cfg.get("checkpoints", {}).get("path")
            or (self.output_dir / "checkpoints")
        )
        self.store = EventStore(self.output_dir)
        self.checkpoints = CheckpointManager(checkpoint_dir)

    # ---- public API -------------------------------------------------------

    def run_once(self, sources: list[str] | None = None) -> IngestionStatus:
        """Run each enabled source once; return the aggregate status."""
        started = utc_now_iso()
        selected = sources or [
            "sophos_api",
            "sophos_syslog",
            "hirschmann_snmp",
            "hirschmann_traps",
            "hirschmann_config",
        ]
        statuses = [self._run_source(name) for name in selected]
        statuses = [s for s in statuses if s is not None]

        total = sum(s.events for s in statuses)
        status = IngestionStatus(
            started_at=started,
            finished_at=utc_now_iso(),
            mode=str(self.live_cfg.get("mode", "offline")),
            read_only=bool(self.live_cfg.get("safety", {}).get("read_only", True)),
            total_events=total,
            sources=statuses,
        )
        self._tally(status)
        return status

    # ---- per-source runner ------------------------------------------------

    def _run_source(self, name: str) -> SourceStatus | None:
        enabled, mode = self._source_enabled(name)
        engine_target = route(name, self.routing)
        if not enabled:
            return SourceStatus(source=name, engine_target=engine_target,
                                status=STATUS_DISABLED, mode=mode, events=0)

        max_attempts = int(self.retry.get("max_attempts", 1) or 1)
        last_error: tuple[str, str] | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                records, parse_errors = self._parse_source(name)
                events_written, raw_written = self._persist(name, records)
                self._checkpoint(name, events_written)
                status = SourceStatus(
                    source=name, engine_target=engine_target, status=STATUS_OK,
                    mode=mode, events=events_written, raw_events=raw_written, attempts=attempt,
                )
                if parse_errors:
                    status.error_category = self._parse_error_category(name)
                    status.error_message = f"{len(parse_errors)} parse error(s); e.g. {parse_errors[0]}"
                return status
            except OSError as exc:
                last_error = ("checkpoint_write_error", str(exc))
            except Exception as exc:  # noqa: BLE001 — isolate every source
                last_error = (self._runtime_error_category(name), str(exc))
                logger.warning("Source %s failed on attempt %d: %s", name, attempt, exc)
        category, message = last_error or ("unknown_error", "unknown")
        return SourceStatus(
            source=name, engine_target=engine_target, status=STATUS_FAILED,
            mode=mode, events=0, attempts=max_attempts,
            error_category=category, error_message=message,
        )

    # ---- parsing dispatch -------------------------------------------------

    def _parse_source(self, name: str) -> ParseResult:
        if name == "sophos_api":
            client = sophos_api.SophosCentralClient(
                self._sophos_block("central_api"), fetcher=self._sophos_fetcher
            )
            return sophos_api.parse_sophos_items(client.fetch()), []
        if name == "sophos_syslog":
            path = self._sophos_block("firewall_syslog").get("offline_sample_path")
            return sophos_syslog.read_offline(path) if path else ([], [])
        if name == "hirschmann_snmp":
            block = self._hirschmann_block("snmp_polling")
            path = block.get("offline_sample_path")
            thresholds = self.hirschmann_cfg.get("thresholds") or {}
            return (hirschmann_snmp.read_offline(path, thresholds) if path else []), []
        if name == "hirschmann_traps":
            path = self._hirschmann_block("traps").get("offline_sample_path")
            return hirschmann_traps.read_offline(path) if path else ([], [])
        if name == "hirschmann_config":
            return self._parse_config_changes(), []
        raise ValueError(f"Unknown source: {name}")

    def _parse_config_changes(self) -> list[dict[str, Any]]:
        block = self._hirschmann_block("config_snapshots")
        snapshot_dir = block.get("snapshot_dir")
        if not snapshot_dir:
            return []
        grouped = hirschmann_config.load_snapshots_dir(snapshot_dir)
        records: list[dict[str, Any]] = []
        for snaps in grouped.values():
            records.extend(config_diff.diff_snapshot_series(snaps))
        return records

    # ---- persistence ------------------------------------------------------

    def _persist(self, name: str, records: list[dict[str, Any]]) -> tuple[int, int]:
        if self.redact_secrets:
            records = [redaction.redact(r, self.secret_env_vars) for r in records]
        events, raws = normalizer.build_batch(records, self.routing)
        self.store.append_normalized(events)
        self.store.append_raw(raws)
        return len(events), len(raws)

    def _checkpoint(self, name: str, events: int) -> None:
        self.checkpoints.save(name, {"last_poll_time": utc_now_iso(), "event_count": events})

    # ---- config helpers ---------------------------------------------------

    def _source_enabled(self, name: str) -> tuple[bool, str]:
        mapping = {
            "sophos_api": self._sophos_block("central_api"),
            "sophos_syslog": self._sophos_block("firewall_syslog"),
            "hirschmann_snmp": self._hirschmann_block("snmp_polling"),
            "hirschmann_traps": self._hirschmann_block("traps"),
            "hirschmann_config": self._hirschmann_block("config_snapshots"),
        }
        block = mapping.get(name, {})
        return bool(block.get("enabled", False)), str(block.get("mode", "offline"))

    def _sophos_block(self, key: str) -> dict[str, Any]:
        return dict((self.sophos_cfg.get("sophos") or self.sophos_cfg).get(key, {}))

    def _hirschmann_block(self, key: str) -> dict[str, Any]:
        return dict((self.hirschmann_cfg.get("hirschmann") or self.hirschmann_cfg).get(key, {}))

    def _collect_secret_env_vars(self) -> list[str]:
        names: list[str] = []
        for cfg in (self.sophos_cfg, self.hirschmann_cfg):
            _walk_env_names(cfg, names)
        return names

    @staticmethod
    def _parse_error_category(name: str) -> str:
        return {
            "sophos_syslog": "syslog_parse_error",
            "hirschmann_traps": "trap_parse_error",
            "hirschmann_config": "config_diff_error",
        }.get(name, "unknown_error")

    @staticmethod
    def _runtime_error_category(name: str) -> str:
        return {
            "sophos_api": "api_auth_error",
            "sophos_syslog": "syslog_parse_error",
            "hirschmann_snmp": "snmp_timeout",
            "hirschmann_traps": "trap_parse_error",
            "hirschmann_config": "config_diff_error",
        }.get(name, "unknown_error")

    @staticmethod
    def _tally(status: IngestionStatus) -> None:
        # Counts are recomputed from persisted events by reports; here we only
        # need per-source rollups the report can enrich.
        for source in status.sources:
            status.events_by_engine[source.engine_target] = (
                status.events_by_engine.get(source.engine_target, 0) + source.events
            )


def run_and_report(
    live_cfg: Mapping[str, Any],
    sophos_cfg: Mapping[str, Any],
    hirschmann_cfg: Mapping[str, Any],
    sources: list[str] | None = None,
    output_dir: str | Path | None = None,
    sophos_fetcher: Optional[Callable[[], list[Mapping[str, Any]]]] = None,
) -> tuple[IngestionStatus, Path]:
    """Convenience: run one ingestion pass and write status + report.

    Returns the enriched :class:`IngestionStatus` and the output directory.
    Used by the CLI scripts so their glue stays minimal.
    """
    from src.live_logging import reports

    logger_ = LiveLogger(
        live_cfg, sophos_cfg, hirschmann_cfg,
        output_dir=output_dir, sophos_fetcher=sophos_fetcher,
    )
    status = logger_.run_once(sources)
    reports.enrich_status(status, logger_.output_dir)
    reports.write_status(status, logger_.output_dir)
    reports.write_report(status, logger_.output_dir)
    return status, logger_.output_dir


def _walk_env_names(obj: Any, out: list[str]) -> None:
    """Collect values of any ``*_env`` keys (they name secret env vars)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key.endswith("_env") and isinstance(value, str):
                out.append(value)
            else:
                _walk_env_names(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_env_names(item, out)

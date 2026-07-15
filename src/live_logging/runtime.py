"""Unified adapter-based ingestion runtime.

One place that turns a set of sources + a requested mode into a run: it builds
each adapter, enforces the live-mode safety gate, runs run_once, aggregates the
outcomes into an :class:`IngestionStatus` and writes the status + report. Used by
every ``scripts.run_*`` entry point so the gating logic lives in exactly one
place.

Live-mode gate (spec section 8):
- ``--live`` runs a source only if its config has ``enabled: true`` AND
  ``mode: live`` AND ``read_only: true`` AND no forbidden flag is set.
- ``--offline`` / ``--mock`` force that (always-safe) mode.
- source failures never crash unrelated sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from src.live_logging import reports
from src.live_logging.adapters import MODE_LIVE, MODE_MOCK, MODE_OFFLINE
from src.live_logging.adapters.base import AdapterRunResult
from src.live_logging.adapters.registry import build_adapter, build_context, resolve_source, SPEC_SOURCES
from src.live_logging.models import (
    STATUS_DISABLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    IngestionStatus,
    SourceStatus,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


def run(
    sources: list[str] | str,
    live_cfg: Mapping[str, Any],
    sophos_cfg: Mapping[str, Any],
    hirschmann_cfg: Mapping[str, Any],
    mode: str | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    write_reports: bool = True,
    mock: Any = None,
) -> tuple[IngestionStatus, Path]:
    """Run the selected sources once and return (status, output_dir).

    ``mode`` is one of None (use config), 'offline', 'mock' or 'live'.
    """
    selected = _resolve_sources(sources)
    context = build_context(live_cfg, sophos_cfg, hirschmann_cfg, output_dir=output_dir, dry_run=dry_run, mock=mock)
    started = utc_now_iso()

    statuses: list[SourceStatus] = []
    for source in selected:
        statuses.append(_run_one(source, sophos_cfg, hirschmann_cfg, context, mode))

    status = IngestionStatus(
        started_at=started,
        finished_at=utc_now_iso(),
        mode=mode or str(live_cfg.get("mode", "offline")),
        read_only=True,
        total_events=sum(s.events for s in statuses),
        sources=statuses,
    )
    for s in statuses:
        status.events_by_engine[s.engine_target] = status.events_by_engine.get(s.engine_target, 0) + s.events

    if write_reports and not dry_run:
        reports.enrich_status(status, context.output_dir)
        reports.write_status(status, context.output_dir)
        reports.write_report(status, context.output_dir)
    return status, context.output_dir


def _run_one(source, sophos_cfg, hirschmann_cfg, context, mode) -> SourceStatus:
    try:
        # Build once to read config/metadata, then apply the requested mode.
        adapter = build_adapter(source, sophos_cfg, hirschmann_cfg, context)
        engine_target = adapter.engine_target

        if mode in (MODE_OFFLINE, MODE_MOCK):
            adapter = build_adapter(source, sophos_cfg, hirschmann_cfg, context, mode=mode)
        elif mode == MODE_LIVE:
            gate = _live_gate(adapter)
            if gate is not None:
                return SourceStatus(source=adapter.source_key, engine_target=engine_target,
                                    status=STATUS_SKIPPED, mode=adapter.mode, events=0,
                                    error_category="blocked", error_message=gate)
        # else: mode is None -> use config mode as-is.

        result = adapter.run_once()
        return _to_status(adapter.source_key, engine_target, result)
    except Exception as exc:  # noqa: BLE001 — isolate the source
        logger.warning("Source %s failed: %s", source, exc)
        return SourceStatus(source=str(source), engine_target="unknown", status=STATUS_FAILED,
                            mode=str(mode or "offline"), events=0,
                            error_category="unknown_error", error_message=str(exc))


def _live_gate(adapter) -> str | None:
    """Return a refusal reason if this adapter may not run live, else None."""
    if not adapter.enabled:
        return "source is disabled (enabled=true required for live)"
    if adapter.mode != MODE_LIVE:
        return f"source mode is '{adapter.mode}', not 'live'"
    if not adapter.read_only:
        return "read_only must be true for live mode"
    problems = adapter.validate_configuration()
    if problems:
        return "; ".join(problems)
    return None


def _to_status(source_key: str, engine_target: str, result: AdapterRunResult) -> SourceStatus:
    if result.error:
        return SourceStatus(source=source_key, engine_target=engine_target, status=STATUS_FAILED,
                            mode=result.mode, events=0, error_category="unknown_error",
                            error_message=result.error)
    status = SourceStatus(source=source_key, engine_target=engine_target, status=STATUS_OK,
                          mode=result.mode, events=result.events, raw_events=result.raw_events)
    if result.parse_errors:
        status.error_category = "parse_error"
        status.error_message = f"{len(result.parse_errors)} parse error(s); e.g. {result.parse_errors[0]}"
    return status


def _resolve_sources(sources: list[str] | str) -> list[str]:
    if sources in ("all", ["all"]):
        return list(SPEC_SOURCES)
    if isinstance(sources, str):
        sources = [sources]
    resolved: list[str] = []
    for s in sources:
        if s == "all":
            return list(SPEC_SOURCES)
        key = resolve_source(s)
        if key not in resolved:
            resolved.append(key)
    return resolved

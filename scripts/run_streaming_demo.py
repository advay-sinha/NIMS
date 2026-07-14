"""Entry point: offline real-time-style monitoring demo (safe replay).

Replays already-persisted Engine A/B/C and correlation artefacts into a unified
event stream, maintains an in-memory monitoring state, writes an append-only
event log, and exposes read-only "current state" artefacts for the dashboard.

This is a **demo/replay layer only**. It never contacts a device, opens SSH,
polls SNMP, captures packets, runs an engine pipeline or executes a remediation
command. Every input and output is a local file.

Usage
-----
    python -m scripts.run_streaming_demo               # paced demo (~1s/event)
    python -m scripts.run_streaming_demo --no-sleep    # instant (CI/one-shot)
    python -m scripts.run_streaming_demo --max-events 20 --loop
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.streaming.runtime import run_stream
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_STREAMING_CONFIG = "configs/streaming.yaml"
_LIVE_FLAGS = ("allow_live_sources", "allow_device_access",
               "allow_packet_capture", "allow_remediation_execution")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Offline streaming monitoring demo (safe replay; no device "
                    "access, no command execution).")
    add_common_arguments(parser)
    parser.add_argument("--streaming-config", default=DEFAULT_STREAMING_CONFIG,
                        help="Streaming config (default: configs/streaming.yaml).")
    parser.add_argument("--tick-seconds", type=float, default=None,
                        help="Override seconds between replayed events.")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Override the maximum number of events to emit.")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the ordered event stream (bounded by a cap).")
    parser.add_argument("--no-sleep", action="store_true",
                        help="Do not pace the replay (emit instantly).")
    parser.add_argument("--correlation-id", default=None,
                        help="Override the correlation run id to replay.")
    parser.add_argument("--snapshot-id", default=None,
                        help="Override the Engine C snapshot id to replay.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` on disabled/unsafe config)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    config = load_yaml(args.streaming_config)

    if not bool((config.get("streaming") or {}).get("enabled", True)):
        logger.warning("Streaming is disabled in %s; nothing to do.",
                       args.streaming_config)
        return 0
    unsafe = [f for f in _LIVE_FLAGS
              if bool((config.get("safety") or {}).get(f, False))]
    if unsafe:
        logger.error("Refusing to run: unsafe live flags enabled (%s). This "
                     "layer is offline-only.", ", ".join(unsafe))
        return 1

    config = _apply_overrides(copy.deepcopy(config), args)
    dirs = _resolve_dirs(ctx, config)
    sleep_fn = None if args.no_sleep else time.sleep

    result = run_stream(config, dirs, sleep_fn=sleep_fn,
                        tick_seconds=args.tick_seconds,
                        max_events=args.max_events,
                        loop=args.loop or None)

    summary = result.state.summary()
    logger.info("Streaming demo complete: %d event(s) replayed; %d active "
                "incident(s) (%d critical) — offline, no commands executed. "
                "Current state: %s", result.events_emitted,
                summary["active_incident_count"],
                summary["critical_incident_count"],
                dirs["current_state_dir"])
    return 0


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.correlation_id:
        config.setdefault("sources", {}).setdefault("correlation", {})[
            "default_correlation_id"] = args.correlation_id
    if args.snapshot_id:
        config.setdefault("sources", {}).setdefault("engine_c", {})[
            "default_snapshot_id"] = args.snapshot_id
    return config


def _resolve_dirs(ctx, config: dict) -> dict:
    """Combine engine artefact dirs (from paths) with streaming output dirs."""
    root = Path(ctx.paths.root)
    streaming = config.get("streaming", {}) or {}

    def _resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (root / p)

    return {
        # engine artefact sources (read-only)
        "correlation_dir": ctx.paths.correlation_dir,
        "network_config_dir": ctx.paths.network_config_dir,
        "network_health_dir": ctx.paths.network_health_dir,
        "registry_dir": ctx.paths.registry_dir,
        "reports_dir": ctx.paths.reports_dir,
        "error_analysis_dir": ctx.paths.error_analysis_dir,
        "visualizations_dir": ctx.paths.visualizations_dir,
        "experiments_dir": ctx.paths.experiments_dir,
        "syslog_ingestion_dir": getattr(ctx.paths, "outputs_dir",
                                         root / "outputs") / "syslog_ingestion",
        # streaming outputs (write)
        "output_dir": _resolve(streaming.get("output_dir", "outputs/streaming")),
        "current_state_dir": _resolve(
            streaming.get("current_state_dir", "outputs/streaming/current")),
        "event_log_path": _resolve(
            streaming.get("event_log_path", "outputs/streaming/events.jsonl")),
    }


if __name__ == "__main__":
    sys.exit(main())

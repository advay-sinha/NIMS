"""Entry point: unified live/offline/mock ingestion runtime (one-shot).

Runs the selected ingestion sources through the adapter layer. Live mode is
DISABLED by default and only runs a source whose config has enabled=true,
mode=live and read_only=true (spec section 8); offline/mock always safe.

Usage
-----
    python -m scripts.run_live_logger --offline --run-once
    python -m scripts.run_live_logger --mock --source all --run-once
    python -m scripts.run_live_logger --source sophos_firewall_syslog --live --run-once
    python -m scripts.run_live_logger --status-only
"""

from __future__ import annotations

import argparse
import logging

from scripts._bootstrap import add_common_arguments, bootstrap
from src.live_logging import runtime
from src.live_logging.adapters.registry import SPEC_SOURCES
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_LIVE_CONFIG = "configs/live_logging.yaml"
DEFAULT_SOPHOS_CONFIG = "configs/sophos_logging.yaml"
DEFAULT_HIRSCHMANN_CONFIG = "configs/hirschmann_logging.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Unified live/offline/mock ingestion runtime.")
    add_common_arguments(parser)
    parser.add_argument("--live-config", default=DEFAULT_LIVE_CONFIG)
    parser.add_argument("--sophos-config", default=DEFAULT_SOPHOS_CONFIG)
    parser.add_argument("--hirschmann-config", default=DEFAULT_HIRSCHMANN_CONFIG)
    parser.add_argument("--source", action="append", default=None,
                        help="Source name or 'all'; repeatable (default: all).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="Force offline mode (saved samples).")
    mode.add_argument("--mock", action="store_true", help="Force mock mode (injected transport).")
    mode.add_argument("--live", action="store_true", help="Attempt live mode (gated; disabled by default).")
    parser.add_argument("--run-once", action="store_true", help="Run one pass (default action).")
    parser.add_argument("--scheduled", action="store_true",
                        help="Poll continuously every --interval seconds until interrupted (Ctrl-C).")
    parser.add_argument("--interval", type=int, default=None,
                        help="Seconds between scheduled cycles (default: SNMP poll_interval_seconds or 60).")
    parser.add_argument("--status-only", action="store_true", help="Print adapter status; do not ingest.")
    parser.add_argument("--dry-run", action="store_true", help="Validate + collect but do not persist.")
    return parser


def _mode(args: argparse.Namespace) -> str | None:
    if args.live:
        return "live"
    if args.mock:
        return "mock"
    if args.offline:
        return "offline"
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success)."""
    args = build_parser().parse_args(argv)
    bootstrap(args)

    live = load_yaml(args.live_config).get("live_logging", {})
    sophos = load_yaml(args.sophos_config)
    hirschmann = load_yaml(args.hirschmann_config)
    sources = args.source or ["all"]

    if args.status_only:
        from src.live_logging.adapters import preflight

        for report in preflight.assess_all(live, sophos, hirschmann, live.get("output_dir")):
            logger.info("[%s] %s mode=%s enabled=%s", report.status, report.friendly_name,
                        report.mode, report.enabled)
        return 0

    if args.scheduled:
        interval = args.interval or _default_interval(hirschmann)
        return _run_scheduled(sources, live, sophos, hirschmann, _mode(args), interval, args.dry_run)

    status, output_dir = runtime.run(
        sources, live, sophos, hirschmann,
        mode=_mode(args), dry_run=args.dry_run,
    )
    _log_cycle(status, output_dir)
    return 0


def _default_interval(hirschmann_cfg: dict) -> int:
    """Poll interval (s) from the SNMP config, falling back to 60."""
    snmp = (hirschmann_cfg.get("hirschmann") or hirschmann_cfg).get("snmp_polling", {})
    try:
        return max(5, int(snmp.get("poll_interval_seconds", 60)))
    except (TypeError, ValueError):
        return 60


def _log_cycle(status, output_dir) -> None:
    logger.info("Ingestion complete: %d events, healthy=%s, output=%s",
                status.total_events, status.healthy, output_dir)
    for s in status.sources:
        logger.info("  %-18s %-9s events=%d%s", s.source, s.status, s.events,
                    f" ({s.error_message})" if s.error_message else "")


def _run_scheduled(sources, live, sophos, hirschmann, mode, interval, dry_run) -> int:
    """Poll continuously every ``interval`` seconds until interrupted.

    Still fully read-only: each cycle is one ``runtime.run`` pass (the same gated
    live/offline path as run-once). A source failure in one cycle never stops the
    loop. Ctrl-C exits cleanly.
    """
    import time

    logger.info("Scheduled ingestion every %ds (mode=%s). Press Ctrl-C to stop.", interval, mode or "config")
    cycle = 0
    try:
        while True:
            cycle += 1
            try:
                status, output_dir = runtime.run(
                    sources, live, sophos, hirschmann, mode=mode, dry_run=dry_run,
                )
                logger.info("cycle %d: %d events, healthy=%s", cycle, status.total_events, status.healthy)
            except Exception as exc:  # noqa: BLE001 — never let one cycle kill the loop
                logger.warning("cycle %d failed: %s", cycle, exc)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Scheduled ingestion stopped by user after %d cycle(s).", cycle)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

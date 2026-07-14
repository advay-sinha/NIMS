"""Entry point: offline ingestion of saved industrial-switch syslog files.

Parses saved Belden/DCN-style switch logs (raw syslog exports and PuTTY terminal
captures) into structured events, Engine B time-window features and Engine C
findings under ``outputs/syslog_ingestion/<run_id>/``.

This script is **offline and file-driven only**. It never contacts a device,
opens SSH/telnet, polls SNMP, captures packets, runs an engine pipeline or
executes a remediation command. Unknown log lines are recorded, never fatal.

Usage
-----
    python -m scripts.ingest_switch_syslog \\
        --input-dir datasets/raw/syslog \\
        --run-id lw_terminal_syslog_sample

    python -m scripts.ingest_switch_syslog \\
        --input-file datasets/raw/syslog/Switch_Log_1.txt \\
        --input-file datasets/raw/syslog/Terminal_IR9.txt \\
        --run-id lw_terminal_syslog_sample
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.syslog_ingestion.artifacts import (
    SAFETY_BANNER,
    ingest,
    read_input_files,
    write_run,
)
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_SYSLOG_CONFIG = "configs/syslog_ingestion.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest saved industrial-switch syslog files (offline, "
                    "read-only; no device access, no packet capture, no "
                    "remediation).")
    add_common_arguments(parser)
    parser.add_argument("--input-file", action="append", default=None,
                        help="A saved log file (repeatable).")
    parser.add_argument("--input-dir", default=None,
                        help="Directory of saved log files (all files read).")
    parser.add_argument("--run-id", default=None,
                        help="Run id (default: timestamped).")
    parser.add_argument("--syslog-config", default=DEFAULT_SYSLOG_CONFIG,
                        help="Syslog ingestion config "
                             "(default: configs/syslog_ingestion.yaml).")
    parser.add_argument("--window-minutes", type=int, default=None,
                        help="Primary Engine B window size in minutes.")
    parser.add_argument("--include-clock-unreliable", action="store_true",
                        help="Include Jan 1 1970 boot-clock events in features.")
    parser.add_argument("--host-holdout", default=None,
                        help="Hostname whose windows are held out as the test "
                             "split.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: "
                             "outputs/syslog_ingestion/<run_id>/).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` only on true fatal errors)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    if not (args.input_file or args.input_dir):
        logger.error("Provide --input-file (repeatable) and/or --input-dir.")
        return 1

    run_id = args.run_id or f"syslog_{_timestamp()}"
    config = load_yaml(args.syslog_config)

    try:
        contents = read_input_files(args.input_file, args.input_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    run = ingest(
        contents, run_id, config,
        window_minutes=args.window_minutes,
        include_clock_unreliable=args.include_clock_unreliable,
        host_holdout=args.host_holdout,
    )

    if not run.events:
        logger.warning("No parseable events found in the provided files; "
                       "writing an empty run for audit.")

    output_dir = Path(args.output_dir) if args.output_dir else \
        ctx.paths.outputs_dir / "syslog_ingestion" / run_id
    paths = write_run(run, output_dir)

    _print_summary(run, paths)
    return 0


def _timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print_summary(run, paths: dict[str, str]) -> None:
    """Log the operator-facing console summary."""
    summary_stamps = sorted(e.timestamp for e in run.events
                            if e.timestamp and not e.clock_unreliable)
    hosts = sorted({e.hostname for e in run.events if e.hostname})
    duplicates = sum(int(d["duplicate_count"]) - 1 for d in run.duplicates)
    time_range = (f"{summary_stamps[0]} -> {summary_stamps[-1]}"
                  if summary_stamps else "n/a")

    logger.info("[syslog-ingestion] %s", SAFETY_BANNER)
    logger.info("Parsed events: %d", len(run.events))
    logger.info("Dropped noise lines: %d", len(run.dropped))
    logger.info("Duplicate lines collapsed: %d", duplicates)
    logger.info("Hosts: %s", ", ".join(hosts) or "n/a")
    logger.info("Time range: %s", time_range)
    logger.info("Engine B windows: %s", paths.get("engine_b_windows", "n/a"))
    logger.info("Engine C findings: %s", paths.get("engine_c_findings", "n/a"))
    logger.info("Report: %s", paths.get("report", "n/a"))


if __name__ == "__main__":
    raise SystemExit(main())

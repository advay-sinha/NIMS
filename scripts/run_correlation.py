"""Entry point: cross-engine correlation of persisted NIMS artefacts.

Reads the already-persisted artefacts of Engine A (cyber), Engine B (network-
health) and Engine C (configuration) and correlates them into unified,
operator-facing incidents under ``outputs/correlation/<correlation_id>/``.

This script is **offline and artefact-driven only**. It never runs an engine
pipeline, never captures packets, never polls SNMP, never contacts a device and
never executes a command. At least one engine input is required; missing
optional artefacts are warned about and skipped, and the run fails only when no
usable signals are found.

Usage
-----
    python -m scripts.run_correlation \\
        --engine-c-snapshot sample_remediation \\
        --engine-b-dataset synthetic \\
        --engine-a-dataset unsw_nb15 \\
        --correlation-id sample_correlation
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.correlation.artifacts import write_correlation
from src.correlation.engine import correlate
from src.correlation.loader import (
    load_engine_a_signals,
    load_engine_b_signals,
    load_engine_c_signals,
)
from src.correlation.models import ENGINE_A, ENGINE_B, ENGINE_C, LoadResult
from src.correlation.syslog_loader import load_syslog_signals
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_CORRELATION_CONFIG = "configs/correlation.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Correlate Engine A/B/C artefacts into unified incidents "
                    "(offline, read-only; no command is executed).")
    add_common_arguments(parser)
    parser.add_argument("--engine-a-dataset", default=None,
                        help="Engine A dataset id (e.g. unsw_nb15). Optional.")
    parser.add_argument("--engine-b-dataset", default=None,
                        help="Engine B network-health dataset id "
                             "(e.g. synthetic). Optional.")
    parser.add_argument("--engine-c-snapshot", default=None,
                        help="Engine C snapshot id under outputs/network_config/. "
                             "Optional but recommended.")
    parser.add_argument("--correlation-id", default=None,
                        help="Correlation run id (default: timestamped).")
    parser.add_argument("--correlation-config", default=DEFAULT_CORRELATION_CONFIG,
                        help="Correlation rules/scoring config "
                             "(default: configs/correlation.yaml).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: "
                             "outputs/correlation/<correlation_id>/).")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any requested engine yields no signals.")
    parser.add_argument("--syslog-run", default=None,
                        help="Syslog ingestion run id under "
                             "outputs/syslog_ingestion/, or 'latest'. Optional.")
    parser.add_argument("--skip-syslog", action="store_true",
                        help="Do not load syslog evidence even if a run exists.")
    parser.add_argument("--require-syslog", action="store_true",
                        help="Fail if syslog evidence is requested but unavailable.")
    parser.add_argument("--syslog-config", default=None,
                        help="Optional path to a syslog ingestion config (unused "
                             "for loading; reserved for future overrides).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` on no usable signals / strict failure)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    if not (args.engine_a_dataset or args.engine_b_dataset
            or args.engine_c_snapshot or args.syslog_run or args.require_syslog):
        logger.error("At least one input is required "
                     "(--engine-a-dataset / --engine-b-dataset / "
                     "--engine-c-snapshot / --syslog-run).")
        return 1

    corr_config = load_yaml(args.correlation_config)
    if not bool((corr_config.get("global") or {}).get("enabled", True)):
        logger.warning("Correlation is disabled in %s (global.enabled=false); "
                       "nothing to do.", args.correlation_config)
        return 0

    results = _load_all(args, ctx, corr_config)
    signals = [s for r in results for s in r.signals]
    for r in results:
        for warning in r.warnings:
            logger.warning("[%s] %s", r.engine, warning)

    # --- optional syslog evidence (backward-compatible; only on request) ---
    syslog_result, syslog_meta = _maybe_load_syslog(args, ctx, corr_config)
    if syslog_result is not None:
        for warning in syslog_result.warnings:
            logger.warning("[syslog] %s", warning)
        if args.require_syslog and not syslog_result.signals:
            logger.error("--require-syslog set but no syslog signals were loaded "
                         "(run --syslog-run must resolve to a valid ingestion "
                         "run). Ingest first: python -m "
                         "scripts.ingest_switch_syslog --input-dir <dir> "
                         "--run-id <id>.")
            return 1
        signals.extend(syslog_result.signals)

    if args.strict:
        empty = [r.engine for r in results if not r.signals]
        if empty:
            logger.error("Strict mode: no signals from requested engine(s): %s.",
                         ", ".join(empty))
            return 1

    if not signals:
        logger.error("No usable signals were found in the provided artefacts; "
                     "nothing to correlate.")
        return 1

    correlation_id = args.correlation_id or _default_id()
    sources = {r.engine: r.source for r in results}
    if syslog_result is not None:
        sources[syslog_result.engine] = syslog_result.source
    result = correlate(signals, corr_config, correlation_id, sources, syslog_meta)

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(ctx.paths.correlation_dir) / correlation_id)
    write_correlation(result, out_dir)

    s = result.summary
    logger.info("Correlation '%s' complete: %d signal(s), %d incident(s) "
                "(%d multi-engine) — offline, no commands executed. Output: %s",
                correlation_id, s.total_signals, s.total_incidents,
                s.multi_engine_incident_count, out_dir)
    return 0


def _load_all(args: argparse.Namespace, ctx, corr_config) -> list[LoadResult]:
    """Load signals from each requested engine (read-only)."""
    results: list[LoadResult] = []
    if args.engine_c_snapshot:
        snapshot_dir = Path(ctx.paths.network_config_dir) / args.engine_c_snapshot
        results.append(load_engine_c_signals(
            snapshot_dir, args.engine_c_snapshot, corr_config))
    if args.engine_b_dataset:
        results.append(load_engine_b_signals(
            ctx.paths.network_health_dir, args.engine_b_dataset, corr_config))
    if args.engine_a_dataset:
        results.append(load_engine_a_signals(
            ctx.paths.experiments_dir, ctx.paths.registry_dir,
            ctx.paths.error_analysis_dir, args.engine_a_dataset, corr_config))
    # Stable engine ordering regardless of CLI argument order.
    order = {ENGINE_A: 0, ENGINE_B: 1, ENGINE_C: 2}
    results.sort(key=lambda r: order.get(r.engine, 9))
    return results


def _maybe_load_syslog(args: argparse.Namespace, ctx, corr_config):
    """Load syslog evidence when requested. Returns ``(LoadResult|None, meta)``.

    Syslog is opt-in: it loads only when ``--syslog-run`` is given (or
    ``--require-syslog``). Existing commands without those flags behave exactly
    as before. ``latest`` resolves the newest valid ingestion run.
    """
    if args.skip_syslog:
        return None, None
    if not (args.syslog_run or args.require_syslog):
        return None, None
    if not bool((corr_config.get("syslog") or {}).get("enabled", True)):
        logger.warning("Syslog correlation is disabled in the config "
                       "(syslog.enabled=false); skipping syslog evidence.")
        return None, None

    run = args.syslog_run or str((corr_config.get("syslog") or {}).get(
        "default_run", "latest"))
    syslog_dir = ctx.paths.outputs_dir / "syslog_ingestion"
    return load_syslog_signals(syslog_dir, run, corr_config)


def _default_id() -> str:
    return "correlation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


if __name__ == "__main__":
    sys.exit(main())

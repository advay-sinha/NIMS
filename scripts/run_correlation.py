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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` on no usable signals / strict failure)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    if not (args.engine_a_dataset or args.engine_b_dataset
            or args.engine_c_snapshot):
        logger.error("At least one engine input is required "
                     "(--engine-a-dataset / --engine-b-dataset / "
                     "--engine-c-snapshot).")
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
    result = correlate(signals, corr_config, correlation_id, sources)

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


def _default_id() -> str:
    return "correlation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


if __name__ == "__main__":
    sys.exit(main())

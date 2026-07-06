"""Entry point: convert a raw network-health dataset to the canonical schema.

Reads a registered dataset's raw files (config-driven, see
``network_health.datasets``), runs its adapter and writes a canonical
telemetry CSV plus an adapter report. The output CSV feeds the existing
validation / preprocessing / training scripts unchanged.

Usage
-----
    python -m scripts.prepare_network_health_dataset --dataset snmp_mib_2016
    python -m scripts.prepare_network_health_dataset --dataset lcore_d --inspect
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_health.artifacts import write_canonical_dataset
from src.network_health.dataset_registry import (
    get_dataset,
    inspect_dataset,
    run_adapter,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Convert a raw network-health dataset to canonical CSV."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--dataset", required=True,
        help="Registered dataset id (see network_health.datasets).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override the dataset's configured output_path.",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Probe the raw files and report inferred column roles only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success, ``1`` on a resolvable failure)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    try:
        definition = get_dataset(ctx.config, args.dataset)
    except (KeyError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.inspect:
        try:
            report = inspect_dataset(definition)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Inspection of dataset '%s':", args.dataset)
        for line in json.dumps(report, indent=2).splitlines():
            logger.info("%s", line)
        return 0

    output_path = args.output or definition.output_path
    if output_path is None:
        logger.error(
            "Dataset '%s' has no output_path; pass --output or set it in "
            "config.", args.dataset,
        )
        return 1

    try:
        result = run_adapter(definition)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    paths = write_canonical_dataset(
        result, output_path, ctx.paths.network_health_dir, args.dataset
    )
    logger.info(
        "Wrote %d canonical row(s) to %s.", result.report.n_rows, paths["csv"]
    )
    logger.info("Adapter report: %s | %s", paths["json"], paths["markdown"])
    for warning in result.report.warnings:
        logger.info("  warning: %s", warning)
    return 0


if __name__ == "__main__":
    sys.exit(main())

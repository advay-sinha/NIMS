"""Entry point: validate network-health telemetry against its schema.

Usage
-----
    python -m scripts.validate_network_health
    python -m scripts.validate_network_health --source path/to/telemetry.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_health.artifacts import write_validation_report
from src.network_health.loader import load_telemetry
from src.network_health.schema import TelemetrySchema
from src.network_health.validation import validate_telemetry

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Validate network-health telemetry CSVs."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--source", default=None,
        help="Telemetry CSV/directory (defaults to network_health.source_path).",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Registered dataset id; resolves its canonical CSV (see "
             "network_health.datasets).",
    )
    parser.add_argument(
        "--device", action="append", default=None,
        help="Restrict to a device id (repeatable).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` when validation passes, ``1`` otherwise)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    cfg = dict(ctx.config.get("network_health") or {})
    schema = TelemetrySchema.from_config(cfg.get("schema") or {})
    dataset_id = str(cfg.get("dataset_id", "telemetry"))

    if args.dataset:
        from src.network_health.dataset_registry import resolve_pipeline_source

        try:
            resolved, dataset_id = resolve_pipeline_source(ctx.config, args.dataset)
        except (KeyError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        source = resolved
    else:
        source = Path(args.source or cfg.get("source_path", ""))
    try:
        frame = load_telemetry(
            source, device_column=schema.device_column,
            device_filter=args.device,
        )
    except (FileNotFoundError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    report = validate_telemetry(frame, schema, dataset_id)
    paths = write_validation_report(report, Path(ctx.paths.network_health_dir))
    logger.info("Reports: %s | %s", paths["json"], paths["markdown"])
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

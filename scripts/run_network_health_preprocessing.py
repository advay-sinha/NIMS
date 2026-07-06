"""Entry point: preprocess network-health telemetry into chronological splits.

Usage
-----
    python -m scripts.run_network_health_preprocessing
    python -m scripts.run_network_health_preprocessing --source telemetry.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_health.artifacts import write_processed_splits
from src.network_health.loader import load_telemetry
from src.network_health.preprocessing import preprocess_telemetry
from src.network_health.schema import TelemetrySchema

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess network-health telemetry (chronological splits)."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    cfg = dict(ctx.config.get("network_health") or {})
    schema = TelemetrySchema.from_config(cfg.get("schema") or {})
    dataset_id = str(cfg.get("dataset_id", "telemetry"))

    if args.dataset:
        from src.network_health.dataset_registry import resolve_pipeline_source

        try:
            source, dataset_id = resolve_pipeline_source(ctx.config, args.dataset)
        except (KeyError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
    else:
        source = Path(args.source or cfg.get("source_path", ""))
    try:
        frame = load_telemetry(source, device_column=schema.device_column)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    result = preprocess_telemetry(
        frame, schema, dict(cfg.get("preprocessing") or {}), dataset_id
    )
    paths = write_processed_splits(
        result, Path(ctx.paths.network_health_dir), dataset_id
    )
    logger.info("Manifest: %s", paths["manifest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

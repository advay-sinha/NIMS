"""Entry point: generate the model validation / benchmark report (Phase 4.1).

Aggregates the latest experiment manifest per (dataset, model) from
``outputs/experiments/`` into ``outputs/reports/model_validation_report.md``.
Read-only over experiments; no model is loaded or trained.

Usage
-----
    python -m scripts.generate_validation_report
    python -m scripts.generate_validation_report --analysis outputs/reports/model_validation_analysis.md
    python -m scripts.generate_validation_report --output custom/path.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.training.reporting import (
    build_validation_report,
    collect_latest_manifests,
    count_manifests,
)

logger = logging.getLogger(__name__)

DEFAULT_REPORT_NAME = "model_validation_report.md"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Generate the NetSentinel model validation report."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--output", default=None,
        help=f"Report path (defaults to <reports_dir>/{DEFAULT_REPORT_NAME}).",
    )
    parser.add_argument(
        "--analysis", default=None,
        help="Optional Markdown file appended to the report "
             "(bottlenecks, recommendations).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success, ``1`` when no experiments were found.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    latest = collect_latest_manifests(Path(ctx.paths.experiments_dir))
    if not latest:
        logger.error("No experiment manifests found under %s.",
                     ctx.paths.experiments_dir)
        return 1

    analysis = None
    if args.analysis:
        analysis_path = Path(args.analysis)
        if not analysis_path.is_file():
            logger.error("Analysis file not found: %s", analysis_path)
            return 1
        analysis = analysis_path.read_text(encoding="utf-8")

    total = count_manifests(Path(ctx.paths.experiments_dir))
    report = build_validation_report(latest, analysis, total_experiments=total)
    output = (
        Path(args.output) if args.output
        else Path(ctx.paths.reports_dir) / DEFAULT_REPORT_NAME
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    runs = sum(len(models) for models in latest.values())
    logger.info("Validation report written: %s (%d dataset(s), %d run(s)).",
                output, len(latest), runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

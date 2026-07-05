"""Entry point: render visualization plots for a completed run.

Locates the latest (or a named) completed experiment and renders plots from
its persisted error-analysis and explainability artefacts. No predictions,
SHAP values or training are recomputed. If an upstream artefact is missing,
its plot is skipped (and recorded in metadata) — run
``scripts.run_error_analysis`` / ``scripts.run_explainability`` first to
produce the sources.

Usage
-----
    python -m scripts.run_visualizations --dataset unsw_nb15 --model xgboost
    python -m scripts.run_visualizations --dataset nsl_kdd --model xgboost \
        --run-id nsl_kdd_xgboost_20260703T185430
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from scripts.run_explainability import _resolve_run_dir
from src.visualization.reporting import visualization_summary
from src.visualization.runner import generate_visualizations

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Render visualization plots for a completed experiment."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. nsl_kdd).")
    parser.add_argument("--model", required=True, help="Model id (e.g. xgboost).")
    parser.add_argument(
        "--run-id", default=None,
        help="Experiment id to visualize (defaults to the newest run).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success (including partially skipped plots), ``1`` when the
        experiment is missing, visualization is disabled, or no plot could
        be generated at all.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    if not ctx.config.get("visualization", {}).get("enabled", True):
        logger.error("Visualization is disabled in configs/visualization.yaml.")
        return 1

    try:
        run_dir = _resolve_run_dir(
            Path(ctx.paths.experiments_dir), args.dataset, args.model, args.run_id
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    result = generate_visualizations(
        run_dir.name,
        dataset_id=args.dataset,
        model_type=args.model,
        error_analysis_dir=Path(ctx.paths.error_analysis_dir),
        explainability_dir=Path(ctx.paths.explainability_dir),
        output_root=Path(ctx.paths.visualizations_dir),
        config=ctx.config,
        preprocessing_dir=Path(ctx.paths.preprocessing_dir),
    )

    for line in visualization_summary(result).splitlines():
        if line.strip():
            logger.info("%s", line)
    if not result["plots"]:
        logger.error(
            "No plots could be generated; run scripts.run_error_analysis and "
            "scripts.run_explainability first."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

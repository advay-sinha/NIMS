"""Entry point: generate error-analysis artefacts for a completed run.

Loads an already-trained model and the feature-engineered split from disk —
nothing is retrained — predicts, and writes the error-analysis artefacts to
``outputs/error_analysis/<experiment_id>/``.

Usage
-----
    python -m scripts.run_error_analysis --dataset unsw_nb15 --model xgboost
    python -m scripts.run_error_analysis --dataset nsl_kdd --model xgboost \
        --run-id nsl_kdd_xgboost_20260703T185430
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from scripts.run_explainability import _resolve_run_dir
from src.error_analysis.analyzer import analyze_model
from src.error_analysis.reporting import error_analysis_summary
from src.models.base import BaseModel
from src.training.trainer import _load_xy
from src.utils.config import load_dataset_config
from src.utils.io import read_parquet

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Generate error-analysis artefacts for a trained model."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. nsl_kdd).")
    parser.add_argument("--model", required=True, help="Model id (e.g. xgboost).")
    parser.add_argument(
        "--run-id", default=None,
        help="Experiment id to analyse (defaults to the newest run).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any expected failure (missing run/split).
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    config = ctx.config

    try:
        run_dir = _resolve_run_dir(
            Path(ctx.paths.experiments_dir), args.dataset, args.model, args.run_id
        )
        model = BaseModel.load(run_dir / "model.joblib")

        split = str(config.get("error_analysis", {}).get("split", "test"))
        label_column = load_dataset_config(args.dataset).get("label_column")
        split_path = Path(ctx.paths.features_out_dir) / args.dataset / f"{split}.parquet"
        if not split_path.is_file():
            raise FileNotFoundError(f"Feature split not found: {split_path}")
        x, y = _load_xy(read_parquet(split_path), str(label_column))

        artefacts, analysis = analyze_model(
            model, x, y,
            experiment_id=run_dir.name,
            dataset_id=args.dataset,
            config=config,
            output_root=Path(ctx.paths.error_analysis_dir),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # Log the reporting summary so the operator sees the headline immediately.
    summary = error_analysis_summary(analysis, artefacts["metadata"].parent)
    for line in summary.splitlines():
        if line.strip():
            logger.info("%s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

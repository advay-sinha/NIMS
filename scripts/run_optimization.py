"""Entry point: hyperparameter optimization for one (dataset, model) pair.

Runs a seeded Optuna study over the model's search space, evaluating every
trial on the VALIDATION split (the test split is never used for tuning).
Study artefacts land in ``outputs/optimization/<study_id>/``; unless
``--no-final-train`` is given, one final model is trained with the best
parameters through the standard experiment pipeline, with the optimization
provenance recorded in its manifest.

Usage
-----
    python -m scripts.run_optimization --dataset unsw_nb15 --model xgboost --n-trials 10
    python -m scripts.run_optimization --dataset nsl_kdd --model lightgbm \
        --metric f1_macro --sampler random --seed 7 --no-final-train
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import add_common_arguments, bootstrap
from src.optimization.runner import run_optimization
from src.optimization.search_spaces import OptimizationError, supported_models
from src.utils.config import deep_merge

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for a registered model."
    )
    add_common_arguments(parser)
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. nsl_kdd).")
    parser.add_argument(
        "--model", required=True,
        help=f"Model id to tune ({', '.join(supported_models())}).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Trials to run (defaults to optimization.n_trials).",
    )
    parser.add_argument(
        "--metric", default=None,
        help="Validation metric to optimize (defaults to optimization.metric).",
    )
    parser.add_argument(
        "--sampler", default=None, choices=["tpe", "random"],
        help="Optuna sampler (defaults to optimization.sampler).",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Wall-clock cap in seconds for the study.",
    )
    parser.add_argument(
        "--no-final-train", action="store_true",
        help="Skip training the final best-params model.",
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    """Collect the optimization-block overrides from parsed CLI arguments."""
    overrides: dict = {}
    if args.n_trials is not None:
        overrides["n_trials"] = args.n_trials
    if args.metric is not None:
        overrides["metric"] = args.metric
    if args.sampler is not None:
        overrides["sampler"] = args.sampler
    if args.timeout is not None:
        overrides["timeout"] = args.timeout
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.no_final_train:
        overrides["train_final_best_model"] = False
    return overrides


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` on success, ``1`` on any expected failure (unsupported model,
        missing artefacts, every trial failed).
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    config = deep_merge(ctx.config, {"optimization": _overrides(args)})

    try:
        result = run_optimization(args.dataset, args.model, config, ctx.paths)
    except (OptimizationError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    summary_path = result.artifact_paths["summary"]
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("```"):
            logger.info("%s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

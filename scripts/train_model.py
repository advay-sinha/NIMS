"""Entry point: train Engine A intrusion-detection models (Phase 4).

For each (dataset, model) this loads the feature-engineered splits, fits the
model on the training split (GPU when available), evaluates on validation/test,
and writes the model, metrics and a manifest to a unique experiment directory.

Usage
-----
    python -m scripts.train_model --dataset nsl_kdd                 # active_model
    python -m scripts.train_model --dataset nsl_kdd --model xgboost
    python -m scripts.train_model --dataset nsl_kdd --all-models
    python -m scripts.train_model --all-datasets --all-models

This launches model training and is intended to be run by the human operator
(CLAUDE.md > Human-in-the-Loop).
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import RuntimeContext, add_common_arguments, bootstrap
from src.data.registry import available_datasets
from src.models.registry import available_models
from src.training.trainer import train_model

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Train NetSentinel Engine A models.")
    add_common_arguments(parser)
    ds_group = parser.add_mutually_exclusive_group(required=True)
    ds_group.add_argument("--dataset", help="Dataset id to train on (e.g. nsl_kdd).")
    ds_group.add_argument(
        "--all-datasets", action="store_true",
        help="Train on every dataset in data.active_datasets.",
    )
    parser.add_argument(
        "--model", choices=available_models(),
        help="Model id to train (defaults to training.active_model).",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Train every model in training.models.",
    )
    return parser


def _select_datasets(args: argparse.Namespace, ctx: RuntimeContext) -> list[str]:
    """Resolve dataset ids from CLI + config."""
    if args.dataset:
        return [args.dataset]
    configured = ctx.config.get("data", {}).get("active_datasets")
    return list(configured) if configured else available_datasets()


def _select_models(args: argparse.Namespace, ctx: RuntimeContext) -> list[str]:
    """Resolve model ids from CLI + config."""
    training_cfg = ctx.config.get("training", {})
    if args.model:
        return [args.model]
    if args.all_models:
        return list(training_cfg.get("models", [])) or available_models()
    return [training_cfg.get("active_model", "xgboost")]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` if at least one run succeeded and none failed; ``1`` otherwise.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    exit_code = 0
    succeeded = 0
    for dataset_id in _select_datasets(args, ctx):
        for model_name in _select_models(args, ctx):
            try:
                result = train_model(dataset_id, model_name, ctx.config, ctx.paths)
            except FileNotFoundError as exc:
                logger.warning("[%s/%s] skipped: %s", dataset_id, model_name, exc)
                continue
            except (KeyError, ValueError) as exc:
                logger.error("[%s/%s] training failed: %s", dataset_id, model_name, exc)
                exit_code = 1
                continue
            succeeded += 1
            val = result.metrics.get("validation") or result.metrics.get("train", {})
            logger.info(
                "[%s/%s] done -> %s | F1=%.4f acc=%.4f",
                dataset_id, model_name, result.experiment_id,
                val.get("f1", 0.0), val.get("accuracy", 0.0),
            )

    if succeeded == 0:
        logger.error("No models were trained.")
        return 1

    logger.info("Training complete: %d run(s), exit %d", succeeded, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

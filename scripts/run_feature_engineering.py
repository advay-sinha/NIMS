"""Entry point: run the Phase 3 feature-engineering pipeline.

For each selected dataset this loads the Phase 2 processed splits, fits the
feature selectors (variance, correlation, statistical selection, optional PCA)
on the training split ONLY, transforms every split, and persists the
transformed datasets, JSON reports and fitted artefacts (namespaced per
dataset).

Usage
-----
    python -m scripts.run_feature_engineering --all
    python -m scripts.run_feature_engineering --dataset nsl_kdd

Read-only with respect to processed data. Performs NO model training (Phase 3).
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import RuntimeContext, add_common_arguments, bootstrap
from src.data.registry import available_datasets
from src.features.pipeline import run_feature_pipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Run NetSentinel feature engineering.")
    add_common_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset id to process (e.g. nsl_kdd).")
    group.add_argument(
        "--all",
        action="store_true",
        help="Process every dataset in data.active_datasets.",
    )
    return parser


def _select_datasets(args: argparse.Namespace, ctx: RuntimeContext) -> list[str]:
    """Resolve the dataset ids to process from CLI + config."""
    if args.dataset:
        return [args.dataset]
    configured = ctx.config.get("data", {}).get("active_datasets")
    return list(configured) if configured else available_datasets()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` if at least one dataset was processed and none failed; ``1`` if
        any dataset errored or nothing was processed.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    exit_code = 0
    processed = 0
    for dataset_id in _select_datasets(args, ctx):
        try:
            result = run_feature_pipeline(dataset_id, ctx.config, ctx.paths)
        except FileNotFoundError as exc:
            logger.warning("[%s] skipped (no processed data): %s", dataset_id, exc)
            continue
        except (ValueError, KeyError) as exc:
            logger.error("[%s] feature engineering failed: %s", dataset_id, exc)
            exit_code = 1
            continue
        processed += 1
        logger.info(
            "[%s] feature engineering OK: %d -> %d feature(s)%s.",
            dataset_id, result.n_original, result.n_retained,
            " (PCA)" if result.pca_enabled else "",
        )

    if processed == 0:
        logger.error("No datasets were processed.")
        return 1

    logger.info("Feature engineering complete: %d dataset(s), exit %d", processed, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

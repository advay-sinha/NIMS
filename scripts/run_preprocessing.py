"""Entry point: run the Phase 2 preprocessing pipeline.

For each selected dataset this loads the raw frame, validates, cleans, splits,
fits encoders/scalers on train only, transforms every split, and persists the
processed datasets, fitted artefacts, per-stage reports and a manifest under
the configured output directories (namespaced per dataset).

Usage
-----
    python -m scripts.run_preprocessing --all
    python -m scripts.run_preprocessing --dataset nsl_kdd

Read-only with respect to raw data. Performs NO model training (Phase 2).
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import RuntimeContext, add_common_arguments, bootstrap
from src.data.pipeline import run_pipeline
from src.data.registry import available_datasets

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Run NetSentinel preprocessing.")
    add_common_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset id to preprocess (e.g. nsl_kdd).")
    group.add_argument(
        "--all",
        action="store_true",
        help="Preprocess every dataset in data.active_datasets.",
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
            result = run_pipeline(dataset_id, ctx.config, ctx.paths)
        except NotImplementedError:
            logger.warning("[%s] loader not implemented yet — skipping.", dataset_id)
            continue
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
            logger.error("[%s] preprocessing failed: %s", dataset_id, exc)
            exit_code = 1
            continue
        processed += 1
        logger.info(
            "[%s] preprocessing OK -> %d artefact(s).",
            dataset_id,
            len(result.output_paths),
        )

    if processed == 0:
        logger.error("No datasets were preprocessed.")
        return 1

    logger.info(
        "Preprocessing complete: %d dataset(s), exit code %d", processed, exit_code
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

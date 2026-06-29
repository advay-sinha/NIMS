"""Entry point: run the data pipeline for one or all datasets.

Usage
-----
    python -m scripts.prepare_data --dataset nsl_kdd
    python -m scripts.prepare_data --all

Delegates to :func:`src.data.pipeline.run_pipeline` / ``run_all``. This script
performs NO training (Phase 1; CLAUDE.md > Human-in-the-Loop Training Policy).
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import add_common_arguments, bootstrap

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Run the NetSentinel data pipeline.")
    add_common_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset id to process (e.g. nsl_kdd).")
    group.add_argument(
        "--all",
        action="store_true",
        help="Process every dataset in data.active_datasets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        Process exit code (0 success, non-zero on failure).
    """
    args = build_parser().parse_args(argv)
    _ = bootstrap(args)
    # TODO(data-engineer): call run_all(...) when args.all else run_pipeline(
    #   args.dataset, ...); log a per-dataset summary; return exit code.
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())

"""Entry point: rebuild the experiment index CSV from run manifests.

Regenerates ``<experiments_dir>/experiment_index.csv`` with one row per run,
derived from every ``manifest.json`` on disk. Use it to backfill runs recorded
before the index existed or to repair a diverged index; new runs append their
own row automatically.

Usage
-----
    python -m scripts.build_experiment_index
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.training.experiment_index import rebuild_index

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Rebuild the NetSentinel experiment index CSV."
    )
    add_common_arguments(parser)
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

    index_path = rebuild_index(Path(ctx.paths.experiments_dir))
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    if rows == 0:
        logger.error("No experiment manifests found under %s.",
                     ctx.paths.experiments_dir)
        return 1
    logger.info("Experiment index ready: %s (%d run(s)).", index_path, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

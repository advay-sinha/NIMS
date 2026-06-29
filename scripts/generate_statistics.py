"""Entry point: generate the Phase 1 data report (statistics + metadata).

Usage
-----
    python -m scripts.generate_statistics --all

For each dataset, computes :mod:`src.data.statistics` and persists a statistics
JSON plus a dataset metadata record (:mod:`src.data.metadata`) under
``paths.metadata_dir`` / ``paths.reports_dir``. Read-only profiling; no training.
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import add_common_arguments, bootstrap

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics and metadata."
    )
    add_common_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset id to profile.")
    group.add_argument("--all", action="store_true", help="Profile all datasets.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    _ = bootstrap(args)
    # TODO(data-engineer): load_raw -> dataset_statistics -> write_json; and
    #   build_metadata -> save_metadata for each selected dataset.
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())

"""Entry point: generate dataset ingestion + validation reports.

Usage
-----
    python -m scripts.generate_data_reports --all
    python -m scripts.generate_data_reports --dataset nsl_kdd

For each selected dataset this loads the raw frame through its registered
loader, runs :func:`src.data.validation.build_report` and writes
``<data_reports_dir>/<id>_report.json`` (e.g. ``outputs/data_reports/``).

Read-only with respect to raw data; performs NO preprocessing or training
(Phase 1; CLAUDE.md > Human-in-the-Loop Training Policy).
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import RuntimeContext, add_common_arguments, bootstrap
from src.data.registry import available_datasets, get_loader_cls
from src.data.validation import DataReport, build_report, save_report
from src.utils.config import load_dataset_config
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Generate dataset ingestion and validation reports."
    )
    add_common_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset id to report on (e.g. nsl_kdd).")
    group.add_argument(
        "--all",
        action="store_true",
        help="Report on every dataset in data.active_datasets.",
    )
    return parser


def _select_datasets(args: argparse.Namespace, ctx: RuntimeContext) -> list[str]:
    """Resolve the list of dataset ids to process from CLI + config."""
    if args.dataset:
        return [args.dataset]
    configured = ctx.config.get("data", {}).get("active_datasets")
    return list(configured) if configured else available_datasets()


def generate_one(dataset_id: str, ctx: RuntimeContext) -> DataReport | None:
    """Load, profile and persist a report for a single dataset.

    Returns ``None`` when the dataset's loader is not yet implemented (the run
    continues with a warning rather than aborting the whole batch).

    Parameters
    ----------
    dataset_id:
        Registered dataset identifier.
    ctx:
        Initialised runtime context (config + paths).

    Returns
    -------
    DataReport | None
    """
    dataset_config = load_dataset_config(dataset_id)
    loader = get_loader_cls(dataset_id)(dataset_config, ctx.paths)

    try:
        raw = loader.load_raw()
    except NotImplementedError:
        logger.warning("[%s] loader not implemented yet — skipping.", dataset_id)
        return None

    report = build_report(raw.frame, dataset_config, label_column=raw.label_column)

    out_dir = ensure_dir(ctx.paths.data_reports_dir)
    out_path = out_dir / f"{dataset_id}_report.json"
    save_report(report, out_path)
    logger.info("[%s] report written to %s", dataset_id, out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` if every processed dataset passed schema validation, else ``1``.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    exit_code = 0
    for dataset_id in _select_datasets(args, ctx):
        try:
            report = generate_one(dataset_id, ctx)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.error("[%s] report generation failed: %s", dataset_id, exc)
            exit_code = 1
            continue
        if report is not None and not report.schema_passed:
            logger.error("[%s] schema validation failed.", dataset_id)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

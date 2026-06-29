"""Entry point: validate raw datasets and write validation reports.

Usage
-----
    python -m scripts.validate_datasets --all
    python -m scripts.validate_datasets --dataset cicids2017

Loads each dataset's raw frame and runs :mod:`src.data.validation`, persisting
a JSON validation report under ``paths.reports_dir``. Read-only; never mutates
raw data.
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import RuntimeContext, add_common_arguments, bootstrap
from src.data.registry import available_datasets, get_loader_cls
from src.data.validation import build_report, save_report
from src.utils.config import load_dataset_config
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

REPORT_SUFFIX = "_validation.json"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Validate NetSentinel datasets.")
    add_common_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset id to validate.")
    group.add_argument("--all", action="store_true", help="Validate all datasets.")
    return parser


def _select_datasets(args: argparse.Namespace, ctx: RuntimeContext) -> list[str]:
    """Resolve the dataset ids to validate from CLI + config."""
    if args.dataset:
        return [args.dataset]
    configured = ctx.config.get("data", {}).get("active_datasets")
    return list(configured) if configured else available_datasets()


def _validate_one(dataset_id: str, ctx: RuntimeContext) -> bool | None:
    """Validate one dataset and persist its report.

    Returns ``True`` if schema validation passed, ``False`` if it failed, or
    ``None`` when the loader is not yet implemented (skipped, non-fatal).
    """
    dataset_config = load_dataset_config(dataset_id)
    loader = get_loader_cls(dataset_id)(dataset_config, ctx.paths)

    try:
        raw = loader.load_raw()
    except NotImplementedError:
        logger.warning("[%s] loader not implemented yet — skipping.", dataset_id)
        return None

    report = build_report(raw.frame, dataset_config, label_column=raw.label_column)

    out_dir = ensure_dir(ctx.paths.reports_dir)
    out_path = out_dir / f"{dataset_id}{REPORT_SUFFIX}"
    save_report(report, out_path)

    n_issues = len(report.validation_issues)
    logger.info(
        "[%s] validation %s (%d issue(s)) -> %s",
        dataset_id,
        "PASSED" if report.schema_passed else "FAILED",
        n_issues,
        out_path,
    )
    return report.schema_passed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        0 if all validations pass, 1 if any dataset fails or errors.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    exit_code = 0
    validated = 0
    for dataset_id in _select_datasets(args, ctx):
        try:
            passed = _validate_one(dataset_id, ctx)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.error("[%s] validation failed: %s", dataset_id, exc)
            exit_code = 1
            continue
        if passed is None:
            continue
        validated += 1
        if not passed:
            exit_code = 1

    if validated == 0:
        logger.error("No datasets were validated.")
        return 1

    logger.info(
        "Validation complete: %d dataset(s), exit code %d", validated, exit_code
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

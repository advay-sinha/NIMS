"""Entry point: run the Phase 1A dataset audit.

For each selected dataset this loads the raw frame once and produces:
  - a per-dataset JSON report   -> ``<data_reports_dir>/<id>_report.json``
  - a fingerprint entry         -> ``<fingerprints_dir>/dataset_fingerprint.json``
  - a section in the audit       -> ``<data_reports_dir>/dataset_audit.md``

Usage
-----
    python -m scripts.run_audit --all
    python -m scripts.run_audit --dataset nsl_kdd

Read-only with respect to raw data. Performs NO preprocessing, feature
engineering or training (Phase 1A).
"""

from __future__ import annotations

import argparse
import logging
import sys

from scripts._bootstrap import RuntimeContext, add_common_arguments, bootstrap
from src.data.audit import render_audit_markdown
from src.data.fingerprint import build_fingerprint
from src.data.registry import available_datasets, get_loader_cls
from src.data.validation import DataReport, build_report, save_report
from src.utils.config import load_dataset_config
from src.utils.io import write_json
from src.utils.paths import ensure_dir

logger = logging.getLogger(__name__)

FINGERPRINT_FILENAME = "dataset_fingerprint.json"
AUDIT_FILENAME = "dataset_audit.md"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(description="Run the NetSentinel dataset audit.")
    add_common_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset id to audit (e.g. nsl_kdd).")
    group.add_argument(
        "--all",
        action="store_true",
        help="Audit every dataset in data.active_datasets.",
    )
    return parser


def _select_datasets(args: argparse.Namespace, ctx: RuntimeContext) -> list[str]:
    """Resolve the dataset ids to process from CLI + config."""
    if args.dataset:
        return [args.dataset]
    configured = ctx.config.get("data", {}).get("active_datasets")
    return list(configured) if configured else available_datasets()


def _audit_one(
    dataset_id: str, ctx: RuntimeContext
) -> tuple[DataReport, dict] | None:
    """Load, profile, validate and fingerprint a single dataset.

    Returns ``None`` if the loader is not yet implemented (run continues).
    """
    dataset_config = load_dataset_config(dataset_id)
    loader = get_loader_cls(dataset_id)(dataset_config, ctx.paths)

    try:
        raw = loader.load_raw()
    except NotImplementedError:
        logger.warning("[%s] loader not implemented yet — skipping.", dataset_id)
        return None

    report = build_report(raw.frame, dataset_config, label_column=raw.label_column)

    report_dir = ensure_dir(ctx.paths.data_reports_dir)
    save_report(report, report_dir / f"{dataset_id}_report.json")

    fingerprint = build_fingerprint(
        dataset_config,
        loader.raw_dir(),
        n_rows=report.n_rows,
        n_features=report.n_features,
    )
    return report, fingerprint


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        ``0`` if every audited dataset passed schema validation, else ``1``.
    """
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    reports: dict[str, DataReport] = {}
    fingerprints: dict[str, dict] = {}
    exit_code = 0

    for dataset_id in _select_datasets(args, ctx):
        try:
            outcome = _audit_one(dataset_id, ctx)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.error("[%s] audit failed: %s", dataset_id, exc)
            exit_code = 1
            continue
        if outcome is None:
            continue
        report, fingerprint = outcome
        reports[dataset_id] = report
        fingerprints[dataset_id] = fingerprint
        if not report.schema_passed:
            logger.error("[%s] schema validation failed.", dataset_id)
            exit_code = 1

    if not reports:
        logger.error("No datasets were audited.")
        return 1

    # Persist the combined fingerprint file.
    fp_dir = ensure_dir(ctx.paths.fingerprints_dir)
    fp_path = fp_dir / FINGERPRINT_FILENAME
    write_json(fingerprints, fp_path)
    logger.info("Fingerprints written to %s", fp_path)

    # Render and persist the Markdown audit.
    audit_md = render_audit_markdown(reports, fingerprints)
    audit_path = ensure_dir(ctx.paths.data_reports_dir) / AUDIT_FILENAME
    audit_path.write_text(audit_md, encoding="utf-8")
    logger.info("Audit report written to %s", audit_path)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

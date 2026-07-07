"""Entry point: consolidated Engine C configuration-intelligence report.

Loads the already-persisted artefacts for one snapshot (and an optional diff)
and writes an operator-facing Markdown report plus a machine summary. Engine C
Phase 7: it reads artefacts only — it never recomputes inventory/topology/
findings, never mutates an artefact, never contacts a device and never executes
a command.

Usage
-----
    python -m scripts.generate_network_config_report --snapshot-id sample_remediation
    python -m scripts.generate_network_config_report \\
        --snapshot-id sample_after --diff-id sample_before__to__sample_after
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_config.intelligence import (
    build_intelligence,
    load_diff_artifacts,
    load_snapshot_artifacts,
)
from src.network_config.intelligence_artifacts import write_intelligence

logger = logging.getLogger(__name__)


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidated network configuration-intelligence report "
                    "(offline, read-only; no command is executed)."
    )
    add_common_arguments(parser)
    parser.add_argument("--snapshot-id", required=True,
                        help="Snapshot id under outputs/network_config/.")
    parser.add_argument("--diff-id", default=None,
                        help="Optional diff id under outputs/network_config/diffs/.")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: the snapshot folder).")
    parser.add_argument("--include-appendix", type=_str2bool, default=True,
                        help="Include the artifact-paths appendix (true/false).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` if the snapshot/inventory is missing)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    root = Path(ctx.paths.network_config_dir)
    snapshot_dir = root / args.snapshot_id

    try:
        artifacts = load_snapshot_artifacts(snapshot_dir, args.snapshot_id)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    for warning in artifacts.warnings:
        logger.warning("[%s] %s", artifacts.snapshot_id, warning)

    diff = None
    if args.diff_id:
        diff = load_diff_artifacts(root / "diffs" / args.diff_id, args.diff_id)
        for warning in diff.warnings:
            logger.warning("[diff:%s] %s", args.diff_id, warning)

    intel = build_intelligence(artifacts, diff)
    out_dir = Path(args.output) if args.output else snapshot_dir
    paths = write_intelligence(intel, out_dir,
                               include_appendix=args.include_appendix)

    logger.info(
        "Intelligence report for '%s': %d finding(s), %d hypothesis(es), "
        "%d action item(s)%s — offline, no commands executed.",
        artifacts.snapshot_id, intel.summary.total_findings,
        intel.summary.root_cause_hypotheses_count,
        intel.summary.operator_action_items_count,
        f", diff '{args.diff_id}'" if diff else "")
    if "report_with_diff" in paths:
        logger.info("Diff-aware report: %s", paths["report_with_diff"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Entry point: offline comparison of two network-config snapshots.

Loads two already-persisted snapshots from ``outputs/network_config/`` and
writes a structured diff plus evidence-based remediation verification under
``outputs/network_config/diffs/<before>__to__<after>/``. Engine C Phase 6:
read-only artefact comparison — it NEVER connects to a device and NEVER
executes a command.

Usage
-----
    python -m scripts.compare_network_snapshots \\
        --before sample_remediation --after sample_after

Optional
--------
    --output-id custom_diff_id
    --include-unchanged
    --strict            (exit non-zero when a verification fails)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_config.diff import SnapshotDiffer, load_snapshot
from src.network_config.diff_artifacts import build_diff_summary, write_diff
from src.network_config.verification import verify_remediation

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Offline diff + verification of two network-config "
                    "snapshots (read-only, no command is executed)."
    )
    add_common_arguments(parser)
    parser.add_argument("--before", required=True,
                        help="Before snapshot id (under outputs/network_config).")
    parser.add_argument("--after", required=True,
                        help="After snapshot id (under outputs/network_config).")
    parser.add_argument("--output-id", default=None,
                        help="Diff output namespace (default <before>__to__<after>).")
    parser.add_argument("--include-unchanged", action="store_true",
                        help="Also record unchanged interfaces/findings.")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any remediation verification fails.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` on missing/invalid inventory)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    root = Path(ctx.paths.network_config_dir)

    try:
        before = load_snapshot(root / args.before, args.before)
        after = load_snapshot(root / args.after, args.after)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    for snap in (before, after):
        for warning in snap.warnings:
            logger.warning("[%s] %s", snap.snapshot_id, warning)

    diff = SnapshotDiffer(include_unchanged=args.include_unchanged).diff(
        before, after)
    verifications = verify_remediation(before, after)

    output_id = args.output_id or f"{args.before}__to__{args.after}"
    out_dir = root / "diffs" / output_id
    write_diff(diff, verifications, out_dir)

    summary = build_diff_summary(diff, verifications)
    logger.info(
        "Diff %s -> %s: %d change(s); findings +%d/-%d; verification "
        "%d passed, %d failed, %d unknown — no commands executed.",
        args.before, args.after, summary["total_changes"],
        summary["findings_new"], summary["findings_resolved"],
        summary["verification_passed"], summary["verification_failed"],
        summary["verification_unknown"])

    if args.strict and summary["verification_failed"] > 0:
        logger.error("Strict mode: %d verification failure(s).",
                     summary["verification_failed"])
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

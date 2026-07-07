"""Entry point: dashboard-ready JSON export for one Engine C snapshot.

Loads the already-persisted artefacts for a snapshot (and an optional diff) and
writes flat, stable, frontend-friendly views under
``outputs/network_config/<snapshot_id>/dashboard/``. Engine C Phase 9: it reads
artefacts only — it never recomputes inventory/topology/findings/remediation,
never runs Batfish, never executes an action, never contacts a device and never
mutates an existing artefact.

Usage
-----
    python -m scripts.export_network_config_dashboard --snapshot-id sample_remediation
    python -m scripts.export_network_config_dashboard \\
        --snapshot-id sample_after --diff-id sample_before__to__sample_after
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_config.dashboard_artifacts import write_dashboard
from src.network_config.dashboard_export import build_dashboard
from src.network_config.intelligence import (
    load_diff_artifacts,
    load_snapshot_artifacts,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Export dashboard-ready JSON views for a network-config "
                    "snapshot (offline, read-only; no command is executed)."
    )
    add_common_arguments(parser)
    parser.add_argument("--snapshot-id", required=True,
                        help="Snapshot id under outputs/network_config/.")
    parser.add_argument("--diff-id", default=None,
                        help="Optional diff id under outputs/network_config/diffs/.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: the snapshot folder).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` if snapshot/inventory is missing)."""
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

    views = build_dashboard(artifacts, diff)
    out_dir = Path(args.output_dir) if args.output_dir else snapshot_dir
    write_dashboard(views, out_dir)

    logger.info(
        "Dashboard export for '%s' complete: %d view(s)%s — offline, no "
        "commands executed.", artifacts.snapshot_id, len(views),
        f", diff '{args.diff_id}'" if diff else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())

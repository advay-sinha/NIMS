"""Entry point: optional Batfish configuration validation (Engine C Phase 8).

Runs the OPTIONAL Batfish adapter for one snapshot and writes external
validation artefacts under ``outputs/network_config/<snapshot_id>/batfish/``.
Batfish is disabled by default; this script never runs automatically from
``analyze_network_config`` and never accesses a live device or executes a
command.

Usage
-----
    python -m scripts.run_batfish_validation --snapshot-id sample_remediation

Optional
--------
    --batfish-config configs/batfish.yaml
    --snapshot-path datasets/samples/network_config/batfish_snapshot
    --fail-if-unavailable      (also forces an attempt; exit non-zero if Batfish
                                cannot be used)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_config.batfish_adapter import (
    STATUS_DISABLED,
    STATUS_SUCCESS,
    load_batfish_config,
    run_batfish_validation,
)
from src.network_config.batfish_artifacts import write_batfish

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Optional Batfish configuration validation (disabled by "
                    "default; no device access, no commands executed)."
    )
    add_common_arguments(parser)
    parser.add_argument("--snapshot-id", required=True,
                        help="Snapshot id (output namespace under "
                             "outputs/network_config/).")
    parser.add_argument("--batfish-config", default="configs/batfish.yaml",
                        help="Path to the Batfish config (configs/batfish.yaml).")
    parser.add_argument("--snapshot-path", default=None,
                        help="Batfish snapshot directory "
                             "(defaults to inputs.snapshot_root).")
    parser.add_argument("--fail-if-unavailable", action="store_true",
                        help="Force an attempt and exit non-zero if Batfish is "
                             "unavailable.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit codes: 0 ok/disabled/skipped; 1 when required."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    try:
        config = load_batfish_config(args.batfish_config)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    config.setdefault("global", {})
    # --fail-if-unavailable is an explicit intent to run Batfish: force enable
    # and honour the strict flag, overriding the disabled-by-default posture.
    if args.fail_if_unavailable:
        config["global"]["enabled"] = True
        config["global"]["fail_if_unavailable"] = True
    fail_if_unavailable = bool(config["global"].get("fail_if_unavailable", False))

    snapshot_path = (args.snapshot_path
                     or (config.get("inputs") or {}).get(
                         "snapshot_root",
                         "datasets/samples/network_config/batfish_snapshot"))

    result = run_batfish_validation(args.snapshot_id, config, snapshot_path)

    if result.status == STATUS_DISABLED:
        logger.info("Batfish validation is disabled in configuration; skipping "
                    "(no pybatfish or Docker required).")
        return 0

    out_dir = Path(ctx.paths.network_config_dir) / args.snapshot_id
    write_batfish(result, out_dir)

    if result.status == STATUS_SUCCESS:
        logger.info(
            "Batfish validation succeeded: %d node(s), %d interface(s), %d L3 "
            "edge(s), %d undefined ref(s), %d finding(s) — external validation "
            "only, no commands executed.", result.node_count,
            result.interface_count, result.l3_edge_count,
            result.undefined_reference_count, len(result.findings))
        return 0

    logger.warning("Batfish validation %s: %s", result.status, result.reason)
    if fail_if_unavailable:
        logger.error("Batfish is unavailable and --fail-if-unavailable was set.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

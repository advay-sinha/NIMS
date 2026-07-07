"""Entry point: dry-run validation of a Phase 4 remediation plan.

Reads ``remediation_plan.json`` from a snapshot directory, safety-validates each
planned action and writes dry-run execution artefacts plus an audit log. This is
Engine C Phase 5: it NEVER connects to a device and NEVER executes a command —
every record is marked ``executed = false`` / ``would_execute = false``.

Usage
-----
    python -m scripts.dry_run_network_actions --snapshot-id sample_remediation

Optional
--------
    --input-dir outputs/network_config/sample_remediation
    --operator advay
    --executor-config configs/network_action_executor.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts._bootstrap import add_common_arguments, bootstrap
from src.network_config.execution_artifacts import write_execution
from src.network_config.executor import (
    DryRunExecutor,
    load_executor_config,
    load_remediation_plan,
    summarise_execution,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Dry-run validation of a remediation plan (read-only, "
                    "no command is executed)."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--snapshot-id", default=None,
        help="Snapshot id (defaults to network_config.snapshot_id).",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Snapshot directory holding remediation_plan.json "
             "(defaults to <network_config_dir>/<snapshot_id>).",
    )
    parser.add_argument(
        "--operator", default=None,
        help="Operator recorded in the audit log (defaults to config).",
    )
    parser.add_argument(
        "--executor-config", default=None,
        help="Executor YAML (defaults to "
             "network_config.action_executor.config_path).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success, ``1`` if the plan is missing/invalid)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)
    cfg = dict(ctx.config.get("network_config") or {})
    ae_cfg = dict(cfg.get("action_executor") or {})

    snapshot_id = args.snapshot_id or str(cfg.get("snapshot_id", "snapshot"))
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(ctx.paths.network_config_dir) / snapshot_id
    plan_path = input_dir / "remediation_plan.json"

    try:
        plan_payload = load_remediation_plan(plan_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    exec_config_path = (args.executor_config or ae_cfg.get("config_path")
                        or "configs/network_action_executor.yaml")
    try:
        exec_config = load_executor_config(exec_config_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not exec_config.get("global", {}).get("enabled", True):
        logger.info("Dry-run executor disabled in configuration; nothing to do.")
        return 0

    result = DryRunExecutor(exec_config, operator=args.operator).execute(
        plan_payload, snapshot_id
    )
    summary = summarise_execution(result)
    write_execution(result, summary, input_dir)

    logger.info(
        "Dry-run execution for '%s' complete: %d action(s) "
        "(%d validated, %d blocked, %d skipped) — NO COMMANDS EXECUTED.",
        snapshot_id, summary.total_actions, summary.validated_actions,
        summary.blocked_actions, summary.skipped_actions,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

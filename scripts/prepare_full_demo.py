"""Entry point: one-command offline preparation of the full-system demo.

Refreshes the Engine C assessment, reuses or conditionally trains Engine A/B
models, discovers/reuses syslog artefacts, runs the unified correlation and the
streaming replay, then validates that the read-only frontend can load every
section — writing a single demo-readiness report.

This is **offline demo preparation only**. It executes only an approved
allowlist of local Python modules (argument arrays, never a shell string). It
never opens SSH/SNMP/syslog listeners, captures packets, contacts a device,
executes remediation, or mutates raw datasets/source artefacts.

Usage
-----
    python -m scripts.prepare_full_demo --dry-run     # inspect the plan
    python -m scripts.prepare_full_demo               # prepare everything
    python -m scripts.prepare_full_demo --skip-training   # reuse existing models
    python -m scripts.prepare_full_demo --launch-dashboard
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from scripts._bootstrap import add_common_arguments, bootstrap
from src.demo import artifacts, planner
from src.demo.models import DemoCommand
from src.demo.runner import DemoRunner
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

DEFAULT_DEMO_CONFIG = "configs/demo.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    p = argparse.ArgumentParser(
        description="Offline one-command full-demo preparation (read-only w.r.t. "
                    "devices; approved local commands only).")
    add_common_arguments(p)
    p.add_argument("--demo-config", default=DEFAULT_DEMO_CONFIG,
                   help="Demo config (default: configs/demo.yaml).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the exact approved commands and execute nothing.")
    p.add_argument("--skip-training", action="store_true",
                   help="Never train; fail readiness if required models missing.")
    p.add_argument("--force-train-engine-a", action="store_true",
                   help="Retrain Engine A even when models exist.")
    p.add_argument("--force-train-engine-b", action="store_true",
                   help="Retrain Engine B even when an experiment exists.")
    p.add_argument("--engine-a-dataset", action="append", default=None,
                   help="Engine A dataset(s) or 'all' (repeatable).")
    p.add_argument("--engine-a-model", default=None, help="Engine A model.")
    p.add_argument("--engine-b-dataset", default=None, help="Engine B dataset.")
    p.add_argument("--engine-c-input-dir", default=None,
                   help="Engine C sample input directory.")
    p.add_argument("--engine-c-snapshot", default=None,
                   help="Engine C snapshot id.")
    p.add_argument("--syslog-run", default=None,
                   help="Syslog run id or 'latest'.")
    p.add_argument("--require-syslog", action="store_true",
                   help="Fail if syslog evidence is unavailable.")
    p.add_argument("--correlation-id", default=None, help="Correlation run id.")
    p.add_argument("--refresh-assessment", action="store_true",
                   help="Force-refresh the Engine C assessment.")
    p.add_argument("--reuse-assessment", action="store_true",
                   help="Reuse the Engine C assessment if it already exists.")
    p.add_argument("--launch-dashboard", action="store_true",
                   help="Launch the read-only dashboard after readiness passes.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue past failures of optional stages only.")
    return p


def _overrides(args: argparse.Namespace) -> dict:
    """Map parsed CLI flags to config overrides (None = fall back to demo.yaml)."""
    ov: dict = {
        "engine_a_datasets": args.engine_a_dataset,
        "engine_a_model": args.engine_a_model,
        "engine_b_dataset": args.engine_b_dataset,
        "engine_c_input_dir": args.engine_c_input_dir,
        "engine_c_snapshot": args.engine_c_snapshot,
        "syslog_run": args.syslog_run,
        "correlation_id": args.correlation_id,
        "dry_run": args.dry_run,
        "skip_training": args.skip_training,
        "force_train_engine_a": args.force_train_engine_a,
        "force_train_engine_b": args.force_train_engine_b,
        "require_syslog": args.require_syslog,
        "reuse_assessment": args.reuse_assessment and not args.refresh_assessment,
        "continue_on_error": args.continue_on_error,
        "launch_dashboard": args.launch_dashboard,
    }
    return ov


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` on success/dry-run; ``1`` on a required failure)."""
    args = build_parser().parse_args(argv)
    ctx = bootstrap(args)

    demo_yaml = load_yaml(args.demo_config)
    config = planner.resolve_config(demo_yaml, _overrides(args))
    stages = planner.build_plan(config, ctx.paths)

    runner = DemoRunner(config, ctx.paths, ctx.paths.root)
    if config.dry_run:
        _print_dry_run(stages)
    roll = runner.run(stages)

    metrics = artifacts.collect_metrics(config, ctx.paths)
    demo_run_id = "demo_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = ctx.paths.outputs_dir / "demo"
    paths = artifacts.write_demo_run(config, stages, roll, metrics, demo_run_id,
                                     out_root)

    _print_summary(config, stages, roll, paths)

    ok = roll.get("all_required_ok", False)
    if config.launch_dashboard and ok and not config.dry_run:
        _launch_dashboard(runner)

    return 0 if (ok or config.dry_run) else 1


def _print_dry_run(stages) -> None:
    logger.info("[demo] DRY RUN — the following approved commands would run "
                "(nothing is executed):")
    for stage in stages:
        for cmd in stage.commands:
            logger.info("  [%s] %s", stage.name, cmd.display)


def _print_summary(config, stages, roll, paths) -> None:
    logger.info("[demo] offline preparation; no device access, no packet "
                "capture, no remediation; approved commands only.")
    for stage in stages:
        logger.info("  %-22s %s", stage.name, stage.status)
    logger.info("Demo report: %s", paths.get("report"))
    logger.info("Latest pointer: %s", paths.get("latest"))
    if roll.get("all_required_ok"):
        logger.info("Frontend ready. Launch: python -m scripts.run_dashboard")
    elif not config.dry_run:
        logger.warning("Demo preparation incomplete — see the report for the "
                       "failed stage(s).")


def _launch_dashboard(runner: DemoRunner) -> None:
    """Launch the read-only dashboard (Streamlit-missing is non-fatal)."""
    logger.info("[demo] launching read-only dashboard (python -m "
                "scripts.run_dashboard)…")
    code, tail = runner._exec(DemoCommand("scripts.run_dashboard", ()))
    if code != 0:
        logger.warning("Dashboard launch returned %d. If Streamlit is not "
                       "installed, install it with `pip install streamlit` and "
                       "re-run `python -m scripts.run_dashboard`. %s", code, tail)


if __name__ == "__main__":
    raise SystemExit(main())

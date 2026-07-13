"""Entry point: run offline Engine A ML workflows in a controlled way.

Selects datasets, models and workflow steps and runs the **existing offline**
Engine A entry points in the correct order, so a user can regenerate artefacts
for a demo and then load them in the dashboard.

Offline-only boundary
---------------------
This orchestrates only local, offline Engine A steps (validate → audit →
preprocess → features → train → reports → explainability/error-analysis/
visualizations → registry → promote → resolve). It never touches live devices,
live traffic, SNMP, SSH, packet capture, firewall logs or remediation.

Training and other Engine A steps can be long-running. Use ``--dry-run`` first
to inspect the exact commands; run without it to execute them locally.

Usage
-----
    python -m scripts.run_offline_ml_workflow --dataset unsw_nb15 --model xgboost
    python -m scripts.run_offline_ml_workflow --dataset all --model all --steps all
    python -m scripts.run_offline_ml_workflow --list
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from src.ml_workflow.planner import (
    DATASETS,
    MODELS,
    STEP_ORDER,
    WorkflowStep,
    build_plan,
)

logger = logging.getLogger(__name__)

_SAFETY_BANNER = (
    "Offline Engine A workflow only — no live devices, SNMP, SSH, packet "
    "capture, firewall logs or remediation are involved.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for this entry point."""
    parser = argparse.ArgumentParser(
        description="Run offline Engine A ML workflows (offline only; no live "
                    "device, traffic or remediation).")
    parser.add_argument("--dataset", action="append", default=[],
                        help="Dataset id or 'all' (repeatable). "
                             f"Valid: {', '.join(DATASETS)}.")
    parser.add_argument("--model", action="append", default=[],
                        help="Model id or 'all' (repeatable). "
                             f"Valid: {', '.join(MODELS)}.")
    parser.add_argument("--steps", default="all",
                        help="Comma-separated workflow steps or 'all'. "
                             f"Valid: {', '.join(STEP_ORDER)}.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without executing anything.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep going if a step fails.")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets, models and steps, then exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``0`` ok; ``1`` on bad selection or a failed step)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list:
        _print_options()
        return 0

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    try:
        plan = build_plan(args.dataset, args.model, steps)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    if not plan:
        logger.error("Empty plan — nothing selected.")
        return 1

    print(_SAFETY_BANNER)
    print(f"\nPlanned steps ({len(plan)}):")
    for i, step in enumerate(plan, 1):
        print(f"  {i:>3}. [{step.step}] {step.display}")

    if args.dry_run:
        print("\nDry run — nothing was executed.")
        return 0

    return _execute(plan, continue_on_error=args.continue_on_error)


def _execute(plan: list[WorkflowStep], *, continue_on_error: bool) -> int:
    """Execute each step as a subprocess; return non-zero if any step fails."""
    failures = 0
    for i, step in enumerate(plan, 1):
        logger.info("[%d/%d] %s", i, len(plan), step.display)
        code = _run_step(step)
        if code != 0:
            failures += 1
            logger.error("Step failed (exit %d): %s", code, step.display)
            if not continue_on_error:
                logger.error("Stopping (use --continue-on-error to proceed).")
                return 1
    if failures:
        logger.error("%d step(s) failed.", failures)
        return 1
    logger.info("All %d step(s) completed.", len(plan))
    return 0


def _run_step(step: WorkflowStep) -> int:
    """Run one workflow step as ``python -m <module> <args>`` (subprocess)."""
    return subprocess.call([sys.executable, "-m", step.module, *step.args])


def _print_options() -> None:
    print(_SAFETY_BANNER)
    print("\nDatasets:", ", ".join(DATASETS), "(or 'all')")
    print("Models:  ", ", ".join(MODELS), "(or 'all')")
    print("Steps:   ", ", ".join(STEP_ORDER), "(or 'all')")


if __name__ == "__main__":
    sys.exit(main())

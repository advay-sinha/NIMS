"""Engine C Phase 4 — remediation artefact persistence.

Writes the dry-run remediation plan alongside the inventory/topology/findings in
``outputs/network_config/<snapshot_id>/``::

    remediation_plan.json
    remediation_plan.md
    remediation_commands.csv
    remediation_summary.json

Plans only — no command is ever executed, and every artefact states so.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import Any

from src.network_config.remediation import RemediationPlan

logger = logging.getLogger(__name__)

_NO_EXEC = "No commands were executed. This is a dry-run plan only."

_COMMAND_FIELDS = [
    "action_id", "finding_id", "rule_id", "device", "interface", "risk_level",
    "requires_confirmation", "dry_run_only", "command", "rollback_command",
    "verification_command",
]


def write_remediation(
    plan: RemediationPlan, summary: dict[str, Any], out_dir: Path
) -> dict[str, Path]:
    """Persist the plan (JSON + Markdown), command CSV and summary."""
    from src.utils.io import write_json

    out = Path(out_dir)
    paths: dict[str, Path] = {}
    paths["plan_json"] = write_json(_plan_payload(plan, summary),
                                    out / "remediation_plan.json")
    paths["summary"] = write_json(summary, out / "remediation_summary.json")
    paths["commands_csv"] = _write_commands_csv(
        out / "remediation_commands.csv", plan
    )
    md_path = out / "remediation_plan.md"
    md_path.write_text(_plan_markdown(plan, summary), encoding="utf-8")
    paths["plan_md"] = md_path
    logger.info("Remediation plan (%d action(s)) written to %s.",
                len(plan.actions), out)
    return paths


def _plan_payload(plan: RemediationPlan, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_id": plan.snapshot_id,
        "generated_at": plan.generated_at,
        "dry_run_only": plan.dry_run_only,
        "requires_confirmation": plan.requires_confirmation,
        "do_not_execute": plan.do_not_execute,
        "notice": _NO_EXEC,
        "summary": summary,
        "actions": [dataclasses.asdict(a) for a in plan.actions],
    }


def _write_commands_csv(path: Path, plan: RemediationPlan) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COMMAND_FIELDS)
        writer.writeheader()
        for action in plan.actions:
            if not action.commands:
                continue
            rollback = action.rollback_commands
            verify = (action.verification_steps[0].command
                      if action.verification_steps else "")
            for index, command in enumerate(action.commands):
                writer.writerow({
                    "action_id": action.action_id,
                    "finding_id": action.finding_id,
                    "rule_id": action.rule_id,
                    "device": action.device or "",
                    "interface": action.interface or "",
                    "risk_level": action.risk_level,
                    "requires_confirmation": str(
                        action.requires_confirmation).lower(),
                    "dry_run_only": str(action.dry_run_only).lower(),
                    "command": command,
                    "rollback_command": (
                        rollback[index] if index < len(rollback) else ""
                    ),
                    "verification_command": verify if index == 0 else "",
                })
    return path


def _plan_markdown(plan: RemediationPlan, summary: dict[str, Any]) -> str:
    lines = [
        f"# Remediation Plan — {plan.snapshot_id}",
        "",
        f"> **{_NO_EXEC}**",
        "",
        f"- Dry-run only: {plan.dry_run_only}",
        f"- Requires confirmation: {plan.requires_confirmation}",
        f"- Do not execute: {plan.do_not_execute}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Findings considered | {summary['total_findings']} |",
        f"| Total actions | {summary['total_actions']} |",
        f"| Command-bearing actions | {summary['command_actions']} |",
        f"| Investigation-only actions | {summary['investigation_actions']} |",
        f"| Blocked actions | {summary['blocked_actions']} |",
        "",
    ]
    if summary["actions_by_risk"]:
        lines += ["Risk breakdown: " + ", ".join(
            f"{risk}={count}"
            for risk, count in sorted(summary["actions_by_risk"].items())
        ), ""]
    lines += ["## Actions", ""]
    for action in plan.actions:
        lines += _action_markdown(action)
    return "\n".join(lines) + "\n"


def _action_markdown(action: Any) -> list[str]:
    header = (f"### {action.action_id} — {action.title} "
              f"({action.severity}/{action.risk_level})")
    lines = [
        header,
        "",
        f"- Rule: `{action.rule_id}` | Finding: `{action.finding_id}`",
        f"- Target: {action.device or 'n/a'}"
        + (f" {action.interface}" if action.interface else ""),
        f"- Status: **{action.status}** | type: `{action.action_type}` | "
        f"requires confirmation: {action.requires_confirmation} | "
        f"dry-run only: {action.dry_run_only}",
    ]
    if action.reason:
        lines.append(f"- Reason: {action.reason}")
    if action.commands:
        lines += ["- Candidate commands (NOT executed):", "", "```"]
        lines += list(action.commands)
        lines += ["```", "- Rollback:", "", "```"]
        lines += list(action.rollback_commands) or ["(none)"]
        lines += ["```"]
    if action.verification_steps:
        label = ("Investigation steps" if action.action_type == "investigation"
                 else "Verification")
        lines.append(f"- {label}:")
        lines += [f"  - `{step.command}` → {step.expected_result}"
                  for step in action.verification_steps]
    lines.append("")
    return lines

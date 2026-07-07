"""Engine C Phase 7 — configuration-intelligence artefact persistence.

Renders the consolidated operator report and its machine summary from a
:class:`~src.network_config.intelligence.ConfigIntelligence` object::

    config_intelligence_report.md
    config_intelligence_summary.json
    config_intelligence_report_with_diff.md   (only when a diff is supplied)

Pure formatting of already-loaded artefacts — nothing here recomputes state,
mutates an artefact, contacts a device or executes a command.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from src.network_config.intelligence import (
    SAFETY_NOTE,
    ConfigIntelligence,
    OperatorActionItem,
    RootCauseHypothesis,
)

logger = logging.getLogger(__name__)

_SAFETY_LINES = [
    "- Offline analysis only — **no commands were executed**.",
    "- Remediation plans are **dry-run** and require **explicit human "
    "confirmation** before any change.",
    "- Dry-run validation is **not** live-device verification.",
    "- All reasoning is deterministic and evidence-based; root-cause "
    "hypotheses use cautious wording.",
]


def write_intelligence(
    intel: ConfigIntelligence, out_dir: Path, include_appendix: bool = True
) -> dict[str, Path]:
    """Persist the summary JSON and Markdown report(s) into the snapshot dir."""
    from src.utils.io import write_json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["summary"] = write_json(dataclasses.asdict(intel.summary),
                                  out / "config_intelligence_summary.json")

    report = _report(intel, include_appendix=include_appendix, with_diff=False)
    report_path = out / "config_intelligence_report.md"
    report_path.write_text(report, encoding="utf-8")
    paths["report"] = report_path

    if intel.diff is not None:
        diff_report = _report(intel, include_appendix=include_appendix,
                              with_diff=True)
        diff_path = out / "config_intelligence_report_with_diff.md"
        diff_path.write_text(diff_report, encoding="utf-8")
        paths["report_with_diff"] = diff_path

    logger.info("Configuration-intelligence report(s) written to %s "
                "(offline; no commands executed).", out)
    return paths


# ------------------------------------------------------------------- report


def _report(intel: ConfigIntelligence, *, include_appendix: bool,
            with_diff: bool) -> str:
    summary = intel.summary
    lines = [
        f"# Configuration Intelligence Report — {summary.snapshot_id}",
        "",
        f"> **No commands were executed.** {SAFETY_NOTE}",
        "",
        f"- Generated: {summary.timestamp}",
        f"- Diff-aware: {with_diff and intel.diff is not None}",
        "",
    ]
    lines += _executive_summary(intel)
    lines += _inventory_overview(intel)
    lines += _topology_overview(intel)
    lines += _findings_section(intel)
    lines += _highest_risk_section(intel)
    lines += _hypotheses_section(intel.hypotheses)
    lines += _remediation_section(intel)
    lines += _audit_section(intel)
    lines += _batfish_section(intel)
    if with_diff and intel.diff is not None:
        lines += _diff_section(intel)
    lines += _action_items_section(intel.action_items)
    lines += _safety_section()
    if include_appendix:
        lines += _appendix(intel, with_diff)
    lines.append("")
    return "\n".join(lines)


def _executive_summary(intel: ConfigIntelligence) -> list[str]:
    s = intel.summary
    return [
        "## Executive Summary", "",
        f"- Findings (open): **{s.total_findings}** across "
        f"{len(s.findings_by_category)} categor(y/ies).",
        f"- Highest-risk issues: **{len(s.top_risks)}** ranked; "
        f"top level: {s.top_risks[0]['risk_level'] if s.top_risks else 'n/a'}.",
        f"- Root-cause hypotheses: **{s.root_cause_hypotheses_count}** "
        "(cautious, evidence-based).",
        f"- Remediation actions: **{s.total_remediation_actions}** "
        f"({s.command_actions} command, {s.investigation_actions} "
        f"investigation, {s.blocked_actions} blocked) — dry-run only.",
        f"- Operator action items: **{s.operator_action_items_count}**.",
        f"- Dry-run audit available: {s.dry_run_available}; diff-aware: "
        f"{s.diff_available}.",
        "",
    ]


def _inventory_overview(intel: ConfigIntelligence) -> list[str]:
    s = intel.summary
    devices = intel.artifacts.inventory.get("devices") or []
    lines = [
        "## Inventory Overview", "",
        "| Metric | Count |", "|---|---|",
        f"| Devices | {len(devices)} |",
        f"| Interfaces | {s.total_interfaces} |",
        f"| VLANs | {s.total_vlans} |",
        f"| Topology edges | {s.total_topology_edges} |",
        "",
    ]
    return lines


def _topology_overview(intel: ConfigIntelligence) -> list[str]:
    topo = intel.artifacts.topology
    if not topo:
        return ["## Topology Overview", "",
                "_No topology artefact available for this snapshot._", ""]
    ts = topo.get("summary") or {}
    return [
        "## Topology Overview", "",
        f"- Nodes: {ts.get('node_count', len(topo.get('nodes') or []))}",
        f"- Edges: {ts.get('edge_count', len(topo.get('edges') or []))}",
        f"- Warnings: {ts.get('warning_count', len(topo.get('warnings') or []))}",
        "",
    ]


def _findings_section(intel: ConfigIntelligence) -> list[str]:
    s = intel.summary
    lines = ["## Configuration Findings", ""]
    if not s.total_findings:
        return lines + ["_No open findings._", ""]
    lines += ["### By severity", ""] + _count_table(s.findings_by_severity,
                                                    "Severity")
    lines += ["", "### By category", ""] + _count_table(s.findings_by_category,
                                                        "Category")
    lines.append("")
    return lines


def _highest_risk_section(intel: ConfigIntelligence) -> list[str]:
    lines = ["## Highest-Risk Issues", ""]
    top = intel.ranked_findings[:10]
    if not top:
        return lines + ["_No scored findings._", ""]
    lines += ["| Risk | Level | Rule | Device | Interface | Title |",
              "|---|---|---|---|---|---|"]
    for finding, risk in top:
        lines.append(
            f"| {risk.risk_score} | {risk.risk_level} | "
            f"{finding.get('rule_id')} | {finding.get('device') or 'n/a'} | "
            f"{finding.get('interface') or '-'} | {finding.get('title')} |")
    lines.append("")
    return lines


def _hypotheses_section(hypotheses: tuple[RootCauseHypothesis, ...]) -> list[str]:
    lines = ["## Root-Cause Hypotheses", ""]
    if not hypotheses:
        return lines + ["_No multi-signal root-cause patterns detected._", ""]
    lines.append("> Hypotheses are conservative and evidence-based; wording "
                 "(`possible`/`candidate`/`likely`) reflects the number of "
                 "corroborating signals.")
    lines.append("")
    for h in hypotheses:
        lines += [
            f"### {h.title} ({h.confidence}, {h.severity})",
            "",
            f"- Affected devices: {', '.join(h.affected_devices) or 'n/a'}",
            f"- Affected interfaces: {', '.join(h.affected_interfaces) or 'n/a'}",
            f"- Explanation: {h.explanation}",
            "- Evidence:",
        ]
        lines += [f"  - {e}" for e in h.evidence]
        lines.append("- Recommended next steps:")
        lines += [f"  - {step}" for step in h.recommended_next_steps]
        lines.append("")
    return lines


def _remediation_section(intel: ConfigIntelligence) -> list[str]:
    s = intel.summary
    lines = ["## Remediation Plan Summary", ""]
    if not intel.artifacts.remediation_plan:
        return lines + ["_No remediation plan artefact for this snapshot._", ""]
    lines += [
        "> Dry-run only — **no command is ever executed**; every command-"
        "bearing action requires explicit human confirmation.",
        "",
        "| Metric | Count |", "|---|---|",
        f"| Total actions | {s.total_remediation_actions} |",
        f"| Command-bearing | {s.command_actions} |",
        f"| Investigation-only | {s.investigation_actions} |",
        f"| Blocked | {s.blocked_actions} |",
        "",
    ]
    return lines


def _audit_section(intel: ConfigIntelligence) -> list[str]:
    exec_sum = intel.artifacts.execution_summary
    if not exec_sum:
        return []
    return [
        "## Dry-Run Audit Summary", "",
        "> Dry-run validation only — **no commands were executed**.", "",
        "| Metric | Count |", "|---|---|",
        f"| Total actions | {exec_sum.get('total_actions', 0)} |",
        f"| Validated | {exec_sum.get('validated_actions', 0)} |",
        f"| Blocked | {exec_sum.get('blocked_actions', 0)} |",
        f"| Skipped | {exec_sum.get('skipped_actions', 0)} |",
        f"| Executed | {exec_sum.get('executed', False)} |",
        "",
    ]


def _batfish_section(intel: ConfigIntelligence) -> list[str]:
    bf = intel.artifacts.batfish
    if not bf:
        return []
    parse = bf.get("parse_status_summary") or {}
    lines = [
        "## Batfish Validation (external, optional)", "",
        "> External configuration validation only — **no device access and no "
        "commands were executed**. Evidence below is from Batfish, not the "
        "offline parsers.", "",
        f"- Status: **{bf.get('status', 'unknown')}**"
        + (f" ({bf.get('reason')})" if bf.get("reason") else ""),
        f"- Nodes: {bf.get('node_count', 0)} | Interfaces: "
        f"{bf.get('interface_count', 0)} | L3 edges: {bf.get('l3_edge_count', 0)}",
        f"- Parse status — passed: {parse.get('passed', 0)}, "
        f"partially parsed: {parse.get('partially_parsed', 0)}, "
        f"failed: {parse.get('failed', 0)}",
        f"- Undefined references: {bf.get('undefined_reference_count', 0)}",
    ]
    findings = bf.get("findings") or []
    if findings:
        lines += ["", "### Batfish findings (external evidence)", ""]
        lines += [
            f"- **{f.get('severity')}** [{f.get('category')}] "
            f"{f.get('title')} — {f.get('device') or 'n/a'}"
            + (f": {f.get('evidence')}" if f.get("evidence") else "")
            for f in findings
        ]
    lines.append("")
    return lines


def _diff_section(intel: ConfigIntelligence) -> list[str]:
    diff = intel.diff
    ds = (diff.diff_summary if diff else None) or {}
    lines = [
        "## Snapshot Diff / Verification Summary", "",
        f"> Comparison `{ds.get('before_snapshot_id', '?')}` → "
        f"`{ds.get('after_snapshot_id', '?')}`; offline only, no commands "
        "executed.", "",
        "| Metric | Count |", "|---|---|",
        f"| Total changes | {ds.get('total_changes', 0)} |",
        f"| Findings new | {ds.get('findings_new', 0)} |",
        f"| Findings resolved | {ds.get('findings_resolved', 0)} |",
        f"| Verification passed | {ds.get('verification_passed', 0)} |",
        f"| Verification failed | {ds.get('verification_failed', 0)} |",
        f"| Verification unknown | {ds.get('verification_unknown', 0)} |",
        "",
    ]
    return lines


def _action_items_section(items: tuple[OperatorActionItem, ...]) -> list[str]:
    lines = ["## Operator Action Items", ""]
    if not items:
        return lines + ["_No action items generated._", ""]
    lines += ["| Priority | Type | Owner | Device | Interface | Next step |",
              "|---|---|---|---|---|---|"]
    for item in items:
        lines.append(
            f"| {item.priority} | {item.action_type} | "
            f"{item.recommended_owner} | {item.device or 'n/a'} | "
            f"{item.interface or '-'} | {item.next_step} |")
    lines.append("")
    return lines


def _safety_section() -> list[str]:
    return ["## Safety Notes", ""] + _SAFETY_LINES + [""]


def _appendix(intel: ConfigIntelligence, with_diff: bool) -> list[str]:
    directory = intel.artifacts.directory
    names = ["inventory.json", "topology.json", "findings.json",
             "rule_summary.json", "remediation_plan.json",
             "remediation_summary.json", "dry_run_execution.json",
             "execution_summary.json"]
    lines = ["## Appendix: Artifact Paths", ""]
    lines += [f"- `{directory}/{name}`" for name in names]
    if with_diff and intel.diff is not None:
        lines += [f"- `diffs/{intel.diff.diff_id}/snapshot_diff.json`",
                  f"- `diffs/{intel.diff.diff_id}/verification_results.json`",
                  f"- `diffs/{intel.diff.diff_id}/diff_summary.json`"]
    lines.append("")
    return lines


def _count_table(counts: dict[str, int], label: str) -> list[str]:
    if not counts:
        return ["_none_"]
    rows = [f"| {label} | Count |", "|---|---|"]
    rows += [f"| {k} | {v} |"
             for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return rows

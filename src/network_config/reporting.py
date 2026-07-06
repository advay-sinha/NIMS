"""Engine C Markdown reporting.

Purpose
-------
Render a human-readable summary of one offline inventory snapshot: devices,
interface status breakdown, VLAN/trunk/neighbor counts, PoE-enabled ports,
STP state distribution and any missing input files. Pure formatting — takes the
inventory and its derived summary, returns Markdown text.
"""

from __future__ import annotations

from typing import Any

from src.network_config.models import NetworkInventory


def network_config_report(
    inventory: NetworkInventory,
    summary: dict[str, Any],
    topology_summary: dict[str, Any] | None = None,
    findings_summary: dict[str, Any] | None = None,
    remediation_summary: dict[str, Any] | None = None,
) -> str:
    """Build the ``network_config_report.md`` text.

    Optional sections are appended in phase order: topology (Phase 2),
    rule findings (Phase 3) and remediation (Phase 4).
    """
    lines: list[str] = [
        f"# Network Configuration Report — {inventory.snapshot_id}",
        "",
        f"- Input directory: `{inventory.input_directory}`",
        f"- Files parsed: {len(inventory.files_parsed)} | "
        f"Files missing: {len(inventory.files_missing)}",
        "",
        "## Devices",
        "",
    ]
    for snap in inventory.devices:
        device = snap.device
        lines.append(
            f"- **{device.device_id}** — hostname: {device.hostname or 'n/a'}, "
            f"platform: {device.platform or 'n/a'}, "
            f"management IP: {device.management_ip or 'n/a'}"
        )
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Devices | {summary['device_count']} |",
        f"| Interfaces | {summary['interface_count']} |",
        f"| Access ports | {summary['access_port_count']} |",
        f"| Trunk ports | {summary['trunk_port_count']} |",
        f"| VLANs | {summary['vlan_count']} |",
        f"| Trunks | {summary['trunk_count']} |",
        f"| Neighbors | {summary['neighbor_count']} |",
        f"| MAC entries | {summary['mac_entry_count']} |",
        f"| PoE-enabled ports | {summary['poe_enabled_port_count']} |",
        f"| Unused/down ports | {len(summary['unused_ports'])} |",
        "",
        "## Interface status",
        "",
    ]
    lines += _count_table(summary["interface_status_counts"], "Status")
    lines += ["", "## Spanning-tree port states", ""]
    lines += _count_table(summary["stp_state_counts"], "State")

    if summary["poe_enabled_ports"]:
        lines += ["", "## PoE-enabled ports", "",
                   ", ".join(summary["poe_enabled_ports"])]
    if summary["unused_ports"]:
        lines += ["", "## Unused/down ports (no description)", "",
                   ", ".join(summary["unused_ports"])]
    if topology_summary is not None:
        lines += _topology_section(topology_summary)
    if findings_summary is not None:
        lines += _findings_section(findings_summary)
    if remediation_summary is not None:
        lines += _remediation_section(remediation_summary)
    if inventory.files_missing:
        lines += ["", "## Missing input files", ""]
        lines += [f"- {name}" for name in inventory.files_missing]
    lines.append("")
    return "\n".join(lines)


def _topology_section(topo: dict[str, Any]) -> list[str]:
    """Render the Phase 2 topology summary section."""
    confidence = topo["confidence_counts"]
    lines = [
        "",
        "## Topology",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Nodes | {topo['node_count']} |",
        f"| Edges | {topo['edge_count']} |",
        f"| Bidirectional edges | {topo['bidirectional_edge_count']} |",
        f"| High-confidence edges | {confidence['high']} |",
        f"| Medium-confidence edges | {confidence['medium']} |",
        f"| Low-confidence edges | {confidence['low']} |",
        f"| LLDP/CDP edges | {topo['lldp_cdp_edge_count']} |",
        f"| Inferred edges | {topo['inferred_edge_count']} |",
        f"| Warnings | {topo['warning_count']} |",
    ]
    if topo["warning_severity_counts"]:
        lines += ["", "### Warnings by severity", ""]
        lines += _count_table(topo["warning_severity_counts"], "Severity")
    if topo["warning_category_counts"]:
        lines += ["", "### Warnings by category", ""]
        lines += _count_table(topo["warning_category_counts"], "Category")
    if topo["top_warnings"]:
        lines += ["", "### Top warnings", ""]
        lines += [f"- **{w['severity']}/{w['category']}** ({w['warning_id']}): "
                  f"{w['message']}" for w in topo["top_warnings"]]
    return lines


def _findings_section(findings: dict[str, Any]) -> list[str]:
    """Render the Phase 3 rule-findings summary section."""
    lines = [
        "",
        "## Rule findings",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Total open findings | {findings['total_findings']} |",
        f"| Suppressed | {findings['suppressed_count']} |",
        f"| Rules evaluated | {len(findings['rules_evaluated'])} |",
        f"| Rules enabled | {len(findings['rules_enabled'])} |",
        f"| Rules disabled | {len(findings['rules_disabled'])} |",
        f"| Rules skipped | {len(findings.get('rules_skipped', []))} |",
    ]
    if findings["findings_by_severity"]:
        lines += ["", "### Findings by severity", ""]
        lines += _count_table(findings["findings_by_severity"], "Severity")
    if findings["findings_by_category"]:
        lines += ["", "### Findings by category", ""]
        lines += _count_table(findings["findings_by_category"], "Category")
    top = findings.get("top_findings") or []
    if top:
        lines += ["", "### Top critical/high findings", ""]
        lines += [
            f"- **{f['severity']}** [{f['rule_id']}] {f['title']} — "
            f"{f.get('device') or 'n/a'}"
            + (f" {f['interface']}" if f.get("interface") else "")
            + (f": {f['evidence']}" if f.get("evidence") else "")
            for f in top
        ]
    if findings.get("rules_skipped"):
        lines += ["", "_Skipped rules (missing inputs): "
                  + ", ".join(findings["rules_skipped"]) + "_"]
    return lines


def _remediation_section(rem: dict[str, Any]) -> list[str]:
    """Render the Phase 4 remediation summary section (dry-run only)."""
    lines = [
        "",
        "## Remediation (dry-run)",
        "",
        "> **No commands were executed.** Plans are dry-run only and require "
        "explicit human confirmation before any change.",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Total actions | {rem['total_actions']} |",
        f"| Command-bearing actions | {rem['command_actions']} |",
        f"| Investigation-only actions | {rem['investigation_actions']} |",
        f"| Blocked actions | {rem['blocked_actions']} |",
    ]
    if rem.get("actions_by_risk"):
        lines += ["", "### Actions by risk", ""]
        lines += _count_table(rem["actions_by_risk"], "Risk")
    top = rem.get("top_actions") or []
    if top:
        lines += ["", "### Top planned actions", ""]
        lines += [
            f"- **{a['risk_level']}** [{a['rule_id']}] {a['title']} — "
            f"{a.get('device') or 'n/a'}"
            + (f" {a['interface']}" if a.get("interface") else "")
            + f" ({a['action_type']})"
            for a in top
        ]
    return lines


def _count_table(counts: dict[str, int], label: str) -> list[str]:
    """Render a ``{key: count}`` mapping as a small Markdown table."""
    if not counts:
        return ["_none_"]
    rows = [f"| {label} | Count |", "|---|---|"]
    rows += [f"| {key} | {value} |"
             for key, value in sorted(counts.items(), key=lambda kv: -kv[1])]
    return rows

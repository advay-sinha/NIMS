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
    inventory: NetworkInventory, summary: dict[str, Any]
) -> str:
    """Build the ``network_config_report.md`` text."""
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
    if inventory.files_missing:
        lines += ["", "## Missing input files", ""]
        lines += [f"- {name}" for name in inventory.files_missing]
    lines.append("")
    return "\n".join(lines)


def _count_table(counts: dict[str, int], label: str) -> list[str]:
    """Render a ``{key: count}`` mapping as a small Markdown table."""
    if not counts:
        return ["_none_"]
    rows = [f"| {label} | Count |", "|---|---|"]
    rows += [f"| {key} | {value} |"
             for key, value in sorted(counts.items(), key=lambda kv: -kv[1])]
    return rows

"""Engine C artefact persistence.

Purpose
-------
Write one offline inventory snapshot to ``outputs/network_config/<snapshot_id>/``:
the full ``inventory.json``, one CSV per object type, a ``metadata.json``
summary and the Markdown report. Read-only analysis in, files out — nothing
here contacts a device.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.network_config.inventory import derive_summary
from src.network_config.models import NetworkInventory
from src.network_config.reporting import network_config_report

logger = logging.getLogger(__name__)

# (filename, dataclass fields) for each per-object CSV, so headers are stable
# even when a table is empty.
_CSV_SPECS: dict[str, tuple[str, Sequence[str]]] = {
    "interfaces": ("interfaces.csv",
                   ["name", "status", "protocol_status", "vlan", "mode",
                    "description", "speed", "duplex", "poe_enabled",
                    "poe_state"]),
    "vlans": ("vlans.csv", ["vlan_id", "name", "status", "ports"]),
    "trunks": ("trunks.csv",
               ["interface", "allowed_vlans", "native_vlan",
                "trunking_status"]),
    "neighbors": ("neighbors.csv",
                  ["local_interface", "remote_device", "remote_interface",
                   "protocol"]),
    "mac_table": ("mac_table.csv",
                  ["vlan", "mac_address", "interface", "entry_type"]),
    "poe_status": ("poe_status.csv",
                   ["interface", "admin_state", "oper_state", "power_watts",
                    "powered_device", "poe_class", "max_watts"]),
    "stp_state": ("stp_state.csv",
                  ["vlan", "interface", "role", "state"]),
}


def write_inventory(
    inventory: NetworkInventory,
    root: Path,
    topology: Any | None = None,
    findings: Any | None = None,
    rule_summary: dict[str, Any] | None = None,
    remediation_plan: Any | None = None,
    remediation_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist a full inventory snapshot; return the written paths by key.

    Optional phase outputs are written alongside the inventory when supplied:
    Phase 2 ``topology`` (``topology.*``), Phase 3 ``findings`` +
    ``rule_summary`` (``findings.*`` / ``rule_summary.json``) and Phase 4
    ``remediation_plan`` + ``remediation_summary`` (``remediation_*``).
    Everything present is summarised in the report and metadata.
    """
    from src.utils.io import write_json
    from src.utils.paths import ensure_dir

    out_dir = ensure_dir(Path(root) / inventory.snapshot_id)
    summary = derive_summary(inventory)
    paths: dict[str, Path] = {}

    paths["inventory"] = write_json(
        _inventory_payload(inventory), out_dir / "inventory.json"
    )

    rows_by_key = {
        "interfaces": inventory.all_interfaces,
        "vlans": inventory.all_vlans,
        "trunks": inventory.all_trunks,
        "neighbors": inventory.all_neighbors,
        "mac_table": inventory.all_mac_entries,
        "poe_status": inventory.all_poe,
        "stp_state": inventory.all_stp_states,
    }
    for key, (filename, fields) in _CSV_SPECS.items():
        paths[key] = _write_csv(out_dir / filename, rows_by_key[key], fields)

    topo_summary = None
    if topology is not None:
        from src.network_config.topology import topology_summary
        from src.network_config.topology_artifacts import write_topology

        topo_summary = topology_summary(topology)
        paths.update(write_topology(topology, out_dir))

    findings_report = None
    if findings is not None and rule_summary is not None:
        from src.network_config.rule_artifacts import write_findings

        paths.update(write_findings(findings, rule_summary, out_dir))
        findings_report = {**rule_summary, "top_findings": _top_findings(findings)}

    remediation_report = None
    if remediation_plan is not None and remediation_summary is not None:
        from src.network_config.remediation_artifacts import write_remediation

        paths.update(write_remediation(remediation_plan, remediation_summary,
                                       out_dir))
        remediation_report = {**remediation_summary,
                              "top_actions": _top_actions(remediation_plan)}

    metadata = _metadata(inventory, summary)
    if topo_summary is not None:
        metadata["topology"] = {
            "node_count": topo_summary["node_count"],
            "edge_count": topo_summary["edge_count"],
            "warning_count": topo_summary["warning_count"],
        }
    if rule_summary is not None:
        metadata["findings"] = {
            "total_findings": rule_summary["total_findings"],
            "suppressed_count": rule_summary["suppressed_count"],
            "rules_evaluated": len(rule_summary["rules_evaluated"]),
        }
    if remediation_summary is not None:
        metadata["remediation"] = {
            "total_actions": remediation_summary["total_actions"],
            "command_actions": remediation_summary["command_actions"],
            "blocked_actions": remediation_summary["blocked_actions"],
            "dry_run_only": remediation_summary["dry_run_only"],
        }
    paths["metadata"] = write_json(metadata, out_dir / "metadata.json")

    report = network_config_report(inventory, summary, topo_summary,
                                   findings_report, remediation_report)
    report_path = out_dir / "network_config_report.md"
    report_path.write_text(report, encoding="utf-8")
    paths["report"] = report_path

    logger.info("Network-config snapshot '%s' written to %s.",
                inventory.snapshot_id, out_dir)
    return paths


def _inventory_payload(inventory: NetworkInventory) -> dict[str, Any]:
    """Full nested, JSON-serialisable inventory."""
    return {
        "snapshot_id": inventory.snapshot_id,
        "input_directory": inventory.input_directory,
        "files_parsed": list(inventory.files_parsed),
        "files_missing": list(inventory.files_missing),
        "warnings": list(inventory.warnings),
        "devices": [dataclasses.asdict(device) for device in inventory.devices],
    }


def _metadata(inventory: NetworkInventory, summary: dict[str, Any]) -> dict[str, Any]:
    """Snapshot metadata block (required fields + a few derived roll-ups)."""
    return {
        "snapshot_id": inventory.snapshot_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_directory": inventory.input_directory,
        "files_parsed": list(inventory.files_parsed),
        "files_missing": list(inventory.files_missing),
        "device_count": summary["device_count"],
        "interface_count": summary["interface_count"],
        "vlan_count": summary["vlan_count"],
        "neighbor_count": summary["neighbor_count"],
        "trunk_count": summary["trunk_count"],
        "poe_enabled_port_count": summary["poe_enabled_port_count"],
    }


def _write_csv(path: Path, rows: Sequence[Any], fields: Sequence[str]) -> Path:
    """Write dataclass rows to a CSV with a fixed header (tuples joined)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            record = dataclasses.asdict(row)
            writer.writerow(
                {f: _cell(record.get(f)) for f in fields}
            )
    return path


def _cell(value: Any) -> Any:
    """Render a CSV cell (join VLAN/port tuples with ``;``)."""
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return "" if value is None else value


def _top_findings(findings: Any, limit: int = 5) -> list[dict[str, Any]]:
    """The most severe open findings for the report (critical/high first)."""
    import dataclasses

    top = [
        dataclasses.asdict(f) for f in findings
        if f.status == "open" and f.severity in {"critical", "high"}
    ]
    return top[:limit]


def _top_actions(plan: Any, limit: int = 5) -> list[dict[str, Any]]:
    """The top planned actions for the report (plan is already severity-sorted)."""
    return [
        {"rule_id": a.rule_id, "title": a.title, "risk_level": a.risk_level,
         "action_type": a.action_type, "device": a.device,
         "interface": a.interface}
        for a in plan.actions if a.status == "planned"
    ][:limit]

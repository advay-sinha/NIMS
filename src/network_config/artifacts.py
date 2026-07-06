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


def write_inventory(inventory: NetworkInventory, root: Path) -> dict[str, Path]:
    """Persist a full inventory snapshot; return the written paths by key."""
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

    paths["metadata"] = write_json(
        _metadata(inventory, summary), out_dir / "metadata.json"
    )
    report = network_config_report(inventory, summary)
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

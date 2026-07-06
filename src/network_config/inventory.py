"""Inventory builder: merge parsed command outputs into one structured view.

Purpose
-------
Read a directory of saved command outputs (offline, read-only), run each
parser, warn on any missing file and continue, then combine the results into a
single :class:`~src.network_config.models.NetworkInventory`. Interface objects
are enriched with PoE state; analytical roll-ups (access/trunk ports, MAC
presence, unused ports, STP state counts) are derived on demand for reporting.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from src.network_config import parsers
from src.network_config.models import (
    NetworkDevice,
    NetworkInterface,
    NetworkInventory,
    ParsedDeviceSnapshot,
)

logger = logging.getLogger(__name__)

# Logical name -> (default filename, parser). ``running_config`` is handled
# separately (it yields device identity, not a list of objects).
_TABLE_PARSERS: dict[str, tuple[str, Callable[[str], list]]] = {
    "interface_status": ("show_interface_status.txt",
                         parsers.parse_interface_status),
    "vlan_brief": ("show_vlan_brief.txt", parsers.parse_vlan_brief),
    "trunk": ("show_interfaces_trunk.txt", parsers.parse_trunk),
    "lldp_neighbors": ("show_lldp_neighbors.txt", parsers.parse_lldp_neighbors),
    "cdp_neighbors": ("show_cdp_neighbors.txt", parsers.parse_cdp_neighbors),
    "mac_table": ("show_mac_address_table.txt", parsers.parse_mac_table),
    "power_inline": ("show_power_inline.txt", parsers.parse_power_inline),
    "spanning_tree": ("show_spanning_tree.txt", parsers.parse_spanning_tree),
}
_RUNNING_CONFIG = ("running_config", "show_running_config.txt")

_POE_ENABLED = {"auto", "static", "on"}
_POE_DISABLED = {"off", "never"}


def _resolve_filenames(config: Mapping[str, Any]) -> dict[str, str]:
    """Merge configured filenames over the built-in defaults."""
    configured = dict(config.get("files") or {})
    names = {key: default for key, (default, _) in _TABLE_PARSERS.items()}
    names[_RUNNING_CONFIG[0]] = _RUNNING_CONFIG[1]
    names.update({k: str(v) for k, v in configured.items()})
    return names


def build_inventory(
    input_dir: str | Path,
    config: Mapping[str, Any],
    snapshot_id: str,
) -> NetworkInventory:
    """Build a network inventory from a directory of saved command outputs.

    Missing files are recorded and warned about; parsing continues with
    whatever is present. The directory itself must exist.
    """
    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    filenames = _resolve_filenames(config)
    parsed: dict[str, list] = {}
    files_parsed: list[str] = []
    files_missing: list[str] = []
    warnings: list[str] = []

    for key, (_, parser) in _TABLE_PARSERS.items():
        path = directory / filenames[key]
        text = _read_optional(path, key, files_missing, warnings)
        parsed[key] = parser(text) if text is not None else []
        if text is not None:
            files_parsed.append(path.name)

    run_path = directory / filenames[_RUNNING_CONFIG[0]]
    run_text = _read_optional(
        run_path, _RUNNING_CONFIG[0], files_missing, warnings
    )
    identity = (
        parsers.parse_running_config(run_text) if run_text is not None else {}
    )
    if run_text is not None:
        files_parsed.append(run_path.name)

    interfaces = _enrich_interfaces(parsed["interface_status"],
                                    parsed["power_inline"])

    hostname = identity.get("hostname")
    device = NetworkDevice(
        device_id=hostname or snapshot_id,
        hostname=hostname,
        platform=identity.get("platform"),
        management_ip=identity.get("management_ip"),
        source_files=tuple(files_parsed),
    )
    snapshot = ParsedDeviceSnapshot(
        device=device,
        interfaces=tuple(interfaces),
        vlans=tuple(parsed["vlan_brief"]),
        trunks=tuple(parsed["trunk"]),
        poe=tuple(parsed["power_inline"]),
        neighbors=tuple(parsed["lldp_neighbors"] + parsed["cdp_neighbors"]),
        mac_entries=tuple(parsed["mac_table"]),
        stp_states=tuple(parsed["spanning_tree"]),
    )
    logger.info(
        "Built inventory '%s': %d interface(s), %d vlan(s), %d neighbor(s); "
        "%d file(s) missing.",
        snapshot_id, len(interfaces), len(snapshot.vlans),
        len(snapshot.neighbors), len(files_missing),
    )
    return NetworkInventory(
        snapshot_id=snapshot_id,
        input_directory=str(directory),
        devices=(snapshot,),
        files_parsed=tuple(files_parsed),
        files_missing=tuple(files_missing),
        warnings=tuple(warnings),
    )


def _read_optional(
    path: Path, key: str, missing: list[str], warnings: list[str]
) -> str | None:
    """Read a file if present; record + warn (and return None) when absent."""
    if not path.is_file():
        missing.append(path.name)
        message = f"Missing '{key}' file: {path.name} (skipped)."
        warnings.append(message)
        logger.warning("%s", message)
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _enrich_interfaces(
    interfaces: list[NetworkInterface], poe: list
) -> list[NetworkInterface]:
    """Attach PoE enabled/state to interfaces by interface name."""
    poe_by_iface = {p.interface: p for p in poe}
    enriched: list[NetworkInterface] = []
    for interface in interfaces:
        entry = poe_by_iface.get(interface.name)
        if entry is None:
            enriched.append(interface)
            continue
        admin = (entry.admin_state or "").lower()
        poe_enabled = (
            True if admin in _POE_ENABLED
            else False if admin in _POE_DISABLED else None
        )
        enriched.append(
            dataclasses.replace(
                interface, poe_enabled=poe_enabled, poe_state=entry.oper_state
            )
        )
    return enriched


# ------------------------------------------------------------ derived views


def derive_summary(inventory: NetworkInventory) -> dict[str, Any]:
    """Compute analytical roll-ups used by metadata and the report."""
    interfaces = inventory.all_interfaces
    status_counts: dict[str, int] = {}
    for interface in interfaces:
        key = (interface.status or "unknown").lower()
        status_counts[key] = status_counts.get(key, 0) + 1

    stp_counts: dict[str, int] = {}
    for state in inventory.all_stp_states:
        key = state.state or "unknown"
        stp_counts[key] = stp_counts.get(key, 0) + 1

    mac_interfaces = {m.interface for m in inventory.all_mac_entries if m.interface}
    poe_enabled_ports = [i.name for i in interfaces if i.poe_enabled]
    unused_ports = [
        i.name for i in interfaces
        if (i.status or "").lower() in {"notconnect", "disabled"}
        and not i.description
    ]
    return {
        "device_count": len(inventory.devices),
        "interface_count": len(interfaces),
        "vlan_count": len(inventory.all_vlans),
        "trunk_count": len(inventory.all_trunks),
        "neighbor_count": len(inventory.all_neighbors),
        "mac_entry_count": len(inventory.all_mac_entries),
        "access_port_count": sum(1 for i in interfaces if i.mode == "access"),
        "trunk_port_count": sum(1 for i in interfaces if i.mode == "trunk"),
        "poe_enabled_port_count": len(poe_enabled_ports),
        "poe_enabled_ports": poe_enabled_ports,
        "interfaces_with_mac": sorted(mac_interfaces),
        "unused_ports": unused_ports,
        "interface_status_counts": status_counts,
        "stp_state_counts": stp_counts,
    }

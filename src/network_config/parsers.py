"""Offline parsers for saved Cisco-style command outputs.

Purpose
-------
Turn the text of a saved ``show ...`` command into typed model objects. Every
parser is small, pure (text in, list of dataclasses out) and tolerant of
missing columns and reasonable formatting differences. Nothing here touches a
live device — inputs are saved files only.

Parsing strategy
----------------
Column-aligned tables (``show interface status``, ``show power inline``,
``show lldp/cdp neighbors``) are sliced by header position so fields with
internal spaces (interface names like ``Gig 0/1``, powered-device names) stay
intact. Single-token tables (VLAN, MAC, trunk, STP) use line regexes.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from src.network_config.models import (
    MACAddressEntry,
    Neighbor,
    NetworkInterface,
    PoEStatus,
    STPState,
    TrunkInterface,
    VLAN,
)

logger = logging.getLogger(__name__)

# Cisco STP status abbreviations -> canonical port state.
_STP_STATE = {
    "FWD": "forwarding", "BLK": "blocking", "LIS": "listening",
    "LRN": "learning", "DIS": "disabled", "BKN": "broken",
}


def _nonblank_lines(text: str) -> list[str]:
    """Split text into lines, dropping blank ones (CRLF-safe)."""
    return [line for line in text.splitlines() if line.strip()]


def _is_separator(line: str) -> bool:
    """True for ruler lines made only of dashes/spaces."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("- ")


def _header_bounds(
    header: str, names: Sequence[str]
) -> list[tuple[str, int, Optional[int]]]:
    """Return ``(name, start, end)`` slices from a header line.

    Columns whose name is absent from the header are skipped (tolerant to
    missing columns); the remaining columns are ordered by position and each
    ends where the next begins (last column runs to end of line).
    """
    found = [(name, header.find(name)) for name in names]
    found = sorted((pair for pair in found if pair[1] >= 0), key=lambda p: p[1])
    bounds: list[tuple[str, int, Optional[int]]] = []
    for idx, (name, start) in enumerate(found):
        end = found[idx + 1][1] if idx + 1 < len(found) else None
        bounds.append((name, start, end))
    return bounds


def _parse_fixed_width(text: str, names: Sequence[str]) -> list[dict[str, str]]:
    """Parse a column-aligned table into row dicts keyed by header name."""
    lines = _nonblank_lines(text)
    # The header is the line matching the most column names (tolerant to
    # missing columns); at least two must match to avoid a stray data line.
    header_idx, best = None, 0
    for i, line in enumerate(lines):
        count = sum(1 for name in names if name in line)
        if count > best:
            best, header_idx = count, i
    if header_idx is None or best < 2:
        return []
    bounds = _header_bounds(lines[header_idx], names)
    rows: list[dict[str, str]] = []
    for line in lines[header_idx + 1:]:
        if _is_separator(line):
            continue
        row = {name: line[start:end].strip() for name, start, end in bounds}
        if row.get(names[0]):
            rows.append(row)
    return rows


# ---------------------------------------------------------------- interfaces


def parse_interface_status(text: str) -> list[NetworkInterface]:
    """Parse ``show interface status``."""
    interfaces: list[NetworkInterface] = []
    for row in _parse_fixed_width(
        text, ["Port", "Name", "Status", "Vlan", "Duplex", "Speed", "Type"]
    ):
        vlan_field = row.get("Vlan", "")
        mode, vlan = _interface_mode(vlan_field)
        interfaces.append(
            NetworkInterface(
                name=row["Port"],
                description=row.get("Name") or None,
                status=row.get("Status") or None,
                vlan=vlan,
                mode=mode,
                duplex=row.get("Duplex") or None,
                speed=row.get("Speed") or None,
            )
        )
    return interfaces


def _interface_mode(vlan_field: str) -> tuple[str, Optional[str]]:
    """Classify an interface-status VLAN cell into (mode, access_vlan)."""
    value = vlan_field.strip().lower()
    if value == "trunk":
        return "trunk", None
    if value == "routed":
        return "routed", None
    if value.isdigit():
        return "access", vlan_field.strip()
    return "unknown", None


# --------------------------------------------------------------------- vlans


def parse_vlan_brief(text: str) -> list[VLAN]:
    """Parse ``show vlan brief`` (ports may wrap onto indented lines)."""
    vlans: list[VLAN] = []
    current: Optional[dict] = None
    row_re = re.compile(r"^(\d+)\s+(\S+)\s+(\S+)\s*(.*)$")
    for line in text.splitlines():
        if not line.strip() or _is_separator(line) or line.lower().startswith(
            "vlan "
        ):
            continue
        match = row_re.match(line)
        if match:
            if current:
                vlans.append(_finish_vlan(current))
            vlan_id, name, status, ports = match.groups()
            current = {"id": vlan_id, "name": name, "status": status,
                       "ports": _split_ports(ports)}
        elif current and line.startswith(" "):
            current["ports"].extend(_split_ports(line))
    if current:
        vlans.append(_finish_vlan(current))
    return vlans


def _split_ports(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _finish_vlan(data: dict) -> VLAN:
    return VLAN(
        vlan_id=data["id"], name=data["name"], status=data["status"],
        ports=tuple(data["ports"]),
    )


# -------------------------------------------------------------------- trunks


def parse_trunk(text: str) -> list[TrunkInterface]:
    """Parse ``show interfaces trunk`` (multi-section Cisco output)."""
    native: dict[str, Optional[str]] = {}
    status: dict[str, Optional[str]] = {}
    allowed: dict[str, tuple[str, ...]] = {}
    section = None
    for line in _nonblank_lines(text):
        low = line.lower()
        if low.startswith("port") and "native vlan" in low:
            section = "summary"
            continue
        if low.startswith("port") and "vlans allowed on trunk" in low:
            section = "allowed"
            continue
        if low.startswith("port"):
            section = "other"
            continue
        parts = line.split()
        if section == "summary" and len(parts) >= 5:
            port = parts[0]
            status[port] = parts[3]
            native[port] = parts[4]
        elif section == "allowed" and len(parts) >= 2:
            allowed[parts[0]] = tuple(_expand_vlan_list(parts[1]))
    ports = set(native) | set(allowed)
    return [
        TrunkInterface(
            interface=port,
            allowed_vlans=allowed.get(port, ()),
            native_vlan=native.get(port),
            trunking_status=status.get(port),
        )
        for port in sorted(ports)
    ]


def _expand_vlan_list(text: str) -> list[str]:
    """Expand a Cisco VLAN list like ``1,10-12,99`` into individual ids."""
    result: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if "-" in token:
            low, high = token.split("-", 1)
            if low.isdigit() and high.isdigit():
                result.extend(str(v) for v in range(int(low), int(high) + 1))
                continue
        if token:
            result.append(token)
    return result


# ----------------------------------------------------------------- neighbors


def parse_lldp_neighbors(text: str) -> list[Neighbor]:
    """Parse ``show lldp neighbors``."""
    rows = _parse_fixed_width(
        text, ["Device ID", "Local Intf", "Hold-time", "Capability", "Port ID"]
    )
    return [
        Neighbor(
            local_interface=row.get("Local Intf", ""),
            remote_device=row.get("Device ID") or None,
            remote_interface=row.get("Port ID") or None,
            protocol="lldp",
        )
        for row in rows
        if row.get("Local Intf")
    ]


def parse_cdp_neighbors(text: str) -> list[Neighbor]:
    """Parse ``show cdp neighbors``."""
    rows = _parse_fixed_width(
        text,
        ["Device ID", "Local Intrfce", "Holdtme", "Capability", "Platform",
         "Port ID"],
    )
    return [
        Neighbor(
            local_interface=row.get("Local Intrfce", ""),
            remote_device=row.get("Device ID") or None,
            remote_interface=row.get("Port ID") or None,
            protocol="cdp",
        )
        for row in rows
        if row.get("Local Intrfce")
    ]


# ----------------------------------------------------------------- mac table


def parse_mac_table(text: str) -> list[MACAddressEntry]:
    """Parse ``show mac address-table``."""
    entries: list[MACAddressEntry] = []
    row_re = re.compile(
        r"^\s*(\d+|All)\s+([0-9a-fA-F]{4}\.[0-9a-fA-F]{4}\.[0-9a-fA-F]{4})"
        r"\s+(\S+)\s+(\S+)"
    )
    for line in text.splitlines():
        match = row_re.match(line)
        if match:
            vlan, mac, entry_type, interface = match.groups()
            entries.append(
                MACAddressEntry(
                    vlan=vlan, mac_address=mac.lower(),
                    interface=interface, entry_type=entry_type.upper(),
                )
            )
    return entries


# ----------------------------------------------------------------------- poe


def parse_power_inline(text: str) -> list[PoEStatus]:
    """Parse ``show power inline`` (interface section)."""
    rows = _parse_fixed_width(
        text,
        ["Interface", "Admin", "Oper", "Power", "Device", "Class", "Max"],
    )
    poe: list[PoEStatus] = []
    for row in rows:
        admin = (row.get("Admin") or "").lower() or None
        device = row.get("Device") or None
        if device and device.lower() in {"n/a", "na", "-"}:
            device = None
        poe.append(
            PoEStatus(
                interface=row["Interface"],
                admin_state=admin,
                oper_state=(row.get("Oper") or "").lower() or None,
                power_watts=_to_float(row.get("Power")),
                powered_device=device,
                poe_class=(row.get("Class") or None)
                if (row.get("Class") or "").lower() not in {"n/a", ""} else None,
                max_watts=_to_float(row.get("Max")),
            )
        )
    return poe


def _to_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value not in (None, "", "n/a") else None
    except ValueError:
        return None


# ----------------------------------------------------------------------- stp


def parse_spanning_tree(text: str) -> list[STPState]:
    """Parse ``show spanning-tree`` (per-VLAN interface roles/states)."""
    states: list[STPState] = []
    current_vlan: Optional[str] = None
    vlan_re = re.compile(r"^VLAN0*(\d+)", re.IGNORECASE)
    row_re = re.compile(
        r"^\s*(\S+)\s+(Root|Desg|Altn|Back|Boun|Mstr)\s+"
        r"(FWD|BLK|LIS|LRN|DIS|BKN)\b",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        vlan_match = vlan_re.match(line.strip())
        if vlan_match:
            current_vlan = vlan_match.group(1)
            continue
        row = row_re.match(line)
        if row:
            interface, role, sts = row.groups()
            states.append(
                STPState(
                    vlan=current_vlan, interface=interface,
                    role=role.capitalize(),
                    state=_STP_STATE.get(sts.upper(), sts.lower()),
                )
            )
    return states


# ------------------------------------------------------------ running config


def parse_running_config(text: str) -> dict[str, Optional[str]]:
    """Extract lightweight device identity from a running-config.

    Returns ``hostname``, ``platform`` (version, when present) and
    ``management_ip`` (first ``ip address`` under a VLAN/management interface).
    """
    hostname: Optional[str] = None
    platform: Optional[str] = None
    management_ip: Optional[str] = None
    in_mgmt = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("hostname "):
            hostname = line.split(None, 1)[1].strip()
        elif line.startswith("version "):
            platform = line.split(None, 1)[1].strip()
        elif line.startswith("interface "):
            target = line.split(None, 1)[1].lower()
            in_mgmt = target.startswith("vlan") or "mgmt" in target
        elif in_mgmt and line.startswith("ip address ") and management_ip is None:
            parts = line.split()
            if len(parts) >= 3:
                management_ip = parts[2]
    return {"hostname": hostname, "platform": platform,
            "management_ip": management_ip}

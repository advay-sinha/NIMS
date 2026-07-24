"""Huawei VRP command-output parsers for Engine C (offline, read-only).

Turns saved ``display ...`` outputs from a Huawei VRP switch (e.g. S5720) into
the shared typed models in :mod:`src.network_config.models`, so the existing
inventory → rules → topology → remediation → dashboard pipeline consumes Huawei
devices without any Cisco/Hirschmann-specific code changing.

Only read-only parsing lives here — no device access, no SNMP, no writes. The
running-config text is expected to be pre-redacted (no credential material).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping

from src.network_config.models import (
    MACAddressEntry,
    NetworkDevice,
    NetworkInterface,
    NetworkInventory,
    Neighbor,
    ParsedDeviceSnapshot,
    PoEStatus,
    STPState,
    TrunkInterface,
    VLAN,
)

logger = logging.getLogger(__name__)

# Default fixture/file names for a saved Huawei snapshot directory.
DEFAULT_FILES: dict[str, str] = {
    "version": "display_version.txt",
    "device": "display_device.txt",
    "interface_brief": "display_interface_brief.txt",
    "vlan": "display_vlan.txt",
    "stp": "display_stp_brief.txt",
    "poe": "display_poe_power.txt",
    "mac": "display_mac_address.txt",
    "lldp": "display_lldp_neighbor_brief.txt",
    "running_config": "display_current_configuration.txt",
}

# Short interface prefixes -> canonical VRP names (for cross-table joins).
_IFACE_PREFIX = {
    "GE": "GigabitEthernet",
    "XGE": "XGigabitEthernet",
    "Eth": "Ethernet",
    "MEth": "MEth",
    "Eth-Trunk": "Eth-Trunk",
    "Vlanif": "Vlanif",
    "NULL": "NULL",
}

_STP_ROLE = {"ROOT": "Root", "DESI": "Desg", "ALTE": "Altn", "BACK": "Back", "MAST": "Master"}
_STP_STATE = {"FORWARDING": "forwarding", "DISCARDING": "blocking",
              "LEARNING": "learning", "DISABLED": "disabled"}


def normalize_ifname(name: str) -> str:
    """Expand a short VRP interface name (``GE0/0/1``) to canonical form."""
    name = re.sub(r"\([^)]*\)\s*$", "", name.strip())  # drop trailing (U)/(D) markers
    match = re.match(r"^([A-Za-z-]+?)(\d.*)$", name)
    if not match:
        return name
    prefix, rest = match.group(1), match.group(2)
    return _IFACE_PREFIX.get(prefix, prefix) + rest


def _status_from_phy(phy: str, proto: str) -> str:
    """Map ``display interface brief`` PHY/Protocol to a canonical status."""
    phy = phy.strip().lower()
    proto = proto.strip().lower()
    if phy.startswith("*"):
        return "disabled"          # administratively down
    if phy == "up" and proto.startswith("up"):
        return "connected"
    return "notconnect"            # link down but admin up (unused-but-up)


# --------------------------------------------------------------- table parsers


def parse_interface_brief(text: str) -> dict[str, dict[str, str]]:
    """Parse ``display interface brief`` into {ifname: {phy, proto}}."""
    out: dict[str, dict[str, str]] = {}
    started = False
    for line in text.splitlines():
        if line.startswith("Interface") and "PHY" in line:
            started = True
            continue
        if not started or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        name = normalize_ifname(parts[0])
        out[name] = {"phy": parts[1], "proto": parts[2]}
    return out


def parse_vlan(text: str) -> dict[str, list[str]]:
    """Parse ``display vlan`` into {vlan_id: [access(untagged) ports]}."""
    ports_by_vlan: dict[str, list[str]] = {}
    current: str | None = None
    collecting = False
    for line in text.splitlines():
        if re.match(r"^VID\s+Type\s+Ports", line):
            collecting = True
            continue
        if not collecting:
            continue
        if re.match(r"^VID\s+Status", line):  # second table starts -> stop
            break
        m = re.match(r"^(\d+)\s+\w+\s+(.*)$", line)
        if m:
            current = m.group(1)
            ports_by_vlan.setdefault(current, [])
            rest = m.group(2)
        elif current and line.strip():
            rest = line
        else:
            continue
        # Untagged (access) ports are prefixed "UT:"; tagged "TG:".
        for chunk in re.findall(r"UT:(.*)", rest):
            for tok in chunk.split():
                ports_by_vlan[current].append(normalize_ifname(tok))
    return ports_by_vlan


def parse_stp_brief(text: str) -> list[STPState]:
    """Parse ``display stp brief`` into per-port STP states."""
    out: list[STPState] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        mstid, port, role, state = parts[0], parts[1], parts[2], parts[3]
        out.append(STPState(
            vlan=mstid, interface=normalize_ifname(port),
            role=_STP_ROLE.get(role.upper(), role),
            state=_STP_STATE.get(state.upper(), state.lower()),
        ))
    return out


def parse_poe_power(text: str) -> list[PoEStatus]:
    """Parse ``display poe power`` into per-port PoE state."""
    out: list[PoEStatus] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[0].lower().startswith(("gig", "ge", "xge", "ten")):
            continue
        name = normalize_ifname(parts[0])
        poe_class = parts[1] if parts[1] != "-" else None
        cur = _to_float(parts[4])            # CURPW (mW)
        usm = _to_float(parts[3])            # USMPW (mW) = user set max
        oper = "on" if (cur or 0) > 0 else "off"
        out.append(PoEStatus(
            interface=name,
            admin_state="auto",              # PoE-capable port, budget assigned
            oper_state=oper,
            power_watts=(cur / 1000.0) if cur is not None else None,
            max_watts=(usm / 1000.0) if usm is not None else None,
            poe_class=poe_class,
        ))
    return out


def parse_mac_address(text: str) -> list[MACAddressEntry]:
    """Parse ``display mac-address`` into MAC table entries."""
    out: list[MACAddressEntry] = []
    for line in text.splitlines():
        m = re.match(r"^([0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4})\s+"
                     r"(\d+)\S*\s+(\S+)\s+(\w+)", line)
        if not m:
            continue
        mac, vlan, iface, etype = m.group(1), m.group(2), m.group(3), m.group(4)
        out.append(MACAddressEntry(
            vlan=vlan, mac_address=mac.lower(),
            interface=normalize_ifname(iface), entry_type=etype.lower(),
        ))
    return out


def parse_lldp_neighbors(text: str) -> list[Neighbor]:
    """Parse ``display lldp neighbor brief`` into neighbour relationships."""
    out: list[Neighbor] = []
    started = False
    for line in text.splitlines():
        if line.startswith("Local Intf"):
            started = True
            continue
        if not started or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        out.append(Neighbor(
            local_interface=normalize_ifname(parts[0]),
            remote_device=parts[1],
            remote_interface=parts[2],
            protocol="lldp",
        ))
    return out


def parse_version(text: str) -> dict[str, str | None]:
    """Extract model + VRP version from ``display version``."""
    model = None
    version = None
    for line in text.splitlines():
        m = re.search(r"(S\d{3,4}\S*)", line)
        if m and model is None and "uptime" in line.lower():
            model = m.group(1)
        v = re.search(r"Version\s+([\d.]+)\s+\((\S+?)\s+(V\d\S+)\)", line)
        if v:
            version = v.group(3)
    return {"model": model, "vrp_version": version}


def parse_running_config(text: str) -> dict[str, Any]:
    """Parse the redacted running-config: identity, VLAN names, per-port config."""
    hostname = None
    platform = None
    management_ip = None
    vlan_names: dict[str, str] = {}
    interfaces: dict[str, dict[str, Any]] = {}
    security: dict[str, Any] = {"telnet_server": False, "ftp_service": False,
                                "snmp_v2c_community": False}

    lines = text.splitlines()
    current_if: str | None = None
    current_vlan: str | None = None
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if line.startswith("!Software Version"):
            platform = stripped.replace("!Software Version", "").strip()
        elif stripped.startswith("sysname "):
            hostname = stripped[len("sysname "):].strip()
        elif stripped == "telnet server enable":
            security["telnet_server"] = True
        elif "service-type" in stripped and "ftp" in stripped:
            security["ftp_service"] = True
        elif stripped.startswith("snmp-agent community"):
            security["snmp_v2c_community"] = True

        # VLAN name blocks
        vm = re.match(r"^vlan (\d+)$", stripped)
        if vm:
            current_vlan = vm.group(1)
            current_if = None
            continue
        if current_vlan and stripped.startswith("name "):
            vlan_names[current_vlan] = stripped[len("name "):].strip()
            continue

        # interface blocks
        im = re.match(r"^interface (\S+)$", stripped)
        if im:
            current_if = normalize_ifname(im.group(1))
            current_vlan = None
            interfaces.setdefault(current_if, {"mode": "unknown"})
            if current_if.startswith("Vlanif"):
                interfaces[current_if]["mode"] = "routed"
            continue
        if current_if:
            cfg = interfaces[current_if]
            if stripped == "port link-type trunk":
                cfg["mode"] = "trunk"
            elif stripped == "port link-type access":
                cfg["mode"] = "access"
            elif stripped.startswith("port default vlan "):
                cfg["vlan"] = stripped.split()[-1]
            elif stripped.startswith("port trunk pvid vlan "):
                cfg["native"] = stripped.split()[-1]
            elif stripped.startswith("port trunk allow-pass vlan "):
                cfg["allow_spec"] = stripped[len("port trunk allow-pass vlan "):].strip()
            elif stripped.startswith("description "):
                cfg["description"] = stripped[len("description "):].strip()
            elif stripped.startswith("ip address ") and current_if == "Vlanif200":
                management_ip = stripped.split()[2]

    return {
        "hostname": hostname,
        "platform": platform,
        "management_ip": management_ip,
        "vlan_names": vlan_names,
        "interfaces": interfaces,
        "security": security,
    }


# ------------------------------------------------------------ inventory build


def _expand_allow_spec(spec: str, defined_vlans: set[str]) -> tuple[str, ...]:
    """Resolve a trunk ``allow-pass`` spec to the DEFINED VLANs it permits.

    ``2 to 4094`` permits every configured VLAN; rather than emit ~4000 ids we
    intersect with the VLANs that actually exist so downstream rules and CSVs
    stay meaningful.
    """
    tokens = spec.split()
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == "to":
            ranges.append((int(tokens[i]), int(tokens[i + 2])))
            i += 3
        else:
            if tokens[i].isdigit():
                ranges.append((int(tokens[i]), int(tokens[i])))
            i += 1
    permitted = {
        v for v in defined_vlans
        if any(lo <= int(v) <= hi for lo, hi in ranges)
    }
    return tuple(sorted(permitted, key=lambda x: int(x)))


def build_inventory(
    input_dir: str | Path,
    config: Mapping[str, Any],
    snapshot_id: str,
) -> NetworkInventory:
    """Build a :class:`NetworkInventory` from a saved Huawei snapshot directory.

    Mirrors :func:`src.network_config.inventory.build_inventory` (same signature
    and return type) so the analysis pipeline treats Huawei like any other
    vendor. Missing files are recorded and skipped, not fatal.
    """
    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    files = {**DEFAULT_FILES, **dict(config.get("huawei_files") or {})}
    files_parsed: list[str] = []
    files_missing: list[str] = []
    warnings: list[str] = []

    def read(key: str) -> str | None:
        path = directory / files[key]
        if not path.is_file():
            files_missing.append(path.name)
            warnings.append(f"Missing '{key}' file: {path.name} (skipped).")
            logger.warning("Missing Huawei '%s' file: %s (skipped).", key, path.name)
            return None
        files_parsed.append(path.name)
        return path.read_text(encoding="utf-8", errors="replace")

    brief_txt = read("interface_brief")
    run_txt = read("running_config")
    vlan_txt = read("vlan")
    stp_txt = read("stp")
    poe_txt = read("poe")
    mac_txt = read("mac")
    lldp_txt = read("lldp")
    ver_txt = read("version")

    brief = parse_interface_brief(brief_txt) if brief_txt else {}
    rc = parse_running_config(run_txt) if run_txt else {}
    vlan_ports = parse_vlan(vlan_txt) if vlan_txt else {}
    poe = parse_poe_power(poe_txt) if poe_txt else []
    stp = parse_stp_brief(stp_txt) if stp_txt else []
    macs = parse_mac_address(mac_txt) if mac_txt else []
    neighbors = parse_lldp_neighbors(lldp_txt) if lldp_txt else []
    version = parse_version(ver_txt) if ver_txt else {}

    rc_ifaces: dict[str, dict[str, Any]] = rc.get("interfaces", {})
    vlan_names: dict[str, str] = rc.get("vlan_names", {})
    defined_vlans = set(vlan_ports) | set(vlan_names)

    poe_by_iface = {p.interface: p for p in poe}
    interfaces: list[NetworkInterface] = []
    all_names = set(brief) | set(rc_ifaces)
    for name in sorted(all_names, key=_iface_sort_key):
        b = brief.get(name, {})
        cfg = rc_ifaces.get(name, {})
        mode = cfg.get("mode", "unknown")
        vlan = cfg.get("vlan")
        if mode == "access" and vlan is None:
            vlan = "1"  # Huawei access default VLAN
        status = _status_from_phy(b.get("phy", "down"), b.get("proto", "down")) if b else None
        p = poe_by_iface.get(name)
        interfaces.append(NetworkInterface(
            name=name, status=status,
            protocol_status=b.get("proto") if b else None,
            vlan=vlan, mode=mode, description=cfg.get("description"),
            poe_enabled=(p.oper_state == "on") if p else None,
            poe_state=p.oper_state if p else None,
        ))

    vlans = tuple(
        VLAN(vlan_id=vid, name=vlan_names.get(vid), status="enable",
             ports=tuple(vlan_ports.get(vid, ())))
        for vid in sorted(defined_vlans, key=int)
    )

    trunks: list[TrunkInterface] = []
    for name, cfg in rc_ifaces.items():
        if cfg.get("mode") != "trunk":
            continue
        allowed = _expand_allow_spec(cfg["allow_spec"], defined_vlans) if cfg.get("allow_spec") else ()
        oper = brief.get(name, {}).get("proto", "")
        trunks.append(TrunkInterface(
            interface=name, allowed_vlans=allowed,
            native_vlan=cfg.get("native"),
            trunking_status="trunking" if oper.lower().startswith("up") else "not-trunking",
        ))

    hostname = rc.get("hostname")
    device = NetworkDevice(
        device_id=hostname or snapshot_id,
        hostname=hostname,
        platform=version.get("model") or rc.get("platform"),
        management_ip=rc.get("management_ip"),
        source_files=tuple(files_parsed),
    )
    snapshot = ParsedDeviceSnapshot(
        device=device,
        interfaces=tuple(interfaces),
        vlans=vlans,
        trunks=tuple(sorted(trunks, key=lambda t: _iface_sort_key(t.interface))),
        poe=tuple(poe),
        neighbors=tuple(neighbors),
        mac_entries=tuple(macs),
        stp_states=tuple(stp),
    )
    logger.info(
        "Built Huawei inventory '%s': %d interface(s), %d vlan(s), %d trunk(s), "
        "%d neighbor(s), %d mac(s); %d file(s) missing.",
        snapshot_id, len(interfaces), len(vlans), len(trunks),
        len(neighbors), len(macs), len(files_missing),
    )
    return NetworkInventory(
        snapshot_id=snapshot_id,
        input_directory=str(directory),
        devices=(snapshot,),
        files_parsed=tuple(files_parsed),
        files_missing=tuple(files_missing),
        warnings=tuple(warnings),
    )


def _iface_sort_key(name: str) -> tuple[str, tuple[int, ...]]:
    nums = tuple(int(n) for n in re.findall(r"\d+", name))
    prefix = re.match(r"^[A-Za-z-]+", name)
    return (prefix.group(0) if prefix else name, nums)


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

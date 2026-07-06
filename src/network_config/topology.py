"""Engine C Phase 2 — topology construction from parsed inventory.

Purpose
-------
Derive a network topology from an already-parsed
:class:`~src.network_config.models.NetworkInventory` (offline, read-only). High-
confidence edges come from LLDP/CDP neighbours (deduplicated across protocols
and normalised so reversed duplicate links collapse to one bidirectional edge).
MAC-table and STP signals are used **conservatively** — as warnings and
low/medium-confidence hints, never as certain links.

Everything here is deterministic and pure: inventory in, topology out. No device
is contacted.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from src.network_config.models import NetworkInventory

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "mac_access_port_threshold": 5,
    "require_stp_on_trunks": True,
    "warn_on_lldp_cdp_mismatch": True,
    "confidence": {"lldp_cdp": "high", "mac_table": "low",
                   "stp_inferred": "medium"},
}

# Cisco interface prefix normalisation (for matching only; display keeps
# the original spelling).
_IFACE_PREFIX = {
    "gigabitethernet": "gi", "gig": "gi", "gi": "gi",
    "tengigabitethernet": "te", "te": "te",
    "fortygigabitethernet": "fo", "fo": "fo",
    "fastethernet": "fa", "fa": "fa",
    "ethernet": "eth", "eth": "eth", "et": "et",
    "port": "port", "po": "po", "portchannel": "po",
}


@dataclass(frozen=True)
class TopologyNode:
    """A device in the topology graph."""

    node_id: str
    hostname: Optional[str] = None
    device_type: Optional[str] = None
    management_ip: Optional[str] = None
    source: str = "local"                    # local / lldp / cdp / inferred


@dataclass(frozen=True)
class TopologyEdge:
    """A link between two devices/interfaces with a confidence and evidence."""

    local_device: str
    local_interface: str
    remote_device: Optional[str]
    remote_interface: Optional[str]
    discovery_protocol: str                  # lldp / cdp / lldp+cdp / mac / stp
    confidence: str                          # high / medium / low
    evidence: str
    bidirectional: bool = False


@dataclass(frozen=True)
class TopologyWarning:
    """A conservative, evidence-backed topology concern."""

    warning_id: str
    severity: str                            # info / warning
    category: str                            # discovery / topology / loop_risk / stp
    message: str
    device: Optional[str] = None
    interface: Optional[str] = None
    evidence: Optional[str] = None


@dataclass(frozen=True)
class NetworkTopology:
    """The derived topology for one snapshot."""

    snapshot_id: str
    nodes: tuple[TopologyNode, ...] = ()
    edges: tuple[TopologyEdge, ...] = ()
    warnings: tuple[TopologyWarning, ...] = ()


# --------------------------------------------------------------- normalisers


def _norm_iface(name: Optional[str]) -> str:
    """Normalise an interface name for matching (``Gig 0/1`` -> ``gi0/1``)."""
    if not name:
        return ""
    token = re.sub(r"\s+", "", name).lower()
    match = re.match(r"^([a-z]+)(.*)$", token)
    if not match:
        return token
    prefix, rest = match.groups()
    return _IFACE_PREFIX.get(prefix, prefix[:2]) + rest


def _norm_device(name: Optional[str]) -> str:
    """Normalise a device name for matching (strip domain, lowercase)."""
    if not name:
        return ""
    return name.split(".")[0].strip().lower()


def _config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge a topology config block over the built-in defaults."""
    merged = {**_DEFAULTS, **dict(config or {})}
    merged["confidence"] = {**_DEFAULTS["confidence"],
                            **dict((config or {}).get("confidence") or {})}
    return merged


# --------------------------------------------------------------------- build


def build_topology(
    inventory: NetworkInventory, config: Mapping[str, Any] | None = None
) -> NetworkTopology:
    """Build a :class:`NetworkTopology` from a parsed inventory."""
    cfg = _config(config)
    nodes = _build_nodes(inventory)
    edges = _build_neighbor_edges(inventory, cfg)
    warnings = _collect_warnings(inventory, edges, cfg)
    logger.info(
        "Topology '%s': %d node(s), %d edge(s), %d warning(s).",
        inventory.snapshot_id, len(nodes), len(edges), len(warnings),
    )
    return NetworkTopology(
        snapshot_id=inventory.snapshot_id,
        nodes=tuple(nodes), edges=tuple(edges), warnings=tuple(warnings),
    )


def _build_nodes(inventory: NetworkInventory) -> list[TopologyNode]:
    """Local devices plus every distinct LLDP/CDP remote device."""
    nodes: dict[str, TopologyNode] = {}
    for snap in inventory.devices:
        dev = snap.device
        nodes[_norm_device(dev.device_id)] = TopologyNode(
            node_id=dev.device_id, hostname=dev.hostname,
            management_ip=dev.management_ip, source="local",
        )
    for snap in inventory.devices:
        for nb in snap.neighbors:
            key = _norm_device(nb.remote_device)
            if key and key not in nodes:
                nodes[key] = TopologyNode(
                    node_id=nb.remote_device or key, hostname=nb.remote_device,
                    source=nb.protocol,
                )
    return [nodes[k] for k in sorted(nodes)]


def _build_neighbor_edges(
    inventory: NetworkInventory, cfg: dict[str, Any]
) -> list[TopologyEdge]:
    """LLDP/CDP edges: merge per local port, then collapse reversed pairs."""
    # 1) One working edge per (local device, local interface); merge protocols.
    by_port: dict[tuple[str, str], dict[str, Any]] = {}
    for snap in inventory.devices:
        local_device = snap.device.device_id
        for nb in snap.neighbors:
            key = (_norm_device(local_device), _norm_iface(nb.local_interface))
            entry = by_port.setdefault(
                key,
                {
                    "local_device": local_device,
                    "local_interface": nb.local_interface,
                    "remote_device": nb.remote_device,
                    "remote_interface": nb.remote_interface,
                    "protocols": set(),
                },
            )
            entry["protocols"].add(nb.protocol)
            # Prefer the more qualified remote name (e.g. FQDN over short host).
            if nb.remote_device and len(nb.remote_device) > len(
                entry["remote_device"] or ""
            ):
                entry["remote_device"] = nb.remote_device
            if nb.remote_interface and not entry["remote_interface"]:
                entry["remote_interface"] = nb.remote_interface

    # 2) Collapse reversed duplicates into single bidirectional edges.
    high = cfg["confidence"]["lldp_cdp"]
    local_ep = {key: e for key, e in by_port.items()}
    consumed: set[tuple[str, str]] = set()
    edges: list[TopologyEdge] = []
    for key in sorted(by_port):
        if key in consumed:
            continue
        entry = by_port[key]
        remote_key = (_norm_device(entry["remote_device"]),
                      _norm_iface(entry["remote_interface"]))
        reverse = local_ep.get(remote_key)
        bidirectional = False
        protocols = set(entry["protocols"])
        if (
            reverse is not None
            and remote_key != key
            and (_norm_device(reverse["remote_device"]),
                 _norm_iface(reverse["remote_interface"])) == key
        ):
            bidirectional = True
            protocols |= set(reverse["protocols"])
            consumed.add(remote_key)
        edges.append(
            TopologyEdge(
                local_device=entry["local_device"],
                local_interface=entry["local_interface"],
                remote_device=entry["remote_device"],
                remote_interface=entry["remote_interface"],
                discovery_protocol="+".join(sorted(protocols)),
                confidence=high,
                evidence=(
                    f"{'/'.join(sorted(protocols))} neighbour on "
                    f"{entry['local_interface']}"
                    + (" (bidirectional)" if bidirectional else "")
                ),
                bidirectional=bidirectional,
            )
        )
    return edges


# ------------------------------------------------------------------ warnings


def _collect_warnings(
    inventory: NetworkInventory, edges: list[TopologyEdge], cfg: dict[str, Any]
) -> list[TopologyWarning]:
    """Run every conservative warning check, then assign deterministic ids."""
    raw: list[dict[str, Any]] = []
    raw += _discovery_mismatch(inventory, cfg)
    raw += _unidirectional(inventory, edges)
    raw += _trunk_without_neighbor(inventory, edges)
    raw += _mac_warnings(inventory, cfg)
    raw += _stp_warnings(inventory, cfg)
    warnings: list[TopologyWarning] = []
    for index, item in enumerate(raw, start=1):
        warnings.append(TopologyWarning(warning_id=f"TW{index:03d}", **item))
    return warnings


def _edge_ports(edges: list[TopologyEdge]) -> set[tuple[str, str]]:
    """Set of (norm device, norm interface) that have a discovered neighbour."""
    return {(_norm_device(e.local_device), _norm_iface(e.local_interface))
            for e in edges}


def _trunk_interfaces(snap: Any) -> list[str]:
    """Trunk interface names for a device (from trunk output + interface mode)."""
    names = {t.interface for t in snap.trunks}
    names |= {i.name for i in snap.interfaces if i.mode == "trunk"}
    return sorted(names)


def _access_interfaces(snap: Any) -> dict[str, Any]:
    """Map of access interface name -> interface object."""
    return {i.name: i for i in snap.interfaces if i.mode == "access"}


def _discovery_mismatch(
    inventory: NetworkInventory, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Neighbour seen in only one of LLDP/CDP when both are available."""
    if not cfg.get("warn_on_lldp_cdp_mismatch", True):
        return []
    out: list[dict[str, Any]] = []
    for snap in inventory.devices:
        protocols_present = {nb.protocol for nb in snap.neighbors}
        if {"lldp", "cdp"} - protocols_present:
            continue  # need both protocols captured for a mismatch to matter
        by_port: dict[str, set[str]] = defaultdict(set)
        for nb in snap.neighbors:
            by_port[_norm_iface(nb.local_interface)].add(nb.protocol)
        for iface in sorted(by_port):
            protos = by_port[iface]
            if len(protos) == 1:
                seen = next(iter(protos))
                missing = "cdp" if seen == "lldp" else "lldp"
                out.append({
                    "severity": "info", "category": "discovery",
                    "message": (f"Neighbour on {iface} seen via {seen} but not "
                                f"{missing} (possible protocol mismatch)."),
                    "device": snap.device.device_id, "interface": iface,
                    "evidence": f"protocols on port: {sorted(protos)}",
                })
    return out


def _unidirectional(
    inventory: NetworkInventory, edges: list[TopologyEdge]
) -> list[dict[str, Any]]:
    """Link to a device we also parsed, but the reverse edge is absent."""
    local_devices = {_norm_device(d.device.device_id) for d in inventory.devices}
    out: list[dict[str, Any]] = []
    for edge in edges:
        remote = _norm_device(edge.remote_device)
        if not edge.bidirectional and remote in local_devices and remote != (
            _norm_device(edge.local_device)
        ):
            out.append({
                "severity": "warning", "category": "topology",
                "message": (f"Neighbour relationship {edge.local_device} -> "
                            f"{edge.remote_device} appears unidirectional "
                            "(reverse link not discovered)."),
                "device": edge.local_device, "interface": edge.local_interface,
                "evidence": edge.evidence,
            })
    return out


def _trunk_without_neighbor(
    inventory: NetworkInventory, edges: list[TopologyEdge]
) -> list[dict[str, Any]]:
    """Trunk port with no discovered LLDP/CDP neighbour."""
    ports = _edge_ports(edges)
    out: list[dict[str, Any]] = []
    for snap in inventory.devices:
        device = _norm_device(snap.device.device_id)
        for iface in _trunk_interfaces(snap):
            if (device, _norm_iface(iface)) not in ports:
                out.append({
                    "severity": "warning", "category": "topology",
                    "message": (f"Trunk port {iface} has no discovered "
                                "LLDP/CDP neighbour (candidate unmanaged "
                                "neighbour or misconfiguration)."),
                    "device": snap.device.device_id, "interface": iface,
                    "evidence": "trunk present; no neighbour edge",
                })
    return out


def _mac_warnings(
    inventory: NetworkInventory, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """MAC-table-derived loop-risk and access-port-saturation hints."""
    threshold = int(cfg.get("mac_access_port_threshold", 5))
    out: list[dict[str, Any]] = []
    for snap in inventory.devices:
        access = _access_interfaces(snap)
        macs_per_iface: dict[str, set[str]] = defaultdict(set)
        iface_per_mac: dict[str, set[str]] = defaultdict(set)
        for entry in snap.mac_entries:
            if not entry.interface:
                continue
            macs_per_iface[entry.interface].add(entry.mac_address)
            iface_per_mac[entry.mac_address].add(entry.interface)
        # Access ports carrying more MACs than expected.
        for iface in sorted(access):
            count = len(macs_per_iface.get(iface, set()))
            if count > threshold:
                out.append({
                    "severity": "warning", "category": "topology",
                    "message": (f"Access port {iface} has {count} MAC(s) "
                                f"(> threshold {threshold}); possible "
                                "unmanaged switch/hub downstream."),
                    "device": snap.device.device_id, "interface": iface,
                    "evidence": f"{count} distinct MAC(s) learned",
                })
        # A MAC learned on multiple interfaces (loop / flap / duplicate).
        for mac in sorted(iface_per_mac):
            ifaces = iface_per_mac[mac]
            if len(ifaces) > 1:
                out.append({
                    "severity": "warning", "category": "loop_risk",
                    "message": (f"MAC {mac} learned on multiple interfaces "
                                f"{sorted(ifaces)}; possible loop, flapping or "
                                "duplicate learning."),
                    "device": snap.device.device_id, "interface": None,
                    "evidence": f"interfaces: {sorted(ifaces)}",
                })
    return out


def _stp_warnings(
    inventory: NetworkInventory, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """STP-derived warnings: blocked access ports, missing STP on trunks."""
    require_stp = bool(cfg.get("require_stp_on_trunks", True))
    out: list[dict[str, Any]] = []
    for snap in inventory.devices:
        access = set(_access_interfaces(snap))
        stp_ifaces = {_norm_iface(s.interface) for s in snap.stp_states}
        for state in snap.stp_states:
            if state.state == "blocking" and state.interface in access:
                out.append({
                    "severity": "warning", "category": "stp",
                    "message": (f"STP is blocking on access port "
                                f"{state.interface} (unexpected)."),
                    "device": snap.device.device_id,
                    "interface": state.interface,
                    "evidence": (f"vlan {state.vlan} role {state.role} "
                                 "state blocking"),
                })
        if require_stp:
            for iface in _trunk_interfaces(snap):
                if _norm_iface(iface) not in stp_ifaces:
                    out.append({
                        "severity": "info", "category": "stp",
                        "message": (f"Trunk port {iface} has no STP data "
                                    "(cannot confirm loop protection)."),
                        "device": snap.device.device_id, "interface": iface,
                        "evidence": "no spanning-tree entry for trunk port",
                    })
    return out


# ------------------------------------------------------------------- summary


def topology_summary(topology: NetworkTopology) -> dict[str, Any]:
    """Counts and highlights for metadata and the report."""
    confidence = Counter(e.confidence for e in topology.edges)
    lldp_cdp = sum(
        1 for e in topology.edges
        if "lldp" in e.discovery_protocol or "cdp" in e.discovery_protocol
    )
    inferred = sum(
        1 for e in topology.edges
        if e.discovery_protocol in {"mac", "stp", "inferred"}
    )
    severity = Counter(w.severity for w in topology.warnings)
    category = Counter(w.category for w in topology.warnings)
    return {
        "node_count": len(topology.nodes),
        "edge_count": len(topology.edges),
        "bidirectional_edge_count": sum(
            1 for e in topology.edges if e.bidirectional
        ),
        "confidence_counts": {
            "high": confidence.get("high", 0),
            "medium": confidence.get("medium", 0),
            "low": confidence.get("low", 0),
        },
        "lldp_cdp_edge_count": lldp_cdp,
        "inferred_edge_count": inferred,
        "warning_count": len(topology.warnings),
        "warning_severity_counts": dict(severity),
        "warning_category_counts": dict(category),
        "top_warnings": [
            {"warning_id": w.warning_id, "severity": w.severity,
             "category": w.category, "message": w.message}
            for w in topology.warnings[:5]
        ],
    }

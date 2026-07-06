"""Typed models for Engine C network-configuration inventory.

Purpose
-------
One frozen dataclass per network object (device, interface, VLAN, trunk, PoE,
neighbor, MAC entry, STP state) plus the per-device snapshot and the aggregate
inventory. These are the single structured contract every parser produces and
every artefact/report consumes — no dict-of-dicts flows through the pipeline.

All objects are JSON-serialisable via :func:`dataclasses.asdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class NetworkInterface:
    """One switch/router interface and its access/PoE state."""

    name: str
    status: Optional[str] = None            # connected / notconnect / disabled
    protocol_status: Optional[str] = None   # up / down (when available)
    vlan: Optional[str] = None              # access VLAN id (access ports)
    mode: str = "unknown"                   # access / trunk / routed / unknown
    description: Optional[str] = None
    speed: Optional[str] = None
    duplex: Optional[str] = None
    poe_enabled: Optional[bool] = None
    poe_state: Optional[str] = None         # on / off / faulty ...


@dataclass(frozen=True)
class VLAN:
    """A VLAN definition and the access ports assigned to it."""

    vlan_id: str
    name: Optional[str] = None
    status: Optional[str] = None
    ports: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrunkInterface:
    """A trunk port and the VLANs it carries."""

    interface: str
    allowed_vlans: tuple[str, ...] = ()
    native_vlan: Optional[str] = None
    trunking_status: Optional[str] = None   # trunking / not-trunking


@dataclass(frozen=True)
class PoEStatus:
    """Power-over-Ethernet state for one interface."""

    interface: str
    admin_state: Optional[str] = None       # auto / static / off / never
    oper_state: Optional[str] = None        # on / off / faulty
    power_watts: Optional[float] = None
    powered_device: Optional[str] = None
    poe_class: Optional[str] = None
    max_watts: Optional[float] = None


@dataclass(frozen=True)
class Neighbor:
    """A discovered LLDP/CDP neighbour relationship."""

    local_interface: str
    remote_device: Optional[str] = None
    remote_interface: Optional[str] = None
    protocol: str = "lldp"                   # lldp / cdp


@dataclass(frozen=True)
class MACAddressEntry:
    """A single MAC address-table entry."""

    vlan: Optional[str]
    mac_address: str
    interface: Optional[str] = None
    entry_type: Optional[str] = None         # DYNAMIC / STATIC


@dataclass(frozen=True)
class STPState:
    """Spanning-tree role/state for one interface within one VLAN."""

    vlan: Optional[str]
    interface: str
    role: Optional[str] = None               # Root / Desg / Altn / Back
    state: Optional[str] = None              # forwarding / blocking / ...


@dataclass(frozen=True)
class NetworkDevice:
    """Identity of one parsed device."""

    device_id: str
    hostname: Optional[str] = None
    platform: Optional[str] = None
    management_ip: Optional[str] = None
    source_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedDeviceSnapshot:
    """All parsed objects for a single device."""

    device: NetworkDevice
    interfaces: tuple[NetworkInterface, ...] = ()
    vlans: tuple[VLAN, ...] = ()
    trunks: tuple[TrunkInterface, ...] = ()
    poe: tuple[PoEStatus, ...] = ()
    neighbors: tuple[Neighbor, ...] = ()
    mac_entries: tuple[MACAddressEntry, ...] = ()
    stp_states: tuple[STPState, ...] = ()


@dataclass(frozen=True)
class NetworkInventory:
    """The aggregate offline inventory for one analysis snapshot."""

    snapshot_id: str
    input_directory: str
    devices: tuple[ParsedDeviceSnapshot, ...] = ()
    files_parsed: tuple[str, ...] = ()
    files_missing: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # -- flat views across every device (used by CSV artefacts) --------------

    @property
    def all_interfaces(self) -> list[NetworkInterface]:
        return [i for d in self.devices for i in d.interfaces]

    @property
    def all_vlans(self) -> list[VLAN]:
        return [v for d in self.devices for v in d.vlans]

    @property
    def all_trunks(self) -> list[TrunkInterface]:
        return [t for d in self.devices for t in d.trunks]

    @property
    def all_poe(self) -> list[PoEStatus]:
        return [p for d in self.devices for p in d.poe]

    @property
    def all_neighbors(self) -> list[Neighbor]:
        return [n for d in self.devices for n in d.neighbors]

    @property
    def all_mac_entries(self) -> list[MACAddressEntry]:
        return [m for d in self.devices for m in d.mac_entries]

    @property
    def all_stp_states(self) -> list[STPState]:
        return [s for d in self.devices for s in d.stp_states]

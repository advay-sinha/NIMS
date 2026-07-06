"""Tests for src.network_config (Engine C Phase 1, offline read-only).

Sample command outputs are built into a tmp directory from committed constants
so the suite is self-contained (the ``datasets/`` sample tree is gitignored and
is only used for manual CLI verification).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.network_config import parsers as P
from src.network_config.artifacts import write_inventory
from src.network_config.inventory import build_inventory, derive_summary


# --------------------------------------------------------- sample builders


def _fixed(rows: list[list[str]], widths: list[int]) -> str:
    """Render column-aligned rows (last width 0 = run to end)."""
    return "\n".join(
        "".join(str(c).ljust(w) if w else str(c) for c, w in zip(r, widths))
        .rstrip()
        for r in rows
    )


INTERFACE_STATUS = _fixed(
    [
        ["Port", "Name", "Status", "Vlan", "Duplex", "Speed", "Type"],
        ["Gi0/1", "Uplink to core", "connected", "trunk", "full", "1000",
         "10/100/1000BaseTX"],
        ["Gi0/2", "", "connected", "10", "a-full", "a-100", "10/100/1000BaseTX"],
        ["Gi0/3", "", "notconnect", "20", "auto", "auto", "10/100/1000BaseTX"],
        ["Gi0/4", "", "disabled", "30", "auto", "auto", "10/100/1000BaseTX"],
        ["Gi0/5", "IP Phone", "connected", "1", "a-full", "a-100",
         "10/100/1000BaseTX"],
        ["Gi0/6", "Access Point", "connected", "1", "a-full", "a-100",
         "10/100/1000BaseTX"],
        ["Gi0/7", "", "notconnect", "40", "auto", "auto", "10/100/1000BaseTX"],
    ],
    [10, 19, 13, 11, 8, 6, 0],
) + "\n"

VLAN_BRIEF = """VLAN Name                             Status    Ports
---- -------------------------------- --------- -------------------------------
1    default                          active    Gi0/5, Gi0/6
10   users                            active    Gi0/2
20   servers                          active    Gi0/3
30   guest                            active    Gi0/4
40   quarantine                       active    Gi0/7
99   native                           active
"""

TRUNK = """Port        Mode             Encapsulation  Status        Native vlan
Gi0/1       on               802.1q         trunking      99

Port        Vlans allowed on trunk
Gi0/1       1,10,20,30,99

Port        Vlans in spanning tree forwarding state and not pruned
Gi0/1       1,10,20
"""

LLDP = _fixed(
    [
        ["Device ID", "Local Intf", "Hold-time", "Capability", "Port ID"],
        ["switch2.example.com", "Gi0/1", "120", "B", "Gi0/24"],
        ["ap1.example.com", "Gi0/6", "120", "B,R", "Fa0"],
        ["phone1.example.com", "Gi0/5", "120", "T", "Port-1"],
    ],
    [20, 15, 11, 16, 0],
) + "\n"

CDP = _fixed(
    [
        ["Device ID", "Local Intrfce", "Holdtme", "Capability", "Platform",
         "Port ID"],
        ["switch2", "Gig 0/1", "156", "S I", "WS-C3560", "Gig 0/24"],
        ["router1", "Gig 0/2", "132", "R S I", "CISCO2901", "Gig 0/0"],
    ],
    [17, 18, 11, 12, 10, 0],
) + "\n"

MAC_TABLE = """          Mac Address Table
-------------------------------------------

Vlan    Mac Address       Type        Ports
----    -----------       ----        -----
  10    0011.2233.4455    DYNAMIC     Gi0/2
  20    00aa.bbcc.ddee    DYNAMIC     Gi0/3
   1    1111.2222.3333    STATIC      Gi0/5
   1    2222.3333.4444    DYNAMIC     Gi0/6
  30    dead.beef.0001    DYNAMIC     Gi0/4

Total Mac Addresses for this criterion: 5
"""

POWER_INLINE = (
    "Module   Available     Used     Remaining\n"
    "          (Watts)     (Watts)    (Watts)\n"
    "------   ---------   --------   ---------\n"
    "1           370.0       19.9       350.1\n\n"
    + _fixed(
        [["Interface", "Admin", "Oper", "Power", "Device", "Class", "Max"]],
        [10, 7, 11, 8, 20, 6, 0],
    )
    + "\n--------- ------ ---------- ------- ------------------- ----- ----\n"
    + _fixed(
        [
            ["Gi0/5", "auto", "on", "15.4", "IP Phone 7965", "3", "15.4"],
            ["Gi0/6", "auto", "on", "4.5", "AIR-AP1131", "2", "15.4"],
            ["Gi0/2", "auto", "off", "0.0", "n/a", "n/a", "15.4"],
            ["Gi0/1", "off", "off", "0.0", "n/a", "n/a", "15.4"],
        ],
        [10, 7, 11, 8, 20, 6, 0],
    )
    + "\n"
)

SPANNING_TREE = """VLAN0001
  Spanning tree enabled protocol rstp
  Root ID    Priority    32769

  Interface           Role Sts Cost      Prio.Nbr Type
  ------------------- ---- --- --------- -------- --------------------
  Gi0/1               Desg FWD 4         128.1    P2p
  Gi0/5               Desg FWD 19        128.5    P2p
  Gi0/6               Desg FWD 19        128.6    P2p

VLAN0010
  Spanning tree enabled protocol rstp

  Interface           Role Sts Cost      Prio.Nbr Type
  ------------------- ---- --- --------- -------- --------------------
  Gi0/1               Root FWD 4         128.1    P2p
  Gi0/2               Desg FWD 19        128.2    P2p

VLAN0020
  Spanning tree enabled protocol rstp

  Interface           Role Sts Cost      Prio.Nbr Type
  ------------------- ---- --- --------- -------- --------------------
  Gi0/1               Root FWD 4         128.1    P2p
  Gi0/3               Altn BLK 19        128.3    P2p
"""

RUNNING_CONFIG = """!
version 15.2
hostname access-sw-01
!
interface Vlan1
 description Management
 ip address 10.0.0.11 255.255.255.0
!
interface GigabitEthernet0/1
 description Uplink to core
 switchport trunk native vlan 99
 switchport mode trunk
!
end
"""

SAMPLES = {
    "show_interface_status.txt": INTERFACE_STATUS,
    "show_vlan_brief.txt": VLAN_BRIEF,
    "show_interfaces_trunk.txt": TRUNK,
    "show_lldp_neighbors.txt": LLDP,
    "show_cdp_neighbors.txt": CDP,
    "show_mac_address_table.txt": MAC_TABLE,
    "show_power_inline.txt": POWER_INLINE,
    "show_spanning_tree.txt": SPANNING_TREE,
    "show_running_config.txt": RUNNING_CONFIG,
}


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    """A directory populated with all nine synthetic command outputs."""
    for name, text in SAMPLES.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------ parsers


def test_interface_status_parsing() -> None:
    interfaces = P.parse_interface_status(INTERFACE_STATUS)
    by_name = {i.name: i for i in interfaces}
    assert set(by_name) >= {"Gi0/1", "Gi0/2", "Gi0/3", "Gi0/4"}
    assert by_name["Gi0/1"].mode == "trunk"
    assert by_name["Gi0/1"].vlan is None
    assert by_name["Gi0/1"].description == "Uplink to core"
    assert by_name["Gi0/2"].mode == "access"
    assert by_name["Gi0/2"].vlan == "10"
    assert by_name["Gi0/2"].status == "connected"
    assert by_name["Gi0/4"].status == "disabled"


def test_interface_status_tolerates_missing_columns() -> None:
    text = (
        "Port      Status       Vlan\n"
        "Gi1/1     connected    5\n"
        "Gi1/2     notconnect   trunk\n"
    )
    interfaces = P.parse_interface_status(text)
    assert interfaces[0].name == "Gi1/1"
    assert interfaces[0].vlan == "5" and interfaces[0].mode == "access"
    assert interfaces[1].mode == "trunk"
    assert interfaces[0].speed is None  # column absent -> None, no crash


def test_vlan_parsing() -> None:
    vlans = P.parse_vlan_brief(VLAN_BRIEF)
    by_id = {v.vlan_id: v for v in vlans}
    assert by_id["1"].name == "default"
    assert set(by_id["1"].ports) == {"Gi0/5", "Gi0/6"}
    assert by_id["10"].ports == ("Gi0/2",)
    assert by_id["99"].ports == ()


def test_vlan_port_wrapping() -> None:
    text = (
        "VLAN Name        Status    Ports\n"
        "---- ----------- --------- -----\n"
        "10   users       active    Gi0/1, Gi0/2, Gi0/3\n"
        "                                   Gi0/4, Gi0/5\n"
    )
    vlans = P.parse_vlan_brief(text)
    assert vlans[0].ports == ("Gi0/1", "Gi0/2", "Gi0/3", "Gi0/4", "Gi0/5")


def test_trunk_parsing() -> None:
    trunks = P.parse_trunk(TRUNK)
    assert len(trunks) == 1
    trunk = trunks[0]
    assert trunk.interface == "Gi0/1"
    assert trunk.native_vlan == "99"
    assert trunk.trunking_status == "trunking"
    assert trunk.allowed_vlans == ("1", "10", "20", "30", "99")


def test_trunk_vlan_range_expansion() -> None:
    text = (
        "Port        Mode             Encapsulation  Status        Native vlan\n"
        "Gi0/1       on               802.1q         trunking      1\n"
        "\n"
        "Port        Vlans allowed on trunk\n"
        "Gi0/1       1,10-12,20\n"
    )
    trunks = P.parse_trunk(text)
    assert trunks[0].allowed_vlans == ("1", "10", "11", "12", "20")


def test_lldp_parsing() -> None:
    neighbors = P.parse_lldp_neighbors(LLDP)
    by_local = {n.local_interface: n for n in neighbors}
    assert by_local["Gi0/1"].remote_device == "switch2.example.com"
    assert by_local["Gi0/1"].remote_interface == "Gi0/24"
    assert all(n.protocol == "lldp" for n in neighbors)


def test_cdp_parsing() -> None:
    neighbors = P.parse_cdp_neighbors(CDP)
    by_local = {n.local_interface: n for n in neighbors}
    # Local/remote interfaces contain an internal space ("Gig 0/1").
    assert by_local["Gig 0/1"].remote_device == "switch2"
    assert by_local["Gig 0/1"].remote_interface == "Gig 0/24"
    assert all(n.protocol == "cdp" for n in neighbors)


def test_mac_table_parsing() -> None:
    entries = P.parse_mac_table(MAC_TABLE)
    assert len(entries) == 5
    first = entries[0]
    assert first.vlan == "10"
    assert first.mac_address == "0011.2233.4455"
    assert first.interface == "Gi0/2"
    assert first.entry_type == "DYNAMIC"


def test_poe_parsing() -> None:
    poe = P.parse_power_inline(POWER_INLINE)
    by_iface = {p.interface: p for p in poe}
    assert by_iface["Gi0/5"].powered_device == "IP Phone 7965"
    assert by_iface["Gi0/5"].oper_state == "on"
    assert by_iface["Gi0/5"].power_watts == pytest.approx(15.4)
    assert by_iface["Gi0/5"].poe_class == "3"
    assert by_iface["Gi0/2"].powered_device is None
    assert by_iface["Gi0/1"].admin_state == "off"


def test_stp_parsing() -> None:
    states = P.parse_spanning_tree(SPANNING_TREE)
    blocking = [s for s in states if s.state == "blocking"]
    assert blocking and blocking[0].interface == "Gi0/3"
    assert blocking[0].vlan == "20" and blocking[0].role == "Altn"
    assert any(s.vlan == "1" and s.state == "forwarding" for s in states)


def test_running_config_identity() -> None:
    identity = P.parse_running_config(RUNNING_CONFIG)
    assert identity["hostname"] == "access-sw-01"
    assert identity["management_ip"] == "10.0.0.11"


# --------------------------------------------------------------- inventory


def test_inventory_merge_behavior(sample_dir: Path) -> None:
    inventory = build_inventory(sample_dir, {}, "sample_offline")
    assert len(inventory.devices) == 1
    device = inventory.devices[0].device
    assert device.hostname == "access-sw-01"
    assert device.management_ip == "10.0.0.11"
    by_name = {i.name: i for i in inventory.all_interfaces}
    assert by_name["Gi0/5"].poe_enabled is True
    assert by_name["Gi0/5"].poe_state == "on"
    assert by_name["Gi0/1"].poe_enabled is False   # admin off
    assert by_name["Gi0/3"].poe_enabled is None     # no PoE entry

    summary = derive_summary(inventory)
    assert summary["interface_count"] == 7
    assert summary["access_port_count"] == 6
    assert summary["trunk_port_count"] == 1
    assert summary["poe_enabled_port_count"] == 3
    assert "Gi0/2" in summary["interfaces_with_mac"]
    assert set(summary["unused_ports"]) == {"Gi0/3", "Gi0/4", "Gi0/7"}


def test_missing_file_behavior(tmp_path: Path) -> None:
    (tmp_path / "show_interface_status.txt").write_text(
        INTERFACE_STATUS, encoding="utf-8"
    )
    (tmp_path / "show_vlan_brief.txt").write_text(VLAN_BRIEF, encoding="utf-8")
    inventory = build_inventory(tmp_path, {}, "partial")
    assert "show_mac_address_table.txt" in inventory.files_missing
    assert "show_power_inline.txt" in inventory.files_missing
    assert inventory.files_parsed == ("show_interface_status.txt",
                                      "show_vlan_brief.txt")
    assert len(inventory.all_interfaces) == 7
    assert len(inventory.all_mac_entries) == 0
    assert inventory.warnings  # warnings recorded, no exception


def test_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        build_inventory(Path("does/not/exist"), {}, "x")


def test_configured_filename_override(tmp_path: Path) -> None:
    (tmp_path / "ifaces.txt").write_text(INTERFACE_STATUS, encoding="utf-8")
    config = {"files": {"interface_status": "ifaces.txt"}}
    inventory = build_inventory(tmp_path, config, "custom")
    assert "ifaces.txt" in inventory.files_parsed
    assert len(inventory.all_interfaces) == 7


# ---------------------------------------------------------------- artifacts


def test_artifact_persistence(sample_dir: Path, tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    inventory = build_inventory(sample_dir, {}, "sample_offline")
    write_inventory(inventory, out_root)

    expected = {
        "inventory.json", "interfaces.csv", "vlans.csv", "trunks.csv",
        "neighbors.csv", "mac_table.csv", "poe_status.csv", "stp_state.csv",
        "metadata.json", "network_config_report.md",
    }
    snap_dir = out_root / "sample_offline"
    assert {p.name for p in snap_dir.iterdir()} == expected

    metadata = json.loads((snap_dir / "metadata.json").read_text("utf-8"))
    for field in ("snapshot_id", "timestamp", "input_directory",
                  "files_parsed", "files_missing", "device_count",
                  "interface_count", "vlan_count", "neighbor_count"):
        assert field in metadata
    assert metadata["interface_count"] == 7

    payload = json.loads((snap_dir / "inventory.json").read_text("utf-8"))
    assert payload["devices"][0]["device"]["hostname"] == "access-sw-01"

    with open(snap_dir / "interfaces.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert {"name", "status", "mode", "poe_enabled"} <= set(rows[0])

    with open(snap_dir / "trunks.csv", encoding="utf-8") as handle:
        trunk_rows = list(csv.DictReader(handle))
    assert trunk_rows[0]["allowed_vlans"] == "1;10;20;30;99"  # tuple joined

    report = (snap_dir / "network_config_report.md").read_text("utf-8")
    assert "Network Configuration Report" in report
    assert "access-sw-01" in report


def test_empty_tables_still_write_headers(tmp_path: Path) -> None:
    (tmp_path / "show_running_config.txt").write_text(
        RUNNING_CONFIG, encoding="utf-8"
    )
    inventory = build_inventory(tmp_path, {}, "identity_only")
    paths = write_inventory(inventory, tmp_path / "out")
    with open(paths["interfaces"], encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "name", "status", "protocol_status", "vlan", "mode",
            "description", "speed", "duplex", "poe_enabled", "poe_state",
        ]
        assert list(reader) == []


# ---------------------------------------------------------------------- CLI


def test_cli_argument_parsing() -> None:
    from scripts.analyze_network_config import build_parser

    args = build_parser().parse_args(
        ["--input-dir", "some/dir", "--snapshot-id", "snap1"]
    )
    assert args.input_dir == "some/dir"
    assert args.snapshot_id == "snap1"
    defaults = build_parser().parse_args([])
    assert defaults.input_dir is None and defaults.snapshot_id is None


class _FakeCtx:
    def __init__(self, config: dict, network_config_dir: Path) -> None:
        self.config = config

        class _P:
            pass

        self.paths = _P()
        self.paths.network_config_dir = network_config_dir


def test_cli_main_end_to_end(sample_dir: Path, tmp_path: Path, monkeypatch,
                             caplog) -> None:
    import scripts.analyze_network_config as cli

    out_root = tmp_path / "outputs"
    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx({"network_config": {}}, out_root),
    )
    with caplog.at_level("INFO"):
        code = cli.main(
            ["--input-dir", str(sample_dir), "--snapshot-id", "cli_snap"]
        )
    assert code == 0
    assert (out_root / "cli_snap" / "inventory.json").is_file()
    assert (out_root / "cli_snap" / "network_config_report.md").is_file()


def test_cli_main_missing_directory_returns_1(tmp_path: Path, monkeypatch,
                                              caplog) -> None:
    import scripts.analyze_network_config as cli

    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx({"network_config": {}}, tmp_path / "out"),
    )
    with caplog.at_level("ERROR"):
        code = cli.main(["--input-dir", str(tmp_path / "nope"),
                         "--snapshot-id", "x"])
    assert code == 1
    assert "not found" in caplog.text

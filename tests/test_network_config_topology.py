"""Tests for src.network_config Phase 2 topology (offline, deterministic)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.network_config.artifacts import write_inventory
from src.network_config.inventory import build_inventory, derive_summary
from src.network_config.models import (
    MACAddressEntry,
    Neighbor,
    NetworkDevice,
    NetworkInterface,
    NetworkInventory,
    ParsedDeviceSnapshot,
    STPState,
    TrunkInterface,
)
from src.network_config.reporting import network_config_report
from src.network_config.topology import (
    build_topology,
    topology_summary,
)
from src.network_config.topology_artifacts import write_topology


# ------------------------------------------------------------- tiny builders


def _device(
    device_id: str,
    *,
    hostname: str | None = None,
    management_ip: str | None = None,
    interfaces=(),
    trunks=(),
    neighbors=(),
    mac=(),
    stp=(),
) -> ParsedDeviceSnapshot:
    return ParsedDeviceSnapshot(
        device=NetworkDevice(device_id=device_id, hostname=hostname,
                             management_ip=management_ip),
        interfaces=tuple(interfaces), trunks=tuple(trunks),
        neighbors=tuple(neighbors), mac_entries=tuple(mac), stp_states=tuple(stp),
    )


def _inv(*devices, snapshot_id: str = "t") -> NetworkInventory:
    return NetworkInventory(
        snapshot_id=snapshot_id, input_directory=".", devices=tuple(devices)
    )


def _categories(topology) -> set[str]:
    return {w.category for w in topology.warnings}


# ---------------------------------------------------------------- edges


def test_lldp_edge_creation() -> None:
    inv = _inv(_device("A", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="B",
                 remote_interface="Gi0/2", protocol="lldp"),
    ]))
    topo = build_topology(inv, {})
    assert len(topo.edges) == 1
    edge = topo.edges[0]
    assert edge.discovery_protocol == "lldp"
    assert edge.confidence == "high"
    assert edge.remote_device == "B"
    assert edge.bidirectional is False


def test_cdp_edge_creation() -> None:
    inv = _inv(_device("A", neighbors=[
        Neighbor(local_interface="Gig 0/2", remote_device="router1",
                 remote_interface="Gig 0/0", protocol="cdp"),
    ]))
    topo = build_topology(inv, {})
    assert topo.edges[0].discovery_protocol == "cdp"
    assert topo.edges[0].remote_device == "router1"


def test_lldp_cdp_deduplication() -> None:
    # Same physical port reports the same neighbour via both protocols.
    inv = _inv(_device("A", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="switch2.example.com",
                 remote_interface="Gi0/24", protocol="lldp"),
        Neighbor(local_interface="Gi0/1", remote_device="switch2",
                 remote_interface="Gi0/24", protocol="cdp"),
    ]))
    topo = build_topology(inv, {})
    assert len(topo.edges) == 1
    edge = topo.edges[0]
    assert edge.discovery_protocol == "cdp+lldp"
    # More qualified (FQDN) remote name retained.
    assert edge.remote_device == "switch2.example.com"
    # Both protocols on the port -> no discovery mismatch warning.
    assert "discovery" not in _categories(topo)


def test_reversed_duplicate_normalization() -> None:
    inv = _inv(
        _device("core-sw1", neighbors=[
            Neighbor(local_interface="Gi1/0/1", remote_device="access-sw1",
                     remote_interface="Gi0/1", protocol="lldp"),
        ]),
        _device("access-sw1", neighbors=[
            Neighbor(local_interface="Gi0/1", remote_device="core-sw1",
                     remote_interface="Gi1/0/1", protocol="lldp"),
        ]),
    )
    topo = build_topology(inv, {})
    assert len(topo.edges) == 1
    assert topo.edges[0].bidirectional is True
    # Both directions present -> not flagged unidirectional.
    assert not any("unidirectional" in w.message.lower()
                   for w in topo.warnings)


# ---------------------------------------------------------------- warnings


def test_unidirectional_neighbor_warning() -> None:
    inv = _inv(
        _device("A", neighbors=[
            Neighbor(local_interface="Gi0/1", remote_device="B",
                     remote_interface="Gi0/2", protocol="lldp"),
        ]),
        _device("B"),  # parsed, but reports no neighbour back
    )
    topo = build_topology(inv, {})
    assert any(w.category == "topology" and "unidirectional" in w.message.lower()
               for w in topo.warnings)


def test_lldp_cdp_mismatch_warning() -> None:
    inv = _inv(_device("A", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="B",
                 remote_interface="Gi0/2", protocol="lldp"),
        Neighbor(local_interface="Gi0/2", remote_device="C",
                 remote_interface="Gi0/3", protocol="cdp"),
    ]))
    topo = build_topology(inv, {})
    mismatches = [w for w in topo.warnings if w.category == "discovery"]
    assert len(mismatches) == 2
    assert all("possible protocol mismatch" in w.message for w in mismatches)


def test_mismatch_suppressed_when_only_one_protocol_captured() -> None:
    # Device only has LLDP output -> a missing CDP is not a mismatch.
    inv = _inv(_device("A", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="B",
                 remote_interface="Gi0/2", protocol="lldp"),
    ]))
    topo = build_topology(inv, {})
    assert "discovery" not in _categories(topo)


def test_trunk_without_neighbor_warning() -> None:
    inv = _inv(_device(
        "A",
        interfaces=[NetworkInterface(name="Gi0/10", mode="trunk",
                                     status="connected")],
        trunks=[TrunkInterface(interface="Gi0/10", native_vlan="1")],
    ))
    topo = build_topology(inv, {})
    assert any(w.category == "topology" and "no discovered LLDP/CDP neighbour"
               in w.message for w in topo.warnings)


def test_mac_threshold_warning_on_access_port() -> None:
    macs = [
        MACAddressEntry(vlan="10", mac_address=f"0000.0000.00{i:02d}",
                        interface="Gi0/1")
        for i in range(6)
    ]
    inv = _inv(_device(
        "A",
        interfaces=[NetworkInterface(name="Gi0/1", mode="access", vlan="10")],
        mac=macs,
    ))
    topo = build_topology(inv, {"mac_access_port_threshold": 5})
    hits = [w for w in topo.warnings
            if w.category == "topology" and "threshold" in w.message]
    assert hits and hits[0].interface == "Gi0/1"


def test_mac_multiple_interfaces_warning() -> None:
    inv = _inv(_device("A", mac=[
        MACAddressEntry(vlan="10", mac_address="aaaa.bbbb.cccc",
                        interface="Gi0/1"),
        MACAddressEntry(vlan="10", mac_address="aaaa.bbbb.cccc",
                        interface="Gi0/2"),
    ]))
    topo = build_topology(inv, {})
    loop = [w for w in topo.warnings if w.category == "loop_risk"]
    assert loop and "aaaa.bbbb.cccc" in loop[0].message


def test_stp_blocked_access_port_warning() -> None:
    inv = _inv(_device(
        "A",
        interfaces=[NetworkInterface(name="Gi0/3", mode="access", vlan="20")],
        stp=[STPState(vlan="20", interface="Gi0/3", role="Altn",
                      state="blocking")],
    ))
    topo = build_topology(inv, {})
    assert any(w.category == "stp" and "blocking on access port" in w.message
               for w in topo.warnings)


def test_missing_stp_for_trunk_warning() -> None:
    inv = _inv(_device(
        "A",
        interfaces=[NetworkInterface(name="Gi0/10", mode="trunk")],
        trunks=[TrunkInterface(interface="Gi0/10")],
        neighbors=[Neighbor(local_interface="Gi0/10", remote_device="B",
                            remote_interface="Gi0/1", protocol="lldp")],
        stp=[],
    ))
    topo = build_topology(inv, {"require_stp_on_trunks": True})
    assert any(w.category == "stp" and "no STP data" in w.message
               for w in topo.warnings)
    # Neighbour present -> the trunk is not also flagged as neighbour-less.
    assert not any("no discovered LLDP/CDP neighbour" in w.message
                   for w in topo.warnings)


def test_deterministic_warning_ids() -> None:
    inv = _inv(_device("A", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="B",
                 remote_interface="Gi0/2", protocol="lldp"),
        Neighbor(local_interface="Gi0/2", remote_device="C",
                 remote_interface="Gi0/3", protocol="cdp"),
    ]))
    first = build_topology(inv, {})
    second = build_topology(inv, {})
    assert [w.warning_id for w in first.warnings] == \
        [w.warning_id for w in second.warnings]
    assert first.warnings[0].warning_id == "TW001"


# ---------------------------------------------------------------- summary


def test_topology_summary_counts() -> None:
    inv = _inv(_device("A", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="B",
                 remote_interface="Gi0/2", protocol="lldp"),
    ]))
    summary = topology_summary(build_topology(inv, {}))
    assert summary["edge_count"] == 1
    assert summary["confidence_counts"]["high"] == 1
    assert summary["lldp_cdp_edge_count"] == 1
    assert summary["node_count"] == 2


# ---------------------------------------------------------------- artifacts


def test_topology_artifact_persistence(tmp_path: Path) -> None:
    inv = _inv(_device("A", management_ip="10.0.0.1", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="B",
                 remote_interface="Gi0/2", protocol="lldp"),
    ]))
    topo = build_topology(inv, {})
    paths = write_topology(topo, tmp_path)
    for key in ("topology", "nodes", "edges", "warnings"):
        assert paths[key].is_file()

    payload = json.loads(paths["topology"].read_text("utf-8"))
    assert payload["snapshot_id"] == "t"
    assert payload["summary"]["edge_count"] == 1
    assert payload["edges"][0]["remote_device"] == "B"

    with open(paths["edges"], encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["confidence"] == "high"
    assert rows[0]["bidirectional"] == "false"  # bool lowercased


def test_write_inventory_includes_topology(tmp_path: Path) -> None:
    inv = _inv(_device("A", hostname="A", neighbors=[
        Neighbor(local_interface="Gi0/1", remote_device="B",
                 remote_interface="Gi0/2", protocol="lldp"),
    ]), snapshot_id="snap")
    topo = build_topology(inv, {})
    paths = write_inventory(inv, tmp_path, topo)
    snap_dir = tmp_path / "snap"
    assert (snap_dir / "topology.json").is_file()
    assert (snap_dir / "topology_edges.csv").is_file()
    metadata = json.loads((snap_dir / "metadata.json").read_text("utf-8"))
    assert metadata["topology"]["edge_count"] == 1


def test_write_inventory_without_topology_omits_files(tmp_path: Path) -> None:
    inv = _inv(_device("A"), snapshot_id="snap")
    write_inventory(inv, tmp_path)  # no topology arg
    snap_dir = tmp_path / "snap"
    assert not (snap_dir / "topology.json").exists()
    metadata = json.loads((snap_dir / "metadata.json").read_text("utf-8"))
    assert "topology" not in metadata


# ---------------------------------------------------------------- reporting


def test_report_topology_section() -> None:
    inv = _inv(_device(
        "A", hostname="A",
        interfaces=[NetworkInterface(name="Gi0/1", mode="access", vlan="10")],
        neighbors=[Neighbor(local_interface="Gi0/2", remote_device="B",
                            remote_interface="Gi0/1", protocol="lldp")],
    ))
    topo = build_topology(inv, {})
    report = network_config_report(inv, derive_summary(inv),
                                   topology_summary(topo))
    assert "## Topology" in report
    assert "| Nodes |" in report
    assert "LLDP/CDP edges" in report


def test_report_without_topology_has_no_section() -> None:
    inv = _inv(_device("A", hostname="A"))
    report = network_config_report(inv, derive_summary(inv))
    assert "## Topology" not in report


# ---------------------------------------------------------------- CLI flag


def _write_min_snapshot(directory: Path) -> None:
    (directory / "show_interface_status.txt").write_text(
        "Port      Status       Vlan\n"
        "Gi0/1     connected    trunk\n",
        encoding="utf-8",
    )
    (directory / "show_lldp_neighbors.txt").write_text(
        "Device ID    Local Intf    Hold-time    Capability    Port ID\n"
        "switchX      Gi0/1         120          B             Gi0/24\n",
        encoding="utf-8",
    )


class _FakeCtx:
    def __init__(self, config: dict, network_config_dir: Path) -> None:
        self.config = config

        class _P:
            pass

        self.paths = _P()
        self.paths.network_config_dir = network_config_dir


def test_cli_runs_topology_by_default(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_min_snapshot(src)
    out = tmp_path / "out"
    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx({"network_config": {}}, out),
    )
    assert cli.main(["--input-dir", str(src), "--snapshot-id", "s1"]) == 0
    assert (out / "s1" / "topology.json").is_file()


def test_cli_skip_topology_flag(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_min_snapshot(src)
    out = tmp_path / "out"
    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx({"network_config": {}}, out),
    )
    code = cli.main(
        ["--input-dir", str(src), "--snapshot-id", "s2", "--skip-topology"]
    )
    assert code == 0
    assert (out / "s2" / "inventory.json").is_file()
    assert not (out / "s2" / "topology.json").exists()


def test_cli_topology_disabled_by_config(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_min_snapshot(src)
    out = tmp_path / "out"
    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx(
            {"network_config": {"topology": {"enabled": False}}}, out
        ),
    )
    assert cli.main(["--input-dir", str(src), "--snapshot-id", "s3"]) == 0
    assert not (out / "s3" / "topology.json").exists()

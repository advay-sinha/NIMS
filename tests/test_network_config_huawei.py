"""Tests for the Huawei VRP Engine C parser (offline, read-only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.network_config.vendors import huawei

FIXTURES = Path("datasets/samples/network_config/huawei_s5720")

pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(), reason="Huawei sample fixtures not present"
)


def _inventory():
    return huawei.build_inventory(FIXTURES, {}, "huawei_test")


def test_normalize_ifname_expands_short_names():
    assert huawei.normalize_ifname("GE0/0/1") == "GigabitEthernet0/0/1"
    assert huawei.normalize_ifname("XGE0/0/4") == "XGigabitEthernet0/0/4"
    assert huawei.normalize_ifname("GE0/0/1(U)") == "GigabitEthernet0/0/1"
    assert huawei.normalize_ifname("Vlanif200") == "Vlanif200"


def test_device_identity_parsed():
    dev = _inventory().devices[0].device
    assert dev.hostname == "ADMIN-ILL-HU-SW"
    assert dev.management_ip == "10.90.10.72"
    assert dev.platform and dev.platform.startswith("S5720")


def test_access_and_trunk_modes():
    ifaces = {i.name: i for i in _inventory().devices[0].interfaces}
    assert ifaces["GigabitEthernet0/0/6"].mode == "access"
    assert ifaces["GigabitEthernet0/0/6"].vlan == "200"
    assert ifaces["GigabitEthernet0/0/1"].mode == "trunk"


def test_trunk_allowed_vlans_intersect_defined():
    trunks = {t.interface: t for t in _inventory().devices[0].trunks}
    ge1 = trunks["GigabitEthernet0/0/1"]
    # "allow-pass vlan 2 to 4094" resolves to the VLANs that actually exist.
    assert "200" in ge1.allowed_vlans and "230" in ge1.allowed_vlans
    assert "1" not in ge1.allowed_vlans          # vlan 1 excluded by the 2-4094 range
    assert ge1.native_vlan == "200"


def test_vlan_names_and_counts():
    inv = _inventory()
    vlans = {v.vlan_id: v for v in inv.devices[0].vlans}
    assert vlans["200"].name == "VLAN-NW-MGMT"
    assert len(inv.devices[0].mac_entries) == 476
    assert len(inv.devices[0].neighbors) == 4


def test_stp_and_poe_parsed():
    dev = _inventory().devices[0]
    stp = {s.interface: s for s in dev.stp_states}
    assert stp["GigabitEthernet0/0/1"].state == "forwarding"
    poe = {p.interface: p for p in dev.poe}
    assert poe["GigabitEthernet0/0/1"].oper_state == "on"
    assert poe["GigabitEthernet0/0/1"].power_watts and poe["GigabitEthernet0/0/1"].power_watts > 0


def test_config_fixture_has_no_secrets():
    cfg = (FIXTURES / "display_current_configuration.txt").read_text(encoding="utf-8")
    assert "%^%#" not in cfg
    assert "irreversible-cipher $" not in cfg

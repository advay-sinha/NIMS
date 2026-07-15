"""Tests for Hirschmann config snapshot loading + diffing (Phase 9, offline)."""

from __future__ import annotations

from pathlib import Path

from src.live_logging import config_diff, hirschmann_config

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = (
    ROOT / "datasets" / "samples" / "live_logging" / "hirschmann" / "config_snapshots"
)


def test_parse_config_text_extracts_structure():
    snap = hirschmann_config.load_snapshot(SNAPSHOTS / "switch_01_2026-07-01.cfg")
    assert snap.device_id == "switch_01"
    assert snap.hostname == "SW-IND-01"
    assert 10 in snap.vlans and 20 in snap.vlans
    assert snap.globals["stp_mode"] == "rstp"
    assert snap.globals["telnet_enabled"] is False
    assert snap.interfaces["1/5"]["access_vlan"] == 10
    assert snap.interfaces["1/7"]["poe_enabled"] is True
    # Secret-bearing SNMP community line must NOT be retained anywhere.
    assert "s3cr3t-community" not in str(snap.to_dict())


def test_diff_detects_expected_changes():
    prev = hirschmann_config.load_snapshot(SNAPSHOTS / "switch_01_2026-07-01.cfg")
    curr = hirschmann_config.load_snapshot(SNAPSHOTS / "switch_01_2026-07-02.cfg")
    events = config_diff.diff_configs(prev, curr)
    subs = {e["subcategory"] for e in events}
    assert {
        "access_vlan_change",
        "poe_change",
        "stp_change",
        "mgmt_protocol_change",
        "trunk_vlan_change",
        "vlan_added",
    } <= subs
    # access VLAN change on 1/5 carries the new vlan id.
    vlan_change = next(e for e in events if e["subcategory"] == "access_vlan_change")
    assert vlan_change["normalized_fields"]["to"] == 20
    assert vlan_change["source_key"] == "hirschmann_config"


def test_load_snapshots_dir_groups_and_orders():
    grouped = hirschmann_config.load_snapshots_dir(SNAPSHOTS)
    assert "switch_01" in grouped
    labels = [s.label for s in grouped["switch_01"]]
    assert labels == sorted(labels)  # chronological
    series_events = config_diff.diff_snapshot_series(grouped["switch_01"])
    assert len(series_events) >= 5


def test_diff_is_deterministic():
    prev = hirschmann_config.load_snapshot(SNAPSHOTS / "switch_01_2026-07-01.cfg")
    curr = hirschmann_config.load_snapshot(SNAPSHOTS / "switch_01_2026-07-02.cfg")
    first = [e["raw_ref"] for e in config_diff.diff_configs(prev, curr)]
    second = [e["raw_ref"] for e in config_diff.diff_configs(prev, curr)]
    assert first == second

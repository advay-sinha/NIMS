"""Tests for src.network_config Phase 3 rule engine (offline, deterministic)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.network_config.artifacts import write_inventory
from src.network_config.findings import make_finding_id
from src.network_config.inventory import derive_summary
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
from src.network_config.rules import (
    RULES,
    RuleEngine,
    load_rules_config,
    run_rules,
)
from src.network_config.topology import build_topology


# ------------------------------------------------------------- tiny builders


def _device(device_id: str = "A", **kw) -> ParsedDeviceSnapshot:
    return ParsedDeviceSnapshot(
        device=NetworkDevice(device_id=device_id,
                             hostname=kw.pop("hostname", None)),
        interfaces=tuple(kw.pop("interfaces", ())),
        trunks=tuple(kw.pop("trunks", ())),
        neighbors=tuple(kw.pop("neighbors", ())),
        mac_entries=tuple(kw.pop("mac", ())),
        stp_states=tuple(kw.pop("stp", ())),
    )


def _inv(*devices, snapshot_id: str = "t") -> NetworkInventory:
    return NetworkInventory(
        snapshot_id=snapshot_id, input_directory=".", devices=tuple(devices)
    )


def _only(rule_id: str, **rule_cfg):
    """A rules config that enables exactly one rule."""
    return {"rules": {rule_id: {"enabled": True, **rule_cfg}}}


# -------------------------------------------------------------- yaml loading


def test_yaml_rule_loading() -> None:
    cfg = load_rules_config("configs/network_rules.yaml")
    assert cfg["global"]["enabled"] is True
    assert "TRUNK_MISSING_REQUIRED_VLAN" in cfg["rules"]
    # Every configured rule id is a registered rule.
    for rule_id in cfg["rules"]:
        assert rule_id in RULES


def test_load_rules_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rules_config(tmp_path / "nope.yaml")


# ------------------------------------------------------------------ rules


def test_disabled_rule_ignored() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/1", mode="access", vlan="999")]))
    cfg = {"rules": {"ACCESS_PORT_DISALLOWED_VLAN":
                     {"enabled": False, "allowed_vlans": [10]}}}
    findings, summary = run_rules(inv, None, cfg)
    assert findings == []
    assert "ACCESS_PORT_DISALLOWED_VLAN" in summary["rules_disabled"]
    assert "ACCESS_PORT_DISALLOWED_VLAN" not in summary["rules_evaluated"]


def test_access_port_disallowed_vlan() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/1", mode="access", vlan="999"),
        NetworkInterface(name="Gi0/2", mode="access", vlan="10"),
    ]))
    findings, _ = run_rules(
        inv, None, _only("ACCESS_PORT_DISALLOWED_VLAN", allowed_vlans=[10, 20])
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "ACCESS_PORT_DISALLOWED_VLAN"
    assert findings[0].interface == "Gi0/1" and findings[0].vlan == "999"


def test_trunk_missing_required_vlan() -> None:
    inv = _inv(_device(trunks=[
        TrunkInterface(interface="Gi0/1", allowed_vlans=("10", "20"))]))
    findings, _ = run_rules(
        inv, None, _only("TRUNK_MISSING_REQUIRED_VLAN",
                         required_vlans=[10, 20, 30])
    )
    assert len(findings) == 1
    assert "30" in findings[0].evidence


def test_trunk_unauthorized_vlan() -> None:
    inv = _inv(_device(trunks=[
        TrunkInterface(interface="Gi0/1", allowed_vlans=("10", "20", "999"))]))
    findings, _ = run_rules(
        inv, None, _only("TRUNK_UNAUTHORIZED_VLAN", authorized_vlans=[10, 20])
    )
    assert len(findings) == 1
    assert "999" in findings[0].evidence


def test_poe_disabled_expected_device() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/5", mode="access", description="IP Phone 1",
                         poe_enabled=False),
        NetworkInterface(name="Gi0/6", mode="access", description="AP-lobby",
                         poe_enabled=True, poe_state="on"),
    ]))
    findings, _ = run_rules(
        inv, None, _only("POE_DISABLED_EXPECTED",
                         expected_poe_keywords=["phone", "ap"])
    )
    assert len(findings) == 1
    assert findings[0].interface == "Gi0/5"


def test_unused_admin_up_port() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/3", mode="access", status="notconnect"),
        NetworkInterface(name="Gi0/4", mode="access", status="disabled"),
        NetworkInterface(name="Gi0/1", mode="trunk", status="connected"),
    ]))
    findings, _ = run_rules(inv, None, _only("UNUSED_PORT_ADMIN_UP"))
    assert [f.interface for f in findings] == ["Gi0/3"]  # not disabled/connected


def test_access_port_too_many_macs() -> None:
    macs = [MACAddressEntry(vlan="10", mac_address=f"0000.0000.00{i:02d}",
                            interface="Gi0/1") for i in range(3)]
    inv = _inv(_device(
        interfaces=[NetworkInterface(name="Gi0/1", mode="access", vlan="10")],
        mac=macs,
    ))
    findings, _ = run_rules(
        inv, None, _only("ACCESS_PORT_TOO_MANY_MACS", threshold=2)
    )
    assert len(findings) == 1 and findings[0].interface == "Gi0/1"


def test_stp_blocking_access_port() -> None:
    inv = _inv(_device(
        interfaces=[NetworkInterface(name="Gi0/3", mode="access", vlan="20")],
        stp=[STPState(vlan="20", interface="Gi0/3", role="Altn",
                      state="blocking")],
    ))
    findings, _ = run_rules(inv, None, _only("STP_BLOCKING_ACCESS_PORT"))
    assert len(findings) == 1 and findings[0].vlan == "20"


def test_trunk_without_neighbor_with_topology() -> None:
    inv = _inv(_device(
        interfaces=[NetworkInterface(name="Gi0/10", mode="trunk")],
        trunks=[TrunkInterface(interface="Gi0/10")],
    ))
    topo = build_topology(inv, {})
    findings, summary = run_rules(inv, topo, _only("TRUNK_WITHOUT_NEIGHBOR"))
    assert len(findings) == 1
    assert findings[0].rule_id == "TRUNK_WITHOUT_NEIGHBOR"
    assert findings[0].source == "topology"
    assert "TRUNK_WITHOUT_NEIGHBOR" in summary["rules_evaluated"]


def test_topology_rule_skipped_when_topology_missing() -> None:
    inv = _inv(_device(
        interfaces=[NetworkInterface(name="Gi0/10", mode="trunk")],
        trunks=[TrunkInterface(interface="Gi0/10")],
    ))
    findings, summary = run_rules(inv, None, _only("TRUNK_WITHOUT_NEIGHBOR"))
    assert findings == []
    assert "TRUNK_WITHOUT_NEIGHBOR" in summary["rules_skipped"]
    assert "TRUNK_WITHOUT_NEIGHBOR" not in summary["rules_evaluated"]
    assert "TRUNK_WITHOUT_NEIGHBOR" in summary["rules_enabled"]


# --------------------------------------------------------------- suppression


def test_suppression_by_rule_id() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/3", mode="access", status="notconnect")]))
    cfg = {
        "rules": {"UNUSED_PORT_ADMIN_UP": {"enabled": True}},
        "suppression": {"enabled": True,
                        "items": [{"rule_id": "UNUSED_PORT_ADMIN_UP"}]},
    }
    findings, summary = run_rules(inv, None, cfg)
    assert len(findings) == 1 and findings[0].status == "suppressed"
    assert summary["suppressed_count"] == 1
    assert summary["total_findings"] == 0  # open count excludes suppressed


def test_suppression_by_device_interface() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/3", mode="access", status="notconnect"),
        NetworkInterface(name="Gi0/7", mode="access", status="notconnect"),
    ]))
    cfg = {
        "rules": {"UNUSED_PORT_ADMIN_UP": {"enabled": True}},
        "suppression": {"enabled": True,
                        "items": [{"device": "A", "interface": "Gi0/7"}]},
    }
    findings, summary = run_rules(inv, None, cfg)
    statuses = {f.interface: f.status for f in findings}
    assert statuses == {"Gi0/3": "open", "Gi0/7": "suppressed"}
    assert summary["total_findings"] == 1 and summary["suppressed_count"] == 1


def test_suppression_by_tag() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/3", mode="access", status="notconnect")]))
    cfg = {
        "rules": {"UNUSED_PORT_ADMIN_UP": {"enabled": True,
                                           "tags": ["hygiene"]}},
        "suppression": {"enabled": True, "items": [{"tag": "hygiene"}]},
    }
    findings, summary = run_rules(inv, None, cfg)
    assert findings[0].status == "suppressed"
    assert summary["suppressed_count"] == 1


# ------------------------------------------------------------ ids & summary


def test_deterministic_finding_ids() -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/3", mode="access", status="notconnect")]))
    first, _ = run_rules(inv, None, _only("UNUSED_PORT_ADMIN_UP"))
    second, _ = run_rules(inv, None, _only("UNUSED_PORT_ADMIN_UP"))
    assert first[0].finding_id == second[0].finding_id
    assert first[0].finding_id == make_finding_id(
        "UNUSED_PORT_ADMIN_UP", "A", "Gi0/3", None
    )


def test_rule_summary_correctness() -> None:
    inv = _inv(_device(
        interfaces=[
            NetworkInterface(name="Gi0/1", mode="access", vlan="999"),
            NetworkInterface(name="Gi0/3", mode="access", status="notconnect"),
        ],
    ))
    cfg = {"rules": {
        "ACCESS_PORT_DISALLOWED_VLAN": {"enabled": True, "allowed_vlans": [10]},
        "UNUSED_PORT_ADMIN_UP": {"enabled": True},
    }}
    findings, summary = run_rules(inv, None, cfg)
    assert summary["total_findings"] == 2
    assert summary["findings_by_severity"] == {"high": 1, "low": 1}
    assert summary["findings_by_category"] == {"port": 1, "vlan": 1}
    assert sorted(summary["rules_evaluated"]) == [
        "ACCESS_PORT_DISALLOWED_VLAN", "UNUSED_PORT_ADMIN_UP"]
    # Findings are sorted most-severe first.
    assert findings[0].severity == "high"


# --------------------------------------------------------------- artifacts


def test_findings_artifact_persistence(tmp_path: Path) -> None:
    inv = _inv(_device(interfaces=[
        NetworkInterface(name="Gi0/1", mode="access", vlan="999")],
    ), snapshot_id="snap")
    findings, summary = run_rules(
        inv, None, _only("ACCESS_PORT_DISALLOWED_VLAN", allowed_vlans=[10])
    )
    paths = write_inventory(inv, tmp_path, None, findings, summary)
    snap = tmp_path / "snap"
    for name in ("findings.json", "findings.csv", "rule_summary.json"):
        assert (snap / name).is_file()

    payload = json.loads((snap / "findings.json").read_text("utf-8"))
    assert payload[0]["rule_id"] == "ACCESS_PORT_DISALLOWED_VLAN"
    assert payload[0]["tags"] == []  # empty tuple -> []

    with open(snap / "findings.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["severity"] == "high"

    metadata = json.loads((snap / "metadata.json").read_text("utf-8"))
    assert metadata["findings"]["total_findings"] == 1


# --------------------------------------------------------------- reporting


def test_report_findings_section() -> None:
    inv = _inv(_device(hostname="A", interfaces=[
        NetworkInterface(name="Gi0/1", mode="access", vlan="999")]))
    findings, summary = run_rules(
        inv, None, _only("ACCESS_PORT_DISALLOWED_VLAN", allowed_vlans=[10])
    )
    report_summary = {**summary, "top_findings": [
        {"severity": f.severity, "rule_id": f.rule_id, "title": f.title,
         "device": f.device, "interface": f.interface, "evidence": f.evidence}
        for f in findings if f.severity in {"critical", "high"}
    ]}
    report = network_config_report(inv, derive_summary(inv), None,
                                   report_summary)
    assert "## Rule findings" in report
    assert "Total open findings" in report
    assert "ACCESS_PORT_DISALLOWED_VLAN" in report


# ----------------------------------------------------------------- CLI flags


def _write_snapshot(directory: Path) -> None:
    (directory / "show_interface_status.txt").write_text(
        "Port      Status       Vlan\n"
        "Gi0/1     connected    trunk\n"
        "Gi0/9     notconnect   40\n",
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


def test_cli_runs_rules_by_default(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_snapshot(src)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap",
                        lambda args: _FakeCtx({"network_config": {}}, out))
    assert cli.main(["--input-dir", str(src), "--snapshot-id", "s1"]) == 0
    assert (out / "s1" / "findings.json").is_file()
    assert (out / "s1" / "rule_summary.json").is_file()


def test_cli_skip_rules(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_snapshot(src)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap",
                        lambda args: _FakeCtx({"network_config": {}}, out))
    code = cli.main(["--input-dir", str(src), "--snapshot-id", "s2",
                     "--skip-rules"])
    assert code == 0
    assert (out / "s2" / "inventory.json").is_file()
    assert not (out / "s2" / "findings.json").exists()


def test_cli_custom_rules_config(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_snapshot(src)  # Gi0/9 is notconnect -> UNUSED_PORT_ADMIN_UP
    rules_file = tmp_path / "custom_rules.yaml"
    rules_file.write_text(
        "global:\n  enabled: true\n"
        "rules:\n"
        "  UNUSED_PORT_ADMIN_UP:\n"
        "    enabled: true\n"
        "    severity: critical\n"   # non-default (default is low)
        "    category: port\n"
        "suppression:\n  enabled: true\n  items: []\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap",
                        lambda args: _FakeCtx({"network_config": {}}, out))
    code = cli.main(["--input-dir", str(src), "--snapshot-id", "s3",
                     "--rules-config", str(rules_file)])
    assert code == 0
    payload = json.loads(
        (out / "s3" / "findings.json").read_text("utf-8")
    )
    # Only the custom rule ran, and its custom severity was applied.
    assert {f["rule_id"] for f in payload} == {"UNUSED_PORT_ADMIN_UP"}
    assert payload[0]["severity"] == "critical"

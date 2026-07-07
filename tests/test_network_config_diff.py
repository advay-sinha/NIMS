"""Tests for src.network_config Phase 6 snapshot diff + verification (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.network_config.diff import (
    SnapshotData,
    SnapshotDiffer,
    load_snapshot,
)
from src.network_config.diff_artifacts import build_diff_summary, write_diff
from src.network_config.verification import verify_remediation


# ------------------------------------------------------------------ builders


def _iface(name, status="connected", vlan=None, mode="access",
           description=None, poe_enabled=None, poe_state=None):
    return {"name": name, "status": status, "protocol_status": None,
            "vlan": vlan, "mode": mode, "description": description,
            "speed": None, "duplex": None, "poe_enabled": poe_enabled,
            "poe_state": poe_state}


def _trunk(interface, allowed_vlans=(), native_vlan=None,
           trunking_status="trunking"):
    return {"interface": interface, "allowed_vlans": list(allowed_vlans),
            "native_vlan": native_vlan, "trunking_status": trunking_status}


def _vlan(vlan_id, name=None, status="active"):
    return {"vlan_id": vlan_id, "name": name, "status": status, "ports": []}


def _poe(interface, admin_state=None, oper_state=None):
    return {"interface": interface, "admin_state": admin_state,
            "oper_state": oper_state, "power_watts": None,
            "powered_device": None, "poe_class": None, "max_watts": None}


def _stp(interface, vlan, state, role=None):
    return {"vlan": vlan, "interface": interface, "role": role, "state": state}


def _inv(snapshot_id="s", device="swA", interfaces=(), vlans=(), trunks=(),
         poe=(), stp=()):
    return {
        "snapshot_id": snapshot_id, "input_directory": ".",
        "files_parsed": [], "files_missing": [], "warnings": [],
        "devices": [{
            "device": {"device_id": device, "hostname": device,
                       "platform": None, "management_ip": None,
                       "source_files": []},
            "interfaces": list(interfaces), "vlans": list(vlans),
            "trunks": list(trunks), "poe": list(poe), "neighbors": [],
            "mac_entries": [], "stp_states": list(stp),
        }],
    }


def _topo(edges=(), nodes=(("swA",), ("swB",)), warnings=()):
    return {
        "snapshot_id": "s", "summary": {},
        "nodes": [{"node_id": n[0], "hostname": n[0], "device_type": "switch",
                   "management_ip": None, "source": "lldp"} for n in nodes],
        "edges": list(edges),
        "warnings": list(warnings),
    }


def _edge(local="swA", li="Gi0/1", remote="swB", ri="Gi0/2",
          confidence="high", protocol="lldp"):
    return {"local_device": local, "local_interface": li,
            "remote_device": remote, "remote_interface": ri,
            "discovery_protocol": protocol, "confidence": confidence,
            "evidence": "", "bidirectional": True}


def _finding(rule_id, finding_id="F1", device="swA", interface="Gi0/1",
             vlan=None, category="vlan", severity="medium", status="open",
             details=None):
    return {"finding_id": finding_id, "rule_id": rule_id, "title": rule_id,
            "severity": severity, "category": category, "device": device,
            "interface": interface, "vlan": vlan, "status": status,
            "evidence": None, "recommendation": None, "confidence": "high",
            "source": "rule_engine", "tags": [], "details": details or {}}


def _action(rule_id, finding_id="F1", device="swA", interface="Gi0/1",
            action_type="vlan_trunk_add", commands=(), status="planned"):
    return {"action_id": f"ACT-{finding_id}", "finding_id": finding_id,
            "rule_id": rule_id, "title": rule_id, "severity": "medium",
            "action_type": action_type, "device": device,
            "interface": interface, "vlan": None, "commands": list(commands),
            "rollback": {"commands": [], "note": ""}, "verification_steps": [],
            "safety_checks": [], "requires_confirmation": True,
            "dry_run_only": True, "status": status, "reason": None,
            "risk_level": "medium", "source": "rule_engine", "tags": []}


def _plan(*actions):
    return {"snapshot_id": "s", "actions": list(actions)}


def _snap(inventory, topology=None, findings=None, remediation=None, sid="s"):
    return SnapshotData(snapshot_id=sid, directory=".", inventory=inventory,
                        topology=topology, findings=findings,
                        remediation=remediation)


def _by(diff, category=None, field=None, change_type=None):
    return [r for r in diff.records
            if (category is None or r.category == category)
            and (field is None or r.field == field)
            and (change_type is None or r.change_type == change_type)]


# ------------------------------------------------------------- inventory diff


def test_interface_status_change_detected():
    diff = SnapshotDiffer().diff(
        _snap(_inv(interfaces=[_iface("Gi0/1", status="connected")])),
        _snap(_inv(interfaces=[_iface("Gi0/1", status="notconnect")])))
    recs = _by(diff, "interface", "status", "changed")
    assert recs and recs[0].before_value == "connected"
    assert recs[0].after_value == "notconnect"


def test_access_vlan_change_detected():
    diff = SnapshotDiffer().diff(
        _snap(_inv(interfaces=[_iface("Gi0/1", vlan="10")])),
        _snap(_inv(interfaces=[_iface("Gi0/1", vlan="20")])))
    recs = _by(diff, "interface", "vlan", "changed")
    assert recs and recs[0].before_value == "10" and recs[0].after_value == "20"


def test_trunk_allowed_vlan_change_detected():
    diff = SnapshotDiffer().diff(
        _snap(_inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10"])])),
        _snap(_inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10", "30"])])))
    recs = _by(diff, "trunk", "allowed_vlans", "changed")
    assert recs and "30" in recs[0].after_value


def test_poe_state_change_detected():
    diff = SnapshotDiffer().diff(
        _snap(_inv(poe=[_poe("Gi0/5", admin_state="never")])),
        _snap(_inv(poe=[_poe("Gi0/5", admin_state="auto")])))
    recs = _by(diff, "poe", "admin_state", "changed")
    assert recs and recs[0].after_value == "auto"


def test_vlan_added_and_removed_detected():
    diff = SnapshotDiffer().diff(
        _snap(_inv(vlans=[_vlan("10")])),
        _snap(_inv(vlans=[_vlan("10"), _vlan("20")])))
    assert _by(diff, "vlan", change_type="added")
    diff2 = SnapshotDiffer().diff(
        _snap(_inv(vlans=[_vlan("10"), _vlan("20")])),
        _snap(_inv(vlans=[_vlan("10")])))
    assert _by(diff2, "vlan", change_type="removed")


def test_stp_state_change_detected():
    diff = SnapshotDiffer().diff(
        _snap(_inv(stp=[_stp("Gi0/3", "20", "forwarding")])),
        _snap(_inv(stp=[_stp("Gi0/3", "20", "blocking")])))
    recs = _by(diff, "stp", "state", "changed")
    assert recs and recs[0].after_value == "blocking"


# -------------------------------------------------------------- topology diff


def test_topology_edge_added_and_removed_detected():
    before = _snap(_inv(), topology=_topo(edges=[_edge()]))
    after = _snap(_inv(), topology=_topo(edges=[]))
    diff = SnapshotDiffer().diff(before, after)
    removed = [r for r in diff.records
               if r.category == "topology" and r.change_type == "removed"
               and r.field == "edge"]
    assert removed
    diff2 = SnapshotDiffer().diff(after, before)
    added = [r for r in diff2.records
             if r.category == "topology" and r.change_type == "added"
             and r.field == "edge"]
    assert added


def test_topology_missing_warns_and_continues():
    before = _snap(_inv(), topology=_topo(edges=[_edge()]))
    after = _snap(_inv(), topology=None)
    diff = SnapshotDiffer().diff(before, after)
    assert any("topology" in w for w in diff.warnings)
    # Non-topology comparison still runs.
    assert isinstance(diff.records, tuple)


# --------------------------------------------------------------- findings diff


def test_finding_resolved_new_persistent():
    before = _snap(_inv(), findings=[
        _finding("R_RESOLVED", finding_id="A", interface="Gi0/1"),
        _finding("R_PERSIST", finding_id="B", interface="Gi0/2"),
    ])
    after = _snap(_inv(), findings=[
        _finding("R_PERSIST", finding_id="B2", interface="Gi0/2"),
        _finding("R_NEW", finding_id="C", interface="Gi0/3"),
    ])
    diff = SnapshotDiffer().diff(before, after)
    assert diff.findings_resolved == 1
    assert diff.findings_new == 1
    assert diff.findings_persistent == 1
    assert _by(diff, "finding", change_type="removed")   # resolved
    assert _by(diff, "finding", change_type="added")     # new


# --------------------------------------------------------- remediation verify


def test_verification_passed_for_trunk_missing_vlan():
    before = _snap(
        _inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10"])]),
        findings=[_finding("TRUNK_MISSING_REQUIRED_VLAN",
                           details={"missing_vlans": ["30"]})],
        remediation=_plan(_action("TRUNK_MISSING_REQUIRED_VLAN")))
    after = _snap(_inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10", "30"])]),
                  findings=[])
    results = verify_remediation(before, after)
    assert results[0].status == "passed"


def test_verification_failed_for_trunk_missing_vlan():
    before = _snap(
        _inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10"])]),
        findings=[_finding("TRUNK_MISSING_REQUIRED_VLAN",
                           details={"missing_vlans": ["30"]})],
        remediation=_plan(_action("TRUNK_MISSING_REQUIRED_VLAN")))
    after = _snap(_inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10"])]),
                  findings=[_finding("TRUNK_MISSING_REQUIRED_VLAN",
                                     details={"missing_vlans": ["30"]})])
    results = verify_remediation(before, after)
    assert results[0].status == "failed"


def test_verification_unknown_when_data_missing():
    before = _snap(
        _inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10"])]),
        findings=[_finding("TRUNK_MISSING_REQUIRED_VLAN",
                           details={"missing_vlans": ["30"]})],
        remediation=_plan(_action("TRUNK_MISSING_REQUIRED_VLAN")))
    after = _snap(_inv(), findings=None)   # no trunk data in after
    results = verify_remediation(before, after)
    assert results[0].status == "unknown"


def test_investigation_action_not_applicable():
    before = _snap(
        _inv(),
        findings=[_finding("STP_BLOCKING_ACCESS_PORT", category="stp")],
        remediation=_plan(_action("STP_BLOCKING_ACCESS_PORT",
                                  action_type="investigation")))
    after = _snap(_inv(),
                  findings=[_finding("STP_BLOCKING_ACCESS_PORT",
                                     category="stp")])   # still present
    results = verify_remediation(before, after)
    assert results[0].status == "not_applicable"


def test_investigation_passed_when_finding_resolved():
    before = _snap(
        _inv(),
        findings=[_finding("STP_BLOCKING_ACCESS_PORT", category="stp")],
        remediation=_plan(_action("STP_BLOCKING_ACCESS_PORT",
                                  action_type="investigation")))
    after = _snap(_inv(), findings=[])   # finding gone
    results = verify_remediation(before, after)
    assert results[0].status == "passed"


# ---------------------------------------------------------------- artifacts


def test_diff_artifact_persistence(tmp_path: Path):
    before = _snap(_inv(interfaces=[_iface("Gi0/1", status="connected")]))
    after = _snap(_inv(interfaces=[_iface("Gi0/1", status="notconnect")]))
    diff = SnapshotDiffer().diff(before, after)
    verifications = verify_remediation(before, after)
    paths = write_diff(diff, verifications, tmp_path)
    for key in ("diff_json", "diff_csv", "verification_json",
                "verification_csv", "summary", "report"):
        assert paths[key].is_file()
    payload = json.loads(paths["diff_json"].read_text("utf-8"))
    assert payload["safety_note"] == "offline comparison only, no commands executed"
    assert payload["records"]


def test_diff_summary_correctness():
    before = _snap(_inv(interfaces=[_iface("Gi0/1", status="connected")]),
                   findings=[_finding("R_OLD", finding_id="A")])
    after = _snap(_inv(interfaces=[_iface("Gi0/1", status="notconnect")]),
                  findings=[_finding("R_NEW", finding_id="B", interface="Gi0/2")])
    diff = SnapshotDiffer().diff(before, after)
    summary = build_diff_summary(diff, [])
    assert summary["total_changes"] >= 1
    assert summary["findings_new"] == 1
    assert summary["findings_resolved"] == 1
    assert summary["safety_note"] == "offline comparison only, no commands executed"
    assert "interface" in summary["changes_by_category"]


def test_report_generation(tmp_path: Path):
    before = _snap(
        _inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10"])]),
        findings=[_finding("TRUNK_MISSING_REQUIRED_VLAN",
                           details={"missing_vlans": ["30"]})],
        remediation=_plan(_action("TRUNK_MISSING_REQUIRED_VLAN")))
    after = _snap(_inv(trunks=[_trunk("Gi0/1", allowed_vlans=["10", "30"])]),
                  findings=[])
    diff = SnapshotDiffer().diff(before, after)
    verifications = verify_remediation(before, after)
    paths = write_diff(diff, verifications, tmp_path)
    report = paths["report"].read_text("utf-8")
    assert "# Network Diff Report" in report
    assert "No commands were executed." in report
    assert "## Remediation verification" in report


# --------------------------------------------------------------- load / CLI


def _write_snapshot_dir(root: Path, name: str, inventory: dict,
                        topology=None, findings=None, remediation=None):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inventory.json").write_text(json.dumps(inventory), "utf-8")
    if topology is not None:
        (directory / "topology.json").write_text(json.dumps(topology), "utf-8")
    if findings is not None:
        (directory / "findings.json").write_text(json.dumps(findings), "utf-8")
    if remediation is not None:
        (directory / "remediation_plan.json").write_text(
            json.dumps(remediation), "utf-8")
    return directory


def test_load_snapshot_missing_inventory(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "empty")


def test_load_snapshot_optional_warnings(tmp_path: Path):
    _write_snapshot_dir(tmp_path, "s", _inv())
    snap = load_snapshot(tmp_path / "s")
    assert snap.topology is None
    assert any("topology" in w for w in snap.warnings)


class _FakeCtx:
    def __init__(self, network_config_dir: Path):
        self.config = {"network_config": {}}

        class _P:
            pass

        self.paths = _P()
        self.paths.network_config_dir = network_config_dir


def test_cli_happy_path(tmp_path: Path, monkeypatch):
    import scripts.compare_network_snapshots as cli

    root = tmp_path / "network_config"
    _write_snapshot_dir(root, "before",
                        _inv(interfaces=[_iface("Gi0/1", status="connected")]))
    _write_snapshot_dir(root, "after",
                        _inv(interfaces=[_iface("Gi0/1", status="notconnect")]))
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    code = cli.main(["--before", "before", "--after", "after"])
    assert code == 0
    out = root / "diffs" / "before__to__after"
    assert (out / "snapshot_diff.json").is_file()
    assert (out / "diff_summary.json").is_file()
    assert (out / "network_diff_report.md").is_file()


def test_cli_missing_inventory_errors(tmp_path: Path, monkeypatch):
    import scripts.compare_network_snapshots as cli

    root = tmp_path / "network_config"
    _write_snapshot_dir(root, "before", _inv())
    (root / "after").mkdir(parents=True)   # no inventory.json
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    code = cli.main(["--before", "before", "--after", "after"])
    assert code == 1


def test_cli_optional_topology_missing_warns(tmp_path: Path, monkeypatch, caplog):
    import scripts.compare_network_snapshots as cli

    root = tmp_path / "network_config"
    _write_snapshot_dir(root, "before", _inv(), topology=_topo(edges=[_edge()]))
    _write_snapshot_dir(root, "after", _inv())   # no topology.json
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    code = cli.main(["--before", "before", "--after", "after"])
    assert code == 0
    summary = json.loads(
        (root / "diffs" / "before__to__after" / "diff_summary.json").read_text(
            "utf-8"))
    assert any("topology" in w for w in summary["warnings"])

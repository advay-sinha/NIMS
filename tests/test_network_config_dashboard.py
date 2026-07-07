"""Tests for src.network_config Phase 9 dashboard export (offline, read-only)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.network_config.dashboard_export import (
    EXPORT_VERSION,
    build_dashboard,
)
from src.network_config.intelligence import DiffArtifacts, SnapshotArtifacts


# ------------------------------------------------------------------ builders


def _iface(name="Gi0/1", status="connected", vlan="10", mode="access",
           poe_enabled=None, poe_state=None):
    return {"name": name, "status": status, "protocol_status": None,
            "vlan": vlan, "mode": mode, "description": None, "speed": None,
            "duplex": None, "poe_enabled": poe_enabled, "poe_state": poe_state}


def _device(device_id="swA", interfaces=None, trunks=(), vlans=(), poe=(),
            stp=(), neighbors=()):
    return {"device": {"device_id": device_id, "hostname": f"{device_id}-host",
                       "platform": "ios", "management_ip": "10.0.0.1",
                       "source_files": []},
            "interfaces": list(interfaces if interfaces is not None
                               else [_iface()]),
            "vlans": list(vlans), "trunks": list(trunks), "poe": list(poe),
            "neighbors": list(neighbors), "mac_entries": [],
            "stp_states": list(stp)}


def _inv(devices=None):
    return {"snapshot_id": "s", "input_directory": ".", "files_parsed": [],
            "files_missing": [], "warnings": [],
            "devices": devices if devices is not None else [_device()]}


def _topo(nodes=(("swA", "switch"),), edges=(), warnings=()):
    return {"snapshot_id": "s",
            "summary": {"node_count": len(nodes), "edge_count": len(edges),
                        "warning_count": len(warnings)},
            "nodes": [{"node_id": n[0], "hostname": f"{n[0]}-host",
                       "device_type": n[1], "management_ip": None,
                       "source": "lldp"} for n in nodes],
            "edges": list(edges), "warnings": list(warnings)}


def _edge(local="swA", li="Gi0/1", remote="swB", ri="Gi0/2",
          confidence="high", protocol="lldp"):
    return {"local_device": local, "local_interface": li,
            "remote_device": remote, "remote_interface": ri,
            "discovery_protocol": protocol, "confidence": confidence,
            "evidence": "", "bidirectional": True}


def _finding(rule_id="R", finding_id=None, device="swA", interface="Gi0/1",
             vlan=None, category="vlan", severity="medium", status="open"):
    return {"finding_id": finding_id or f"{rule_id}-x", "rule_id": rule_id,
            "title": rule_id, "severity": severity, "category": category,
            "device": device, "interface": interface, "vlan": vlan,
            "status": status, "evidence": "e", "recommendation": "r",
            "confidence": "high", "source": "rule", "tags": [], "details": {}}


def _action(rule_id="R", finding_id="R-x", device="swA", interface="Gi0/1",
            action_type="port_shutdown", commands=("shutdown",),
            status="planned", risk_level="medium"):
    return {"action_id": f"ACT-{finding_id}", "finding_id": finding_id,
            "rule_id": rule_id, "title": rule_id, "severity": "medium",
            "action_type": action_type, "device": device, "interface": interface,
            "vlan": None, "commands": list(commands),
            "rollback": {"commands": [], "note": ""}, "verification_steps": [],
            "safety_checks": [], "requires_confirmation": True,
            "dry_run_only": True, "status": status, "reason": None,
            "risk_level": risk_level, "source": "rule", "tags": []}


def _artifacts(directory="/snap", **kw):
    defaults = dict(snapshot_id="s", inventory=_inv(), findings=[])
    defaults.update(kw)
    return SnapshotArtifacts(directory=directory, **defaults)


def _diff(diff_id="d", diff_summary=None, snapshot_diff=None, verifications=()):
    return DiffArtifacts(diff_id=diff_id, snapshot_diff=snapshot_diff,
                         verification_results=list(verifications),
                         diff_summary=diff_summary or {})


# ------------------------------------------------------------------ summary


def test_dashboard_summary_generation():
    artifacts = _artifacts(
        inventory=_inv([_device(interfaces=[_iface(), _iface("Gi0/2")])]),
        findings=[_finding(severity="high")],
        remediation_summary={"total_actions": 1, "command_actions": 1,
                             "investigation_actions": 0, "blocked_actions": 0})
    summary = build_dashboard(artifacts)["dashboard_summary"]
    assert summary["device_count"] == 1
    assert summary["interface_count"] == 2
    assert summary["finding_count"] == 1
    assert summary["command_action_count"] == 1
    assert summary["batfish_available"] is False
    assert summary["diff_available"] is False
    assert "no commands were executed" in summary["safety_note"].lower()
    assert any(d["device"] == "swA" for d in summary["top_risk_devices"])


# --------------------------------------------------------------- inventory


def test_inventory_view_grouping():
    inv = _inv([_device("swA", interfaces=[_iface("Gi0/1")],
                        trunks=[{"interface": "Gi0/1", "allowed_vlans": ["10"],
                                 "native_vlan": "1", "trunking_status": "on"}]),
                _device("swB", interfaces=[_iface("Gi0/9")])])
    view = build_dashboard(_artifacts(inventory=inv))["inventory_view"]
    assert {d["device_id"] for d in view["devices"]} == {"swA", "swB"}
    assert "swA" in view["interfaces_by_device"]
    assert view["trunks_by_device"]["swA"][0]["interface"] == "Gi0/1"
    assert view["interfaces_by_device"]["swB"][0]["name"] == "Gi0/9"


# --------------------------------------------------------------- topology


def test_topology_view_generation():
    topo = _topo(nodes=(("swA", "switch"), ("swB", "switch")),
                 edges=[_edge()],
                 warnings=[{"warning_id": "TW001", "severity": "medium",
                            "category": "topology", "message": "m",
                            "device": "swA", "interface": "Gi0/1",
                            "evidence": ""}])
    artifacts = _artifacts(topology=topo, findings=[_finding(device="swA")])
    view = build_dashboard(artifacts)["topology_view"]
    assert len(view["nodes"]) == 2
    node_a = next(n for n in view["nodes"] if n["id"] == "swA")
    assert node_a["finding_count"] == 1
    assert node_a["risk_score"] >= 0
    edge = view["edges"][0]
    assert edge["source"] == "swA" and edge["target"] == "swB"
    assert edge["warning_count"] == 1


def test_topology_view_absent():
    view = build_dashboard(_artifacts(topology=None))["topology_view"]
    assert view["available"] is False
    assert view["nodes"] == []


# --------------------------------------------------------------- findings


def test_findings_grouping():
    findings = [
        _finding("R1", finding_id="a", severity="high", category="vlan",
                 device="swA"),
        _finding("R2", finding_id="b", severity="low", category="poe",
                 device="swB", interface="Gi0/9"),
        _finding("R3", finding_id="c", severity="high", category="vlan",
                 device="swA", interface="Gi0/2"),
    ]
    view = build_dashboard(_artifacts(findings=findings))["findings_view"]
    assert view["grouped_by_severity"]["high"] == 2
    assert view["grouped_by_category"]["vlan"] == 2
    assert set(view["grouped_by_device"]) == {"swA", "swB"}
    # Sorted by severity: a high finding comes first.
    assert view["findings"][0]["severity"] == "high"
    assert "risk_score" in view["findings"][0]


# -------------------------------------------------------------- remediation


def test_remediation_grouping():
    plan = {"actions": [
        _action(finding_id="a", device="swA", risk_level="high"),
        _action(finding_id="b", device="swB", risk_level="medium",
                action_type="investigation", commands=()),
        _action(finding_id="c", device="swA", risk_level="medium",
                status="blocked", commands=()),
    ]}
    view = build_dashboard(_artifacts(remediation_plan=plan))["remediation_view"]
    assert view["human_confirmation_required"] is True
    assert view["dry_run_only"] is True
    assert len(view["command_actions"]) == 1
    assert len(view["investigation_actions"]) == 1
    assert len(view["blocked_actions"]) == 1
    assert set(view["grouped_by_device"]) == {"swA", "swB"}
    assert set(view["grouped_by_risk"]) == {"high", "medium"}


def test_remediation_view_absent():
    view = build_dashboard(_artifacts())["remediation_view"]
    assert view["available"] is False
    assert view["dry_run_only"] is True


# -------------------------------------------------------------- action audit


def test_action_audit_view_from_dry_run():
    execution = {"records": [
        {"status": "validated", "executed": False},
        {"status": "blocked", "executed": False},
        {"status": "validated", "executed": False}]}
    summary = {"total_actions": 3, "validated_actions": 2, "executed": False}
    view = build_dashboard(_artifacts(dry_run_execution=execution,
                                      execution_summary=summary))[
        "action_audit_view"]
    assert view["available"] is True
    assert view["executed_count"] == 0
    assert len(view["records_by_status"]["validated"]) == 2


def test_action_audit_view_missing():
    view = build_dashboard(_artifacts())["action_audit_view"]
    assert view["available"] is False
    assert "reason" in view


# -------------------------------------------------------- device health cards


def test_device_health_card_status():
    inv = _inv([
        _device("critA", interfaces=[_iface()], stp=[
            {"interface": "Gi0/1", "vlan": "20", "role": "Altn",
             "state": "blocking"}]),
        _device("warnB", interfaces=[_iface("Gi0/2")]),
        _device("healthyC", interfaces=[_iface("Gi0/3")]),
        _device("unknownD", interfaces=[]),
    ])
    findings = [_finding(device="critA", severity="high"),
                _finding(device="warnB", severity="medium", interface="Gi0/2")]
    cards = {c["device_id"]: c
             for c in build_dashboard(_artifacts(inventory=inv, findings=findings))[
                 "device_health_cards"]["cards"]}
    assert cards["critA"]["status"] == "critical"
    assert cards["critA"]["stp_blocked_count"] == 1
    assert cards["warnB"]["status"] == "warning"
    assert cards["healthyC"]["status"] == "healthy"
    assert cards["unknownD"]["status"] == "unknown"


# -------------------------------------------------------------- risk timeline


def test_risk_timeline_generation(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"timestamp": "2026-07-07T10:00:00+00:00"}), "utf-8")
    artifacts = _artifacts(
        directory=str(tmp_path), topology=_topo(), findings=[_finding()],
        remediation_summary={"timestamp": "2026-07-07T10:05:00+00:00"},
        execution_summary={"timestamp": "2026-07-07T10:10:00+00:00"})
    timeline = build_dashboard(artifacts)["risk_timeline"]
    steps = [e["step"] for e in timeline["events"]]
    assert "snapshot_generated" in steps
    assert "remediation_planned" in steps
    assert "dry_run_executed" in steps
    assert timeline["kind"] == "artifact_lifecycle"
    # Events are time-ordered.
    times = [e["timestamp"] for e in timeline["events"]]
    assert times == sorted(times)


# -------------------------------------------------------------- export meta


def test_export_metadata_records_missing(tmp_path: Path):
    (tmp_path / "inventory.json").write_text(json.dumps(_inv()), "utf-8")
    artifacts = _artifacts(directory=str(tmp_path))
    meta = build_dashboard(artifacts)["export_metadata"]
    assert "inventory.json" in meta["source_artifacts_used"]
    assert "findings.json" in meta["source_artifacts_missing"]
    assert meta["export_version"] == EXPORT_VERSION
    assert meta["diff_id"] is None


# --------------------------------------------------------------- diff views


def test_diff_and_verification_views():
    diff = _diff(
        diff_summary={"before_snapshot_id": "b", "after_snapshot_id": "a",
                      "total_changes": 3,
                      "changes_by_category": {"interface": 2, "finding": 1},
                      "changes_by_type": {"changed": 2, "added": 1},
                      "findings_new": 1, "findings_resolved": 1,
                      "findings_persistent": 0},
        snapshot_diff={"records": [{"category": "interface",
                                    "change_type": "changed"}]},
        verifications=[{"status": "passed", "rule_id": "R"},
                       {"status": "failed", "rule_id": "R2"},
                       {"status": "passed", "rule_id": "R3"}])
    views = build_dashboard(_artifacts(), diff)
    assert views["diff_view"]["total_changes"] == 3
    assert views["diff_view"]["changes_by_category"]["interface"] == 2
    assert views["verification_view"]["passed"] == 2
    assert views["verification_view"]["failed"] == 1
    assert len(views["verification_view"]["grouped_by_status"]["passed"]) == 2


def test_diff_views_absent_without_diff():
    views = build_dashboard(_artifacts())
    assert "diff_view" not in views
    assert "verification_view" not in views


# ----------------------------------------------------- no recompute / mutate


def test_build_does_not_write_or_mutate(tmp_path: Path):
    inv = _inv([_device(interfaces=[_iface(), _iface("Gi0/2")])])
    findings = [_finding(severity="high")]
    before_inv = copy.deepcopy(inv)
    before_findings = copy.deepcopy(findings)
    artifacts = _artifacts(directory=str(tmp_path), inventory=inv,
                           findings=findings)
    build_dashboard(artifacts)
    # No files written by the pure builder, and inputs are untouched.
    assert not (tmp_path / "dashboard").exists()
    assert inv == before_inv
    assert findings == before_findings


# ------------------------------------------------------------------- CLI


class _FakeCtx:
    def __init__(self, network_config_dir: Path):
        self.config = {"network_config": {}}

        class _P:
            pass

        self.paths = _P()
        self.paths.network_config_dir = network_config_dir


def _write_snapshot(root: Path, name: str, findings=None):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inventory.json").write_text(json.dumps(_inv()), "utf-8")
    if findings is not None:
        (directory / "findings.json").write_text(json.dumps(findings), "utf-8")
    return directory


def test_cli_happy_path(tmp_path: Path, monkeypatch):
    import scripts.export_network_config_dashboard as cli

    root = tmp_path / "network_config"
    _write_snapshot(root, "snap", findings=[_finding(severity="high")])
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    code = cli.main(["--snapshot-id", "snap"])
    assert code == 0
    dash = root / "snap" / "dashboard"
    for name in ("dashboard_summary", "inventory_view", "topology_view",
                 "findings_view", "remediation_view", "action_audit_view",
                 "risk_timeline", "device_health_cards", "export_metadata"):
        assert (dash / f"{name}.json").is_file()


def test_cli_missing_snapshot_error(tmp_path: Path, monkeypatch):
    import scripts.export_network_config_dashboard as cli

    root = tmp_path / "network_config"
    root.mkdir(parents=True)
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    assert cli.main(["--snapshot-id", "nope"]) == 1

"""Tests for src.network_config Phase 7 configuration intelligence (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.network_config.intelligence import (
    DiffArtifacts,
    SnapshotArtifacts,
    build_intelligence,
    generate_action_items,
    generate_hypotheses,
    load_snapshot_artifacts,
    score_finding,
)
from src.network_config.intelligence_artifacts import write_intelligence


# ------------------------------------------------------------------ builders


def _finding(rule_id, finding_id=None, device="swA", interface="Gi0/1",
             vlan=None, category="vlan", severity="medium", status="open",
             confidence="high", details=None):
    return {"finding_id": finding_id or f"{rule_id}-x", "rule_id": rule_id,
            "title": rule_id.replace("_", " ").title(), "severity": severity,
            "category": category, "device": device, "interface": interface,
            "vlan": vlan, "status": status, "evidence": "e",
            "recommendation": "r", "confidence": confidence, "source": "rule",
            "tags": [], "details": details or {}}


def _iface(name="Gi0/1"):
    return {"name": name, "status": "connected", "protocol_status": None,
            "vlan": "10", "mode": "access", "description": None, "speed": None,
            "duplex": None, "poe_enabled": None, "poe_state": None}


def _device(device_id="swA", interfaces=None, stp=(), mac=(), neighbors=()):
    return {"device": {"device_id": device_id, "hostname": device_id,
                       "platform": None, "management_ip": None,
                       "source_files": []},
            "interfaces": list(interfaces if interfaces is not None else [_iface()]),
            "vlans": [], "trunks": [],
            "poe": [], "neighbors": list(neighbors), "mac_entries": list(mac),
            "stp_states": list(stp)}


def _inv(devices=None):
    return {"snapshot_id": "s", "input_directory": ".", "files_parsed": [],
            "files_missing": [], "warnings": [],
            "devices": devices or [_device()]}


def _artifacts(findings=(), inventory=None, topology=None, rule_summary=None,
               remediation_plan=None, remediation_summary=None,
               execution_summary=None, snapshot_id="s"):
    return SnapshotArtifacts(
        snapshot_id=snapshot_id, directory="/snap", inventory=inventory or _inv(),
        topology=topology, findings=list(findings), rule_summary=rule_summary,
        remediation_plan=remediation_plan,
        remediation_summary=remediation_summary,
        execution_summary=execution_summary)


def _action(rule_id, finding_id, action_type="port_shutdown",
            commands=("shutdown",), status="planned", severity="high",
            device="swA", interface="Gi0/1"):
    return {"action_id": f"ACT-{finding_id}", "finding_id": finding_id,
            "rule_id": rule_id, "title": rule_id, "severity": severity,
            "action_type": action_type, "device": device, "interface": interface,
            "vlan": None, "commands": list(commands),
            "rollback": {"commands": [], "note": ""}, "verification_steps": [],
            "safety_checks": [], "requires_confirmation": True,
            "dry_run_only": True, "status": status, "reason": None,
            "risk_level": "medium", "source": "rule", "tags": []}


def _plan(*actions):
    return {"snapshot_id": "s", "actions": list(actions), "dry_run_only": True}


def _verification(status, rule_id="R", finding_id="F", device="swA",
                  interface="Gi0/1", vid="V1", aid="A1"):
    return {"verification_id": vid, "action_id": aid, "finding_id": finding_id,
            "rule_id": rule_id, "device": device, "interface": interface,
            "expected_outcome": "x", "observed_outcome": "y", "status": status,
            "evidence": None, "recommendation": None}


def _diff(verifications=(), snapshot_diff=None, diff_summary=None, diff_id="d"):
    return DiffArtifacts(diff_id=diff_id, snapshot_diff=snapshot_diff,
                         verification_results=list(verifications),
                         diff_summary=diff_summary or {})


# ------------------------------------------------------------------- summary


def test_intelligence_summary_generation():
    artifacts = _artifacts(
        findings=[_finding("R1", severity="high"),
                  _finding("R2", severity="low", interface="Gi0/2")],
        remediation_summary={"total_actions": 2, "command_actions": 1,
                             "investigation_actions": 1, "blocked_actions": 0})
    intel = build_intelligence(artifacts)
    s = intel.summary
    assert s.total_findings == 2
    assert s.total_interfaces == 1
    assert s.command_actions == 1
    assert s.investigation_actions == 1
    assert "no commands were executed" in s.safety_note.lower()


# ---------------------------------------------------------------- risk score


def test_risk_scoring_severity_ordering():
    crit = score_finding(_finding("R", severity="critical")).risk_score
    high = score_finding(_finding("R", severity="high")).risk_score
    med = score_finding(_finding("R", severity="medium")).risk_score
    assert crit > high > med
    assert score_finding(_finding("R", severity="critical")).risk_level \
        == "critical"


def test_risk_scoring_deterministic():
    finding = _finding("R", severity="high", category="security")
    a = score_finding(finding, topology_relevant=True)
    b = score_finding(finding, topology_relevant=True)
    assert a == b
    assert a.contributing_factors == b.contributing_factors


def test_verification_failed_raises_risk():
    finding = _finding("R", severity="medium")
    base = score_finding(finding).risk_score
    failed = score_finding(finding, verification_status="failed").risk_score
    passed = score_finding(finding, verification_status="passed").risk_score
    assert failed > base > passed


# ----------------------------------------------------------- root-cause hyp


def test_hypothesis_stp_blocking_plus_mac_loop():
    artifacts = _artifacts(findings=[
        _finding("STP_BLOCKING_ACCESS_PORT", category="stp", interface="Gi0/3"),
        _finding("MAC_ON_MULTIPLE_INTERFACES", category="stp", interface="Gi0/3",
                 finding_id="mac-1"),
    ])
    hyps = generate_hypotheses(artifacts)
    assert any("loop" in h.tags for h in hyps)
    loop = next(h for h in hyps if "loop" in h.tags)
    assert loop.confidence in ("possible", "candidate", "likely")


def test_hypothesis_trunk_no_neighbor_missing_stp():
    # Inventory has no STP state for Gi0/1 -> missing STP on the trunk.
    artifacts = _artifacts(findings=[
        _finding("TRUNK_WITHOUT_NEIGHBOR", category="topology",
                 interface="Gi0/1")])
    hyps = generate_hypotheses(artifacts)
    assert any("stale trunk" in h.title.lower()
               or "undocumented uplink" in h.title.lower() for h in hyps)


def test_hypothesis_absent_without_evidence():
    artifacts = _artifacts(findings=[_finding("POE_DISABLED_EXPECTED",
                                              category="poe")])
    assert generate_hypotheses(artifacts) == []


# --------------------------------------------------------------- action items


def test_action_item_command_bearing_remediation():
    artifacts = _artifacts(
        findings=[_finding("UNUSED_PORT_ADMIN_UP", finding_id="F1",
                           severity="high", category="port")],
        remediation_plan=_plan(_action("UNUSED_PORT_ADMIN_UP", "F1")))
    items = generate_action_items(artifacts)
    plan_items = [i for i in items if i.action_type == "config-plan"]
    assert plan_items
    assert plan_items[0].priority == "P1"
    assert "confirmation" in plan_items[0].safety_status


def test_action_item_investigation_only_remediation():
    artifacts = _artifacts(
        findings=[_finding("STP_BLOCKING_ACCESS_PORT", finding_id="F2",
                           category="stp")],
        remediation_plan=_plan(_action("STP_BLOCKING_ACCESS_PORT", "F2",
                                       action_type="investigation",
                                       commands=())))
    items = generate_action_items(artifacts)
    assert any(i.action_type == "investigate" for i in items)


def test_failed_verification_becomes_high_priority_item():
    artifacts = _artifacts(findings=[])
    diff = _diff(verifications=[_verification("failed",
                                              rule_id="TRUNK_MISSING_REQUIRED_VLAN")])
    items = generate_action_items(artifacts, diff)
    verify_items = [i for i in items if i.action_type == "verify"]
    assert verify_items and verify_items[0].priority == "P1"


def test_passed_verification_becomes_monitor_item():
    diff = _diff(verifications=[_verification("passed")])
    items = generate_action_items(_artifacts(findings=[]), diff)
    assert any(i.action_type == "monitor" and i.priority == "P3" for i in items)


# --------------------------------------------------------- graceful degrade


def test_missing_optional_artifacts_handled():
    # Only an inventory, nothing else.
    intel = build_intelligence(_artifacts())
    assert intel.summary.total_findings == 0
    assert intel.summary.total_remediation_actions == 0
    assert intel.hypotheses == ()
    assert intel.summary.diff_available is False


# --------------------------------------------------------------- artifacts


def test_summary_json_persistence(tmp_path: Path):
    intel = build_intelligence(_artifacts(findings=[_finding("R1")]))
    paths = write_intelligence(intel, tmp_path)
    assert paths["summary"].is_file()
    assert paths["report"].is_file()
    payload = json.loads(paths["summary"].read_text("utf-8"))
    assert payload["snapshot_id"] == "s"
    assert payload["total_findings"] == 1
    assert "safety_note" in payload


def test_report_has_all_sections(tmp_path: Path):
    intel = build_intelligence(_artifacts(
        findings=[_finding("R1", severity="high")],
        remediation_summary={"total_actions": 1, "command_actions": 1,
                             "investigation_actions": 0, "blocked_actions": 0},
        remediation_plan=_plan(_action("R1", "R1-x"))))
    report = write_intelligence(intel, tmp_path)["report"].read_text("utf-8")
    for header in ("## Executive Summary", "## Inventory Overview",
                   "## Topology Overview", "## Configuration Findings",
                   "## Highest-Risk Issues", "## Root-Cause Hypotheses",
                   "## Remediation Plan Summary", "## Operator Action Items",
                   "## Safety Notes", "## Appendix: Artifact Paths"):
        assert header in report


def test_report_contains_safety_notes(tmp_path: Path):
    intel = build_intelligence(_artifacts(findings=[_finding("R1")]))
    report = write_intelligence(intel, tmp_path)["report"].read_text("utf-8")
    assert "No commands were executed." in report
    assert "dry-run" in report.lower()
    assert "not" in report.lower() and "live-device verification" in report.lower()


def test_diff_aware_report_section(tmp_path: Path):
    diff = _diff(
        verifications=[_verification("failed")],
        diff_summary={"before_snapshot_id": "b", "after_snapshot_id": "a",
                      "total_changes": 3, "findings_new": 1,
                      "findings_resolved": 1, "verification_passed": 0,
                      "verification_failed": 1, "verification_unknown": 0})
    intel = build_intelligence(_artifacts(findings=[_finding("R1")]), diff)
    paths = write_intelligence(intel, tmp_path)
    assert "report_with_diff" in paths
    text = paths["report_with_diff"].read_text("utf-8")
    assert "## Snapshot Diff / Verification Summary" in text
    assert intel.summary.diff_available is True


# --------------------------------------------------------------- load / CLI


def test_load_snapshot_missing_inventory(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_snapshot_artifacts(tmp_path / "empty")


def test_load_snapshot_optional_warnings(tmp_path: Path):
    directory = tmp_path / "s"
    directory.mkdir()
    (directory / "inventory.json").write_text(json.dumps(_inv()), "utf-8")
    artifacts = load_snapshot_artifacts(directory)
    assert artifacts.findings == []
    assert any("findings" in w for w in artifacts.warnings)


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
    import scripts.generate_network_config_report as cli

    root = tmp_path / "network_config"
    _write_snapshot(root, "snap", findings=[_finding("R1", severity="high")])
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    code = cli.main(["--snapshot-id", "snap"])
    assert code == 0
    assert (root / "snap" / "config_intelligence_report.md").is_file()
    assert (root / "snap" / "config_intelligence_summary.json").is_file()


def test_cli_missing_snapshot_error(tmp_path: Path, monkeypatch):
    import scripts.generate_network_config_report as cli

    root = tmp_path / "network_config"
    root.mkdir(parents=True)
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(root))
    assert cli.main(["--snapshot-id", "nope"]) == 1

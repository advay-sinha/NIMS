"""Tests for src.network_config Phase 4 remediation planning (dry-run only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.network_config.findings import Finding, make_finding_id
from src.network_config.inventory import derive_summary
from src.network_config.models import NetworkDevice, NetworkInventory
from src.network_config.remediation import (
    RemediationGenerator,
    generate_remediation,
    load_remediation_config,
)
from src.network_config.remediation_artifacts import write_remediation
from src.network_config.reporting import network_config_report


def _finding(rule_id: str, **kw) -> Finding:
    device = kw.get("device", "A")
    interface = kw.get("interface")
    vlan = kw.get("vlan")
    return Finding(
        finding_id=make_finding_id(rule_id, device, interface, vlan),
        rule_id=rule_id, title=kw.get("title", rule_id),
        severity=kw.get("severity", "medium"),
        category=kw.get("category", "port"),
        device=device, interface=interface, vlan=vlan,
        status=kw.get("status", "open"), evidence=kw.get("evidence"),
        details=kw.get("details", {}),
    )


def _cfg(**templates) -> dict:
    return {
        "global": {"enabled": True, "dry_run_only": True,
                   "require_confirmation": True, "require_rollback": True,
                   "require_verification": True},
        "templates": templates,
    }


# --------------------------------------------------------------- config load


def test_remediation_config_loading() -> None:
    cfg = load_remediation_config("configs/remediation.yaml")
    assert cfg["global"]["dry_run_only"] is True
    assert cfg["global"]["require_rollback"] is True
    assert "TRUNK_MISSING_REQUIRED_VLAN" in cfg["templates"]


def test_load_remediation_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_remediation_config(tmp_path / "nope.yaml")


# ------------------------------------------------------------ command plans


def test_trunk_missing_vlan_plan() -> None:
    f = _finding("TRUNK_MISSING_REQUIRED_VLAN", interface="Gi0/1",
                 details={"missing_vlans": ["30"]})
    plan, _ = generate_remediation(
        [f], "s", _cfg(TRUNK_MISSING_REQUIRED_VLAN={"enabled": True,
                                                    "risk_level": "medium"})
    )
    action = plan.actions[0]
    assert action.status == "planned"
    assert "switchport trunk allowed vlan add 30" in action.commands
    assert "switchport trunk allowed vlan remove 30" in action.rollback_commands
    assert action.verification_steps[0].command == "show interfaces trunk"
    assert action.risk_level == "medium"


def test_trunk_unauthorized_vlan_plan() -> None:
    f = _finding("TRUNK_UNAUTHORIZED_VLAN", interface="Gi0/1",
                 details={"unauthorized_vlans": ["999"]})
    plan, _ = generate_remediation(
        [f], "s", _cfg(TRUNK_UNAUTHORIZED_VLAN={"enabled": True,
                                                "risk_level": "high"})
    )
    action = plan.actions[0]
    assert "switchport trunk allowed vlan remove 999" in action.commands
    assert "switchport trunk allowed vlan add 999" in action.rollback_commands
    assert action.risk_level == "high"


def test_unused_admin_up_shutdown_plan() -> None:
    f = _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/3")
    plan, _ = generate_remediation(
        [f], "s", _cfg(UNUSED_PORT_ADMIN_UP={"enabled": True})
    )
    action = plan.actions[0]
    assert action.action_type == "port_shutdown"
    assert "shutdown" in action.commands
    assert "no shutdown" in action.rollback_commands
    assert action.verification_steps[0].command == "show interface status"


def test_poe_disabled_expected_plan() -> None:
    f = _finding("POE_DISABLED_EXPECTED", interface="Gi0/5", category="poe")
    plan, _ = generate_remediation(
        [f], "s", _cfg(POE_DISABLED_EXPECTED={"enabled": True})
    )
    action = plan.actions[0]
    assert "power inline auto" in action.commands
    assert "power inline never" in action.rollback_commands


# ---------------------------------------------------- investigation / blocked


def test_stp_blocking_investigation_only() -> None:
    f = _finding("STP_BLOCKING_ACCESS_PORT", interface="Gi0/3", vlan="20",
                 category="stp")
    plan, _ = generate_remediation(
        [f], "s", _cfg(STP_BLOCKING_ACCESS_PORT={"enabled": True,
                                                 "mode": "investigation_only"})
    )
    action = plan.actions[0]
    assert action.action_type == "investigation"
    assert action.commands == ()          # never config-changing
    assert action.rollback_commands == ()
    assert action.verification_steps      # read-only inspection steps present


def test_mac_multiple_interfaces_investigation_only() -> None:
    f = _finding("MAC_ON_MULTIPLE_INTERFACES", category="stp")
    plan, _ = generate_remediation(
        [f], "s", _cfg(MAC_ON_MULTIPLE_INTERFACES={"enabled": True,
                                                   "mode": "investigation_only"})
    )
    assert plan.actions[0].action_type == "investigation"
    assert plan.actions[0].commands == ()


def test_unsupported_finding_blocked() -> None:
    f = _finding("TOTALLY_UNKNOWN_RULE", interface="Gi0/1")
    plan, _ = generate_remediation([f], "s", _cfg())
    action = plan.actions[0]
    assert action.status == "blocked"
    assert action.commands == ()
    assert "no remediation template" in action.reason


def test_disabled_template_skipped() -> None:
    f = _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/3")
    plan, summary = generate_remediation(
        [f], "s", _cfg(UNUSED_PORT_ADMIN_UP={"enabled": False})
    )
    action = plan.actions[0]
    assert action.status == "skipped"
    assert action.commands == ()
    assert "disabled" in action.reason
    assert summary["actions_by_status"].get("skipped") == 1


# --------------------------------------------------------------- safety


def test_command_actions_have_rollback_and_verification() -> None:
    findings = [
        _finding("TRUNK_MISSING_REQUIRED_VLAN", interface="Gi0/1",
                 details={"missing_vlans": ["30"]}),
        _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/3"),
        _finding("POE_DISABLED_EXPECTED", interface="Gi0/5"),
    ]
    plan, _ = generate_remediation([f for f in findings], "s", _cfg(
        TRUNK_MISSING_REQUIRED_VLAN={"enabled": True},
        UNUSED_PORT_ADMIN_UP={"enabled": True},
        POE_DISABLED_EXPECTED={"enabled": True},
    ))
    command_actions = [a for a in plan.actions if a.is_command_bearing]
    assert len(command_actions) == 3
    for action in command_actions:
        assert action.rollback_commands, "rollback required"
        assert action.verification_steps, "verification required"
        assert action.requires_confirmation is True
        assert action.dry_run_only is True
        names = {c.name: c.satisfied for c in action.safety_checks}
        assert names["rollback_present"] and names["verification_present"]
        assert names["dry_run_only"] and names["requires_confirmation"]


def test_plan_is_dry_run_and_do_not_execute() -> None:
    f = _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/3")
    plan, summary = generate_remediation(
        [f], "s", _cfg(UNUSED_PORT_ADMIN_UP={"enabled": True})
    )
    assert plan.dry_run_only is True
    assert plan.do_not_execute is True
    assert summary["dry_run_only"] is True
    assert summary["requires_confirmation"] is True
    assert summary["do_not_execute"] is True


def test_command_blocked_when_no_concrete_commands() -> None:
    # Trunk-add with no VLANs cannot produce a safe concrete change -> the
    # engine blocks it rather than emitting a no-op or unsafe action.
    f = _finding("TRUNK_MISSING_REQUIRED_VLAN", interface="Gi0/1",
                 details={"missing_vlans": []})
    plan, _ = generate_remediation(
        [f], "s", _cfg(TRUNK_MISSING_REQUIRED_VLAN={"enabled": True})
    )
    action = plan.actions[0]
    assert action.status == "blocked"
    assert action.commands == ()
    assert "no candidate commands" in action.reason


# --------------------------------------------------------------- summary


def test_remediation_summary_correctness() -> None:
    findings = [
        _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/3"),
        _finding("STP_BLOCKING_ACCESS_PORT", interface="Gi0/4", vlan="20",
                 category="stp"),
    ]
    plan, summary = generate_remediation(findings, "snap", _cfg(
        UNUSED_PORT_ADMIN_UP={"enabled": True},
        STP_BLOCKING_ACCESS_PORT={"enabled": True, "mode": "investigation_only"},
    ))
    assert summary["total_findings"] == 2
    assert summary["total_actions"] == 2
    assert summary["command_actions"] == 1
    assert summary["investigation_actions"] == 1
    assert summary["blocked_actions"] == 0
    assert summary["actions_by_status"] == {"planned": 2}


def test_only_open_findings_are_planned() -> None:
    findings = [
        _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/3", status="open"),
        _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/9",
                 status="suppressed"),
    ]
    plan, summary = generate_remediation(
        findings, "s", _cfg(UNUSED_PORT_ADMIN_UP={"enabled": True})
    )
    assert summary["total_findings"] == 1
    assert len(plan.actions) == 1


# --------------------------------------------------------------- artifacts


def test_remediation_artifact_persistence(tmp_path: Path) -> None:
    findings = [
        _finding("TRUNK_MISSING_REQUIRED_VLAN", interface="Gi0/1",
                 details={"missing_vlans": ["30"]}),
        _finding("STP_BLOCKING_ACCESS_PORT", interface="Gi0/3", vlan="20",
                 category="stp"),
    ]
    plan, summary = generate_remediation(findings, "snap", _cfg(
        TRUNK_MISSING_REQUIRED_VLAN={"enabled": True},
        STP_BLOCKING_ACCESS_PORT={"enabled": True, "mode": "investigation_only"},
    ))
    paths = write_remediation(plan, summary, tmp_path)
    for key in ("plan_json", "plan_md", "commands_csv", "summary"):
        assert paths[key].is_file()

    payload = json.loads(paths["plan_json"].read_text("utf-8"))
    assert payload["do_not_execute"] is True
    assert "No commands were executed" in payload["notice"]
    assert len(payload["actions"]) == 2

    with open(paths["commands_csv"], encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Only the command-bearing action contributes command rows.
    assert rows and all(r["rule_id"] == "TRUNK_MISSING_REQUIRED_VLAN"
                        for r in rows)
    assert any("switchport trunk allowed vlan add 30" == r["command"]
               for r in rows)

    md = paths["plan_md"].read_text("utf-8")
    assert "No commands were executed." in md


# --------------------------------------------------------------- reporting


def test_report_remediation_section() -> None:
    inv = NetworkInventory(snapshot_id="s", input_directory=".",
                           devices=(_dev(),))
    f = _finding("UNUSED_PORT_ADMIN_UP", interface="Gi0/3")
    plan, summary = generate_remediation(
        [f], "s", _cfg(UNUSED_PORT_ADMIN_UP={"enabled": True})
    )
    report_summary = {**summary, "top_actions": [
        {"rule_id": a.rule_id, "title": a.title, "risk_level": a.risk_level,
         "action_type": a.action_type, "device": a.device,
         "interface": a.interface}
        for a in plan.actions if a.status == "planned"
    ]}
    report = network_config_report(inv, derive_summary(inv), None, None,
                                   report_summary)
    assert "## Remediation (dry-run)" in report
    assert "No commands were executed." in report


def _dev():
    from src.network_config.models import ParsedDeviceSnapshot
    return ParsedDeviceSnapshot(device=NetworkDevice(device_id="A"))


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


def test_cli_runs_remediation_by_default(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_snapshot(src)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap",
                        lambda args: _FakeCtx({"network_config": {}}, out))
    assert cli.main(["--input-dir", str(src), "--snapshot-id", "s1"]) == 0
    assert (out / "s1" / "remediation_plan.json").is_file()
    assert (out / "s1" / "remediation_commands.csv").is_file()


def test_cli_skip_remediation(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_snapshot(src)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap",
                        lambda args: _FakeCtx({"network_config": {}}, out))
    code = cli.main(["--input-dir", str(src), "--snapshot-id", "s2",
                     "--skip-remediation"])
    assert code == 0
    assert (out / "s2" / "findings.json").is_file()
    assert not (out / "s2" / "remediation_plan.json").exists()


def test_cli_custom_remediation_config(tmp_path: Path, monkeypatch) -> None:
    import scripts.analyze_network_config as cli

    src = tmp_path / "src"
    src.mkdir()
    _write_snapshot(src)  # Gi0/9 notconnect -> UNUSED_PORT_ADMIN_UP finding
    rem_file = tmp_path / "custom_rem.yaml"
    # Custom config forces the unused-port rule to investigation-only, so it
    # yields no command action (default would emit a shutdown command).
    rem_file.write_text(
        "global:\n  enabled: true\n  dry_run_only: true\n"
        "  require_confirmation: true\n  require_rollback: true\n"
        "  require_verification: true\n"
        "templates:\n"
        "  UNUSED_PORT_ADMIN_UP:\n"
        "    enabled: true\n    mode: investigation_only\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "bootstrap",
                        lambda args: _FakeCtx({"network_config": {}}, out))
    code = cli.main(["--input-dir", str(src), "--snapshot-id", "s3",
                     "--remediation-config", str(rem_file)])
    assert code == 0
    summary = json.loads(
        (out / "s3" / "remediation_summary.json").read_text("utf-8")
    )
    assert summary["command_actions"] == 0
    assert summary["investigation_actions"] >= 1

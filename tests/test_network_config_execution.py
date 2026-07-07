"""Tests for src.network_config Phase 5 dry-run execution + audit (no exec)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.network_config.audit import build_audit_records, write_audit_log
from src.network_config.execution_artifacts import write_execution
from src.network_config.executor import (
    DryRunExecutor,
    load_executor_config,
    load_remediation_plan,
    summarise_execution,
)


# ------------------------------------------------------------------ helpers


def _action(status="planned", commands=(), rollback=(), verification=(),
            action_type="port_shutdown", requires_confirmation=True,
            dry_run_only=True, **kw) -> dict:
    return {
        "action_id": kw.get("action_id", "ACT-1"),
        "finding_id": kw.get("finding_id", "F1"),
        "rule_id": kw.get("rule_id", "R1"),
        "title": kw.get("title", "t"),
        "severity": kw.get("severity", "medium"),
        "action_type": action_type,
        "device": kw.get("device", "swA"),
        "interface": kw.get("interface", "Gi0/1"),
        "vlan": kw.get("vlan"),
        "commands": list(commands),
        "rollback": {"commands": list(rollback), "note": ""},
        "verification_steps": [
            {"command": c, "expected_result": e} for c, e in verification
        ],
        "safety_checks": [],
        "requires_confirmation": requires_confirmation,
        "dry_run_only": dry_run_only,
        "status": status,
        "reason": kw.get("reason"),
        "risk_level": kw.get("risk_level", "medium"),
        "source": "rule_engine",
        "tags": [],
    }


def _plan(*actions, snapshot_id="s") -> dict:
    return {
        "snapshot_id": snapshot_id,
        "generated_at": "t0",
        "dry_run_only": True,
        "do_not_execute": True,
        "actions": list(actions),
    }


def _ecfg(**safety) -> dict:
    base = {
        "block_config_commands_in_investigation_actions": True,
        "block_if_missing_rollback": True,
        "block_if_missing_verification": True,
        "block_if_not_dry_run": True,
    }
    base.update(safety)
    return {
        "global": {"enabled": True, "mode": "dry_run",
                   "default_operator": "offline_dry_run"},
        "safety": base,
    }


def _command_action(**kw) -> dict:
    return _action(
        commands=["interface Gi0/1", "shutdown"],
        rollback=["interface Gi0/1", "no shutdown"],
        verification=[("show interface status", "administratively down")],
        **kw,
    )


# ----------------------------------------------------------------- validation


def test_valid_command_action_validated() -> None:
    result = DryRunExecutor(_ecfg()).execute(_plan(_command_action()))
    record = result.records[0]
    assert record.status == "validated"
    assert record.executed is False
    assert record.would_execute is False
    assert record.execution_mode == "dry_run"
    assert record.execution_id.startswith("EXE-")


def test_missing_rollback_blocks() -> None:
    action = _action(commands=["shutdown"], rollback=[],
                     verification=[("show x", "y")])
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "blocked"
    assert "rollback" in record.reason


def test_missing_verification_blocks() -> None:
    action = _action(commands=["shutdown"], rollback=["no shutdown"],
                     verification=[])
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "blocked"
    assert "verification" in record.reason


def test_not_dry_run_blocks() -> None:
    action = _command_action(dry_run_only=False)
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "blocked"
    assert "dry_run_only" in record.reason


def test_requires_confirmation_false_blocks_command_action() -> None:
    action = _command_action(requires_confirmation=False)
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "blocked"
    assert "confirmation" in record.reason


def test_investigation_with_config_command_blocked() -> None:
    action = _action(action_type="investigation", commands=["shutdown"],
                     rollback=[], verification=[("show x", "y")])
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "blocked"
    assert "investigation" in record.reason


def test_investigation_without_commands_validated() -> None:
    action = _action(action_type="investigation", commands=[], rollback=[],
                     verification=[("show mac address-table", "locate MAC")],
                     requires_confirmation=False)
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "validated"
    assert record.requested_commands == ()


def test_blocked_plan_action_remains_blocked() -> None:
    action = _action(status="blocked", commands=[], rollback=[],
                     reason="no template", requires_confirmation=False)
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "blocked"
    assert record.reason == "no template"


def test_skipped_plan_action_remains_skipped() -> None:
    action = _action(status="skipped", commands=[], rollback=[],
                     reason="template disabled", requires_confirmation=False)
    record = DryRunExecutor(_ecfg()).execute(_plan(action)).records[0]
    assert record.status == "skipped"
    assert record.reason == "template disabled"


# ------------------------------------------------------------------- summary


def test_execution_summary_correctness() -> None:
    plan = _plan(
        _command_action(action_id="ACT-1"),                       # validated
        _action(action_id="ACT-2", commands=["shutdown"], rollback=[],
                verification=[("x", "y")]),                        # blocked
        _action(action_id="ACT-3", status="skipped", commands=[],
                requires_confirmation=False),                     # skipped
    )
    result = DryRunExecutor(_ecfg()).execute(plan)
    summary = summarise_execution(result)
    assert summary.total_actions == 3
    assert summary.validated_actions == 1
    assert summary.blocked_actions == 1
    assert summary.skipped_actions == 1
    assert summary.failed_actions == 0
    assert summary.executed is False
    assert summary.would_execute is False


# ----------------------------------------------------------------- audit log


def test_audit_log_written(tmp_path: Path) -> None:
    result = DryRunExecutor(_ecfg(), operator="tester").execute(
        _plan(_command_action())
    )
    records = build_audit_records(result)
    path = write_audit_log(records, tmp_path / "action_audit_log.jsonl")
    lines = path.read_text("utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["executed"] is False
    assert entry["dry_run_only"] is True
    assert entry["operator"] == "tester"
    assert entry["status"] == "validated"


# ----------------------------------------------------------------- artifacts


def test_execution_artifacts_persistence(tmp_path: Path) -> None:
    result = DryRunExecutor(_ecfg()).execute(_plan(_command_action()))
    summary = summarise_execution(result)
    paths = write_execution(result, summary, tmp_path)
    for key in ("execution_json", "execution_csv", "audit_log", "summary"):
        assert paths[key].is_file()

    payload = json.loads(paths["execution_json"].read_text("utf-8"))
    assert payload["executed"] is False
    assert payload["would_execute"] is False
    assert payload["execution_mode"] == "dry_run"
    assert "No commands were executed" in payload["notice"]

    summary_json = json.loads(paths["summary"].read_text("utf-8"))
    assert summary_json["audit_log_path"]
    assert "No commands were executed" in summary_json["notice"]


def test_execution_csv_persistence(tmp_path: Path) -> None:
    result = DryRunExecutor(_ecfg()).execute(_plan(_command_action()))
    write_execution(result, summarise_execution(result), tmp_path)
    with open(tmp_path / "dry_run_execution.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(r["executed"] == "false" for r in rows)
    assert all(r["would_execute"] == "false" for r in rows)
    assert all(r["execution_mode"] == "dry_run" for r in rows)


def test_report_execution_section_appended(tmp_path: Path) -> None:
    report = tmp_path / "network_config_report.md"
    report.write_text("# Network Configuration Report — s\n", encoding="utf-8")
    result = DryRunExecutor(_ecfg()).execute(_plan(_command_action()))
    write_execution(result, summarise_execution(result), tmp_path)
    text = report.read_text("utf-8")
    assert "## Dry-run execution (Phase 5)" in text
    assert "No commands were executed." in text
    # Re-running must replace, not duplicate, the section.
    write_execution(result, summarise_execution(result), tmp_path)
    assert report.read_text("utf-8").count("## Dry-run execution (Phase 5)") == 1


# --------------------------------------------------------------- config load


def test_executor_config_loading() -> None:
    cfg = load_executor_config("configs/network_action_executor.yaml")
    assert cfg["global"]["mode"] == "dry_run"
    assert cfg["global"]["allow_live_execution"] is False
    assert cfg["safety"]["block_if_missing_rollback"] is True


def test_load_executor_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_executor_config(tmp_path / "nope.yaml")


def test_load_remediation_plan_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_remediation_plan(tmp_path / "remediation_plan.json")


def test_load_remediation_plan_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "remediation_plan.json"
    bad.write_text('{"snapshot_id": "s"}', encoding="utf-8")  # no "actions"
    with pytest.raises(ValueError):
        load_remediation_plan(bad)


# ------------------------------------------------------------------- no exec


def test_no_live_execution_imports() -> None:
    """Guard: Phase 5 modules never import a device-control or SSH library."""
    forbidden = ("netmiko", "napalm", "paramiko", "import socket", "pysnmp")
    for module in ("executor", "audit", "execution_artifacts"):
        source = Path(f"src/network_config/{module}.py").read_text("utf-8").lower()
        for token in forbidden:
            assert token not in source, f"{module}.py references {token!r}"


# ----------------------------------------------------------------- CLI flags


class _FakeCtx:
    def __init__(self, config: dict, network_config_dir: Path) -> None:
        self.config = config

        class _P:
            pass

        self.paths = _P()
        self.paths.network_config_dir = network_config_dir


def _write_plan_file(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = _plan(_command_action(), snapshot_id="cli")
    (directory / "remediation_plan.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_cli_happy_path(tmp_path: Path, monkeypatch) -> None:
    import scripts.dry_run_network_actions as cli

    snap_dir = tmp_path / "out" / "cli"
    _write_plan_file(snap_dir)
    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx({"network_config": {}}, tmp_path / "out"),
    )
    code = cli.main(["--snapshot-id", "cli"])
    assert code == 0
    assert (snap_dir / "dry_run_execution.json").is_file()
    assert (snap_dir / "dry_run_execution.csv").is_file()
    assert (snap_dir / "action_audit_log.jsonl").is_file()
    assert (snap_dir / "execution_summary.json").is_file()


def test_cli_missing_plan_errors(tmp_path: Path, monkeypatch) -> None:
    import scripts.dry_run_network_actions as cli

    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx({"network_config": {}}, tmp_path / "out"),
    )
    code = cli.main(["--snapshot-id", "nope"])
    assert code == 1


def test_cli_custom_operator_recorded(tmp_path: Path, monkeypatch) -> None:
    import scripts.dry_run_network_actions as cli

    snap_dir = tmp_path / "out" / "cli"
    _write_plan_file(snap_dir)
    monkeypatch.setattr(
        cli, "bootstrap",
        lambda args: _FakeCtx({"network_config": {}}, tmp_path / "out"),
    )
    code = cli.main(["--snapshot-id", "cli", "--operator", "advay"])
    assert code == 0
    lines = (snap_dir / "action_audit_log.jsonl").read_text("utf-8").splitlines()
    assert all(json.loads(line)["operator"] == "advay" for line in lines)

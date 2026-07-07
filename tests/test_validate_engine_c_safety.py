"""Tests for the Engine C static safety validator."""

from __future__ import annotations

from pathlib import Path

from scripts.validate_engine_c_safety import (
    check_config_defaults,
    run_audit,
    scan_source_tree,
)


def test_repo_passes_safety_audit():
    result = run_audit()
    assert result["passed"], result["violations"]
    assert result["violations"] == []
    assert all(ok for _, ok in result["config_checks"])


def test_mocked_forbidden_import_fails(tmp_path: Path):
    bad = tmp_path / "live_client.py"
    bad.write_text("import paramiko\n\n\ndef go():\n    return 1\n", "utf-8")
    violations = scan_source_tree([tmp_path])
    assert any(v.token == "paramiko" for v in violations)


def test_mocked_connect_handler_fails(tmp_path: Path):
    bad = tmp_path / "netmiko_client.py"
    bad.write_text(
        "from netmiko import ConnectHandler\n\n"
        "def push(dev):\n    c = ConnectHandler(**dev)\n"
        "    c.send_config_set(['interface Gi0/1', 'shutdown'])\n", "utf-8")
    tokens = {v.token for v in scan_source_tree([tmp_path])}
    assert {"netmiko", "ConnectHandler", "send_config_set"} <= tokens


def test_prose_mentions_do_not_trip(tmp_path: Path):
    # A docstring/comment mentioning the libraries must not be flagged.
    ok = tmp_path / "safe.py"
    ok.write_text(
        '"""This module never uses paramiko, netmiko or napalm."""\n'
        "# no ConnectHandler here either\n"
        "VALUE = 1\n", "utf-8")
    assert scan_source_tree([tmp_path]) == []


def test_config_defaults_are_safe():
    checks = dict(check_config_defaults(Path("configs")))
    assert checks["batfish.global.enabled is false (disabled by default)"]
    assert checks["remediation.global.dry_run_only is true"]
    assert checks["action_executor.global.allow_live_execution is false"]


def test_config_defaults_detect_unsafe(tmp_path: Path):
    (tmp_path / "batfish.yaml").write_text(
        "global:\n  enabled: true\n", "utf-8")
    checks = dict(check_config_defaults(tmp_path))
    assert checks["batfish.global.enabled is false (disabled by default)"] is False

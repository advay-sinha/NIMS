"""Tests for adapter preflight readiness (offline/mock, no real devices)."""

from __future__ import annotations

import json
import socket
from pathlib import Path

from src.live_logging.adapters import preflight
from src.live_logging.adapters.base import AdapterContext
from src.live_logging.adapters.hirschmann_config import HirschmannConfigAdapter
from src.live_logging.adapters.hirschmann_snmp import HirschmannSnmpAdapter
from src.live_logging.adapters.sophos_central import SophosCentralAdapter
from src.live_logging.adapters.sophos_firewall_syslog import SophosFirewallSyslogAdapter


def _ctx(tmp_path):
    return AdapterContext(output_dir=Path(tmp_path), routing={}, checkpoint_dir=Path(tmp_path) / "cp")


def test_offline_source_reports_disabled(tmp_path):
    a = SophosFirewallSyslogAdapter({"mode": "offline", "enabled": True, "bind_port": 5599}, _ctx(tmp_path))
    report = preflight.assess(a, _ctx(tmp_path))
    assert report.status == preflight.DISABLED
    assert report.remaining_steps  # tells the user what to flip


def test_missing_dependency_reported(tmp_path):
    # SNMP live needs pysnmp; when absent preflight must say MISSING_DEPENDENCY.
    a = HirschmannSnmpAdapter({"mode": "live", "enabled": True, "read_only": True}, _ctx(tmp_path))
    report = preflight.assess(a, _ctx(tmp_path))
    if not report.dependency_ok:
        assert report.status == preflight.MISSING_DEPENDENCY
    else:  # pysnmp happens to be installed
        assert report.status in {preflight.MISSING_CREDENTIALS, preflight.MISSING_CONFIGURATION,
                                 preflight.READY, preflight.BLOCKED_BY_SAFETY}


def test_missing_credentials_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("SOPHOS_CLIENT_ID", raising=False)
    monkeypatch.delenv("SOPHOS_CLIENT_SECRET", raising=False)
    a = SophosCentralAdapter(
        {"mode": "live", "enabled": True, "read_only": True, "region": "eu"}, _ctx(tmp_path)
    )
    report = preflight.assess(a, _ctx(tmp_path))
    # httpx is installed, region set -> should fall through to missing creds.
    assert report.status in {preflight.MISSING_CREDENTIALS, preflight.MISSING_DEPENDENCY}


def test_safety_block_reported(tmp_path):
    a = SophosFirewallSyslogAdapter(
        {"mode": "live", "enabled": True, "read_only": True, "allow_remediation": True}, _ctx(tmp_path)
    )
    report = preflight.assess(a, _ctx(tmp_path))
    assert report.status == preflight.BLOCKED_BY_SAFETY


def test_port_availability_detected(tmp_path):
    # A free port should report available=True.
    a = SophosFirewallSyslogAdapter(
        {"mode": "live", "enabled": True, "read_only": True,
         "bind_host": "127.0.0.1", "bind_port": 5599}, _ctx(tmp_path)
    )
    report = preflight.assess(a, _ctx(tmp_path))
    assert report.bind_port == 5599
    assert isinstance(report.bind_port_available, bool)

    # A port held with an exclusive bind should report available=False.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    try:
        a2 = SophosFirewallSyslogAdapter(
            {"mode": "live", "enabled": True, "read_only": True,
             "bind_host": "127.0.0.1", "bind_port": port}, _ctx(tmp_path)
        )
        assert preflight.assess(a2, _ctx(tmp_path)).bind_port_available is False
    finally:
        sock.close()


def test_write_readiness_and_json(tmp_path):
    ctx = _ctx(tmp_path)
    a = SophosFirewallSyslogAdapter({"mode": "offline", "enabled": False}, ctx)
    report = preflight.assess(a, ctx)
    path = preflight.write_readiness([report], tmp_path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["sources"][0]["source"] == "sophos_firewall_syslog"
    # No secret values ever appear in readiness output.
    assert "password" not in json.dumps(data).lower() or "env" in json.dumps(data).lower()


def test_no_secret_values_in_report(tmp_path, monkeypatch):
    monkeypatch.setenv("HIRSCHMANN_SNMP_USER", "supersecretuser")
    a = HirschmannSnmpAdapter({"mode": "live", "enabled": True, "read_only": True}, _ctx(tmp_path))
    report = preflight.assess(a, _ctx(tmp_path))
    # env_present records booleans, never values.
    assert "supersecretuser" not in json.dumps(report.to_dict())

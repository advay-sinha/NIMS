"""Tests for the live adapter layer (offline/mock, no real devices)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.live_logging.adapters import MODE_LIVE, MODE_MOCK, MODE_OFFLINE
from src.live_logging.adapters.base import AdapterContext
from src.live_logging.adapters.errors import ConfigurationError
from src.live_logging.adapters.hirschmann_config import HirschmannConfigAdapter
from src.live_logging.adapters.hirschmann_snmp import HirschmannSnmpAdapter
from src.live_logging.adapters.hirschmann_traps import HirschmannTrapAdapter
from src.live_logging.adapters.registry import build_adapter, resolve_source, SPEC_SOURCES
from src.live_logging.adapters.sophos_central import SophosCentralAdapter
from src.live_logging.adapters.sophos_firewall_syslog import SophosFirewallSyslogAdapter

SYSLOG_LINE = (
    '<134>2026-07-14T08:44:01Z SFW01 log_type="Firewall" log_subtype="Denied" '
    'severity="high" src_ip="203.0.113.55" dst_ip="10.10.10.9" message="deny"'
)


def _ctx(tmp_path, mock=None):
    return AdapterContext(output_dir=Path(tmp_path), routing={}, mock=mock)


# ------------------------------------------------------------ interface/registry


def test_registry_resolves_all_sources(tmp_path):
    for name in SPEC_SOURCES:
        adapter = build_adapter(name, {"sophos": {}}, {"hirschmann": {}}, _ctx(tmp_path))
        assert adapter.name == name
        assert adapter.engine_target in {"cyber", "network_health", "network_config"}


def test_alias_resolution():
    assert resolve_source("firewall_syslog") == "sophos_firewall_syslog"
    assert resolve_source("snmp") == "hirschmann_snmp"
    with pytest.raises(KeyError):
        resolve_source("does_not_exist")


def test_status_and_health_shape(tmp_path):
    a = SophosFirewallSyslogAdapter({"mode": MODE_OFFLINE}, _ctx(tmp_path))
    assert a.health()["source"] == "sophos_syslog"
    assert "problems" in a.status()


# ------------------------------------------------------------ mode validation


def test_invalid_mode_flagged(tmp_path):
    a = SophosFirewallSyslogAdapter({"mode": "banana"}, _ctx(tmp_path))
    assert any("mode" in p for p in a.validate_configuration())


def test_live_disabled_by_default(tmp_path):
    a = SophosCentralAdapter({"mode": MODE_LIVE, "enabled": False}, _ctx(tmp_path))
    result = a.run_once()
    assert result.error is not None  # collect() refused (enabled=false)
    assert result.persisted is False


# ------------------------------------------------------------ Sophos syslog UDP


def test_syslog_mock_parses_datagram(tmp_path):
    ctx = _ctx(tmp_path, mock={"datagrams": [{"src": "203.0.113.55", "line": SYSLOG_LINE}]})
    a = SophosFirewallSyslogAdapter(
        {"mode": MODE_MOCK, "enabled": True, "allowed_sources": ["203.0.113.55"]}, ctx
    )
    result = a.run_once()
    assert result.events == 1
    assert result.mode == MODE_MOCK


def test_syslog_source_allowlist_drops_unlisted(tmp_path):
    ctx = _ctx(tmp_path, mock={"datagrams": [{"src": "10.0.0.9", "line": SYSLOG_LINE}]})
    a = SophosFirewallSyslogAdapter(
        {"mode": MODE_MOCK, "enabled": True, "allowed_sources": ["203.0.113.55"]}, ctx
    )
    records, errors = a.collect()
    assert records == [] and any("not in allowed_sources" in e for e in errors)


def test_syslog_datagram_size_limit(tmp_path):
    ctx = _ctx(tmp_path, mock={"datagrams": [{"src": "203.0.113.55", "line": SYSLOG_LINE}]})
    a = SophosFirewallSyslogAdapter(
        {"mode": MODE_MOCK, "enabled": True, "max_datagram_bytes": 10}, ctx
    )
    records, errors = a.collect()
    assert records == [] and any("max_datagram_bytes" in e for e in errors)


# ------------------------------------------------------------ Sophos Central mock


def test_central_mock_returns_events(tmp_path):
    items = [{"id": "a2", "type": "Event::IPS", "severity": "critical",
              "description": "IPS", "src_ip": "1.2.3.4"}]
    a = SophosCentralAdapter({"mode": MODE_MOCK, "enabled": True}, _ctx(tmp_path, mock={"items": items}))
    result = a.run_once()
    assert result.events == 1


def test_central_requires_region_for_live(tmp_path):
    a = SophosCentralAdapter({"mode": MODE_LIVE, "enabled": True}, _ctx(tmp_path))
    assert any("region" in p for p in a.validate_configuration())


# ------------------------------------------------------------ Hirschmann SNMP


def test_snmp_mock_poll(tmp_path):
    readings = [{"device_id": "SW-1", "interface": "1/1", "in_errors": 1500, "utilization": 95}]
    a = HirschmannSnmpAdapter({"mode": MODE_MOCK, "enabled": True}, _ctx(tmp_path, mock={"readings": readings}))
    result = a.run_once()
    assert result.events >= 2  # high_error_rate + high_utilization


def test_snmp_rejects_arbitrary_oids(tmp_path):
    a = HirschmannSnmpAdapter(
        {"mode": MODE_OFFLINE, "oids": ["1.3.6.1.4.1.9999"]}, _ctx(tmp_path)
    )
    assert any("arbitrary OID" in p for p in a.validate_configuration())


def test_snmp_has_no_set_path():
    src = Path("src/live_logging/adapters/hirschmann_snmp.py").read_text(encoding="utf-8")
    # No SNMP write primitive in any casing (pysnmp v4 setCmd / v6+ set_cmd / CLI snmpset).
    assert "setCmd" not in src and "set_cmd" not in src and "snmpset" not in src
    assert "get_cmd" in src  # read-only GET is used (pysnmp v6/7 API)


# ------------------------------------------------------------ Hirschmann traps


def test_trap_mock_parsing_and_allowlist(tmp_path):
    good = {"src": "192.168.10.20", "line": '2026-07-14T08:47:00Z 192.168.10.20 linkDown ifName="1/3"'}
    a = HirschmannTrapAdapter(
        {"mode": MODE_MOCK, "enabled": True, "allowed_sources": ["192.168.10.20"]},
        _ctx(tmp_path, mock={"datagrams": [good]}),
    )
    assert a.run_once().events == 1


# ------------------------------------------------------------ Hirschmann config


def test_config_mock_retrieval_diff(tmp_path):
    prev = "hostname SW\n!\ninterface 1/5\n switchport access vlan 10\n poe enabled\n!\n"
    curr = "hostname SW\n!\ninterface 1/5\n switchport access vlan 20\n poe disabled\n!\n"
    a = HirschmannConfigAdapter(
        {"mode": MODE_MOCK, "enabled": True},
        _ctx(tmp_path, mock={"device_id": "SW", "previous_config": prev, "current_config": curr}),
    )
    result = a.run_once()
    assert result.events >= 2  # access_vlan_change + poe_change


def test_config_command_allowlist_rejects_write(tmp_path):
    a = HirschmannConfigAdapter(
        {"mode": MODE_OFFLINE, "allowed_commands": ["configure terminal"]}, _ctx(tmp_path)
    )
    assert any("forbidden token" in p for p in a.validate_configuration())


def test_config_live_requires_known_hosts(tmp_path):
    a = HirschmannConfigAdapter(
        {"mode": MODE_LIVE, "enabled": True, "allowed_commands": ["show running-config"]}, _ctx(tmp_path)
    )
    assert any("known_hosts" in p for p in a.validate_configuration())


# ------------------------------------------------------------ persistence/isolation


def test_checkpoint_persisted_after_run(tmp_path):
    readings = [{"device_id": "SW-1", "interface": "1/1", "in_errors": 1500}]
    a = HirschmannSnmpAdapter({"mode": MODE_MOCK, "enabled": True}, _ctx(tmp_path, mock={"readings": readings}))
    a.run_once()
    assert a.checkpoint()["cursor"].get("event_count", 0) >= 1


def test_dry_run_does_not_persist(tmp_path):
    ctx = AdapterContext(output_dir=Path(tmp_path), routing={}, dry_run=True,
                         mock={"readings": [{"device_id": "SW", "interface": "1/1", "in_errors": 1500}]})
    a = HirschmannSnmpAdapter({"mode": MODE_MOCK, "enabled": True}, ctx)
    result = a.run_once()
    assert result.events >= 1 and result.persisted is False
    assert not (Path(tmp_path) / "events.jsonl").exists()


def test_source_failure_isolated_in_runtime(tmp_path):
    from src.live_logging import runtime

    sophos = {"sophos": {"firewall_syslog": {"enabled": True, "mode": "offline",
                                             "offline_sample_path": "does/not/exist.log"}}}
    # Bad path yields zero events but must not raise.
    status, _ = runtime.run(["sophos_firewall_syslog"], {"output_dir": str(tmp_path)},
                            sophos, {"hirschmann": {}}, mode="offline")
    assert status.sources[0].status in {"ok", "failed"}

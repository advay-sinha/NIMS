"""Tests for Hirschmann SNMP metric + trap offline ingestion (Phase 9)."""

from __future__ import annotations

from pathlib import Path

from src.live_logging import hirschmann_snmp, hirschmann_traps

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "datasets" / "samples" / "live_logging" / "hirschmann"


def test_port_down_event():
    events = hirschmann_snmp.metric_events(
        {"device_id": "SW-1", "interface": "1/3", "if_oper_status": "down"}
    )
    subs = {e["subcategory"] for e in events}
    assert "port_down" in subs


def test_high_error_and_utilization_events():
    events = hirschmann_snmp.metric_events(
        {"device_id": "SW-1", "interface": "1/1", "in_errors": 1500, "utilization": 95}
    )
    subs = {e["subcategory"] for e in events}
    assert {"high_error_rate", "high_utilization"} <= subs


def test_temperature_and_power_events():
    temp = hirschmann_snmp.metric_events({"device_id": "SW-2", "temperature_c": 82})
    assert temp[0]["subcategory"] == "temperature"
    assert temp[0]["severity"] == "high"  # >= critical threshold
    power = hirschmann_snmp.metric_events(
        {"device_id": "SW-2", "power_supply_1": "ok", "power_supply_2": "failed"}
    )
    assert power[0]["subcategory"] == "power_supply"


def test_unreachable_event():
    events = hirschmann_snmp.metric_events({"device_id": "SW-3", "reachable": False})
    assert events[0]["subcategory"] == "device_unreachable"


def test_parse_snmp_metrics_from_sample():
    records = hirschmann_snmp.read_offline(SAMPLES / "snmp_metrics.json")
    assert len(records) == 6  # 1 down + (errors+util) + (temp+power) + unreachable


def test_trap_parsing_and_errors():
    lines = (SAMPLES / "traps.log").read_text(encoding="utf-8").splitlines()
    records, errors = hirschmann_traps.parse_traps(lines)
    subs = {r["subcategory"] for r in records}
    assert "port_down" in subs and "auth_failure" in subs
    assert len(errors) == 1  # the garbage line
    assert all(r["source_key"] == "hirschmann_traps" for r in records)

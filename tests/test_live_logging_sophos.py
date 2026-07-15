"""Tests for Sophos offline ingestion (Phase 9, offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.live_logging import sophos_api, sophos_syslog

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "datasets" / "samples" / "live_logging" / "sophos"


def test_parse_sophos_items_from_sample():
    import json

    items = json.loads((SAMPLES / "central_api_alerts.json").read_text(encoding="utf-8"))
    records = sophos_api.parse_sophos_items(items)
    assert len(records) == len(items)
    ips = next(r for r in records if r["category"] == "ips")
    assert ips["source_key"] == "sophos_api"
    assert ips["correlation_keys"]["src_ip"] == "203.0.113.10"
    assert ips["severity"] == "critical"


def test_sophos_client_offline_reads_sample():
    client = sophos_api.SophosCentralClient(
        {"enabled": True, "mode": "offline",
         "offline_sample_path": str(SAMPLES / "central_api_alerts.json")}
    )
    assert len(client.fetch()) == 4


def test_sophos_client_live_disabled_raises():
    client = sophos_api.SophosCentralClient({"enabled": False, "mode": "live"})
    with pytest.raises(RuntimeError):
        client.fetch()


def test_sophos_client_live_without_fetcher_raises():
    client = sophos_api.SophosCentralClient({"enabled": True, "mode": "live"})
    with pytest.raises(RuntimeError):
        client.fetch()


def test_syslog_parsing_and_errors():
    lines = (SAMPLES / "firewall_syslog.log").read_text(encoding="utf-8").splitlines()
    records, errors = sophos_syslog.parse_syslog_lines(lines)
    assert len(records) == 4          # four valid lines
    assert len(errors) == 1           # one malformed line captured
    denied = records[0]
    assert denied["category"] == "firewall"
    assert denied["correlation_keys"]["dst_ip"] == "10.10.10.9"
    assert records[1]["category"] == "ips"

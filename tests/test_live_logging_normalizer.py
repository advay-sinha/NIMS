"""Tests for src.live_logging.normalizer (Phase 9, offline)."""

from __future__ import annotations

import pytest

from src.live_logging.models import ENGINE_CYBER, ENGINE_NETWORK_HEALTH
from src.live_logging.normalizer import build_batch, build_normalized


def _record(**over):
    base = dict(
        source_vendor="sophos", source_product="sophos_firewall", source_type="syslog",
        source_key="sophos_syslog", category="ips", severity="Critical",
        message="IPS drop", timestamp="2026-07-14T08:00:00Z", device_ip="10.0.0.1",
        correlation_keys={"src_ip": "1.2.3.4"}, raw_ref="r1",
    )
    base.update(over)
    return base


def test_build_normalized_routes_and_normalizes():
    event, raw = build_normalized(_record())
    assert event.engine_target == ENGINE_CYBER
    assert event.severity == "critical"
    assert event.raw_event_ref == "r1"
    assert raw.raw_ref == "r1"
    assert event.correlation_keys["src_ip"] == "1.2.3.4"


def test_event_id_is_deterministic_for_same_record():
    e1, _ = build_normalized(_record())
    e2, _ = build_normalized(_record())
    assert e1.event_id == e2.event_id


def test_missing_required_key_raises():
    bad = _record()
    del bad["category"]
    with pytest.raises(KeyError):
        build_normalized(bad)


def test_routing_override_applies():
    event, _ = build_normalized(
        _record(source_key="hirschmann_snmp", category="interface"),
        routing={"hirschmann_snmp": "network_health"},
    )
    assert event.engine_target == ENGINE_NETWORK_HEALTH


def test_build_batch_parallel_lengths():
    events, raws = build_batch([_record(raw_ref="a"), _record(raw_ref="b")])
    assert len(events) == len(raws) == 2

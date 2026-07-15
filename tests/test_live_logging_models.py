"""Tests for src.live_logging.models (Phase 9, offline)."""

from __future__ import annotations

import pytest

from src.live_logging.models import (
    ENGINE_CYBER,
    NormalizedEvent,
    SourceStatus,
    IngestionStatus,
    make_event_id,
    normalize_severity,
)


def _event(**over):
    base = dict(
        event_id="evt_x", timestamp="2026-07-14T00:00:00Z", source_vendor="sophos",
        source_product="sophos_firewall", source_type="syslog", category="ips",
        severity="high", message="m", engine_target=ENGINE_CYBER,
    )
    base.update(over)
    return NormalizedEvent(**base)


def test_required_fields_present_in_to_dict():
    data = _event().to_dict()
    for field in NormalizedEvent.REQUIRED_FIELDS:
        assert field in data and data[field] not in (None, "")
    assert "REQUIRED_FIELDS" not in data


def test_invalid_engine_target_raises():
    with pytest.raises(ValueError):
        _event(engine_target="bogus")


@pytest.mark.parametrize(
    "raw,expected",
    [("Critical", "critical"), ("WARN", "medium"), ("informational", "info"),
     ("err", "high"), (None, "unknown"), ("", "unknown"), ("7", "unknown")],
)
def test_normalize_severity(raw, expected):
    assert normalize_severity(raw) == expected


def test_make_event_id_deterministic():
    assert make_event_id("a", "b", 1) == make_event_id("a", "b", 1)
    assert make_event_id("a", "b", 1) != make_event_id("a", "b", 2)


def test_severity_rank_orders_most_severe_first():
    assert _event(severity="critical").severity_rank < _event(severity="low").severity_rank


def test_ingestion_status_healthy_flag():
    ok = SourceStatus(source="s1", engine_target="cyber", status="ok", events=3)
    bad = SourceStatus(source="s2", engine_target="cyber", status="failed")
    healthy = IngestionStatus("a", "b", "offline", True, 3, sources=[ok])
    unhealthy = IngestionStatus("a", "b", "offline", True, 3, sources=[ok, bad])
    assert healthy.healthy is True
    assert unhealthy.healthy is False
    assert isinstance(healthy.to_dict()["sources"][0], dict)

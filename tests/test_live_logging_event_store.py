"""Tests for src.live_logging.event_store (Phase 9, offline)."""

from __future__ import annotations

from src.live_logging.event_store import EventStore
from src.live_logging.models import ENGINE_CYBER, NormalizedEvent, RawEvent


def _event(i):
    return NormalizedEvent(
        event_id=f"evt_{i}", timestamp="2026-07-14T00:00:00Z", source_vendor="sophos",
        source_product="sophos_firewall", source_type="syslog", category="ips",
        severity="high", message=f"m{i}", engine_target=ENGINE_CYBER,
    )


def _raw(i):
    return RawEvent(raw_ref=f"r{i}", source_vendor="sophos", source_product="p",
                    source_type="syslog", source_name="s", received_at="t", payload={"i": i})


def test_append_and_read_roundtrip(tmp_path):
    store = EventStore(tmp_path)
    assert store.append_normalized([_event(1), _event(2)]) == 2
    rows = store.read_normalized()
    assert [r["event_id"] for r in rows] == ["evt_1", "evt_2"]


def test_append_only_accumulates(tmp_path):
    store = EventStore(tmp_path)
    store.append_normalized([_event(1)])
    store.append_normalized([_event(2)])
    assert len(store.read_normalized()) == 2


def test_raw_and_normalized_are_separate_files(tmp_path):
    store = EventStore(tmp_path)
    store.append_normalized([_event(1)])
    store.append_raw([_raw(1)])
    assert store.normalized_path.exists()
    assert store.raw_path.exists()
    assert len(store.read_raw()) == 1


def test_missing_files_read_as_empty(tmp_path):
    store = EventStore(tmp_path / "nope")
    assert store.read_normalized() == []
    assert store.read_raw() == []

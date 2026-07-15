"""Tests for src.live_logging.routing and checkpointing (Phase 9, offline)."""

from __future__ import annotations

from src.live_logging.checkpoint import CheckpointManager
from src.live_logging.models import (
    ENGINE_CYBER,
    ENGINE_NETWORK_CONFIG,
    ENGINE_NETWORK_HEALTH,
    ENGINE_UNKNOWN,
)
from src.live_logging.routing import route


def test_default_routing():
    assert route("sophos_syslog") == ENGINE_CYBER
    assert route("hirschmann_snmp") == ENGINE_NETWORK_HEALTH
    assert route("hirschmann_config") == ENGINE_NETWORK_CONFIG


def test_unknown_source_routes_unknown():
    assert route("mystery_source") == ENGINE_UNKNOWN


def test_config_override_and_invalid_target():
    assert route("sophos_syslog", {"sophos_syslog": "network_health"}) == ENGINE_NETWORK_HEALTH
    assert route("sophos_syslog", {"sophos_syslog": "not_a_target"}) == ENGINE_UNKNOWN


def test_checkpoint_roundtrip(tmp_path):
    mgr = CheckpointManager(tmp_path)
    assert mgr.load("sophos_api").cursor == {}
    saved = mgr.save("sophos_api", {"last_event_id": "a4"})
    assert saved.cursor["last_event_id"] == "a4"
    assert mgr.load("sophos_api").cursor["last_event_id"] == "a4"
    assert saved.updated_at  # timestamp recorded

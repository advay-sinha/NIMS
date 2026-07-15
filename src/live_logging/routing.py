"""Deterministic routing of normalized events to engine targets (Phase 9).

Purpose
-------
Every normalized event is assigned to exactly one engine target — ``cyber``,
``network_health`` or ``network_config`` (or ``unknown``) — from a
configuration-driven map keyed by source key (spec Phase 9 > Routing). Routing
is pure and deterministic: same source key + config always yields the same
target.
"""

from __future__ import annotations

from typing import Mapping

from src.live_logging.models import (
    ENGINE_CYBER,
    ENGINE_NETWORK_CONFIG,
    ENGINE_NETWORK_HEALTH,
    ENGINE_TARGETS,
    ENGINE_UNKNOWN,
)

# Fallback routing map used when config omits one (matches the spec example).
DEFAULT_ROUTING: dict[str, str] = {
    "sophos": ENGINE_CYBER,
    "sophos_api": ENGINE_CYBER,
    "sophos_syslog": ENGINE_CYBER,
    "hirschmann_snmp": ENGINE_NETWORK_HEALTH,
    "hirschmann_traps": ENGINE_NETWORK_HEALTH,
    "hirschmann_config": ENGINE_NETWORK_CONFIG,
}


def route(source_key: str, routing: Mapping[str, str] | None = None) -> str:
    """Return the engine target for a source key using ``routing`` (or default).

    An unknown source key or an invalid configured target both resolve to
    ``unknown`` rather than raising, so one bad mapping never crashes ingestion.
    """
    table = {**DEFAULT_ROUTING, **(dict(routing) if routing else {})}
    target = table.get(source_key, ENGINE_UNKNOWN)
    return target if target in ENGINE_TARGETS else ENGINE_UNKNOWN

"""Real-time / streaming monitoring foundation (Phase 11) — offline & safe.

This package simulates near-real-time monitoring by **replaying existing,
already-persisted artefacts** (Engine A/B/C findings and correlation incidents)
into a unified event stream, maintaining an in-memory monitoring state, writing
an append-only event log, and exposing read-only "current state" artefacts for
the dashboard.

Hard safety boundary
--------------------
This is a demo/replay layer only. It **never** contacts a device, opens SSH,
polls SNMP, captures packets, runs an engine pipeline, trains, infers or
executes a remediation command. Every input is a local file; every output is a
local file. See ``configs/streaming.yaml`` (``safety`` block) — all live flags
are ``false`` and there is no code path that honours ``true``.
"""

from __future__ import annotations

__all__ = [
    "models",
    "event_log",
    "sources",
    "state",
    "replay",
    "artifacts",
    "formatting",
    "runtime",
]

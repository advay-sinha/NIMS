"""Monitoring dashboard prototype (Phase 10) — offline, artefact-driven.

A thin, read-only visualisation layer over the artefacts already produced by
Engine A (cyber), Engine B (network-health), Engine C (configuration) and the
Phase 9 correlation engine. It **never** recomputes an artefact, runs an engine
pipeline, trains, infers, polls SNMP, captures packets, contacts a device or
executes a command — it only reads and renders what is on disk.

Import discipline
-----------------
``loader`` and ``formatting`` are pure Python and carry **no** Streamlit
dependency, so they (and the test-suite) import cleanly in environments without
Streamlit. ``components``, ``views`` and ``app`` are the only modules that touch
Streamlit and are imported solely by the running app. The package ``__init__``
therefore imports neither, keeping ``import src.dashboard`` Streamlit-free.
"""

from __future__ import annotations

__all__ = [
    "loader",
    "formatting",
    "components",
    "views",
    "app",
]

"""Correlation Engine (Phase 9) — unified cross-engine incident generation.

Purpose
-------
Combine the already-persisted artefacts of the three NIMS engines into a single
operator-facing view:

- Engine A — cyber intrusion-detection outputs (aggregate model/error-analysis
  artefacts; there is no per-flow alert log yet, so cyber signals are marked
  ``aggregate``).
- Engine B — network-health anomaly experiment metrics.
- Engine C — network-configuration findings, topology warnings and device
  health cards.

This package is **offline and artefact-driven only**. It never runs an engine
pipeline, never captures packets, never polls SNMP, never contacts a device,
never executes a command and never mutates an Engine A/B/C artefact. All
correlation is deterministic and rule-based (no ML); incident and signal ids are
content-addressed so re-running on the same inputs yields identical output.
"""

from __future__ import annotations

__all__ = [
    "models",
    "loader",
    "rules",
    "engine",
    "artifacts",
    "reporting",
]

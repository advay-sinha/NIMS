"""Live-ready, offline-first logging & telemetry ingestion (Phase 9).

A shared ingestion layer that collects, normalizes, redacts, persists and
routes cyber, health and configuration events from Sophos and Hirschmann
infrastructure into the common :class:`~src.live_logging.models.NormalizedEvent`
schema.

Safety contract
---------------
This package is READ-ONLY. It never changes Sophos/Hirschmann configuration,
shuts ports, disables PoE, changes VLANs or executes remediation. Live device
clients are disabled by default and gated behind explicit configuration; every
live client is mockable and the offline path (saved samples) is the default.
Secrets are never persisted — see :mod:`src.live_logging.redaction`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

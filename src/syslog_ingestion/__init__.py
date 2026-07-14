"""Offline industrial-switch syslog ingestion (Phase 12).

Parses *saved* Belden/DCN-style switch logs (raw syslog exports and PuTTY
terminal captures) into structured :class:`~src.syslog_ingestion.models.SyslogEvent`
objects, then derives Engine B time-window features and Engine C findings.

This package is strictly offline and read-only: it never contacts a device,
opens SSH/telnet, polls SNMP, captures packets, runs an engine pipeline or
executes any remediation. Every input is a local file; every output is a local
artefact under ``outputs/syslog_ingestion/<run_id>/``.
"""

from __future__ import annotations

from src.syslog_ingestion.models import (
    EngineHints,
    SyslogEntities,
    SyslogEvent,
)

__all__ = ["SyslogEvent", "SyslogEntities", "EngineHints"]

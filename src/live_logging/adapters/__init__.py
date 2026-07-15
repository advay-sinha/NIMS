"""Live-ready ingestion adapters (Phase — Live-Ready Adapter Implementation).

Each adapter wraps one source (Sophos Firewall syslog, Sophos Central API,
Hirschmann SNMP, Hirschmann traps, Hirschmann read-only config) behind a common
interface with three interchangeable modes — ``offline`` (saved samples),
``mock`` (injected transport, used by tests) and ``live`` (real transport,
DISABLED by default and lazily imported).

All modes reuse the existing parser / normalizer / redaction / event-store /
routing layers; no parsing or normalization is duplicated here. Every adapter is
READ-ONLY: there is no SNMP SET, no configuration mode, no write command and no
remediation path anywhere in this package.
"""

from __future__ import annotations

__all__ = ["MODE_OFFLINE", "MODE_MOCK", "MODE_LIVE", "MODES"]

MODE_OFFLINE = "offline"
MODE_MOCK = "mock"
MODE_LIVE = "live"
MODES = (MODE_OFFLINE, MODE_MOCK, MODE_LIVE)

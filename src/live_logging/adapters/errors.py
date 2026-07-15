"""Adapter error types (Phase — Live-Ready Adapter Implementation)."""

from __future__ import annotations


class AdapterError(Exception):
    """Base class for all adapter errors."""


class ConfigurationError(AdapterError):
    """The adapter configuration is invalid or incomplete."""


class DependencyMissing(AdapterError):
    """A live-mode dependency (httpx / pysnmp / paramiko) is not installed."""


class SafetyViolation(AdapterError):
    """A configuration would breach the read-only safety boundary."""


class ConnectionTestError(AdapterError):
    """A non-destructive live connection test failed."""

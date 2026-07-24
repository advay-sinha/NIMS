"""Vendor-specific Engine C parsers (offline, read-only).

Each vendor module turns saved device command output into the shared typed
models in :mod:`src.network_config.models`, so the inventory/rules/topology/
remediation/dashboard pipeline stays vendor-agnostic. No live device access.
"""

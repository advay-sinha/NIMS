"""Engine C — Network Configuration Intelligence & Remediation.

Phase 1 (this package): offline, READ-ONLY parsing of saved network device
command outputs into a typed, structured inventory. No live device access, no
SNMP polling, no remediation execution — those are later, human-gated phases
and are intentionally absent here (see CLAUDE.md > Engine C safety rules).
"""

from __future__ import annotations

__all__ = ["models", "parsers", "inventory", "artifacts", "reporting"]

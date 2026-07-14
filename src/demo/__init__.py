"""Offline end-to-end demo orchestration (frontend readiness).

Prepares or refreshes every artefact the read-only frontend needs — Engine C
assessment, Engine A/B model readiness, syslog ingestion, unified correlation,
streaming current-state — in one command, then validates that the dashboard
loaders can serve every section.

Strictly offline and read-only with respect to devices: the orchestrator only
executes an **allowlisted** set of local Python module entry points (argument
arrays, never ``shell=True``). It never opens SSH/SNMP/syslog listeners, never
captures packets, never contacts a device, never executes remediation and never
mutates raw datasets or source artefacts. It does not add a second training or
correlation pipeline — it reuses the existing scripts.
"""

from __future__ import annotations

__all__ = ["models", "planner", "readiness", "runner", "artifacts", "reporting"]

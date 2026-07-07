"""Engine C Phase 9 — dashboard export artefact persistence.

Writes the frontend-friendly dashboard views under
``outputs/network_config/<snapshot_id>/dashboard/``::

    dashboard_summary.json
    inventory_view.json
    topology_view.json
    findings_view.json
    remediation_view.json
    action_audit_view.json
    risk_timeline.json
    device_health_cards.json
    export_metadata.json
    diff_view.json            (only when a diff is supplied)
    verification_view.json    (only when a diff is supplied)

Pure serialisation of already-built views — nothing here recomputes state,
mutates an artefact, contacts a device or executes a command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_dashboard(
    views: dict[str, dict[str, Any]], out_dir: Path
) -> dict[str, Path]:
    """Write each dashboard view as ``<out_dir>/dashboard/<key>.json``."""
    from src.utils.io import write_json

    out = Path(out_dir) / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, view in views.items():
        paths[key] = write_json(view, out / f"{key}.json")
    logger.info("Dashboard export (%d view(s)) written to %s "
                "(offline; no commands executed).", len(views), out)
    return paths

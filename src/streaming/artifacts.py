"""Read-only 'current state' artefact persistence for the dashboard.

Writes the live monitoring snapshot and a small run summary under
``outputs/streaming/`` so the dashboard can poll a stable, JSON view of the
stream. Pure serialisation — nothing here recomputes or executes anything.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.streaming.state import MonitoringState

logger = logging.getLogger(__name__)


def write_current_state(state: MonitoringState, current_dir: str | Path
                        ) -> dict[str, Path]:
    """Write ``current_state.json`` (+ split incident/event views)."""
    from src.utils.io import write_json

    out = Path(current_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = state.snapshot()
    paths = {
        "current_state": write_json(snapshot, out / "current_state.json"),
        "active_incidents": write_json(
            snapshot["active_incidents"], out / "active_incidents.json"),
        "recent_events": write_json(
            snapshot["recent_events"], out / "recent_events.json"),
    }
    return paths


def write_summary(state: MonitoringState, output_dir: str | Path) -> Path:
    """Write ``stream_summary.json`` for the run."""
    from src.utils.io import write_json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return write_json(state.summary(), out / "stream_summary.json")


def load_current_state(current_dir: str | Path) -> dict[str, Any]:
    """Read back ``current_state.json`` (tolerant: ``available`` flag)."""
    import json

    path = Path(current_dir) / "current_state.json"
    if not path.is_file():
        return {"available": False,
                "message": "No live monitoring state yet. Run: "
                           "python -m scripts.run_streaming_demo"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["available"] = True
        return data
    except (OSError, ValueError) as exc:
        logger.warning("could not read current state %s: %s", path, exc)
        return {"available": False, "message": f"Unreadable current state: {exc}"}

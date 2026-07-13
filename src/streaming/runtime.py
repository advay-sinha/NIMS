"""Streaming demo orchestrator.

Ties the pieces together: collect events from offline artefacts, replay them
into an in-memory :class:`~src.streaming.state.MonitoringState` while appending
to the event log, then write the read-only current-state artefacts. Pacing is
injectable so tests and one-shot runs never block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from src.streaming import sources
from src.streaming.artifacts import write_current_state, write_summary
from src.streaming.event_log import EventLog
from src.streaming.replay import replay
from src.streaming.state import MonitoringState

logger = logging.getLogger(__name__)


def _cfg(config: dict[str, Any], dotted: str, default: Any) -> Any:
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


@dataclass
class StreamRunResult:
    """Outcome of one demo replay run."""

    events_emitted: int
    state: MonitoringState
    event_log_path: Path
    current_state_paths: dict[str, Path]
    summary_path: Path


def run_stream(
    config: dict[str, Any],
    dirs: dict[str, Any],
    *,
    sleep_fn: Optional[Callable[[float], None]] = None,
    tick_seconds: Optional[float] = None,
    max_events: Optional[int] = None,
    loop: Optional[bool] = None,
    reset_log: bool = True,
) -> StreamRunResult:
    """Run one offline demo replay and persist the current-state artefacts.

    ``sleep_fn`` defaults to a no-op (no pacing); the CLI passes ``time.sleep``
    for a near-real-time feel. Every input and output is a local file — nothing
    here contacts a device, captures packets or executes a command.
    """
    tick = float(_cfg(config, "streaming.tick_seconds", 1.0)
                 if tick_seconds is None else tick_seconds)
    cap = _cfg(config, "streaming.max_events", None) if max_events is None \
        else max_events
    do_loop = bool(_cfg(config, "streaming.loop", False)) if loop is None else loop

    events = sources.collect_events(config, dirs)

    log = EventLog(dirs["event_log_path"])
    if reset_log:
        log.reset()
    state = MonitoringState()

    def on_event(event) -> None:
        state.apply(event)
        log.append(event)

    emitted = replay(events, on_event, tick_seconds=tick, max_events=cap,
                     loop=do_loop, sleep_fn=sleep_fn)

    current_paths = {}
    summary_path = Path(dirs["output_dir"]) / "stream_summary.json"
    if _cfg(config, "dashboard.write_current_state", True):
        current_paths = write_current_state(state, dirs["current_state_dir"])
    if _cfg(config, "dashboard.write_summary", True):
        summary_path = write_summary(state, dirs["output_dir"])

    logger.info("Streaming demo replayed %d event(s) (offline; no device "
                "access, no command execution).", emitted)
    return StreamRunResult(
        events_emitted=emitted, state=state,
        event_log_path=Path(dirs["event_log_path"]),
        current_state_paths=current_paths, summary_path=summary_path)

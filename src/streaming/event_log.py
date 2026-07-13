"""Append-only local event log (JSON Lines).

A tiny, dependency-free writer/reader for the streaming event log. Append-only
by construction — it never rewrites or truncates existing lines, mirroring the
audit-log discipline used elsewhere in NIMS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from src.streaming.models import StreamEvent, event_from_dict

logger = logging.getLogger(__name__)


class EventLog:
    """Append-only JSONL event log at a fixed path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def reset(self) -> None:
        """Start a fresh demo log (removes a previous demo file if present).

        Used only at the start of a demo replay run so repeated demos do not
        accumulate; real deployments would keep appending.
        """
        if self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: StreamEvent) -> None:
        """Append one event as a JSON line (creates parent dirs on first write)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), default=str) + "\n")

    def append_many(self, events: Iterable[StreamEvent]) -> int:
        """Append several events; return the count written."""
        count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event.to_dict(), default=str) + "\n")
                count += 1
        return count

    def read_all(self) -> list[StreamEvent]:
        """Read every logged event back into :class:`StreamEvent` objects."""
        if not self.path.is_file():
            return []
        events: list[StreamEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(event_from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skipping malformed event-log line: %s", exc)
        return events

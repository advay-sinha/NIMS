"""Append-only JSONL persistence for raw and normalized events (Phase 9).

Purpose
-------
Raw and normalized event streams are persisted as append-only JSONL so history
is never overwritten (spec Phase 9 > Output Rules). Raw and normalized events
are kept in separate files. All writes go through :func:`redaction.redact`'d
objects upstream; this module does not itself hold secrets.

The store is filesystem-only: no device access, no network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from src.live_logging.models import NormalizedEvent, RawEvent

logger = logging.getLogger(__name__)

NORMALIZED_FILENAME = "events.jsonl"
RAW_FILENAME = "raw_events.jsonl"


class EventStore:
    """Append-only JSONL event store rooted at ``base_dir``."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.normalized_path = self.base_dir / NORMALIZED_FILENAME
        self.raw_path = self.base_dir / RAW_FILENAME

    def _ensure_dir(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _append(path: Path, rows: Iterable[dict[str, Any]]) -> int:
        count = 0
        with open(path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
                count += 1
        return count

    def append_normalized(self, events: Iterable[NormalizedEvent]) -> int:
        """Append normalized events; returns the number written."""
        self._ensure_dir()
        written = self._append(self.normalized_path, (e.to_dict() for e in events))
        if written:
            logger.info("Appended %d normalized events to %s", written, self.normalized_path)
        return written

    def append_raw(self, events: Iterable[RawEvent]) -> int:
        """Append raw (already-redacted) events; returns the number written."""
        self._ensure_dir()
        written = self._append(self.raw_path, (e.to_dict() for e in events))
        if written:
            logger.info("Appended %d raw events to %s", written, self.raw_path)
        return written

    def append_to(self, filename: str, rows: Iterable[dict[str, Any]]) -> int:
        """Append arbitrary dict rows to a named JSONL file under the store.

        Used for per-source streams (e.g. ``sophos/firewall_syslog.jsonl``).
        """
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return self._append(path, rows)

    def read_normalized(self) -> list[dict[str, Any]]:
        """Read every persisted normalized event (empty list if none)."""
        return self._read(self.normalized_path)

    def read_raw(self) -> list[dict[str, Any]]:
        """Read every persisted raw event (empty list if none)."""
        return self._read(self.raw_path)

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in %s", path)
        return rows

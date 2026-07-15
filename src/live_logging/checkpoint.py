"""Per-source checkpoint persistence (Phase 9).

Purpose
-------
Each source keeps its own JSON checkpoint (cursor, last event id, last poll
time, last config hash…) so polling can resume without reprocessing or losing
events (spec Phase 9 > Checkpointing Strategy). Writes are atomic (temp file +
replace) so a crash mid-write never corrupts the last good checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.live_logging.models import Checkpoint, utc_now_iso

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Reads/writes per-source JSON checkpoints under ``base_dir``."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _path(self, source: str) -> Path:
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in source)
        return self.base_dir / f"{safe}_checkpoint.json"

    def load(self, source: str) -> Checkpoint:
        """Load a source's checkpoint, or an empty one when none exists."""
        path = self._path(source)
        if not path.is_file():
            return Checkpoint(source=source, cursor={}, updated_at="")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Unreadable checkpoint %s (%s); starting fresh", path, exc)
            return Checkpoint(source=source, cursor={}, updated_at="")
        return Checkpoint(
            source=data.get("source", source),
            cursor=dict(data.get("cursor", {})),
            updated_at=str(data.get("updated_at", "")),
        )

    def save(self, source: str, cursor: dict[str, Any]) -> Checkpoint:
        """Atomically persist a source's cursor; returns the saved checkpoint.

        Raises
        ------
        OSError
            If the checkpoint cannot be written (caller maps this to the
            ``checkpoint_write_error`` failure category).
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = Checkpoint(source=source, cursor=dict(cursor), updated_at=utc_now_iso())
        path = self._path(source)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(checkpoint.to_dict(), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
        logger.info("Saved checkpoint for %s -> %s", source, path)
        return checkpoint

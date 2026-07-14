"""Time-window bucketing for syslog events.

Small, dependency-free helpers that floor event timestamps into fixed-width
bins (5/15/60 min) used by the Engine B feature builder. Boot-clock
(``clock_unreliable``) events carry no trustworthy time and are excluded by the
feature builder, not here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp string into a datetime, or ``None``."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def floor_to_bin(moment: datetime, window_minutes: int) -> datetime:
    """Floor ``moment`` down to the start of its ``window_minutes`` bin."""
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    total_minutes = moment.hour * 60 + moment.minute
    floored = (total_minutes // window_minutes) * window_minutes
    base = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return base + timedelta(minutes=floored)


def window_bounds(moment: datetime, window_minutes: int) -> tuple[str, str]:
    """Return ``(start_iso, end_iso)`` for the bin containing ``moment``."""
    start = floor_to_bin(moment, window_minutes)
    end = start + timedelta(minutes=window_minutes)
    return start.isoformat(), end.isoformat()

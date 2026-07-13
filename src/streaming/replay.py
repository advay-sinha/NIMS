"""Deterministic event replay.

Orders collected events and replays them one "tick" at a time, stamping each
with a monotonic sequence and an emit timestamp and invoking a callback. Pacing
is injectable (``sleep_fn``) so tests and one-shot runs never actually block;
the CLI passes ``time.sleep`` for a near-real-time demo feel.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from src.streaming.models import StreamEvent

logger = logging.getLogger(__name__)


def order_events(events: list[StreamEvent]) -> list[StreamEvent]:
    """Stable order by source timestamp (missing last), then original position."""
    indexed = list(enumerate(events))
    indexed.sort(key=lambda pair: (pair[1].timestamp or "~", pair[0]))
    return [event for _, event in indexed]


def replay(
    events: list[StreamEvent],
    on_event: Callable[[StreamEvent], None],
    *,
    tick_seconds: float = 0.0,
    max_events: Optional[int] = None,
    loop: bool = False,
    sleep_fn: Callable[[float], None] | None = None,
    base_time: Optional[datetime] = None,
    max_total: int = 10_000,
) -> int:
    """Replay ``events`` in order, stamping emit time/seq and calling ``on_event``.

    Returns the number of events emitted. ``loop`` repeats the ordered stream
    (bounded by ``max_events`` or the ``max_total`` safety cap so a demo can
    never run unbounded). ``sleep_fn`` defaults to a no-op — pacing is opt-in.
    """
    ordered = order_events(events)
    if not ordered:
        return 0
    sleep = sleep_fn or (lambda _seconds: None)
    base = base_time or datetime.now(timezone.utc)

    emitted = 0
    seq = 0
    while True:
        for event in ordered:
            if max_events is not None and emitted >= max_events:
                return emitted
            if emitted >= max_total:
                logger.warning("replay hit safety cap of %d events", max_total)
                return emitted
            emit_time = (base + timedelta(seconds=seq * max(tick_seconds, 0.0))
                         ).isoformat()
            stamped = event.with_emission(seq=seq, emitted_at=emit_time)
            on_event(stamped)
            emitted += 1
            seq += 1
            if tick_seconds > 0:
                sleep(tick_seconds)
        if not loop:
            return emitted

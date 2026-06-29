"""Lightweight timing utilities.

Purpose
-------
Measure wall-clock duration of pipeline stages for the metrics the project
requires (e.g. preprocessing time). Usable as a context manager or decorator
and emits via logging — never ``print``.

Examples
--------
>>> from src.utils.timer import Timer
>>> with Timer("load nsl-kdd") as t:     # doctest: +SKIP
...     ...                               # doctest: +SKIP
>>> t.elapsed                             # doctest: +SKIP

Limitations
-----------
Measures wall-clock (``perf_counter``), not CPU time.
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Optional, Type

logger = logging.getLogger(__name__)


class Timer:
    """Context manager that records elapsed wall-clock seconds.

    Parameters
    ----------
    label:
        Human-readable name logged on exit.
    log_level:
        Logging level for the completion message.
    """

    def __init__(self, label: str = "block", log_level: int = logging.INFO) -> None:
        self.label = label
        self.log_level = log_level
        self._start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # TODO(data-engineer): compute self.elapsed and log at self.log_level.
        #   Do not suppress exceptions (return None / False).
        raise NotImplementedError

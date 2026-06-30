"""Memory profiling and cleanup utilities.

Purpose
-------
Instrument the preprocessing pipeline so peak RAM can be measured per stage on
constrained (8 GB) development machines, and provide opt-in garbage collection
between stages. Measurement uses the resident set size (RSS) of the current
process via :mod:`psutil`; a lightweight background sampler captures the true
peak during a stage (pandas/numpy allocate outside the Python heap, so
``tracemalloc`` alone would undercount).

All instrumentation is opt-in and cheap when disabled (no sampler thread).

Examples
--------
>>> report = MemoryReport(enabled=True)            # doctest: +SKIP
>>> with report.stage("cleaning"):                 # doctest: +SKIP
...     cleaned = clean(frame)                      # doctest: +SKIP
>>> report.to_dict()["peak_mb"]                    # doctest: +SKIP
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Default RSS sampling interval (seconds) for the peak tracker.
_SAMPLE_INTERVAL: float = 0.05


def rss_bytes() -> int:
    """Return the current process resident set size in bytes.

    Returns ``0`` if :mod:`psutil` is unavailable, so callers degrade
    gracefully rather than failing.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - environment without psutil
        return 0


def _mb(num_bytes: int) -> float:
    """Convert a byte count to megabytes, rounded to 3 decimals."""
    return round(num_bytes / 1024**2, 3)


def collect_garbage(enabled: bool = True) -> None:
    """Invoke a full garbage collection when ``enabled``.

    Used between major pipeline stages to release intermediate DataFrames.
    """
    if enabled:
        gc.collect()


class StageProfiler:
    """Context manager recording memory + timing for one pipeline stage.

    On exit, appends a result dict to the owning report's ``stages`` list with
    keys: ``stage``, ``memory_before_mb``, ``memory_after_mb``, ``peak_mb``,
    ``delta_mb`` and ``elapsed_seconds``. When disabled, only timing is
    recorded and no sampler thread is spawned.

    Parameters
    ----------
    name:
        Stage label.
    stages:
        Shared list the result is appended to.
    enabled:
        When ``False``, skip RSS sampling (timing only).
    interval:
        RSS sampling interval in seconds.
    """

    def __init__(
        self,
        name: str,
        stages: list[dict[str, Any]],
        enabled: bool = True,
        interval: float = _SAMPLE_INTERVAL,
    ) -> None:
        self.name = name
        self._stages = stages
        self.enabled = enabled
        self.interval = interval
        self._before = 0
        self._peak = 0
        self._t0 = 0.0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self.result: dict[str, Any] = {}

    def __enter__(self) -> "StageProfiler":
        self._t0 = time.perf_counter()
        if self.enabled:
            self._before = rss_bytes()
            self._peak = self._before
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self) -> None:
        """Background loop tracking the maximum RSS until stopped."""
        assert self._stop is not None
        while not self._stop.is_set():
            current = rss_bytes()
            if current > self._peak:
                self._peak = current
            self._stop.wait(self.interval)

    def __exit__(self, *exc: Any) -> bool:
        elapsed = time.perf_counter() - self._t0
        if self.enabled:
            assert self._stop is not None and self._thread is not None
            self._stop.set()
            self._thread.join(timeout=1.0)
            after = rss_bytes()
            self._peak = max(self._peak, after)
            self.result = {
                "stage": self.name,
                "memory_before_mb": _mb(self._before),
                "memory_after_mb": _mb(after),
                "peak_mb": _mb(self._peak),
                "delta_mb": _mb(after - self._before),
                "elapsed_seconds": round(elapsed, 4),
            }
            logger.debug(
                "[mem] %s: before=%.1fMB after=%.1fMB peak=%.1fMB (%.3fs)",
                self.name,
                self.result["memory_before_mb"],
                self.result["memory_after_mb"],
                self.result["peak_mb"],
                elapsed,
            )
        else:
            self.result = {"stage": self.name, "elapsed_seconds": round(elapsed, 4)}
        self._stages.append(self.result)
        return False  # never suppress exceptions


class MemoryReport:
    """Aggregates per-stage profiles for one pipeline run.

    Parameters
    ----------
    enabled:
        Master switch; when ``False`` stages only record timing.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stages: list[dict[str, Any]] = []

    def stage(self, name: str) -> StageProfiler:
        """Return a :class:`StageProfiler` context manager for ``name``."""
        return StageProfiler(name, self.stages, enabled=self.enabled)

    def to_dict(self) -> dict[str, Any]:
        """Return the aggregated, JSON-serialisable memory profile."""
        peak = max((s.get("peak_mb", 0.0) for s in self.stages), default=0.0)
        total_elapsed = round(
            sum(s.get("elapsed_seconds", 0.0) for s in self.stages), 4
        )
        return {
            "enabled": self.enabled,
            "stages": self.stages,
            "peak_mb": peak,
            "total_elapsed_seconds": total_elapsed,
        }

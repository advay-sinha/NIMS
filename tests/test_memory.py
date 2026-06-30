"""Tests for src.utils.memory."""

from __future__ import annotations

from src.utils import memory


def test_rss_bytes_non_negative() -> None:
    assert memory.rss_bytes() >= 0


def test_collect_garbage_runs_without_error() -> None:
    memory.collect_garbage(True)
    memory.collect_garbage(False)  # disabled path is a no-op


def test_stage_profiler_records_metrics() -> None:
    report = memory.MemoryReport(enabled=True)
    with report.stage("unit"):
        _ = [0] * 10000
    assert len(report.stages) == 1
    entry = report.stages[0]
    assert entry["stage"] == "unit"
    for key in ("memory_before_mb", "memory_after_mb", "peak_mb", "elapsed_seconds"):
        assert key in entry
    assert entry["peak_mb"] >= entry["memory_before_mb"]


def test_disabled_report_records_timing_only() -> None:
    report = memory.MemoryReport(enabled=False)
    with report.stage("unit"):
        pass
    entry = report.stages[0]
    assert entry["stage"] == "unit"
    assert "elapsed_seconds" in entry
    assert "peak_mb" not in entry  # no sampling when disabled


def test_to_dict_aggregates_stages() -> None:
    report = memory.MemoryReport(enabled=True)
    with report.stage("a"):
        pass
    with report.stage("b"):
        pass
    summary = report.to_dict()
    assert summary["enabled"] is True
    assert len(summary["stages"]) == 2
    assert "peak_mb" in summary
    assert "total_elapsed_seconds" in summary


def test_profiler_does_not_suppress_exceptions() -> None:
    report = memory.MemoryReport(enabled=True)
    import pytest

    with pytest.raises(ValueError):
        with report.stage("boom"):
            raise ValueError("x")
    # The stage is still recorded despite the exception.
    assert report.stages[-1]["stage"] == "boom"

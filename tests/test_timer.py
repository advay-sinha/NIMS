"""Tests for src.utils.timer."""

from __future__ import annotations

import pytest

from src.utils.timer import Timer


def test_timer_enter_returns_self() -> None:
    timer = Timer("unit")
    assert timer.__enter__() is timer


def test_timer_records_elapsed() -> None:
    with Timer("unit") as timer:
        pass
    assert timer.elapsed >= 0.0


def test_timer_does_not_suppress_exceptions() -> None:
    with pytest.raises(ValueError):
        with Timer("unit"):
            raise ValueError("boom")

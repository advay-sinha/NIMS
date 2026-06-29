"""Tests for src.utils.timer."""

from __future__ import annotations

import pytest

from src.utils.timer import Timer


def test_timer_enter_returns_self() -> None:
    timer = Timer("unit")
    assert timer.__enter__() is timer


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_timer_exit_not_yet_implemented() -> None:
    with Timer("unit"):
        pass

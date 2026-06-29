"""Tests for src.utils.seed."""

from __future__ import annotations

import pytest

from src.utils import seed


def test_negative_seed_raises_value_error() -> None:
    with pytest.raises(ValueError):
        seed.set_global_seed(-1)


def test_set_global_seed_is_reproducible() -> None:
    import numpy as np

    seed.set_global_seed(42)
    first = np.random.rand(5).tolist()
    seed.set_global_seed(42)
    second = np.random.rand(5).tolist()
    assert first == second

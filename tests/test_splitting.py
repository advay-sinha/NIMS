"""Tests for src.data.splitting."""

from __future__ import annotations

import pytest

from src.data import splitting


def test_valid_ratios_pass() -> None:
    # Should not raise.
    splitting._validate_ratios(0.7, 0.15, 0.15)


@pytest.mark.parametrize(
    "train,val,test",
    [
        (0.8, 0.15, 0.15),  # sums to 1.10
        (0.6, 0.2, 0.1),    # sums to 0.90
    ],
)
def test_ratios_not_summing_to_one_raise(train: float, val: float, test: float) -> None:
    with pytest.raises(ValueError):
        splitting._validate_ratios(train, val, test)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_out_of_range_ratio_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        splitting._validate_ratios(bad, 0.5, 0.5)


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_split_not_yet_implemented() -> None:
    cfg = {"train_size": 0.7, "val_size": 0.15, "test_size": 0.15}
    splitting.train_val_test_split(x=None, y=None, split_config=cfg, seed=42)

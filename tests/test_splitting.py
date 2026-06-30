"""Tests for src.data.splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import splitting


@pytest.fixture()
def classification_frame() -> tuple[pd.DataFrame, pd.Series]:
    """A small balanced binary classification dataset (200 rows)."""
    n = 200
    x = pd.DataFrame({"f1": range(n), "f2": [i % 7 for i in range(n)]})
    y = pd.Series(["attack"] * (n // 2) + ["normal"] * (n // 2), name="label")
    return x, y


_CFG = {
    "train_size": 0.70,
    "val_size": 0.15,
    "test_size": 0.15,
    "stratify": True,
    "shuffle": True,
}


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


def test_split_partitions_are_disjoint_and_complete(classification_frame) -> None:
    x, y = classification_frame
    x_tr, x_va, x_te, y_tr, y_va, y_te = splitting.train_val_test_split(
        x, y, _CFG, seed=42
    )
    assert len(x_tr) + len(x_va) + len(x_te) == len(x)
    idx = set(x_tr.index) | set(x_va.index) | set(x_te.index)
    assert idx == set(x.index)  # disjoint + complete
    # Approximate ratios.
    assert abs(len(x_tr) / len(x) - 0.70) < 0.02
    assert abs(len(x_te) / len(x) - 0.15) < 0.02


def test_split_is_deterministic_with_seed(classification_frame) -> None:
    x, y = classification_frame
    a = splitting.train_val_test_split(x, y, _CFG, seed=7)
    b = splitting.train_val_test_split(x, y, _CFG, seed=7)
    assert list(a[0].index) == list(b[0].index)
    assert list(a[2].index) == list(b[2].index)


def test_split_different_seeds_differ(classification_frame) -> None:
    x, y = classification_frame
    a = splitting.train_val_test_split(x, y, _CFG, seed=1)
    b = splitting.train_val_test_split(x, y, _CFG, seed=2)
    assert list(a[0].index) != list(b[0].index)


def test_stratified_split_preserves_class_distribution(classification_frame) -> None:
    x, y = classification_frame
    _, _, _, y_tr, y_va, y_te = splitting.train_val_test_split(x, y, _CFG, seed=42)
    # 50/50 source -> each partition should stay close to balanced.
    for part in (y_tr, y_va, y_te):
        frac = (part == "attack").mean()
        assert abs(frac - 0.5) < 0.05


def test_split_without_target_returns_none_targets(classification_frame) -> None:
    x, _ = classification_frame
    cfg = {**_CFG, "stratify": False}
    x_tr, x_va, x_te, y_tr, y_va, y_te = splitting.train_val_test_split(
        x, None, cfg, seed=42
    )
    assert y_tr is None and y_va is None and y_te is None
    assert len(x_tr) + len(x_va) + len(x_te) == len(x)


def test_build_split_report_records_distribution(classification_frame) -> None:
    x, y = classification_frame
    _, _, _, y_tr, y_va, y_te = splitting.train_val_test_split(x, y, _CFG, seed=42)
    report = splitting.build_split_report(y_tr, y_va, y_te, _CFG, seed=42)
    assert report.n_total == len(x)
    assert report.stratified is True
    assert set(report.class_distribution["train"]) == {"attack", "normal"}

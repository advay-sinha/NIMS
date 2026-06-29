"""Tests for src.data.statistics (all profiling lives here now)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data import statistics as st


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": [0.0, 1.0, 2.0, 2.0, np.inf],
            "protocol_type": ["tcp", "udp", "tcp", "tcp", None],
            "label": ["normal", "attack", "attack", "attack", "normal"],
        }
    )


def test_count_missing(frame: pd.DataFrame) -> None:
    assert st.count_missing(frame) == {"protocol_type": 1}


def test_count_infinite(frame: pd.DataFrame) -> None:
    assert st.count_infinite(frame) == {"duration": 1}


def test_count_duplicates(frame: pd.DataFrame) -> None:
    # rows index 2 and 3 are identical.
    assert st.count_duplicates(frame) == 1


def test_memory_usage_positive(frame: pd.DataFrame) -> None:
    assert st.memory_usage_bytes(frame) > 0


def test_label_distribution(frame: pd.DataFrame) -> None:
    assert st.label_distribution(frame, "label") == {"attack": 3, "normal": 2}


def test_label_distribution_no_label(frame: pd.DataFrame) -> None:
    assert st.label_distribution(frame, None) == {}


def test_class_imbalance_ratio(frame: pd.DataFrame) -> None:
    dist = st.label_distribution(frame, "label")
    imb = st.class_imbalance(dist)
    assert imb["n_classes"] == 2
    assert imb["majority_class"] == "attack"
    assert imb["minority_class"] == "normal"
    assert imb["imbalance_ratio"] == pytest.approx(1.5)


def test_class_imbalance_empty() -> None:
    out = st.class_imbalance({})
    assert out["n_classes"] == 0
    assert out["imbalance_ratio"] is None


def test_numerical_summary(frame: pd.DataFrame) -> None:
    summary = st.numerical_summary(frame)
    assert "duration" in summary
    assert summary["duration"]["count"] == pytest.approx(5.0)
    assert "protocol_type" not in summary  # non-numeric excluded


def test_categorical_summary(frame: pd.DataFrame) -> None:
    summary = st.categorical_summary(frame, categorical_columns=("protocol_type",))
    assert summary["protocol_type"]["top"] == "tcp"
    assert summary["protocol_type"]["unique"] == 2
    assert summary["protocol_type"]["missing"] == 1


def test_dataset_statistics_aggregate(frame: pd.DataFrame) -> None:
    stats: dict[str, Any] = st.dataset_statistics(
        frame, label_column="label", categorical_columns=("protocol_type",)
    )
    for key in (
        "n_rows",
        "n_features",
        "memory_usage_bytes",
        "memory_usage_mb",
        "duplicate_rows",
        "total_missing",
        "missing_values",
        "infinite_values",
        "dtypes",
        "label_distribution",
        "class_imbalance",
        "numerical_summary",
        "categorical_summary",
    ):
        assert key in stats
    assert stats["n_rows"] == 5
    assert stats["n_features"] == 3
    assert stats["duplicate_rows"] == 1
    assert stats["total_missing"] == 1

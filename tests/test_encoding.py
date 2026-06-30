"""Tests for src.data.encoding."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import encoding


@pytest.fixture()
def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "proto": ["tcp", "udp", "tcp", "icmp"],
            "flag": ["S0", "SF", "SF", "S0"],
            "bytes": [100, 200, 300, 400],
        }
    )


def test_onehot_encoder_fit_transform(feature_frame: pd.DataFrame) -> None:
    fitted = encoding.fit_encoder(
        feature_frame, ["proto", "flag"], {"categorical_strategy": "onehot"}
    )
    out = encoding.apply_encoder(fitted, feature_frame)
    # Original categorical columns are gone; numeric passthrough remains.
    assert "proto" not in out.columns
    assert "bytes" in out.columns
    # tcp/udp/icmp + S0/SF = 5 indicator columns.
    assert len(fitted.feature_names_out) == 5


def test_ordinal_encoder_preserves_column_names(feature_frame: pd.DataFrame) -> None:
    fitted = encoding.fit_encoder(
        feature_frame, ["proto"], {"categorical_strategy": "ordinal"}
    )
    out = encoding.apply_encoder(fitted, feature_frame)
    assert "proto" in out.columns
    assert out["proto"].nunique() == 3


def test_encoder_handles_unknown_category_safely(feature_frame: pd.DataFrame) -> None:
    fitted = encoding.fit_encoder(
        feature_frame.iloc[:2], ["proto"], {"categorical_strategy": "onehot"}
    )
    # Row with unseen 'icmp' must not raise; ignored -> all-zero indicators.
    out = encoding.apply_encoder(fitted, feature_frame)
    assert len(out) == len(feature_frame)


def test_no_categorical_columns_is_noop() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    fitted = encoding.fit_encoder(frame, [], {"categorical_strategy": "onehot"})
    out = encoding.apply_encoder(fitted, frame)
    # Returns the original object (no defensive copy) — memory optimization.
    assert out is frame


def test_unknown_strategy_raises(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        encoding.fit_encoder(feature_frame, ["proto"], {"categorical_strategy": "bogus"})


def test_label_encoder_roundtrip() -> None:
    y = pd.Series(["normal", "attack", "attack", "normal"])
    enc = encoding.fit_label_encoder(y, column="label")
    out = encoding.apply_label_encoder(enc, y)
    assert set(out.tolist()) == {0, 1}
    assert set(enc.classes) == {"attack", "normal"}


def test_label_encoder_unknown_maps_to_minus_one() -> None:
    enc = encoding.fit_label_encoder(pd.Series(["a", "b"]))
    out = encoding.apply_label_encoder(enc, pd.Series(["a", "c"]))
    assert out[0] == enc.encoder.transform(["a"])[0]
    assert out[1] == -1

"""Tests for src.training.feature_audit (pre-fit feature-matrix audit)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.feature_audit import (
    FeatureAuditError,
    audit_features,
    load_expected_features,
)
from src.utils.io import write_json


def _frame(columns: list[str], n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(size=n) for c in columns})


def _splits(**frames: pd.DataFrame) -> dict:
    return {name: (frame, pd.Series(np.zeros(len(frame)))) for name, frame in frames.items()}


def test_audit_passes_on_consistent_splits() -> None:
    cols = ["f1", "f2", "f3"]
    splits = _splits(train=_frame(cols), validation=_frame(cols, seed=1))
    audit_features(splits, cols, "demo", "xgboost")  # must not raise


def test_audit_rejects_expected_feature_mismatch() -> None:
    splits = _splits(train=_frame(["f1", "f2"]))
    with pytest.raises(FeatureAuditError, match="do not match"):
        audit_features(splits, ["f1", "f2", "f3"], "demo", "xgboost")


def test_audit_rejects_wrong_feature_order() -> None:
    splits = _splits(train=_frame(["f2", "f1"]))
    with pytest.raises(FeatureAuditError, match="do not match"):
        audit_features(splits, ["f1", "f2"], "demo", "xgboost")


def test_audit_rejects_split_column_drift() -> None:
    splits = _splits(train=_frame(["f1", "f2"]), test=_frame(["f2", "f1"]))
    with pytest.raises(FeatureAuditError, match="differ from train"):
        audit_features(splits, None, "demo", "xgboost")


def test_audit_rejects_missing_values() -> None:
    frame = _frame(["f1", "f2"])
    frame.loc[0, "f1"] = np.nan
    with pytest.raises(FeatureAuditError, match="missing values"):
        audit_features(_splits(train=frame), None, "demo", "xgboost")


def test_audit_rejects_non_numeric_columns() -> None:
    frame = _frame(["f1"])
    frame["f2"] = "text"
    with pytest.raises(FeatureAuditError, match="non-numeric"):
        audit_features(_splits(train=frame), None, "demo", "xgboost")


def test_audit_rejects_duplicated_columns() -> None:
    frame = pd.concat([_frame(["f1", "f2"]), _frame(["f1"], seed=1)], axis=1)
    with pytest.raises(FeatureAuditError, match="duplicated"):
        audit_features(_splits(train=frame), None, "demo", "xgboost")


def test_load_expected_features_prefers_metadata(tmp_path: Path) -> None:
    # feature_metadata.json reflects post-PCA columns; it must win.
    write_json({"retained_features": ["pc_1", "pc_2"]}, tmp_path / "feature_metadata.json")
    write_json({"selected": ["f1", "f2", "f3"]}, tmp_path / "selected_features.json")
    assert load_expected_features(tmp_path) == ["pc_1", "pc_2"]


def test_load_expected_features_falls_back_to_selected(tmp_path: Path) -> None:
    write_json({"selected": ["f1", "f2"]}, tmp_path / "selected_features.json")
    assert load_expected_features(tmp_path) == ["f1", "f2"]


def test_load_expected_features_none_when_absent(tmp_path: Path) -> None:
    assert load_expected_features(tmp_path) is None


# --------------------------------------------- provenance-column consistency


def test_expected_features_exclude_provenance_columns(tmp_path: Path) -> None:
    """Stale artefacts recording provenance columns (source_file, or any
    future metadata column) must not leak into the expected model features."""
    write_json(
        {"retained_features": ["f1", "source_file", "f2", "origin"]},
        tmp_path / "feature_metadata.json",
    )
    assert load_expected_features(tmp_path) == ["f1", "f2"]


def test_expected_features_filter_applies_to_selected_fallback(tmp_path: Path) -> None:
    write_json({"selected": ["source_file", "f1"]}, tmp_path / "selected_features.json")
    assert load_expected_features(tmp_path) == ["f1"]


def test_clean_expected_features_pass_through_unchanged(tmp_path: Path) -> None:
    """Datasets without stale provenance entries (NSL-KDD, UNSW-NB15) keep
    their expected list byte-identical."""
    cols = [f"f{i}" for i in range(30)]
    write_json({"retained_features": cols}, tmp_path / "feature_metadata.json")
    assert load_expected_features(tmp_path) == cols


def test_audit_passes_when_matrix_lacks_stale_provenance_column(tmp_path: Path) -> None:
    """The CICIDS2017 failure mode: metadata records source_file, the trainer
    (correctly) drops it — the audit must pass on the reduced matrix."""
    write_json(
        {"retained_features": ["f1", "f2", "source_file"]},
        tmp_path / "feature_metadata.json",
    )
    expected = load_expected_features(tmp_path)
    splits = _splits(train=_frame(["f1", "f2"]), test=_frame(["f1", "f2"], seed=1))
    audit_features(splits, expected, "cicids2017", "mlp")  # must not raise


def test_audit_still_rejects_missing_real_feature_after_filter(tmp_path: Path) -> None:
    write_json(
        {"retained_features": ["f1", "f2", "source_file"]},
        tmp_path / "feature_metadata.json",
    )
    expected = load_expected_features(tmp_path)
    with pytest.raises(FeatureAuditError, match=r"missing=\['f2'\]"):
        audit_features(_splits(train=_frame(["f1"])), expected, "demo", "mlp")


def test_audit_still_rejects_unexpected_extra_feature(tmp_path: Path) -> None:
    write_json({"retained_features": ["f1", "f2"]}, tmp_path / "feature_metadata.json")
    expected = load_expected_features(tmp_path)
    with pytest.raises(FeatureAuditError, match=r"unexpected=\['rogue'\]"):
        audit_features(
            _splits(train=_frame(["f1", "f2", "rogue"])), expected, "demo", "mlp"
        )


def test_audit_still_rejects_order_mismatch_after_filter(tmp_path: Path) -> None:
    write_json(
        {"retained_features": ["f1", "source_file", "f2"]},
        tmp_path / "feature_metadata.json",
    )
    expected = load_expected_features(tmp_path)
    with pytest.raises(FeatureAuditError, match="do not match"):
        audit_features(_splits(train=_frame(["f2", "f1"])), expected, "demo", "mlp")


_REAL_FEATURES = Path("outputs/features")


@pytest.mark.skipif(
    not (_REAL_FEATURES / "cicids2017" / "train.parquet").is_file(),
    reason="local feature artefacts not present",
)
@pytest.mark.parametrize("dataset_id", ["cicids2017", "nsl_kdd", "unsw_nb15"])
def test_real_artifacts_expected_features_match_trainer_matrix(dataset_id: str) -> None:
    """On this machine, the audit's expected list must equal the trainer's
    post-reduction matrix columns for every engineered dataset."""
    import pyarrow.parquet as pq

    from src.features.metadata import filter_provenance_names

    feat_dir = _REAL_FEATURES / dataset_id
    expected = load_expected_features(feat_dir)
    schema = pq.read_schema(feat_dir / "train.parquet")
    # Reproduce the trainer's reduction on the persisted columns (label last).
    matrix_columns = filter_provenance_names(schema.names[:-1])
    assert expected == matrix_columns

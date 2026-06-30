"""Tests for the data preprocessing stage modules (Phase 2)."""

from __future__ import annotations

import pandas as pd

from src.data import cleaning, encoding, scaling, validation


def test_clean_dataset_returns_frame_and_report() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned, report = cleaning.clean_dataset(frame, {"drop_duplicates": True})
    assert report.rows_before == 3
    assert report.rows_after == 2
    assert report.duplicates_removed == 1


def test_fit_encoder_returns_fitted_encoder() -> None:
    x = pd.DataFrame({"proto": ["tcp", "udp", "tcp"], "n": [1, 2, 3]})
    fitted = encoding.fit_encoder(x, ["proto"], {"categorical_strategy": "onehot"})
    assert fitted.strategy == "onehot"
    assert fitted.columns == ("proto",)
    assert len(fitted.feature_names_out) >= 2


def test_fit_scaler_returns_fitted_scaler() -> None:
    x = pd.DataFrame({"n": [1.0, 2.0, 3.0]})
    fitted = scaling.fit_scaler(x, ["n"], {"numeric_strategy": "standard"})
    assert fitted.strategy == "standard"
    assert fitted.columns == ("n",)
    assert fitted.scaler is not None


def test_validation_report_records_error_and_flips_passed() -> None:
    report = validation.ValidationReport(dataset_id="nsl_kdd")
    assert report.passed is True
    report.add("warning", "minor", "a warning does not fail validation")
    assert report.passed is True
    report.add("error", "missing_column", "label column absent")
    assert report.passed is False
    assert len(report.issues) == 2

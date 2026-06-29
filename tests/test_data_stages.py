"""Tests for the data preprocessing stage modules.

Phase 1 stages are stubs; these tests assert the not-yet-implemented contract
(``NotImplementedError``) so that implementing a stage flips its test to pass.
Replace each xfail with behavioural assertions as stages land.
"""

from __future__ import annotations

import pytest

from src.data import cleaning, encoding, scaling, validation


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_clean_dataset_stub() -> None:
    cleaning.clean_dataset(frame=None, cleaning_config={})


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_fit_encoder_stub() -> None:
    encoding.fit_encoder(x_train=None, categorical_columns=[], encoding_config={})


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_fit_scaler_stub() -> None:
    scaling.fit_scaler(x_train=None, numeric_columns=[], scaling_config={})


def test_validation_report_records_error_and_flips_passed() -> None:
    report = validation.ValidationReport(dataset_id="nsl_kdd")
    assert report.passed is True
    report.add("warning", "minor", "a warning does not fail validation")
    assert report.passed is True
    report.add("error", "missing_column", "label column absent")
    assert report.passed is False
    assert len(report.issues) == 2

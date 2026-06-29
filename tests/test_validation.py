"""Tests for src.data.validation (schema/datatype/rule + orchestration)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data import validation as v


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration": [0.0, 1.0, 2.0, 2.0, np.inf],
            "protocol_type": ["tcp", "udp", "tcp", "tcp", "tcp"],
            "label": ["normal", "attack", "attack", "attack", "normal"],
        }
    )


@pytest.fixture()
def dataset_config() -> dict[str, Any]:
    return {
        "id": "nsl_kdd",
        "name": "NSL-KDD",
        "engine": "A",
        "columns": ["duration", "protocol_type", "label"],
        "categorical_columns": ["protocol_type"],
        "label_column": "label",
    }


# -- ValidationReport ------------------------------------------------------- #
def test_validation_report_error_flips_passed() -> None:
    report = v.ValidationReport(dataset_id="x")
    report.add("warning", "w", "non-fatal")
    assert report.passed is True
    report.add("error", "e", "fatal")
    assert report.passed is False
    assert len(report.as_list()) == 2


# -- Individual rules ------------------------------------------------------- #
def test_validate_required_columns_missing_label() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    report = v.ValidationReport(dataset_id="x")
    v.validate_required_columns(frame, {"label_column": "label"}, report)
    assert report.passed is False
    assert report.issues[0].code == "missing_label_column"


def test_validate_datatypes_categorical_as_numeric() -> None:
    frame = pd.DataFrame({"proto": [0, 1, 2], "label": [0, 1, 0]})
    report = v.ValidationReport(dataset_id="x")
    v.validate_datatypes(
        frame, {"categorical_columns": ["proto"], "label_column": "label"}, report
    )
    codes = {i.code for i in report.issues}
    assert "categorical_stored_as_numeric" in codes


def test_validate_rules_flags_quality_issues() -> None:
    stats = {
        "n_rows": 100,
        "duplicate_rows": 5,
        "total_missing": 3,
        "missing_values": {"a": 3},
        "infinite_values": {"b": 2},
        "class_imbalance": {"imbalance_ratio": 50.0},
    }
    report = v.ValidationReport(dataset_id="x")
    v.validate_rules(stats, {}, report)
    codes = {i.code for i in report.issues}
    assert {
        "duplicate_rows",
        "missing_values",
        "infinite_values",
        "severe_class_imbalance",
    } <= codes


def test_validate_rules_all_null_column_is_error() -> None:
    stats = {"n_rows": 10, "missing_values": {"c": 10}, "class_imbalance": {}}
    report = v.ValidationReport(dataset_id="x")
    v.validate_rules(stats, {}, report)
    assert report.passed is False
    assert any(i.code == "all_null_column" for i in report.issues)


# -- Orchestration ---------------------------------------------------------- #
def test_build_report_delegates_to_statistics(
    frame: pd.DataFrame, dataset_config: dict[str, Any]
) -> None:
    report = v.build_report(frame, dataset_config)
    assert report.dataset_id == "nsl_kdd"
    assert report.n_rows == 5
    assert report.n_features == 3
    # Profiling now lives under `statistics`.
    assert report.statistics["duplicate_rows"] == 1
    assert report.statistics["infinite_values"] == {"duration": 1}
    assert report.statistics["label_distribution"] == {"attack": 3, "normal": 2}
    assert report.schema_passed is True
    assert isinstance(report.to_dict(), dict)
    assert "statistics" in report.to_dict()


def test_build_report_flags_missing_label(frame: pd.DataFrame) -> None:
    report = v.build_report(frame, {"id": "x", "label_column": "nope"})
    assert report.schema_passed is False
    codes = {issue["code"] for issue in report.validation_issues}
    assert "missing_label_column" in codes


def test_save_report_writes_json(tmp_path, frame, dataset_config) -> None:
    import json

    report = v.build_report(frame, dataset_config)
    out = tmp_path / "reports" / "nsl_kdd_report.json"
    written = v.save_report(report, out)
    assert written.exists()
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["dataset_id"] == "nsl_kdd"
    assert loaded["statistics"]["n_rows"] == 5

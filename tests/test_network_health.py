"""Tests for src.network_health (Engine B foundation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.network_health.baseline import train_baseline
from src.network_health.features import build_features
from src.network_health.preprocessing import (
    chronological_split,
    compute_counter_rates,
    preprocess_telemetry,
)
from src.network_health.schema import TelemetrySchema
from src.network_health.validation import validate_telemetry

_SCHEMA_CONFIG = {
    "timestamp_column": "timestamp",
    "device_column": "device_id",
    "interface_column": "interface_id",
    "label_column": "label",
    "required_columns": [
        "timestamp", "device_id", "interface_id",
        "ifInOctets", "ifInErrors", "ifOperStatus", "cpu_usage",
    ],
    "counter_columns": ["ifInOctets", "ifInErrors"],
    "gauge_columns": ["cpu_usage"],
    "status_columns": ["ifOperStatus"],
    "bounded_columns": {"cpu_usage": [0, 100]},
}

_FEATURES_CONFIG = {
    "rolling_windows": [3], "rolling_stats": ["mean", "std", "max"], "lags": [1],
}


def _schema(**overrides) -> TelemetrySchema:
    return TelemetrySchema.from_config({**_SCHEMA_CONFIG, **overrides})


def _telemetry(
    n_per_series: int = 60,
    *,
    with_labels: bool = True,
    anomaly_rows: int = 6,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic telemetry: 2 devices x 1 interface, monotonic counters."""
    rng = np.random.default_rng(seed)
    frames = []
    for device in ("sw1", "sw2"):
        times = pd.date_range("2026-01-01", periods=n_per_series, freq="5min")
        octets = np.cumsum(rng.integers(1_000, 5_000, n_per_series))
        errors = np.cumsum(rng.integers(0, 2, n_per_series))
        cpu = rng.uniform(10, 40, n_per_series)
        # Benign excursions: healthy telemetry occasionally spikes, so the
        # trained forest has in-range density structure to isolate against.
        population = max(1, n_per_series - 10)
        n_spikes = min(max(2, n_per_series // 15), population)
        spike_rows = rng.choice(population, n_spikes, replace=False)
        cpu[spike_rows] = rng.uniform(70, 92, n_spikes)
        errors[spike_rows] = errors[spike_rows] + rng.integers(
            20, 60, n_spikes
        )
        errors = np.maximum.accumulate(errors)  # keep the counter monotonic
        label = np.zeros(n_per_series, dtype=int)
        if with_labels and device == "sw1":
            n_sick = min(anomaly_rows, n_per_series)
            sick = slice(n_per_series - n_sick, n_per_series)
            errors[sick] = errors[sick] + np.cumsum(
                rng.integers(500, 900, n_sick)
            )
            cpu[sick] = rng.uniform(95, 100, n_sick)
            label[sick] = 1
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": times,
                    "device_id": device,
                    "interface_id": "eth0",
                    "ifInOctets": octets,
                    "ifInErrors": errors,
                    "ifOperStatus": 1,
                    "cpu_usage": cpu,
                    "label": label,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    if not with_labels:
        frame = frame.drop(columns=["label"])
    return frame


# ---------------------------------------------------------------- validation


def test_schema_validation_success() -> None:
    report = validate_telemetry(_telemetry(), _schema(), "demo")
    assert report.passed
    assert report.n_devices == 2
    assert report.n_interfaces == 2
    assert report.coverage["n_series"] == 2


def test_missing_required_column_fails() -> None:
    frame = _telemetry().drop(columns=["cpu_usage"])
    report = validate_telemetry(frame, _schema(), "demo")
    assert not report.passed
    assert any(
        i["check"] == "required_columns" and "cpu_usage" in i["message"]
        for i in report.issues
    )


def test_unparseable_timestamps_fail() -> None:
    frame = _telemetry()
    frame["timestamp"] = frame["timestamp"].astype(str)  # as read from CSV
    frame.loc[0, "timestamp"] = "not-a-time"
    report = validate_telemetry(frame, _schema(), "demo")
    assert not report.passed
    assert any(i["check"] == "timestamp" for i in report.issues)


def test_negative_counter_detected() -> None:
    frame = _telemetry()
    frame.loc[3, "ifInOctets"] = -5
    report = validate_telemetry(frame, _schema(), "demo")
    assert not report.passed
    assert any(i["check"] == "negative_values" for i in report.issues)


def test_counter_reset_warning() -> None:
    frame = _telemetry()
    frame.loc[10, "ifInOctets"] = 1  # mid-series drop = reset
    report = validate_telemetry(frame, _schema(), "demo")
    assert report.passed  # warning, not error
    assert any(i["check"] == "counter_monotonicity" for i in report.issues)


def test_duplicate_detection() -> None:
    frame = _telemetry()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate_telemetry(frame, _schema(), "demo")
    assert any(i["check"] == "duplicates" for i in report.issues)


# ------------------------------------------------------------- preprocessing


def test_counter_delta_and_rate_calculation() -> None:
    schema = _schema()
    frame = _telemetry(n_per_series=5).sort_values(
        ["device_id", "interface_id", "timestamp"]
    )
    result = compute_counter_rates(frame, schema)
    sw1 = result[result.device_id == "sw1"]
    expected_delta = sw1["ifInOctets"].diff().fillna(0.0)
    assert list(result.columns).count("ifInOctets_delta") == 1
    assert np.allclose(sw1["ifInOctets_delta"], expected_delta)
    # 5-minute interval -> rate = delta / 300 s (first row 0).
    assert np.allclose(
        sw1["ifInOctets_rate"].iloc[1:], expected_delta.iloc[1:] / 300.0
    )
    assert sw1["ifInOctets_rate"].iloc[0] == 0.0


def test_counter_reset_clipped_to_zero() -> None:
    schema = _schema()
    frame = _telemetry(n_per_series=5)
    frame.loc[2, "ifInOctets"] = 0  # reset mid-series
    frame = frame.sort_values(["device_id", "interface_id", "timestamp"])
    result = compute_counter_rates(frame, schema)
    assert (result["ifInOctets_delta"] >= 0).all()


def test_chronological_split_no_time_overlap() -> None:
    schema = _schema()
    frame = _telemetry()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    splits = chronological_split(
        frame, schema, {"train": 0.7, "validation": 0.15, "test": 0.15}
    )
    assert all(len(part) > 0 for part in splits.values())
    assert splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()
    total = sum(len(part) for part in splits.values())
    assert total == len(frame)


def test_preprocess_telemetry_end_to_end() -> None:
    result = preprocess_telemetry(
        _telemetry(), _schema(),
        {"split": {"train": 0.7, "validation": 0.15, "test": 0.15}},
        "demo",
    )
    assert set(result.splits) == {"train", "validation", "test"}
    assert result.manifest["n_duplicates_removed"] == 0
    train = result.splits["train"]
    assert "ifInOctets_rate" in train.columns
    assert not train[_schema().numeric_columns].isna().any().any()


# ------------------------------------------------------------------ features


def test_rolling_lag_and_status_features() -> None:
    schema = _schema()
    processed = preprocess_telemetry(
        _telemetry(), schema, {"split": {"train": 1.0, "validation": 0.0,
                                         "test": 0.0}}, "demo",
    ).splits["train"]
    features, metadata = build_features(processed, schema, _FEATURES_CONFIG)

    assert "traffic_in_rate" in features.columns
    assert "error_rate_in" in features.columns
    assert "traffic_in_rate_roll3_mean" in features.columns
    assert "cpu_usage_roll3_std" in features.columns
    assert "cpu_usage_roll3_max" in features.columns
    assert "traffic_in_rate_lag1" in features.columns
    assert "ifOperStatus_changed" in features.columns
    assert metadata["n_features"] == len(metadata["feature_columns"])
    assert not features[metadata["feature_columns"]].isna().any().any()

    # Rolling mean sanity: window-3 mean of a constant series is constant.
    sw1 = features[features.device_id == "sw1"]
    manual = sw1["cpu_usage"].rolling(3, min_periods=1).mean()
    assert np.allclose(sw1["cpu_usage_roll3_mean"], manual)


# ------------------------------------------------------------------ baseline


def _feature_splits(with_labels: bool) -> tuple[dict, list[str], TelemetrySchema]:
    schema = _schema(label_column="label" if with_labels else None)
    result = preprocess_telemetry(
        _telemetry(with_labels=with_labels), schema,
        {"split": {"train": 0.7, "validation": 0.15, "test": 0.15}}, "demo",
    )
    splits = {}
    metadata = {}
    for name, frame in result.splits.items():
        splits[name], metadata = build_features(frame, schema, _FEATURES_CONFIG)
    return splits, list(metadata["feature_columns"]), schema


def test_isolation_forest_training_labeled_metrics() -> None:
    splits, columns, schema = _feature_splits(with_labels=True)
    baseline, metrics = train_baseline(
        splits, columns, schema,
        {"params": {"n_estimators": 50, "random_state": 42},
         "threshold_quantile": 0.99},
        seed=42,
    )
    assert baseline.labeled is True
    assert metrics["test"]["mode"] == "labeled"
    for key in ("precision", "recall", "f1", "roc_auc", "confusion_matrix"):
        assert key in metrics["test"]
    # The injected error/CPU burst lives at the end of sw1 => test split.
    assert metrics["test"]["recall"] > 0.5


def test_unlabeled_dataset_behavior() -> None:
    splits, columns, schema = _feature_splits(with_labels=False)
    baseline, metrics = train_baseline(
        splits, columns, schema,
        {"params": {"n_estimators": 50, "random_state": 42},
         "threshold_quantile": 0.95},
        seed=42,
    )
    assert baseline.labeled is False
    assert metrics["train"]["mode"] == "unlabeled"
    assert 0.0 <= metrics["train"]["anomaly_rate"] <= 0.1
    assert "score_distribution" in metrics["train"]
    assert metrics["train"]["threshold"] == pytest.approx(baseline.threshold)


def test_artifact_persistence(tmp_path: Path) -> None:
    import joblib

    from src.network_health.artifacts import (
        write_experiment,
        write_feature_splits,
        write_processed_splits,
        write_validation_report,
    )

    schema = _schema()
    frame = _telemetry()
    report = validate_telemetry(frame, schema, "demo")
    paths = write_validation_report(report, tmp_path)
    assert paths["json"].is_file() and paths["markdown"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["passed"] is True

    result = preprocess_telemetry(
        frame, schema, {"split": {"train": 0.7, "validation": 0.15,
                                  "test": 0.15}}, "demo",
    )
    processed = write_processed_splits(result, tmp_path, "demo")
    assert processed["train"].is_file() and processed["manifest"].is_file()

    splits, columns, schema2 = _feature_splits(with_labels=True)
    metadata = {"feature_columns": columns, "n_features": len(columns)}
    features = write_feature_splits(splits, metadata, tmp_path, "demo")
    assert features["train"].is_file() and features["metadata"].is_file()

    baseline, metrics = train_baseline(
        splits, columns, schema2,
        {"params": {"n_estimators": 20, "random_state": 42}}, seed=42,
    )
    experiment = write_experiment(
        baseline, metrics, tmp_path, "demo", config_snapshot={}, seed=42,
    )
    assert experiment["model"].is_file()
    manifest = json.loads(experiment["manifest"].read_text(encoding="utf-8"))
    assert manifest["engine"] == "network_health"
    assert manifest["n_features"] == len(columns)
    reloaded = joblib.load(experiment["model"])
    assert reloaded.feature_columns == columns

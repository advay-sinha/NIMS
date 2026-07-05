"""Tests for src.api (loader, service, endpoints)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.api.loader import ModelBundle, _required_input_columns
from src.utils.io import write_json

_FEATURES = ["f1", "f2"]


class _FakeModel:
    """Two-class stub predicting class 1 for every row."""

    def __init__(self, with_proba: bool = True) -> None:
        self.with_proba = with_proba
        self.classes_ = np.array([0, 1])

    def predict(self, x):  # noqa: ANN001, ANN201
        return np.ones(len(x), dtype=int)

    def predict_proba(self, x):  # noqa: ANN001, ANN201
        if not self.with_proba:
            return None
        return np.tile([0.2, 0.8], (len(x), 1))


def _label_encoder() -> SimpleNamespace:
    """A saved-label-encoder stand-in with a real sklearn inverse_transform."""
    from sklearn.preprocessing import LabelEncoder

    sk = LabelEncoder()
    sk.classes_ = np.array(["normal", "attack"])  # id 0 -> normal, 1 -> attack
    return SimpleNamespace(classes=("normal", "attack"), encoder=sk)


def _bundle(with_proba: bool = True, feature_selector=None) -> ModelBundle:
    return ModelBundle(
        dataset="demo",
        stage="production",
        resolution={"model_type": "xgboost", "experiment_id": "demo_exp_1"},
        model=_FakeModel(with_proba),
        encoder=None,
        scaler=None,
        label_encoder=_label_encoder(),
        feature_selector=feature_selector,
        expected_features=list(_FEATURES),
        required_columns=list(_FEATURES),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    """A TestClient whose bundle loader returns the fake bundle."""
    from fastapi.testclient import TestClient

    import src.api.loader as loader_module
    from src.api.app import create_app

    calls = {"n": 0}

    def _fake_load(dataset, stage, registry_dir):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if dataset != "demo":
            from src.api.errors import ModelNotAvailableError

            raise ModelNotAvailableError(
                f"No production model assigned for dataset '{dataset}'."
            )
        return _bundle()

    monkeypatch.setattr(loader_module, "load_bundle", _fake_load)
    app = create_app(
        config={"api": {"max_rows_per_request": 3}},
        registry_dir=tmp_path / "registry",
    )
    test_client = TestClient(app)
    test_client.load_calls = calls  # type: ignore[attr-defined]
    return test_client


# ---------------------------------------------------------------- endpoints


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "netsentinel-inference"
    assert payload["loaded_models"] == 0


def test_models_endpoint_lists_production(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    write_json(
        {"entries": [{
            "experiment_id": "demo_exp_1", "dataset": "demo",
            "model_type": "xgboost", "status": "production",
            "metrics": {"test": {"f1": 0.95}},
        }]},
        registry_dir / "registry.json",
    )
    write_json(
        {"demo": {"experiment_id": "demo_exp_1", "model_type": "xgboost"}},
        registry_dir / "production.json",
    )
    client = TestClient(create_app(config={}, registry_dir=registry_dir))
    payload = client.get("/models").json()
    assert payload["models"] == [{
        "dataset": "demo", "model_type": "xgboost",
        "experiment_id": "demo_exp_1",
        "metrics": {"test": {"f1": 0.95}}, "status": "production",
    }]


def test_json_prediction(client) -> None:
    response = client.post(
        "/predict-json/demo",
        json={"rows": [{"f1": 1.0, "f2": 2.0}, {"f1": 0.5, "f2": 0.1}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == "demo"
    assert payload["model_type"] == "xgboost"
    assert payload["experiment_id"] == "demo_exp_1"
    assert payload["n_rows"] == 2
    assert payload["predictions"] == ["attack", "attack"]  # decoded labels
    assert payload["probabilities"] == [[0.2, 0.8], [0.2, 0.8]]
    assert payload["class_labels"] == ["normal", "attack"]
    assert payload["warnings"] == []


def test_csv_upload_prediction(client) -> None:
    response = client.post(
        "/predict/demo",
        files={"file": ("rows.csv", b"f1,f2\n1.0,2.0\n3.0,4.0\n", "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_rows"] == 2
    assert payload["predictions"] == ["attack", "attack"]


def test_missing_production_model_is_404(client) -> None:
    response = client.post(
        "/predict-json/unknown_ds", json={"rows": [{"f1": 1.0, "f2": 2.0}]}
    )
    assert response.status_code == 404
    assert "No production model" in response.json()["detail"]


def test_invalid_csv_is_422(client) -> None:
    response = client.post(
        "/predict/demo", files={"file": ("rows.csv", b"", "text/csv")}
    )
    assert response.status_code == 422
    assert "CSV" in response.json()["detail"]


def test_too_many_rows_is_413(client) -> None:
    rows = [{"f1": 1.0, "f2": 2.0}] * 4  # limit configured to 3
    response = client.post("/predict-json/demo", json={"rows": rows})
    assert response.status_code == 413
    assert "limit is 3" in response.json()["detail"]


def test_missing_columns_is_422(client) -> None:
    response = client.post(
        "/predict-json/demo", json={"rows": [{"f1": 1.0, "wrong": 2.0}]}
    )
    assert response.status_code == 422
    assert "f2" in response.json()["detail"]


def test_extra_columns_produce_warning(client) -> None:
    response = client.post(
        "/predict-json/demo",
        json={"rows": [{"f1": 1.0, "f2": 2.0, "extra": 9}]},
    )
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert len(warnings) == 1 and "extra" in warnings[0]


def test_model_cache_reuse(client) -> None:
    body = {"rows": [{"f1": 1.0, "f2": 2.0}]}
    assert client.post("/predict-json/demo", json=body).status_code == 200
    assert client.post("/predict-json/demo", json=body).status_code == 200
    assert client.load_calls["n"] == 1  # loaded once, cached after
    assert client.get("/health").json()["loaded_models"] == 1


# ------------------------------------------------------------ service level


def test_probabilities_omitted_without_predict_proba() -> None:
    from src.api.service import run_inference
    import pandas as pd

    payload = run_inference(
        _bundle(with_proba=False),
        pd.DataFrame({"f1": [1.0], "f2": [2.0]}),
        max_rows=10,
    )
    assert payload["predictions"] == ["attack"]
    assert "probabilities" not in payload
    assert "class_labels" not in payload


def test_nan_rows_rejected() -> None:
    from src.api.errors import RequestValidationError
    from src.api.service import run_inference
    import pandas as pd

    frame = pd.DataFrame({"f1": [1.0, None], "f2": [2.0, 3.0]})
    with pytest.raises(RequestValidationError, match="missing value"):
        run_inference(_bundle(), frame, max_rows=10)


def test_feature_selector_transform_applied() -> None:
    """The saved selector reduces the transformed frame before alignment."""
    import pandas as pd

    from src.api.service import run_inference
    from src.features.selection import FeatureSelector

    selector = FeatureSelector(selected_features=["f2", "f1"], method="test")
    bundle = _bundle(feature_selector=selector)
    frame = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    payload = run_inference(bundle, frame, max_rows=10)
    assert payload["predictions"] == ["attack"]


def test_label_decode_uses_inverse_transform() -> None:
    from src.api.service import _decode_labels

    bundle = _bundle()
    assert _decode_labels(bundle, [0, 1]) == ["normal", "attack"]
    # Out-of-range / unknown sentinel values pass through as integers.
    assert _decode_labels(bundle, [1, -1, 7]) == ["attack", -1, 7]


def test_loader_rejects_missing_preprocessing_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """A missing saved transform is a clear load error, never a silent skip."""
    import joblib

    import src.api.loader as loader_module
    from src.api.errors import ModelNotAvailableError

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    joblib.dump(_FakeModel(), run_dir / "model.joblib")
    pre_dir = tmp_path / "pre"
    pre_dir.mkdir()
    joblib.dump(object(), pre_dir / "encoder.joblib")  # scaler & co. absent

    monkeypatch.setattr(
        "src.registry.resolver.resolve_model",
        lambda dataset, stage, registry_dir: {
            "experiment_id": "run",
            "model_type": "xgboost",
            "model_artifact_path": str(run_dir / "model.joblib"),
            "manifest_path": str(run_dir / "manifest.json"),
            "metrics": {},
            "status": "production",
            "artifacts": {"preprocessing_artifacts": str(pre_dir)},
        },
    )
    with pytest.raises(ModelNotAvailableError, match="scaler.joblib"):
        loader_module.load_bundle("demo", "production", tmp_path)


def test_required_columns_derivation() -> None:
    encoder = SimpleNamespace(
        columns=("proto", "service"),
        feature_names_out=("proto_tcp", "proto_udp", "service_http"),
    )
    # The scaler was fit before feature selection, so its full column set is
    # required even when the final model kept only a subset (e.g. dbytes).
    scaler = SimpleNamespace(columns=("sttl", "rate", "dbytes"))
    required = _required_input_columns(
        ["proto_tcp", "sttl", "service_http", "rate"], encoder, scaler
    )
    assert required == ["proto", "service", "sttl", "rate", "dbytes"]
    assert _required_input_columns(["f1", "f2"], None) == ["f1", "f2"]

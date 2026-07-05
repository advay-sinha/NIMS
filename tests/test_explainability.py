"""Tests for src.explainability (SHAP backends, registry, artefacts, runner)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explainability.artifacts import (
    feature_importance_frame,
    write_explanation_artifacts,
)
from src.explainability.base import ExplainabilityError, resolve_feature_names
from src.explainability.registry import available_explainers, get_explainer
from src.explainability.runner import explain_model, maybe_explain_after_training
from src.models.registry import build_model

_FEATURES = ["f1", "f2", "f3", "f4"]


@pytest.fixture(scope="module")
def fitted_xgboost() -> tuple:
    """A small fitted 3-class XGBoost model plus its training matrix."""
    rng = np.random.default_rng(3)
    n = 150
    y = pd.Series(np.arange(n) % 3, name="label")
    x = pd.DataFrame(
        {
            "f1": y + rng.normal(0, 0.1, n),
            "f2": rng.normal(size=n),
            "f3": -y + rng.normal(0, 0.2, n),
            "f4": rng.normal(size=n),
        }
    )
    model = build_model(
        "xgboost", {"gpu": False, "params": {"n_estimators": 10}}, use_gpu=False,
        seed=42,
    )
    model.fit(x, y)
    return model, x


def _config(**explainability: object) -> dict:
    return {
        "project": {"seed": 42},
        "training": {"random_seed": 42},
        "explainability": {"enabled": True, "split": "test", **explainability},
    }


# ------------------------------------------------------------------ registry


def test_registry_lists_xgboost_only() -> None:
    assert available_explainers() == ["xgboost"]


def test_registry_rejects_unsupported_model() -> None:
    with pytest.raises(ExplainabilityError, match="No explainability backend"):
        get_explainer("lightgbm")


def test_explainer_rejects_wrong_model_wrapper(fitted_xgboost) -> None:
    _, x = fitted_xgboost
    wrong = build_model(
        "lightgbm", {"gpu": False, "params": {"n_estimators": 5}}, use_gpu=False,
        seed=42,
    )
    with pytest.raises(ExplainabilityError, match="cannot explain model"):
        get_explainer("xgboost").explain(wrong, x)


def test_explainer_rejects_unfitted_model() -> None:
    model = build_model(
        "xgboost", {"gpu": False, "params": {"n_estimators": 5}}, use_gpu=False,
        seed=42,
    )
    with pytest.raises(ExplainabilityError, match="not fitted"):
        get_explainer("xgboost").explain(model, pd.DataFrame({"f1": [1.0]}))


# ------------------------------------------------------------- feature names


def test_missing_feature_names_rejected(fitted_xgboost) -> None:
    model, x = fitted_xgboost
    with pytest.raises(ExplainabilityError, match="no named columns"):
        get_explainer("xgboost").explain(model, x.to_numpy())


def test_resolve_feature_names_returns_strings(fitted_xgboost) -> None:
    _, x = fitted_xgboost
    assert resolve_feature_names(x) == _FEATURES


# ------------------------------------------------------------- SHAP results


def test_multiclass_shap_values_shape(fitted_xgboost) -> None:
    model, x = fitted_xgboost
    result = get_explainer("xgboost").explain(model, x)
    assert result.values.shape == (len(x), len(_FEATURES), 3)
    assert result.feature_names == _FEATURES
    assert result.n_samples == len(x)
    assert len(result.class_labels) == 3


def test_feature_importance_ranks_descending(fitted_xgboost) -> None:
    model, x = fitted_xgboost
    frame = feature_importance_frame(get_explainer("xgboost").explain(model, x))
    assert list(frame.columns) == ["feature", "mean_abs_shap", "rank"]
    assert list(frame["rank"]) == [1, 2, 3, 4]
    assert frame["mean_abs_shap"].is_monotonic_decreasing
    # f1/f3 carry the class signal; a noise feature cannot rank first.
    assert frame.iloc[0]["feature"] in {"f1", "f3"}


# ---------------------------------------------------------------- artefacts


def test_artifact_generation_and_persistence(fitted_xgboost, tmp_path: Path) -> None:
    model, x = fitted_xgboost
    paths = explain_model(
        model, x,
        experiment_id="demo_xgboost_20260101T000000",
        dataset_id="demo",
        config=_config(max_samples=64),
        output_root=tmp_path,
    )
    out_dir = tmp_path / "demo_xgboost_20260101T000000"
    assert set(paths) == {
        "metadata", "feature_importance", "shap_values", "global_summary",
        "global_feature_importance", "local_explanations",
    }
    for name, path in paths.items():
        if name == "local_explanations":
            assert path.is_dir() and path.parent == out_dir
        else:
            assert path.is_file() and path.parent == out_dir

    import json

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment_id"] == "demo_xgboost_20260101T000000"
    assert metadata["model_type"] == "xgboost"
    assert metadata["dataset"] == "demo"
    assert metadata["n_samples_explained"] == 64  # max_samples cap applied
    assert "timestamp" in metadata

    import shap

    assert metadata["shap_version"] == shap.__version__

    frame = pd.read_csv(out_dir / "feature_importance.csv")
    assert list(frame.columns) == ["feature", "mean_abs_shap", "rank"]
    assert set(frame["feature"]) == set(_FEATURES)

    import joblib

    payload = joblib.load(out_dir / "shap_values.pkl")
    assert payload["values"].shape == (64, len(_FEATURES), 3)
    assert payload["feature_names"] == _FEATURES
    assert len(payload["base_values"]) == 3

    assert (out_dir / "global_summary.png").stat().st_size > 0


def test_sampling_is_deterministic(fitted_xgboost, tmp_path: Path) -> None:
    import joblib

    model, x = fitted_xgboost
    values = []
    for run in ("a", "b"):
        explain_model(
            model, x,
            experiment_id=run, dataset_id="demo",
            config=_config(max_samples=32), output_root=tmp_path,
        )
        values.append(joblib.load(tmp_path / run / "shap_values.pkl")["values"])
    assert np.array_equal(values[0], values[1])


def test_binary_model_artifacts(tmp_path: Path) -> None:
    """Binary targets yield a single-output explanation and still persist."""
    rng = np.random.default_rng(5)
    n = 120
    y = pd.Series(np.arange(n) % 2, name="label")
    x = pd.DataFrame({"f1": y + rng.normal(0, 0.1, n), "f2": rng.normal(size=n)})
    model = build_model(
        "xgboost", {"gpu": False, "params": {"n_estimators": 10}}, use_gpu=False,
        seed=42,
    )
    model.fit(x, y)
    result = get_explainer("xgboost").explain(model, x)
    assert result.values.shape == (n, 2, 1)
    paths = explain_model(
        model, x, experiment_id="bin", dataset_id="demo",
        config=_config(), output_root=tmp_path,
    )
    assert all(p.exists() for p in paths.values())
    assert paths["local_explanations"].is_dir()


# ------------------------------------------- Phase 1.2: global + local files


_GLOBAL_COLUMNS = [
    "feature", "mean_abs_shap", "std_abs_shap", "percentage_contribution",
    "cumulative_percentage", "rank",
]
_LOCAL_COLUMNS = [
    "feature", "feature_value", "shap_contribution", "abs_contribution",
    "contribution_rank",
]


@pytest.fixture()
def explained_dir(fitted_xgboost, tmp_path: Path) -> Path:
    model, x = fitted_xgboost
    explain_model(
        model, x,
        experiment_id="p12", dataset_id="demo",
        config=_config(
            max_samples=64,
            **{"local_explanations": {"enabled": True, "max_samples": 5}},
        ),
        output_root=tmp_path,
    )
    return tmp_path / "p12"


def test_global_feature_importance_created_with_columns(explained_dir: Path) -> None:
    frame = pd.read_csv(explained_dir / "global_feature_importance.csv")
    assert list(frame.columns) == _GLOBAL_COLUMNS
    assert list(frame["rank"]) == [1, 2, 3, 4]
    assert frame["mean_abs_shap"].is_monotonic_decreasing


def test_global_percentages_sum_to_100(explained_dir: Path) -> None:
    frame = pd.read_csv(explained_dir / "global_feature_importance.csv")
    assert frame["percentage_contribution"].sum() == pytest.approx(100.0)
    assert frame["cumulative_percentage"].iloc[-1] == pytest.approx(100.0)
    assert frame["cumulative_percentage"].is_monotonic_increasing


def test_local_folder_and_columns(explained_dir: Path) -> None:
    local_dir = explained_dir / "local"
    assert local_dir.is_dir()
    files = sorted(local_dir.glob("sample_*.csv"))
    assert [f.name for f in files] == [f"sample_{i:04d}.csv" for i in range(1, 6)]
    frame = pd.read_csv(files[0])
    assert list(frame.columns) == _LOCAL_COLUMNS
    assert list(frame["contribution_rank"]) == [1, 2, 3, 4]
    assert frame["abs_contribution"].is_monotonic_decreasing
    assert set(frame["feature"]) == set(_FEATURES)


def test_local_sample_selection_is_deterministic(
    fitted_xgboost, tmp_path: Path
) -> None:
    model, x = fitted_xgboost
    frames = []
    for run in ("a", "b"):
        explain_model(
            model, x,
            experiment_id=run, dataset_id="demo",
            config=_config(
                max_samples=64,
                **{"local_explanations": {"enabled": True, "max_samples": 3}},
            ),
            output_root=tmp_path,
        )
        frames.append(
            [pd.read_csv(p) for p in sorted((tmp_path / run / "local").glob("*.csv"))]
        )
    assert len(frames[0]) == 3
    for left, right in zip(frames[0], frames[1]):
        pd.testing.assert_frame_equal(left, right)


def test_local_explanations_can_be_disabled(fitted_xgboost, tmp_path: Path) -> None:
    model, x = fitted_xgboost
    paths = explain_model(
        model, x,
        experiment_id="off", dataset_id="demo",
        config=_config(**{
            "local_explanations": {"enabled": False},
            "global_importance": {"enabled": False},
        }),
        output_root=tmp_path,
    )
    assert "local_explanations" not in paths
    assert "global_feature_importance" not in paths
    assert not (tmp_path / "off" / "local").exists()
    # Phase 1.1 artefacts unaffected by the toggles.
    assert (tmp_path / "off" / "feature_importance.csv").is_file()


def test_empty_dataset_handled_gracefully(tmp_path: Path) -> None:
    """Zero explained samples: metadata + raw values persist, frames/plot skip."""
    from src.explainability.artifacts import write_explanation_artifacts
    from src.explainability.base import ExplanationResult

    result = ExplanationResult(
        values=np.zeros((0, 2, 3)),
        base_values=np.zeros(3),
        feature_names=["f1", "f2"],
        class_labels=["0", "1", "2"],
        n_samples=0,
    )
    paths = write_explanation_artifacts(
        result, pd.DataFrame(columns=["f1", "f2"]),
        experiment_id="empty", model_type="xgboost", dataset_id="demo",
        output_root=tmp_path,
    )
    assert set(paths) == {"metadata", "shap_values"}
    assert not (tmp_path / "empty" / "local").exists()
    assert not (tmp_path / "empty" / "global_feature_importance.csv").exists()


def test_binary_local_contribution_is_exact_signed_value(tmp_path: Path) -> None:
    """Single-output case: shap_contribution == signed value, abs == |signed|."""
    from src.explainability.artifacts import local_explanation_frame
    from src.explainability.base import ExplanationResult

    values = np.array([[[0.5], [-1.25]]])  # 1 sample, 2 features, 1 output
    result = ExplanationResult(
        values=values, base_values=np.zeros(1),
        feature_names=["f1", "f2"], class_labels=["output_0"], n_samples=1,
    )
    x = pd.DataFrame({"f1": [10.0], "f2": [20.0]})
    frame = local_explanation_frame(result, x, 0)
    by_feature = frame.set_index("feature")
    assert by_feature.loc["f1", "shap_contribution"] == 0.5
    assert by_feature.loc["f2", "shap_contribution"] == -1.25
    assert by_feature.loc["f2", "abs_contribution"] == 1.25
    assert by_feature.loc["f2", "contribution_rank"] == 1  # largest magnitude


# ------------------------------------------------------------- trainer hook


class _Result:
    experiment_id = "demo_xgboost_20260101T000000"
    dataset_id = "demo"
    model_name = "xgboost"


class _Paths:
    def __init__(self, root: Path) -> None:
        self.explainability_dir = root


def _splits(x: pd.DataFrame) -> dict:
    y = pd.Series(np.zeros(len(x)))
    return {"train": (x, y), "test": (x, y)}


def test_hook_disabled_is_noop(fitted_xgboost, tmp_path: Path) -> None:
    model, x = fitted_xgboost
    config = {"explainability": {"enabled": False}}
    out = maybe_explain_after_training(
        _Result(), model, _splits(x), config, _Paths(tmp_path)
    )
    assert out is None
    assert not any(tmp_path.iterdir())


def test_hook_skips_unsupported_model(fitted_xgboost, tmp_path: Path) -> None:
    _, x = fitted_xgboost
    result = _Result()
    result.model_name = "lightgbm"
    out = maybe_explain_after_training(
        result, object(), _splits(x), _config(), _Paths(tmp_path)
    )
    assert out is None


def test_hook_skips_missing_split(fitted_xgboost, tmp_path: Path) -> None:
    model, x = fitted_xgboost
    splits = {"train": _splits(x)["train"]}  # no test split
    out = maybe_explain_after_training(
        _Result(), model, splits, _config(), _Paths(tmp_path)
    )
    assert out is None


def test_hook_generates_artifacts_when_enabled(fitted_xgboost, tmp_path: Path) -> None:
    model, x = fitted_xgboost
    out = maybe_explain_after_training(
        _Result(), model, _splits(x), _config(max_samples=32), _Paths(tmp_path)
    )
    assert out is not None
    assert (tmp_path / _Result.experiment_id / "metadata.json").is_file()


def test_hook_never_raises_on_backend_failure(
    fitted_xgboost, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashing backend must be logged, not propagated (run already saved)."""
    model, x = fitted_xgboost
    monkeypatch.setattr(
        "src.explainability.runner.get_explainer",
        lambda name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = maybe_explain_after_training(
        _Result(), model, _splits(x), _config(), _Paths(tmp_path)
    )
    assert out is None

"""Tests for src.models.deep_learning (Engine B framework).

Networks are trained for 2-3 epochs on tiny synthetic matrices (CPU) so the
suite stays fast; no real training runs are performed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from src.models import registry
from src.models.base import BaseModel
from src.models.deep_learning.registry import DEEP_MODEL_REGISTRY

DEEP_MODELS = sorted(DEEP_MODEL_REGISTRY)

# Tiny architecture + training settings so every model fits in well under a
# second on CPU.
_TINY_TRAINING = {
    "batch_size": 32,
    "epochs": 2,
    "learning_rate": 0.01,
    "optimizer": "adam",
    "scheduler": "none",
    "mixed_precision": False,
    "progress_bar": False,
    "early_stopping": {"enabled": False},
}
_TINY_ARCH: dict[str, dict[str, Any]] = {
    "mlp": {"hidden_layers": [16], "dropout": 0.1, "batch_norm": True},
    "cnn": {"channels": [4, 8], "kernel_sizes": [3, 3]},
    "lstm": {"hidden_size": 8, "num_layers": 1, "bidirectional": True},
    "transformer": {"embedding_dim": 8, "num_heads": 2, "num_layers": 1},
}


def _params(name: str, **training_overrides: Any) -> dict[str, Any]:
    return {
        **_TINY_ARCH[name],
        "training": {**_TINY_TRAINING, **training_overrides},
    }


def _build(name: str, seed: int = 42, **training_overrides: Any) -> BaseModel:
    return registry.build_model(
        name,
        {"gpu": False, "params": _params(name, **training_overrides)},
        use_gpu=False,
        seed=seed,
    )


@pytest.fixture()
def xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 96
    y = rng.integers(0, 3, size=n)
    x = pd.DataFrame(
        {
            "f1": y + rng.normal(0, 0.1, n),
            "f2": -y + rng.normal(0, 0.1, n),
            "f3": rng.normal(size=n),
            "f4": rng.normal(size=n),
        }
    )
    return x, pd.Series(y, name="label")


# ------------------------------------------------------------------ registry


def test_registry_exposes_deep_models() -> None:
    assert set(registry.available_models()) >= set(DEEP_MODELS)
    for name in DEEP_MODELS:
        assert registry.get_model_cls(name).name == name


def test_classical_registry_untouched() -> None:
    assert set(registry.MODEL_REGISTRY) == {"xgboost", "lightgbm", "isolation_forest"}


# --------------------------------------------------------------- fit/predict


@pytest.mark.parametrize("name", DEEP_MODELS)
def test_fit_predict_proba_contract(name: str, xy) -> None:
    x, y = xy
    model = _build(name)
    model.fit(x, y, x, y)

    preds = model.predict(x)
    assert preds.shape == (len(x),)
    assert set(np.unique(preds)) <= set(np.unique(y))

    proba = model.predict_proba(x)
    assert proba.shape == (len(x), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    assert list(model.classes_) == [0, 1, 2]
    assert model.device == "cpu"
    assert model.is_supervised is True


def test_non_contiguous_labels_are_mapped_back(xy) -> None:
    x, y = xy
    remapped = y.map({0: 0, 1: 2, 2: 5})
    model = _build("mlp")
    model.fit(x, remapped)
    assert list(model.classes_) == [0, 2, 5]
    assert set(np.unique(model.predict(x))) <= {0, 2, 5}


def test_fit_without_labels_raises(xy) -> None:
    x, _ = xy
    with pytest.raises(ValueError, match="supervised"):
        _build("mlp").fit(x, None)


# ------------------------------------------------------- training mechanics


def test_early_stopping_halts_training(xy) -> None:
    x, y = xy
    # An impossible min_delta means no epoch after the first ever "improves".
    model = _build(
        "mlp",
        epochs=50,
        early_stopping={"enabled": True, "patience": 2, "min_delta": 1e9},
    )
    model.fit(x, y, x, y)
    assert model.fitted_params["epochs_trained"] == 3  # 1 best + 2 bad


def test_same_seed_is_deterministic(xy) -> None:
    x, y = xy
    proba_a = _build("mlp", seed=7).fit(x, y).predict_proba(x)
    proba_b = _build("mlp", seed=7).fit(x, y).predict_proba(x)
    np.testing.assert_array_equal(proba_a, proba_b)


def test_gradient_accumulation_and_clipping_run(xy) -> None:
    x, y = xy
    model = _build(
        "mlp", gradient_accumulation_steps=2, gradient_clipping=1.0
    )
    model.fit(x, y)
    assert model.fitted_params["epochs_trained"] == 2


def test_read_only_float32_input_does_not_warn(xy) -> None:
    """Arrow-backed parquet frames yield read-only float32 arrays; converting
    them to tensors must copy instead of emitting the non-writable warning."""
    import warnings

    x, y = xy
    frozen = x.to_numpy(dtype="float32")
    frozen.setflags(write=False)
    model = _build("mlp")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(frozen, y.to_numpy())
        model.predict_proba(frozen)
    assert model.fitted_params["epochs_trained"] == 2


def test_dataloader_worker_config_is_honoured(xy) -> None:
    x, y = xy
    model = _build("mlp")
    model.fit(x, y)  # sets device/seed
    loader = model._loader(
        x.to_numpy(dtype="float32"),
        y.to_numpy(),
        {**_TINY_TRAINING, "num_workers": 0, "prefetch_factor": 2,
         "persistent_workers": False, "batch_size": 16},
        shuffle=False,
    )
    assert loader.num_workers == 0
    assert loader.batch_size == 16


def test_cuda_performance_config_is_noop_on_cpu() -> None:
    from src.utils import hardware

    # Must never raise, with or without CUDA present.
    hardware.configure_cuda_performance(tf32=True, cudnn_benchmark=True)
    hardware.configure_cuda_performance(tf32=False, cudnn_benchmark=False)


def test_throughput_keys_have_defaults(xy) -> None:
    x, y = xy
    model = _build("mlp")
    model.fit(x, y)
    training = model.fitted_params["training"]
    assert {"tf32", "cudnn_benchmark", "num_workers", "prefetch_factor",
            "persistent_workers"} <= set(training)


def test_scheduler_variants_run(xy) -> None:
    x, y = xy
    for scheduler in ("plateau", "cosine", "step"):
        model = _build("mlp", scheduler=scheduler)
        model.fit(x, y, x, y)
        assert model.history  # trained without error


def test_unknown_optimizer_raises(xy) -> None:
    x, y = xy
    with pytest.raises(ValueError, match="optimizer"):
        _build("mlp", optimizer="nope").fit(x, y)


def test_unknown_scheduler_raises(xy) -> None:
    x, y = xy
    with pytest.raises(ValueError, match="scheduler"):
        _build("mlp", scheduler="nope").fit(x, y)


def test_checkpointing_writes_best_state(xy, tmp_path: Path) -> None:
    x, y = xy
    model = _build(
        "mlp", checkpointing={"enabled": True, "dir": str(tmp_path / "ckpt")}
    )
    model.fit(x, y, x, y)
    assert (tmp_path / "ckpt" / "mlp_best.pt").is_file()


def test_history_logs_epoch_metrics(xy) -> None:
    x, y = xy
    model = _build("mlp")
    model.fit(x, y, x, y)
    assert len(model.history) == 2
    entry = model.history[0]
    assert {"epoch", "train_loss", "val_loss", "lr", "seconds", "gpu_mb"} <= set(entry)
    assert model.describe()["epochs_trained"] == 2


# -------------------------------------------------- architecture validation


def test_cnn_rejects_mismatched_kernel_config(xy) -> None:
    x, y = xy
    model = registry.build_model(
        "cnn",
        {"gpu": False, "params": {"channels": [4, 8], "kernel_sizes": [3],
                                  "training": _TINY_TRAINING}},
        use_gpu=False, seed=42,
    )
    with pytest.raises(ValueError, match="kernel_sizes"):
        model.fit(x, y)


def test_transformer_rejects_indivisible_heads(xy) -> None:
    x, y = xy
    model = registry.build_model(
        "transformer",
        {"gpu": False, "params": {"embedding_dim": 9, "num_heads": 2,
                                  "training": _TINY_TRAINING}},
        use_gpu=False, seed=42,
    )
    with pytest.raises(ValueError, match="divisible"):
        model.fit(x, y)


def test_gpu_smoke_mixed_precision(xy) -> None:
    """AMP + pinned-memory path (runs only when a CUDA GPU is present)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x, y = xy
    model = registry.build_model(
        "mlp",
        {"gpu": True, "params": _params("mlp", mixed_precision=True)},
        use_gpu=True, seed=42,
    )
    model.fit(x, y, x, y)
    assert model.device == "cuda"
    assert model.predict(x).shape == (len(x),)
    assert model.history[0]["gpu_mb"] > 0


# ------------------------------------------------------------- persistence


def test_save_load_roundtrip(xy, tmp_path: Path) -> None:
    x, y = xy
    model = _build("mlp")
    model.fit(x, y)
    path = model.save(tmp_path / "model.joblib")

    loaded = BaseModel.load(path)
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))


# ------------------------------------------------------------- integration


def test_trainer_end_to_end_with_mlp(
    monkeypatch: pytest.MonkeyPatch, make_paths: Callable[[dict], Any], xy
) -> None:
    """The existing Engine A trainer must run a deep model unchanged."""
    from src.training import trainer
    from src.utils.io import read_json, write_parquet

    x, y = xy
    frame = x.copy()
    frame["label"] = y
    paths = make_paths({})
    feat_dir = Path(paths.features_out_dir) / "demo"
    for split in ("train", "validation", "test"):
        write_parquet(frame, feat_dir / f"{split}.parquet")
    monkeypatch.setattr(
        trainer, "load_dataset_config", lambda did: {"label_column": "label"}
    )

    config = {
        "project": {"seed": 42},
        "training": {
            "random_seed": 42, "use_gpu": False, "min_train_rows": 50,
            "evaluation": {"average": "weighted"},
        },
        "models": {"mlp": {"gpu": False, "params": _params("mlp")}},
    }
    result = trainer.train_model("demo", "mlp", config, paths)

    assert {"model", "metrics", "manifest"} <= set(result.output_paths)

    import csv
    index_path = Path(paths.experiments_dir) / "experiment_index.csv"
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == result.experiment_id
    assert rows[0]["best_epoch"] == "2"

    manifest = read_json(result.output_paths["manifest"])
    assert manifest["model"]["fitted_params"]["n_classes"] == 3
    assert manifest["model"]["fitted_params"]["epochs_trained"] == 2
    for split in ("train", "validation", "test"):
        assert result.metrics[split]["roc_auc"] is not None


def test_trainer_survives_locked_experiment_index(
    monkeypatch: pytest.MonkeyPatch, make_paths: Callable[[dict], Any], xy
) -> None:
    """A locked index CSV (e.g. open in Excel) must not fail a finished run."""
    from src.training import trainer
    from src.utils.io import write_parquet

    x, y = xy
    frame = x.copy()
    frame["label"] = y
    paths = make_paths({})
    feat_dir = Path(paths.features_out_dir) / "demo"
    for split in ("train", "validation", "test"):
        write_parquet(frame, feat_dir / f"{split}.parquet")
    monkeypatch.setattr(
        trainer, "load_dataset_config", lambda did: {"label_column": "label"}
    )

    def _locked(manifest: Any, root: Any) -> None:
        raise PermissionError("experiment_index.csv is locked")

    monkeypatch.setattr(trainer, "append_index_row", _locked)

    config = {
        "project": {"seed": 42},
        "training": {
            "random_seed": 42, "use_gpu": False, "min_train_rows": 50,
            "evaluation": {"average": "weighted"},
        },
        "models": {"mlp": {"gpu": False, "params": _params("mlp")}},
    }
    result = trainer.train_model("demo", "mlp", config, paths)
    assert Path(result.output_paths["manifest"]).is_file()


def test_deep_learning_config_is_wired_in() -> None:
    """configs/deep_learning.yaml must merge into the effective config with a
    training block on every deep model (validates the YAML anchor)."""
    from src.utils.config import load_config

    config = load_config()
    for name in DEEP_MODELS:
        block = config["models"][name]
        assert block["params"]["training"]["batch_size"] > 0
        assert "epochs" in block["params"]["training"]
        assert block["params"]["training"]["early_stopping"]["patience"] > 0

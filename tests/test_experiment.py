"""Tests for src.training.experiment."""

from __future__ import annotations

from pathlib import Path

from src.training import experiment


def test_create_experiment_makes_isolated_dir(tmp_path: Path) -> None:
    exp = experiment.create_experiment("nsl_kdd", "xgboost", tmp_path)
    assert exp.output_dir.is_dir()
    assert exp.dataset_id == "nsl_kdd"
    assert exp.model_name == "xgboost"
    assert "nsl_kdd_xgboost_" in exp.experiment_id
    # Nested under <root>/<dataset>/<model>/.
    assert exp.output_dir.parent.name == "xgboost"
    assert exp.output_dir.parent.parent.name == "nsl_kdd"


def test_experiments_never_overwrite(tmp_path: Path) -> None:
    a = experiment.create_experiment("d", "m", tmp_path)
    b = experiment.create_experiment("d", "m", tmp_path)
    # Same timestamp second -> suffixed, so directories differ.
    assert a.output_dir != b.output_dir


def test_build_manifest_has_required_sections(tmp_path: Path) -> None:
    exp = experiment.create_experiment("d", "m", tmp_path)
    manifest = experiment.build_manifest(
        experiment=exp,
        model_description={"name": "m"},
        config_snapshot={"training": {}},
        hardware={"device": "cpu"},
        metrics={"validation": {"f1": 1.0}},
        timings={"train_seconds": 0.1},
        artefacts={"model": "model.joblib"},
        seed=42,
    )
    for key in ("experiment_id", "seed", "model", "config_snapshot", "hardware",
                "metrics", "timings", "artefacts"):
        assert key in manifest
    assert manifest["seed"] == 42

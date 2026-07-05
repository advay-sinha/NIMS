"""Tests for src.registry (registration, rebuild, promotion, resolution)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.registry.artifacts import (
    load_production,
    load_registry,
    promote,
    rebuild_registry,
)
from src.registry.registry import (
    RegistryError,
    entry_from_manifest,
    metric_value,
    select_best_per_dataset,
)
from src.registry.reporting import registry_summary
from src.registry.resolver import resolve_model
from src.utils.io import write_json


def _write_experiment(
    root: Path,
    dataset: str,
    model: str,
    stamp: str,
    test_f1: float,
    *,
    provenance: dict | None = None,
    with_test: bool = True,
    with_model_file: bool = True,
) -> str:
    experiment_id = f"{dataset}_{model}_{stamp}"
    run_dir = root / dataset / model / experiment_id
    run_dir.mkdir(parents=True)
    if with_model_file:
        (run_dir / "model.joblib").write_bytes(b"stub")
    metrics: dict = {
        "validation": {"accuracy": test_f1, "f1": test_f1 - 0.001},
    }
    if with_test:
        metrics["test"] = {"accuracy": test_f1 + 0.001, "f1": test_f1,
                           "roc_auc": 0.99}
    manifest = {
        "experiment_id": experiment_id,
        "dataset_id": dataset,
        "model_name": model,
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": 42,
        "metrics": metrics,
        "artefacts": {"model": str(run_dir / "model.joblib")},
    }
    if provenance:
        manifest["provenance"] = provenance
    write_json(manifest, run_dir / "manifest.json")
    return experiment_id


@pytest.fixture()
def experiments(tmp_path: Path) -> Path:
    root = tmp_path / "experiments"
    _write_experiment(root, "ds_a", "xgboost", "20260101T000000", 0.92)
    _write_experiment(root, "ds_a", "xgboost", "20260102T000000", 0.95)
    _write_experiment(root, "ds_a", "mlp", "20260101T000000", 0.90)
    _write_experiment(
        root, "ds_b", "xgboost", "20260101T000000", 0.99,
        provenance={"source": "optimization", "study_id": "s1",
                    "best_trial_number": 3},
    )
    return root


def _rebuild(experiments: Path, tmp_path: Path, **policy) -> dict:
    return rebuild_registry(experiments, tmp_path / "registry", {}, **policy)


# ------------------------------------------------------------- registration


def test_register_valid_experiment(experiments: Path) -> None:
    manifest = (
        experiments / "ds_a" / "xgboost" / "ds_a_xgboost_20260102T000000"
        / "manifest.json"
    )
    entry = entry_from_manifest(manifest, {})
    assert entry["registry_id"] == "reg-ds_a_xgboost_20260102T000000"
    assert entry["dataset"] == "ds_a" and entry["model_type"] == "xgboost"
    assert entry["test_f1"] == 0.95
    assert entry["validation_f1"] == pytest.approx(0.949)
    assert entry["status"] == "candidate" and entry["source"] == "benchmark"
    assert Path(entry["model_artifact_path"]).is_file()


def test_reject_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="Manifest not found"):
        entry_from_manifest(tmp_path / "nope" / "manifest.json", {})


def test_reject_missing_model_artifact(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _write_experiment(root, "ds_a", "xgboost", "20260101T000000", 0.9,
                      with_model_file=False)
    manifest = next(root.glob("*/*/*/manifest.json"))
    with pytest.raises(RegistryError, match="model artefact not found"):
        entry_from_manifest(manifest, {})


def test_optimized_experiment_provenance_included(
    experiments: Path, tmp_path: Path
) -> None:
    document = _rebuild(experiments, tmp_path)
    optimized = next(e for e in document["entries"] if e["dataset"] == "ds_b")
    assert optimized["source"] == "optimization"
    assert optimized["optimization_study_id"] == "s1"


def test_optional_artifact_references(experiments: Path, tmp_path: Path) -> None:
    experiment_id = "ds_a_xgboost_20260102T000000"
    (tmp_path / "explain" / experiment_id).mkdir(parents=True)
    (tmp_path / "features" / "ds_a").mkdir(parents=True)
    manifest = (
        experiments / "ds_a" / "xgboost" / experiment_id / "manifest.json"
    )
    entry = entry_from_manifest(
        manifest,
        {"explainability": tmp_path / "explain",
         "features": tmp_path / "features",
         "error_analysis": tmp_path / "missing"},
    )
    assert entry["artifacts"]["explainability"].endswith(experiment_id)
    assert entry["artifacts"]["features"].endswith("ds_a")
    assert "error_analysis" not in entry["artifacts"]


# --------------------------------------------------------- rebuild/selection


def test_best_per_dataset_selection(experiments: Path, tmp_path: Path) -> None:
    _rebuild(experiments, tmp_path)
    best = json.loads(
        (tmp_path / "registry" / "best_per_dataset.json").read_text("utf-8")
    )
    assert best["ds_a"]["experiment_id"] == "ds_a_xgboost_20260102T000000"
    assert best["ds_a"]["value"] == 0.95
    assert best["ds_a"]["metric"] == "test_f1"
    assert best["ds_b"]["value"] == 0.99


def test_config_metric_selection(experiments: Path, tmp_path: Path) -> None:
    """Switching the selection metric changes the recorded policy/winner."""
    _rebuild(experiments, tmp_path, selection_metric="validation_accuracy")
    best = json.loads(
        (tmp_path / "registry" / "best_per_dataset.json").read_text("utf-8")
    )
    assert best["ds_a"]["metric"] == "validation_accuracy"
    assert best["ds_a"]["value"] == 0.95


def test_require_test_metrics_policy(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _write_experiment(root, "ds_a", "xgboost", "20260101T000000", 0.9,
                      with_test=False)
    document = _rebuild(root, tmp_path)
    assert document["entries"] == []
    permissive = rebuild_registry(
        root, tmp_path / "registry2", {}, require_test_metrics=False
    )
    assert len(permissive["entries"]) == 1


def test_registry_rebuild_idempotence(experiments: Path, tmp_path: Path) -> None:
    first = _rebuild(experiments, tmp_path)
    second = _rebuild(experiments, tmp_path)
    # registered_at, statuses, tags and content survive the rebuild.
    assert first["entries"] == second["entries"]


# ---------------------------------------------------------------- promotion


def test_production_promotion_and_status(experiments: Path, tmp_path: Path) -> None:
    _rebuild(experiments, tmp_path)
    registry_dir = tmp_path / "registry"
    assignment = promote(
        registry_dir, dataset="ds_a",
        experiment_id="ds_a_xgboost_20260102T000000", reason="best baseline",
    )
    assert assignment["model_type"] == "xgboost"
    production = load_production(registry_dir)
    assert production["ds_a"]["experiment_id"] == "ds_a_xgboost_20260102T000000"
    assert production["ds_a"]["reason"] == "best baseline"

    entries = {e["experiment_id"]: e
               for e in load_registry(registry_dir)["entries"]}
    assert entries["ds_a_xgboost_20260102T000000"]["status"] == "production"

    # Re-promoting another run demotes the previous production entry.
    promote(registry_dir, dataset="ds_a",
            experiment_id="ds_a_mlp_20260101T000000", reason="switch")
    entries = {e["experiment_id"]: e
               for e in load_registry(registry_dir)["entries"]}
    assert entries["ds_a_xgboost_20260102T000000"]["status"] == "candidate"
    assert entries["ds_a_mlp_20260101T000000"]["status"] == "production"

    # Rebuild preserves the production status (production.json authoritative).
    _rebuild(experiments, tmp_path)
    entries = {e["experiment_id"]: e
               for e in load_registry(registry_dir)["entries"]}
    assert entries["ds_a_mlp_20260101T000000"]["status"] == "production"


def test_promotion_rejects_unregistered_experiment(
    experiments: Path, tmp_path: Path
) -> None:
    _rebuild(experiments, tmp_path)
    with pytest.raises(RegistryError, match="not registered"):
        promote(tmp_path / "registry", dataset="ds_a",
                experiment_id="ghost", reason="nope")


# ------------------------------------------------------------------ resolver


def test_resolver_production_lookup(experiments: Path, tmp_path: Path) -> None:
    _rebuild(experiments, tmp_path)
    registry_dir = tmp_path / "registry"
    promote(registry_dir, dataset="ds_a",
            experiment_id="ds_a_xgboost_20260102T000000", reason="best")
    resolved = resolve_model("ds_a", "production", registry_dir=registry_dir)
    assert resolved["experiment_id"] == "ds_a_xgboost_20260102T000000"
    assert resolved["model_type"] == "xgboost"
    assert Path(resolved["model_artifact_path"]).is_file()
    assert Path(resolved["manifest_path"]).is_file()
    assert resolved["status"] == "production"
    assert resolved["metrics"]["test"]["f1"] == 0.95


def test_resolver_best_stage(experiments: Path, tmp_path: Path) -> None:
    _rebuild(experiments, tmp_path)
    resolved = resolve_model("ds_b", "best", registry_dir=tmp_path / "registry")
    assert resolved["experiment_id"] == "ds_b_xgboost_20260101T000000"
    assert resolved["stage"] == "best"


def test_resolver_missing_production_assignment(
    experiments: Path, tmp_path: Path
) -> None:
    _rebuild(experiments, tmp_path)
    with pytest.raises(RegistryError, match="No production model"):
        resolve_model("ds_a", "production", registry_dir=tmp_path / "registry")
    with pytest.raises(RegistryError, match="Unknown stage"):
        resolve_model("ds_a", "nope", registry_dir=tmp_path / "registry")


# ----------------------------------------------------------- archive status


def test_archived_entries_excluded_from_best(experiments: Path,
                                             tmp_path: Path) -> None:
    document = _rebuild(experiments, tmp_path)
    registry_dir = tmp_path / "registry"
    for entry in document["entries"]:
        if entry["experiment_id"] == "ds_a_xgboost_20260102T000000":
            entry["status"] = "archived"
    write_json(document, registry_dir / "registry.json")
    # Rebuild preserves the archived status and skips it for best selection.
    second = _rebuild(experiments, tmp_path)
    entries = {e["experiment_id"]: e for e in second["entries"]}
    assert entries["ds_a_xgboost_20260102T000000"]["status"] == "archived"
    best = json.loads(
        (registry_dir / "best_per_dataset.json").read_text("utf-8")
    )
    assert best["ds_a"]["experiment_id"] == "ds_a_xgboost_20260101T000000"


# ------------------------------------------------------- reporting + script


def test_registry_summary_reports_gaps(experiments: Path, tmp_path: Path) -> None:
    _rebuild(experiments, tmp_path)
    registry_dir = tmp_path / "registry"
    promote(registry_dir, dataset="ds_a",
            experiment_id="ds_a_xgboost_20260102T000000", reason="best")
    best = json.loads((registry_dir / "best_per_dataset.json").read_text("utf-8"))
    summary = registry_summary(
        load_registry(registry_dir), best, load_production(registry_dir),
        registry_dir,
    )
    assert "## Model Registry" in summary
    assert "ds_a_xgboost_20260102T000000" in summary
    assert "Missing production assignments: ds_b" in summary


def test_metric_value_parsing() -> None:
    entry = {"metrics": {"test": {"f1": 0.9}, "validation": {"accuracy": 0.8}}}
    assert metric_value(entry, "test_f1") == 0.9
    assert metric_value(entry, "validation_accuracy") == 0.8
    assert metric_value(entry, "test_missing") is None


def test_select_best_deterministic_on_ties() -> None:
    entries = [
        {"dataset": "d", "experiment_id": "b", "model_type": "m",
         "status": "candidate", "metrics": {"test": {"f1": 0.9}}},
        {"dataset": "d", "experiment_id": "a", "model_type": "m",
         "status": "candidate", "metrics": {"test": {"f1": 0.9}}},
    ]
    best = select_best_per_dataset(entries)
    assert best["d"]["experiment_id"] == "a"  # lexicographic tie-break


# --------------------------------------------------- resolve_model CLI


class _FakeCtx:
    def __init__(self, registry_dir: Path) -> None:
        class _P:
            pass

        self.paths = _P()
        self.paths.registry_dir = registry_dir


@pytest.fixture()
def promoted_registry(experiments: Path, tmp_path: Path, monkeypatch) -> Path:
    """A built registry with ds_a promoted, wired into the CLI's bootstrap."""
    import scripts.resolve_model as cli

    _rebuild(experiments, tmp_path)
    registry_dir = tmp_path / "registry"
    promote(registry_dir, dataset="ds_a",
            experiment_id="ds_a_xgboost_20260102T000000", reason="best")
    monkeypatch.setattr(cli, "bootstrap", lambda args: _FakeCtx(registry_dir))
    return registry_dir


def test_resolve_cli_successful_resolution(promoted_registry, caplog) -> None:
    from scripts.resolve_model import main

    with caplog.at_level("INFO"):
        assert main(["--dataset", "ds_a"]) == 0
    assert "ds_a_xgboost_20260102T000000" in caplog.text
    assert "xgboost" in caplog.text


def test_resolve_cli_missing_production_assignment(promoted_registry,
                                                   caplog) -> None:
    from scripts.resolve_model import main

    with caplog.at_level("ERROR"):
        assert main(["--dataset", "ds_b"]) == 1
    assert "No production model" in caplog.text


def test_resolve_cli_json_output(promoted_registry, capsys) -> None:
    from scripts.resolve_model import main

    assert main(["--dataset", "ds_a", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == "ds_a_xgboost_20260102T000000"
    assert payload["model_type"] == "xgboost"
    assert payload["metrics"]["test"]["f1"] == 0.95
    assert payload["status"] == "production"


def test_resolve_cli_model_type_mismatch(promoted_registry, caplog) -> None:
    from scripts.resolve_model import main

    with caplog.at_level("ERROR"):
        assert main(["--dataset", "ds_a", "--model", "mlp"]) == 1
    assert "not the" in caplog.text


def test_resolve_cli_argument_parsing() -> None:
    from scripts.resolve_model import build_parser

    args = build_parser().parse_args(
        ["--dataset", "unsw_nb15", "--stage", "best", "--model", "xgboost",
         "--json"]
    )
    assert args.dataset == "unsw_nb15" and args.stage == "best"
    assert args.model == "xgboost" and args.as_json is True
    defaults = build_parser().parse_args(["--dataset", "d"])
    assert defaults.stage == "production" and defaults.as_json is False
    with pytest.raises(SystemExit):  # --dataset is required
        build_parser().parse_args([])


def test_promote_script_argument_parsing() -> None:
    from scripts.promote_model import build_parser

    args = build_parser().parse_args(
        ["--dataset", "unsw_nb15", "--model", "xgboost",
         "--experiment-id", "run1", "--reason", "why"]
    )
    assert args.dataset == "unsw_nb15" and args.experiment_id == "run1"
    with pytest.raises(SystemExit):  # --reason is required
        build_parser().parse_args(["--dataset", "d", "--model", "m"])

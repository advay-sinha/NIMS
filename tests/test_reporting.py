"""Tests for src.training.reporting (validation report generation)."""

from __future__ import annotations

from pathlib import Path

from src.training.reporting import (
    build_validation_report,
    collect_latest_manifests,
    count_manifests,
)
from src.utils.io import write_json


def _manifest(dataset: str, model: str, stamp: str, created: str, f1: float) -> dict:
    return {
        "experiment_id": f"{dataset}_{model}_{stamp}",
        "dataset_id": dataset,
        "model_name": model,
        "created_at": created,
        "model": {"device": "cpu"},
        "timings": {
            "train_seconds": 1.5,
            "predict_test_seconds": 0.1,
            "model_size_bytes": 2048,
        },
        "metrics": {
            "validation": {
                "accuracy": f1, "precision": f1, "recall": f1, "f1": f1,
                "roc_auc": None,
                "false_positive_rate": {"macro": 0.01},
            },
        },
    }


def _write_run(root: Path, dataset: str, model: str, created: str, f1: float) -> None:
    # Run ids use the filesystem-safe compact stamp (as the trainer does).
    stamp = created.replace("-", "").replace(":", "")[:15]
    run_dir = root / dataset / model / f"{dataset}_{model}_{stamp}"
    run_dir.mkdir(parents=True)
    write_json(_manifest(dataset, model, stamp, created, f1), run_dir / "manifest.json")


def test_collect_latest_manifests_picks_newest_run(tmp_path: Path) -> None:
    _write_run(tmp_path, "demo", "xgboost", "2026-01-01T00:00:00+00:00", 0.5)
    _write_run(tmp_path, "demo", "xgboost", "2026-02-01T00:00:00+00:00", 0.9)
    latest = collect_latest_manifests(tmp_path)
    assert latest["demo"]["xgboost"]["metrics"]["validation"]["f1"] == 0.9


def test_collect_latest_manifests_empty_dir(tmp_path: Path) -> None:
    assert collect_latest_manifests(tmp_path) == {}


def test_build_validation_report_renders_all_sections(tmp_path: Path) -> None:
    _write_run(tmp_path, "demo", "xgboost", "2026-01-01T00:00:00+00:00", 0.5)
    _write_run(tmp_path, "demo", "lightgbm", "2026-01-01T00:00:00+00:00", 0.4)
    report = build_validation_report(
        collect_latest_manifests(tmp_path), analysis="## Bottlenecks\n\nnone"
    )
    assert "# Model Validation Report" in report
    assert "## Dataset: demo" in report
    assert "### xgboost" in report and "### lightgbm" in report
    assert "| Model | Val F1 |" in report  # comparison table header
    assert "## Bottlenecks" in report  # analysis appended
    assert "—" in report  # None metric rendered as em dash


# ------------------------------------------------------- benchmark sections


def _bench_manifest(
    dataset: str,
    model: str,
    f1: float,
    *,
    train_seconds: float = 10.0,
    size_bytes: int = 1_048_576,
    roc_auc: float | None = 0.99,
) -> dict:
    return {
        "experiment_id": f"{dataset}_{model}_20260101T000000",
        "dataset_id": dataset,
        "model_name": model,
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": 42,
        "model": {"device": "cuda"},
        "hardware": {"gpu_name": "RTX 4060", "device": "cuda"},
        "timings": {
            "train_seconds": train_seconds,
            "predict_test_seconds": 0.5,
            "model_size_bytes": size_bytes,
        },
        "metrics": {
            "test": {
                "accuracy": f1, "precision": f1, "recall": f1, "f1": f1,
                "roc_auc": roc_auc,
                "false_positive_rate": {"macro": 0.01},
            },
        },
    }


def _bench_latest() -> dict[str, dict[str, dict]]:
    """Two datasets, one classical and one deep model each."""
    return {
        "ds_a": {
            "xgboost": _bench_manifest("ds_a", "xgboost", 0.99, train_seconds=5),
            "mlp": _bench_manifest(
                "ds_a", "mlp", 0.95, train_seconds=100, size_bytes=262_144
            ),
        },
        "ds_b": {
            "xgboost": _bench_manifest("ds_b", "xgboost", 0.97, train_seconds=50),
            "mlp": _bench_manifest(
                "ds_b", "mlp", 0.91, train_seconds=200, size_bytes=262_144
            ),
        },
    }


def test_executive_summary_counts_and_hardware() -> None:
    report = build_validation_report(_bench_latest(), total_experiments=17)
    assert "## 1. Executive Summary" in report
    assert "- Datasets benchmarked: 2 (ds_a, ds_b)" in report
    assert "- Models benchmarked: 2 (mlp, xgboost)" in report
    assert "- Completed experiments: 17" in report
    assert "- Hardware: RTX 4060" in report


def test_best_per_dataset_sorted_by_f1() -> None:
    report = build_validation_report(_bench_latest())
    section = report.split("## 2.")[1].split("## 3.")[0]
    rows = [line for line in section.splitlines() if line.startswith("| ds_")]
    # ds_a's best F1 (0.99) outranks ds_b's (0.97); xgboost wins both.
    assert rows[0].startswith("| ds_a | xgboost | 0.9900")
    assert rows[1].startswith("| ds_b | xgboost | 0.9700")


def test_ranking_bolds_winner_per_dataset() -> None:
    report = build_validation_report(_bench_latest())
    section = report.split("## 3.")[1].split("## 4.")[0]
    assert "| 1 | **xgboost** |" in section
    assert "| 2 | mlp |" in section
    assert "**mlp**" not in section


def test_overall_ranking_averages_across_datasets() -> None:
    report = build_validation_report(_bench_latest())
    section = report.split("## 4.")[1].split("## 5.")[0]
    # xgboost: mean F1 (0.99+0.97)/2, mean train (5+50)/2, 2 datasets.
    assert "| 1 | xgboost | 2 | 0.9800 | 0.9800 | 0.9900 | 27.50 |" in section
    assert "| 2 | mlp | 2 |" in section


def test_family_comparison_classifies_models() -> None:
    report = build_validation_report(_bench_latest())
    section = report.split("## 5.")[1].split("## 6.")[0]
    assert "| Classical | 1 | 0.9800 |" in section
    assert "| Deep | 1 | 0.9300 |" in section
    assert "Unclassified" not in section


def test_efficiency_leaders() -> None:
    report = build_validation_report(_bench_latest())
    section = report.split("## 6.")[1].split("## 7.")[0]
    assert "| Fastest training | xgboost | ds_a | 5.00 s |" in section
    assert "| Smallest model | mlp | ds_a | 256.0 KB |" in section
    # 0.99 F1 / (5 s / 60) = 11.88 F1/min.
    assert "| Best F1 per training minute | xgboost | ds_a | 11.88 F1/min |" in section
    # mlp: 0.95 F1 / 0.25 MB = 3.80 F1/MB beats xgboost's 0.99 F1/MB.
    assert "| Best F1 per model MB | mlp | ds_a | 3.80 F1/MB |" in section


def test_key_findings_are_derived_from_data() -> None:
    report = build_validation_report(_bench_latest())
    section = report.split("## 7.")[1].split("## 8.")[0]
    assert "xgboost achieves the highest F1 on all 2 datasets." in section
    assert "Classical models lead every dataset benchmarked." in section
    assert "Strongest deep model by average F1: mlp (0.9300" in section


def test_reproducibility_lists_seed() -> None:
    report = build_validation_report(_bench_latest())
    section = report.split("## 8.")[1]
    assert "- Deterministic seed: 42" in section
    assert "experiment_index.csv" in section


def test_legacy_detail_sections_preserved() -> None:
    """Backwards compatibility: the pre-existing per-dataset tables remain."""
    report = build_validation_report(_bench_latest())
    assert "## Detailed Results" in report
    assert "## Dataset: ds_a" in report
    assert "| Model | Val F1 |" in report
    assert report.index("## 8.") < report.index("## Detailed Results")


def test_count_manifests_counts_all_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "demo", "xgboost", "2026-01-01T00:00:00+00:00", 0.5)
    _write_run(tmp_path, "demo", "xgboost", "2026-02-01T00:00:00+00:00", 0.9)
    _write_run(tmp_path, "demo", "mlp", "2026-01-01T00:00:00+00:00", 0.7)
    assert count_manifests(tmp_path) == 3
    assert count_manifests(tmp_path / "missing") == 0

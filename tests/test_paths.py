"""Tests for src.utils.paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.utils import paths
from src.utils.paths import Paths


def test_ensure_dir_creates_nested_directory(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    created = paths.ensure_dir(target)
    assert created.is_dir()


def test_raw_dir_unknown_dataset_raises_key_error() -> None:
    p = Paths(
        root=Path("."),
        data_dir=Path("data"),
        datasets_dir=Path("datasets"),
        models_dir=Path("models"),
        outputs_dir=Path("outputs"),
        logs_dir=Path("outputs/logs"),
        interim_dir=Path("data/interim"),
        processed_dir=Path("data/processed"),
        features_dir=Path("data/features"),
        metadata_dir=Path("data/metadata"),
        reports_dir=Path("outputs/reports"),
        data_reports_dir=Path("outputs/data_reports"),
        fingerprints_dir=Path("outputs/metadata"),
        figures_dir=Path("outputs/figures"),
        preprocessing_dir=Path("outputs/preprocessing"),
        processed_out_dir=Path("outputs/processed"),
        artifacts_dir=Path("outputs/artifacts"),
        features_out_dir=Path("outputs/features"),
        experiments_dir=Path("outputs/experiments"),
        explainability_dir=Path("outputs/explainability"),
        error_analysis_dir=Path("outputs/error_analysis"),
        visualizations_dir=Path("outputs/visualizations"),
        optimization_dir=Path("outputs/optimization"),
        registry_dir=Path("outputs/registry"),
        raw={"nsl_kdd": Path("datasets/NSL-KDD")},
    )
    assert p.raw_dir("nsl_kdd") == Path("datasets/NSL-KDD")
    with pytest.raises(KeyError):
        p.raw_dir("does_not_exist")


def test_from_config_resolves_absolute_paths(sample_config: dict[str, Any]) -> None:
    p = Paths.from_config(sample_config)
    assert p.processed_dir.is_absolute()
    assert p.processed_dir == paths.PROJECT_ROOT / "data" / "processed"
    assert p.raw_dir("nsl_kdd") == paths.PROJECT_ROOT / "datasets" / "NSL-KDD"


def test_from_config_missing_paths_section_raises() -> None:
    with pytest.raises(KeyError):
        Paths.from_config({})

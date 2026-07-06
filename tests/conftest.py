"""Shared pytest fixtures.

Provides lightweight, dependency-free fixtures so unit tests can run without
the large raw datasets on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

# DLL load-order guard (see scripts/_bootstrap.py): pyarrow must load its
# native libraries before torch on this platform or parquet reads segfault.
try:  # pragma: no cover
    import pyarrow.dataset  # noqa: F401
except ImportError:
    pass

import pytest

from src.utils.paths import Paths


@pytest.fixture()
def make_paths(tmp_path: Path) -> Callable[[Mapping[str, Path]], Paths]:
    """Factory building a :class:`Paths` rooted under a temp dir.

    The returned callable takes a ``{dataset_id: raw_dir}`` mapping so loader
    tests can point a dataset at fixture files without touching real data.
    """

    def _factory(raw: Mapping[str, Path]) -> Paths:
        return Paths(
            root=tmp_path,
            data_dir=tmp_path / "data",
            datasets_dir=tmp_path / "datasets",
            models_dir=tmp_path / "models",
            outputs_dir=tmp_path / "outputs",
            logs_dir=tmp_path / "outputs/logs",
            interim_dir=tmp_path / "data/interim",
            processed_dir=tmp_path / "data/processed",
            features_dir=tmp_path / "data/features",
            metadata_dir=tmp_path / "data/metadata",
            reports_dir=tmp_path / "outputs/reports",
            data_reports_dir=tmp_path / "outputs/data_reports",
            fingerprints_dir=tmp_path / "outputs/metadata",
            figures_dir=tmp_path / "outputs/figures",
            preprocessing_dir=tmp_path / "outputs/preprocessing",
            processed_out_dir=tmp_path / "outputs/processed",
            artifacts_dir=tmp_path / "outputs/artifacts",
            features_out_dir=tmp_path / "outputs/features",
            experiments_dir=tmp_path / "outputs/experiments",
            explainability_dir=tmp_path / "outputs/explainability",
            error_analysis_dir=tmp_path / "outputs/error_analysis",
            visualizations_dir=tmp_path / "outputs/visualizations",
            optimization_dir=tmp_path / "outputs/optimization",
            registry_dir=tmp_path / "outputs/registry",
            network_health_dir=tmp_path / "outputs/network_health",
            network_config_dir=tmp_path / "outputs/network_config",
            raw=dict(raw),
        )

    return _factory


@pytest.fixture()
def nsl_kdd_config() -> dict[str, Any]:
    """A minimal NSL-KDD dataset config with a tiny 4-column schema."""
    return {
        "id": "nsl_kdd",
        "name": "NSL-KDD",
        "engine": "A",
        "raw_dir_key": "nsl_kdd",
        "files": {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"},
        "columns": ["duration", "protocol_type", "label", "difficulty"],
        "categorical_columns": ["protocol_type"],
        "label_column": "label",
        "drop_columns": ["difficulty"],
    }


@pytest.fixture()
def sample_config() -> dict[str, Any]:
    """A minimal but structurally valid effective configuration."""
    return {
        "project": {"name": "netsentinel", "version": "1.0", "seed": 42},
        "paths": {
            "data_dir": "data",
            "datasets_dir": "datasets",
            "models_dir": "models",
            "outputs_dir": "outputs",
            "logs_dir": "outputs/logs",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
            "features_dir": "data/features",
            "metadata_dir": "data/metadata",
            "reports_dir": "outputs/reports",
            "data_reports_dir": "outputs/data_reports",
            "fingerprints_dir": "outputs/metadata",
            "figures_dir": "outputs/figures",
            "preprocessing_dir": "outputs/preprocessing",
            "processed_out_dir": "outputs/processed",
            "artifacts_dir": "outputs/artifacts",
            "features_out_dir": "outputs/features",
            "experiments_dir": "outputs/experiments",
            "explainability_dir": "outputs/explainability",
            "error_analysis_dir": "outputs/error_analysis",
            "visualizations_dir": "outputs/visualizations",
            "optimization_dir": "outputs/optimization",
            "registry_dir": "outputs/registry",
            "network_health_dir": "outputs/network_health",
            "network_config_dir": "outputs/network_config",
            "raw": {"nsl_kdd": "datasets/NSL-KDD"},
        },
        "data": {
            "active_datasets": ["nsl_kdd"],
            "split": {
                "train_size": 0.70,
                "val_size": 0.15,
                "test_size": 0.15,
                "stratify": True,
                "shuffle": True,
            },
        },
    }


@pytest.fixture()
def tmp_artifact_path(tmp_path: Path) -> Path:
    """A nested, not-yet-existing path to exercise parent-dir creation."""
    return tmp_path / "nested" / "artifact.json"

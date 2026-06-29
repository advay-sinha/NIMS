"""Tests for src.utils.config."""

from __future__ import annotations

from typing import Any

import pytest

from src.utils import config


def test_get_resolves_nested_dotted_key(sample_config: dict[str, Any]) -> None:
    assert config.get(sample_config, "data.split.train_size") == 0.70


def test_get_returns_default_for_missing_key(sample_config: dict[str, Any]) -> None:
    assert config.get(sample_config, "data.split.missing", default=-1) == -1


def test_get_returns_default_when_traversing_non_mapping(
    sample_config: dict[str, Any],
) -> None:
    # "name" is a str; descending further must fall back to default.
    assert config.get(sample_config, "project.name.nope", default="x") == "x"


def test_load_yaml_reads_mapping(tmp_path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    assert config.load_yaml(path) == {"a": 1, "b": {"c": 2}}


def test_load_yaml_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        config.load_yaml("does/not/exist.yaml")


def test_load_yaml_empty_file_returns_empty_dict(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert config.load_yaml(path) == {}


def test_deep_merge_recurses_and_overrides() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 10}
    override = {"a": {"y": 20, "z": 3}, "c": 30}
    assert config.deep_merge(base, override) == {
        "a": {"x": 1, "y": 20, "z": 3},
        "b": 10,
        "c": 30,
    }


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    config.deep_merge(base, {"a": {"x": 9}})
    assert base == {"a": {"x": 1}}


def test_load_config_composes_defaults_and_root() -> None:
    # The real repo config: defaults (paths/logging/data) merged under root.
    cfg = config.load_config()
    assert cfg["project"]["name"] == "netsentinel"
    assert "paths" in cfg and "data" in cfg and "logging" in cfg


def test_load_dataset_config_returns_dataset_block() -> None:
    block = config.load_dataset_config("nsl_kdd")
    assert block["id"] == "nsl_kdd"
    assert block["label_column"] == "label"


def test_load_dataset_config_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        config.load_dataset_config("nope")

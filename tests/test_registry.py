"""Tests for src.data.registry."""

from __future__ import annotations

import pytest

from src.data import registry
from src.data.base import BaseDatasetLoader


def test_available_datasets_lists_all_four() -> None:
    assert set(registry.available_datasets()) == {
        "nsl_kdd",
        "unsw_nb15",
        "cicids2017",
        "snmp",
    }


@pytest.mark.parametrize("dataset_id", ["nsl_kdd", "unsw_nb15", "cicids2017", "snmp"])
def test_get_loader_cls_returns_loader_subclass(dataset_id: str) -> None:
    cls = registry.get_loader_cls(dataset_id)
    assert issubclass(cls, BaseDatasetLoader)


def test_get_loader_cls_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        registry.get_loader_cls("nope")

"""Tests for src.data.fingerprint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.data import fingerprint as fp


def test_resolve_data_files_dict() -> None:
    config = {"files": {"train": "a.csv", "test": "b.csv", "feature_dictionary": "f.csv"}}
    # Only train/test are data files; the dictionary is excluded.
    assert fp.resolve_data_files(config) == ["a.csv", "b.csv"]


def test_resolve_data_files_list() -> None:
    config = {"files": ["d1.csv", "d2.csv"]}
    assert fp.resolve_data_files(config) == ["d1.csv", "d2.csv"]


def test_resolve_data_files_snmp_data_key() -> None:
    config = {"files": {"data": "all.csv", "reference": "ref.pdf"}}
    assert fp.resolve_data_files(config) == ["all.csv"]


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    content = b"a,b,c\n1,2,3\n"
    path.write_bytes(content)
    assert fp.sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fp.sha256_file(tmp_path / "nope.csv")


def test_combined_checksum_order_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_bytes(b"AAAA")
    b.write_bytes(b"BBBB")
    assert fp.combined_checksum([a, b]) == fp.combined_checksum([b, a])


def test_combined_checksum_empty() -> None:
    assert fp.combined_checksum([]) == ""


def test_build_fingerprint_fields(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.csv").write_bytes(b"x,y\n1,2\n")
    (raw_dir / "test.csv").write_bytes(b"x,y\n3,4\n")
    config = {
        "id": "demo",
        "name": "Demo",
        "schema_version": "2.1",
        "files": {"train": "train.csv", "test": "test.csv"},
    }

    out = fp.build_fingerprint(config, raw_dir, n_rows=2, n_features=2)

    assert out["dataset_name"] == "Demo"
    assert out["row_count"] == 2
    assert out["column_count"] == 2
    assert out["schema_version"] == "2.1"
    assert len(out["sha256"]) == 64
    assert out["source_path"] == str(raw_dir)
    assert set(out["source_files"]) == {"train.csv", "test.csv"}
    assert "generated_at" in out

"""Tests for the dataset loaders (src.data.loaders).

Covers, per the Phase 1 requirements: a valid dataset, a missing file, an
invalid schema, an empty dataset and malformed rows. Fixture CSVs are tiny and
created under ``tmp_path`` so no real raw data is touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from src.data.base import RawDataset
from src.data.loaders.cicids2017 import CICIDS2017Loader
from src.data.loaders.nsl_kdd import NSLKDDLoader
from src.data.loaders.unsw_nb15 import UNSWNB15Loader
from src.utils.paths import Paths

PathsFactory = Callable[[dict[str, Path]], Paths]


# --------------------------------------------------------------------------- #
# NSL-KDD                                                                      #
# --------------------------------------------------------------------------- #
def _write_nsl(raw_dir: Path, train_rows: str, test_rows: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "KDDTrain+.txt").write_text(train_rows, encoding="utf-8")
    (raw_dir / "KDDTest+.txt").write_text(test_rows, encoding="utf-8")


def test_nsl_kdd_loads_valid_dataset(
    tmp_path: Path,
    make_paths: PathsFactory,
    nsl_kdd_config: dict[str, Any],
) -> None:
    raw_dir = tmp_path / "nsl"
    _write_nsl(
        raw_dir,
        train_rows="0,tcp,normal,20\n1,udp,neptune,18\n",
        test_rows="2,tcp,normal,15\n",
    )
    loader = NSLKDDLoader(nsl_kdd_config, make_paths({"nsl_kdd": raw_dir}))

    raw = loader.load_raw()

    assert isinstance(raw, RawDataset)
    assert raw.label_column == "label"
    assert raw.categorical_columns == ("protocol_type",)
    # difficulty dropped; split provenance added; 3 rows total.
    assert "difficulty" not in raw.frame.columns
    assert "split" in raw.frame.columns
    assert len(raw.frame) == 3
    assert set(raw.frame["label"]) == {"normal", "neptune"}


def test_nsl_kdd_missing_file_raises(
    tmp_path: Path,
    make_paths: PathsFactory,
    nsl_kdd_config: dict[str, Any],
) -> None:
    raw_dir = tmp_path / "nsl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "KDDTrain+.txt").write_text("0,tcp,normal,20\n", encoding="utf-8")
    # KDDTest+.txt intentionally absent.
    loader = NSLKDDLoader(nsl_kdd_config, make_paths({"nsl_kdd": raw_dir}))

    with pytest.raises(FileNotFoundError):
        loader.load_raw()


def test_nsl_kdd_malformed_rows_raise(
    tmp_path: Path,
    make_paths: PathsFactory,
    nsl_kdd_config: dict[str, Any],
) -> None:
    raw_dir = tmp_path / "nsl"
    # Second row has 5 fields for a 4-column schema -> parser error.
    _write_nsl(
        raw_dir,
        train_rows="0,tcp,normal,20\n1,udp,neptune,18,EXTRA\n",
        test_rows="2,tcp,normal,15\n",
    )
    loader = NSLKDDLoader(nsl_kdd_config, make_paths({"nsl_kdd": raw_dir}))

    with pytest.raises(ValueError):
        loader.load_raw()


# --------------------------------------------------------------------------- #
# UNSW-NB15                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def unsw_config() -> dict[str, Any]:
    return {
        "id": "unsw_nb15",
        "name": "UNSW-NB15",
        "engine": "A",
        "raw_dir_key": "unsw_nb15",
        "files": {
            "train": "UNSW_NB15_training-set.csv",
            "test": "UNSW_NB15_testing-set.csv",
        },
        "categorical_columns": ["proto", "service", "state"],
        "label_column": "label",
        "drop_columns": ["id"],
    }


def test_unsw_loads_and_drops_id(
    tmp_path: Path, make_paths: PathsFactory, unsw_config: dict[str, Any]
) -> None:
    raw_dir = tmp_path / "unsw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    header = "id,proto,service,state,label\n"
    (raw_dir / "UNSW_NB15_training-set.csv").write_text(
        header + "1,tcp,http,FIN,0\n2,udp,dns,CON,1\n", encoding="utf-8"
    )
    (raw_dir / "UNSW_NB15_testing-set.csv").write_text(
        header + "3,tcp,ftp,FIN,1\n", encoding="utf-8"
    )
    loader = UNSWNB15Loader(unsw_config, make_paths({"unsw_nb15": raw_dir}))

    raw = loader.load_raw()

    assert "id" not in raw.frame.columns
    assert "split" in raw.frame.columns
    assert len(raw.frame) == 3
    assert raw.label_column == "label"


def test_unsw_empty_dataset_raises(
    tmp_path: Path, make_paths: PathsFactory, unsw_config: dict[str, Any]
) -> None:
    raw_dir = tmp_path / "unsw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    header = "id,proto,service,state,label\n"
    # Header only -> zero data rows.
    (raw_dir / "UNSW_NB15_training-set.csv").write_text(header, encoding="utf-8")
    (raw_dir / "UNSW_NB15_testing-set.csv").write_text(header, encoding="utf-8")
    loader = UNSWNB15Loader(unsw_config, make_paths({"unsw_nb15": raw_dir}))

    with pytest.raises(ValueError):
        loader.load_raw()


def test_unsw_invalid_schema_missing_label_raises(
    tmp_path: Path, make_paths: PathsFactory, unsw_config: dict[str, Any]
) -> None:
    raw_dir = tmp_path / "unsw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # No `label` column -> require_label_column should fail.
    header = "id,proto,service,state\n"
    (raw_dir / "UNSW_NB15_training-set.csv").write_text(
        header + "1,tcp,http,FIN\n", encoding="utf-8"
    )
    (raw_dir / "UNSW_NB15_testing-set.csv").write_text(
        header + "2,udp,dns,CON\n", encoding="utf-8"
    )
    loader = UNSWNB15Loader(unsw_config, make_paths({"unsw_nb15": raw_dir}))

    with pytest.raises(ValueError):
        loader.load_raw()


# --------------------------------------------------------------------------- #
# CICIDS2017                                                                   #
# --------------------------------------------------------------------------- #
def test_cicids_strips_column_whitespace(
    tmp_path: Path, make_paths: PathsFactory
) -> None:
    raw_dir = tmp_path / "cic"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Note leading spaces in headers, as in the real CICIDS captures.
    (raw_dir / "day1.csv").write_text(
        " Flow Duration, Total Fwd Packets, Label\n100,5,BENIGN\n200,9,DDoS\n",
        encoding="utf-8",
    )
    config = {
        "id": "cicids2017",
        "name": "CICIDS2017",
        "engine": "A",
        "raw_dir_key": "cicids2017",
        "files": ["day1.csv"],
        "strip_column_whitespace": True,
        "label_column": "Label",
        "read_options": {"chunksize": 100},
    }
    loader = CICIDS2017Loader(config, make_paths({"cicids2017": raw_dir}))

    raw = loader.load_raw()

    assert "Flow Duration" in raw.frame.columns  # whitespace stripped
    assert raw.label_column == "Label"
    assert len(raw.frame) == 2


def test_cicids_missing_file_raises(
    tmp_path: Path, make_paths: PathsFactory
) -> None:
    raw_dir = tmp_path / "cic"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "id": "cicids2017",
        "raw_dir_key": "cicids2017",
        "files": ["missing.csv"],
        "strip_column_whitespace": True,
        "label_column": "Label",
    }
    loader = CICIDS2017Loader(config, make_paths({"cicids2017": raw_dir}))

    with pytest.raises(FileNotFoundError):
        loader.load_raw()

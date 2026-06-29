"""Abstract dataset loader interface.

Purpose
-------
Define the single interface every dataset is loaded through (Phase 1
Definition of Done: "Every dataset can be loaded using one interface").
Concrete loaders in :mod:`src.data.loaders` subclass :class:`BaseDatasetLoader`
and implement only the dataset-specific reading / labelling logic; the shared
preprocessing pipeline is applied uniformly on top.

A loader is responsible for:
    - locating and reading the raw, READ-ONLY files,
    - normalising column names,
    - identifying feature kinds and the label column,
    - returning a tidy raw DataFrame.

It is NOT responsible for cleaning, encoding, scaling or splitting — those are
shared stages (see :mod:`src.data.pipeline`).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawDataset:
    """A dataset as read from disk, before shared preprocessing.

    Attributes
    ----------
    frame:
        The tidy raw ``pandas.DataFrame`` (column names normalised).
    label_column:
        Target column name, or ``None`` for unsupervised data.
    categorical_columns:
        Columns to be treated as categorical downstream.
    """

    frame: "Any"
    label_column: str | None
    categorical_columns: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplit:
    """A reproducible train / validation / test partition.

    Each attribute is a ``(X, y)`` tuple; ``y`` is ``None`` for unsupervised
    datasets. Features have already been cleaned / encoded / scaled.
    """

    x_train: "Any"
    y_train: "Any | None"
    x_val: "Any"
    y_val: "Any | None"
    x_test: "Any"
    y_test: "Any | None"


class BaseDatasetLoader(ABC):
    """Abstract base for all dataset loaders.

    Parameters
    ----------
    dataset_config:
        The ``dataset`` block from ``configs/datasets/<id>.yaml``.
    paths:
        Resolved :class:`src.utils.paths.Paths` for locating raw files.
    """

    def __init__(self, dataset_config: Mapping[str, Any], paths: Any) -> None:
        self.config = dataset_config
        self.paths = paths
        self.dataset_id: str = dataset_config.get("id", "unknown")

    @abstractmethod
    def load_raw(self) -> RawDataset:
        """Read the raw files into a tidy :class:`RawDataset`.

        Must not modify the source files on disk (CLAUDE.md > Dataset Rules).

        Returns
        -------
        RawDataset
        """
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Return a short description of the loader's target dataset.

        Returns
        -------
        dict
            Keys: ``id``, ``name``, ``engine``.
        """
        return {
            "id": self.dataset_id,
            "name": self.config.get("name"),
            "engine": self.config.get("engine"),
        }

    # ------------------------------------------------------------------ #
    # Shared helpers for concrete loaders (keep loaders DRY).             #
    # These perform schema-level ingestion only — never feature           #
    # preprocessing, encoding, scaling or splitting.                      #
    # ------------------------------------------------------------------ #
    def raw_dir(self) -> "Any":
        """Return the read-only raw directory for this dataset.

        Resolved via ``paths.raw_dir`` using the config's ``raw_dir_key``,
        falling back to the dataset id.

        Returns
        -------
        pathlib.Path
        """
        key = self.config.get("raw_dir_key", self.dataset_id)
        return self.paths.raw_dir(key)

    def require_file(self, filename: str) -> "Any":
        """Resolve a raw file path and assert it exists.

        Parameters
        ----------
        filename:
            File name relative to the dataset's raw directory.

        Returns
        -------
        pathlib.Path

        Raises
        ------
        FileNotFoundError
            If the file is absent (with the resolved path in the message).
        """
        path = self.raw_dir() / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"[{self.dataset_id}] expected raw file is missing: {path}"
            )
        return path

    @staticmethod
    def normalize_columns(frame: "Any") -> "Any":
        """Strip leading/trailing whitespace from column names in place-safe way.

        CICIDS2017 ships headers with leading spaces; normalising makes column
        lookups deterministic across datasets.

        Parameters
        ----------
        frame:
            DataFrame whose columns may carry stray whitespace.

        Returns
        -------
        pandas.DataFrame
            The same frame with cleaned column labels.
        """
        frame.columns = [str(col).strip() for col in frame.columns]
        return frame

    def apply_drop_columns(self, frame: "Any") -> "Any":
        """Drop configured non-predictive columns (e.g. ``id``, ``difficulty``).

        Only removes columns explicitly declared under ``drop_columns`` in the
        dataset config. This is schema shaping, not feature preprocessing.

        Parameters
        ----------
        frame:
            Input DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        drop = [c for c in self.config.get("drop_columns", []) if c in frame.columns]
        if drop:
            logger.debug("[%s] dropping configured columns: %s", self.dataset_id, drop)
            frame = frame.drop(columns=drop)
        return frame

    def guard_non_empty(self, frame: "Any") -> None:
        """Raise if the assembled dataset has no rows.

        Parameters
        ----------
        frame:
            Assembled DataFrame.

        Raises
        ------
        ValueError
            If ``frame`` is empty.
        """
        if len(frame) == 0:
            raise ValueError(f"[{self.dataset_id}] loaded dataset is empty (0 rows).")

    def require_label_column(self, frame: "Any", label_column: str | None) -> None:
        """Validate that a configured label column is present.

        Parameters
        ----------
        frame:
            Assembled DataFrame.
        label_column:
            Expected label column name, or ``None`` (unsupervised) to skip.

        Raises
        ------
        ValueError
            If ``label_column`` is set but absent from ``frame``.
        """
        if label_column is not None and label_column not in frame.columns:
            raise ValueError(
                f"[{self.dataset_id}] configured label column '{label_column}' "
                f"not found in loaded data."
            )

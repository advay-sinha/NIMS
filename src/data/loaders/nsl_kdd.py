"""NSL-KDD loader (Engine A — cyber intrusion detection).

Purpose
-------
Read the headerless NSL-KDD ``KDDTrain+.txt`` / ``KDDTest+.txt`` files, attach
the configured column names, drop the auxiliary ``difficulty`` column and
expose a tidy raw dataset. Multi-class attack labels can be collapsed to
binary (normal vs attack) per config.

Limitations
-----------
NSL-KDD test labels include attack types absent from train; downstream
encoding uses ``handle_unknown=ignore`` to tolerate this.
"""

from __future__ import annotations

import logging

from src.data.base import BaseDatasetLoader, RawDataset

logger = logging.getLogger(__name__)


class NSLKDDLoader(BaseDatasetLoader):
    """Loader for the NSL-KDD dataset."""

    def load_raw(self) -> RawDataset:
        """Read and assemble the raw NSL-KDD dataframe.

        Reads the headerless ``KDDTrain+`` / ``KDDTest+`` files, applies the
        configured column names, tags each row with its source ``split`` for
        provenance and drops configured auxiliary columns (``difficulty``).
        Labels are left untouched (no encoding).

        Returns
        -------
        RawDataset

        Raises
        ------
        FileNotFoundError
            If a configured train/test file is missing under the raw dir.
        ValueError
            If the column count does not match the configured schema, or the
            assembled dataset is empty.
        """
        import pandas as pd

        from src.utils.io import read_csv

        columns: list[str] = list(self.config["columns"])
        files: dict[str, str] = self.config["files"]

        frames: list[pd.DataFrame] = []
        for split_name, filename in files.items():
            path = self.require_file(filename)
            try:
                # index_col=False forbids silent index absorption, so rows with
                # the wrong field count surface as a parser error rather than
                # being misaligned.
                frame = read_csv(path, header=None, names=columns, index_col=False)
            except pd.errors.ParserError as exc:
                raise ValueError(
                    f"[{self.dataset_id}] malformed rows in {filename}: {exc}"
                ) from exc
            if frame.shape[1] != len(columns):
                raise ValueError(
                    f"[{self.dataset_id}] {filename}: expected {len(columns)} "
                    f"columns, found {frame.shape[1]} (malformed file)."
                )
            frame["split"] = split_name
            frames.append(frame)
            logger.info(
                "[%s] loaded %s rows from %s", self.dataset_id, len(frame), filename
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = self.normalize_columns(combined)
        combined = self.apply_drop_columns(combined)

        label_column = self.config.get("label_column")
        self.require_label_column(combined, label_column)
        self.guard_non_empty(combined)

        return RawDataset(
            frame=combined,
            label_column=label_column,
            categorical_columns=tuple(self.config.get("categorical_columns", [])),
        )

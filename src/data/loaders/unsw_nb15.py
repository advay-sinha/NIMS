"""UNSW-NB15 loader (Engine A — cyber intrusion detection).

Purpose
-------
Read the headed UNSW-NB15 train/test CSVs, drop the non-predictive ``id``
column and expose a tidy raw dataset. Binary target is ``label``; the
multi-class attack family is ``attack_cat``.

Limitations
-----------
``attack_cat`` contains NaN for benign rows in some releases; cleaning fills it
with an explicit "Normal" category. TODO(data-engineer): confirm per file.
"""

from __future__ import annotations

import logging

from src.data.base import BaseDatasetLoader, RawDataset

logger = logging.getLogger(__name__)


class UNSWNB15Loader(BaseDatasetLoader):
    """Loader for the UNSW-NB15 dataset."""

    def load_raw(self) -> RawDataset:
        """Read and assemble the raw UNSW-NB15 dataframe.

        Reads the headed train/test CSVs, tags each row with its source
        ``split`` and drops the non-predictive ``id`` column per config.
        Feature columns and labels are left untouched.

        Returns
        -------
        RawDataset

        Raises
        ------
        FileNotFoundError
            If a configured train/test file is missing under the raw dir.
        ValueError
            If the configured label column is absent or the dataset is empty.
        """
        import pandas as pd

        from src.utils.io import read_csv

        files: dict[str, str] = self.config["files"]
        # Only the partitioned data files carry `split`; auxiliary files
        # (feature dictionary) are ignored here.
        split_files = {k: v for k, v in files.items() if k in {"train", "test"}}

        frames: list[pd.DataFrame] = []
        for split_name, filename in split_files.items():
            path = self.require_file(filename)
            frame = read_csv(path)
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

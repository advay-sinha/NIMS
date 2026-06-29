"""CICIDS2017 loader (Engine A — cyber intrusion detection).

Purpose
-------
Read and concatenate the eight daily CICFlowMeter CSVs, normalise the
whitespace-prefixed column names, and expose a tidy raw dataset. Textual
labels (``BENIGN``, ``DDoS``, ``PortScan``, ...) can be collapsed to binary
per config.

Limitations
-----------
Files total ~1 GB and ratio features contain ``+/-Inf`` / NaN. Loaders may read
in chunks (``read_options.chunksize``); infinity handling is deferred to the
cleaning stage. TODO(data-engineer): consider dtype downcasting to cut memory.
"""

from __future__ import annotations

import logging

from src.data.base import BaseDatasetLoader, RawDataset

logger = logging.getLogger(__name__)


class CICIDS2017Loader(BaseDatasetLoader):
    """Loader for the CICIDS2017 dataset."""

    def load_raw(self) -> RawDataset:
        """Read, concatenate and normalise the raw CICIDS2017 dataframe.

        Iterates the eight configured daily captures, strips leading/trailing
        whitespace from the (notoriously space-prefixed) column names and
        concatenates them. Infinity / NaN handling and label collapsing are
        deferred to later stages — this layer only ingests.

        Returns
        -------
        RawDataset

        Raises
        ------
        FileNotFoundError
            If any configured capture file is missing under the raw dir.
        ValueError
            If the configured label column is absent after normalisation, or
            the assembled dataset is empty.
        """
        import pandas as pd

        from src.utils.io import read_csv

        files: list[str] = list(self.config["files"])
        read_options: dict = dict(self.config.get("read_options", {}) or {})
        # `chunksize` belongs to streaming concerns we don't use here; reading
        # whole files keeps the ingestion contract simple and deterministic.
        read_options.pop("chunksize", None)

        frames: list[pd.DataFrame] = []
        for filename in files:
            path = self.require_file(filename)
            frame = read_csv(path, **read_options)
            if self.config.get("strip_column_whitespace", False):
                frame = self.normalize_columns(frame)
            frame["source_file"] = filename
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

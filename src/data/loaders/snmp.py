"""SNMP 2016 loader (Engine B — network health prediction).

Purpose
-------
Read the SNMP MIB-counter telemetry CSV and expose a tidy raw dataset for
unsupervised anomaly / degradation modelling in later phases (Isolation
Forest, LSTM Autoencoder). Counters are cumulative; first-difference feature
engineering is deferred to the feature layer.

Limitations
-----------
Timestamp and label columns are not yet confirmed from the reference PDF; the
config leaves them ``null`` and the loader treats the data as unsupervised
until set. TODO(data-engineer): populate ``timestamp_column`` / ``label_column``.
"""

from __future__ import annotations

import logging

from src.data.base import BaseDatasetLoader, RawDataset

logger = logging.getLogger(__name__)


class SNMPLoader(BaseDatasetLoader):
    """Loader for the SNMP 2016 telemetry dataset."""

    def load_raw(self) -> RawDataset:
        """Read the raw SNMP telemetry dataframe.

        Returns
        -------
        RawDataset
            ``label_column`` is ``None`` while the dataset is unsupervised.

        Raises
        ------
        FileNotFoundError
            If the configured data file is missing under the raw dir.
        """
        # TODO(data-engineer): read config["files"]["data"] via read_csv;
        #   parse timestamp_column when configured; return RawDataset with
        #   label_column from config (may be None).
        raise NotImplementedError

"""Network-health feature engineering.

Purpose
-------
Derive model features from preprocessed telemetry: canonical health rates
(traffic/error/discard), rolling statistics, lags and status-change
indicators — all computed per (device, interface) series with configurable
windows. Rolling/lag features are strictly causal (past values only), so
computing them per split cannot leak future information.

Outputs
-------
The feature frame plus a metadata dict (persisted by
:mod:`src.network_health.artifacts`).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from src.network_health.schema import TelemetrySchema

logger = logging.getLogger(__name__)

# Canonical health features derived from counter rates when available.
_RATE_ALIASES = {
    "traffic_in_rate": "ifInOctets_rate",
    "traffic_out_rate": "ifOutOctets_rate",
    "error_rate_in": "ifInErrors_rate",
    "error_rate_out": "ifOutErrors_rate",
    "discard_rate_in": "ifInDiscards_rate",
    "discard_rate_out": "ifOutDiscards_rate",
}


def build_features(
    frame: "Any",
    schema: TelemetrySchema,
    features_config: Mapping[str, Any],
) -> tuple["Any", dict[str, Any]]:
    """Add health features to one preprocessed split.

    Parameters
    ----------
    frame:
        Preprocessed telemetry (with ``*_rate`` columns).
    schema:
        Column roles.
    features_config:
        The ``network_health.features`` block (``rolling_windows``,
        ``rolling_stats``, ``lags``).

    Returns
    -------
    tuple
        ``(feature_frame, metadata)`` — metadata lists every generated
        feature and the configuration used.
    """
    result = frame.copy(deep=False)
    groups = lambda: result.groupby(schema.series_columns, sort=False)  # noqa: E731

    # Canonical rate aliases (only when the source counter exists).
    aliases = {}
    for alias, source in _RATE_ALIASES.items():
        if source in result.columns:
            result[alias] = result[source]
            aliases[alias] = source

    base_columns = list(aliases)
    for gauge in schema.gauge_columns:
        if gauge in result.columns and gauge not in base_columns:
            base_columns.append(gauge)

    # Rolling statistics per series (causal windows: current + past samples).
    windows = [int(w) for w in features_config.get("rolling_windows", (3,))]
    stats = [str(s) for s in features_config.get("rolling_stats", ("mean",))]
    rolling_features = []
    for window in windows:
        rolled = groups()[base_columns].rolling(
            window=window, min_periods=1
        )
        for stat in stats:
            stat_frame = getattr(rolled, stat)().reset_index(drop=True)
            stat_frame = stat_frame.fillna(0.0)  # std of a single sample
            for column in base_columns:
                name = f"{column}_roll{window}_{stat}"
                result[name] = stat_frame[column].to_numpy()
                rolling_features.append(name)

    # Lag features per series (missing history -> 0).
    lags = [int(lag) for lag in features_config.get("lags", (1,))]
    lag_features = []
    for lag in lags:
        shifted = groups()[base_columns].shift(lag).fillna(0.0)
        for column in base_columns:
            name = f"{column}_lag{lag}"
            result[name] = shifted[column].to_numpy()
            lag_features.append(name)

    # Status-change indicators per series.
    status_features = []
    for column in schema.status_columns:
        if column not in result.columns:
            continue
        changed = (
            groups()[column].shift(1).ne(result[column])
            & groups()[column].shift(1).notna()
        )
        name = f"{column}_changed"
        result[name] = changed.astype(int).to_numpy()
        status_features.append(name)

    feature_columns = base_columns + rolling_features + lag_features + status_features
    metadata = {
        "base_features": base_columns,
        "rate_aliases": aliases,
        "rolling_features": rolling_features,
        "lag_features": lag_features,
        "status_change_features": status_features,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "config": {"rolling_windows": windows, "rolling_stats": stats, "lags": lags},
    }
    logger.info(
        "Built %d feature column(s) (%d base, %d rolling, %d lag, %d status).",
        len(feature_columns), len(base_columns), len(rolling_features),
        len(lag_features), len(status_features),
    )
    return result, metadata

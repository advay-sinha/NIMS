"""Isolation Forest anomaly baseline.

Purpose
-------
The first Engine B model: an Isolation Forest over the engineered health
features. With labels, it trains on the healthy rows only and is evaluated
supervised; without labels it trains on the full training split and flags
scores above a configured train-score quantile.

Outputs
-------
A fitted :class:`NetworkHealthBaseline` and per-split metric dicts
(persisted by :mod:`src.network_health.artifacts`).

Limitations
-----------
The LSTM autoencoder is a later phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.network_health.metrics import labeled_metrics, unlabeled_metrics
from src.network_health.schema import TelemetrySchema

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD_QUANTILE = 0.99


@dataclass
class NetworkHealthBaseline:
    """Isolation Forest wrapper for telemetry anomaly detection.

    Attributes
    ----------
    params:
        Estimator hyperparameters from configuration.
    threshold_quantile:
        Unlabeled decision threshold: the train-score quantile above which a
        reading is flagged anomalous.
    feature_columns:
        Feature names the model was fitted on (order preserved).
    """

    params: dict[str, Any] = field(default_factory=dict)
    threshold_quantile: float = _DEFAULT_THRESHOLD_QUANTILE
    feature_columns: list[str] = field(default_factory=list)
    model: Any = None
    threshold: float | None = None
    labeled: bool = False

    def fit(self, x: "Any", y: "Any | None" = None) -> "NetworkHealthBaseline":
        """Fit on the training split (healthy rows only when labels exist)."""
        from sklearn.ensemble import IsolationForest

        self.feature_columns = [str(c) for c in x.columns]
        self.labeled = y is not None
        train_x = x
        if y is not None:
            healthy = (y == 0).to_numpy()
            if healthy.any():
                train_x = x[healthy]
            logger.info(
                "Labels present: fitting on %d/%d healthy row(s).",
                len(train_x), len(x),
            )
        self.model = IsolationForest(**self.params)
        self.model.fit(train_x)
        # Higher = more anomalous (negated sklearn decision_function).
        train_scores = self.score(x)
        self.threshold = float(
            __import__("numpy").quantile(train_scores, self.threshold_quantile)
        )
        return self

    def score(self, x: "Any") -> "Any":
        """Anomaly scores for ``x`` (higher = more anomalous)."""
        return -self.model.decision_function(x[self.feature_columns])

    def predict(self, x: "Any") -> "Any":
        """Anomaly flags (1 = anomalous) via the fitted threshold."""
        return (self.score(x) > self.threshold).astype(int)


def train_baseline(
    feature_splits: Mapping[str, "Any"],
    feature_columns: list[str],
    schema: TelemetrySchema,
    model_config: Mapping[str, Any],
    seed: int,
) -> tuple[NetworkHealthBaseline, dict[str, Any]]:
    """Train the Isolation Forest baseline and evaluate every split.

    Parameters
    ----------
    feature_splits:
        ``{split: feature DataFrame}`` from the feature stage.
    feature_columns:
        Model feature names (from the feature metadata).
    schema:
        Column roles (label column, when configured and present).
    model_config:
        The ``network_health.model.isolation_forest`` block.
    seed:
        Reproducibility seed (used when the params omit ``random_state``).

    Returns
    -------
    tuple
        ``(fitted baseline, {split: metrics})``.
    """
    params = dict(model_config.get("params") or {})
    params.setdefault("random_state", seed)
    baseline = NetworkHealthBaseline(
        params=params,
        threshold_quantile=float(
            model_config.get("threshold_quantile", _DEFAULT_THRESHOLD_QUANTILE)
        ),
    )

    train = feature_splits["train"]
    label = schema.label_column if (
        schema.label_column and schema.label_column in train.columns
    ) else None
    x_train = train[feature_columns]
    y_train = train[label] if label else None
    baseline.fit(x_train, y_train)

    metrics: dict[str, Any] = {}
    for split_name, split_frame in feature_splits.items():
        if not len(split_frame):
            continue
        x = split_frame[feature_columns]
        scores = baseline.score(x)
        y_pred = baseline.predict(x)
        if label:
            metrics[split_name] = labeled_metrics(
                split_frame[label], y_pred, scores
            )
        else:
            metrics[split_name] = unlabeled_metrics(
                scores, y_pred, baseline.threshold or 0.0
            )
    return baseline, metrics

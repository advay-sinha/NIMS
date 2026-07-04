"""Isolation Forest anomaly-detection model.

An unsupervised baseline (CLAUDE.md > Engine A). It fits on features only and
flags anomalies; ``y`` is ignored during fitting. scikit-learn estimators run
on CPU (no fake GPU acceleration — CLAUDE.md > Scikit-learn).
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class IsolationForestModel(BaseModel):
    """Isolation Forest wrapper (unsupervised; predicts 1=anomaly, 0=normal)."""

    name = "isolation_forest"
    is_supervised = False

    def fit(
        self,
        x_train: "Any",
        y_train: "Any | None" = None,
        x_val: "Any | None" = None,
        y_val: "Any | None" = None,
    ) -> "IsolationForestModel":
        """Fit the Isolation Forest on training features (labels ignored)."""
        from sklearn.ensemble import IsolationForest

        self.device = "cpu"
        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        # Config-provided n_jobs wins; only default to all cores when unset.
        params.setdefault("n_jobs", -1)

        logger.info("Training Isolation Forest on device=cpu (%s trees).",
                    params.get("n_estimators", 100))
        self.model = IsolationForest(**params)
        self.fitted_params = self.model.get_params()
        logger.info("Isolation Forest final parameters: %s", self.fitted_params)
        self.model.fit(x_train)
        return self

    def predict(self, x: "Any") -> "Any":
        """Return anomaly labels: 1 for anomalies (outliers), 0 for normal."""
        import numpy as np

        raw = self.model.predict(x)  # +1 inlier, -1 outlier
        return (raw == -1).astype(np.int64)

    def score_samples(self, x: "Any") -> "Any":
        """Return anomaly scores (higher = more anomalous)."""
        return -self.model.decision_function(x)

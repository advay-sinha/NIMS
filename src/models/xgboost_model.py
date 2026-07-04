"""XGBoost intrusion-detection model.

Uses GPU training (``device="cuda"``, ``tree_method="hist"``) when a CUDA build
and GPU are available, falling back to CPU otherwise (CLAUDE.md > XGBoost). All
hyperparameters come from configuration.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

# Multiclass eval metrics and their binary counterparts, used when a
# configured multiclass objective is reconciled to a binary target.
_BINARY_EVAL_METRICS = {"mlogloss": "logloss", "merror": "error"}


def _reconcile_multiclass_objective(params: dict[str, Any], n_classes: int) -> None:
    """Align a configured multiclass objective with the observed class count.

    The XGBoost sklearn wrapper injects ``num_class`` only when it sees more
    than two classes, so an explicitly configured ``multi:*`` objective on a
    binary target reaches the core booster with ``num_class=0`` and aborts
    with "value 0 for Parameter num_class should be greater equal to 1".
    For two-class targets the objective is switched to ``binary:logistic``
    (the wrapper's own choice when no objective is configured) and multiclass
    eval metrics are mapped to their binary counterparts. Targets with more
    than two classes are left untouched.

    Mutates ``params`` in place; the model's configured ``self.params`` is
    never modified (``fit`` operates on a copy).
    """
    objective = str(params.get("objective", ""))
    if n_classes > 2 or not objective.startswith("multi:"):
        return
    params["objective"] = "binary:logistic"
    metric = params.get("eval_metric")
    if isinstance(metric, str):
        params["eval_metric"] = _BINARY_EVAL_METRICS.get(metric, metric)
    elif isinstance(metric, (list, tuple)):
        params["eval_metric"] = [
            _BINARY_EVAL_METRICS.get(str(m), str(m)) for m in metric
        ]
    logger.info(
        "Configured objective %r is multiclass but the training labels have "
        "%d classes; training with objective=%r eval_metric=%s instead.",
        objective, n_classes, params["objective"], params.get("eval_metric"),
    )


class XGBoostModel(BaseModel):
    """XGBoost classifier wrapper."""

    name = "xgboost"
    is_supervised = True

    def fit(
        self,
        x_train: "Any",
        y_train: "Any | None" = None,
        x_val: "Any | None" = None,
        y_val: "Any | None" = None,
    ) -> "XGBoostModel":
        """Fit the XGBoost classifier (GPU when available)."""
        from xgboost import XGBClassifier

        from src.utils.hardware import supports_xgboost_gpu

        gpu = self.use_gpu and supports_xgboost_gpu()
        self.device = "cuda" if gpu else "cpu"

        params = dict(self.params)
        params.setdefault("tree_method", "hist")
        params["device"] = self.device
        params.setdefault("random_state", self.seed)

        if y_train is not None:
            import numpy as np

            n_classes = int(np.unique(np.asarray(y_train)).size)
            _reconcile_multiclass_objective(params, n_classes)

        eval_set = None
        if x_val is not None and y_val is not None:
            eval_set = [(x_val, y_val)]

        logger.info("Training XGBoost on device=%s (%d trees, %d rows).",
                    self.device, params.get("n_estimators", 100),
                    len(x_train) if hasattr(x_train, "__len__") else -1)
        self.model = XGBClassifier(**params)
        self.fitted_params = self.model.get_params()
        logger.info("XGBoost final parameters: %s", self.fitted_params)
        self.model.fit(x_train, y_train, eval_set=eval_set, verbose=False)

        if self.device == "cuda":
            # Predictions are made on host (CPU) arrays. Pin the booster's
            # inference device to CPU so XGBoost uses inplace_predict directly
            # instead of falling back to a DMatrix copy on a device mismatch
            # (removes the "mismatched devices" warning). Training is unaffected;
            # the fitted trees are identical.
            self.model.get_booster().set_param({"device": "cpu"})
        return self

    def predict(self, x: "Any") -> "Any":
        """Return predicted class labels."""
        return self.model.predict(x)

    def predict_proba(self, x: "Any") -> "Any | None":
        """Return class probabilities."""
        return self.model.predict_proba(x)

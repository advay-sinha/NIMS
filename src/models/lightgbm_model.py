"""LightGBM intrusion-detection model.

Attempts GPU training (``device="gpu"``) first; if the installed build is
CPU-only the hardware probe detects it, logs a warning and falls back to CPU —
never silently (CLAUDE.md > LightGBM). All hyperparameters come from config.
"""

from __future__ import annotations

import logging
from typing import Any

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM classifier wrapper."""

    name = "lightgbm"
    is_supervised = True

    def fit(
        self,
        x_train: "Any",
        y_train: "Any | None" = None,
        x_val: "Any | None" = None,
        y_val: "Any | None" = None,
    ) -> "LightGBMModel":
        """Fit the LightGBM classifier.

        Attempts GPU training when permitted and CUDA is present; if the
        installed LightGBM build is CPU-only the GPU attempt fails and we fall
        back to CPU with a warning (never silently — CLAUDE.md > LightGBM).
        """
        from src.utils.hardware import supports_lightgbm_gpu

        eval_set = None
        if x_val is not None and y_val is not None:
            eval_set = [(x_val, y_val)]

        if self.use_gpu and supports_lightgbm_gpu():
            try:
                self._fit_on_device("gpu", x_train, y_train, eval_set)
                return self
            except Exception as exc:
                logger.warning(
                    "LightGBM GPU training unavailable (CPU-only build?); "
                    "falling back to CPU: %s", exc,
                )

        self._fit_on_device("cpu", x_train, y_train, eval_set)
        return self

    def _fit_on_device(
        self,
        device: str,
        x_train: "Any",
        y_train: "Any | None",
        eval_set: "Any | None",
    ) -> None:
        """Construct and fit an ``LGBMClassifier`` on the given device."""
        from lightgbm import LGBMClassifier

        params = dict(self.params)
        params["device"] = device
        params.setdefault("random_state", self.seed)
        params.setdefault("verbosity", -1)

        logger.info(
            "Training LightGBM on device=%s (%d trees, %d rows).",
            device, params.get("n_estimators", 100),
            len(x_train) if hasattr(x_train, "__len__") else -1,
        )
        model = LGBMClassifier(**params)
        # The FINAL parameter dictionary of the constructed estimator — what
        # LightGBM will actually train with (config + estimator defaults).
        # Logged before fitting and persisted into the run manifest so every
        # experiment records the effective (not just configured) parameters.
        self.fitted_params = model.get_params()
        logger.info("LightGBM final parameters: %s", self.fitted_params)

        model.fit(x_train, y_train, eval_set=eval_set)
        self.model = model
        self.device = device
        logger.info(
            "LightGBM fitted: objective=%s n_classes=%d n_features=%d",
            model.objective_, model.n_classes_, model.n_features_,
        )

    def predict(self, x: "Any") -> "Any":
        """Return predicted class labels."""
        return self.model.predict(x)

    def predict_proba(self, x: "Any") -> "Any | None":
        """Return class probabilities."""
        return self.model.predict_proba(x)

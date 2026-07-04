"""Abstract model interface for Engine A.

Purpose
-------
Define the single interface every intrusion-detection model is trained and
served through, so the trainer and evaluation layers are model-agnostic
(CLAUDE.md > Project Architecture: "Layers communicate only through clearly
defined interfaces"). Concrete models implement only ``fit`` / ``predict`` /
``predict_proba``; serialisation is shared.

A model is responsible for:
    - choosing its execution device via :mod:`src.utils.hardware`,
    - fitting on training data (optionally using a validation set),
    - producing predictions (and probabilities when supported).

It is NOT responsible for loading data, computing metrics or experiment
bookkeeping — those are the trainer's job.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all Engine A models.

    Parameters
    ----------
    params:
        Model hyperparameters from configuration.
    use_gpu:
        Whether GPU execution is permitted (negotiated with hardware support).
    seed:
        Reproducibility seed injected into the underlying estimator.
    """

    name: str = "base"
    is_supervised: bool = True

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        use_gpu: bool = True,
        seed: int = 42,
    ) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.use_gpu = use_gpu
        self.seed = seed
        self.device: str = "cpu"
        self.model: Any = None
        # Final parameters of the constructed estimator (estimator.get_params()
        # at fit time). ``params`` above is the *configured* dict; the two can
        # differ (e.g. LightGBM device fallback), so both are persisted.
        self.fitted_params: dict[str, Any] | None = None

    @abstractmethod
    def fit(
        self,
        x_train: "Any",
        y_train: "Any | None" = None,
        x_val: "Any | None" = None,
        y_val: "Any | None" = None,
    ) -> "BaseModel":
        """Fit the model on training data; return ``self``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: "Any") -> "Any":
        """Return predicted labels for ``x``."""
        raise NotImplementedError

    def predict_proba(self, x: "Any") -> "Any | None":
        """Return class probabilities, or ``None`` when unsupported."""
        return None

    @property
    def classes_(self) -> "Any | None":
        """Return the full set of labels the model was fitted on, or ``None``.

        For supervised classifiers this is the complete training class list
        (e.g. ``LabelEncoder``-encoded labels), used so multiclass ROC-AUC is
        computed over every class — not only those present in a given split.
        """
        return getattr(self.model, "classes_", None)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model wrapper via joblib."""
        from src.utils.io import save_artifact

        return save_artifact(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load a previously saved model wrapper."""
        from src.utils.io import load_artifact

        return load_artifact(path)

    def describe(self) -> dict[str, Any]:
        """Return a short description of the configured model.

        ``params`` is the configuration as provided; ``fitted_params`` is the
        estimator's final parameter dictionary captured at fit time (``None``
        before fitting).
        """
        return {
            "name": self.name,
            "supervised": self.is_supervised,
            "device": self.device,
            "params": self.params,
            "fitted_params": self.fitted_params,
        }

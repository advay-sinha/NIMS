"""Model bundle loading and caching.

Purpose
-------
Resolve a dataset's serving model through the registry and load everything a
prediction needs into one immutable bundle: the fitted model, the saved
preprocessing transformers (feature encoder, scaler, label encoder) and the
canonical expected-feature list. Bundles are cached in memory keyed by
``(dataset, stage)`` so repeated requests never reload joblib files.

Inputs
------
Dataset id, stage and the registry directory.

Outputs
-------
:class:`ModelBundle` instances.

Limitations
-----------
Registry paths are authoritative — nothing is fitted or recomputed here.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.errors import ModelNotAvailableError

logger = logging.getLogger(__name__)

_PREPROCESSING_FILES = {
    "encoder": "encoder.joblib",
    "scaler": "scaler.joblib",
    "label_encoder": "label_encoder.joblib",
    "feature_selector": "feature_selector.joblib",
}


@dataclass(frozen=True)
class ModelBundle:
    """Everything needed to serve one dataset's model.

    Attributes
    ----------
    dataset, stage:
        Registry lookup identity.
    resolution:
        The registry resolver's output (experiment id, metrics, paths).
    model:
        The loaded :class:`src.models.base.BaseModel` wrapper.
    encoder, scaler, label_encoder, feature_selector:
        Saved pipeline transformers, applied via ``transform`` /
        ``inverse_transform`` only — never refit. All four are required at
        load time (a missing artefact is a load error, never a silent skip).
    expected_features:
        Final model-feature names in training order (canonical list from the
        feature-engineering artefacts).
    required_columns:
        Raw input columns a request must provide: the encoder's categorical
        columns plus every expected feature the encoder does not produce.
    loaded_at:
        UTC ISO timestamp of the load.
    """

    dataset: str
    stage: str
    resolution: dict[str, Any]
    model: Any
    encoder: Any | None
    scaler: Any | None
    label_encoder: Any | None
    feature_selector: Any | None
    expected_features: list[str]
    required_columns: list[str]
    loaded_at: str


def _required_input_columns(
    expected_features: list[str], encoder: Any | None, scaler: Any | None = None
) -> list[str]:
    """Derive the raw columns a request must contain (deterministic order).

    Preprocessing replays on the PRE-selection schema: the encoder needs its
    categorical columns and the scaler every numeric column it was fit on
    (feature selection ran after scaling, so the final feature list is a
    subset). Expected features the encoder does not produce are also
    required directly.
    """
    encoded_outputs = set(
        getattr(encoder, "feature_names_out", ()) or ()
    ) if encoder is not None else set()
    categorical = list(getattr(encoder, "columns", ()) or ()) if encoder else []
    scaled = [
        c for c in (getattr(scaler, "columns", ()) or ())
        if c not in encoded_outputs
    ] if scaler is not None else []
    passthrough = [f for f in expected_features if f not in encoded_outputs]
    seen: set[str] = set()
    required = []
    for column in categorical + scaled + passthrough:
        if column not in seen:
            seen.add(column)
            required.append(column)
    return required


def load_bundle(dataset: str, stage: str, registry_dir: Path) -> ModelBundle:
    """Load the serving bundle for a dataset from registry-resolved paths.

    Parameters
    ----------
    dataset, stage:
        Registry lookup identity.
    registry_dir:
        Registry directory containing the built registry files.

    Returns
    -------
    ModelBundle

    Raises
    ------
    ModelNotAvailableError
        When no model is assigned, or any required artefact is missing.
    """
    from src.models.base import BaseModel
    from src.registry.registry import RegistryError
    from src.registry.resolver import resolve_model
    from src.training.feature_audit import load_expected_features
    from src.utils.io import load_artifact

    try:
        resolution = resolve_model(dataset, stage, registry_dir=registry_dir)
    except RegistryError as exc:
        raise ModelNotAvailableError(str(exc)) from exc

    model_path = Path(resolution["model_artifact_path"])
    if not model_path.is_file():
        raise ModelNotAvailableError(
            f"Model artefact for '{dataset}' is missing on disk: {model_path}"
        )
    model = BaseModel.load(model_path)

    transformers: dict[str, Any | None] = dict.fromkeys(_PREPROCESSING_FILES)
    pre_dir = resolution.get("artifacts", {}).get("preprocessing_artifacts")
    missing = []
    for name, filename in _PREPROCESSING_FILES.items():
        artifact_path = Path(pre_dir) / filename if pre_dir else None
        if artifact_path is not None and artifact_path.is_file():
            transformers[name] = load_artifact(artifact_path)
        else:
            missing.append(filename)
    if missing:
        # Saved transforms are never silently bypassed at inference time.
        raise ModelNotAvailableError(
            f"Preprocessing artefacts for '{dataset}' are missing: "
            f"{', '.join(missing)} (expected under {pre_dir or 'unresolved'})."
        )

    features_dir = resolution.get("artifacts", {}).get("features")
    expected = (
        load_expected_features(Path(features_dir)) if features_dir else None
    )
    if not expected:
        raise ModelNotAvailableError(
            f"Feature-engineering artefacts for '{dataset}' are missing; the "
            f"expected model features cannot be determined."
        )

    bundle = ModelBundle(
        dataset=dataset,
        stage=stage,
        resolution=resolution,
        model=model,
        encoder=transformers["encoder"],
        scaler=transformers["scaler"],
        label_encoder=transformers["label_encoder"],
        feature_selector=transformers["feature_selector"],
        expected_features=list(expected),
        required_columns=_required_input_columns(
            list(expected), transformers["encoder"], transformers["scaler"]
        ),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "Loaded bundle %s/%s -> %s (%d features, %d required input columns).",
        dataset, stage, resolution["experiment_id"],
        len(bundle.expected_features), len(bundle.required_columns),
    )
    return bundle


@dataclass
class BundleCache:
    """Thread-safe in-memory ``(dataset, stage) -> ModelBundle`` cache."""

    registry_dir: Path
    enabled: bool = True
    _bundles: dict[tuple[str, str], ModelBundle] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, dataset: str, stage: str) -> ModelBundle:
        """Return the cached bundle, loading it on first use."""
        key = (dataset, stage)
        if not self.enabled:
            return load_bundle(dataset, stage, self.registry_dir)
        with self._lock:
            bundle = self._bundles.get(key)
            if bundle is None:
                bundle = load_bundle(dataset, stage, self.registry_dir)
                self._bundles[key] = bundle
            return bundle

    def clear(self) -> None:
        """Drop every cached bundle (safe reload: next request reloads)."""
        with self._lock:
            self._bundles.clear()

    @property
    def count(self) -> int:
        """Number of currently loaded bundles."""
        return len(self._bundles)

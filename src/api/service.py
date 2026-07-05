"""Inference service: raw rows -> transformed features -> predictions.

Purpose
-------
Run the training pipeline's transform chain on request rows using the SAVED
artefacts only (nothing is fitted here)::

    raw rows -> column validation -> encoder.transform -> scaler.transform
             -> feature_selector.transform -> expected-feature alignment
             -> predict -> label_encoder.inverse_transform -> response

Inputs
------
A :class:`src.api.loader.ModelBundle` and a pandas DataFrame of raw rows.

Outputs
-------
The prediction response payload (dict, schema-validated by the app layer).

Limitations
-----------
Batch scoring only; no packet capture, streaming or training.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from src.api.errors import (
    InferenceFailureError,
    PayloadTooLargeError,
    RequestValidationError,
)

logger = logging.getLogger(__name__)

_MAX_LISTED_COLUMNS = 20


def frame_from_csv(payload: bytes) -> "Any":
    """Parse an uploaded CSV into a DataFrame.

    Parameters
    ----------
    payload:
        Raw request body bytes.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    RequestValidationError
        For unparseable or empty CSV content.
    """
    import pandas as pd

    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001 - any parser failure is a 422
        raise RequestValidationError(f"Invalid CSV upload: {exc}") from exc
    if frame.empty:
        raise RequestValidationError("The uploaded CSV contains no data rows.")
    return frame


def frame_from_rows(rows: list[dict[str, Any]]) -> "Any":
    """Build a DataFrame from JSON rows.

    Parameters
    ----------
    rows:
        List of ``{column: value}`` mappings.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    RequestValidationError
        When ``rows`` is empty.
    """
    import pandas as pd

    if not rows:
        raise RequestValidationError("Request contains no rows.")
    return pd.DataFrame(rows)


def _validate_columns(bundle: Any, frame: "Any", warnings: list[str]) -> "Any":
    """Check required input columns; ignore extras with a warning."""
    columns = set(map(str, frame.columns))
    missing = [c for c in bundle.required_columns if c not in columns]
    if missing:
        listed = ", ".join(missing[:_MAX_LISTED_COLUMNS])
        suffix = "..." if len(missing) > _MAX_LISTED_COLUMNS else ""
        raise RequestValidationError(
            f"Missing {len(missing)} required column(s): {listed}{suffix}"
        )
    extra = sorted(columns - set(bundle.required_columns))
    if extra:
        warnings.append(
            f"Ignored {len(extra)} column(s) not used by the model: "
            f"{', '.join(extra[:_MAX_LISTED_COLUMNS])}"
        )
    return frame[bundle.required_columns]


def _transform(bundle: Any, frame: "Any") -> "Any":
    """Apply the saved preprocessing transforms and align to model features."""
    from src.data.encoding import apply_encoder
    from src.data.scaling import apply_scaler

    x = frame
    try:
        if bundle.encoder is not None:
            x = apply_encoder(bundle.encoder, x)
        if bundle.scaler is not None:
            x = apply_scaler(bundle.scaler, x)
        if bundle.feature_selector is not None:
            x = bundle.feature_selector.transform(x)
    except Exception as exc:  # noqa: BLE001 - reported as a clean 500
        logger.exception("Preprocessing transform failed for %s.", bundle.dataset)
        raise InferenceFailureError(
            "Preprocessing transform failed; see the server logs."
        ) from exc

    produced = set(map(str, x.columns))
    absent = [f for f in bundle.expected_features if f not in produced]
    if absent:
        raise RequestValidationError(
            f"Transformed input lacks {len(absent)} expected feature(s): "
            f"{', '.join(absent[:_MAX_LISTED_COLUMNS])}"
        )
    x = x[bundle.expected_features]

    # Audit: the matrix must be fully numeric with no missing values —
    # the same invariant the trainer enforces before every fit.
    import numpy as np

    try:
        matrix = x.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(
            f"Feature matrix contains non-numeric values: {exc}"
        ) from exc
    n_missing = int(matrix.isna().sum().sum())
    if n_missing:
        raise RequestValidationError(
            f"Feature matrix contains {n_missing} missing value(s) after "
            f"the transform; fill or drop incomplete rows."
        )
    return matrix


def _decode_labels(bundle: Any, encoded: "Any") -> list[Any]:
    """Invert encoded integer predictions via the saved label encoder.

    Uses ``label_encoder.encoder.inverse_transform`` for every value inside
    the fitted class range; out-of-range values (e.g. the ``-1`` unknown
    sentinel) pass through as integers rather than failing.
    """
    import numpy as np

    sk_encoder = getattr(bundle.label_encoder, "encoder", None)
    fitted = getattr(sk_encoder, "classes_", None) if sk_encoder else None
    values = np.asarray(list(encoded), dtype=int)
    if fitted is None or len(fitted) == 0:
        return [int(v) for v in values]

    in_range = (values >= 0) & (values < len(fitted))
    decoded: list[Any] = [int(v) for v in values]
    if in_range.any():
        names = sk_encoder.inverse_transform(values[in_range])
        for position, name in zip(np.flatnonzero(in_range), names):
            decoded[int(position)] = str(name)
    return decoded


def run_inference(
    bundle: Any,
    frame: "Any",
    *,
    max_rows: int,
    return_probabilities: bool = True,
) -> dict[str, Any]:
    """Score raw rows with a loaded bundle.

    Parameters
    ----------
    bundle:
        :class:`src.api.loader.ModelBundle`.
    frame:
        Raw input rows.
    max_rows:
        Request row cap.
    return_probabilities:
        Include class probabilities when the model supports them.

    Returns
    -------
    dict
        The prediction response payload.

    Raises
    ------
    PayloadTooLargeError, RequestValidationError, InferenceFailureError
    """
    if len(frame) > max_rows:
        raise PayloadTooLargeError(
            f"Request has {len(frame):,} rows; the limit is {max_rows:,}."
        )
    warnings: list[str] = []
    matrix = _transform(bundle, _validate_columns(bundle, frame, warnings))

    try:
        encoded = bundle.model.predict(matrix)
        proba = (
            bundle.model.predict_proba(matrix) if return_probabilities else None
        )
    except Exception as exc:  # noqa: BLE001 - reported as a clean 500
        logger.exception("Prediction failed for %s.", bundle.dataset)
        raise InferenceFailureError(
            "Prediction failed; see the server logs."
        ) from exc

    response: dict[str, Any] = {
        "dataset": bundle.dataset,
        "model_type": bundle.resolution["model_type"],
        "experiment_id": bundle.resolution["experiment_id"],
        "n_rows": int(len(matrix)),
        "predictions": _decode_labels(bundle, encoded),
        "warnings": warnings,
    }
    if proba is not None:
        response["probabilities"] = [
            [float(p) for p in row] for row in proba
        ]
        model_classes = getattr(bundle.model, "classes_", None)
        response["class_labels"] = _decode_labels(
            bundle, [] if model_classes is None else list(model_classes)
        )
    logger.info(
        "Scored %d row(s) for %s with %s.",
        len(matrix), bundle.dataset, bundle.resolution["experiment_id"],
    )
    return response

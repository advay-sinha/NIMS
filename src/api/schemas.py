"""API request/response schemas (pydantic).

Purpose
-------
Typed contracts for every endpoint, so the OpenAPI documentation and response
validation come for free from FastAPI.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """``GET /health`` payload."""

    status: str
    service: str
    loaded_models: int


class ModelInfo(BaseModel):
    """One production model listed by ``GET /models``."""

    dataset: str
    model_type: str
    experiment_id: str
    metrics: dict[str, dict[str, float]]
    status: str


class ModelsResponse(BaseModel):
    """``GET /models`` payload."""

    models: list[ModelInfo]


class PredictJsonRequest(BaseModel):
    """``POST /predict-json/{dataset}`` body."""

    rows: list[dict[str, Any]] = Field(
        ..., description="Raw input rows as {column: value} mappings."
    )


class PredictionResponse(BaseModel):
    """Prediction payload shared by the CSV and JSON endpoints."""

    dataset: str
    model_type: str
    experiment_id: str
    n_rows: int
    predictions: list[Any]
    probabilities: list[list[float]] | None = None
    class_labels: list[Any] | None = None
    warnings: list[str] = Field(default_factory=list)

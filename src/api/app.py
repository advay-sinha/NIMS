"""FastAPI application factory and endpoints.

Purpose
-------
The thin HTTP layer over the registry-driven loader and inference service:
``/health``, ``/models``, ``/predict/{dataset}`` (CSV upload) and
``/predict-json/{dataset}``. All heavy lifting lives in
:mod:`src.api.loader` / :mod:`src.api.service`; endpoints only translate
between HTTP and those modules.

Run with::

    python -m scripts.run_api
    # or
    uvicorn src.api.app:app --reload

Limitations
-----------
Batch inference only — no packet capture, no dashboard, no training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

# DLL load-order guard (see scripts/_bootstrap.py): pyarrow must load before
# any torch import when deep-learning models are served on Windows.
try:  # pragma: no cover - depends on the installed environment
    import pyarrow.dataset  # noqa: F401  (import for DLL side effect only)
except ImportError:
    pass

# FastAPI types must be importable at module scope so endpoint annotations
# (evaluated lazily under `from __future__ import annotations`) resolve.
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.errors import ApiError
from src.api.loader import BundleCache
from src.api.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictJsonRequest,
    PredictionResponse,
)
from src.api.service import frame_from_csv, frame_from_rows, run_inference

logger = logging.getLogger(__name__)

SERVICE_NAME = "netsentinel-inference"

_DEFAULTS = {
    "stage": "production",
    "max_rows_per_request": 10_000,
    "return_probabilities": True,
    "model_cache_enabled": True,
}


def create_app(
    config: Mapping[str, Any] | None = None,
    registry_dir: Path | None = None,
) -> "Any":
    """Build the FastAPI application.

    Parameters
    ----------
    config:
        Effective configuration (loaded from ``configs/config.yaml`` when
        omitted); the ``api`` block controls stage, row limits and caching.
    registry_dir:
        Registry location override (tests); defaults to the configured path.

    Returns
    -------
    fastapi.FastAPI
    """
    if config is None:
        from src.utils.config import load_config

        config = load_config()
    api_cfg = {**_DEFAULTS, **dict(config.get("api") or {})}

    if registry_dir is None:
        from src.utils.paths import Paths

        registry_dir = Path(Paths.from_config(config).registry_dir)

    app = FastAPI(
        title="NetSentinel Inference API",
        description="Batch intrusion-detection inference over the "
                    "registry-promoted production models.",
        version="1.0",
    )
    cache = BundleCache(
        registry_dir=Path(registry_dir),
        enabled=bool(api_cfg["model_cache_enabled"]),
    )
    app.state.cache = cache
    app.state.api_config = api_cfg

    @app.exception_handler(ApiError)
    async def _api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code,
                            content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unexpected_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error: %s", exc)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error."})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok", service=SERVICE_NAME, loaded_models=cache.count
        )

    @app.get("/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        from src.registry.artifacts import load_production, load_registry

        entries = {
            e["experiment_id"]: e
            for e in load_registry(cache.registry_dir).get("entries", [])
        }
        listed = []
        for dataset, assignment in sorted(
            load_production(cache.registry_dir).items()
        ):
            entry = entries.get(assignment["experiment_id"])
            if entry is None:
                logger.warning(
                    "Production assignment %s has no registry entry; skipping.",
                    assignment["experiment_id"],
                )
                continue
            listed.append(
                ModelInfo(
                    dataset=dataset,
                    model_type=entry["model_type"],
                    experiment_id=entry["experiment_id"],
                    metrics=entry["metrics"],
                    status=entry["status"],
                )
            )
        return ModelsResponse(models=listed)

    def _predict(dataset: str, frame: "Any") -> PredictionResponse:
        bundle = cache.get(dataset, str(api_cfg["stage"]))
        payload = run_inference(
            bundle,
            frame,
            max_rows=int(api_cfg["max_rows_per_request"]),
            return_probabilities=bool(api_cfg["return_probabilities"]),
        )
        return PredictionResponse(**payload)

    @app.post("/predict/{dataset}", response_model=PredictionResponse)
    async def predict_csv(
        dataset: str, file: UploadFile = File(...)
    ) -> PredictionResponse:
        payload = await file.read()
        return _predict(dataset, frame_from_csv(payload))

    @app.post("/predict-json/{dataset}", response_model=PredictionResponse)
    async def predict_json(
        dataset: str, request: PredictJsonRequest
    ) -> PredictionResponse:
        return _predict(dataset, frame_from_rows(request.rows))

    return app


# Module-level application for ``uvicorn src.api.app:app``.
app = create_app()

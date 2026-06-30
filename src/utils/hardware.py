"""Centralized hardware / GPU detection (CLAUDE.md > GPU Acceleration Policy).

Purpose
-------
The single source of truth for compute-device detection. CUDA detection is
centralized here so no training entry point re-implements it. Every model and
trainer asks this module which device to use and how to configure each
framework for GPU execution, falling back to CPU only when a GPU path is
unavailable (never silently).

Inputs / Outputs
----------------
No inputs; reads the local machine via PyTorch / framework probes and returns
plain dicts and booleans. Heavy imports (torch) are lazy so importing this
module stays cheap.

Examples
--------
>>> from src.utils import hardware
>>> hardware.get_device()                 # doctest: +SKIP
'cuda'
>>> hardware.log_hardware_summary()        # doctest: +SKIP

Limitations
-----------
GPU-capability probes for LightGBM are best-effort (they attempt a tiny train
and cache the result). XGBoost/CatBoost capability is inferred from CUDA
availability and the installed build.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Return ``True`` when a CUDA-capable GPU is visible to PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - torch import/runtime failure
        logger.debug("CUDA detection failed: %s", exc)
        return False


def get_device(prefer_gpu: bool = True) -> str:
    """Return the execution device string.

    Parameters
    ----------
    prefer_gpu:
        When ``False``, always returns ``"cpu"`` regardless of availability
        (used to honour a config opt-out).

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"``.
    """
    if prefer_gpu and cuda_available():
        return CUDA_DEVICE
    return CPU_DEVICE


def torch_version() -> str | None:
    """Return the installed PyTorch version, or ``None`` if absent."""
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # pragma: no cover
        return None


def cuda_version() -> str | None:
    """Return the CUDA toolkit version PyTorch was built against, if any."""
    try:
        import torch

        return torch.version.cuda
    except Exception:  # pragma: no cover
        return None


def get_vram() -> dict[str, float] | None:
    """Return GPU VRAM figures in megabytes, or ``None`` when no GPU.

    Returns
    -------
    dict | None
        ``{"total_mb", "free_mb", "used_mb"}``.
    """
    if not cuda_available():
        return None
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        return {
            "total_mb": round(total / 1024**2, 1),
            "free_mb": round(free / 1024**2, 1),
            "used_mb": round((total - free) / 1024**2, 1),
        }
    except Exception as exc:  # pragma: no cover
        logger.debug("VRAM query failed: %s", exc)
        return None


def get_gpu_info() -> dict[str, Any]:
    """Return a structured description of the active compute hardware.

    Returns
    -------
    dict
        Keys: ``device``, ``cuda_available``, ``gpu_name``, ``cuda_version``,
        ``torch_version``, ``vram``.
    """
    info: dict[str, Any] = {
        "device": get_device(),
        "cuda_available": cuda_available(),
        "gpu_name": None,
        "cuda_version": cuda_version(),
        "torch_version": torch_version(),
        "vram": get_vram(),
    }
    if cuda_available():
        try:
            import torch

            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            logger.debug("GPU name query failed: %s", exc)
    return info


def log_hardware_summary() -> dict[str, Any]:
    """Log and return a one-line summary of the compute hardware.

    Every training run calls this to record the GPU model, CUDA / PyTorch
    versions, VRAM and selected device (CLAUDE.md > GPU Acceleration Policy).
    """
    info = get_gpu_info()
    if info["cuda_available"]:
        vram = info["vram"] or {}
        logger.info(
            "Hardware: GPU=%s | CUDA=%s | torch=%s | VRAM=%.0f/%.0f MB free | device=%s",
            info["gpu_name"], info["cuda_version"], info["torch_version"],
            vram.get("free_mb", 0.0), vram.get("total_mb", 0.0), info["device"],
        )
    else:
        logger.info(
            "Hardware: no CUDA GPU detected (torch=%s) | device=cpu",
            info["torch_version"],
        )
    return info


def supports_xgboost_gpu() -> bool:
    """Return whether XGBoost can train on GPU (CUDA build + visible GPU)."""
    if not cuda_available():
        return False
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def supports_catboost_gpu() -> bool:
    """Return whether CatBoost can train on GPU."""
    if not cuda_available():
        return False
    try:
        import catboost  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


@lru_cache(maxsize=1)
def supports_lightgbm_gpu() -> bool:
    """Return a best-effort indication of LightGBM GPU capability.

    This is intentionally lightweight: it reports whether a CUDA GPU is present
    and LightGBM is importable, but does NOT run a training probe (a probe would
    emit misleading tiny-dataset training logs and cannot reliably distinguish a
    CPU-only build). The authoritative decision is made in
    :class:`src.models.lightgbm_model.LightGBMModel`, which attempts GPU
    training and falls back to CPU with a warning if the build is CPU-only
    (CLAUDE.md > LightGBM: "attempt GPU first ... never silently fall back").
    """
    if not cuda_available():
        return False
    try:
        import lightgbm  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False

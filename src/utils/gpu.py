"""Device detection with automatic CPU fallback.

Purpose
-------
Centralise CUDA detection so that no CUDA-specific code leaks into business
logic (CLAUDE.md > GPU Rules: "Always detect CUDA automatically. Fallback to
CPU. No CUDA-specific code should exist outside utility modules").

Phase 1 (data engineering) does not require a GPU; this module exists so that
later training phases share one detection path.

Inputs / Outputs
----------------
No inputs; returns a device descriptor string and capability info.

Examples
--------
>>> from src.utils.gpu import get_device, device_info
>>> get_device()                 # doctest: +SKIP
'cuda'

Limitations
-----------
torch is an optional import; on a torch-less environment this reports ``cpu``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute device capabilities."""

    device: str            # "cuda" | "cpu"
    cuda_available: bool
    device_name: str | None
    total_memory_mb: float | None


def get_device(prefer_cuda: bool = True) -> str:
    """Return the preferred available device string.

    Parameters
    ----------
    prefer_cuda:
        When ``True``, return ``"cuda"`` if available, else ``"cpu"``.

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"``.
    """
    # TODO(ml-engineer): import torch lazily; return "cuda" if prefer_cuda and
    #   torch.cuda.is_available() else "cpu". Default to "cpu" when torch is
    #   not installed.
    raise NotImplementedError


def device_info(prefer_cuda: bool = True) -> DeviceInfo:
    """Return detailed information about the selected device.

    Parameters
    ----------
    prefer_cuda:
        See :func:`get_device`.

    Returns
    -------
    DeviceInfo
    """
    # TODO(ml-engineer): populate DeviceInfo from torch.cuda APIs with a safe
    #   CPU fallback when CUDA / torch is unavailable.
    raise NotImplementedError

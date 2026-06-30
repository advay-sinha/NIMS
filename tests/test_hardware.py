"""Tests for src.utils.hardware."""

from __future__ import annotations

from src.utils import hardware


def test_cuda_available_returns_bool() -> None:
    assert isinstance(hardware.cuda_available(), bool)


def test_get_device_is_cpu_or_cuda() -> None:
    assert hardware.get_device() in {"cpu", "cuda"}
    # Opting out of GPU always yields CPU.
    assert hardware.get_device(prefer_gpu=False) == "cpu"


def test_get_gpu_info_has_expected_keys() -> None:
    info = hardware.get_gpu_info()
    for key in ("device", "cuda_available", "gpu_name", "cuda_version",
                "torch_version", "vram"):
        assert key in info


def test_log_hardware_summary_returns_info() -> None:
    info = hardware.log_hardware_summary()
    assert info["device"] in {"cpu", "cuda"}


def test_gpu_support_flags_are_bool() -> None:
    assert isinstance(hardware.supports_xgboost_gpu(), bool)
    assert isinstance(hardware.supports_catboost_gpu(), bool)
    # LightGBM support is now a cheap, non-training check (no tiny-dataset probe).
    assert isinstance(hardware.supports_lightgbm_gpu(), bool)


def test_vram_consistent_with_cuda() -> None:
    vram = hardware.get_vram()
    if hardware.cuda_available():
        assert vram is not None and vram["total_mb"] > 0
    else:
        assert vram is None

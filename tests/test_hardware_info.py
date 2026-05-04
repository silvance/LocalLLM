"""Tests for hardware_info — pure recommendation logic.

We don't unit-test the actual nvidia-smi / rocm-smi shellouts (those are
side effects). The detect_hardware function returns a populated
HardwareInfo even on systems without a GPU; we test the recommendation
mapping by constructing HardwareInfo directly.
"""
from __future__ import annotations

from app.utils.hardware_info import (
    GPU,
    HardwareInfo,
    Recommendation,
    _gpu_name_to_vram,
    _guess_vendor,
    _is_real_gpu,
    recommend_models,
    to_dict,
)


def make_hw(*, vram_gb: float = 0, ram_gb: float = 16, has_gpu: bool = False) -> HardwareInfo:
    gpus: list[GPU] = []
    if has_gpu or vram_gb > 0:
        gpus = [GPU(vendor="nvidia", name="Test GPU", vram_gb=vram_gb if vram_gb > 0 else None)]
    return HardwareInfo(
        os_name="TestOS", os_release="1.0",
        cpu_model="TestCPU", cpu_cores=8,
        ram_gb=ram_gb, gpus=gpus,
    )


def test_high_vram_gets_full_lineup() -> None:
    rec = recommend_models(make_hw(vram_gb=24, ram_gb=64))
    assert rec.tier == "high"
    assert "qwen3-coder:30b" in rec.models
    assert "nomic-embed-text" in rec.models


def test_3070_class_recommends_mid_with_qwen() -> None:
    """The actual airgap target: 8 GB VRAM."""
    rec = recommend_models(make_hw(vram_gb=8, ram_gb=128))
    assert rec.tier == "mid"
    assert "qwen3-coder:30b" in rec.models
    assert "granite4:tiny-h" in rec.models


def test_low_vram_drops_qwen() -> None:
    rec = recommend_models(make_hw(vram_gb=6, ram_gb=32))
    assert rec.tier == "low-gpu"
    assert "qwen3-coder:30b" not in rec.models
    assert "granite4:latest" in rec.models


def test_cpu_only_high_ram_keeps_medium_models() -> None:
    rec = recommend_models(make_hw(ram_gb=128))
    assert rec.tier == "cpu-high"
    assert "qwen3-coder:30b" not in rec.models
    assert "gemma4:latest" in rec.models


def test_cpu_only_low_ram_smallest_only() -> None:
    rec = recommend_models(make_hw(ram_gb=8))
    assert rec.tier == "cpu-low"
    assert "qwen3-coder:30b" not in rec.models
    assert "granite4:tiny-h" in rec.models


def test_to_dict_marks_installed_vs_missing() -> None:
    hw = make_hw(vram_gb=8, ram_gb=128)
    rec = recommend_models(hw)
    payload = to_dict(hw, rec, installed=["granite4:latest", "nomic-embed-text:latest"])
    statuses = payload["recommendation"]["model_status"]
    assert statuses["granite4:latest"] == "installed"
    assert statuses["nomic-embed-text"] == "installed"  # bare-name match
    assert statuses["qwen3-coder:30b"] == "missing"
    assert "qwen3-coder:30b" in payload["recommendation"]["missing"]


def test_recommendation_includes_rationale() -> None:
    rec = recommend_models(make_hw(vram_gb=8, ram_gb=128))
    assert rec.rationale
    assert "VRAM" in rec.rationale or "RAM" in rec.rationale


# Windows GPU lookup helpers (the actual Win32_VideoController shellout
# is best-effort; tests target the pure name->vram mapping).

def test_known_amd_gpu_vram_lookup() -> None:
    assert _gpu_name_to_vram("AMD Radeon RX 7900 XT") == 20.0
    assert _gpu_name_to_vram("AMD Radeon RX 7900 XTX") == 24.0


def test_known_nvidia_gpu_vram_lookup() -> None:
    assert _gpu_name_to_vram("NVIDIA GeForce RTX 3070") == 8.0
    assert _gpu_name_to_vram("NVIDIA GeForce RTX 4090") == 24.0


def test_specific_sku_wins_over_substring() -> None:
    """RTX 4070 Ti Super must not match the shorter 'rtx 4070' entry."""
    assert _gpu_name_to_vram("NVIDIA GeForce RTX 4070 Ti SUPER") == 16.0
    assert _gpu_name_to_vram("NVIDIA GeForce RTX 4070") == 12.0


def test_unknown_gpu_returns_none() -> None:
    assert _gpu_name_to_vram("Some Future GPU 99 XL Pro") is None


def test_vendor_inference() -> None:
    assert _guess_vendor("AMD Radeon RX 7900 XT") == "amd"
    assert _guess_vendor("NVIDIA GeForce RTX 3070") == "nvidia"
    assert _guess_vendor("Intel Arc A770") == "intel"
    assert _guess_vendor("Mystery Card") == "unknown"


def test_filter_synthetic_adapters() -> None:
    assert _is_real_gpu("AMD Radeon RX 7900 XT") is True
    assert _is_real_gpu("Microsoft Basic Display Adapter") is False
    assert _is_real_gpu("Microsoft Remote Display Adapter") is False
    assert _is_real_gpu("Microsoft Hyper-V Video") is False

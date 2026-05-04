"""Local hardware detection + model recommendation.

Used by both `scripts/recommend_models.py` and the web `/hardware` page.

Detection: GPU vendor/VRAM (via nvidia-smi or rocm-smi if available),
total RAM, CPU cores. Pure-function recommendation logic maps that to a
model lineup tuned for this app's lineup of granite/gemma/qwen-coder.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class GPU:
    vendor: str
    name: str
    vram_gb: float | None
    driver: str | None = None


@dataclass
class HardwareInfo:
    os_name: str
    os_release: str
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    gpus: list[GPU] = field(default_factory=list)

    def best_vram_gb(self) -> float:
        """Largest single-GPU VRAM in GB, 0 if no GPU."""
        candidates = [g.vram_gb for g in self.gpus if g.vram_gb is not None]
        return max(candidates) if candidates else 0.0


@dataclass
class Recommendation:
    tier: str
    label: str
    models: list[str]
    rationale: str


# --------------------------------------------------------------------------
# GPU detection
# --------------------------------------------------------------------------

def _detect_nvidia() -> list[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    gpus: list[GPU] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram_mb = float(parts[1])
        except ValueError:
            vram_mb = 0
        gpus.append(GPU(
            vendor="nvidia",
            name=parts[0],
            vram_gb=round(vram_mb / 1024.0, 1) if vram_mb else None,
            driver=parts[2] if len(parts) > 2 else None,
        ))
    return gpus


def _detect_amd() -> list[GPU]:
    """Try rocm-smi (mostly Linux; rare on Windows). ROCm-on-Windows is
    hit-or-miss so the Windows fallback below picks up AMD cards via
    Win32_VideoController instead."""
    if not shutil.which("rocm-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if "GPU" in out:
        return [GPU(vendor="amd", name="AMD GPU (rocm-smi)", vram_gb=None)]
    return []


# ---------------------------------------------------------------------------
# Windows fallback: Get-CimInstance Win32_VideoController
#
# Win32_VideoController.AdapterRAM is a 32-bit field, so it caps at 4 GB
# even on a 24 GB card. We work around this with a lookup table for common
# 2020-2026 SKUs. Unknown cards fall back to AdapterRAM (which at least
# proves a GPU exists, even if the VRAM number is wrong).
# ---------------------------------------------------------------------------

_KNOWN_GPU_VRAM: dict[str, float] = {
    # NVIDIA — RTX 50 / 40 / 30 series
    "rtx 5090": 32.0, "rtx 5080": 16.0, "rtx 5070 ti": 16.0, "rtx 5070": 12.0,
    "rtx 4090": 24.0, "rtx 4080 super": 16.0, "rtx 4080": 16.0,
    "rtx 4070 ti super": 16.0, "rtx 4070 ti": 12.0, "rtx 4070 super": 12.0,
    "rtx 4070": 12.0, "rtx 4060 ti": 8.0, "rtx 4060": 8.0,
    "rtx 3090 ti": 24.0, "rtx 3090": 24.0,
    "rtx 3080 ti": 12.0, "rtx 3080": 10.0,
    "rtx 3070 ti": 8.0, "rtx 3070": 8.0,
    "rtx 3060 ti": 8.0, "rtx 3060": 12.0,
    "rtx 3050": 8.0, "rtx 2080 ti": 11.0, "rtx 2080": 8.0, "rtx 2070": 8.0,
    # NVIDIA workstation
    "rtx a6000": 48.0, "rtx a5000": 24.0, "rtx a4000": 16.0,
    "rtx 6000 ada": 48.0, "rtx 5000 ada": 32.0, "rtx 4000 ada": 20.0,
    # AMD — RX 7000 / 6000 series
    "rx 7900 xtx": 24.0, "rx 7900 xt": 20.0, "rx 7900 gre": 16.0,
    "rx 7800 xt": 16.0, "rx 7700 xt": 12.0,
    "rx 7600 xt": 16.0, "rx 7600": 8.0,
    "rx 6950 xt": 16.0, "rx 6900 xt": 16.0,
    "rx 6800 xt": 16.0, "rx 6800": 16.0,
    "rx 6750 xt": 12.0, "rx 6700 xt": 12.0,
    "rx 6650 xt": 8.0, "rx 6600 xt": 8.0, "rx 6600": 8.0,
    "rx 6500 xt": 4.0, "rx 6400": 4.0,
    # Intel Arc
    "arc a770": 16.0, "arc a750": 8.0, "arc a580": 8.0, "arc a380": 6.0,
    "arc b580": 12.0, "arc b570": 10.0,
}

_VENDOR_HINTS = (
    ("nvidia", "nvidia"), ("geforce", "nvidia"), ("rtx ", "nvidia"),
    ("gtx ", "nvidia"), ("quadro", "nvidia"), ("tesla", "nvidia"),
    ("amd", "amd"), ("radeon", "amd"), ("rx ", "amd"), ("vega", "amd"),
    ("intel", "intel"), ("arc ", "intel"), ("uhd", "intel"), ("iris", "intel"),
)


def _guess_vendor(name: str) -> str:
    n = name.lower()
    for needle, vendor in _VENDOR_HINTS:
        if needle in n:
            return vendor
    return "unknown"


def _gpu_name_to_vram(name: str) -> float | None:
    """Look up VRAM for a known SKU. Longest match wins so that
    'rtx 4070 ti super' isn't shadowed by 'rtx 4070'."""
    n = name.lower()
    matches = [(k, v) for k, v in _KNOWN_GPU_VRAM.items() if k in n]
    if not matches:
        return None
    matches.sort(key=lambda x: len(x[0]), reverse=True)
    return matches[0][1]


def _is_real_gpu(name: str) -> bool:
    """Filter out Windows synthetic / virtual display adapters."""
    n = name.lower()
    return not any(
        bad in n
        for bad in (
            "basic display",
            "remote display",
            "remotefx",
            "remote desktop",
            "virtual display",
            "microsoft hyper-v",
        )
    )


def _detect_windows_gpu() -> list[GPU]:
    if platform.system() != "Windows":
        return []
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name, AdapterRAM | "
        "ConvertTo-Json -Depth 2",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=10)
    except Exception:
        return []

    import json as _json
    try:
        data = _json.loads(out) if out.strip() else []
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    gpus: list[GPU] = []
    for entry in data:
        name = (entry.get("Name") or "").strip()
        if not name or not _is_real_gpu(name):
            continue
        # Prefer the SKU lookup; fall back to AdapterRAM (capped at 4 GB).
        vram_gb = _gpu_name_to_vram(name)
        if vram_gb is None:
            adapter_ram = entry.get("AdapterRAM") or 0
            if adapter_ram:
                vram_gb = round(adapter_ram / (1024 ** 3), 1)
        gpus.append(GPU(vendor=_guess_vendor(name), name=name, vram_gb=vram_gb))
    return gpus


def _detect_cpu() -> tuple[str, int]:
    name = platform.processor() or platform.machine() or "unknown"
    try:
        import psutil
        cores = psutil.cpu_count(logical=True) or 1
    except Exception:
        cores = 1
    return name, cores


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return 0.0


def detect_hardware() -> HardwareInfo:
    cpu_name, cores = _detect_cpu()
    raw_gpus = _detect_nvidia() + _detect_amd() + _detect_windows_gpu()

    # Deduplicate by name — Windows might surface the same card via both
    # nvidia-smi and Win32_VideoController. Prefer the entry with VRAM set.
    by_name: dict[str, GPU] = {}
    for g in raw_gpus:
        existing = by_name.get(g.name)
        if existing is None or (existing.vram_gb is None and g.vram_gb is not None):
            by_name[g.name] = g

    return HardwareInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        cpu_model=cpu_name,
        cpu_cores=cores,
        ram_gb=_ram_gb(),
        gpus=list(by_name.values()),
    )


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------

EMBEDDING_MODEL = "nomic-embed-text"


def recommend_models(hw: HardwareInfo) -> Recommendation:
    """Map hardware to a model lineup. The embedding model is always
    included so RAG works even on CPU-only setups."""
    vram = hw.best_vram_gb()
    has_gpu = bool(hw.gpus)

    if vram >= 18:
        return Recommendation(
            tier="high",
            label="High-end GPU (18+ GB VRAM)",
            models=["granite4:latest", "gemma4:latest", "qwen3-coder:30b", EMBEDDING_MODEL],
            rationale=(
                f"{vram:.0f} GB VRAM fits qwen3-coder:30b fully on GPU. "
                "Full lineup recommended."
            ),
        )

    if vram >= 12:
        return Recommendation(
            tier="upper-mid",
            label="Upper-mid GPU (12-18 GB VRAM)",
            models=["granite4:latest", "granite4:tiny-h", "gemma4:latest",
                    "qwen3-coder:30b", EMBEDDING_MODEL],
            rationale=(
                f"{vram:.0f} GB VRAM. qwen3-coder:30b will spill a few "
                "layers to CPU — usable, with granite4:tiny-h as the fast "
                f"daily driver. {hw.ram_gb:.0f} GB system RAM helps."
            ),
        )

    if vram >= 8:
        return Recommendation(
            tier="mid",
            label="Mid GPU (8-12 GB VRAM, e.g. RTX 3070)",
            models=["granite4:latest", "granite4:tiny-h", "gemma4:latest",
                    "qwen3-coder:30b", EMBEDDING_MODEL],
            rationale=(
                f"{vram:.0f} GB VRAM. qwen3-coder:30b partial-offloads to "
                f"system RAM ({hw.ram_gb:.0f} GB available); MoE keeps it at "
                "~4-8 tok/s. granite4 / gemma4 fit fully on GPU."
            ),
        )

    if vram >= 4:
        return Recommendation(
            tier="low-gpu",
            label="Low-VRAM GPU (4-8 GB)",
            models=["granite4:latest", "granite4:tiny-h", "gemma4:latest",
                    EMBEDDING_MODEL],
            rationale=(
                f"{vram:.0f} GB VRAM. qwen3-coder:30b would be very slow "
                "(deep CPU offload); skipping. Stick with the smaller models."
            ),
        )

    if vram > 0 and has_gpu:
        return Recommendation(
            tier="tiny-gpu",
            label="Very small GPU",
            models=["granite4:latest", "granite4:tiny-h", EMBEDDING_MODEL],
            rationale=(
                f"{vram:.1f} GB VRAM. Only the smallest models fit; "
                "consider running CPU-only for anything bigger."
            ),
        )

    if hw.ram_gb >= 64:
        return Recommendation(
            tier="cpu-high",
            label="CPU-only, lots of RAM",
            models=["granite4:latest", "granite4:tiny-h", "gemma4:latest",
                    EMBEDDING_MODEL],
            rationale=(
                f"No GPU detected. {hw.ram_gb:.0f} GB RAM is enough for "
                "small/medium models on CPU. Inference will be slow but "
                "functional."
            ),
        )

    return Recommendation(
        tier="cpu-low",
        label="CPU-only, modest RAM",
        models=["granite4:latest", "granite4:tiny-h", EMBEDDING_MODEL],
        rationale=(
            f"No GPU and {hw.ram_gb:.0f} GB RAM — only the smallest models "
            "are practical. Expect single-digit tokens/sec on CPU."
        ),
    )


def to_dict(hw: HardwareInfo, rec: Recommendation, installed: list[str] | None = None) -> dict:
    """Serialise everything for the JSON API / template context."""
    installed_set = set(installed or [])

    def status(model: str) -> str:
        # Match either "name:tag" or bare "name" against installed.
        if model in installed_set:
            return "installed"
        if ":" in model:
            base = model.split(":")[0]
            if base in installed_set or any(i.startswith(base + ":") for i in installed_set):
                return "installed"
        else:
            if any(i == model or i.startswith(model + ":") for i in installed_set):
                return "installed"
        return "missing"

    return {
        "hardware": {
            **asdict(hw),
            "best_vram_gb": hw.best_vram_gb(),
            "has_gpu": bool(hw.gpus),
        },
        "recommendation": {
            **asdict(rec),
            "model_status": {m: status(m) for m in rec.models},
            "missing": [m for m in rec.models if status(m) == "missing"],
        },
    }

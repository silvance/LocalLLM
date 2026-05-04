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
    """Try rocm-smi (Linux) or fall back to silent no-op. ROCm-on-Windows
    is hit-or-miss; we don't try to interrogate it here. The user's actual
    runtime check is whether Ollama can reach the GPU, which we don't
    duplicate."""
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
    # rocm-smi --csv format varies between versions; just return a single
    # generic AMD GPU entry without VRAM if we can't parse cleanly. Better
    # than missing the GPU entirely.
    if "GPU" in out:
        return [GPU(vendor="amd", name="AMD GPU (rocm-smi)", vram_gb=None)]
    return []


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
    return HardwareInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        cpu_model=cpu_name,
        cpu_cores=cores,
        ram_gb=_ram_gb(),
        gpus=_detect_nvidia() + _detect_amd(),
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

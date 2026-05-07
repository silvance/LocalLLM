"""Tests for scripts/smart_install.py — pure-function pieces.

We don't unit-test the hardware-detection shellouts (that's covered in
test_hardware_info.py); we test the pieces smart_install layers on top:
the env-file updater and the tier→default-model mapping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# scripts/ is its own importable namespace alongside app/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.utils.hardware_info import (  # noqa: E402
    GPU,
    HardwareInfo,
    recommend_models,
)
from scripts.smart_install import (  # noqa: E402
    TIER_DEFAULT_MODEL,
    pick_default_model,
    update_env_settings,
    write_env,
)


# ---------------------------------------------------------------------------
# update_env_settings — preserves comments, replaces existing keys, appends new
# ---------------------------------------------------------------------------

def test_replaces_existing_key() -> None:
    lines = [
        "# top comment",
        "DEFAULT_MODEL=granite",
        "RAG_ENABLED=false",
    ]
    out = update_env_settings(lines, {"DEFAULT_MODEL": "auto"})
    assert out == [
        "# top comment",
        "DEFAULT_MODEL=auto",
        "RAG_ENABLED=false",
    ]


def test_appends_missing_key() -> None:
    lines = ["RAG_ENABLED=false"]
    out = update_env_settings(lines, {"DEFAULT_MODEL": "auto"})
    assert out == ["RAG_ENABLED=false", "DEFAULT_MODEL=auto"]


def test_preserves_blank_lines_and_comments() -> None:
    lines = [
        "# header",
        "",
        "DEFAULT_MODEL=granite",
        "",
        "# trailing",
    ]
    out = update_env_settings(lines, {"DEFAULT_MODEL": "qwen"})
    assert out == ["# header", "", "DEFAULT_MODEL=qwen", "", "# trailing"]


def test_multi_key_update() -> None:
    lines = ["DEFAULT_MODEL=a"]
    out = update_env_settings(lines, {"DEFAULT_MODEL": "b", "RAG_ENABLED": "true"})
    assert "DEFAULT_MODEL=b" in out
    assert "RAG_ENABLED=true" in out


def test_does_not_touch_lines_with_no_equals() -> None:
    lines = ["just a string", "DEFAULT_MODEL=x"]
    out = update_env_settings(lines, {"DEFAULT_MODEL": "y"})
    # First line is preserved; second is updated.
    assert out == ["just a string", "DEFAULT_MODEL=y"]


# ---------------------------------------------------------------------------
# write_env — file-IO behaviour, .env.example seeding
# ---------------------------------------------------------------------------

def test_writes_new_file_when_no_env_or_example(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    write_env(target, {"DEFAULT_MODEL": "auto"})
    contents = target.read_text(encoding="utf-8")
    assert "DEFAULT_MODEL=auto" in contents


def test_seeds_from_env_example_when_env_missing(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text(
        "# example\nRAG_ENABLED=false\nDEFAULT_MODEL=granite\n",
        encoding="utf-8",
    )
    target = tmp_path / ".env"
    write_env(target, {"DEFAULT_MODEL": "qwen"})
    contents = target.read_text(encoding="utf-8")
    assert "DEFAULT_MODEL=qwen" in contents
    assert "RAG_ENABLED=false" in contents
    assert "# example" in contents


def test_updates_existing_env_in_place(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text(
        "DEFAULT_MODEL=qwen\nRAG_ENABLED=true\n",
        encoding="utf-8",
    )
    write_env(target, {"DEFAULT_MODEL": "auto"})
    contents = target.read_text(encoding="utf-8")
    assert "DEFAULT_MODEL=auto" in contents
    assert "RAG_ENABLED=true" in contents
    assert "DEFAULT_MODEL=qwen" not in contents


# ---------------------------------------------------------------------------
# Tier mapping — every recommendation tier should yield a default
# ---------------------------------------------------------------------------

def _hw(*, vram_gb: float = 0, ram_gb: float = 16) -> HardwareInfo:
    gpus = []
    if vram_gb > 0:
        gpus = [GPU(vendor="nvidia", name="Test GPU", vram_gb=vram_gb)]
    return HardwareInfo(
        os_name="TestOS", os_release="1.0",
        cpu_model="TestCPU", cpu_cores=8,
        ram_gb=ram_gb, gpus=gpus,
    )


@pytest.mark.parametrize("vram_gb,ram_gb,expected_tier,expected_default", [
    (24, 64,  "high",      "auto"),
    (16, 64,  "upper-mid", "auto"),
    (8,  128, "mid",       "auto"),       # 3070-class — qwen partial-offload OK
    (6,  32,  "low-gpu",   "gemma"),
    (2,  16,  "tiny-gpu",  "granite"),
    (0,  128, "cpu-high",  "granite"),
    (0,  16,  "cpu-low",   "granite"),
])
def test_tier_picks_appropriate_default(
    vram_gb: float, ram_gb: float, expected_tier: str, expected_default: str
) -> None:
    hw = _hw(vram_gb=vram_gb, ram_gb=ram_gb)
    rec = recommend_models(hw)
    assert rec.tier == expected_tier
    assert pick_default_model(rec) == expected_default


def test_tier_default_model_covers_every_recommendation_tier() -> None:
    """Belt-and-suspenders: any tier the recommendation logic produces must
    have a matching default-model entry, otherwise pick_default_model
    falls through to the safe 'granite' fallback silently."""
    expected_tiers = {
        "high", "upper-mid", "mid", "low-gpu", "tiny-gpu", "cpu-high", "cpu-low",
    }
    assert expected_tiers <= TIER_DEFAULT_MODEL.keys()

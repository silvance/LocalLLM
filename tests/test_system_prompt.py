"""Tests for app/utils/system_prompt.py — the centralized prompt
assembly that prepends the always-on anti-hallucination baseline."""
from __future__ import annotations

import pytest

from app.utils.system_prompt import BASELINE, baseline_enabled, compose


# ---------------------------------------------------------------------------
# baseline_enabled — env-var toggle
# ---------------------------------------------------------------------------

def test_baseline_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALLLM_DISABLE_BASELINE_PROMPT", raising=False)
    assert baseline_enabled() is True


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_baseline_disabled_when_env_truthy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("LOCALLLM_DISABLE_BASELINE_PROMPT", val)
    assert baseline_enabled() is False


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "  "])
def test_baseline_enabled_when_env_falsy(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("LOCALLLM_DISABLE_BASELINE_PROMPT", val)
    assert baseline_enabled() is True


# ---------------------------------------------------------------------------
# compose — layering behaviour
# ---------------------------------------------------------------------------

def test_compose_includes_baseline_with_no_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALLLM_DISABLE_BASELINE_PROMPT", raising=False)
    out = compose()
    assert out == BASELINE.strip()


def test_compose_baseline_first_then_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALLLM_DISABLE_BASELINE_PROMPT", raising=False)
    out = compose("Task: do X precisely.", "User: be concise.")
    assert out.startswith(BASELINE.strip())
    assert "Task: do X precisely." in out
    assert "User: be concise." in out
    # Order: baseline → task-specific → user
    baseline_idx = out.index(BASELINE.strip())
    task_idx = out.index("Task: do X precisely.")
    user_idx = out.index("User: be concise.")
    assert baseline_idx < task_idx < user_idx


def test_compose_skips_empty_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALLLM_DISABLE_BASELINE_PROMPT", raising=False)
    out = compose("", "  ", "real layer", "")
    assert out.startswith(BASELINE.strip())
    assert out.endswith("real layer")
    # Only one blank-line separator between baseline and the real layer
    assert out.count("\n\n") == 1


def test_compose_returns_empty_when_disabled_and_no_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller treats empty result as 'don't add a system message at all'."""
    monkeypatch.setenv("LOCALLLM_DISABLE_BASELINE_PROMPT", "1")
    assert compose() == ""
    assert compose("", "  ") == ""


def test_compose_disabled_uses_only_user_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALLLM_DISABLE_BASELINE_PROMPT", "1")
    out = compose("custom prompt only")
    assert out == "custom prompt only"
    assert "Calibration" not in out


def test_baseline_mentions_no_fabrication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: the actual anti-hallucination words must stay
    in the baseline. Without them this whole feature is decorative."""
    text = BASELINE.lower()
    assert "invent" in text or "fabricat" in text
    assert "verify" in text or "confident" in text


def test_compose_trims_user_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALLLM_DISABLE_BASELINE_PROMPT", raising=False)
    padded = "\n\n  Be brief.\n  "
    out = compose(padded)
    assert out.endswith("Be brief.")
    assert "  Be brief." not in out  # leading whitespace stripped


def test_default_system_prompt_does_not_force_code_for_normal_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEFAULT_SYSTEM_PROMPT", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    prompt = get_settings().default_system_prompt.lower()
    assert "for normal questions" in prompt
    assert "do not turn those answers into code" in prompt
    assert "only produce code when the user asks" in prompt

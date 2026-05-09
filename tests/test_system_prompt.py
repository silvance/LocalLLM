"""Tests for app/utils/system_prompt.py — the centralized prompt
assembly that prepends the always-on anti-hallucination baseline."""
from __future__ import annotations

import pytest

from app.utils.system_prompt import (
    BASELINE,
    REVIEW_WRITER_GUIDANCE,
    baseline_enabled,
    compose,
    compose_for_writer,
)


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


# ---------------------------------------------------------------------------
# REVIEW_WRITER_GUIDANCE + compose_for_writer
# ---------------------------------------------------------------------------

def test_writer_guidance_names_busy_loop_anti_pattern() -> None:
    """The named anti-pattern from the user's BLE/Wi-Fi sniffer
    screenshot must be in the writer guidance verbatim — small models
    route on concrete patterns, not abstract guidance, so the literal
    `while True: pass` string must appear so the writer can see what
    NOT to write."""
    g = REVIEW_WRITER_GUIDANCE
    assert "while True: pass" in g
    # Suggested replacements — the writer needs to know what TO do,
    # not just what to avoid.
    assert "signal.pause" in g or "Event().wait" in g or "time.sleep" in g


def test_writer_guidance_names_conditional_import_anti_pattern() -> None:
    """Second anti-pattern from the screenshot: imports inside
    `if __name__ == "__main__":` whose names are referenced from
    module-level functions. Must mention the structural rule with
    examples concrete enough for a small model to recognize."""
    g = REVIEW_WRITER_GUIDANCE
    assert '__main__' in g
    assert "top of the file" in g.lower() or "module top" in g.lower()
    # NameError is the symptom — naming it makes the rule concrete.
    assert "NameError" in g


def test_writer_guidance_names_subprocess_wait_anti_pattern() -> None:
    """`.terminate()` without `.wait()` leaks zombies — was visible
    in the screenshot's KeyboardInterrupt handler."""
    g = REVIEW_WRITER_GUIDANCE
    assert ".terminate()" in g
    assert ".wait()" in g


def test_writer_guidance_names_fake_cli_flag_anti_pattern() -> None:
    """The screenshot's `btmon --output` is a hallucinated flag
    (real flag is `-w`). The guidance must explicitly call out the
    "verify CLI flags" rule and use a concrete example so the
    pattern lands."""
    g = REVIEW_WRITER_GUIDANCE
    assert "btmon" in g  # concrete example
    assert "-w" in g  # the real flag
    # Generalization to other tools — so the model doesn't think
    # btmon is the only one to be careful about.
    assert "iw" in g or "tshark" in g or "nmcli" in g


def test_compose_for_writer_stacks_baseline_guidance_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compose_for_writer order: BASELINE → REVIEW_WRITER_GUIDANCE →
    user prompt. User prompt is last so a specifically-instructed
    user override wins."""
    monkeypatch.delenv("LOCALLLM_DISABLE_BASELINE_PROMPT", raising=False)
    out = compose_for_writer("Be terse.")
    # All three layers present.
    assert "Calibration" in out  # from BASELINE
    assert "Code-quality requirements" in out  # from REVIEW_WRITER_GUIDANCE
    assert "Be terse." in out
    # Order
    i_base = out.index("Calibration")
    i_guide = out.index("Code-quality requirements")
    i_user = out.index("Be terse.")
    assert i_base < i_guide < i_user


def test_compose_for_writer_honors_baseline_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same env-var that disables BASELINE in compose() must
    also disable it here — otherwise debugging the writer chain
    while comparing to chat would give different results."""
    monkeypatch.setenv("LOCALLLM_DISABLE_BASELINE_PROMPT", "1")
    out = compose_for_writer("Be terse.")
    assert "Calibration" not in out
    # But the writer guidance is INDEPENDENT of the baseline toggle —
    # disabling the anti-hallucination guard doesn't disable the
    # code-quality rules. Different concerns.
    assert "Code-quality requirements" in out


def test_compose_for_writer_works_with_no_user_prompt() -> None:
    """User leaves the system-prompt textarea empty — compose_for_writer
    must still produce the BASELINE + GUIDANCE pair."""
    out = compose_for_writer("")
    assert "Calibration" in out
    assert "Code-quality requirements" in out


# ---------------------------------------------------------------------------
# End-to-end: the writer in /review actually receives the guidance
# ---------------------------------------------------------------------------

def test_writer_guidance_reaches_review_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """Wiring smoke check: when the review pipeline builds its
    base_messages(), the system message it emits must contain the
    REVIEW_WRITER_GUIDANCE layer. Otherwise we added the guidance
    but it never reaches the model — silent regression."""
    pytest.importorskip("chromadb")
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    import importlib

    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(tmp_path))
    from app import config
    config.get_settings.cache_clear()
    from app.web import app as web_app
    importlib.reload(web_app)
    from app.utils.review_storage import ReviewStorage
    web_app.review_storage = ReviewStorage(tmp_path / "reviews")

    captured: dict = {}

    def _capture(*, job, review_id, prompt, writer_model, reviewer_model,
                 fallback_writer_model, rounds, system_prompt, temperature,
                 max_tokens, num_ctx, apply_prior_lessons, loop):
        # base_messages() (line 1020 of app.py) calls
        # compose_writer_system_prompt(system_prompt) — re-run that
        # exact path here so we test what the writer actually receives.
        captured["composed"] = web_app.compose_writer_system_prompt(system_prompt)

    monkeypatch.setattr(web_app, "_start_review_thread", _capture)

    with TestClient(web_app.app) as client:
        r = client.post("/api/review", json={
            "prompt": "build a TCP echo server",
            "writer_model": "qwen",
            "reviewer_model": "gemma",
            "rounds": 2,
            "system_prompt": "Use type hints.",
        })
        assert r.status_code == 200, r.text

    composed = captured["composed"]
    assert "Calibration" in composed              # BASELINE
    assert "Code-quality requirements" in composed  # REVIEW_WRITER_GUIDANCE
    assert "Use type hints." in composed          # user

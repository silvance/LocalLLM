"""Tests for app/services/ollama_models.py — model-name validation
and the streaming wrapper around Ollama's /api/pull endpoint.

Network is mocked; we never actually contact a daemon."""
from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from app.services.ollama_models import (
    pull_model,
    validate_model_name,
)


# ---------------------------------------------------------------------------
# validate_model_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "granite4",
    "granite4:latest",
    "qwen3-coder:30b",
    "qwen2.5-coder:32b",
    "library/llama3:8b",
    "registry.example.com/team/model:tag",
])
def test_valid_model_names(name: str) -> None:
    assert validate_model_name(name) == name


@pytest.mark.parametrize("name", [
    "",
    "   ",
    "model with spaces",
    "model;rm -rf /",
    "model && curl evil",
    "model | sh",
    "model$variable",
    "model`backtick`",
    "<model>",
    "..",
])
def test_invalid_model_names_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        validate_model_name(name)


def test_strips_surrounding_whitespace() -> None:
    assert validate_model_name("  granite4:latest  ") == "granite4:latest"


def test_rejects_overlong_names() -> None:
    with pytest.raises(ValueError):
        validate_model_name("a" * 250)


# ---------------------------------------------------------------------------
# pull_model — network mocked
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal stand-in for urlopen's HTTPResponse iterator."""
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
    def __iter__(self):
        for line in self._lines:
            yield (line + "\n").encode("utf-8")
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


def test_pull_model_yields_each_ndjson_event() -> None:
    events = [
        json.dumps({"status": "pulling manifest"}),
        json.dumps({"status": "downloading", "digest": "sha256:abc",
                    "total": 100, "completed": 25}),
        json.dumps({"status": "downloading", "digest": "sha256:abc",
                    "total": 100, "completed": 100}),
        json.dumps({"status": "success"}),
    ]
    with patch("app.services.ollama_models.urllib.request.urlopen",
               return_value=_FakeResponse(events)):
        out = list(pull_model("granite4", base_url="http://x"))
    assert len(out) == 4
    assert out[0]["status"] == "pulling manifest"
    assert out[1]["completed"] == 25
    assert out[-1]["status"] == "success"


def test_pull_model_skips_blank_lines() -> None:
    events = ["", json.dumps({"status": "x"}), "  "]
    with patch("app.services.ollama_models.urllib.request.urlopen",
               return_value=_FakeResponse(events)):
        out = list(pull_model("granite4", base_url="http://x"))
    assert len(out) == 1
    assert out[0]["status"] == "x"


def test_pull_model_yields_raw_string_for_unparseable_lines() -> None:
    events = ["not-json-but-still-meaningful"]
    with patch("app.services.ollama_models.urllib.request.urlopen",
               return_value=_FakeResponse(events)):
        out = list(pull_model("granite4", base_url="http://x"))
    assert out == [{"status": "not-json-but-still-meaningful"}]


def test_pull_model_returns_error_on_http_error() -> None:
    err = urllib.error.HTTPError(
        url="http://x/api/pull", code=404,
        msg="Not Found", hdrs=None,
        fp=io.BytesIO(json.dumps({"error": "model not found"}).encode("utf-8")),
    )
    with patch("app.services.ollama_models.urllib.request.urlopen", side_effect=err):
        out = list(pull_model("ghost", base_url="http://x"))
    assert len(out) == 1
    assert "model not found" in out[0]["error"]
    assert out[0]["http_status"] == 404


def test_pull_model_validates_before_dialing() -> None:
    """Bad model names must raise BEFORE we touch the network — defense
    in depth against any caller that forgets to validate."""
    with pytest.raises(ValueError):
        list(pull_model("evil; rm -rf /", base_url="http://x"))


def test_pull_model_returns_transport_error_on_connection_failure() -> None:
    err = urllib.error.URLError("connection refused")
    with patch("app.services.ollama_models.urllib.request.urlopen", side_effect=err):
        out = list(pull_model("granite4", base_url="http://x"))
    assert len(out) == 1
    assert "transport error" in out[0]["error"]

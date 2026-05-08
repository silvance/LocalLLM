"""Tests for the lazy adapter cache in ChatService.

Covers the change that lets the dropdowns surface every installed
Ollama model instead of just the three preset slots:
- get_adapter("granite") → preset slot, uses settings.model_map
- get_adapter("qwen2.5-coder:32b") → raw model name, no model_map
  entry needed
- repeated lookups return the cached instance
"""
from __future__ import annotations

import pytest

chromadb = pytest.importorskip("chromadb")
ollama = pytest.importorskip("ollama")

from app.services.chat_service import ChatService  # noqa: E402


def test_preset_slot_returns_eager_adapter() -> None:
    svc = ChatService()
    a = svc.get_adapter("granite")
    assert a is svc.adapters["granite"]
    # Slot resolves through model_map → real Ollama name like "granite4"
    assert a.model_name != "granite"


def test_raw_model_name_lazily_creates_adapter() -> None:
    svc = ChatService()
    raw = "qwen2.5-coder:32b"
    assert raw not in svc.adapters
    a = svc.get_adapter(raw)
    assert raw in svc.adapters
    # Raw names are passed through to OllamaAdapter as-is.
    assert a.model_name == raw
    # Subsequent lookup is cached.
    assert svc.get_adapter(raw) is a


def test_unknown_model_falls_back_to_raw_name_via_adapter() -> None:
    """OllamaAdapter is the layer that resolves model_key → model_name.
    Names not in model_map become the literal Ollama tag — caller is
    responsible for the model actually being installed."""
    svc = ChatService()
    a = svc.get_adapter("definitely-not-a-real-model:99b")
    assert a.model_name == "definitely-not-a-real-model:99b"

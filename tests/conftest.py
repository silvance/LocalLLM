"""Shared pytest setup for the LocalLLM test suite.

Adds the repo root to ``sys.path`` once so individual test files don't
have to repeat the boilerplate. Existing tests that still do their own
``sys.path.insert`` keep working — the conftest insert is idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

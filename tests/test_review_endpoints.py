"""Endpoint tests for the /review page (writer ↔ reviewer adversarial loop).

Skipped in the minimal CI test job (chromadb not installed). Runs locally
or on the full-dep build.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest

chromadb = pytest.importorskip("chromadb")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(tmp_path))
    import importlib

    from app import config
    config.get_settings.cache_clear()

    from app.web import app as web_app
    importlib.reload(web_app)
    # Re-point on-disk storage at tmp_path so the test doesn't trample
    # the dev box's data/reviews/ tree.
    from app.utils.review_storage import ReviewStorage
    web_app.review_storage = ReviewStorage(tmp_path / "reviews")

    with TestClient(web_app.app) as c:
        yield c


def test_review_redirects_to_in_flight_detail_on_navigation(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """User starts a review, navigates away, comes back to /review.
    The route should detect the running review and redirect to its
    detail page so the JS can re-subscribe to its SSE stream."""
    from app.web import app as web_app

    # Stub the writer/reviewer thread so the review never finishes —
    # simulates the user navigating away mid-loop.
    started = {"flag": False}

    def _hang(*_args, **_kwargs):
        started["flag"] = True
        time.sleep(60)

    monkeypatch.setattr(web_app, "_start_review_thread", _hang)

    r = client.post("/api/review", json={
        "prompt": "review resume probe",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        "rounds": 2,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]
    time.sleep(0.05)
    assert started["flag"] is True

    # /review (no id) should now redirect to /review/<id> for the
    # in-flight review.
    r2 = client.get("/review", follow_redirects=False)
    assert r2.status_code in (302, 303)
    assert r2.headers["location"].endswith(f"/review/{review_id}")

    # The detail page should embed the active_job_id so JS can subscribe.
    detail = client.get(f"/review/{review_id}")
    assert detail.status_code == 200
    assert "activeJobId" in detail.text
    assert "review resume probe" in detail.text


def test_review_no_redirect_when_no_in_flight(client: TestClient) -> None:
    """No running review → /review serves the empty form, not a redirect."""
    r = client.get("/review", follow_redirects=False)
    assert r.status_code == 200
    # active_job_id should be null in the embedded JS hook
    assert "activeJobId" in r.text
    assert "null" in r.text  # the tojson serialization of None

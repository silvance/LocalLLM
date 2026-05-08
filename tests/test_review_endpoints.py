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


# ---------------------------------------------------------------------------
# Static-analysis gates fence the reviewer from broken code
# ---------------------------------------------------------------------------

def _stub_chat_service(monkeypatch: pytest.MonkeyPatch, scripted_outputs: list[str]) -> list[dict]:
    """Replace chat_service.stream_chat with a deterministic generator.
    Returns the call log so tests can assert what was sent to the model."""
    from app.web import app as web_app

    calls: list[dict] = []
    pending = list(scripted_outputs)

    class _FakeChunk:
        def __init__(self, content: str) -> None:
            self.content = content
            self.done = False
            self.prompt_tokens = None
            self.completion_tokens = None
            self.eval_duration_ns = None
            self.total_duration_ns = None

    def _fake_stream_chat(*, request, selection, use_rag=False, **_kw):
        calls.append({"selection": selection, "messages": list(request.messages)})
        text = pending.pop(0) if pending else ""

        class _Execution:
            stream = iter([_FakeChunk(text)])
            selected_model = selection
            routing_decision = None
            retrievals: list = []

        return _Execution()

    monkeypatch.setattr(web_app.chat_service, "stream_chat", _fake_stream_chat)
    return calls


def test_review_skips_reviewer_when_gates_fail(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Writer produces code with an undefined name. Gates 1-2 should
    catch it; the reviewer model should NEVER be called for round 1.
    Instead a `gates` section is persisted with the lint findings."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    # Writer's only output: code that lints clean syntactically but
    # references an undefined name. Reviewer should never be reached.
    broken = (
        "Here is your tool:\n\n"
        "```python\n"
        "def discover():\n"
        "    return ghost_var  # undefined → pyflakes catches it\n"
        "```\n"
    )
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[broken])

    r = client.post("/api/review", json={
        "prompt": "build a discovery tool",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        "rounds": 2,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    # Wait for the daemon thread to finish (rounds=2 means writer + 1
    # follow-up — the follow-up should be a `gates` section, not reviewer).
    deadline = time.monotonic() + 5.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done", f"review never finished (status={saved and saved.status})"

    roles = [s.role for s in saved.sections]
    # Round 0: writer. Round 1: gates (NOT reviewer).
    assert roles == ["writer", "gates"], f"unexpected role sequence: {roles}"
    assert "ghost_var" in saved.sections[1].text, "gate feedback should mention the undefined name"
    # Reviewer model selection should never have been invoked.
    selections = [c["selection"] for c in calls]
    assert "gemma" not in selections, f"reviewer should not have been called when gates failed; selections={selections}"


def test_review_stops_early_when_reviewer_approves(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Reviewer's JSON verdict says `approve` → orchestrator should
    break out of the loop instead of running another pointless
    writer round."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    clean = "```python\ndef add(a, b):\n    return a + b\n```\n"
    approve_review = (
        "Looks good.\n\n```json\n"
        '{"pass": true, "blockers": [], "safe_to_rebuild": true, '
        '"recommended_next_action": "approve"}\n```\n'
    )
    # rounds=4 budget would normally be writer/reviewer/writer/reviewer.
    # With early-break on approve we expect just writer/reviewer.
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[clean, approve_review])

    r = client.post("/api/review", json={
        "prompt": "implement add",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        "rounds": 4,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    deadline = time.monotonic() + 5.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done"
    roles = [s.role for s in saved.sections]
    assert roles == ["writer", "reviewer"], f"expected early break, got {roles}"
    # Only the writer + reviewer model calls should have happened
    # (not a second writer rebuild).
    selections = [c["selection"] for c in calls]
    assert selections.count("qwen") == 1, f"writer ran more than once: {selections}"


def test_review_engages_fallback_when_reviewer_recommends(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Reviewer's verdict has `recommended_next_action=rebuild_different_writer`
    → fallback engages on the next round, even though gates passed
    and the threshold isn't met."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    # Code that PASSES gates (clean python) but reviewer flags as a
    # domain mistake.
    code = (
        "```python\n"
        "import socket\n"
        "def sniff_ble():\n"
        "    s = socket.socket(socket.AF_INET, socket.SOCK_RAW)\n"
        "    return s.recv(1024)\n"
        "```\n"
    )
    domain_blocker = (
        "BLE can't be sniffed via raw IPv4 sockets — needs HCI on Linux.\n\n"
        "```json\n"
        '{"pass": false, "blockers": ["wrong stack: BLE != AF_INET"], '
        '"safe_to_rebuild": true, '
        '"recommended_next_action": "rebuild_different_writer"}\n```\n'
    )
    fixed = "```python\nimport bluetooth\n\ndef sniff_ble():\n    pass\n```\n"
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[code, domain_blocker, fixed])

    r = client.post("/api/review", json={
        "prompt": "sniff BLE packets",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        "fallback_writer_model": "granite",
        "rounds": 4,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    deadline = time.monotonic() + 5.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done"

    roles = [s.role for s in saved.sections]
    assert "fallback" in roles, f"fallback notice not emitted; roles={roles}"
    # The writer call AFTER the fallback notice should target granite.
    fallback_idx = next(i for i, c in enumerate(calls) if c["selection"] == "granite")
    assert fallback_idx > 0, "granite writer never called — fallback didn't engage"


def test_fallback_writer_engages_after_repeated_gate_failures(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Primary writer fails gate twice in a row → orchestrator emits a
    `fallback` notice and uses the operator-specified fallback model
    for the next rebuild round."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    broken = (
        "Try this:\n\n"
        "```python\n"
        "def discover():\n"
        "    return ghost  # undefined\n"
        "```\n"
    )
    fixed = (
        "OK now:\n\n"
        "```python\n"
        "def discover():\n"
        "    return 1\n"
        "```\n"
    )
    # 4 rounds budget: writer / gates / writer / gates / writer / gates ...
    # Round 0 writer: broken
    # Round 1 gates: fail (consecutive=1)
    # Round 2 writer: broken again (consecutive_gate_fails still 1, no swap yet —
    #                  threshold check happens AFTER the gate run, so we need
    #                  another gate fail before the swap fires)
    # Round 3 gates: fail (consecutive=2 → swap on next round)
    # Round 4 writer: fallback active (returns clean code)
    # Round 5 gates: pass → reviewer (we don't have rounds=6 budget though)
    scripted = [broken, broken, fixed]
    calls = _stub_chat_service(monkeypatch, scripted_outputs=scripted)

    r = client.post("/api/review", json={
        "prompt": "build a discoverer",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        "fallback_writer_model": "granite",
        "rounds": 5,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    deadline = time.monotonic() + 5.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done"

    roles = [s.role for s in saved.sections]
    # Must contain a fallback notice somewhere after at least 2 gate fails
    assert "fallback" in roles, f"no fallback section emitted; roles={roles}"
    fallback_idx = roles.index("fallback")
    gates_before = [r for r in roles[:fallback_idx] if r == "gates"]
    assert len(gates_before) >= 2, "fallback should fire only after ≥2 gate fails"

    # Writer rounds AFTER the fallback notice must use the fallback model.
    selections_after_fallback = [c["selection"] for c in calls[len([c for c in calls if c["selection"] == "qwen"]):]]  # noqa: E501
    assert "granite" in [c["selection"] for c in calls], (
        f"fallback model never invoked; selections={[c['selection'] for c in calls]}"
    )


def test_review_runs_reviewer_when_gates_pass(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Clean code → all gates pass → reviewer runs as before."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    clean = (
        "Here you go:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
    )
    review = "Looks fine. Maybe add type hints."
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[clean, review])

    r = client.post("/api/review", json={
        "prompt": "implement add()",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        "rounds": 2,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    deadline = time.monotonic() + 5.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done"
    assert [s.role for s in saved.sections] == ["writer", "reviewer"]
    assert "gemma" in [c["selection"] for c in calls]

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


def test_fallback_writer_gets_fresh_start_prompt(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """When the fallback writer engages, its first prompt must be a
    fresh-start kickoff (no broken-code-as-starting-point) and must
    contain the accumulated gate failures so it knows what NOT to do.
    Captured directly from the chat_service stub — no LLM round-trip."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    broken = (
        "Try this:\n\n"
        "```python\n"
        "from scapy.all import *\n"  # F403 — wildcard
        "def discover():\n"
        "    return ghost  # F821 — undefined name\n"
        "```\n"
    )
    fixed = "```python\ndef discover():\n    return 1\n```\n"
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[broken, broken, fixed])

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

    # The third writer call (after two gate fails + fallback engages)
    # is the fallback. Its prompt must be the kickoff, not the regular
    # revise template.
    granite_calls = [c for c in calls if c["selection"] == "granite"]
    assert granite_calls, "fallback (granite) was never invoked"
    fallback_msgs = granite_calls[0]["messages"]
    # Kickoff format: single user message, no broken-code preamble.
    user_content = "\n".join(m.content for m in fallback_msgs if m.role == "user")
    assert "fallback builder" in user_content.lower()
    assert "from scratch" in user_content.lower()
    # Must NOT include the broken code as a starting point — that's
    # the bug we're fixing. (`ghost` shows up in the gate-failure
    # echo, but `def discover():` is the actual function body the
    # fallback would have copied if we used the regular revise prompt.)
    assert "def discover" not in user_content
    assert "Your previous code:\n" not in user_content
    # Kickoff must restate the original task, not "fix what came before".
    assert "Original user request" in user_content
    # Must list at least one gate failure the previous writer kept
    # hitting so the fallback knows what to avoid.
    assert "import *" in user_content or "wildcard" in user_content.lower()


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


def test_domain_failure_triggers_swap_on_first_reviewer_blocker(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """A reviewer round that returns ANY blockers should immediately
    engage the fallback (default max_same_writer_domain_failures=1).
    Distinct from the existing `rebuild_different_writer` path —
    the reviewer doesn't have to opt in; just having blockers is
    enough."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    clean_code = "```python\ndef add(a, b):\n    return a + b\n```\n"
    review_with_blockers = (
        "Bug: missing edge case.\n\n```json\n"
        '{"pass": false, "blockers": ["does not handle negative numbers"], '
        '"safe_to_rebuild": true, "recommended_next_action": "rebuild_same_writer"}\n```'
    )
    fixed = "```python\ndef add(a, b):\n    return a + b if a or b else 0\n```\n"
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[clean_code, review_with_blockers, fixed])

    r = client.post("/api/review", json={
        "prompt": "implement add",
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
    # writer / reviewer / fallback notice / writer (granite)
    assert "fallback" in roles, f"domain swap not engaged; roles={roles}"
    assert any(c["selection"] == "granite" for c in calls), \
        "fallback model never invoked after domain failure"


def test_max_unique_builders_caps_chain(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """fallback_writer_model accepts a comma-separated chain. After
    using MAX_UNIQUE_BUILDERS (=3) total writers, no more swaps fire
    even if more failures happen."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    # Every writer round produces broken code → gate fails forever.
    broken = "```python\ndef foo():\n    return ghost\n```\n"
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[broken] * 20)

    r = client.post("/api/review", json={
        "prompt": "thing",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        # Chain: 3 fallbacks proposed. Cap should clip at 2 swaps
        # (primary + 2 fallbacks = 3 total = MAX_UNIQUE_BUILDERS).
        "fallback_writer_model": "granite, gemma, qwen2.5-coder:32b",
        "rounds": 12,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    deadline = time.monotonic() + 8.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done"

    # Distinct writer models in the call log (selection is whatever
    # `model_key` got passed to chat_service.stream_chat).
    distinct_writers = {
        c["selection"] for c in calls
        if c["selection"] in {"qwen", "granite", "gemma", "qwen2.5-coder:32b"}
    }
    assert len(distinct_writers) <= 3, (
        f"used too many builders, expected ≤3 (MAX_UNIQUE_BUILDERS): {distinct_writers}"
    )


def test_review_routes_to_reviewer_when_only_known_dep_missing(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Writer code uses scapy (a real third-party package) but scapy
    isn't installed in the gate env. The orchestrator must:
      1. NOT bump writer_static_fails (the writer didn't hallucinate),
      2. emit a `dependency_blocked` section, and
      3. still call the reviewer (with a gate-note prepended).
    """
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    # Force scapy + scapy.layers.bluetooth4LE to look uninstalled
    # regardless of whether the dev box has them.
    import importlib.util
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *a, **kw):
        if name == "scapy":
            return None
        return real_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    writer_code = (
        "Here you go:\n\n"
        "```python\n"
        "import scapy\n"
        "from scapy.layers.bluetooth4LE import BTLE_ADV\n"
        "def cap_ble():\n"
        "    return scapy, BTLE_ADV\n"
        "```\n"
    )
    review_text = "Architecture looks fine; assuming scapy is installed it works."
    calls = _stub_chat_service(
        monkeypatch,
        scripted_outputs=[writer_code, review_text],
    )

    r = client.post("/api/review", json={
        "prompt": "BLE sniffer",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        # rounds=2 means: writer round 0 + gate_or_review round 1.
        # No revise loop, so the 2 scripted outputs (writer + reviewer)
        # are exactly enough.
        "rounds": 2,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    deadline = time.monotonic() + 8.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done"

    roles = [s.role for s in saved.sections]
    assert "dependency_blocked" in roles, (
        f"no dependency_blocked section emitted; roles: {roles}"
    )
    assert "gates" not in roles, (
        f"writer was incorrectly bounced to gates path; roles: {roles}"
    )
    assert "reviewer" in roles, (
        f"reviewer should still run on dependency-blocked code; roles: {roles}"
    )

    # Reviewer prompt must include the GATE NOTE so the reviewer
    # doesn't reject the code as "uses an unavailable package".
    reviewer_calls = [c for c in calls if c["selection"] == "gemma"]
    assert reviewer_calls, "reviewer was never called"
    reviewer_prompt = "\n".join(m.content for m in reviewer_calls[0]["messages"])
    assert "GATE NOTE" in reviewer_prompt
    assert "scapy" in reviewer_prompt


def test_review_repeated_blocker_triggers_constrained_kickoff(
    monkeypatch: pytest.MonkeyPatch, client: TestClient,
) -> None:
    """Same blocker text appearing twice → orchestrator escalates to a
    constrained-architecture-template rebuild BEFORE aborting. This is
    the one-shot escalation path: abort only fires if the same blocker
    survives the constrained round too."""
    from app.web import app as web_app
    from app.utils.review_storage import ReviewStorage

    web_app.review_storage = ReviewStorage(web_app.review_storage.base_dir)

    clean1 = "```python\ndef sniff_ble():\n    pass\n```\n"
    blocker_review = (
        "Same architectural mistake.\n\n```json\n"
        '{"pass": false, "blockers": ["BLE cannot be sniffed from wlan0"], '
        '"safe_to_rebuild": true, "recommended_next_action": "rebuild_same_writer"}\n```'
    )
    clean2 = "```python\ndef sniff_ble2():\n    pass\n```\n"
    same_blocker_again = blocker_review  # threshold hits → constrained kickoff
    constrained_writer = "```python\ndef sniff_ble3():\n    pass\n```\n"
    post_constrained_review = (
        "Still wrong.\n\n```json\n"
        '{"pass": false, "blockers": ["BLE cannot be sniffed from wlan0"], '
        '"safe_to_rebuild": true, "recommended_next_action": "rebuild_same_writer"}\n```'
    )
    extra = "```python\npass\n```\n"
    calls = _stub_chat_service(monkeypatch, scripted_outputs=[
        clean1, blocker_review, clean2, same_blocker_again,
        constrained_writer, post_constrained_review, extra,
    ])

    r = client.post("/api/review", json={
        "prompt": "sniff BLE",
        "writer_model": "qwen",
        "reviewer_model": "gemma",
        "rounds": 10,
    })
    assert r.status_code == 200
    review_id = r.json()["review_id"]

    deadline = time.monotonic() + 8.0
    saved = None
    while time.monotonic() < deadline:
        saved = web_app.review_storage.load(review_id)
        if saved and saved.status in ("done", "error"):
            break
        time.sleep(0.05)
    assert saved is not None and saved.status == "done"

    # The constrained-kickoff section MUST be emitted between the
    # second blocker reveal and the abort.
    constrained_sections = [s for s in saved.sections if s.role == "constrained_kickoff"]
    assert constrained_sections, (
        f"no constrained_kickoff section; sections: "
        f"{[(s.role, s.text[:60]) for s in saved.sections]}"
    )
    assert any(
        "constrained" in s.text.lower() or "escalation" in s.text.lower()
        for s in constrained_sections
    )

    # The constrained writer round must have been called with the
    # architecture-template prompt — visible in the stubbed call log
    # via the user message text.
    constrained_calls = [
        c for c in calls
        if any(
            "ESCALATION" in m.content or "HARD CONSTRAINTS" in m.content
            for m in c["messages"]
        )
    ]
    assert constrained_calls, "constrained-kickoff prompt never reached the writer"

    # And after the constrained round still fails, the abort notice
    # must fire — that's the final stop, not just the escalation.
    fallback_sections = [s for s in saved.sections if s.role == "fallback"]
    assert any(
        "stopping:" in s.text.lower()
        or "constrained" in s.text.lower()
        or "abort" in s.text.lower()
        for s in fallback_sections
    ), (
        f"no abort notice after constrained round; "
        f"fallback sections: {[s.text for s in fallback_sections]}"
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

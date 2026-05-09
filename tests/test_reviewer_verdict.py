"""Tests for app/services/reviewer_verdict.py — JSON parsing of
reviewer output. Covers the structured-verdict feature so the
orchestrator can branch on `recommended_next_action` instead of
counting failures."""
from __future__ import annotations

import pytest

from app.services.reviewer_verdict import (
    SCHEMA_INSTRUCTION,
    VALID_ACTIONS,
    ReviewerVerdict,
    parse,
)


# ---------------------------------------------------------------------------
# parse — happy paths
# ---------------------------------------------------------------------------

def test_parse_extracts_valid_json_block() -> None:
    text = (
        "Here are the issues I found:\n"
        "- Missing import\n"
        "- Bad signature\n\n"
        "```json\n"
        '{"pass": false, "blockers": ["missing import os", "bad signature on add()"], '
        '"safe_to_rebuild": true, "recommended_next_action": "rebuild_same_writer"}\n'
        "```\n"
    )
    v = parse(text)
    assert v.passed is False
    assert v.blockers == ["missing import os", "bad signature on add()"]
    assert v.safe_to_rebuild is True
    assert v.next_action == "rebuild_same_writer"


def test_parse_approve_when_pass_true() -> None:
    text = (
        "All good.\n\n```json\n"
        '{"pass": true, "blockers": [], "safe_to_rebuild": true, '
        '"recommended_next_action": "approve"}\n```'
    )
    v = parse(text)
    assert v.passed is True
    assert v.next_action == "approve"


def test_parse_rebuild_different_writer_passthrough() -> None:
    text = (
        "Writer keeps claiming BLE works on wlan0.\n\n```json\n"
        '{"pass": false, "blockers": ["BLE on wlan0"], "safe_to_rebuild": true, '
        '"recommended_next_action": "rebuild_different_writer"}\n```'
    )
    v = parse(text)
    assert v.next_action == "rebuild_different_writer"
    assert v.blocker_categories == ["protocol_interface_mismatch"]


def test_parse_accepts_explicit_blocker_categories() -> None:
    text = (
        "```json\n"
        '{"pass": false, "blockers": ["uses fake BLE packets"], '
        '"blocker_categories": ["fake_implementation"], '
        '"safe_to_rebuild": true, '
        '"recommended_next_action": "rebuild_different_writer"}\n```'
    )
    v = parse(text)
    assert v.blocker_categories == ["fake_implementation"]


def test_parse_abort_when_unsalvageable() -> None:
    text = (
        "Hardware not present on the target.\n\n```json\n"
        '{"pass": false, "blockers": ["no SDR available"], "safe_to_rebuild": false, '
        '"recommended_next_action": "abort"}\n```'
    )
    v = parse(text)
    assert v.next_action == "abort"
    assert v.safe_to_rebuild is False


# ---------------------------------------------------------------------------
# parse — fallback / error paths
# ---------------------------------------------------------------------------

def test_parse_no_json_block_treats_as_one_blocker() -> None:
    """A reviewer model that ignores the schema and just emits prose
    still has to drive the orchestrator forward — treat its prose as
    a single blocker so the writer gets the feedback."""
    v = parse("This code is wrong because the imports are missing.")
    assert v.passed is False
    assert len(v.blockers) == 1
    assert "imports are missing" in v.blockers[0]
    assert v.next_action == "rebuild_same_writer"


def test_parse_malformed_json_falls_back() -> None:
    text = "Some prose.\n\n```json\n{not valid json}\n```\n"
    v = parse(text)
    assert v.passed is False
    assert any("unparseable" in b.lower() for b in v.blockers)
    assert v.next_action == "rebuild_same_writer"


def test_parse_empty_text_treated_as_approve() -> None:
    v = parse("")
    assert v.passed is True
    assert v.next_action == "approve"


def test_parse_pass_true_with_rebuild_same_writer_forces_approve() -> None:
    """Some local models emit `pass: true` and `rebuild_same_writer`
    in the same JSON — the orchestrator only short-circuits on
    `approve`, so without the parser forcing consistency the loop
    runs another pointless writer round on already-approved code."""
    text = (
        "```json\n"
        '{"pass": true, "blockers": [], "safe_to_rebuild": true, '
        '"recommended_next_action": "rebuild_same_writer"}\n```'
    )
    v = parse(text)
    assert v.passed is True
    assert v.next_action == "approve"


def test_parse_pass_false_with_approve_clamps_to_rebuild() -> None:
    """Inverse contradiction — pass=false but action=approve. Treat
    as the safest repairable interpretation."""
    text = (
        "```json\n"
        '{"pass": false, "blockers": ["bug"], "safe_to_rebuild": true, '
        '"recommended_next_action": "approve"}\n```'
    )
    v = parse(text)
    assert v.passed is False
    assert v.next_action == "rebuild_same_writer"


def test_parse_unknown_action_clamps_to_safest() -> None:
    text = (
        "```json\n"
        '{"pass": false, "blockers": ["x"], "safe_to_rebuild": true, '
        '"recommended_next_action": "send_carrier_pigeon"}\n```'
    )
    v = parse(text)
    # Unknown action with pass=false → safest fallback is rebuild_same.
    assert v.next_action == "rebuild_same_writer"


def test_parse_picks_last_json_block_when_multiple() -> None:
    """Reviewer might quote the schema as an example earlier in its
    response; we want the *final* JSON block as the actual verdict."""
    text = (
        "Schema you asked me to follow:\n\n"
        '```json\n{"pass": true, "blockers": [], "safe_to_rebuild": true, '
        '"recommended_next_action": "approve"}\n```\n\n'
        "Actual verdict:\n\n"
        '```json\n{"pass": false, "blockers": ["bug"], "safe_to_rebuild": true, '
        '"recommended_next_action": "rebuild_same_writer"}\n```'
    )
    v = parse(text)
    assert v.passed is False
    assert v.next_action == "rebuild_same_writer"


def test_display_text_strips_json_block() -> None:
    text = (
        "Issues:\n- foo\n\n```json\n"
        '{"pass": false, "blockers": ["foo"], "safe_to_rebuild": true, '
        '"recommended_next_action": "rebuild_same_writer"}\n```\n'
    )
    v = parse(text)
    rendered = v.display_text()
    assert "Issues:" in rendered
    assert "```json" not in rendered
    assert "rebuild_same_writer" not in rendered


# ---------------------------------------------------------------------------
# Schema instruction content
# ---------------------------------------------------------------------------

def test_schema_instruction_lists_every_valid_action() -> None:
    """Regression guard: the prompt's enum must match the parser's."""
    for action in VALID_ACTIONS:
        assert action in SCHEMA_INSTRUCTION, f"missing action in schema doc: {action}"


def test_reviewer_instruction_includes_hardware_grounding() -> None:
    """Regression guard for the BLE-on-wlan0 class of bug — the
    reviewer prompt MUST tell the model to flag wireless / hardware
    confusion. If someone trims this back, the test catches it."""
    pytest.importorskip("chromadb")  # app.web.app imports chromadb
    from app.web.app import REVIEWER_INSTRUCTION
    text = REVIEWER_INSTRUCTION.lower()
    # Hardware/protocol reality checks
    assert "ble" in text and "bluetooth" in text
    assert "wlan0" in text or "monitor mode" in text
    assert "hci" in text  # the actual BT stack on Linux
    assert "sdr" in text  # the SDR-vs-network confusion
    assert "domain" in text  # a "domain checks" section header is present


def test_reviewer_instruction_includes_real_vs_simulated() -> None:
    """Regression guard for Class 3 — reviewer must flag fake /
    simulated implementations and recommend rebuild_different_writer
    rather than letting the same writer keep faking output."""
    pytest.importorskip("chromadb")
    from app.web.app import REVIEWER_INSTRUCTION
    text = REVIEWER_INSTRUCTION.lower()
    assert "simulated" in text or "placeholder" in text
    assert "rebuild_different_writer" in text
    # The exact failure mode ChatGPT highlighted.
    assert "for demonstration" in text or "would do x in production" in text


def test_reviewer_verdict_dataclass_fields_default_safe() -> None:
    """A bare ReviewerVerdict (no JSON parsed) defaults to a state
    that's safe for the orchestrator: rebuild same, not approved."""
    v = ReviewerVerdict(passed=False)
    assert v.next_action == "rebuild_same_writer"
    assert v.safe_to_rebuild is True
    assert v.blockers == []
    assert v.blocker_categories == []

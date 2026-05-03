import json
import tempfile
from pathlib import Path

import pytest

from app.utils.comparison_storage import (
    ComparisonRun,
    ComparisonStorage,
    ModelOutput,
    new_run,
)


@pytest.fixture
def tmp_storage() -> ComparisonStorage:
    with tempfile.TemporaryDirectory() as td:
        yield ComparisonStorage(Path(td))


def test_new_run_has_id_and_timestamp() -> None:
    r = new_run(prompt="What is X?", system_prompt="Be brief.")
    assert r.id
    assert r.timestamp
    assert r.prompt == "What is X?"
    assert r.outputs == []
    assert r.winner is None


def test_save_load_round_trip(tmp_storage: ComparisonStorage) -> None:
    r = new_run("explain Kerberos")
    r.outputs = [
        ModelOutput(model_key="qwen", model_name="qwen3-coder:30b", text="answer", elapsed_s=2.1),
        ModelOutput(model_key="gemma", model_name="gemma4", text="other answer", elapsed_s=1.4),
    ]
    r.winner = "qwen"
    tmp_storage.save(r)
    loaded = tmp_storage.load(r.id)
    assert loaded is not None
    assert loaded.id == r.id
    assert loaded.winner == "qwen"
    assert {o.model_key for o in loaded.outputs} == {"qwen", "gemma"}


def test_list_runs_sorted_desc(tmp_storage: ComparisonStorage) -> None:
    older = {
        "id": "old", "timestamp": "2026-05-01T10:00:00.000+00:00",
        "prompt": "p1", "system_prompt": "", "outputs": [], "winner": None,
    }
    newer = {
        "id": "new", "timestamp": "2026-05-03T10:00:00.000+00:00",
        "prompt": "p2", "system_prompt": "", "outputs": [], "winner": None,
    }
    (tmp_storage.base_dir / "old.json").write_text(json.dumps(older))
    (tmp_storage.base_dir / "new.json").write_text(json.dumps(newer))
    runs = tmp_storage.list_runs()
    assert [r.id for r in runs] == ["new", "old"]


def test_winner_tally_aggregates(tmp_storage: ComparisonStorage) -> None:
    for winner in ("qwen", "qwen", "gemma", None):
        r = new_run("p")
        r.winner = winner
        tmp_storage.save(r)
    tally = tmp_storage.winner_tally()
    assert tally == {"qwen": 2, "gemma": 1}


def test_path_traversal_blocked(tmp_storage: ComparisonStorage) -> None:
    with pytest.raises(ValueError):
        tmp_storage.load("../etc/passwd")


def test_load_unknown_id_returns_none(tmp_storage: ComparisonStorage) -> None:
    assert tmp_storage.load("nonexistent") is None

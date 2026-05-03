from app.services.rag_fusion import reciprocal_rank_fusion


def test_combines_all_ids() -> None:
    a = ["x", "y", "z"]
    b = ["z", "a", "y"]
    fused = reciprocal_rank_fusion([a, b])
    assert set(fused) == {"x", "y", "z", "a"}


def test_doc_in_both_rankings_wins() -> None:
    a = ["x", "y", "z", "w"]
    b = ["z", "a", "y", "q"]
    fused = reciprocal_rank_fusion([a, b])
    top = max(fused, key=lambda k: fused[k])
    # z appears at rank 0 in b and rank 2 in a — best total.
    # y also appears in both. One of these should win.
    assert top in {"z", "y"}


def test_unique_to_one_ranking_at_same_rank_ties() -> None:
    a = ["x", "y"]
    b = ["a", "b"]
    fused = reciprocal_rank_fusion([a, b])
    assert fused["x"] == fused["a"]
    assert fused["y"] == fused["b"]


def test_higher_rank_scores_higher() -> None:
    fused = reciprocal_rank_fusion([["first", "second", "third"]])
    assert fused["first"] > fused["second"] > fused["third"]


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[]]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}

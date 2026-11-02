from policy_assistant.retrieval.fusion import rrf


def test_rrf_hand_computed() -> None:
    fused = rrf([[1, 2], [2, 3]], 60)
    assert fused == [
        (2, 1 / 62 + 1 / 61),
        (1, 1 / 61),
        (3, 1 / 62),
    ]


def test_rrf_ties_break_by_chunk_id() -> None:
    assert rrf([[5], [7]], 60) == [(5, 1 / 61), (7, 1 / 61)]

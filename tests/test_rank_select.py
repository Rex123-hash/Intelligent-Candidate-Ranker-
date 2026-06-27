from src import rank


def test_select_top_tiebreaks_on_rounded_score_by_candidate_id():
    # Two candidates whose full scores differ but round to the same 4-decimal value.
    # The higher full score is the LARGER candidate_id; after rounding they tie, so
    # the output must order them by candidate_id ascending (validator rule).
    scored = [
        {"candidate_id": "CAND_0000050", "score": 0.74103},  # higher full score, bigger id
        {"candidate_id": "CAND_0000010", "score": 0.74097},  # lower full score, smaller id
        {"candidate_id": "CAND_0000099", "score": 0.90000},
    ]
    top = rank.select_top(scored, top_n=3)
    assert [s["candidate_id"] for s in top] == [
        "CAND_0000099", "CAND_0000010", "CAND_0000050"
    ]
    # both near-tied rows print the same rounded score
    assert top[1]["score4"] == top[2]["score4"] == 0.7410


def test_select_top_respects_top_n():
    scored = [{"candidate_id": f"CAND_{i:07d}", "score": i / 100} for i in range(1, 11)]
    top = rank.select_top(scored, top_n=3)
    assert len(top) == 3
    assert top[0]["candidate_id"] == "CAND_0000010"  # highest score first

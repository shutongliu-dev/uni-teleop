from scripts.diagnostics.cloth_clearance_witness import isolate_pair


def test_subset_bisection_finds_cross_half_pair_without_changing_indices():
    calls = []
    def score(ids):
        calls.append(ids)
        return 3. if 7 in ids and 996 in ids else 1. if 1 in ids and 3 in ids else 0.
    assert set(isolate_pair(range(1200), score, 3.)) == {7, 996}
    assert len(calls) < 30


def test_subset_bisection_finds_same_half_pair_with_tied_maxima():
    def score(ids):
        return 3. if ({17, 21} <= set(ids) or {790, 810} <= set(ids)) else 0.
    assert set(isolate_pair(range(1200), score, 3.)) in ({17, 21}, {790, 810})

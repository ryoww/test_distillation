from scripts.select_verified_best import pick


def test_the_feasible_candidate_with_the_lowest_recomputed_cost_wins():
    assert pick([("fft", True, 120.0), ("dsh", True, 100.0)]) == "dsh"


def test_an_infeasible_candidate_never_wins_over_a_feasible_one():
    assert pick([("fft", True, 120.0), ("dsh", False, None)]) == "fft"
    assert pick([("fft", False, None), ("dsh", True, 300.0)]) == "dsh"


def test_the_first_run_is_kept_when_nothing_is_feasible():
    assert pick([("fft", False, None), ("dsh", False, None)]) == "fft"

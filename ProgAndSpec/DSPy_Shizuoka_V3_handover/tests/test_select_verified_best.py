from scripts.select_verified_best import pick, pick_cascade


def test_the_feasible_candidate_with_the_lowest_recomputed_cost_wins():
    assert pick([("fft", "feasible", 120.0), ("dsh", "feasible", 100.0)]) == "dsh"


def test_a_failed_candidate_never_wins_over_a_feasible_one():
    assert pick([("fft", "feasible", 120.0), ("dsh", "infeasible", None)]) == "fft"
    assert pick([("fft", "exec_error", None), ("dsh", "feasible", 300.0)]) == "dsh"


def test_feasible_without_an_objective_beats_unverified_and_failed():
    assert pick([("fft", "unverified", None), ("dsh", "feasible", None)]) == "dsh"
    assert pick([("fft", "infeasible", None), ("dsh", "unverified", None)]) == "dsh"


def test_ties_keep_the_run_order():
    assert pick([("fft", "unverified", None), ("dsh", "unverified", None)]) == "fft"
    assert pick([("fft", "exec_error", None), ("dsh", "infeasible", None)]) == "fft"


def test_cascade_stops_at_a_feasible_student_even_if_a_later_run_is_better():
    assert pick_cascade([("student", "feasible", 120.0), ("agent", "feasible", 100.0)]) == "student"


def test_cascade_falls_back_when_the_student_fails():
    assert pick_cascade([("student", "exec_error", None), ("agent", "feasible", 100.0)]) == "agent"

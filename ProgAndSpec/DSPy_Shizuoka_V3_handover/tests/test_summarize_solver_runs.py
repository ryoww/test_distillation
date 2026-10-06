import json

from scripts.summarize_solver_runs import gap_of, is_correct, kinds_of, summarize


def _row(status, cost, ref, score=1.0):
    return {"status": status, "cost": cost, "reference_value": ref, "score": score, "usage": {}}


def test_correct_requires_feasible_status_and_gap_within_limit():
    rows = [
        _row("exact_match", 100.0, 100.0),
        _row("worse", 109.0, 100.0),
        _row("worse", 120.0, 100.0),
        _row("infeasible", 50.0, 100.0),
        _row("unverified", 100.0, 100.0),
    ]
    summary = summarize(rows, max_gap=0.10)
    assert summary["correct"] == 2
    assert summary["at_least_reference"] == 1
    assert summary["feasible"] == 3


def test_zero_reference_counts_only_zero_cost():
    assert gap_of(_row("exact_match", 0.0, 0.0)) == 0.0
    assert gap_of(_row("worse", 1.0, 0.0)) == float("inf")


def test_exact_reference_match_is_correct():
    assert is_correct(_row("exact_match", 100.0, 100.0), max_gap=0.10)
    assert not is_correct(_row("infeasible", 100.0, 100.0), max_gap=0.10)


def test_kinds_of_maps_generated_ids_to_their_kind(tmp_path):
    (tmp_path / "prob_4001.json").write_text(json.dumps({"id": 4001, "provenance": {"kind": "clsp"}}))
    (tmp_path / "prob_0001.json").write_text(json.dumps({"id": 1}))
    assert kinds_of([tmp_path]) == {"prob_4001": "clsp"}

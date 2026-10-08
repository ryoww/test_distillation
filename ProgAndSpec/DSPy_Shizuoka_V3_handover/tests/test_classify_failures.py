import json

from scripts.classify_failures import classify, kinds_from


def _row(status, detail="", tokens=100, cost=None, ref=None):
    return {"status": status, "detail": detail, "usage": {"completion_tokens": tokens}, "cost": cost,
            "reference_value": ref, "score": 0.0}


def test_truncation_wins_over_the_resulting_syntax_error():
    assert classify(_row("exec_error", "SYNTAX: '(' was never closed", tokens=8192), 8192) == "truncated"
    assert classify(_row("gen_error", "no solve() in response", tokens=8192), 8192) == "truncated"


def test_basic_coding_failures_are_split_by_error_head():
    assert classify(_row("exec_error", "SYNTAX: invalid syntax"), 8192) == "syntax"
    assert classify(_row("exec_error", "KEYERROR: 'flights'"), 8192) == "data_access"
    assert classify(_row("exec_error", "TIMEOUT: Timeout after 900.0s"), 8192) == "timeout"
    assert classify(_row("unverified", "ref=1, cost=2, feasibility=unverified"), 8192) == "output_shape"


def test_violations_separate_declared_mismatch_from_structural_breakage():
    declared = "constraint violation (1/327 violated, partial_score=0.99): declared objective_value 0.00 != recomputed 10.0"
    assert classify(_row("partial_feasible", declared), 8192) == "declared_only"
    most = "constraint violation (32/36 violated, partial_score=0.11): holdings below minimum"
    assert classify(_row("partial_feasible", most), 8192) == "violation_most"
    few = "constraint violation (2/52 violated, partial_score=0.9): route 3 exceeds capacity"
    assert classify(_row("partial_feasible", few), 8192) == "violation_partial"


def test_feasible_but_worse_is_split_at_fifty_percent():
    assert classify(_row("worse", cost=300.0, ref=100.0), 8192) == "gap_large"
    assert classify(_row("worse", cost=120.0, ref=100.0), 8192) == "gap_moderate"


def test_original_problems_take_their_kind_from_the_requirements(tmp_path):
    generated = {"id": 4001, "provenance": {"kind": "clsp"}, "requirements": {"kind": "clsp"}}
    original = {"id": 302, "provenance": {"source": "hard"}, "requirements": {"kind": "crew_pairing"}}
    for record in (generated, original):
        (tmp_path / f"prob_{record['id']}.json").write_text(json.dumps(record), encoding="utf-8")
    assert kinds_from([tmp_path]) == {"prob_4001": "clsp", "prob_302": "crew_pairing"}

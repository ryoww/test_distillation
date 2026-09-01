"""提出前検証と修復ループの振る舞いを検証する。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.verify_loop import (
    generate_verified,
    summarize_attempts,
    verify_solution,
)

BASE_DIR = Path(__file__).resolve().parents[1]
PROBLEM_DIR = BASE_DIR / "data" / "problems"

KNAPSACK = json.loads((PROBLEM_DIR / "prob_055.json").read_text(encoding="utf-8"))
INSTANCE = KNAPSACK["instance"]
CORE_TYPE = f"{KNAPSACK['domain']}_{KNAPSACK['math_type']}"
REFERENCE = KNAPSACK["reference_solution"]


def _code_returning(payload: str) -> str:
    return f"def solve(instance):\n    return {payload}\n"


GOOD_CODE = _code_returning(
    f"{{'max_value': {REFERENCE['max_value']}, 'chosen_items': {REFERENCE['chosen_items']}}}"
)


@dataclass
class FakePrediction:
    algorithm_code: str
    parse_code: str = "# helpers"


class FakeProgram:
    """生成と修復の呼び出しを記録するだけのスタブ。"""

    def __init__(self, first: str, revisions: list[str] | None = None):
        self.first = first
        self.revisions = list(revisions or [])
        self.improve_calls: list[dict] = []

    def __call__(self, requirement: str, core_type: str) -> FakePrediction:
        return FakePrediction(algorithm_code=self.first)

    def improve_forward(self, **kwargs) -> FakePrediction:
        self.improve_calls.append(kwargs)
        nxt = self.revisions.pop(0) if self.revisions else kwargs["original_code"]
        return FakePrediction(algorithm_code=nxt)


# --- verify_solution ---------------------------------------------------------


def test_code_that_raises_is_reported_as_exec_error():
    verdict = verify_solution(
        "def solve(instance):\n    raise ValueError('boom')\n", INSTANCE, CORE_TYPE
    )
    assert verdict.ok is False
    assert verdict.kind == "exec_error"
    assert "boom" in verdict.feedback


def test_code_returning_an_error_dict_is_reported_as_exec_error():
    code = _code_returning("{'error': 'internal failure'}")
    verdict = verify_solution(code, INSTANCE, CORE_TYPE)
    assert verdict.kind == "exec_error"
    assert "internal failure" in verdict.feedback


def test_empty_solution_is_rejected():
    verdict = verify_solution(_code_returning("{}"), INSTANCE, CORE_TYPE)
    assert verdict.ok is False
    assert verdict.kind == "empty"


def test_constraint_violation_is_reported_with_the_violations():
    every_item = list(range(len(INSTANCE["items"])))
    code = _code_returning(f"{{'max_value': 9999, 'chosen_items': {every_item}}}")
    verdict = verify_solution(code, INSTANCE, CORE_TYPE)
    assert verdict.ok is False
    assert verdict.kind == "infeasible"
    assert any("exceeds capacity" in v for v in verdict.violations)


def test_a_correct_solution_passes():
    verdict = verify_solution(GOOD_CODE, INSTANCE, CORE_TYPE)
    assert verdict.ok is True
    assert verdict.kind == "feasible"


def test_unrecognized_shape_is_not_treated_as_a_violation():
    """検証できないことと制約違反は別物。突き返さない。"""
    verdict = verify_solution(_code_returning("{'unexpected': 1}"), INSTANCE, CORE_TYPE)
    assert verdict.ok is True
    assert verdict.kind == "unverified"


def test_feedback_never_carries_the_target_value():
    code = _code_returning("{'max_value': 1, 'chosen_items': [0, 0]}")
    verdict = verify_solution(code, INSTANCE, CORE_TYPE)
    assert str(REFERENCE["max_value"]) not in verdict.feedback


# --- generate_verified -------------------------------------------------------


def test_single_attempt_records_the_verdict_without_revising():
    program = FakeProgram(first=GOOD_CODE)
    code, _, attempts = generate_verified(program, "req", CORE_TYPE, INSTANCE, max_attempts=1)
    assert code == GOOD_CODE
    assert len(attempts) == 1
    assert program.improve_calls == []


def test_a_broken_first_pass_is_repaired_and_reverified():
    program = FakeProgram(first=_code_returning("{}"), revisions=[GOOD_CODE])
    code, _, attempts = generate_verified(
        program, "req", CORE_TYPE, INSTANCE, max_attempts=2, return_schema="- max_value: numeric"
    )
    assert code == GOOD_CODE
    assert [a.kind for a in attempts] == ["empty", "feasible"]
    assert summarize_attempts(attempts)["repaired"] is True


def test_the_return_schema_reaches_the_reviser():
    program = FakeProgram(first=_code_returning("{}"), revisions=[GOOD_CODE])
    generate_verified(
        program, "req", CORE_TYPE, INSTANCE, max_attempts=2, return_schema="- max_value: numeric"
    )
    assert program.improve_calls[0]["return_schema"] == "- max_value: numeric"
    assert program.improve_calls[0]["core_type"] == CORE_TYPE


def test_loop_stops_when_the_reviser_returns_the_same_code():
    broken = _code_returning("{}")
    program = FakeProgram(first=broken, revisions=[broken])
    _, _, attempts = generate_verified(program, "req", CORE_TYPE, INSTANCE, max_attempts=4)
    assert len(attempts) == 1
    assert len(program.improve_calls) == 1


def test_loop_respects_the_attempt_budget():
    program = FakeProgram(
        first=_code_returning("{}"),
        revisions=[_code_returning("[]"), _code_returning("None"), GOOD_CODE],
    )
    _, _, attempts = generate_verified(program, "req", CORE_TYPE, INSTANCE, max_attempts=3)
    assert len(attempts) == 3
    assert attempts[-1].ok is False


def test_a_custom_reviser_can_replace_the_default():
    calls: list[dict] = []

    def reviser(**kwargs):
        calls.append(kwargs)
        return FakePrediction(algorithm_code=GOOD_CODE)

    program = FakeProgram(first=_code_returning("{}"))
    code, _, _ = generate_verified(
        program, "req", CORE_TYPE, INSTANCE, max_attempts=2, reviser=reviser
    )
    assert code == GOOD_CODE
    assert len(calls) == 1
    assert program.improve_calls == []


# --- summarize_attempts ------------------------------------------------------


def test_summary_keeps_first_pass_and_final_apart():
    program = FakeProgram(first=_code_returning("{}"), revisions=[GOOD_CODE])
    _, _, attempts = generate_verified(program, "req", CORE_TYPE, INSTANCE, max_attempts=2)
    summary = summarize_attempts(attempts)
    assert summary["first_pass_ok"] is False
    assert summary["final_ok"] is True
    assert summary["attempts"] == 2


def test_summary_of_a_clean_first_pass_is_not_a_repair():
    program = FakeProgram(first=GOOD_CODE)
    _, _, attempts = generate_verified(program, "req", CORE_TYPE, INSTANCE, max_attempts=2)
    assert summarize_attempts(attempts)["repaired"] is False


def test_empty_summary_is_well_formed():
    assert summarize_attempts([])["attempts"] == 0

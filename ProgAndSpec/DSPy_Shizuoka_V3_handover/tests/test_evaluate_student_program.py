from scripts.evaluate_student_program import score_row
from src.agent import AgentResult, Attempt


class _Agent:
    def solve(self, requirement, core_type, instance):
        return AgentResult(
            status="failed", code="def f():\n    pass", objective=None, violations=[],
            attempts=[Attempt("student", "exec_error", "boom", 1.0), Attempt("student-repair", "exec_error", "boom", 1.0)],
            fallback_used=True,
        )


def test_row_without_solve_is_a_generation_error_and_keeps_agent_fields():
    example = {"instance_id": "prob_1", "core_type": "x", "instance": {}, "record": {"id": 1, "instance": {}}}
    row = score_row(_Agent(), example, exec_timeout=1.0)
    assert row["status"] == "gen_error" and row["score"] == -0.5
    assert row["first_verdict"] == "exec_error" and row["repaired"] and row["fallback_used"]
    assert row["attempts"] == 2 and row["agent_status"] == "failed"


class _Refusing:
    def solve(self, requirement, core_type, instance):
        return AgentResult(status="unsupported", code="", objective=None, violations=[],
                           failure_reason="kind 'portfolio' is outside", kind="portfolio", supported=False)


def test_unsupported_rows_score_zero_and_keep_the_kind():
    example = {"instance_id": "prob_2", "core_type": "x", "instance": {}, "record": {"id": 2, "instance": {}}}
    row = score_row(_Refusing(), example, exec_timeout=1.0)
    assert row["status"] == "unsupported" and row["score"] == 0.0
    assert row["kind"] == "portfolio" and row["supported"] is False

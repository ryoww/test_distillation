import dspy

from scripts.evaluate_student_program import score_row


class _Program:
    def __call__(self, requirement, core_type, problem_instance):
        return dspy.Prediction(algorithm_code="def f():\n    pass", first_verdict="exec_error", repaired=True)


def test_row_without_solve_is_a_generation_error_and_keeps_repair_fields():
    example = {"instance_id": "prob_1", "core_type": "x", "instance": {}, "record": {"id": 1, "instance": {}}}
    row = score_row(_Program(), example, exec_timeout=1.0)
    assert row["status"] == "gen_error" and row["score"] == -0.5
    assert row["first_verdict"] == "exec_error" and row["repaired"] is True

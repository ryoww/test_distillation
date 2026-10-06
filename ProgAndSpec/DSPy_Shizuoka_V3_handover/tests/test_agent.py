import dspy

from src import agent as ag
from src.verify_loop import Verdict
from tests.test_student_program import _ScriptedLM

CODE_A = "def solve(instance):\n    return {'a': 1}"
CODE_B = "def solve(instance):\n    return {'b': 2}"
CODE_C = "def solve(instance):\n    return {'c': 3}"


def _verdicts(monkeypatch, kinds):
    it = iter(kinds)

    def fake(code, instance, core_type, timeout):
        kind = next(it)
        ok = kind in ("feasible", "unverified")
        return Verdict(ok=ok, kind=kind, feedback="" if ok else f"bad: {kind}", solution={"x": 1} if ok else None)

    monkeypatch.setattr(ag, "verify_solution", fake)
    monkeypatch.setattr(ag, "check_feasibility_detailed", lambda *a, **k: {"cost": 42.0})


def test_solved_first_try_has_no_repair(monkeypatch):
    _verdicts(monkeypatch, ["feasible"])
    lm = _ScriptedLM([CODE_A])
    result = ag.OptimizationAgent(lm, max_repairs=2).solve("REQ", "x", {})
    assert result.status == "solved" and result.objective == 42.0 and not result.fallback_used
    assert [a.stage for a in result.attempts] == ["student"]


def test_repairs_at_most_max_repairs_then_fails_with_reason(monkeypatch):
    _verdicts(monkeypatch, ["exec_error", "infeasible", "infeasible"])
    lm = _ScriptedLM([CODE_A, CODE_B, CODE_C])
    result = ag.OptimizationAgent(lm, max_repairs=2).solve("REQ", "x", {})
    assert result.status == "failed" and "infeasible" in result.failure_reason
    assert [a.stage for a in result.attempts] == ["student", "student-repair", "student-repair"]


def test_fallback_runs_when_student_fails(monkeypatch):
    _verdicts(monkeypatch, ["exec_error", "feasible"])
    student = _ScriptedLM([CODE_A, CODE_A])  # repair returns the same code -> stop
    fallback = _ScriptedLM([CODE_B])
    result = ag.OptimizationAgent(student, fallback_lm=fallback, max_repairs=2).solve("REQ", "x", {})
    assert result.status == "solved" and result.fallback_used and result.code == CODE_B
    assert [a.stage for a in result.attempts] == ["student", "fallback"]


def test_result_serialises_to_plain_dict(monkeypatch):
    _verdicts(monkeypatch, ["unverified"])
    result = ag.OptimizationAgent(_ScriptedLM([CODE_A])).solve("REQ", "x", {})
    payload = result.to_dict()
    assert payload["status"] == "unverified" and isinstance(payload["attempts"][0], dict)
    assert isinstance(dspy.Prediction(**payload), dspy.Prediction)

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


def _hard_instance(kind_name):
    import json
    from pathlib import Path

    for path in sorted(Path("data/problems_hard_gen/test").glob("prob_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["provenance"]["kind"] == kind_name:
            return record["instance"]
    raise LookupError(kind_name)


def test_detect_kind_uses_instance_shape_for_hard_problems_and_core_type_otherwise():
    assert ag.detect_kind("whatever", _hard_instance("pdptw")) == "pdptw"
    assert ag.detect_kind("配送・輸送_混合整数計画", {"x": 1}) == "配送・輸送_混合整数計画"


def test_unsupported_kind_is_reported_without_calling_the_student():
    lm = _ScriptedLM([])
    agent = ag.OptimizationAgent(lm, supported_kinds={"pdptw"})
    result = agent.solve("REQ", "x", _hard_instance("portfolio"))
    assert result.status == "unsupported" and result.kind == "portfolio" and result.supported is False
    assert lm.seen == [] and "outside the kinds" in result.failure_reason


def test_unsupported_kind_goes_to_the_routing_lm_when_given(monkeypatch):
    _verdicts(monkeypatch, ["feasible"])
    student, router = _ScriptedLM([]), _ScriptedLM([CODE_B])
    agent = ag.OptimizationAgent(student, supported_kinds={"pdptw"}, unsupported_lm=router)
    result = agent.solve("REQ", "x", _hard_instance("portfolio"))
    assert result.status == "solved" and result.routed_unsupported and result.code == CODE_B
    assert student.seen == [] and [a.stage for a in result.attempts] == ["unsupported-route"]


def test_supported_kind_runs_the_student(monkeypatch):
    _verdicts(monkeypatch, ["feasible"])
    result = ag.OptimizationAgent(_ScriptedLM([CODE_A]), supported_kinds={"pdptw"}).solve(
        "REQ", "x", _hard_instance("pdptw")
    )
    assert result.status == "solved" and result.supported is True

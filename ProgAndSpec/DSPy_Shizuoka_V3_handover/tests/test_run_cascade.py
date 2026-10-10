import argparse

import scripts.run_cascade as cascade


def _args(tmp_path):
    return argparse.Namespace(work_root=tmp_path, label="t", solve_limit=300, task_timeout=60,
                              dsh=tmp_path / "dsh", exec_timeout=60)


def _example(kind):
    return {"instance_id": "prob_1", "core_type": kind, "instance": {}, "record": {}}


def _wire(monkeypatch, verdicts, calls):
    monkeypatch.setattr(cascade, "student_code", lambda *a: calls.append("student") or "S")
    monkeypatch.setattr(
        cascade, "run_dsh", lambda *a, **k: calls.append("dsh") or ("D", {"dsh_returncode": 0})
    )
    monkeypatch.setattr(cascade, "verify", lambda code, ex, t: verdicts[code])
    monkeypatch.setattr(cascade, "score", lambda args, ex, code: {"code": code, "status": "ok", "score": 1.0})


def test_unsupported_kinds_go_straight_to_the_agent(monkeypatch, tmp_path):
    calls = []
    _wire(monkeypatch, {"D": ("feasible", 1.0)}, calls)
    row = cascade.solve_one(_args(tmp_path), "sys", {"known"}, _example("unknown"))
    assert calls == ["dsh"] and row["picked_from"] == "dsh"


def test_a_feasible_student_answer_stops_the_cascade(monkeypatch, tmp_path):
    calls = []
    _wire(monkeypatch, {"S": ("feasible", 5.0)}, calls)
    row = cascade.solve_one(_args(tmp_path), "sys", {"known"}, _example("known"))
    assert calls == ["student"] and row["code"] == "S"


def test_a_failed_student_answer_calls_the_agent_and_keeps_the_better_version(monkeypatch, tmp_path):
    calls = []
    _wire(monkeypatch, {"S": ("infeasible", None), "D": ("feasible", 7.0)}, calls)
    row = cascade.solve_one(_args(tmp_path), "sys", {"known"}, _example("known"))
    assert calls == ["student", "dsh"] and row["picked_from"] == "dsh"

import argparse

import scripts.sample_and_select as sampler


def test_the_verified_best_sample_is_scored_and_greedy_wins_ties(monkeypatch):
    outputs = iter(["G", "A", "B"])
    verdicts = {"G": ("feasible", 10.0), "A": ("feasible", 8.0), "B": ("infeasible", None)}
    scored = []
    monkeypatch.setattr(sampler, "generate", lambda *a: next(outputs))
    monkeypatch.setattr(sampler, "verify", lambda code, ex, t: verdicts[code])
    monkeypatch.setattr(
        sampler, "score", lambda args, ex, code: scored.append(code) or {"status": code, "score": 1.0}
    )
    args = argparse.Namespace(samples=3, temperature=0.7, solve_limit=300)
    row = sampler.solve_one(args, "sys", {"instance_id": "p", "core_type": "k", "instance": {}, "record": {}})
    assert row["picked_index"] == 1 and scored == ["A", "G"]
    assert row["status"] == "A" and row["greedy_status"] == "G"
    assert [s["temperature"] for s in row["samples"]] == [0.0, 0.7, 0.7]


def test_ties_fall_back_to_the_greedy_sample(monkeypatch):
    outputs = iter(["G", "A"])
    monkeypatch.setattr(sampler, "generate", lambda *a: next(outputs))
    monkeypatch.setattr(sampler, "verify", lambda code, ex, t: ("unverified", None))
    monkeypatch.setattr(sampler, "score", lambda args, ex, code: {"status": "ok", "score": 1.0, "code": code})
    args = argparse.Namespace(samples=2, temperature=0.7, solve_limit=300)
    row = sampler.solve_one(args, "sys", {"instance_id": "p", "core_type": "k", "instance": {}, "record": {}})
    assert row["picked_index"] == 0

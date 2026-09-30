"""大規模問題集 shop_cutting 群（FJSP と カッティングストック）の検証器の振る舞いを検証する。"""

from __future__ import annotations

import copy
import json
from itertools import pairwise
from pathlib import Path

import pytest

from src.utils.feasibility import check_feasibility_detailed
from src.utils.hard import find_kind
from src.utils.scorer import compute_score

BASE_DIR = Path(__file__).resolve().parents[1]
HARD_DIR = BASE_DIR / "data" / "problems_hard"
BUNDLED_DIR = BASE_DIR / "data" / "problems"

EXPECTED_KINDS = {
    "prob_305": "fjsp",
    "prob_315": "fjsp",
    "prob_322": "fjsp_setup",
    "prob_307": "cutting_1d",
    "prob_317": "cutting_1d",
    "prob_324": "cutting_2d",
}


def _load(pid: str) -> dict:
    return json.loads((HARD_DIR / f"{pid}.json").read_text())


def _reference(record: dict) -> dict:
    return {
        k: v for k, v in record["reference_solution"].items()
        if k not in ("objective_value", "note")
    }


def _core_type(record: dict) -> str:
    return f"{record['domain']}_{record['math_type']}"


@pytest.mark.parametrize("pid", sorted(EXPECTED_KINDS))
def test_detects_expected_kind(pid):
    kind = find_kind(_load(pid)["instance"])
    assert kind is not None and kind.name == EXPECTED_KINDS[pid]


@pytest.mark.parametrize("pid", sorted(EXPECTED_KINDS))
def test_reference_solution_is_feasible_and_reproduces_objective(pid):
    record = _load(pid)
    kind = find_kind(record["instance"])
    result = kind.check(record["instance"], _reference(record))
    objective = record["reference_solution"]["objective_value"]
    assert result["verified"] is True
    assert result["feasible"] is True
    assert result["violation_count"] == 0
    assert abs(result["cost"] - objective) <= 1e-3 * max(1, abs(objective))


@pytest.mark.parametrize("pid", sorted(EXPECTED_KINDS))
def test_dispatch_routes_to_hard_checker_and_scores_negative_cost(pid):
    record = _load(pid)
    instance, reference = record["instance"], _reference(record)
    objective = record["reference_solution"]["objective_value"]
    detailed = check_feasibility_detailed(_core_type(record), instance, reference)
    assert detailed["verified"] is True and detailed["feasible"] is True
    score = compute_score(_core_type(record), instance, reference)
    assert score == pytest.approx(-objective, rel=1e-3)


@pytest.mark.parametrize("pid", sorted(EXPECTED_KINDS))
def test_unparseable_solution_is_unverified_not_infeasible(pid):
    record = _load(pid)
    kind = find_kind(record["instance"])
    for garbage in (None, [], {"answer": 1}, {"schedule": 3, "patterns": "x"}):
        result = kind.check(record["instance"], garbage)
        assert result["verified"] is False
        assert result["violation_count"] == 0


def test_bundled_problems_are_not_detected_as_hard_kinds():
    hits = []
    for path in sorted(BUNDLED_DIR.glob("*.json")):
        kind = find_kind(json.loads(path.read_text())["instance"])
        if kind is not None:
            hits.append((path.name, kind.name))
    assert hits == []


# ---------------------------------------------------------------- 参照解を壊す
def _first_op(reference: dict) -> dict:
    """辞書形式・リスト形式どちらの schedule でも最初の工程行を返す。"""
    schedule = reference["schedule"]
    return schedule[0] if isinstance(schedule, list) else next(iter(schedule.values()))[0]


def _fjsp_rows(reference: dict) -> list[dict]:
    schedule = reference["schedule"]
    if isinstance(schedule, list):
        return schedule
    # 破壊を参照解へ反映させるため、コピーせず行そのものに job_id を書き足す
    rows = []
    for job, ops in schedule.items():
        for row in ops:
            row["job_id"] = int(job)
            rows.append(row)
    return rows


def _shift(row: dict, delta: float) -> None:
    for key in ("start", "start_time"):
        if key in row:
            row[key] += delta
    for key in ("end", "end_time"):
        if key in row:
            row[key] += delta


def _break_fjsp(pid: str, how: str) -> tuple[dict, dict, str]:
    record = _load(pid)
    instance, reference = record["instance"], copy.deepcopy(_reference(record))
    rows = _fjsp_rows(reference)
    if how == "drop_operation":
        schedule = reference["schedule"]
        if isinstance(schedule, list):
            schedule.pop()
        else:
            next(iter(schedule.values())).pop()
        return instance, reference, "not scheduled"
    if how == "wrong_machine":
        row = _first_op(reference)
        for key in ("machine", "machine_id"):
            if key in row:
                row[key] = -1
        return instance, reference, "not in its options"
    if how == "precedence":
        # 第 2 工程を第 1 工程の完了前へ動かす
        job = next(r for r in rows if r["op_index"] == 2)
        first = next(r for r in rows if r["op_index"] == 1 and r["job_id"] == job["job_id"])
        first_end = first.get("end", first.get("end_time"))
        _shift(job, (first_end - 1) - job.get("start", job.get("start_time")))
        return instance, reference, "starts"
    if how == "machine_overlap":
        # 同じ機械の 2 工程を重ねる
        by_machine: dict = {}
        for r in rows:
            by_machine.setdefault(r.get("machine", r.get("machine_id")), []).append(r)
        ops = next(v for v in by_machine.values() if len(v) >= 2)
        ops.sort(key=lambda r: r.get("start", r.get("start_time")))
        _shift(ops[1], ops[0].get("end", ops[0].get("end_time")) - 1
               - ops[1].get("start", ops[1].get("start_time")))
        return instance, reference, "machine"
    if how == "declared_objective":
        reference["makespan"] = reference["makespan"] * 1.1
        return instance, reference, "declared objective"
    raise AssertionError(how)


@pytest.mark.parametrize("pid", ["prob_305", "prob_315", "prob_322"])
@pytest.mark.parametrize(
    "how", ["drop_operation", "wrong_machine", "precedence", "machine_overlap", "declared_objective"]
)
def test_broken_fjsp_solution_is_reported(pid, how):
    instance, broken, needle = _break_fjsp(pid, how)
    result = find_kind(instance).check(instance, broken)
    assert result["verified"] is True
    assert result["feasible"] is False
    assert any(needle in v for v in result["violations"])


def test_fjsp_setup_rejects_insufficient_setup_gap():
    record = _load("prob_322")
    instance, reference = record["instance"], copy.deepcopy(_reference(record))
    setups = instance["setups"]
    by_machine: dict = {}
    for row in reference["schedule"]:
        by_machine.setdefault(row["machine_id"], []).append(row)
    # 正の段取り時間を持つ隣接ペアを見つけ、後続を段取り分だけ前倒しする
    for machine, ops in by_machine.items():
        ops.sort(key=lambda r: r["start_time"])
        for a, b in pairwise(ops):
            setup = setups[str(machine)][str(a["job_id"])][str(b["job_id"])]
            if setup > 0 and b["start_time"] - a["end_time"] >= setup:
                _shift(b, a["end_time"] - b["start_time"])
                # 前倒しで次の工程と重ならないよう、ジョブ内の後続もそのまま（違反は段取りで出る）
                result = find_kind(instance).check(instance, reference)
                assert result["feasible"] is False
                assert any(f"setup {setup}" in v for v in result["violations"])
                return
    raise AssertionError("no adjacent pair with positive setup time")


def test_fjsp_setup_rejects_processing_during_breakdown():
    record = _load("prob_322")
    instance, reference = record["instance"], copy.deepcopy(_reference(record))
    breakdown = instance["breakdowns"]["2"][0]
    row = next(r for r in reference["schedule"] if r["machine_id"] == 2)
    _shift(row, breakdown["start"] - row["start_time"])
    result = find_kind(instance).check(instance, reference)
    assert result["feasible"] is False
    assert any("breakdown" in v for v in result["violations"])


def _break_cutting(pid: str, how: str) -> tuple[dict, dict, str]:
    record = _load(pid)
    instance, reference = record["instance"], copy.deepcopy(_reference(record))
    patterns = reference["patterns"]
    if how == "drop_pattern":
        patterns.pop()
        return instance, reference, "demand"
    if how == "too_many_patterns":
        # 新しい品目 1 個だけの余分なパターンを足して種類数上限を超えさせる
        extra = copy.deepcopy(patterns[0])
        extra["runs"] = 1
        if "cuts" in extra:
            extra["cuts"] = {str(instance["items"][-1]["id"]): 1}
        else:
            extra["strips"] = [{"height": instance["items"][-1]["height"],
                                "items": {str(instance["items"][-1]["id"]): 1}}]
        patterns.append(extra)
        return instance, reference, "distinct patterns exceed"
    if how == "overfull":
        pat = patterns[0]
        if "cuts" in pat:
            item = next(iter(pat["cuts"]))
            pat["cuts"][item] += 100
            return instance, reference, "exceeds stock"
        strip = pat["strips"][0]
        item = next(iter(strip["items"]))
        strip["items"][item] += 100
        return instance, reference, "exceeds plate width"
    if how == "declared_objective":
        reference["total_cost"] *= 1.1
        return instance, reference, "declared objective"
    raise AssertionError(how)


@pytest.mark.parametrize("pid", ["prob_307", "prob_317", "prob_324"])
@pytest.mark.parametrize(
    "how", ["drop_pattern", "too_many_patterns", "overfull", "declared_objective"]
)
def test_broken_cutting_solution_is_reported(pid, how):
    instance, broken, needle = _break_cutting(pid, how)
    result = find_kind(instance).check(instance, broken)
    assert result["verified"] is True
    assert result["feasible"] is False
    assert any(needle in v for v in result["violations"])


def test_cutting_1d_cost_is_recomputed_from_stock_costs():
    record = _load("prob_307")
    instance, reference = record["instance"], copy.deepcopy(_reference(record))
    del reference["total_cost"]
    reference["patterns"][0]["runs"] += 1
    result = find_kind(instance).check(instance, reference)
    stock_cost = {s["id"]: s["cost"] for s in instance["stocks"]}
    expected = sum(p["runs"] * stock_cost[p["stock_id"]] for p in reference["patterns"])
    assert result["cost"] == pytest.approx(expected)
    assert result["feasible"] is True


def test_cutting_2d_rejects_min_lot_and_stacked_height():
    record = _load("prob_324")
    instance, reference = record["instance"], copy.deepcopy(_reference(record))
    reference["patterns"][0]["runs"] = instance["min_lot"] - 1
    reference["patterns"][1]["strips"][0]["height"] = instance["plates"][0]["height"]
    reference["patterns"][2]["strips"][0]["height"] = 1  # 品目より低いストリップ
    result = find_kind(instance).check(instance, reference)
    assert result["feasible"] is False
    joined = "\n".join(result["violations"])
    assert "min_lot" in joined
    assert "stacked strip height" in joined
    assert "exceeds strip height" in joined

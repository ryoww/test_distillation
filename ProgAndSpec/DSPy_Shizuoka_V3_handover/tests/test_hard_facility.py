"""facility 群（prob_304 / 314 / 328 / 329）の検証器が参照解を再現し、壊した解を弾くことを確かめる。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.utils.feasibility import check_feasibility_detailed
from src.utils.hard import find_kind
from src.utils.scorer import compute_score

BASE_DIR = Path(__file__).resolve().parents[1]
HARD_DIR = BASE_DIR / "data" / "problems_hard"
SHIPPED_DIR = BASE_DIR / "data" / "problems"

KIND_BY_ID = {
    "prob_304": "facility_multi",
    "prob_314": "facility_multi",
    "prob_328": "facility_2ech",
    "prob_329": "facility_robust",
}


def _load(pid: str) -> tuple[dict, dict, float]:
    record = json.loads((HARD_DIR / f"{pid}.json").read_text(encoding="utf-8"))
    ref = {
        k: v
        for k, v in record["reference_solution"].items()
        if k not in ("objective_value", "note")
    }
    return record, ref, float(record["reference_solution"]["objective_value"])


def _check(record: dict, solution: object) -> dict:
    core_type = f"{record['domain']}_{record['math_type']}"
    return check_feasibility_detailed(core_type, record["instance"], solution)


def _score(record: dict, solution: object) -> float | None:
    core_type = f"{record['domain']}_{record['math_type']}"
    return compute_score(core_type, record["instance"], solution)


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_detects_each_problem_as_its_kind(pid: str):
    record, _, _ = _load(pid)
    kind = find_kind(record["instance"])
    assert kind is not None and kind.name == KIND_BY_ID[pid]


@pytest.mark.parametrize("pid", ["prob_304", "prob_314", "prob_329"])
def test_reference_solution_is_feasible_and_reproduces_objective(pid: str):
    record, ref, objective = _load(pid)
    result = _check(record, ref)
    assert result["verified"] and result["feasible"], result["violations"][:3]
    assert result["violation_count"] == 0
    assert abs(result["cost"] - objective) <= 1e-3 * max(1.0, abs(objective))
    assert _score(record, ref) == pytest.approx(-result["cost"])


def test_prob_328_reference_double_counts_production_cost():
    # 参照解の objective_value は生産費を二重に数えている（差がちょうど生産費に一致する）。
    # 検証器は要件どおり生産費を 1 回だけ数えるので、申告値との不一致だけが違反として残る。
    record, ref, objective = _load("prob_328")
    result = _check(record, ref)
    assert result["verified"]
    assert result["violations"] == [
        f"declared total_cost {ref['total_cost']} differs from recomputed {result['cost']:.2f}"
    ]
    assert result["cost"] == pytest.approx(12172143.43, abs=0.01)
    assert result["cost"] < objective
    without_declared = {k: v for k, v in ref.items() if k != "total_cost"}
    clean = _check(record, without_declared)
    assert clean["feasible"] and clean["violation_count"] == 0
    assert _score(record, without_declared) == pytest.approx(-clean["cost"])


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_declared_objective_shift_is_reported(pid: str):
    record, ref, _ = _load(pid)
    broken = copy.deepcopy(ref)
    broken["total_cost"] = ref["total_cost"] * 0.9
    result = _check(record, broken)
    assert any("declared total_cost" in v for v in result["violations"])


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_dropped_customer_assignment_is_reported(pid: str):
    record, ref, _ = _load(pid)
    broken = copy.deepcopy(ref)
    del broken["customer_assignment"]["1"]
    result = _check(record, broken)
    assert result["verified"] and not result["feasible"]
    assert any("customer 1 is not assigned" in v for v in result["violations"])


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_assignment_to_non_candidate_is_reported(pid: str):
    record, ref, _ = _load(pid)
    customer = record["instance"]["customers"][0]
    cand_key = "candidate_dcs" if pid == "prob_328" else "candidate_facilities"
    pool_key = "dcs" if pid == "prob_328" else "facilities"
    outsider = next(
        f["id"] for f in record["instance"][pool_key] if f["id"] not in customer[cand_key]
    )
    broken = copy.deepcopy(ref)
    broken["customer_assignment"][str(customer["id"])] = outsider
    result = _check(record, broken)
    assert any("non-candidate" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_304", "prob_314"])
def test_procurement_over_throughput_is_reported(pid: str):
    record, ref, _ = _load(pid)
    fid = ref["opened_facilities"][0]
    facility = next(f for f in record["instance"]["facilities"] if f["id"] == fid)
    broken = copy.deepcopy(ref)
    broken["procurement_plan"][str(fid)][0] = facility["throughput_per_period"] + 10
    result = _check(record, broken)
    assert any("over throughput" in v for v in result["violations"])
    assert any("balance broken" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_304", "prob_314"])
def test_safety_stock_shortfall_is_reported(pid: str):
    record, ref, _ = _load(pid)
    fid = str(ref["opened_facilities"][0])
    broken = copy.deepcopy(ref)
    # 期末在庫を 0 にして調達も同じだけ減らすと収支は保たれ、安全在庫だけが破れる。
    shortfall = broken["inventory_plan"][fid][-1]
    broken["inventory_plan"][fid][-1] = 0.0
    broken["procurement_plan"][fid][-1] -= shortfall
    result = _check(record, broken)
    assert any("below safety stock" in v for v in result["violations"])
    assert not any("balance broken" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_304", "prob_314"])
def test_missing_plans_are_unverified(pid: str):
    record, ref, _ = _load(pid)
    broken = {k: v for k, v in ref.items() if k != "inventory_plan"}
    result = _check(record, broken)
    assert not result["verified"]


def test_closing_a_used_facility_is_reported_for_robust():
    record, ref, _ = _load("prob_329")
    used = ref["customer_assignment"]["1"]
    broken = copy.deepcopy(ref)
    broken["opened_facilities"] = [f for f in ref["opened_facilities"] if f != used]
    result = _check(record, broken)
    assert any(f"closed facility {used}" in v for v in result["violations"])


def test_robust_cost_is_worst_scenario():
    record, ref, _ = _load("prob_329")
    result = _check(record, ref)
    assert result["cost"] == pytest.approx(max(ref["scenario_costs"]), rel=1e-6)


def test_dc_on_closed_plant_is_reported_for_two_echelon():
    record, ref, _ = _load("prob_328")
    broken = copy.deepcopy(ref)
    broken["opened_plants"] = [p for p in ref["opened_plants"] if p != 4]
    result = _check(record, broken)
    assert any("closed plant 4" in v for v in result["violations"])


def test_closing_a_dc_with_customers_is_reported_for_two_echelon():
    record, ref, _ = _load("prob_328")
    dc = ref["customer_assignment"]["1"]
    broken = copy.deepcopy(ref)
    broken["opened_dcs"] = [d for d in ref["opened_dcs"] if d != dc]
    result = _check(record, broken)
    assert any(f"closed facility {dc}" in v for v in result["violations"])
    assert any(f"DC {dc} is assigned to a plant but not opened" in v for v in result["violations"])


def test_plant_capacity_shortfall_leaves_cost_undefined():
    record, ref, _ = _load("prob_328")
    instance = copy.deepcopy(record["instance"])
    # 第 1 期は前期在庫がないので、工場 4 の能力を配下需要より下げると生産計画が組めない。
    next(p for p in instance["plants"] if p["id"] == 4)["capacity_per_period"] = 1000
    result = check_feasibility_detailed("x", instance, ref)
    assert not result["feasible"] and result["cost"] is None
    assert any("plant 4: no feasible production plan" in v for v in result["violations"])


def test_facility_throughput_shortfall_leaves_cost_undefined_for_robust():
    record, ref, _ = _load("prob_329")
    instance = copy.deepcopy(record["instance"])
    next(f for f in instance["facilities"] if f["id"] == 30)["throughput_per_period"] = 50
    result = check_feasibility_detailed("x", instance, ref)
    assert not result["feasible"] and result["cost"] is None
    assert any("facility 30 cannot meet demand" in v for v in result["violations"])


def test_unreadable_solution_is_unverified():
    record, _, _ = _load("prob_304")
    assert not _check(record, [1, 2, 3])["verified"]
    assert not _check(record, {"opened_facilities": "1,2"})["verified"]


def test_shipped_problems_are_not_detected_as_hard_kinds():
    for path in sorted(SHIPPED_DIR.glob("prob_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert find_kind(record["instance"]) is None, path.name

"""production 群（clsp / prp / prp_tw）の大規模問題検証器のテスト。"""

from __future__ import annotations

import copy
import glob
import json
import os

import pytest

from src.utils.feasibility import check_feasibility_detailed
from src.utils.hard import KINDS, find_kind
from src.utils.scorer import compute_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PRODUCTION_KINDS = ("clsp", "prp", "prp_tw")
PROBLEMS = {"prob_301": "clsp", "prob_311": "clsp", "prob_320": "prp", "prob_327": "prp_tw"}


def _load(pid: str) -> tuple[dict, dict, dict, float]:
    with open(os.path.join(ROOT, "data", "problems_hard", f"{pid}.json"), encoding="utf-8") as f:
        record = json.load(f)
    ref = record["reference_solution"]
    solution = {k: v for k, v in ref.items() if k not in ("objective_value", "note")}
    return record, record["instance"], solution, float(ref["objective_value"])


def _check(instance: dict, solution) -> dict:
    kind = find_kind(instance)
    assert kind is not None
    return kind.check(instance, solution)


@pytest.mark.parametrize("pid,expected_kind", sorted(PROBLEMS.items()))
def test_reference_solution_is_feasible_and_reproduces_objective(pid, expected_kind):
    _record, instance, solution, objective = _load(pid)
    kind = find_kind(instance)
    assert kind is not None and kind.name == expected_kind
    result = kind.check(instance, solution)
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"][:5]
    assert result["violation_count"] == 0
    assert abs(result["cost"] - objective) <= 1e-3 * max(1.0, abs(objective))


@pytest.mark.parametrize("pid", sorted(PROBLEMS))
def test_feasibility_and_scorer_dispatch_to_hard_checker(pid):
    record, instance, solution, objective = _load(pid)
    core_type = f"{record['domain']}_{record['math_type']}"
    detailed = check_feasibility_detailed(core_type, instance, solution)
    assert detailed["feasible"] is True
    assert abs(detailed["cost"] - objective) <= 1e-3 * max(1.0, abs(objective))
    score = compute_score(core_type, instance, solution)
    assert score == pytest.approx(-detailed["cost"])


@pytest.mark.parametrize("pid", sorted(PROBLEMS))
def test_shifted_declared_objective_is_reported(pid):
    _record, instance, solution, objective = _load(pid)
    solution["objective_value"] = objective * 1.01
    result = _check(instance, solution)
    assert result["feasible"] is False
    assert any("declared objective_value" in v for v in result["violations"])


@pytest.mark.parametrize("pid", sorted(PROBLEMS))
def test_unreadable_solutions_are_unverified_not_infeasible(pid):
    _record, instance, _solution, _objective = _load(pid)
    for bad in (None, [], {}, {"production_plan": 3}, {"production_plan": [], "routes": "x"}):
        result = _check(instance, bad)
        assert result["verified"] is False
        assert result["feasible"] is True
        assert result["cost"] is None


# ---------------- clsp ----------------


@pytest.mark.parametrize("pid", ["prob_301", "prob_311"])
def test_clsp_incompatible_machine_is_a_violation(pid):
    _record, instance, solution, _objective = _load(pid)
    item = instance["items"][0]
    bad_machine = next(m["id"] for m in instance["machines"] if m["id"] not in item["compatible_machines"])
    solution["production_plan"][str(item["id"])][0]["machine"] = bad_machine
    result = _check(instance, solution)
    assert result["feasible"] is False
    assert any("incompatible machines" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_301", "prob_311"])
def test_clsp_exceeding_machine_capacity_is_a_violation(pid):
    _record, instance, solution, _objective = _load(pid)
    solution["production_plan"]["1"][0]["qty"] = 1e6
    result = _check(instance, solution)
    assert result["feasible"] is False
    assert any("> capacity" in v for v in result["violations"])
    assert result["cost"] is not None


@pytest.mark.parametrize("pid", ["prob_301", "prob_311"])
def test_clsp_lot_outside_horizon_and_wrong_setup_count_are_violations(pid):
    _record, instance, solution, _objective = _load(pid)
    solution["production_plan"]["1"][0]["period"] = instance["num_periods"] + 1
    result = _check(instance, solution)
    assert any("outside periods" in v for v in result["violations"])
    assert any("num_setups" in v for v in result["violations"])


def test_clsp_missing_item_counts_its_demand_as_lost_sales():
    _record, instance, solution, objective = _load("prob_311")
    # 参照解は品目 69 を計画に含めておらず、その需要は全て欠品として費用に入る。
    assert "69" not in solution["production_plan"]
    result = _check(instance, solution)
    assert result["feasible"] is True
    assert result["cost"] == pytest.approx(objective, rel=1e-6)
    explicit = copy.deepcopy(solution)
    explicit["production_plan"]["69"] = []
    assert _check(instance, explicit)["cost"] == pytest.approx(result["cost"])
    dropped = copy.deepcopy(solution)
    del dropped["production_plan"]["1"]
    assert _check(instance, dropped)["cost"] > result["cost"]


# ---------------- prp / prp_tw ----------------


@pytest.mark.parametrize("pid", ["prob_320", "prob_327"])
def test_prp_overloaded_vehicle_is_a_violation(pid):
    _record, instance, solution, _objective = _load(pid)
    solution["routes"][0]["quantities"][0] += instance["vehicle_capacity"]
    result = _check(instance, solution)
    assert result["feasible"] is False
    assert any("> vehicle capacity" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_320", "prob_327"])
def test_prp_dropping_a_route_causes_customer_stockouts(pid):
    _record, instance, solution, _objective = _load(pid)
    solution["routes"].pop(0)
    result = _check(instance, solution)
    assert result["feasible"] is False
    assert any("stockout" in v for v in result["violations"])
    assert any("declared end_inventory" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_320", "prob_327"])
def test_prp_production_without_setup_is_a_violation(pid):
    _record, instance, solution, _objective = _load(pid)
    entry = next(e for e in solution["production_plan"] if e["production"] > 0)
    entry["setup"] = 0
    result = _check(instance, solution)
    assert result["violation_count"] == 1
    assert "without a setup" in result["violations"][0]


@pytest.mark.parametrize("pid", ["prob_320", "prob_327"])
def test_prp_too_many_vehicles_per_plant_period_is_a_violation(pid):
    _record, instance, solution, _objective = _load(pid)
    first = solution["routes"][0]
    for _ in range(instance["max_vehicles_per_plant"]):
        solution["routes"].append({**first, "customers": [], "quantities": []})
    result = _check(instance, solution)
    assert any("vehicles >" in v for v in result["violations"])


def test_prp_tw_reversed_route_breaks_time_window_and_route_time():
    _record, instance, solution, _objective = _load("prob_327")
    route = solution["routes"][0]
    route["customers"] = route["customers"][::-1]
    route["quantities"] = route["quantities"][::-1]
    result = _check(instance, solution)
    assert result["feasible"] is False
    assert any("after the window" in v for v in result["violations"])
    assert any("max route time" in v for v in result["violations"])


def test_prp_allocation_inconsistent_with_routes_is_a_violation():
    _record, instance, solution, _objective = _load("prob_320")
    route = solution["routes"][0]
    customer, period = str(route["customers"][0]), str(route["period"])
    solution["allocation"][customer][period] = {}
    result = _check(instance, solution)
    assert any("allocation differs" in v for v in result["violations"])


def test_prp_route_quantities_define_deliveries_when_allocation_is_absent():
    _record, instance, solution, objective = _load("prob_320")
    del solution["allocation"]
    result = _check(instance, copy.deepcopy(solution))
    assert result["feasible"] is True
    assert result["cost"] == pytest.approx(objective, rel=1e-6)


# ---------------- 誤検知 ----------------


def test_production_kinds_do_not_match_shipped_problems():
    detectors = [k for k in KINDS if k.name in PRODUCTION_KINDS]
    assert len(detectors) == len(PRODUCTION_KINDS)
    files = sorted(glob.glob(os.path.join(ROOT, "data", "problems", "*.json")))
    assert files
    for path in files:
        with open(path, encoding="utf-8") as f:
            instance = json.load(f).get("instance")
        kind = find_kind(instance)
        assert kind is None, path
        assert not any(k.detect(instance) for k in detectors if isinstance(instance, dict)), path

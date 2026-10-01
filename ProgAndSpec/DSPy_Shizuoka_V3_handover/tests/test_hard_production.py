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
    bad_shapes = [None, [], {}, {"production_plan": 3}, {"production_plan": "x"}]
    if PROBLEMS[pid] != "clsp":
        bad_shapes.append({"production_plan": [], "routes": "x"})
    for bad in bad_shapes:
        result = _check(instance, bad)
        assert result["verified"] is False
        assert result["feasible"] is True
        assert result["cost"] is None


@pytest.mark.parametrize("pid", sorted(PROBLEMS))
def test_empty_plan_is_read_and_its_violations_are_counted(pid):
    _record, instance, _solution, _objective = _load(pid)
    empty = {"production_plan": [], "routes": []}
    result = _check(instance, empty)
    assert result["verified"] is True
    assert result["cost"] is not None
    if PROBLEMS[pid] == "clsp":
        lost = sum(it["lost_sale_cost"] * sum(it["demand"]) for it in instance["items"])
        assert result["cost"] == pytest.approx(lost)
    else:
        assert any("stockout" in v for v in result["violations"])


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


def _clsp_records(plan: dict) -> list[dict]:
    """参照解の {item: [{machine, period, qty}]} を品目 id 付きのフラットな record 列にする。"""
    return [
        {"item_id": int(item), "machine_id": lot["machine"], "period": lot["period"], "quantity": lot["qty"]}
        for item, lots in plan.items()
        for lot in lots
    ]


def test_clsp_flat_record_list_with_item_ids_reproduces_reference():
    _record, instance, solution, objective = _load("prob_301")
    records = _clsp_records(solution["production_plan"])
    # 保存解にあった余分な 0 始まりの period_index は無視し、1 始まりの period を読む。
    records[0]["period_index"] = records[0]["period"] - 1
    solution["production_plan"] = records
    result = _check(instance, solution)
    assert result["feasible"] is True, result["violations"][:3]
    assert result["cost"] == pytest.approx(objective, rel=1e-6)


def test_clsp_record_without_item_id_is_unverified():
    _record, instance, solution, _objective = _load("prob_301")
    solution["production_plan"] = [{"machine": 4, "period": 5, "qty": 273.0}]
    assert _check(instance, solution)["verified"] is False


def test_clsp_per_item_production_dict_with_auxiliary_series_is_read():
    _record, instance, solution, objective = _load("prob_311")
    # 保存解の形: {item: {"production": [{period, machine, quantity}], "inventory": [...], "lost_sales": [...]}}
    solution["production_plan"] = {
        item: {
            "production": [
                {"period": lot["period"], "machine": lot["machine"], "quantity": lot["qty"]}
                for lot in lots
            ],
            "inventory": [0.0] * instance["num_periods"],
            "lost_sales": [0.0] * instance["num_periods"],
        }
        for item, lots in solution["production_plan"].items()
    }
    result = _check(instance, solution)
    assert result["feasible"] is True, result["violations"][:3]
    assert result["cost"] == pytest.approx(objective, rel=1e-6)


def test_clsp_period_ordered_machine_dicts_are_read_only_at_horizon_length():
    _record, instance, solution, objective = _load("prob_311")
    num_periods = instance["num_periods"]
    # 保存解の形: {item: [{machine: qty, ...} per period]}（index 0 が第 1 期）。
    positional = {}
    for item, lots in solution["production_plan"].items():
        periods = [{} for _ in range(num_periods)]
        for lot in lots:
            periods[lot["period"] - 1][str(lot["machine"])] = lot["qty"]
        positional[item] = periods
    solution["production_plan"] = positional
    result = _check(instance, copy.deepcopy(solution))
    assert result["feasible"] is True, result["violations"][:3]
    assert result["cost"] == pytest.approx(objective, rel=1e-6)
    solution["production_plan"]["1"].append({})
    assert _check(instance, solution)["verified"] is False


def test_clsp_capacity_check_tolerates_decimal_rounding_but_not_real_overload():
    _record, instance, _solution, _objective = _load("prob_301")
    item = instance["items"][0]
    machine = next(m for m in instance["machines"] if m["id"] == item["compatible_machines"][0])
    fill = (machine["capacity_per_period"] - item["setup_time"]) / item["unit_process_time"]
    plan = {str(item["id"]): [{"machine": machine["id"], "period": 1, "qty": round(fill + 5e-7, 6)}]}
    assert _check(instance, {"production_plan": plan})["feasible"] is True
    plan[str(item["id"])][0]["qty"] = fill + 1e-3
    assert any("> capacity" in v for v in _check(instance, {"production_plan": plan})["violations"])


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


def _assert_reproduces(instance: dict, solution: dict, objective: float) -> dict:
    result = _check(instance, solution)
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"][:3]
    assert result["cost"] == pytest.approx(objective, rel=1e-6)
    return result


@pytest.mark.parametrize("pid", ["prob_320", "prob_327"])
def test_prp_plan_records_with_synonym_keys_are_read(pid):
    _record, instance, solution, objective = _load(pid)
    # 保存解の形: plant_id / quantity / inventory、setup は bool。
    solution["production_plan"] = [
        {
            "plant_id": e["plant"],
            "period": e["period"],
            "quantity": e["production"],
            "setup": bool(e["setup"]),
            "inventory": e["end_inventory"],
        }
        for e in solution["production_plan"]
    ]
    _assert_reproduces(instance, solution, objective)


def _plan_by_plant(plan: list[dict], num_periods: int) -> dict:
    """参照解の record 列を {plant: {production: [期順], setup: [期順], inventory: [期順]}} にする。"""
    by_plant: dict = {}
    for e in plan:
        series = by_plant.setdefault(
            str(e["plant"]),
            {"production": [0.0] * num_periods, "setup": [0] * num_periods, "inventory": [0.0] * num_periods},
        )
        idx = e["period"] - 1
        series["production"][idx] = e["production"]
        series["setup"][idx] = e["setup"]
        series["inventory"][idx] = e["end_inventory"]
    return by_plant


def test_prp_tw_per_plant_period_series_plan_is_read_only_at_horizon_length():
    _record, instance, solution, objective = _load("prob_327")
    solution["production_plan"] = _plan_by_plant(solution["production_plan"], instance["periods"])
    _assert_reproduces(instance, copy.deepcopy(solution), objective)
    solution["production_plan"]["1"]["setup"].append(0)
    assert _check(instance, solution)["verified"] is False


def test_prp_zero_based_periods_are_reported_as_violations_not_shifted():
    _record, instance, solution, _objective = _load("prob_320")
    for route in solution["routes"]:
        route["period"] -= 1
    result = _check(instance, solution)
    assert result["verified"] is True
    assert result["feasible"] is False
    assert any("period 0) has an unknown plant or period" in v for v in result["violations"])


def _allocation_records(allocation: dict) -> list[dict]:
    """参照解の {customer: {period: {plant: qty}}} を record 列にする。"""
    return [
        {"customer": int(c), "plant": int(p), "period": int(t), "quantity": q}
        for c, per_period in allocation.items()
        for t, by_plant in per_period.items()
        for p, q in by_plant.items()
    ]


def test_prp_allocation_record_list_is_compared_with_route_deliveries():
    _record, instance, solution, objective = _load("prob_320")
    solution["allocation"] = _allocation_records(solution["allocation"])
    _assert_reproduces(instance, copy.deepcopy(solution), objective)
    solution["allocation"][0]["quantity"] += 1
    assert any("allocation differs" in v for v in _check(instance, solution)["violations"])


def test_prp_allocation_records_without_quantities_are_not_compared():
    _record, instance, solution, objective = _load("prob_320")
    # 保存解の形: 顧客→工場の割当だけで配送量が無い。照合できないので読み飛ばす。
    solution["allocation"] = [{"customer_id": 1, "plant_id": 3}]
    _assert_reproduces(instance, solution, objective)


def test_prp_tw_allocation_series_by_plant_then_customer_is_oriented_by_ids():
    _record, instance, solution, objective = _load("prob_327")
    num_periods = instance["periods"]
    by_plant: dict = {}
    for rec in _allocation_records(solution["allocation"]):
        row = by_plant.setdefault(str(rec["plant"]), {}).setdefault(str(rec["customer"]), [0.0] * num_periods)
        row[rec["period"] - 1] += rec["quantity"]
    solution["allocation"] = by_plant
    _assert_reproduces(instance, copy.deepcopy(solution), objective)
    next(iter(by_plant["1"].values()))[0] += 1
    assert any("allocation differs" in v for v in _check(instance, solution)["violations"])
    # 工場 id とも顧客 id とも取れる {1: {2: [...]}} は向きが決まらないので照合しない。
    solution["allocation"] = {"1": {"2": [999.0] * num_periods}}
    _assert_reproduces(instance, solution, objective)


def test_prp_tw_allocation_deliveries_by_period_is_read():
    _record, instance, solution, objective = _load("prob_327")
    # 保存解の形: {customer: {"plant": p, "deliveries": {period: qty}, "inventory": [...]}}
    deliveries: dict = {}
    for rec in _allocation_records(solution["allocation"]):
        entry = deliveries.setdefault(str(rec["customer"]), {"plant": rec["plant"], "deliveries": {}})
        entry["deliveries"][str(rec["period"])] = entry["deliveries"].get(str(rec["period"]), 0.0) + rec["quantity"]
    solution["allocation"] = deliveries
    _assert_reproduces(instance, solution, objective)


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

"""追加した20 core_type ぶんの制約チェッカーの振る舞いを検証する。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.utils.feasibility import CHECKERS_DETAILED, check_feasibility_detailed
from src.utils.feasibility_v3_ext import EXTRA_CHECKERS

BASE_DIR = Path(__file__).resolve().parents[1]
PROBLEM_DIR = BASE_DIR / "data" / "problems"

# 目的値を再計算で確定できない問題。理由はモジュールの docstring に記載。
UNPINNED_OBJECTIVE_IDS = {31, 53, 87}


def _load_problems() -> list[dict]:
    records = []
    for path in sorted(PROBLEM_DIR.glob("prob_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["core_type"] = f"{record['domain']}_{record['math_type']}"
        records.append(record)
    return records


PROBLEMS = _load_problems()
EXT_PROBLEMS = [r for r in PROBLEMS if r["core_type"] in EXTRA_CHECKERS]


def _check(record: dict, solution: object) -> dict:
    return check_feasibility_detailed(record["core_type"], record["instance"], solution)


def test_extension_registers_twenty_core_types():
    assert len(EXTRA_CHECKERS) == 20


def test_every_problem_core_type_has_a_checker():
    uncovered = sorted({r["core_type"] for r in PROBLEMS} - set(CHECKERS_DETAILED))
    assert uncovered == []


def test_extension_covers_seventy_problems():
    assert len(EXT_PROBLEMS) == 70


@pytest.mark.parametrize("record", EXT_PROBLEMS, ids=lambda r: f"prob_{r['id']:03d}")
def test_reference_solution_is_verified_and_feasible(record):
    result = _check(record, record["reference_solution"])
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]
    assert result["violation_count"] == 0


@pytest.mark.parametrize(
    "record",
    [r for r in EXT_PROBLEMS if r["id"] not in UNPINNED_OBJECTIVE_IDS],
    ids=lambda r: f"prob_{r['id']:03d}",
)
def test_corrupted_objective_is_rejected(record):
    solution = copy.deepcopy(record["reference_solution"])
    key = next(
        (k for k, v in solution.items() if isinstance(v, (int, float)) and not isinstance(v, bool)),
        None,
    )
    if key is None:
        pytest.skip("no numeric objective field")
    solution[key] = solution[key] * 0.5 - 7 if solution[key] else -13
    result = _check(record, solution)
    assert result["verified"] is True
    assert result["feasible"] is False


def test_unrecognized_solution_shape_stays_unverified():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 55)
    result = _check(record, {"unexpected_field": 1})
    assert result["verified"] is False
    assert result["feasible"] is True


def test_knapsack_rejects_capacity_overflow():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 55)
    every_item = list(range(len(record["instance"]["items"])))
    result = _check(record, {"chosen_items": every_item, "max_value": 999})
    assert result["feasible"] is False
    assert any("exceeds capacity" in v for v in result["violations"])


def test_knapsack_rejects_duplicate_items():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 55)
    result = _check(record, {"chosen_items": [0, 0], "max_value": 0})
    assert any("duplicates" in v for v in result["violations"])


def test_max_flow_rejects_value_above_the_source_cut():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 31)
    result = _check(record, {"max_flow": 10_000})
    assert result["feasible"] is False
    assert any("upper bound" in v for v in result["violations"])


def test_set_cover_rejects_uncovered_items():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 43)
    result = _check(record, {"chosen_sets": [0], "min_cost": 5})
    assert result["feasible"] is False
    assert any("not covered" in v for v in result["violations"])


def test_graph_coloring_rejects_adjacent_same_color():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 90)
    num_nodes = record["instance"]["num_nodes"]
    result = _check(
        record, {"coloring": dict.fromkeys(map(str, range(num_nodes)), 0), "min_colors": 1}
    )
    assert result["feasible"] is False
    assert any("share color" in v for v in result["violations"])


def test_clique_rejects_non_adjacent_members():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 99)
    result = _check(record, {"clique": [0, 1, 2], "max_clique_size": 3})
    assert result["feasible"] is False
    assert any("not adjacent" in v for v in result["violations"])


def test_tsp_rejects_incomplete_tour():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 89)
    result = _check(record, {"tour": [0, 1], "min_distance": 1})
    assert result["feasible"] is False
    assert any("visits" in v for v in result["violations"])


def test_stable_marriage_rejects_a_blocking_pair():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 69)
    size = record["instance"]["num_pairs"]
    # 各男性に自分の最下位の相手を割り当てると、必ずブロッキングペアが生じる。
    matching = {str(m): record["instance"]["men_preferences"][m][-1] for m in range(size)}
    result = _check(record, {"stable_matching": matching})
    assert result["verified"] is True
    assert result["feasible"] is False


def test_newsvendor_rejects_wrong_expected_profit():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 80)
    quantity = record["reference_solution"]["optimal_order_quantity"]
    result = _check(record, {"optimal_order_quantity": quantity, "expected_profit": 10_000})
    assert result["feasible"] is False
    assert any("expected_profit" in v for v in result["violations"])


def test_duty_roster_accepts_both_key_orientations():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 74)
    reference = record["reference_solution"]
    by_staff = _check(record, reference)
    assert by_staff["feasible"] is True

    by_day: dict[str, list[int]] = {str(day): [] for day in range(record["instance"]["num_days"])}
    for staff, days in reference["schedule"].items():
        for day in days:
            by_day[str(day)].append(int(staff))
    flipped = {**reference, "schedule": by_day}
    assert _check(record, flipped)["feasible"] is True


def test_production_plan_rejects_capacity_overflow():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 84)
    capacity = record["instance"]["production_capacity"]
    plan = [capacity * 10] * record["instance"]["periods"]
    result = _check(record, {"production_plan": plan, "min_cost": 1})
    assert result["feasible"] is False
    assert any("exceeds capacity" in v for v in result["violations"])


def test_strip_packing_rejects_overlap():
    record = next(r for r in EXT_PROBLEMS if r["id"] == 62)
    rectangles = record["instance"]["rectangles"]
    stacked = [{"id": r["id"], "x": 0, "y": 0, "w": r["w"], "h": r["h"]} for r in rectangles]
    result = _check(record, {"placement": stacked, "min_height": 99})
    assert result["feasible"] is False
    assert any("overlap" in v for v in result["violations"])


# --- 実際のモデル出力で観測された形状ゆれ ---
#
# 参照解と違う形で返された正解を infeasible に落とすと、未検証のまま放置するより
# 悪い結果になる。観測済みの形をここで固定する。


def _problem(problem_id: int) -> dict:
    return next(r for r in EXT_PROBLEMS if r["id"] == problem_id)


def test_unit_commitment_accepts_an_output_only_series():
    record = _problem(100)
    reference = record["reference_solution"]["schedule"]
    schedule = {unit: entry["output"] for unit, entry in reference.items()}
    result = _check(
        record, {"min_cost": record["reference_solution"]["min_cost"], "schedule": schedule}
    )
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_unit_commitment_accepts_per_period_records():
    record = _problem(100)
    reference = record["reference_solution"]["schedule"]
    schedule = {
        unit: [
            {"period": period, "output": value, "on": bool(entry["on"][period])}
            for period, value in enumerate(entry["output"])
        ]
        for unit, entry in reference.items()
    }
    result = _check(
        record, {"min_cost": record["reference_solution"]["min_cost"], "schedule": schedule}
    )
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_bipartite_matching_accepts_plain_pairs():
    record = _problem(70)
    reference = record["reference_solution"]
    pairs = [[m["left"], m["right"]] for m in reference["matching"]]
    result = _check(record, {"max_weight": reference["max_weight"], "matching": pairs})
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_selective_assignment_accepts_plain_pairs():
    record = _problem(76)
    reference = record["reference_solution"]
    pairs = [[p["agent"], p["job"]] for p in reference["pairs"]]
    result = _check(record, {"max_profit": reference["max_profit"], "pairs": pairs})
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_classroom_assignment_accepts_room_slot_pairs():
    record = _problem(75)
    reference = record["reference_solution"]
    assignment = {course: [v["room"], v["slot"]] for course, v in reference["assignment"].items()}
    result = _check(
        record, {"max_preference": reference["max_preference"], "assignment": assignment}
    )
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_bond_holdings_accept_id_quantity_records():
    record = _problem(82)
    reference = record["reference_solution"]
    holdings = [
        {"id": int(bond), "quantity": amount} for bond, amount in reference["bond_holdings"].items()
    ]
    result = _check(record, {"min_cost": reference["min_cost"], "bond_holdings": holdings})
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_workforce_plan_accepts_a_bare_level_series():
    record = _problem(85)
    reference = record["reference_solution"]
    levels = [entry["workforce"] for entry in reference["plan"]]
    result = _check(record, {"min_cost": reference["min_cost"], "plan": levels})
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_workforce_plan_accepts_plural_hire_and_fire_keys():
    record = _problem(85)
    reference = record["reference_solution"]
    plan = [
        {
            "period": entry["period"] + 1,
            "workforce": entry["workforce"],
            "hires": entry["hire"],
            "fires": entry["fire"],
        }
        for entry in reference["plan"]
    ]
    result = _check(record, {"min_cost": reference["min_cost"], "plan": plan})
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_shipments_accept_a_plant_by_customer_matrix():
    record = _problem(94)
    reference = record["reference_solution"]
    plants = record["instance"]["num_plants"]
    customers = record["instance"]["num_customers"]
    matrix = [[0.0] * customers for _ in range(plants)]
    for plant, row in reference["shipments"].items():
        for customer, qty in row.items():
            matrix[int(plant)][int(customer)] = qty
    result = _check(record, {"min_total_cost": reference["min_total_cost"], "shipments": matrix})
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_shipments_accept_flow_records():
    record = _problem(94)
    reference = record["reference_solution"]
    flows = [
        {"plant": int(plant), "customer": int(customer), "amount": qty}
        for plant, row in reference["shipments"].items()
        for customer, qty in row.items()
    ]
    result = _check(record, {"min_total_cost": reference["min_total_cost"], "shipments": flows})
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]


def test_cash_flow_rejects_a_sentinel_cost_that_contradicts_the_holdings():
    """実測で出た「緊急フォールバック」解。保有量と申告費用が食い違う。"""
    record = _problem(82)
    result = _check(
        record,
        {"min_cost": 9999.0, "feasible": False, "bond_holdings": {"0": 10.0, "1": 10.0, "2": 10.0}},
    )
    assert result["verified"] is True
    assert result["feasible"] is False

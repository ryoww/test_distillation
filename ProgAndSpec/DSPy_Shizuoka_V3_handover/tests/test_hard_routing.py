"""routing 群（vrptw_md: prob_303/313、pdptw: prob_321）の検証器の振る舞いを固定する。"""

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
BUNDLED_DIR = BASE_DIR / "data" / "problems"


def _load(problem_id: str) -> dict:
    record = json.loads((HARD_DIR / f"{problem_id}.json").read_text(encoding="utf-8"))
    record["core_type"] = f"{record['domain']}_{record['math_type']}"
    record["ref"] = {
        k: v for k, v in record["reference_solution"].items() if k not in ("objective_value", "note")
    }
    return record


def _check(record: dict, solution: object) -> dict:
    return find_kind(record["instance"]).check(record["instance"], solution)


def _assert_violation(result: dict, keyword: str) -> None:
    assert not result["feasible"]
    assert any(keyword in v for v in result["violations"]), result["violations"]


# ----------------------------------------------------------------------------
# 種別判定
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("problem_id", "kind"), [("prob_303", "vrptw_md"), ("prob_313", "vrptw_md"), ("prob_321", "pdptw")]
)
def test_detects_routing_kinds_from_instance_keys(problem_id, kind):
    assert find_kind(_load(problem_id)["instance"]).name == kind


def test_bundled_problems_are_not_detected_as_routing_kinds():
    for path in sorted(BUNDLED_DIR.glob("prob_*.json")):
        instance = json.loads(path.read_text(encoding="utf-8"))["instance"]
        kind = find_kind(instance)
        assert kind is None or kind.name not in ("vrptw_md", "pdptw"), path.name


# ----------------------------------------------------------------------------
# 参照解の再現と採点系への振り分け
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("problem_id", ["prob_303", "prob_313", "prob_321"])
def test_reference_solution_is_feasible_and_reproduces_objective(problem_id):
    record = _load(problem_id)
    result = _check(record, record["ref"])
    objective = record["reference_solution"]["objective_value"]
    assert result["feasible"] and result["verified"]
    assert result["violation_count"] == 0
    assert abs(result["cost"] - objective) <= 1e-3 * max(1.0, abs(objective))


@pytest.mark.parametrize("problem_id", ["prob_303", "prob_313", "prob_321"])
def test_scorer_and_feasibility_route_to_hard_checker(problem_id):
    record = _load(problem_id)
    detailed = check_feasibility_detailed(record["core_type"], record["instance"], record["ref"])
    score = compute_score(record["core_type"], record["instance"], record["ref"])
    assert detailed["feasible"] and detailed["verified"]
    assert score == pytest.approx(-detailed["cost"])


@pytest.mark.parametrize("problem_id", ["prob_303", "prob_313", "prob_321"])
def test_unparseable_solution_is_unverified_without_violations(problem_id):
    record = _load(problem_id)
    for bad in (None, [], {"routes": "x"}, {"routes": [{"depot": 1}]}):
        result = _check(record, bad)
        assert result["verified"] is False
        assert result["violation_count"] == 0


# ----------------------------------------------------------------------------
# vrptw_md: 参照解を 1 箇所壊す
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("problem_id", ["prob_303", "prob_313"])
def test_vrptw_declared_cost_mismatch_is_reported(problem_id):
    record = _load(problem_id)
    solution = copy.deepcopy(record["ref"])
    solution["total_cost"] *= 0.98
    _assert_violation(_check(record, solution), "declared total_cost")


@pytest.mark.parametrize("problem_id", ["prob_303", "prob_313"])
def test_vrptw_dropped_customer_changes_cost_and_unserved_list(problem_id):
    record = _load(problem_id)
    solution = copy.deepcopy(record["ref"])
    solution["routes"][0]["customers"].pop()
    result = _check(record, solution)
    _assert_violation(result, "unserved_customers disagrees")
    # 落とした顧客ぶんのペナルティが再計算に乗る。
    assert result["cost"] > record["reference_solution"]["objective_value"]


@pytest.mark.parametrize("problem_id", ["prob_303", "prob_313"])
def test_vrptw_overloaded_route_is_reported(problem_id):
    record = _load(problem_id)
    solution = copy.deepcopy(record["ref"])
    solution["routes"][0]["customers"] *= 4
    result = _check(record, solution)
    _assert_violation(result, "exceeds capacity")
    _assert_violation(result, "visited more than once")


@pytest.mark.parametrize("problem_id", ["prob_303", "prob_313"])
def test_vrptw_fleet_limit_is_reported(problem_id):
    record = _load(problem_id)
    solution = copy.deepcopy(record["ref"])
    for route in solution["routes"]:
        route["depot"] = record["instance"]["depots"][0]["id"]
        route["vehicle_type"] = "small"
    _assert_violation(_check(record, solution), "vehicles but holds")


def test_vrptw_unknown_vehicle_type_is_reported():
    record = _load("prob_303")
    solution = copy.deepcopy(record["ref"])
    solution["routes"][0]["vehicle_type"] = "jumbo"
    _assert_violation(_check(record, solution), "unknown depot or vehicle_type")


def test_vrptw_accepts_sequence_key_for_route_stops():
    record = _load("prob_303")
    solution = copy.deepcopy(record["ref"])
    for route in solution["routes"]:
        route["sequence"] = route.pop("customers")
    assert _check(record, solution)["feasible"]


# ----------------------------------------------------------------------------
# vrptw_md: 時間制約の意味づけ（小さな手作り instance で固定する）
# ----------------------------------------------------------------------------


def _tiny_vrptw(customers: list[dict]) -> dict:
    return {
        "depots": [
            {"id": 1, "x": 0.0, "y": 0.0, "open_time": 6.0, "close_time": 21.0, "fleet": {"van": 1}}
        ],
        "customers": customers,
        "vehicle_types": [
            {
                "type": "van",
                "capacity": 100,
                "fixed_cost": 100,
                "cost_per_km": 2,
                "speed_kmh": 10,
                "max_route_hours": 5,
            }
        ],
        "unserved_penalty": 1000,
        "distance": "ユークリッド距離(km)、移動時間 = 距離 / 車種の平均速度",
    }


def _tiny_solution(customer_ids: list[int]) -> dict:
    return {"routes": [{"depot": 1, "vehicle_type": "van", "customers": customer_ids}]}


def test_vrptw_cost_is_fixed_plus_distance_plus_penalty():
    instance = _tiny_vrptw(
        [
            {"id": 1, "x": 10.0, "y": 0.0, "demand": 1, "tw_start": 6, "tw_end": 20, "service_time": 0},
            {"id": 2, "x": 0.0, "y": 10.0, "demand": 1, "tw_start": 6, "tw_end": 20, "service_time": 0},
        ]
    )
    result = find_kind(instance).check(instance, _tiny_solution([1]))
    assert result["feasible"]
    assert result["cost"] == pytest.approx(100 + 2 * 20 + 1000)


def test_vrptw_waiting_before_window_can_be_absorbed_by_delayed_departure():
    # 到着 7 時、時間枠 12 時開始: 最早出発だと 7 時間拘束だが、出発を遅らせれば 2 時間で済む。
    instance = _tiny_vrptw(
        [{"id": 1, "x": 10.0, "y": 0.0, "demand": 1, "tw_start": 12, "tw_end": 13, "service_time": 0}]
    )
    assert find_kind(instance).check(instance, _tiny_solution([1]))["feasible"]


def test_vrptw_route_longer_than_max_hours_is_reported():
    instance = _tiny_vrptw(
        [{"id": 1, "x": 30.0, "y": 0.0, "demand": 1, "tw_start": 6, "tw_end": 20, "service_time": 0}]
    )
    _assert_violation(find_kind(instance).check(instance, _tiny_solution([1])), "route duration")


def test_vrptw_late_arrival_and_late_return_are_reported():
    instance = _tiny_vrptw(
        [
            {"id": 1, "x": 10.0, "y": 0.0, "demand": 1, "tw_start": 6, "tw_end": 6.5, "service_time": 0},
            {"id": 2, "x": 10.0, "y": 0.0, "demand": 1, "tw_start": 20.5, "tw_end": 21, "service_time": 0},
        ]
    )
    _assert_violation(find_kind(instance).check(instance, _tiny_solution([1])), "after tw_end")
    _assert_violation(find_kind(instance).check(instance, _tiny_solution([2])), "after depot close")


# ----------------------------------------------------------------------------
# pdptw
# ----------------------------------------------------------------------------


def test_pdptw_declared_cost_mismatch_is_reported():
    record = _load("prob_321")
    solution = copy.deepcopy(record["ref"])
    solution["total_cost"] *= 0.98
    _assert_violation(_check(record, solution), "declared total_cost")


def test_pdptw_delivery_before_pickup_is_reported():
    record = _load("prob_321")
    solution = copy.deepcopy(record["ref"])
    solution["routes"][0]["stops"].reverse()
    _assert_violation(_check(record, solution), "pickup must precede delivery")


def test_pdptw_pickup_without_delivery_is_reported():
    record = _load("prob_321")
    solution = copy.deepcopy(record["ref"])
    solution["routes"][0]["stops"] = solution["routes"][0]["stops"][:1]
    _assert_violation(_check(record, solution), "pickup must precede delivery")


def test_pdptw_pair_split_across_routes_is_reported():
    record = _load("prob_321")
    solution = copy.deepcopy(record["ref"])
    solution["routes"][1]["stops"].extend(solution["routes"][0]["stops"])
    _assert_violation(_check(record, solution), "served more than once or split")


def test_pdptw_dropped_pair_changes_cost_and_unserved_list():
    record = _load("prob_321")
    solution = copy.deepcopy(record["ref"])
    solution["routes"].pop()
    result = _check(record, solution)
    _assert_violation(result, "unserved_pairs disagrees")
    assert result["cost"] != pytest.approx(record["reference_solution"]["objective_value"])


def _tiny_pdptw() -> dict:
    return {
        "depots": [
            {"id": 1, "x": 0.0, "y": 0.0, "open_time": 6.0, "close_time": 20.0, "fleet": {"van": 1}}
        ],
        "vehicle_types": [
            {
                "type": "van",
                "capacity": 80,
                "fixed_cost": 100,
                "cost_per_km": 1,
                "speed_kmh": 10,
                "max_route_hours": 8,
            }
        ],
        "pairs": [
            {
                "id": 1,
                "pickup": {"x": 5.0, "y": 0.0, "demand": 50, "tw_start": 6, "tw_end": 20},
                "delivery": {"x": 10.0, "y": 0.0, "demand": 40, "tw_start": 6, "tw_end": 20},
            }
        ],
        "unserved_penalty": 1000,
        "note": "precedence: pickup 訪問後に delivery 訪問; 積載は累計需要の最大値",
    }


def _pdptw_route(stops: list) -> dict:
    return {"routes": [{"depot": 1, "vehicle_type": "van", "stops": stops}]}


def test_pdptw_load_counts_cumulative_demand_of_every_stop():
    # note「積載は累計需要の最大値」: 受取 50 + 配送 40 = 90 が積載量 80 を超える。
    instance = _tiny_pdptw()
    result = find_kind(instance).check(instance, _pdptw_route([[1, "pickup"], [1, "delivery"]]))
    _assert_violation(result, "exceeds capacity")
    assert result["cost"] == pytest.approx(100 + 20)


# ----------------------------------------------------------------------------
# pdptw: モデルが実際に返した停留所の形（保存解からの抜粋）
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stops",
    [
        # Qwen3.8: "P<id>" / "D<id>" の文字列
        ["P1", "D1"],
        ["pickup_1", "delivery-1"],
        # Claude: pair_id キーと座標・到着時刻などの付随キーを持つ dict
        [
            {"pair_id": 1, "type": "pickup", "x": 5.0, "y": 0.0, "arrival_time": 6.5, "load_after": 50},
            {"pair_id": 1, "type": "delivery", "x": 10.0, "y": 0.0, "arrival_time": 7.0, "load_after": 90},
        ],
        # Qwen3.6: 先頭と末尾にデポ id、停留所は "P<id>" / "D<id>"
        [1, "P1", "D1", 1],
        # 数値 id が文字列化されている
        [["1", "pickup"], ["1", "delivery"]],
    ],
)
def test_pdptw_reads_observed_stop_forms_as_the_reference_form(stops):
    instance = _tiny_pdptw()
    reference = find_kind(instance).check(instance, _pdptw_route([[1, "pickup"], [1, "delivery"]]))
    result = find_kind(instance).check(instance, _pdptw_route(stops))
    assert result["verified"] is True
    assert result["violations"] == reference["violations"]
    assert result["cost"] == pytest.approx(reference["cost"])


def test_pdptw_observed_depot_markers_do_not_hide_precedence_errors():
    # デポ印を外した後の順序で precedence を見る: D が先なら違反。
    instance = _tiny_pdptw()
    result = find_kind(instance).check(instance, _pdptw_route([1, "D1", "P1", 1]))
    assert result["verified"] is True
    _assert_violation(result, "pickup must precede delivery")


@pytest.mark.parametrize(
    "stops",
    [
        [1, "P1", 1, "D1", 1],  # 途中の裸の数値は受取か配送か決まらない
        [2, "P1", "D1", 2],  # 端でもデポ id と一致しない裸の数値は pair id かもしれない
        [{"pair_id": 1, "x": 5.0}, {"pair_id": 1, "x": 10.0}],  # kind が無い
        ["X1", "Y1"],
    ],
)
def test_pdptw_stops_without_a_unique_meaning_stay_unverified(stops):
    instance = _tiny_pdptw()
    result = find_kind(instance).check(instance, _pdptw_route(stops))
    assert result["verified"] is False
    assert result["violation_count"] == 0


def test_pdptw_unknown_pair_id_in_observed_form_is_a_violation_not_unverified():
    # 0 始まりなど id 規約のずれは推測で補正せず、そのまま違反として報告する。
    instance = _tiny_pdptw()
    result = find_kind(instance).check(instance, _pdptw_route(["P0", "D0"]))
    assert result["verified"] is True
    _assert_violation(result, "unknown stop")


@pytest.mark.parametrize(
    ("condition", "expected_cost"),
    [
        ("compact__qwen3_6_27b", 454415.58),
        ("compact__qwen3_8_27b", 311965.28),
        ("fable__claude", 233524.77),
    ],
)
def test_pdptw_saved_model_solutions_are_verified_and_reproduce_declared_cost(condition, expected_cost):
    # 保存解（あれば）が unverified にならず、申告費用を再計算で再現することを確認する。
    path = BASE_DIR / "outputs" / "rescore_hard" / "solutions" / condition / "prob_321.json"
    if not path.exists():
        pytest.skip(f"{path} is not available")
    record = _load("prob_321")
    result = _check(record, json.loads(path.read_text(encoding="utf-8"))["solution"])
    assert result["verified"] is True
    assert result["cost"] == pytest.approx(expected_cost, rel=1e-4)

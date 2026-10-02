"""routing 群（vrptw_md: prob_303/313、pdptw: prob_321）の instance 生成器の振る舞いを固定する。"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import pytest

from src.datagen.pipeline import shape_signature
from src.hardgen import GENERATORS
from src.utils.hard import find_kind

BASE_DIR = Path(__file__).resolve().parents[1]
HARD_DIR = BASE_DIR / "data" / "problems_hard"
CASES = [("vrptw_md", 303), ("vrptw_md", 313), ("pdptw", 321)]


def _base(problem_id: int) -> dict:
    record = json.loads((HARD_DIR / f"prob_{problem_id}.json").read_text(encoding="utf-8"))
    return record["instance"]


def _generate(kind: str, problem_id: int, seed: str = "test") -> dict:
    return GENERATORS[kind](random.Random(f"{seed}:{problem_id}"), _base(problem_id))


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _capacity(instance: dict) -> float:
    capacity = {v["type"]: v["capacity"] for v in instance["vehicle_types"]}
    return sum(n * capacity[t] for d in instance["depots"] for t, n in d["fleet"].items())


def _stops(instance: dict) -> list[dict]:
    if "customers" in instance:
        return instance["customers"]
    return [p["pickup"] for p in instance["pairs"]] + [p["delivery"] for p in instance["pairs"]]


def _demand_ratio(instance: dict) -> float:
    stops = instance.get("customers") or [p["pickup"] for p in instance["pairs"]]
    return sum(s["demand"] for s in stops) / _capacity(instance)


# ----------------------------------------------------------------------------
# 形・検出・決定性
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind,problem_id", CASES)
def test_generated_instance_keeps_base_shape(kind: str, problem_id: int) -> None:
    generated = _generate(kind, problem_id)
    assert shape_signature(generated) == shape_signature(_base(problem_id))


@pytest.mark.parametrize("kind,problem_id", CASES)
def test_generated_instance_is_detected_as_same_kind(kind: str, problem_id: int) -> None:
    assert find_kind(_generate(kind, problem_id)).name == kind


@pytest.mark.parametrize("kind,problem_id", CASES)
def test_same_seed_gives_same_instance_and_other_seed_differs(
    kind: str, problem_id: int
) -> None:
    first = _generate(kind, problem_id, seed="a")
    assert first == _generate(kind, problem_id, seed="a")
    assert first != _generate(kind, problem_id, seed="b")


# ----------------------------------------------------------------------------
# 固定した量と揺らした量
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind,problem_id", CASES)
def test_depots_and_vehicle_types_follow_base_with_fleet_jitter(
    kind: str, problem_id: int
) -> None:
    base = _base(problem_id)
    generated = _generate(kind, problem_id)
    assert generated["vehicle_types"] == base["vehicle_types"]
    assert generated["unserved_penalty"] == base["unserved_penalty"]
    for new, old in zip(generated["depots"], base["depots"]):
        assert {k: v for k, v in new.items() if k != "fleet"} == {
            k: v for k, v in old.items() if k != "fleet"
        }
        assert new["fleet"].keys() == old["fleet"].keys()
        assert all(1 <= new["fleet"][t] and abs(new["fleet"][t] - old["fleet"][t]) <= 1
                   for t in old["fleet"])
    # 台数はデポ間で移すだけなので、車種ごとの総台数（＝総容量）は元のまま。
    for vtype in base["depots"][0]["fleet"]:
        assert sum(d["fleet"][vtype] for d in generated["depots"]) == sum(
            d["fleet"][vtype] for d in base["depots"]
        )


@pytest.mark.parametrize("kind,problem_id", CASES)
def test_ids_and_coordinate_box_follow_base(kind: str, problem_id: int) -> None:
    base = _base(problem_id)
    generated = _generate(kind, problem_id)
    key = "customers" if kind == "vrptw_md" else "pairs"
    assert [c["id"] for c in generated[key]] == [c["id"] for c in base[key]]
    base_stops, new_stops = _stops(base), _stops(generated)
    for axis in ("x", "y"):
        lo, hi = min(s[axis] for s in base_stops), max(s[axis] for s in base_stops)
        assert all(lo <= s[axis] <= hi for s in new_stops)
    assert all(float(s["demand"]).is_integer() for s in new_stops)


# ----------------------------------------------------------------------------
# 統計量が base の近傍にあること
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind,problem_id", CASES)
def test_demand_to_capacity_ratio_stays_near_base(kind: str, problem_id: int) -> None:
    base_ratio = _demand_ratio(_base(problem_id))
    for seed in range(5):
        ratio = _demand_ratio(_generate(kind, problem_id, seed=str(seed)))
        assert abs(ratio - base_ratio) <= 0.15 * base_ratio


@pytest.mark.parametrize("kind,problem_id", CASES)
def test_window_widths_and_demands_follow_base_values(kind: str, problem_id: int) -> None:
    base_stops, new_stops = _stops(_base(problem_id)), _stops(_generate(kind, problem_id))
    base_widths = {round(s["tw_end"] - s["tw_start"], 2) for s in base_stops}
    assert all(round(s["tw_end"] - s["tw_start"], 2) in base_widths for s in new_stops)
    assert {s["demand"] for s in new_stops} <= {s["demand"] for s in base_stops}
    base_mean = statistics.mean(s["tw_end"] - s["tw_start"] for s in base_stops)
    new_mean = statistics.mean(s["tw_end"] - s["tw_start"] for s in new_stops)
    assert abs(new_mean - base_mean) <= 0.15 * base_mean


@pytest.mark.parametrize("problem_id", [303, 313])
def test_vrptw_customers_are_reachable_by_the_slowest_vehicle(problem_id: int) -> None:
    generated = _generate("vrptw_md", problem_id)
    speed = min(v["speed_kmh"] for v in generated["vehicle_types"])
    base_customers = _base(problem_id)["customers"]
    start_lo = min(c["tw_start"] for c in base_customers)
    start_hi = max(c["tw_start"] for c in base_customers)
    for customer in generated["customers"]:
        depot = min(generated["depots"], key=lambda d: _dist(d, customer))
        travel = _dist(depot, customer) / speed
        assert start_lo <= customer["tw_start"] <= start_hi
        assert depot["open_time"] + travel < customer["tw_end"]
        assert customer["tw_start"] + customer["service_time"] + travel < depot["close_time"]
        assert customer["service_time"] in {c["service_time"] for c in base_customers}


def test_vrptw_nearest_depot_distance_stays_near_base() -> None:
    for problem_id in (303, 313):
        base = _base(problem_id)
        generated = _generate("vrptw_md", problem_id)

        def mean_nearest(instance: dict) -> float:
            return statistics.mean(
                min(_dist(d, c) for d in instance["depots"]) for c in instance["customers"]
            )

        assert abs(mean_nearest(generated) - mean_nearest(base)) <= 0.2 * mean_nearest(base)


def test_pdptw_delivery_window_is_reachable_from_pickup_window() -> None:
    generated = _generate("pdptw", 321)
    speed = min(v["speed_kmh"] for v in generated["vehicle_types"])
    base_pairs = _base(321)["pairs"]
    start_lo = min(p["pickup"]["tw_start"] for p in base_pairs)
    start_hi = max(p["pickup"]["tw_start"] for p in base_pairs)
    for pair in generated["pairs"]:
        pickup, delivery = pair["pickup"], pair["delivery"]
        assert start_lo <= pickup["tw_start"] <= start_hi
        travel = _dist(pickup, delivery) / speed
        # 受取枠の始まりに積んで直行しても配送枠より早く、受取枠の終わりに積んでも間に合う。
        assert pickup["tw_start"] + travel <= delivery["tw_start"] + 1e-6
        assert pickup["tw_end"] + travel <= delivery["tw_end"] + 1e-6
        depot = min(generated["depots"], key=lambda d: _dist(d, pickup) + _dist(delivery, d))
        assert depot["open_time"] + _dist(depot, pickup) / speed < pickup["tw_end"]
        assert delivery["tw_start"] + _dist(delivery, depot) / speed < depot["close_time"]


def test_pdptw_pair_distance_stays_near_base() -> None:
    base = _base(321)
    generated = _generate("pdptw", 321)

    def mean_pair_distance(instance: dict) -> float:
        return statistics.mean(_dist(p["pickup"], p["delivery"]) for p in instance["pairs"])

    assert abs(mean_pair_distance(generated) - mean_pair_distance(base)) <= 0.2 * mean_pair_distance(
        base
    )

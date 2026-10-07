"""production 群（clsp / prp / prp_tw）の instance 生成器が、元 instance と同じ形・同じ分布で
決定的に新しい instance を作ることを固定する。教師コードの実行は重いので含めない。"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import pytest

from src.datagen.pipeline import shape_signature
from src.hardgen import GENERATORS
from src.hardgen.production import clsp_load_ratio, coordinate_bound, direct_trip_fits
from src.utils.hard import find_kind

PROBLEM_DIR = Path(__file__).resolve().parents[1] / "data" / "problems_hard"
CASES = [("clsp", 301), ("clsp", 311), ("prp", 320), ("prp_tw", 327)]


def _base(problem_id: int) -> dict:
    return json.loads((PROBLEM_DIR / f"prob_{problem_id}.json").read_text(encoding="utf-8"))[
        "instance"
    ]


def _generate(kind: str, problem_id: int, seed: str = "seed-a") -> dict:
    return GENERATORS[kind](random.Random(seed), _base(problem_id))


@pytest.mark.parametrize(("kind", "problem_id"), CASES)
def test_generated_instance_keeps_base_shape(kind: str, problem_id: int) -> None:
    base = _base(problem_id)
    generated = _generate(kind, problem_id)
    assert shape_signature(generated) == shape_signature(base)


@pytest.mark.parametrize(("kind", "problem_id"), CASES)
def test_generated_instance_is_detected_as_same_kind(kind: str, problem_id: int) -> None:
    detected = find_kind(_generate(kind, problem_id))
    assert detected is not None and detected.name == kind


@pytest.mark.parametrize(("kind", "problem_id"), CASES)
def test_generation_is_deterministic_per_seed(kind: str, problem_id: int) -> None:
    first = _generate(kind, problem_id, "seed-a")
    again = _generate(kind, problem_id, "seed-a")
    other = _generate(kind, problem_id, "seed-b")
    assert first == again
    assert first != other
    assert first != _base(problem_id)


@pytest.mark.parametrize("problem_id", [301, 311])
def test_clsp_matches_base_load_and_value_ranges(problem_id: int) -> None:
    base = _base(problem_id)
    generated = _generate("clsp", problem_id)
    # きつさ: 加工時間/能力の比は能力の補正で base と一致する（整数丸めの分だけずれる）。
    assert clsp_load_ratio(generated) == pytest.approx(clsp_load_ratio(base), rel=1e-3)
    assert generated["num_periods"] == base["num_periods"]
    machine_ids = {m["id"] for m in generated["machines"]}
    assert machine_ids == {m["id"] for m in base["machines"]}
    assert [it["id"] for it in generated["items"]] == [it["id"] for it in base["items"]]
    assert generated["items"][0]["name"] == base["items"][0]["name"]
    assert generated["machines"][-1]["name"] == base["machines"][-1]["name"]
    base_sizes = {len(it["compatible_machines"]) for it in base["items"]}
    pool = {d for it in base["items"] for d in it["demand"]}
    for item in generated["items"]:
        assert len(item["compatible_machines"]) in base_sizes
        assert set(item["compatible_machines"]) <= machine_ids
        assert item["compatible_machines"] == sorted(item["compatible_machines"])
        assert len(item["demand"]) == base["num_periods"]
        assert set(item["demand"]) <= pool
        for key in ("unit_process_time", "holding_cost", "setup_cost", "setup_time", "lost_sale_cost"):
            values = [it[key] for it in base["items"]]
            assert min(values) <= item[key] <= max(values)
            assert isinstance(item[key], type(values[0]))
    capacities = [m["capacity_per_period"] for m in generated["machines"]]
    base_capacities = [m["capacity_per_period"] for m in base["machines"]]
    assert all(isinstance(c, int) for c in capacities)
    assert 0.8 * min(base_capacities) <= min(capacities)
    assert max(capacities) <= 1.2 * max(base_capacities)
    assert statistics.mean(it["setup_cost"] for it in generated["items"]) == pytest.approx(
        statistics.mean(it["setup_cost"] for it in base["items"]), rel=0.15
    )


@pytest.mark.parametrize(("kind", "problem_id"), [("prp", 320), ("prp_tw", 327)])
def test_prp_keeps_scalars_and_matches_base_ranges(kind: str, problem_id: int) -> None:
    base = _base(problem_id)
    generated = _generate(kind, problem_id)
    for key, value in base.items():
        if key not in ("plants", "customers"):
            assert generated[key] == value
    bound = coordinate_bound(base)
    assert coordinate_bound(generated) <= bound
    assert [p["id"] for p in generated["plants"]] == [p["id"] for p in base["plants"]]
    assert [c["id"] for c in generated["customers"]] == [c["id"] for c in base["customers"]]
    assert generated["plants"][0]["name"] == base["plants"][0]["name"]
    assert generated["customers"][-1]["name"] == base["customers"][-1]["name"]
    for entries, keys in (
        (
            "plants",
            (
                "capacity_per_period",
                "setup_cost",
                "unit_production_cost",
                "inventory_cost",
                "max_inventory",
                "initial_inventory",
            ),
        ),
        ("customers", ("inventory_cost", "max_inventory", "initial_inventory")),
    ):
        for entry in generated[entries]:
            assert 0 <= entry["x"] <= bound and 0 <= entry["y"] <= bound
            for key in keys:
                values = [e[key] for e in base[entries]]
                assert min(values) <= entry[key] <= max(values)
                assert isinstance(entry[key], type(values[0]))
    pool = {d for c in base["customers"] for d in c["demand"]}
    for customer in generated["customers"]:
        assert len(customer["demand"]) == base["periods"]
        assert set(customer["demand"]) <= pool
        # 期ごとの JIT 配送が常に可能: 初期在庫は上限以下、需要は上限以下。
        assert customer["initial_inventory"] <= customer["max_inventory"]
        assert max(customer["demand"]) <= customer["max_inventory"]
    # きつさ: 総需要/総能力と 1 期あたりの必要台数が base の近傍にある。
    def tightness(instance: dict) -> tuple[float, float]:
        demand = sum(sum(c["demand"]) for c in instance["customers"])
        capacity = sum(p["capacity_per_period"] for p in instance["plants"]) * instance["periods"]
        return demand / capacity, demand / instance["periods"] / instance["vehicle_capacity"]

    base_ratio, base_vehicles = tightness(base)
    ratio, vehicles = tightness(generated)
    assert ratio == pytest.approx(base_ratio, rel=0.35)
    assert vehicles == pytest.approx(base_vehicles, rel=0.1)
    assert max(p["initial_inventory"] for p in generated["plants"]) <= max(
        p["max_inventory"] for p in generated["plants"]
    )


def test_prp_tw_windows_keep_width_and_every_customer_reachable() -> None:
    base = _base(327)
    generated = _generate("prp_tw", 327)
    base_windows = [c["delivery_window"] for c in base["customers"]]
    width = round(base_windows[0]["end"] - base_windows[0]["start"], 2)
    starts = [w["start"] for w in base_windows]
    for customer in generated["customers"]:
        window = customer["delivery_window"]
        assert min(starts) <= window["start"] <= max(starts)
        assert round(window["end"] - window["start"], 2) == width
        assert direct_trip_fits(generated, customer)
    assert statistics.mean(c["delivery_window"]["start"] for c in generated["customers"]) == (
        pytest.approx(statistics.mean(starts), abs=0.6)
    )


def test_prp_without_windows_has_no_window_key() -> None:
    generated = _generate("prp", 320)
    assert all("delivery_window" not in c for c in generated["customers"])
    assert find_kind(generated).name == "prp"

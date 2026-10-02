"""facility 群の生成器が、元 instance と同じ形・同じ種別・同じきつさの instance を決定的に作ることを確かめる。"""

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

KIND_BY_ID = {
    "prob_304": "facility_multi",
    "prob_314": "facility_multi",
    "prob_328": "facility_2ech",
    "prob_329": "facility_robust",
}


def _base(pid: str) -> dict:
    return json.loads((HARD_DIR / f"{pid}.json").read_text(encoding="utf-8"))["instance"]


def _generate(pid: str, seed: str = "test") -> tuple[dict, dict]:
    base = _base(pid)
    return base, GENERATORS[KIND_BY_ID[pid]](random.Random(f"{seed}:{pid}"), base)


def _sites(instance: dict) -> tuple[list[dict], str]:
    """顧客が割り当てられる側の拠点群と、顧客側の候補キー。"""
    if "dcs" in instance:
        return instance["dcs"], "candidate_dcs"
    return instance["facilities"], "candidate_facilities"


def _period_totals(instance: dict) -> list[float]:
    periods = instance["num_periods"]
    return [sum(c["demand"][t] for c in instance["customers"]) for t in range(periods)]


def _capacity_ratio(instance: dict) -> float:
    """能力合計 / ピーク期需要。きつさの主指標。"""
    sites, _ = _sites(instance)
    return sum(s["throughput_per_period"] for s in sites) / max(_period_totals(instance))


def _storage_ratio(instance: dict) -> float:
    sites, _ = _sites(instance)
    total_throughput = sum(s["throughput_per_period"] for s in sites)
    return sum(s["storage_capacity"] for s in sites) / total_throughput


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_generated_instance_keeps_shape_and_kind(pid: str) -> None:
    base, new = _generate(pid)
    assert shape_signature(new) == shape_signature(base)
    assert find_kind(new).name == KIND_BY_ID[pid]
    assert new is not base
    # 文字列・スカラーは base の値を保つ。
    for key, value in base.items():
        if not isinstance(value, list):
            assert new[key] == value


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_generation_is_deterministic(pid: str) -> None:
    _, first = _generate(pid, "seed-a")
    _, second = _generate(pid, "seed-a")
    _, other = _generate(pid, "seed-b")
    assert first == second
    assert first != other


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_ids_names_and_candidates_follow_base_conventions(pid: str) -> None:
    base, new = _generate(pid)
    sites, cand_key = _sites(new)
    base_sites, _ = _sites(base)
    prefix = base_sites[0]["name"].rstrip("0123456789")
    assert [s["id"] for s in sites] == list(range(1, len(base_sites) + 1))
    assert [s["name"] for s in sites] == [f"{prefix}{i}" for i in range(1, len(base_sites) + 1)]
    assert [c["id"] for c in new["customers"]] == list(range(1, len(base["customers"]) + 1))
    n_cand = len(base["customers"][0][cand_key])
    by_id = {s["id"]: s for s in sites}
    for customer in new["customers"]:
        cand = customer[cand_key]
        assert len(cand) == n_cand and len(set(cand)) == n_cand
        dists = [math.hypot(by_id[f]["x"] - customer["x"], by_id[f]["y"] - customer["y"]) for f in cand]
        assert dists == sorted(dists)
        farthest = dists[-1]
        others = [
            math.hypot(s["x"] - customer["x"], s["y"] - customer["y"])
            for s in sites
            if s["id"] not in cand
        ]
        assert all(d >= farthest - 1e-9 for d in others)


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_demand_and_capacity_ratios_stay_near_base(pid: str) -> None:
    base, new = _generate(pid)
    assert sum(_period_totals(new)) == pytest.approx(sum(_period_totals(base)), rel=0.08)
    assert _capacity_ratio(new) == pytest.approx(_capacity_ratio(base), rel=0.2)
    assert _storage_ratio(new) == pytest.approx(_storage_ratio(base), rel=0.2)
    base_values = [d for c in base["customers"] for d in c["demand"]]
    new_values = [d for c in new["customers"] for d in c["demand"]]
    assert min(new_values) >= min(base_values)
    assert max(new_values) <= max(base_values) * 1.15
    assert statistics.fmean(new_values) == pytest.approx(statistics.fmean(base_values), rel=0.08)
    # 期ごとの総需要の山谷（季節性）は base と同程度。
    base_totals, new_totals = _period_totals(base), _period_totals(new)
    assert max(new_totals) / min(new_totals) == pytest.approx(
        max(base_totals) / min(base_totals), abs=0.1
    )


@pytest.mark.parametrize("pid", sorted(KIND_BY_ID))
def test_coordinates_and_attributes_stay_in_base_ranges(pid: str) -> None:
    base, new = _generate(pid)
    extent = math.ceil(max(max(c["x"], c["y"]) for c in base["customers"]) / 10) * 10
    for c in new["customers"]:
        assert 0 <= c["x"] <= extent and 0 <= c["y"] <= extent
    site_keys = [k for k in ("facilities", "plants", "dcs") if k in base]
    for key in site_keys:
        for attr in base[key][0]:
            if attr in ("id", "name"):
                continue
            base_values = [s[attr] for s in base[key]]
            new_values = [s[attr] for s in new[key]]
            assert all(isinstance(v, type(base_values[0])) for v in new_values)
            margin = 0.25 * (max(base_values) - min(base_values))
            assert min(new_values) >= min(base_values) - margin
            assert max(new_values) <= max(base_values) + margin
            if attr in ("x", "y"):
                continue
            assert statistics.fmean(new_values) == pytest.approx(
                statistics.fmean(base_values), rel=0.25
            )


def test_robust_scenarios_cover_every_customer_with_base_factor_range() -> None:
    base, new = _generate("prob_329")
    base_factors = [v for s in base["scenarios"] for v in s["factors"].values()]
    assert [s["id"] for s in new["scenarios"]] == [s["id"] for s in base["scenarios"]]
    for scenario in new["scenarios"]:
        assert list(scenario["factors"]) == [str(c["id"]) for c in new["customers"]]
        values = list(scenario["factors"].values())
        assert min(base_factors) <= min(values) and max(values) <= max(base_factors)
        assert statistics.fmean(values) == pytest.approx(statistics.fmean(base_factors), abs=0.03)


def test_2ech_plants_keep_initial_inventory_and_cover_peak_demand() -> None:
    base, new = _generate("prob_328")
    assert all(p["initial_inventory"] == 0 for p in new["plants"])
    assert all(d["initial_inventory"] == 0 for d in new["dcs"])
    plant_ratio = sum(p["capacity_per_period"] for p in new["plants"]) / max(_period_totals(new))
    base_ratio = sum(p["capacity_per_period"] for p in base["plants"]) / max(_period_totals(base))
    assert plant_ratio == pytest.approx(base_ratio, rel=0.3)

"""network_finance 群（網設計・ポートフォリオ）の大規模 instance 生成器の振る舞いを検証する。"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import pytest

from src.datagen.pipeline import shape_signature
from src.hardgen import GENERATORS
from src.hardgen.network_finance import _disjoint_pair_lengths
from src.utils.hard import find_kind

BASE_DIR = Path(__file__).resolve().parents[1]
HARD_DIR = BASE_DIR / "data" / "problems_hard"

KIND_OF = {
    "prob_308": "mcnd",
    "prob_318": "mcnd",
    "prob_325": "mcnd_surv",
    "prob_319": "portfolio",
    "prob_326": "portfolio_cvar",
}
NETWORK_PIDS = ["prob_308", "prob_318", "prob_325"]
PORTFOLIO_PIDS = ["prob_319", "prob_326"]


@cache
def _base(pid: str) -> dict:
    return json.loads((HARD_DIR / f"{pid}.json").read_text(encoding="utf-8"))["instance"]


@cache
def _generated(pid: str, seed: str = "seed") -> dict:
    return GENERATORS[KIND_OF[pid]](random.Random(f"{pid}:{seed}"), _base(pid))


def _demand_ratio(instance: dict) -> float:
    """需要合計 / 候補アーク容量合計 — 網設計のきつさ。"""
    return sum(o["volume"] for o in instance["od_demands"]) / sum(
        a["capacity"] for a in instance["arcs"]
    )


def _reachable(instance: dict, source: int) -> set[int]:
    adj = defaultdict(list)
    for arc in instance["arcs"]:
        adj[arc["from"]].append(arc["to"])
    seen = {source}
    stack = [source]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _column_sd_mean(instance: dict) -> float:
    """銘柄ごとのシナリオ収益の標準偏差の平均 — リスク尺度の規模。"""
    columns = zip(*instance["scenario_returns"])
    return statistics.fmean(statistics.pstdev(col) for col in columns)


# ---------------------------------------------------------------------------
# 共通契約
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pid", list(KIND_OF))
def test_generated_instance_keeps_shape_and_kind(pid: str) -> None:
    base, new = _base(pid), _generated(pid)
    assert shape_signature(new) == shape_signature(base)
    assert find_kind(new).name == KIND_OF[pid]
    assert new != base


@pytest.mark.parametrize("pid", list(KIND_OF))
def test_generation_is_deterministic_for_same_seed(pid: str) -> None:
    kind = KIND_OF[pid]
    first = GENERATORS[kind](random.Random("same"), _base(pid))
    second = GENERATORS[kind](random.Random("same"), _base(pid))
    assert first == second
    assert GENERATORS[kind](random.Random("other"), _base(pid)) != first


# ---------------------------------------------------------------------------
# 網設計
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pid", NETWORK_PIDS)
def test_network_nodes_and_arcs_follow_base_layout(pid: str) -> None:
    base, new = _base(pid), _generated(pid)
    bound = math.ceil(max(max(v["x"], v["y"]) for v in base["nodes"]) / 100) * 100
    assert [v["id"] for v in new["nodes"]] == list(range(1, len(base["nodes"]) + 1))
    assert all(0 <= v["x"] <= bound and 0 <= v["y"] <= bound for v in new["nodes"])
    assert new["nodes"][0]["name"] == base["nodes"][0]["name"]
    assert [a["id"] for a in new["arcs"]] == list(range(1, len(base["arcs"]) + 1))
    assert all(a["from"] != a["to"] for a in new["arcs"])
    assert len({(a["from"], a["to"]) for a in new["arcs"]}) == len(new["arcs"])
    positions = {v["id"]: (v["x"], v["y"]) for v in new["nodes"]}
    for arc in new["arcs"]:
        assert arc["distance"] == pytest.approx(
            math.dist(positions[arc["from"]], positions[arc["to"]]), abs=0.051
        )
    assert {a["capacity"] for a in new["arcs"]} <= {a["capacity"] for a in base["arcs"]}


@pytest.mark.parametrize("pid", NETWORK_PIDS)
def test_network_costs_stay_in_base_range(pid: str) -> None:
    base, new = _base(pid), _generated(pid)
    for key, slack in (("fixed_cost", 1.4), ("flow_cost", 1.4)):
        lo = min(a[key] for a in base["arcs"])
        hi = max(a[key] for a in base["arcs"])
        assert min(a[key] for a in new["arcs"]) >= lo * 0.9
        assert max(a[key] for a in new["arcs"]) <= hi * slack
        base_mean = statistics.fmean(a[key] for a in base["arcs"])
        assert statistics.fmean(a[key] for a in new["arcs"]) == pytest.approx(base_mean, rel=0.25)


@pytest.mark.parametrize("pid", NETWORK_PIDS)
def test_network_is_strongly_connected_with_demand_ratio_near_base(pid: str) -> None:
    base, new = _base(pid), _generated(pid)
    n = len(new["nodes"])
    assert all(len(_reachable(new, v)) == n for v in range(1, n + 1))
    assert _demand_ratio(new) == pytest.approx(_demand_ratio(base), rel=0.3)
    volumes = [o["volume"] for o in new["od_demands"]]
    assert set(volumes) <= {o["volume"] for o in base["od_demands"]}
    assert all(o["origin"] != o["destination"] for o in new["od_demands"])
    distinct = len({(o["origin"], o["destination"]) for o in base["od_demands"]})
    if distinct == len(base["od_demands"]):
        assert len({(o["origin"], o["destination"]) for o in new["od_demands"]}) == len(volumes)


def test_mcnd_surv_has_two_disjoint_paths_within_max_delay() -> None:
    base, new = _base("prob_325"), _generated("prob_325")
    assert new["disjoint_paths"] == base["disjoint_paths"]
    assert new["note"] == base["note"]
    ratios = []
    for od in new["od_demands"]:
        lengths = _disjoint_pair_lengths(new["arcs"], od["origin"], od["destination"])
        assert lengths is not None and lengths[1] <= od["max_delay"]
        ratios.append(od["max_delay"] / lengths[1])
    assert 1.2 <= min(ratios) and max(ratios) <= 1.4
    assert isinstance(new["od_demands"][0]["max_delay"], int)


def test_knn_base_keeps_bidirectional_arcs_and_ring_base_keeps_ring() -> None:
    knn = {(a["from"], a["to"]) for a in _generated("prob_308")["arcs"]}
    assert all((b, a) in knn for a, b in knn)
    ring = {(a["from"], a["to"]) for a in _generated("prob_325")["arcs"]}
    n = len(_base("prob_325")["nodes"])
    assert all((i, i % n + 1) in ring and (i % n + 1, i) in ring for i in range(1, n + 1))


# ---------------------------------------------------------------------------
# ポートフォリオ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pid", PORTFOLIO_PIDS)
def test_portfolio_assets_follow_base_attribute_ranges(pid: str) -> None:
    base, new = _base(pid), _generated(pid)
    assert new["sectors"] == base["sectors"]
    assert new["limits"] == base["limits"]
    assert new["target_return"] == base["target_return"]
    assert [a["id"] for a in new["assets"]] == list(range(1, len(base["assets"]) + 1))
    assert new["assets"][0]["name"] == base["assets"][0]["name"]
    for key in ("expected_return", "beta"):
        lo = min(a[key] for a in base["assets"])
        hi = max(a[key] for a in base["assets"])
        assert all(lo <= a[key] <= hi for a in new["assets"])
    assert all(a["mean_return"] == a["expected_return"] for a in new["assets"])
    assert set(Counter(a["sector"] for a in new["assets"])) == set(base["sectors"])
    base_fraction = statistics.fmean(
        a["expected_return"] >= base["target_return"] for a in base["assets"]
    )
    new_fraction = statistics.fmean(
        a["expected_return"] >= new["target_return"] for a in new["assets"]
    )
    assert new_fraction == pytest.approx(base_fraction, abs=0.08)


@pytest.mark.parametrize("pid", PORTFOLIO_PIDS)
def test_portfolio_scenarios_match_base_dispersion(pid: str) -> None:
    base, new = _base(pid), _generated(pid)
    assert len(new["scenario_returns"]) == len(base["scenario_returns"])
    assert all(len(row) == len(new["assets"]) for row in new["scenario_returns"])
    assert _column_sd_mean(new) == pytest.approx(_column_sd_mean(base), rel=0.15)
    grand_mean = statistics.fmean(v for row in new["scenario_returns"] for v in row)
    assert abs(grand_mean) < 0.02


def test_portfolio_current_holdings_sum_to_one_like_base() -> None:
    base, new = _base("prob_319"), _generated("prob_319")
    current = new["current_portfolio"]
    assert len(current) == len(base["current_portfolio"])
    assert sum(current.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(1 <= int(k) <= len(new["assets"]) for k in current)
    assert all(0 < w <= base["limits"]["max_weight"] for w in current.values())


def test_portfolio_cvar_current_holdings_leave_room_and_fit_cardinality() -> None:
    base, new = _base("prob_326"), _generated("prob_326")
    assert new["risk"] == base["risk"]
    assert new["min_holding"] == base["min_holding"]
    current = {int(k): w for k, w in new["current_portfolio"].items()}
    assert len(current) == len(base["current_portfolio"])
    # 売却不可なので、現保有の合計が 1 だと解が現保有に固定される。新規組入れの余地を残す。
    assert 0.55 <= sum(current.values()) <= 0.85
    sector_of = {a["id"]: a["sector"] for a in new["assets"]}
    sector_weight: dict[int, float] = defaultdict(float)
    for asset_id, w in current.items():
        sector_weight[sector_of[asset_id]] += w
    assert max(sector_weight.values()) <= base["limits"]["max_sector_weight"]
    assert len(sector_weight) >= base["limits"]["min_sectors"]
    held = Counter(sector_of[a] for a in current)
    cardinality = new["sector_cardinality"]
    assert len(cardinality) == len(base["sector_cardinality"])
    for sector, (lo, hi) in cardinality.items():
        assert 1 <= lo <= held[int(sector)] <= hi
    best = max(a["expected_return"] for a in new["assets"])
    expected = sum(w * new["assets"][a - 1]["expected_return"] for a, w in current.items())
    assert expected + (1 - sum(current.values())) * best >= new["target_return"]

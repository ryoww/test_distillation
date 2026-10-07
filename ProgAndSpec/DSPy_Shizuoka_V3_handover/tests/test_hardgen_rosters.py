"""rosters 群（乗務員ペアリング・勤務表）の大規模 instance 生成器の振る舞いを検証する。"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from src.datagen.pipeline import shape_signature
from src.hardgen import GENERATORS
from src.utils.hard import find_kind

BASE_DIR = Path(__file__).resolve().parents[1]
HARD_DIR = BASE_DIR / "data" / "problems_hard"

KIND_OF = {
    "prob_302": "crew_pairing",
    "prob_312": "crew_pairing",
    "prob_330": "crew_pairing_seniority",
    "prob_306": "nurse_roster",
    "prob_316": "nurse_roster",
    "prob_323": "role_roster",
}
CREW_PIDS = ["prob_302", "prob_312", "prob_330"]
ROSTER_PIDS = {"prob_306": "nurses", "prob_316": "nurses", "prob_323": "staff"}


def _base(pid: str) -> dict:
    return json.loads((HARD_DIR / f"{pid}.json").read_text(encoding="utf-8"))["instance"]


def _generate(pid: str, seed: str = "seed") -> tuple[dict, dict]:
    base = _base(pid)
    return base, GENERATORS[KIND_OF[pid]](random.Random(f"{pid}:{seed}"), base)


def _connection_arcs(instance: dict) -> int:
    """同一空港・接続時間 min〜max の便ペア数（接続可能性の密度）。"""
    by_origin: dict[str, list[dict]] = {}
    for f in instance["flights"]:
        by_origin.setdefault(f["origin"], []).append(f)
    lo, hi = instance["min_connect_hours"], instance["max_connect_hours"]
    return sum(
        1
        for a in instance["flights"]
        for b in by_origin.get(a["destination"], [])
        if lo <= b["dep_time"] - a["arr_time"] <= hi
    )


def _requirement_ratio(instance: dict, people_key: str) -> float:
    """必要人数合計 / (人数 × 勤務可能日数) — 勤務表のきつさ。"""
    reqs = instance["daily_requirements"]
    total = sum(v for r in reqs for k, v in r.items() if k != "day")
    workable = len(reqs) - instance["rules"]["min_off_days"]
    return total / (len(instance[people_key]) * workable)


@pytest.mark.parametrize("pid", sorted(KIND_OF))
def test_generated_instance_keeps_shape_and_kind(pid):
    base, new = _generate(pid)
    assert shape_signature(new) == shape_signature(base)
    assert find_kind(new).name == KIND_OF[pid]


@pytest.mark.parametrize("pid", sorted(KIND_OF))
def test_generation_is_deterministic_and_seed_sensitive(pid):
    _, first = _generate(pid, "a")
    _, again = _generate(pid, "a")
    _, other = _generate(pid, "b")
    assert first == again
    assert first != other


@pytest.mark.parametrize("pid", CREW_PIDS)
def test_flights_keep_time_window_and_cost_rules(pid):
    base, new = _generate(pid)
    deps = [f["dep_time"] for f in base["flights"]]
    durs = [f["duration"] for f in base["flights"]]
    costs = [f["flight_cost"] for f in base["flights"]]
    airports = set(base["airports"])
    ids = [f["id"] for f in new["flights"]]
    assert ids == list(range(1, len(base["flights"]) + 1))
    for f in new["flights"]:
        assert f["origin"] in airports and f["destination"] in airports
        assert f["origin"] != f["destination"]
        assert min(deps) <= f["dep_time"] <= max(deps)
        assert min(durs) <= f["duration"] <= max(durs)
        assert min(costs) <= f["flight_cost"] <= max(costs)
        assert abs(f["arr_time"] - f["dep_time"] - f["duration"]) < 1e-6
        assert round(f["dep_time"], 2) == f["dep_time"]
    # 文字列定数と費用係数は base のまま
    for key in ("bases", "airports", "fixed_cost_per_pairing", "wait_cost_per_hour"):
        assert new[key] == base[key]


def test_prob_302_flights_stay_proportional_to_duration_and_sorted():
    _, new = _generate("prob_302")
    assert all(f["flight_cost"] == round(f["duration"] * 1500) for f in new["flights"])
    deps = [f["dep_time"] for f in new["flights"]]
    assert deps == sorted(deps)


@pytest.mark.parametrize("pid", CREW_PIDS)
def test_connection_density_matches_base(pid):
    base, new = _generate(pid)
    base_arcs = _connection_arcs(base)
    assert abs(_connection_arcs(new) - base_arcs) <= 0.1 * base_arcs
    base_from_base = sum(f["origin"] in base["bases"] for f in base["flights"])
    new_from_base = sum(f["origin"] in new["bases"] for f in new["flights"])
    assert abs(new_from_base - base_from_base) <= 0.15 * base_from_base


def test_crews_follow_base_attribute_ranges():
    base, new = _generate("prob_330")
    base_crews, crews = base["crews"], new["crews"]
    assert [c["id"] for c in crews] == list(range(1, len(base_crews) + 1))
    assert [c["name"] for c in crews][:2] == ["クルー1", "クルー2"]
    assert {c["max_duties"] for c in crews} <= {c["max_duties"] for c in base_crews}
    assert {c["home_base"] for c in crews} <= set(base["bases"])
    span = [c["span_pref"] for c in base_crews]
    assert all(min(span) <= c["span_pref"] <= max(span) for c in crews)
    base_duties = sum(c["max_duties"] for c in base_crews)
    assert abs(sum(c["max_duties"] for c in crews) - base_duties) <= 0.1 * base_duties
    assert new["unassigned_pairing_cost"] == base["unassigned_pairing_cost"]
    assert new["note"] == base["note"]


@pytest.mark.parametrize("pid", sorted(ROSTER_PIDS))
def test_roster_people_follow_base_conventions(pid):
    key = ROSTER_PIDS[pid]
    base, new = _generate(pid)
    days = len(base["daily_requirements"])
    counts = [len(p["preferred_off_days"]) for p in base[key]]
    prefix = base[key][0]["name"].rstrip("0123456789")
    assert [p["id"] for p in new[key]] == list(range(1, len(base[key]) + 1))
    for p in new[key]:
        assert p["name"] == f"{prefix}{p['id']}"
        offs = p["preferred_off_days"]
        assert offs == sorted(set(offs))
        assert min(counts) <= len(offs) <= max(counts)
        assert all(1 <= d <= days for d in offs)
    qualified = sum(p["qualified"] for p in new[key])
    base_qualified = sum(p["qualified"] for p in base[key])
    assert abs(qualified - base_qualified) <= 0.3 * base_qualified
    for fixed in ("rules", "penalties", "shifts"):
        assert new[fixed] == base[fixed]


@pytest.mark.parametrize("pid", sorted(ROSTER_PIDS))
def test_daily_requirements_stay_in_base_range_and_tightness(pid):
    key = ROSTER_PIDS[pid]
    base, new = _generate(pid)
    assert [r["day"] for r in new["daily_requirements"]] == list(
        range(1, len(base["daily_requirements"]) + 1)
    )
    for col in base["daily_requirements"][0]:
        if col == "day":
            continue
        values = [r[col] for r in base["daily_requirements"]]
        assert all(min(values) <= r[col] <= max(values) for r in new["daily_requirements"])
    base_ratio = _requirement_ratio(base, key)
    assert abs(_requirement_ratio(new, key) - base_ratio) <= 0.03 * base_ratio


def test_role_roster_keeps_role_sequence_and_regions():
    base, new = _generate("prob_323")
    assert [p["role"] for p in new["staff"]] == [p["role"] for p in base["staff"]]
    regions = {p["region"] for p in base["staff"]}
    assert {p["region"] for p in new["staff"]} == regions
    region_counts = Counter(p["region"] for p in new["staff"])
    assert min(region_counts.values()) >= 0.3 * len(new["staff"])
    seniority = [p["seniority"] for p in base["staff"]]
    assert all(min(seniority) <= p["seniority"] <= max(seniority) for p in new["staff"])
    assert new["roles"] == base["roles"]
    assert new["note"] == base["note"]

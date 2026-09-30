"""rosters 群（乗務員ペアリング・勤務表）の大規模問題検証器の振る舞いを検証する。"""

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

EXPECTED_KINDS = {
    "prob_302": "crew_pairing",
    "prob_312": "crew_pairing",
    "prob_330": "crew_pairing_seniority",
    "prob_306": "nurse_roster",
    "prob_316": "nurse_roster",
    "prob_323": "role_roster",
}


def _load(pid: str) -> dict:
    record = json.loads((HARD_DIR / f"{pid}.json").read_text(encoding="utf-8"))
    record["core_type"] = f"{record['domain']}_{record['math_type']}"
    return record


def _reference(record: dict) -> dict:
    return {
        k: v for k, v in record["reference_solution"].items() if k not in ("objective_value", "note")
    }


def _check(record: dict, solution: object) -> dict:
    return check_feasibility_detailed(record["core_type"], record["instance"], solution)


@pytest.mark.parametrize("pid", sorted(EXPECTED_KINDS))
def test_reference_solution_is_feasible_and_objective_is_reproduced(pid):
    record = _load(pid)
    kind = find_kind(record["instance"])
    assert kind is not None and kind.name == EXPECTED_KINDS[pid]
    result = _check(record, _reference(record))
    objective = record["reference_solution"]["objective_value"]
    assert result["verified"] and result["feasible"], result["violations"][:5]
    assert result["violation_count"] == 0
    assert abs(result["cost"] - objective) <= 1e-3 * max(1.0, abs(objective))


@pytest.mark.parametrize("pid", sorted(EXPECTED_KINDS))
def test_scorer_returns_negative_recomputed_cost(pid):
    record = _load(pid)
    reference = _reference(record)
    score = compute_score(record["core_type"], record["instance"], reference)
    assert score == pytest.approx(-_check(record, reference)["cost"])


@pytest.mark.parametrize("pid", sorted(EXPECTED_KINDS))
def test_declared_objective_far_from_recomputed_is_a_violation(pid):
    record = _load(pid)
    solution = _reference(record)
    solution["objective_value"] = record["reference_solution"]["objective_value"] * 0.9
    result = _check(record, solution)
    assert not result["feasible"]
    assert any("declared objective" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_302", "prob_312"])
def test_crew_pairing_dropped_pairing_changes_cost_and_uncovered_list(pid):
    record = _load(pid)
    solution = _reference(record)
    dropped = solution["pairings"].pop()
    result = _check(record, solution)
    assert not result["feasible"]
    assert any("uncovered_flights" in v for v in result["violations"])
    expected = (
        record["reference_solution"]["objective_value"]
        - dropped["cost"]
        + len(dropped["flights"]) * record["instance"]["uncovered_flight_cost"]
    )
    assert result["cost"] == pytest.approx(expected)


@pytest.mark.parametrize("pid", ["prob_302", "prob_312"])
def test_crew_pairing_flags_bad_connection_and_leg_limit(pid):
    record = _load(pid)
    solution = _reference(record)
    longest = max(solution["pairings"], key=lambda p: len(p["flights"]))
    other = next(p for p in solution["pairings"] if p is not longest)
    longest["flights"] += other["flights"]
    result = _check(record, solution)
    assert any("invalid connections" in v for v in result["violations"])
    assert any("legs > 6" in v for v in result["violations"])


def test_crew_pairing_departure_outside_base_is_a_violation():
    record = _load("prob_302")
    bases = set(record["instance"]["bases"])
    flight = next(f for f in record["instance"]["flights"] if f["origin"] not in bases)
    solution = {"pairings": [{"flights": [flight["id"]]}], "uncovered_flights": []}
    result = _check(record, solution)
    assert any("non-base" in v for v in result["violations"])
    assert any("uncovered_flights" in v for v in result["violations"])


def test_crew_pairing_seniority_overloaded_crew_is_a_violation():
    record = _load("prob_330")
    solution = _reference(record)
    crew = record["instance"]["crews"][0]
    for key in solution["stage2"]["unassigned_pairings"][: crew["max_duties"] + 1]:
        solution["stage2"]["assignments"][str(key)] = {"crew": crew["id"]}
    result = _check(record, solution)
    assert any("max_duties" in v for v in result["violations"])
    assert any("unassigned_pairings" in v for v in result["violations"])


def test_crew_pairing_seniority_home_base_mismatch_adds_5000():
    record = _load("prob_330")
    solution = _reference(record)
    key, assignment = next(iter(solution["stage2"]["assignments"].items()))
    flights = {f["id"]: f for f in record["instance"]["flights"]}
    origin = flights[assignment["pairing"][0]]["origin"]
    crews = record["instance"]["crews"]
    home = next(c for c in crews if c["home_base"] == origin)
    away = next(c for c in crews if c["home_base"] != origin)
    base_cost = _check(record, solution)["cost"]
    for crew, delta in ((home, 0.0), (away, 5000.0)):
        variant = copy.deepcopy(solution)
        variant["stage2"]["assignments"][key] = {"crew": crew["id"]}
        span = flights[assignment["pairing"][-1]]["arr_time"] - flights[assignment["pairing"][0]]["dep_time"]
        expected = base_cost - assignment["bid"] + span * crew["span_pref"] * 100 + delta
        assert _check(record, variant)["cost"] == pytest.approx(expected)


@pytest.mark.parametrize("pid", ["prob_306", "prob_316", "prob_323"])
def test_roster_rules_are_checked_per_person(pid):
    record = _load(pid)
    solution = _reference(record)
    key, row = next(iter(solution["roster"].items()))
    cell = {"state": "night"} if isinstance(row[0], dict) else "night"
    solution["roster"][key] = [cell] * len(row)
    result = _check(record, solution)
    messages = " ".join(result["violations"])
    assert "night blocks outside 2-3 days" in messages
    assert "consecutive work days" in messages
    assert "off days" in messages
    assert "nights >" in messages


@pytest.mark.parametrize("pid", ["prob_306", "prob_316"])
def test_nurse_roster_forbidden_transition_is_detected(pid):
    record = _load(pid)
    solution = _reference(record)
    row = next(iter(solution["roster"].values()))
    row[0], row[1] = "late", "early"
    result = _check(record, solution)
    assert any("forbidden shift transitions" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_306", "prob_316"])
def test_nurse_roster_all_off_costs_the_full_shortage(pid):
    record = _load(pid)
    days = len(record["instance"]["daily_requirements"])
    solution = {"roster": {n["id"]: ["off"] * days for n in record["instance"]["nurses"]}}
    result = _check(record, solution)
    demand = sum(
        v for req in record["instance"]["daily_requirements"] for k, v in req.items() if k != "day"
    )
    assert result["feasible"]
    assert result["cost"] == pytest.approx(200 * demand)


def test_role_roster_region_and_qualification_are_hard_rules():
    record = _load("prob_323")
    solution = _reference(record)
    staff = record["instance"]["staff"]
    unqualified = next(s for s in staff if not s["qualified"])
    row = solution["roster"][str(unqualified["id"])]
    row[0] = {"state": "night", "region": "A" if unqualified["region"] == "B" else "B"}
    row[1] = {"state": "night", "region": unqualified["region"]}
    result = _check(record, solution)
    messages = " ".join(result["violations"])
    assert "outside home region" in messages
    assert "without qualification" in messages


def test_role_roster_preference_penalty_is_weighted_by_seniority():
    record = _load("prob_323")
    solution = _reference(record)
    staff = next(
        s
        for s in record["instance"]["staff"]
        if s["seniority"] > 1
        and solution["roster"][str(s["id"])][s["preferred_off_days"][0] - 1]["state"] == "off"
    )
    base = _check(record, solution)["cost"]
    row = solution["roster"][str(staff["id"])]
    row[staff["preferred_off_days"][0] - 1] = {"state": "early", "region": staff["region"]}
    changed = _check(record, solution)["cost"]
    # 希望休違反 8×seniority に加え、勤務日数が 1 増えるのでバランス項も動く
    assert changed - base >= 8 * staff["seniority"] - 3


@pytest.mark.parametrize(
    ("pid", "solution"),
    [
        ("prob_302", None),
        ("prob_302", {"pairings": [{"flights": ["x"]}]}),
        ("prob_330", {"stage1": {"pairings": []}, "stage2": {"assignments": []}}),
        ("prob_306", {"roster": {"1": ["off"] * 28}}),
        ("prob_323", {"roster": []}),
    ],
)
def test_unparseable_solution_is_unverified(pid, solution):
    record = _load(pid)
    result = _check(record, solution)
    assert result["verified"] is False
    assert result["cost"] is None


def test_shipped_problems_do_not_match_roster_kinds():
    names = set(EXPECTED_KINDS.values())
    for path in sorted(SHIPPED_DIR.glob("prob_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        kind = find_kind(record["instance"])
        assert kind is None or kind.name not in names, path.name

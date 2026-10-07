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


# prob_302 の同梱参照解は、問題文の「基地に戻る」を 200 ペアリング中 116 で破っている（既知欠陥）。
SHIPPED_REFERENCE_DEFECTS = {"prob_302"}


@pytest.mark.parametrize("pid", sorted(set(EXPECTED_KINDS) - SHIPPED_REFERENCE_DEFECTS))
def test_reference_solution_is_feasible_and_objective_is_reproduced(pid):
    record = _load(pid)
    kind = find_kind(record["instance"])
    assert kind is not None and kind.name == EXPECTED_KINDS[pid]
    result = _check(record, _reference(record))
    objective = record["reference_solution"]["objective_value"]
    assert result["verified"] and result["feasible"], result["violations"][:5]
    assert result["violation_count"] == 0
    assert abs(result["cost"] - objective) <= 1e-3 * max(1.0, abs(objective))


def test_shipped_302_reference_breaks_base_return_but_covers_each_flight_once():
    record = _load("prob_302")
    result = _check(record, _reference(record))
    assert result["verified"] and not result["feasible"]
    assert sum("ends at non-base" in v for v in result["violations"]) == 116
    assert not any("covered" in v for v in result["violations"])


def test_302_rules_apply_only_to_full_width_core_type():
    record = _load("prob_312")
    solution = _reference(record)
    solution["pairings"].append(copy.deepcopy(solution["pairings"][0]))
    lenient = _check(record, solution)
    assert not any("covered" in v or "ends at non-base" in v for v in lenient["violations"])
    strict = check_feasibility_detailed(
        _load("prob_302")["core_type"], record["instance"], solution
    )
    assert any("covered 2 times" in v for v in strict["violations"])
    assert any("ends at non-base" in v for v in strict["violations"])


@pytest.mark.parametrize("pid", sorted(set(EXPECTED_KINDS) - SHIPPED_REFERENCE_DEFECTS))
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
        ("prob_330", {"stage1": {"pairings": []}, "stage2": {"assignments": [{"crew_id": 5}]}}),
        ("prob_330", {"stage1": {"pairings": [[1], [1]]}, "stage2": {"p0": 1, "p1": 2}}),
        ("prob_306", {"roster": {"999": ["off"] * 28}}),
        ("prob_306", {"roster": {"1": [3] * 28, "2": [3] * 28}, "breakdown": {}}),
        ("prob_323", {"roster": [{"staff_id": 1, "day": 1, "shift": "off"}]}),
    ],
)
def test_unparseable_solution_is_unverified(pid, solution):
    record = _load(pid)
    result = _check(record, solution)
    assert result["verified"] is False
    assert result["cost"] is None


def test_empty_pairing_and_assignment_lists_are_read_as_all_uncovered():
    record = _load("prob_330")
    result = _check(record, {"stage1": {"pairings": []}, "stage2": {"assignments": []}})
    assert result["verified"] and result["feasible"]
    assert result["cost"] == pytest.approx(
        len(record["instance"]["flights"]) * record["instance"]["uncovered_flight_cost"]
    )


# ---------------------------------------------------------------- モデルが実際に返した形
@pytest.mark.parametrize(
    ("pid", "solution"),
    [
        ("prob_306", {"roster": {"1": ["off"] * 28}}),
        ("prob_306", {"roster": [{"nurse_id": 1, "schedule": ["off"] * 28}]}),
        ("prob_306", {"roster": {}}),
        ("prob_323", {"roster": []}),
    ],
)
def test_roster_missing_people_are_read_as_all_off(pid, solution):
    record = _load(pid)
    result = _check(record, solution)
    demand = sum(
        v for req in record["instance"]["daily_requirements"] for k, v in req.items() if k != "day"
    )
    assert result["verified"] and result["feasible"]
    assert result["cost"] == pytest.approx(200 * demand)


def test_nurse_roster_missing_nurses_are_charged_as_shortage_alongside_declared_mismatch():
    record = _load("prob_306")
    reference = _reference(record)
    key, row = next(iter(reference["roster"].items()))
    solution = {"objective_value": 0.0, "roster": {key: row}}
    result = _check(record, solution)
    full = _check(record, {"roster": {**{k: ["off"] * 28 for k in reference["roster"]}, key: row}})
    assert result["verified"] and not result["feasible"]
    assert result["cost"] == pytest.approx(full["cost"])
    assert any("declared objective 0.0" in v for v in result["violations"])


@pytest.mark.parametrize("pid", ["prob_306", "prob_316"])
def test_nurse_roster_reads_list_of_nurse_records(pid):
    record = _load(pid)
    reference = _reference(record)
    expected = _check(record, reference)
    records = [{"nurse_id": int(k), "schedule": row} for k, row in reference["roster"].items()]
    result = _check(record, {"roster": records})
    assert result["verified"] and result["feasible"]
    assert result["cost"] == pytest.approx(expected["cost"])


def test_role_roster_reads_id_schedule_records_without_region():
    record = _load("prob_323")
    reference = _reference(record)
    expected = _check(record, reference)
    records = [
        {"id": k, "schedule": [c["state"] for c in row], "shifts": [c["state"] for c in row]}
        for k, row in reference["roster"].items()
    ]
    result = _check(record, {"roster": records})
    assert result["verified"] and result["feasible"]
    assert result["cost"] == pytest.approx(expected["cost"])


def test_role_roster_reads_per_staff_day_records():
    record = _load("prob_323")
    reference = _reference(record)
    expected = _check(record, reference)
    records = [
        {"staff_id": int(k), "day": d, "shift": cell["state"]}
        for k, row in reference["roster"].items()
        for d, cell in enumerate(row, 1)
    ]
    result = _check(record, {"roster": records})
    assert result["verified"] and result["feasible"]
    assert result["cost"] == pytest.approx(expected["cost"])


def test_role_roster_falls_back_to_roster_by_staff_key():
    record = _load("prob_323")
    reference = _reference(record)
    expected = _check(record, reference)
    by_staff = {k: [c["state"] for c in row] for k, row in reference["roster"].items()}
    result = _check(record, {"roster_by_staff": by_staff})
    assert result["verified"]
    assert result["cost"] == pytest.approx(expected["cost"])


@pytest.mark.parametrize("pid", ["prob_306", "prob_323"])
def test_integer_coded_roster_without_legend_stays_unverified(pid):
    record = _load(pid)
    people = record["instance"].get("nurses") or record["instance"]["staff"]
    days = len(record["instance"]["daily_requirements"])
    result = _check(record, {"roster": {str(p["id"]): [0] * days for p in people}})
    assert result["verified"] is False


def test_crew_pairing_seniority_reads_assignment_records_and_outsourced_flights():
    record = _load("prob_330")
    reference = _reference(record)
    expected = _check(record, reference)
    stage1 = {
        "pairings": [
            {"pairing_id": i, "flights": p["flights"]}
            for i, p in enumerate(reference["stage1"]["pairings"], 1)
        ],
        "outsourced_flights": reference["stage1"]["uncovered_flights"],
    }
    stage2 = {
        "assignments": [
            {"pairing_id": int(k), "crew_id": a["crew"], "cost": 0}
            for k, a in reference["stage2"]["assignments"].items()
        ],
        "unassigned_pairings": reference["stage2"]["unassigned_pairings"],
    }
    result = _check(record, {"stage1": stage1, "stage2": stage2})
    assert result["verified"] and result["feasible"], result["violations"][:3]
    assert result["cost"] == pytest.approx(expected["cost"])


def test_crew_pairing_seniority_reads_label_keyed_stages():
    record = _load("prob_330")
    reference = _reference(record)
    expected = _check(record, reference)
    stage1 = {f"p{i}": p["flights"] for i, p in enumerate(reference["stage1"]["pairings"], 1)}
    stage2 = {f"p{k}": a["crew"] for k, a in reference["stage2"]["assignments"].items()}
    result = _check(record, {"stage1": stage1, "stage2": stage2})
    assert result["verified"] and result["feasible"], result["violations"][:3]
    assert result["cost"] == pytest.approx(expected["cost"])


def test_crew_pairing_seniority_assignment_to_unknown_pairing_label_is_a_violation():
    record = _load("prob_330")
    flight = record["instance"]["flights"][0]
    crew = record["instance"]["crews"][0]["id"]
    assignments = [{"pairing_id": 10, "crew": crew}, {"pairing_id": 11, "crew": crew}]
    solution = {
        "stage1": {"pairings": [{"id": 10, "flights": [flight["id"]]}]},
        "stage2": {"assignments": assignments},
    }
    result = _check(record, solution)
    assert result["verified"]
    assert any("does not name a stage1 pairing" in v for v in result["violations"])


def test_shipped_problems_do_not_match_roster_kinds():
    names = set(EXPECTED_KINDS.values())
    for path in sorted(SHIPPED_DIR.glob("prob_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        kind = find_kind(record["instance"])
        assert kind is None or kind.name not in names, path.name

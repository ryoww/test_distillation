"""大規模問題集の network_finance 群（mcnd / mcnd_surv / portfolio / portfolio_cvar）の検証器の振る舞い。"""

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

EXPECTED_KINDS = {
    "prob_308": "mcnd",
    "prob_318": "mcnd",
    "prob_325": "mcnd_surv",
    "prob_319": "portfolio",
    "prob_326": "portfolio_cvar",
}


def _load(problem_id: str) -> dict:
    record = json.loads((HARD_DIR / f"{problem_id}.json").read_text(encoding="utf-8"))
    record["core_type"] = f"{record['domain']}_{record['math_type']}"
    record["ref"] = {
        k: v for k, v in record["reference_solution"].items() if k not in ("objective_value", "note")
    }
    return record


RECORDS = {pid: _load(pid) for pid in EXPECTED_KINDS}


def _check(record: dict, solution: object) -> dict:
    return check_feasibility_detailed(record["core_type"], record["instance"], solution)


def _messages(result: dict) -> str:
    return "\n".join(result["violations"])


@pytest.mark.parametrize("problem_id", sorted(EXPECTED_KINDS))
def test_detects_expected_kind(problem_id):
    kind = find_kind(RECORDS[problem_id]["instance"])
    assert kind is not None and kind.name == EXPECTED_KINDS[problem_id]


@pytest.mark.parametrize("problem_id", sorted(EXPECTED_KINDS))
def test_reference_solution_is_feasible_and_cost_matches_declared(problem_id):
    record = RECORDS[problem_id]
    result = _check(record, record["ref"])
    objective = record["reference_solution"]["objective_value"]
    assert result["verified"] is True
    assert result["feasible"] is True, result["violations"]
    assert result["violation_count"] == 0
    assert abs(result["cost"] - objective) <= 1e-3 * max(1.0, abs(objective))


@pytest.mark.parametrize("problem_id", sorted(EXPECTED_KINDS))
def test_scorer_returns_negative_recomputed_cost(problem_id):
    record = RECORDS[problem_id]
    result = _check(record, record["ref"])
    score = compute_score(record["core_type"], record["instance"], record["ref"])
    assert score == pytest.approx(-result["cost"])


@pytest.mark.parametrize("problem_id", sorted(EXPECTED_KINDS))
def test_declared_objective_off_by_two_percent_is_reported(problem_id):
    record = RECORDS[problem_id]
    objective = record["reference_solution"]["objective_value"]
    solution = dict(record["ref"], objective_value=objective * 1.02)
    result = _check(record, solution)
    assert "declared objective_value" in _messages(result)
    exact = dict(record["ref"], objective_value=objective)
    assert _check(record, exact)["violation_count"] == 0


@pytest.mark.parametrize("problem_id", sorted(EXPECTED_KINDS))
def test_unparseable_solution_is_unverified_without_violations(problem_id):
    record = RECORDS[problem_id]
    for bad in ({"foo": 1}, [1, 2, 3], None):
        result = _check(record, bad)
        assert result.get("verified") is False
        assert result["violation_count"] == 0


def test_bundled_problems_do_not_match_any_hard_kind():
    for path in sorted(BUNDLED_DIR.glob("prob_*.json")):
        instance = json.loads(path.read_text(encoding="utf-8"))["instance"]
        assert find_kind(instance) is None, path.name


# ---------------------------------------------------------------------------
# mcnd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("problem_id", ["prob_308", "prob_318"])
def test_mcnd_dropped_od_is_reported_as_unmet_demand(problem_id):
    record = RECORDS[problem_id]
    solution = copy.deepcopy(record["ref"])
    del solution["od_paths"]["0"]
    result = _check(record, solution)
    assert result["feasible"] is False
    assert "OD 0 carries 0 of demand" in _messages(result)


@pytest.mark.parametrize("problem_id", ["prob_308", "prob_318"])
def test_mcnd_flow_on_closed_arc_is_reported(problem_id):
    record = RECORDS[problem_id]
    solution = copy.deepcopy(record["ref"])
    arc_id = solution["od_paths"]["0"][0]["path"][0]
    solution["opened_arc_ids"] = [a for a in solution["opened_arc_ids"] if a != arc_id]
    result = _check(record, solution)
    assert f"flows on closed arc {arc_id}" in _messages(result)
    arcs = {a["id"]: a for a in record["instance"]["arcs"]}
    reference_cost = _check(record, record["ref"])["cost"]
    assert result["cost"] == pytest.approx(reference_cost - arcs[arc_id]["fixed_cost"])


@pytest.mark.parametrize("problem_id", ["prob_308", "prob_318"])
def test_mcnd_capacity_overflow_is_reported(problem_id):
    record = RECORDS[problem_id]
    solution = copy.deepcopy(record["ref"])
    entry = solution["od_paths"]["0"][0]
    arcs = {a["id"]: a for a in record["instance"]["arcs"]}
    entry["volume"] += arcs[entry["path"][0]]["capacity"]
    result = _check(record, solution)
    assert f"arc {entry['path'][0]} carries" in _messages(result)
    assert "over capacity" in _messages(result)


def test_mcnd_disconnected_path_is_reported():
    record = RECORDS["prob_308"]
    solution = copy.deepcopy(record["ref"])
    solution["od_paths"]["0"][0]["path"] = solution["od_paths"]["0"][0]["path"][1:]
    result = _check(record, solution)
    assert "OD 0 is disconnected" in _messages(result)


def test_mcnd_arc_flows_are_checked_against_paths():
    record = RECORDS["prob_308"]
    solution = copy.deepcopy(record["ref"])
    first = next(iter(solution["arc_flows"]))
    solution["arc_flows"][first] += 10.0
    result = _check(record, solution)
    assert "arc_flows disagrees with od_paths on 1 arcs" in _messages(result)


# ---------------------------------------------------------------------------
# mcnd_surv
# ---------------------------------------------------------------------------


def test_surv_shared_arc_between_paths_is_reported():
    record = RECORDS["prob_325"]
    solution = copy.deepcopy(record["ref"])
    solution["od_paths"]["0"]["path2"] = list(solution["od_paths"]["0"]["path1"])
    result = _check(record, solution)
    assert "OD 0 shares arc" in _messages(result)


def test_surv_delay_limit_is_checked():
    record = RECORDS["prob_325"]
    instance = copy.deepcopy(record["instance"])
    arcs = {a["id"]: a for a in instance["arcs"]}
    length = sum(arcs[a]["distance"] for a in record["ref"]["od_paths"]["0"]["path1"])
    instance["od_demands"][0]["max_delay"] = length - 1.0
    result = check_feasibility_detailed(record["core_type"], instance, record["ref"])
    assert "OD 0 path1 length" in _messages(result)
    assert "exceeds max_delay" in _messages(result)


def test_surv_half_split_and_closed_arcs_are_checked():
    record = RECORDS["prob_325"]
    solution = copy.deepcopy(record["ref"])
    solution["od_paths"]["0"]["volume_each"] *= 1.5
    arc_id = solution["od_paths"]["1"]["path2"][0]
    solution["opened_arc_ids"] = [a for a in solution["opened_arc_ids"] if a != arc_id]
    result = _check(record, solution)
    assert "OD 0 splits" in _messages(result)
    assert f"OD 1 path2 flows on closed arc {arc_id}" in _messages(result)


# ---------------------------------------------------------------------------
# portfolio / portfolio_cvar
# ---------------------------------------------------------------------------


def test_portfolio_weight_above_max_is_reported():
    record = RECORDS["prob_319"]
    solution = copy.deepcopy(record["ref"])
    asset_id = next(iter(solution["weights"]))
    solution["weights"][asset_id] = 0.2
    result = _check(record, solution)
    assert f"asset {asset_id} weight 0.2 above max_weight" in _messages(result)
    assert "weights sum to" in _messages(result)


def test_portfolio_too_few_holdings_is_reported():
    record = RECORDS["prob_319"]
    solution = copy.deepcopy(record["ref"])
    kept = dict(list(solution["weights"].items())[:10])
    scale = 1.0 / sum(kept.values())
    solution["weights"] = {k: v * scale for k, v in kept.items()}
    result = _check(record, solution)
    assert "10 holdings below minimum 20" in _messages(result)


def test_portfolio_turnover_limit_is_reported():
    record = RECORDS["prob_319"]
    instance = record["instance"]
    held = {int(k) for k in record["ref"]["weights"]} | {int(k) for k in instance["current_portfolio"]}
    fresh = [a["id"] for a in instance["assets"] if a["id"] not in held][:25]
    solution = {"weights": {str(a): 1.0 / len(fresh) for a in fresh}}
    result = _check(record, solution)
    assert "turnover 2 above 0.9" in _messages(result)


def test_portfolio_accepts_weight_list_and_recomputes_mad():
    record = RECORDS["prob_319"]
    instance = record["instance"]
    weights = [0.0] * len(instance["assets"])
    for key, value in record["ref"]["weights"].items():
        weights[int(key) - 1] = value
    result = _check(record, {"weights": weights})
    assert result["feasible"] is True
    assert result["cost"] == pytest.approx(record["reference_solution"]["objective_value"], abs=1e-6)


def test_cvar_selling_current_holding_is_reported():
    record = RECORDS["prob_326"]
    solution = copy.deepcopy(record["ref"])
    asset_id, weight = next(iter(record["instance"]["current_portfolio"].items()))
    solution["weights"][asset_id] = weight / 2
    result = _check(record, solution)
    assert f"asset {asset_id} sold below current weight" in _messages(result)


def test_cvar_sector_cardinality_is_checked():
    record = RECORDS["prob_326"]
    instance = record["instance"]
    sector_of = {a["id"]: a["sector"] for a in instance["assets"]}
    solution = copy.deepcopy(record["ref"])
    # 業種 5 は保有数 1〜2 が要件で参照解は 1 銘柄。3 銘柄に増やして上限超過を作る。
    extra = [a["id"] for a in instance["assets"] if sector_of[a["id"]] == 5][:3]
    for asset_id in extra:
        solution["weights"][str(asset_id)] = 0.02
    result = _check(record, solution)
    assert "sector 5 holds" in _messages(result)
    assert "outside [1, 2]" in _messages(result)


def test_cvar_expected_return_below_target_is_reported():
    record = RECORDS["prob_326"]
    instance = copy.deepcopy(record["instance"])
    instance["target_return"] = 0.2
    result = check_feasibility_detailed(record["core_type"], instance, record["ref"])
    assert "expected return" in _messages(result)
    assert "below target 0.2" in _messages(result)

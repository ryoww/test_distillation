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


# ---------------------------------------------------------------------------
# 保存解（outputs/rescore_hard）で実際に見られた形
# ---------------------------------------------------------------------------


def _nodes(instance: dict, path: list[int]) -> list[int]:
    """アーク id 列を同じ経路のノード列に直す。"""
    arcs = {a["id"]: a for a in instance["arcs"]}
    return [arcs[path[0]]["from"]] + [arcs[a]["to"] for a in path]


def _assert_same_as_reference(record: dict, solution: dict) -> None:
    expected = _check(record, record["ref"])
    result = _check(record, solution)
    assert result["verified"] is True
    assert result["violations"] == []
    assert result["cost"] == pytest.approx(expected["cost"])


def test_mcnd_reads_records_with_origin_destination_and_nested_paths():
    # Qwen3.8 / Claude の prob_318: list の要素が {origin, destination, paths: [{path: ノード列, arc_ids, flow}]}
    record = RECORDS["prob_318"]
    instance = record["instance"]
    od_paths = []
    for index, entries in record["ref"]["od_paths"].items():
        od = instance["od_demands"][int(index)]
        od_paths.append(
            {
                "origin": od["origin"],
                "destination": od["destination"],
                "volume": od["volume"],
                "path": _nodes(instance, entries[0]["path"]),
                "paths": [
                    {"path": _nodes(instance, e["path"]), "arc_ids": e["path"], "flow": e["volume"]}
                    for e in entries
                ],
            }
        )
    _assert_same_as_reference(record, dict(record["ref"], od_paths=od_paths))


def test_mcnd_reads_node_paths_when_no_arc_ids_are_given():
    record = RECORDS["prob_318"]
    instance = record["instance"]
    od_paths = {
        index: [{"path": _nodes(instance, e["path"]), "volume": e["volume"]} for e in entries]
        for index, entries in record["ref"]["od_paths"].items()
    }
    _assert_same_as_reference(record, dict(record["ref"], od_paths=od_paths))


def test_mcnd_pair_keyed_arc_lists_route_the_whole_pair_demand():
    # Gemma の prob_318: od_paths が {"origin-destination": [arc ids]}。同じ対の OD はまとめて需要を満たす。
    record = RECORDS["prob_318"]
    instance = record["instance"]
    od_paths: dict[str, list[int]] = {}
    for index, entries in record["ref"]["od_paths"].items():
        od = instance["od_demands"][int(index)]
        od_paths.setdefault(f"{od['origin']}-{od['destination']}", entries[0]["path"])
    assert len(od_paths) == 290
    result = _check(record, {"opened_arc_ids": record["ref"]["opened_arc_ids"], "od_paths": od_paths})
    assert result["verified"] is True
    # 分流していた OD を 1 本にまとめたので容量超過は出るが、需要の未達と未知の対は出ない。
    assert "of demand" not in _messages(result)
    assert "unknown" not in _messages(result)


def test_mcnd_zero_based_pair_keys_are_reported_not_corrected():
    # Qwen3.6 の prob_318: ノード番号を 0 始まりにした "34_9" キー。instance に無い対はそのまま違反にする。
    record = RECORDS["prob_318"]
    result = _check(record, {"opened_arc_ids": [240, 385], "od_paths": {"34_9": [240, 385]}})
    assert result["verified"] is True
    assert "od_paths names unknown OD pair 34->9" in _messages(result)
    assert "OD 0 carries 0 of demand 15" in _messages(result)


def test_mcnd_reads_flow_distribution_and_checks_conservation():
    # Qwen3.6 の prob_308: {origin, destination, volume, flow_distribution: {arc_id: 流量}} の list
    record = RECORDS["prob_308"]
    instance = record["instance"]
    od_paths = []
    for index, entries in record["ref"]["od_paths"].items():
        od = instance["od_demands"][int(index)]
        distribution: dict[str, float] = {}
        for entry in entries:
            for arc_id in entry["path"]:
                distribution[str(arc_id)] = distribution.get(str(arc_id), 0.0) + entry["volume"]
        od_paths.append(dict(od, flow_distribution=distribution))
    _assert_same_as_reference(record, dict(record["ref"], od_paths=od_paths))

    del od_paths[0]["flow_distribution"][next(iter(od_paths[0]["flow_distribution"]))]
    result = _check(record, {"opened_arc_ids": record["ref"]["opened_arc_ids"], "od_paths": od_paths})
    assert "OD 0 flow is unbalanced at node" in _messages(result)


def test_mcnd_split_over_paths_without_volumes_is_unverified():
    record = RECORDS["prob_308"]
    path = record["ref"]["od_paths"]["0"][0]["path"]
    result = _check(record, {"opened_arc_ids": path, "od_paths": {"0": [path, path]}})
    assert result["verified"] is False
    assert result["violation_count"] == 0


def test_mcnd_positional_arc_flows_list_is_checked():
    # Qwen3.6 の prob_308: arc_flows がアーク順の list
    record = RECORDS["prob_308"]
    declared = {int(k): v for k, v in record["ref"]["arc_flows"].items()}
    arc_flows = [declared.get(a["id"], 0.0) for a in record["instance"]["arcs"]]
    _assert_same_as_reference(record, dict(record["ref"], arc_flows=arc_flows))
    arc_flows[0] += 10.0
    result = _check(record, dict(record["ref"], arc_flows=arc_flows))
    assert "arc_flows disagrees with od_paths on 1 arcs" in _messages(result)


def _surv_records(record: dict) -> list[dict]:
    """Claude の prob_325 の形: {origin, destination, paths: [{arcs, nodes, flow}, ...]} の list。"""
    instance = record["instance"]
    out = []
    for index, entry in record["ref"]["od_paths"].items():
        od = instance["od_demands"][int(index)]
        out.append(
            {
                "origin": od["origin"],
                "destination": od["destination"],
                "paths": [
                    {"arcs": entry[k], "nodes": _nodes(instance, entry[k]), "flow": entry["volume_each"]}
                    for k in ("path1", "path2")
                ],
            }
        )
    return out


def test_surv_reads_paths_records_with_arcs_and_flow():
    record = RECORDS["prob_325"]
    solution = {"opened_arc_ids": record["ref"]["opened_arc_ids"], "od_paths": _surv_records(record)}
    _assert_same_as_reference(record, solution)


def test_surv_unequal_split_between_paths_is_reported():
    record = RECORDS["prob_325"]
    od_paths = _surv_records(record)
    od_paths[0]["paths"][0]["flow"] = 50.0
    od_paths[0]["paths"][1]["flow"] = 24.0
    result = _check(record, {"opened_arc_ids": record["ref"]["opened_arc_ids"], "od_paths": od_paths})
    assert "OD 0 splits unequally [50.0, 24.0]" in _messages(result)


def test_surv_pair_keyed_node_paths_are_converted_to_arcs():
    # Gemma の prob_325: {"origin": {"to": destination, "paths": [ノード列, ノード列]}}。無い OD は違反として積む。
    record = RECORDS["prob_325"]
    od_paths = {"14": {"to": 2, "paths": [[14, 13, 11, 2], [14, 23, 24, 9, 2]]}}
    result = _check(record, {"opened_arc_ids": record["ref"]["opened_arc_ids"], "od_paths": od_paths})
    assert result["verified"] is True
    assert "OD 0 " not in _messages(result)
    assert "OD 1 has no paths" in _messages(result)


def test_surv_node_path_over_parallel_arcs_is_unverified():
    # prob_325 ではノード 11→10 に並行アーク 48, 150 があり、ノード列からはどちらか決められない。
    record = RECORDS["prob_325"]
    od_paths = {"0": {"paths": [{"nodes": [11, 10]}, {"nodes": [11, 10]}]}}
    result = _check(record, {"opened_arc_ids": [], "od_paths": od_paths})
    assert result["verified"] is False
    assert "parallel arcs" in result["violations"][0]


def test_mcnd_origin_destination_pairs_without_routes_are_unmet_demand():
    # Qwen3.8 (gepa) の prob_318: od_paths が OD 順の [origin, destination] だけで、開設も流量も無い。
    record = RECORDS["prob_318"]
    instance = record["instance"]
    od_paths = [[od["origin"], od["destination"]] for od in instance["od_demands"]]
    result = _check(record, {"opened_arc_ids": [], "arc_flows": [], "od_paths": od_paths})
    assert result["verified"] is True
    assert result["feasible"] is False
    assert "OD 0 carries 0 of demand 15" in _messages(result)
    assert "closed arc" not in _messages(result)
    assert result["cost"] is None


def test_mcnd_direct_arc_node_path_in_route_dict_is_still_a_path():
    # {path: [origin, destination], volume} は経路の記述なので、直行アークのノード列として読む。
    record = RECORDS["prob_318"]
    instance = record["instance"]
    arcs = {a["id"]: a for a in instance["arcs"]}
    index, entries = next(
        (i, e) for i, e in record["ref"]["od_paths"].items() if len(e[0]["path"]) == 1
    )
    arc = arcs[entries[0]["path"][0]]
    od_paths = {index: [{"path": [arc["from"], arc["to"]], "volume": entries[0]["volume"]}]}
    result = _check(record, {"opened_arc_ids": [arc["id"]], "od_paths": od_paths})
    assert result["verified"] is True
    assert f"OD {index} " not in _messages(result)


def test_surv_list_of_empty_routes_is_unmet_not_unverified():
    # Qwen3.8 (gepa) の prob_325: od_paths が OD 順の [[], []] で、開設アークも無い。
    record = RECORDS["prob_325"]
    od_paths = [[[], []] for _ in record["instance"]["od_demands"]]
    result = _check(record, {"opened_arc_ids": [], "od_paths": od_paths})
    assert result["verified"] is True
    assert result["feasible"] is False
    assert "OD 0 path1 ends at node 14, not destination 2" in _messages(result)
    assert "OD 0 path2 ends at node 14, not destination 2" in _messages(result)
    assert result["cost"] is None


def test_surv_list_of_arc_routes_matches_reference():
    # path1/path2 を dict にせず [path1, path2] と並べた形は参照解と同じ値になる。
    record = RECORDS["prob_325"]
    od_paths = [
        [entry["path1"], entry["path2"]]
        for _, entry in sorted(record["ref"]["od_paths"].items(), key=lambda kv: int(kv[0]))
    ]
    _assert_same_as_reference(record, {"opened_arc_ids": record["ref"]["opened_arc_ids"], "od_paths": od_paths})

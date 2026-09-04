"""feasibility.py に直接登録された旧チェッカーが、同梱の参照解を弾かないことを確認する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.feasibility import check_feasibility_detailed

BASE_DIR = Path(__file__).resolve().parents[1]
PROBLEM_DIR = BASE_DIR / "data" / "problems"


def _load_problems() -> list[dict]:
    records = []
    for path in sorted(PROBLEM_DIR.glob("prob_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["core_type"] = f"{record['domain']}_{record['math_type']}"
        records.append(record)
    return records


PROBLEMS = _load_problems()
# prob_012 の同梱参照解は total_priority を 6 と申告するが、割り当てたジョブの優先度合計は 14。
# 参照解側の欠陥なので、チェッカーが弾くことを別のテストで固定する。
KNOWN_DEFECTIVE_REFERENCES = {12}


@pytest.mark.parametrize(
    "record",
    [r for r in PROBLEMS if r["id"] not in KNOWN_DEFECTIVE_REFERENCES],
    ids=lambda r: f"prob_{r['id']:03d}",
)
def test_every_shipped_reference_passes_its_own_checker(record):
    result = check_feasibility_detailed(
        record["core_type"], record["instance"], record["reference_solution"]
    )
    assert result["feasible"] is True, result["violations"]
    assert result["violation_count"] == 0


def test_prob_012_reference_misreports_its_objective():
    record = next(r for r in PROBLEMS if r["id"] == 12)
    result = check_feasibility_detailed(
        record["core_type"], record["instance"], record["reference_solution"]
    )
    assert result["feasible"] is False
    assert result["violations"] == ["total_priority 6 != sum of assigned priorities 14"]


def test_parallel_machine_reference_shape_is_read():
    instance = {
        "num_machines": 2,
        "jobs": [{"id": 1, "processing_time": 3}, {"id": 2, "processing_time": 4}],
    }
    solution = {
        "machine_assignment": {"1": 1, "2": 2},
        "makespan": 4,
        "machine_loads": {"1": 3, "2": 4},
    }
    result = check_feasibility_detailed("スケジューリング_整数計画", instance, solution)
    assert result["feasible"] is True, result["violations"]


def test_parallel_machine_rejects_unknown_machine_and_wrong_makespan():
    instance = {
        "num_machines": 2,
        "jobs": [{"id": 1, "processing_time": 3}, {"id": 2, "processing_time": 4}],
    }
    zero_based = {"machine_assignment": {"1": 0, "2": 1}, "makespan": 4, "machine_loads": [3, 4]}
    assert check_feasibility_detailed("スケジューリング_整数計画", instance, zero_based)["feasible"]
    bad_machine = {"machine_assignment": {"1": 1, "2": 5}, "makespan": 4}
    result = check_feasibility_detailed("スケジューリング_整数計画", instance, bad_machine)
    assert result["feasible"] is False
    assert any("unknown machine" in v for v in result["violations"])
    wrong_makespan = {"machine_assignment": {"1": 1, "2": 1}, "makespan": 4}
    result = check_feasibility_detailed("スケジューリング_整数計画", instance, wrong_makespan)
    assert result["feasible"] is False
    assert any("max machine load" in v for v in result["violations"])


def test_flow_shop_sequence_must_be_a_permutation_with_matching_makespan():
    instance = {
        "num_stages": 2,
        "jobs": [
            {"id": 1, "processing_times": {"stage_1": 3, "stage_2": 2}},
            {"id": 2, "processing_times": {"stage_1": 1, "stage_2": 4}},
        ],
    }
    good = {"optimal_sequence": [2, 1], "makespan": 7, "note": ""}
    assert check_feasibility_detailed("スケジューリング_混合整数計画", instance, good)["feasible"]
    duplicated = {"optimal_sequence": [1, 1], "makespan": 7, "note": ""}
    result = check_feasibility_detailed("スケジューリング_混合整数計画", instance, duplicated)
    assert result["feasible"] is False
    understated = {"optimal_sequence": [2, 1], "makespan": 5, "note": ""}
    result = check_feasibility_detailed("スケジューリング_混合整数計画", instance, understated)
    assert any("flow shop makespan" in v for v in result["violations"])


def test_cluster_assignment_checks_each_job_fits_its_node():
    instance = {
        "nodes": [{"id": 1, "cpu_cores": 4, "gpu_count": 0, "memory_gb": 8}],
        "jobs": [
            {"id": 1, "cpu_required": 2, "gpu_required": 0, "memory_required_gb": 4, "priority": 3},
            {"id": 2, "cpu_required": 8, "gpu_required": 0, "memory_required_gb": 4, "priority": 9},
        ],
    }
    fits = {"node_assignment": {"1": 1}, "total_priority": 3, "assigned_count": 1}
    assert check_feasibility_detailed("スケジューリング_混合整数計画", instance, fits)["feasible"]
    too_big = {"node_assignment": {"1": 1, "2": 1}, "total_priority": 12, "assigned_count": 2}
    result = check_feasibility_detailed("スケジューリング_混合整数計画", instance, too_big)
    assert result["feasible"] is False
    assert any("cpu_required" in v for v in result["violations"])
    lying = {"node_assignment": {"1": 1}, "total_priority": 30, "assigned_count": 1}
    result = check_feasibility_detailed("スケジューリング_混合整数計画", instance, lying)
    assert any("total_priority" in v for v in result["violations"])


def test_missing_objective_is_still_a_violation():
    instance = {"jobs": [{"id": 1}, {"id": 2}]}
    solution = {"schedule": {"1": {}, "2": {}}, "note": "no numbers"}
    result = check_feasibility_detailed("スケジューリング_混合整数計画", instance, solution)
    assert result["feasible"] is False
    assert any("makespan" in v for v in result["violations"])


def test_day1_routes_must_start_and_end_at_the_warehouse():
    instance = {
        "customers": [{"id": 1}, {"id": 2}],
        "warehouse": {"id": 0},
        "num_vehicles": 1,
    }
    good = {"day1_routes": {"1": {"route": [0, 1, 2, 0], "distance": 9}}, "total_distance": 9}
    assert check_feasibility_detailed("配送・輸送_混合整数計画", instance, good)["feasible"]
    partial_list = {"day1_routes": [[0, 2, 0]], "total_distance": 5.5}
    assert check_feasibility_detailed("配送・輸送_混合整数計画", instance, partial_list)["feasible"]
    no_depot = {"day1_routes": {"1": {"route": [1, 2, 1], "distance": 9}}, "total_distance": 9}
    result = check_feasibility_detailed("配送・輸送_混合整数計画", instance, no_depot)
    assert result["feasible"] is False
    too_many = {
        "day1_routes": {
            "1": {"route": [0, 1, 0], "distance": 4},
            "2": {"route": [0, 2, 0], "distance": 5},
        },
        "total_distance": 9,
    }
    result = check_feasibility_detailed("配送・輸送_混合整数計画", instance, too_many)
    assert any("vehicles" in v for v in result["violations"])

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
# 同梱参照解が壊れていると分かっている問題。参照解側の欠陥なので、チェッカーが弾くことを
# 別のテストで固定する。
# - prob_011: 「greedy+shifting近似解」。活動 5 と 7 が先行活動 4 の完了前に始まり、
#   作業員が 7 人（peak_workers も 7 と申告）で容量 6 を超える。
# - prob_012: total_priority を 6 と申告するが割り当てたジョブの優先度合計は 14。さらにノード 1
#   （64 GB）に 96 GB、ノード 2（32 GB）に 64 GB のジョブを同居させている。
# - prob_015: 週 3 時限の科目 3・4・7 に 1 時限しか置いておらず、num_periods を無視している。
# - prob_020: 14 活動のうち 3 つしか置かず、その所要日数も合わない。ピーク 19 は容量 8 を超え、
#   置いた 3 活動から再計算した値 4 とも一致しない。
KNOWN_DEFECTIVE_REFERENCES = {11, 12, 15, 20}


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
    assert result["violations"] == [
        "node 1 memory_gb=64 < total memory_required_gb 96",
        "node 2 memory_gb=32 < total memory_required_gb 64",
        "total_priority 6 != sum of assigned priorities 14",
    ]


@pytest.mark.parametrize(
    ("problem_id", "expected"),
    [
        (11, "workers usage peaks at 7 > capacity 6"),
        (15, "class 3 has 1 periods, needs 3"),
        (20, "peak_resource_usage 19 != recomputed 4"),
    ],
    ids=lambda v: f"prob_{v:03d}" if isinstance(v, int) else "",
)
def test_other_defective_references_are_rejected_for_the_recorded_reason(problem_id, expected):
    record = next(r for r in PROBLEMS if r["id"] == problem_id)
    result = check_feasibility_detailed(
        record["core_type"], record["instance"], record["reference_solution"]
    )
    assert result["verified"] is True
    assert result["feasible"] is False
    assert expected in result["violations"], result["violations"]


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


def test_parallel_machine_accepts_machine_to_jobs_mapping():
    instance = {
        "num_machines": 2,
        "jobs": [
            {"id": 1, "processing_time": 3},
            {"id": 2, "processing_time": 4},
            {"id": 3, "processing_time": 2},
        ],
    }
    solution = {
        "machine_assignment": {"0": [1, 3], "1": [2]},
        "makespan": 5,
        "machine_loads": [5, 4],
    }
    result = check_feasibility_detailed("スケジューリング_整数計画", instance, solution)
    assert result["feasible"] is True, result["violations"]
    wrong = {"machine_assignment": {"0": [1, 3], "1": [2]}, "makespan": 4}
    result = check_feasibility_detailed("スケジューリング_整数計画", instance, wrong)
    assert any("max machine load" in v for v in result["violations"])

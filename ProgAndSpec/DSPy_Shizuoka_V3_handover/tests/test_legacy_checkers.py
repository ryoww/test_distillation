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


@pytest.mark.parametrize("record", PROBLEMS, ids=lambda r: f"prob_{r['id']:03d}")
def test_every_shipped_reference_passes_its_own_checker(record):
    result = check_feasibility_detailed(
        record["core_type"], record["instance"], record["reference_solution"]
    )
    assert result["feasible"] is True, result["violations"]
    assert result["violation_count"] == 0


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


def test_priority_objective_counts_as_objective():
    instance = {"nodes": [{"id": 1}], "jobs": [{"id": 1}, {"id": 2}]}
    solution = {"node_assignment": {"1": 1, "2": 1}, "total_priority": 5, "assigned_count": 2}
    result = check_feasibility_detailed("スケジューリング_混合整数計画", instance, solution)
    assert result["feasible"] is True, result["violations"]


def test_missing_objective_is_still_a_violation():
    instance = {"jobs": [{"id": 1}, {"id": 2}]}
    solution = {"schedule": {"1": {}, "2": {}}, "note": "no numbers"}
    result = check_feasibility_detailed("スケジューリング_混合整数計画", instance, solution)
    assert result["feasible"] is False
    assert any("objective" in v for v in result["violations"])


def test_day1_routes_are_read_as_routes():
    instance = {"customers": [{"id": 1}, {"id": 2}], "warehouse": {"id": 0}}
    solution = {"day1_routes": {"1": {"route": [0, 1, 2, 0], "distance": 9}}, "total_distance": 9}
    result = check_feasibility_detailed("配送・輸送_混合整数計画", instance, solution)
    assert result["feasible"] is True, result["violations"]

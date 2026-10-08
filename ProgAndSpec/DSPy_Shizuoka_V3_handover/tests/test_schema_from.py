import json

from scripts.evaluate_solver_model import schema_table


def _write(tmp_path, pid, solution):
    record = {
        "id": pid,
        "name": "toy",
        "domain": "スケジューリング",
        "math_type": "整数計画",
        "requirements": {"objective": "min"},
        "instance": {"items": [1, 2]},
        "reference_solution": solution,
    }
    (tmp_path / f"prob_{pid:03d}.json").write_text(json.dumps(record), encoding="utf-8")


def test_most_common_return_shape_wins_per_kind(tmp_path):
    _write(tmp_path, 1, {"objective_value": 1, "schedule": [{"job_id": 1}]})
    _write(tmp_path, 2, {"objective_value": 2, "schedule": [{"job_id": 3}]})
    _write(tmp_path, 3, {"objective_value": 3, "plan": {"1": []}, "lns_iterations": 9})
    table = schema_table(tmp_path)
    assert list(table) == ["スケジューリング_整数計画"]
    assert set(table["スケジューリング_整数計画"]) == {"objective_value", "schedule"}

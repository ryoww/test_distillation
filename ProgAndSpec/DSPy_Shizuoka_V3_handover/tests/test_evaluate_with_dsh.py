import json

from scripts.evaluate_with_dsh import RESULT_FILENAME, count_events, load_drafts, prepare


def test_workspace_holds_the_problem_without_the_reference_value(tmp_path):
    record = {
        "id": 7,
        "name": "toy",
        "domain": "スケジューリング",
        "math_type": "整数計画",
        "description": "Schedule the jobs.",
        "requirements": {"objective": "min makespan"},
        "instance": {"jobs": [1, 2]},
        "reference_solution": {"objective_value": 123.456},
    }
    example = {"record": record, "instance": record["instance"], "core_type": "toy_kind"}
    prepare(tmp_path, example, 300)
    assert "123.46" not in (tmp_path / "problem.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "instance.json").read_text()) == {"jobs": [1, 2]}
    assert json.loads((tmp_path / "meta.json").read_text()) == {"core_type": "toy_kind", "timeout": 300}
    assert (tmp_path / "check.sh").stat().st_mode & 0o111
    assert "verify_solve.py" in (tmp_path / "check.sh").read_text()


def test_event_stream_is_counted_by_type_and_noise_is_skipped():
    stream = "\n".join(
        [json.dumps({"type": "tool_call"}), "not json", json.dumps({"type": "tool_call"}),
         json.dumps({"type": "final"})]
    )
    assert count_events(stream) == {"tool_call": 2, "final": 1}


def test_drafts_are_read_per_problem_and_rows_without_code_are_skipped(tmp_path):
    rows = [{"instance_id": "prob_301", "code": "def solve(i):\n    return {}\n"},
            {"instance_id": "prob_302", "code": None}]
    (tmp_path / RESULT_FILENAME).write_text(json.dumps({"test": {"results": rows}}), encoding="utf-8")
    assert load_drafts(tmp_path) == {"prob_301": "def solve(i):\n    return {}\n"}

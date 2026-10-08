from scripts.build_self_sft_dataset import to_rows


def _record(i):
    return {"id": i, "name": "n", "domain": "d", "math_type": "m", "description": "desc",
            "requirements": {"objective": "min"}, "instance": {"x": i}, "reference_solution": {"objective_value": 5}}


def test_only_solved_rows_become_pairs_and_validation_is_split_by_id():
    records = {"prob_020": _record(20), "prob_021": _record(21), "prob_022": _record(22)}
    results = [
        {"instance_id": "prob_020", "status": "exact_match", "code": "def solve(i):\n    return 1"},
        {"instance_id": "prob_021", "status": "beat_reference", "code": "def solve(i):\n    return 2"},
        {"instance_id": "prob_022", "status": "worse", "code": "def solve(i):\n    return 3"},
    ]
    rows = to_rows(results, records, "SYS", every=20)
    assert [r["instance_id"] for r in rows["validation"]] == ["prob_020"]
    assert [r["instance_id"] for r in rows["train"]] == ["prob_021"]
    message = rows["train"][0]["messages"]
    assert message[0] == {"role": "system", "content": "SYS"} and message[2]["content"].startswith("```python")

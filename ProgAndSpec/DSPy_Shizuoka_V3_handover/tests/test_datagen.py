"""雛形ベースの問題生成器が、既存の参照値を再現し、検証済みの新問題を作れることを確認する。"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from src.datagen import (
    TEMPLATES,
    ValidationError,
    generate_dataset,
    load_base,
    make_problem,
    shape_signature,
    validate_problem,
)

BASE_DIR = Path(__file__).resolve().parents[1]
PROBLEM_DIR = BASE_DIR / "data" / "problems"
TEMPLATE_IDS = sorted(TEMPLATES)


def _objective_of(record: dict, key: str) -> float:
    return float(record["reference_solution"][key])


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_solver_reproduces_original_reference_value(template_id):
    template = TEMPLATES[template_id]
    base = load_base(PROBLEM_DIR, template_id)
    solved = template.solve(base["instance"])
    assert solved[template.objective_key] == pytest.approx(
        _objective_of(base, template.objective_key), rel=2e-3
    )
    assert set(solved) == set(base["reference_solution"])


@pytest.mark.parametrize("template_id", TEMPLATE_IDS)
def test_generated_problem_keeps_shape_and_passes_checker(template_id):
    template = TEMPLATES[template_id]
    base = load_base(PROBLEM_DIR, template_id)
    record = make_problem(
        base, template, random.Random(f"t:{template_id}"), new_id=5000, seed_label="t"
    )
    assert shape_signature(record["instance"]) == shape_signature(base["instance"])
    assert record["instance"] != base["instance"]
    assert record["description"] == base["description"]
    assert record["provenance"]["template_id"] == template_id


def test_generation_is_deterministic_for_a_seed():
    first = generate_dataset(PROBLEM_DIR, template_ids=[1, 55], per_template=2, seed=7, start_id=1)
    second = generate_dataset(PROBLEM_DIR, template_ids=[1, 55], per_template=2, seed=7, start_id=1)
    assert first == second
    assert [r["id"] for r in first] == [1, 2, 3, 4]


def test_generation_matches_across_processes_for_tie_rich_templates():
    """同点最適解の多い雛形（ビンパッキング・TSP）でも別プロセスで同じ参照解になる。"""
    ids = [59, 89]
    here = generate_dataset(PROBLEM_DIR, template_ids=ids, per_template=2, seed=11, start_id=1)
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "from pathlib import Path; from src.datagen import generate_dataset;"
        "print(json.dumps(generate_dataset(Path(sys.argv[2]), template_ids=[59, 89],"
        " per_template=2, seed=11, start_id=1), ensure_ascii=False))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(BASE_DIR), str(PROBLEM_DIR)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(proc.stdout) == json.loads(json.dumps(here))


def test_different_seeds_give_different_instances():
    a = generate_dataset(PROBLEM_DIR, template_ids=[67], per_template=1, seed=1, start_id=1)
    b = generate_dataset(PROBLEM_DIR, template_ids=[67], per_template=1, seed=2, start_id=1)
    assert a[0]["instance"] != b[0]["instance"]


def test_validation_rejects_wrong_objective():
    template = TEMPLATES[55]
    base = load_base(PROBLEM_DIR, 55)
    record = make_problem(base, template, random.Random(0), new_id=1, seed_label="x")
    record["reference_solution"]["max_value"] += 1
    with pytest.raises(ValidationError):
        validate_problem(record, base, template)


def test_validation_rejects_shape_drift():
    template = TEMPLATES[55]
    base = load_base(PROBLEM_DIR, 55)
    record = make_problem(base, template, random.Random(0), new_id=1, seed_label="x")
    record["instance"]["items"].append({"id": 99, "weight": 1, "value": 1})
    with pytest.raises(ValidationError):
        validate_problem(record, base, template)

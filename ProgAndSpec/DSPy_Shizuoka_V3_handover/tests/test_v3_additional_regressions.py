"""Additional regression tests for reference extraction and feasibility trust."""

import json
from pathlib import Path

import pytest

pytest.importorskip("dspy")

from src.best_known import BestKnownRegistry
from src.data_loader import reference_value_from_solution
from src.metrics_v3 import evaluate_algorithm_v3
from src.utils.feasibility import CHECKERS_DETAILED, check_feasibility_detailed

PROBLEMS_DIR = Path(__file__).parents[1] / "data" / "problems"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("prob_020.json", 19.0),
        ("prob_080.json", 141.0),
        ("prob_009.json", None),
    ],
)
def test_reference_value_matches_selected_problem_objective(filename, expected):
    """What: selected problem files expose the intended reference objective value."""
    record = json.loads((PROBLEMS_DIR / filename).read_text(encoding="utf-8"))

    assert reference_value_from_solution(record) == expected


def test_boolean_objective_value_is_not_numeric():
    """What: boolean objective markers cannot become numeric reference values."""
    record = {
        "requirements": {"objective": "Check whether every item is assigned"},
        "reference_solution": {"objective_value": True, "all_assigned": True},
    }

    assert reference_value_from_solution(record) is None


def test_unregistered_non_empty_solution_is_feasible_but_unverified(monkeypatch):
    """What: an unregistered checker accepts shape but marks feasibility unverified."""
    core_type = "regression_unregistered_feasibility"
    monkeypatch.delitem(CHECKERS_DETAILED, core_type, raising=False)

    result = check_feasibility_detailed(
        core_type,
        {"items": [1]},
        {"items": [1]},
    )

    assert result["feasible"] is True
    assert result["verified"] is False


def test_raising_registered_checker_is_infeasible_and_unverified(monkeypatch):
    """What: a checker exception produces an infeasible, unverified result."""
    core_type = "regression_raising_feasibility"

    def raising_checker(_instance, _solution):
        raise RuntimeError("regression checker failure")

    monkeypatch.setitem(CHECKERS_DETAILED, core_type, raising_checker)

    result = check_feasibility_detailed(
        core_type,
        {"items": [1]},
        {"items": [1]},
    )

    assert result["feasible"] is False
    assert result["verified"] is False


def test_unregistered_solution_is_unverified_and_does_not_update_best_known(monkeypatch):
    """What: an unregistered but scoreable solution cannot change best-known state."""
    core_type = "regression_unregistered_evaluation"
    instance_id = "regression-unverified-best-known"
    monkeypatch.delitem(CHECKERS_DETAILED, core_type, raising=False)
    registry = BestKnownRegistry()
    registry.register(instance_id, 100.0)

    result = evaluate_algorithm_v3(
        code="def solve(instance):\n    return {'items': [1], 'cost': 12}",
        instance={"items": [1]},
        core_type=core_type,
        instance_id=instance_id,
        registry=registry,
        reference_value=100.0,
        reference_solution={"items": [1], "cost": 100.0},
        objective_text="Minimize cost",
        timeout=5,
    )

    assert result["status"] == "unverified"
    assert result["feasibility_verified"] is False
    assert result["best_known"] == 100.0
    assert registry.get(instance_id) == 100.0


def test_unverified_solution_does_not_seed_reference_free_baseline(monkeypatch):
    """What: an unverified solution cannot anchor later reference-free rewards."""
    core_type = "regression_unverified_baseline"
    instance_id = "regression-unverified-baseline"
    monkeypatch.delitem(CHECKERS_DETAILED, core_type, raising=False)
    registry = BestKnownRegistry()

    result = evaluate_algorithm_v3(
        code="def solve(instance):\n    return {'items': [1], 'cost': 12}",
        instance={"items": [1]},
        core_type=core_type,
        instance_id=instance_id,
        registry=registry,
        reference_value=100.0,
        reference_solution={"items": [1], "cost": 100.0},
        objective_text="Minimize cost",
        use_reference=False,
        timeout=5,
    )

    assert result["status"] == "unverified"
    assert registry.get_baseline(instance_id) is None

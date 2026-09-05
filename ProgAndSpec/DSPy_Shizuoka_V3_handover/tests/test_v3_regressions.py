"""Regression tests for V3 requirement construction, scoring, and runtime loading."""

import ast
import json
from pathlib import Path

import pytest

pytest.importorskip("dspy")

from src.best_known import BestKnownRegistry
from src.data_loader import convert_to_dspy_example
from src.metrics_v3 import (
    evaluate_algorithm_v3,
    generic_ref_guided_cost,
    generic_reference_free_cost,
)
from src.modules import AlgorithmGenerator
from src.signatures import GenerateOptimizationAlgorithm
from src.utils.safe_exec import safe_run
from src.utils.scorer import SCORERS
from train_gepa_v3 import bootstrap_parse_codes_v3


@pytest.fixture
def reference_record():
    """Return a record whose reference values are distinguishable from instance data."""
    return {
        "id": 7,
        "name": "Reference-shape regression",
        "domain": "optimization",
        "math_type": "maximization",
        "difficulty": "easy",
        "description": "Select items under the stated constraints.",
        "requirements": {
            "objective": "Maximize total value",
            "constraints": ["All selected items must be valid."],
        },
        "instance": {
            "items": [{"id": "instance-item", "value": 4}],
            "capacity": 10,
        },
        "reference_value": 987654.321,
        "reference_solution": {
            "items": [314159, 271828],
            "max_value": 115,
        },
    }


def test_requirement_without_reference_contains_shape_but_no_reference_values(
    reference_record,
):
    """What: reference-free requirements retain reference keys/types without leaking values."""
    example = convert_to_dspy_example(reference_record, use_reference=False)
    requirement = example["requirement"]

    assert "items" in requirement
    assert "max_value" in requirement
    assert "list" in requirement.lower()
    assert "int" in requirement.lower()
    assert "## Reference" not in requirement
    assert "reference solution" not in requirement.lower()
    assert "list[2]" not in requirement
    assert "reference_value" not in requirement
    assert "987654.321" not in requirement
    assert "314159" not in requirement
    assert "271828" not in requirement
    assert "115" not in requirement


def test_base_generation_signature_uses_reference_neutral_schema_wording():
    """What: fresh reference-free training does not receive reference wording via the signature."""
    instructions = GenerateOptimizationAlgorithm.instructions.lower()

    assert "required return schema" in instructions
    assert "reference solution" not in instructions


def test_requirement_with_reference_contains_raw_instance_and_reference_shape(reference_record):
    """What: reference-guided requirements include raw instance JSON and reference structure."""
    example = convert_to_dspy_example(reference_record, use_reference=True)
    requirement = example["requirement"]
    raw_instance = json.dumps(reference_record["instance"], ensure_ascii=False, indent=2)

    assert raw_instance in requirement
    assert "Reference Solution Structure" in requirement
    assert "items" in requirement
    assert "max_value" in requirement


def test_missing_maximization_objective_is_invalid(monkeypatch):
    """What: a non-empty solution missing max_value is rejected as invalid."""
    reference_solution = {"items": ["reference-item"], "max_value": 115}
    solution = {"items": ["candidate-item"], "total_value": 115}

    assert (
        generic_ref_guided_cost(
            solution,
            reference_solution,
            objective_text="Maximize total value",
        )
        is None
    )

    core_type = "regression_maximization"
    monkeypatch.setitem(SCORERS, core_type, lambda _instance, _solution: None)
    result = evaluate_algorithm_v3(
        code="def solve(instance):\n    return {'items': ['candidate-item'], 'total_value': 115}",
        instance={"items": ["instance-item"]},
        core_type=core_type,
        instance_id="regression-missing-max-value",
        registry=BestKnownRegistry(),
        reference_value=115,
        reference_solution=reference_solution,
        objective_text="Maximize total value",
        timeout=5,
    )

    assert result["status"] == "invalid_solution"
    assert result["status"] not in {"beat_reference", "exact_match"}


def test_maximization_cost_is_normalized_against_reference():
    """What: max_value=115 normalizes to 115, while max_value=230 is better."""
    reference_solution = {"items": ["reference-item"], "max_value": 115}

    equal_cost = generic_ref_guided_cost(
        {"items": ["candidate-item"], "max_value": 115},
        reference_solution,
        objective_text="Maximize total value",
    )
    better_cost = generic_ref_guided_cost(
        {"items": ["candidate-item"], "max_value": 230},
        reference_solution,
        objective_text="Maximize total value",
    )

    assert equal_cost == pytest.approx(115)
    assert better_cost == pytest.approx(57.5)
    assert better_cost < equal_cost


def test_reference_free_maximization_cost_does_not_depend_on_reference():
    """What: reference-free objective normalization reads only the candidate solution."""
    candidate = {"optimal_order_quantity": 40, "expected_profit": 100}

    assert generic_reference_free_cost(candidate, "Maximize expected profit") == pytest.approx(0.01)


def test_bootstrap_parse_code_is_valid_python():
    """What: bootstrapped helper functions and assignments form one valid Python block."""
    example = pytest.importorskip("dspy").Example(
        core_type="regression_parse",
        instance={"jobs": [{"id": 1}], "capacity": 2},
    )

    parse_code = bootstrap_parse_codes_v3([example])["regression_parse"]

    ast.parse(parse_code)


def test_phase_e_compiled_program_loads_into_algorithm_generator():
    """What: the saved Phase E program loads and restores exactly two demonstrations."""
    program_path = Path(__file__).parents[1] / "compiled_program_v3_gepa_phaseE.json"
    program = AlgorithmGenerator()

    program.load(str(program_path))

    assert len(program.generate.predict.demos) == 2


def test_safe_run_executes_minimal_solve_in_spawn_child():
    """What: safe_run returns a child-process result from the minimal solve contract."""
    ok, result = safe_run(
        "def solve(instance):\n    return {'answer': instance.get('answer')}",
        {"answer": 42},
        timeout=5,
    )

    assert ok is True
    assert result == {"answer": 42}


def test_safe_run_rejects_dunder_import_escape():
    """What: generated code cannot recover unrestricted import through module internals."""
    ok, result = safe_run(
        'def solve(instance):\n    return json.__dict__["__builtins__"]["__import__"]("os")',
        {},
        timeout=5,
    )

    assert ok is False
    assert result.startswith(("Banned dunder attribute", "Banned subscript key"))


def test_safe_run_rejects_attrgetter_import_escape():
    """What: dynamic attribute lookup cannot reconstruct module internals."""
    code = """\
def solve(instance):
    module_dict = operator.attrgetter("__dict__")(json)
    builtins_dict = module_dict["__" + "builtins__"]
    importer = builtins_dict["__" + "import__"]
    return importer("os").getcwd()
"""

    ok, result = safe_run(code, {}, timeout=5)

    assert ok is False
    assert result.startswith(("Banned attribute", "Banned dunder attribute"))


def test_safe_run_rejects_numpy_ctypes_escape():
    """What: allowed numeric modules cannot expose native dynamic loaders."""
    code = "def solve(instance):\n    return numpy.ctypeslib.ctypes.CDLL(None)"

    ok, result = safe_run(code, {}, timeout=5)

    assert ok is False
    assert result.startswith("Banned attribute")


def test_safe_run_supports_from_import_of_allowed_submodules():
    """What: `from scipy.optimize import linprog` and ortools' cp_model import work in the sandbox."""
    code = (
        "from scipy.optimize import linprog\n"
        "from ortools.sat.python import cp_model\n"
        "def solve(instance):\n"
        "    return {'ok': linprog is not None and cp_model.CpModel is not None}\n"
    )
    ok, result = safe_run(code, {}, timeout=30)
    assert ok, result
    assert result == {"ok": True}


def test_safe_run_returns_plain_python_types_for_numpy_results():
    """What: numpy scalars and arrays in solve() output come back as plain floats and lists."""
    code = (
        "import numpy as np\n"
        "def solve(instance):\n"
        "    return {'cost': np.float64(1.5), 'x': np.array([1, 2]), 'n': np.int64(3)}\n"
    )
    ok, result = safe_run(code, {}, timeout=30)
    assert ok, result
    assert result == {"cost": 1.5, "x": [1, 2], "n": 3}
    assert type(result["cost"]) is float and type(result["n"]) is int


def test_safe_run_provides_iteration_and_numeric_builtins():
    """What: next/iter/divmod and friends are available to generated code."""
    code = (
        "def solve(instance):\n"
        "    it = iter([3, 4])\n"
        "    q, r = divmod(7, 2)\n"
        "    return {'first': next(it), 'q': q, 'r': r, 'fs': sorted(frozenset({1, 1, 2}))}\n"
    )
    ok, result = safe_run(code, {}, timeout=30)
    assert ok, result
    assert result == {"first": 3, "q": 3, "r": 1, "fs": [1, 2]}


def test_safe_run_allows_public_getattr_but_blocks_private_names():
    """What: getattr on public attributes works; dunder names built at run time are refused."""
    ok, result = safe_run(
        "from scipy.optimize import linprog\n"
        "def solve(instance):\n"
        "    res = linprog(c=[1.0], bounds=[(0, 1)], method='highs')\n"
        "    return {'ok': bool(getattr(res, 'success', False)), 'has': hasattr(res, 'x')}\n",
        {},
        timeout=30,
    )
    assert ok, result
    assert result == {"ok": True, "has": True}
    ok, result = safe_run(
        "def solve(instance):\n    return getattr((1).__class__, '__cl' + 'ass__')\n",
        {},
        timeout=10,
    )
    assert not ok
    ok, result = safe_run(
        "def solve(instance):\n    return getattr(1, '__cl' + 'ass__')\n", {}, timeout=10
    )
    assert not ok and "not allowed" in str(result)

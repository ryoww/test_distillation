"""効果分解スクリプトの振る舞いを検証する。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import diagnose_effects as diag

BASE_DIR = Path(__file__).resolve().parents[1]
PROBLEM_DIR = BASE_DIR / "data" / "problems"
PROGRAM = BASE_DIR / "compiled_program_v3_gepa_phaseE.json"


def test_historical_boundary_reproduces_the_tail_slice():
    """過去の結果を診断するには、修正前の分割規則が要る。"""
    train27, val13 = diag.historical_train_val(PROBLEM_DIR)
    assert len(train27) == 27
    assert len(val13) == 13
    assert set(train27) & set(val13) == set()
    # 当時は instance_id でソート済みの train を末尾から切っていた。
    assert val13 == sorted(val13)
    assert min(val13) > max(train27)


def test_demo_problems_are_identified_from_the_compiled_program():
    demos = diag.find_demo_ids(PROGRAM, PROBLEM_DIR)
    assert demos == ["prob_001", "prob_021"]


def test_missing_program_yields_no_demos(tmp_path):
    assert diag.find_demo_ids(tmp_path / "absent.json", PROBLEM_DIR) == []


def test_bootstrap_interval_brackets_the_mean():
    deltas = [0.4, -0.1, 0.9, 0.2, 0.5, -0.3, 0.7, 0.1]
    mean, low, high = diag.paired_bootstrap(deltas, rounds=2000, seed=42)
    assert mean == pytest.approx(sum(deltas) / len(deltas))
    assert low < mean < high


def test_bootstrap_is_deterministic_for_a_seed():
    deltas = [0.3, -0.2, 0.8, 0.1]
    assert diag.paired_bootstrap(deltas, 500, 7) == diag.paired_bootstrap(deltas, 500, 7)


def test_constant_deltas_give_a_zero_width_interval():
    mean, low, high = diag.paired_bootstrap([0.25] * 12, rounds=500, seed=1)
    assert mean == pytest.approx(0.25)
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.25)


def test_deltas_pair_by_instance_and_skip_unmatched():
    scores = {
        "before__m": {"prob_001": 1.0, "prob_002": 0.5, "prob_003": 0.0},
        "after__m": {"prob_001": 1.5, "prob_002": 0.5},
    }
    assert diag.deltas_for(scores, "m", ["prob_001", "prob_002", "prob_003"]) == [0.5, 0.0]


def test_deltas_for_unknown_model_is_empty():
    assert diag.deltas_for({}, "nope", ["prob_001"]) == []

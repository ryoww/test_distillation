"""保存済みコードの再採点スクリプトの振る舞いを検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import rescore_with_checkers as rescore

BASE_DIR = Path(__file__).resolve().parents[1]
PROBLEM_DIR = BASE_DIR / "data" / "problems"


def test_condition_label_strips_the_shard_suffix():
    assert rescore._condition_label("before__qwen3_6_27b__shard01of02") == "before__qwen3_6_27b"
    assert rescore._condition_label("after__qwen3_8_27b") == "after__qwen3_8_27b"


def test_discover_shard_dirs_skips_folders_without_results(tmp_path):
    good = tmp_path / "before__m__shard01of02"
    good.mkdir()
    (good / rescore.RESULT_FILENAME).write_text("{}", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    assert rescore._discover_shard_dirs(tmp_path) == [good]


def test_summarise_reports_old_and_new_side_by_side():
    records = [
        {
            "instance_id": "prob_001",
            "score": 1.5,
            "status": "exact_match",
            "old_score": 1.0,
            "old_status": "unverified",
        },
        {
            "instance_id": "prob_002",
            "score": 0.0,
            "status": "infeasible",
            "old_score": 1.0,
            "old_status": "unverified",
        },
    ]
    summary = rescore._summarise(records, {"pair": ["prob_001", "prob_002"]})["pair"]
    assert summary["total_count"] == 2
    assert summary["mean_score"] == pytest.approx(0.75)
    assert summary["old_mean_score"] == pytest.approx(1.0)
    assert summary["status_counts"] == {"exact_match": 1, "infeasible": 1}
    assert summary["old_status_counts"] == {"unverified": 2}


def test_summarise_ignores_ids_outside_the_subset():
    records = [{"instance_id": "prob_001", "score": 2.0, "status": "new_best"}]
    summary = rescore._summarise(records, {"only": ["prob_001", "prob_999"]})["only"]
    assert summary["total_count"] == 1


def _write_shard(tmp_path: Path, rows: list[dict]) -> Path:
    shard = tmp_path / "before__fake__shard01of01"
    shard.mkdir(parents=True)
    (shard / rescore.RESULT_FILENAME).write_text(
        json.dumps({"train": {"results": []}, "test": {"results": rows}}), encoding="utf-8"
    )
    return shard


def test_rescore_shard_replays_stored_code_and_upgrades_unverified(tmp_path):
    """チェッカーが付いた core_type は unverified から昇格する。"""
    record = json.loads((PROBLEM_DIR / "prob_055.json").read_text(encoding="utf-8"))
    reference = record["reference_solution"]
    code = (
        "def solve(instance):\n"
        f"    return {{'max_value': {reference['max_value']}, "
        f"'chosen_items': {reference['chosen_items']}}}\n"
    )
    shard = _write_shard(
        tmp_path,
        [
            {
                "instance_id": "prob_055",
                "status": "unverified",
                "score": 1.0,
                "cost": None,
                "code": code,
            }
        ],
    )
    examples = rescore._load_examples(PROBLEM_DIR)

    outcome = rescore._rescore_shard(shard, examples, timeout=30.0)

    assert len(outcome.records) == 1
    row = outcome.records[0]
    assert row["rescored"] is True
    assert row["old_status"] == "unverified"
    assert row["status"] != "unverified"
    assert row["feasibility_verified"] is True
    assert row["score"] >= row["old_score"]


def test_rescore_shard_keeps_records_without_code(tmp_path):
    shard = _write_shard(
        tmp_path,
        [{"instance_id": "prob_004", "status": "gen_error", "score": -0.5, "error": "boom"}],
    )
    examples = rescore._load_examples(PROBLEM_DIR)

    outcome = rescore._rescore_shard(shard, examples, timeout=30.0)

    assert outcome.records == [
        {
            "instance_id": "prob_004",
            "status": "gen_error",
            "score": -0.5,
            "error": "boom",
            "rescored": False,
        }
    ]


def test_rescore_shard_flags_a_cost_that_does_not_replay(tmp_path):
    """保存コストと再実行コストの食い違いは非決定の印として残す。"""
    record = json.loads((PROBLEM_DIR / "prob_055.json").read_text(encoding="utf-8"))
    reference = record["reference_solution"]
    code = (
        "def solve(instance):\n"
        f"    return {{'max_value': {reference['max_value']}, "
        f"'chosen_items': {reference['chosen_items']}}}\n"
    )
    shard = _write_shard(
        tmp_path,
        [
            {
                "instance_id": "prob_055",
                "status": "unverified",
                "score": 1.0,
                "cost": -12345.0,
                "code": code,
            }
        ],
    )
    examples = rescore._load_examples(PROBLEM_DIR)

    outcome = rescore._rescore_shard(shard, examples, timeout=30.0)

    assert len(outcome.cost_mismatches) == 1
    assert "prob_055" in outcome.cost_mismatches[0]

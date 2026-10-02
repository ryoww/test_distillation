"""shop_cutting 群（fjsp / fjsp_setup / cutting_1d / cutting_2d）の instance 生成器が、元 instance と
同じ形・同じ分布で決定的に新しい instance を作ることを固定する。教師コードの実行は重いので含めない。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

import pytest

from src.datagen.pipeline import shape_signature
from src.hardgen import GENERATORS
from src.utils.hard import find_kind

PROBLEM_DIR = Path(__file__).resolve().parents[1] / "data" / "problems_hard"
CASES = [
    ("fjsp", 305), ("fjsp", 315), ("fjsp_setup", 322),
    ("cutting_1d", 307), ("cutting_1d", 317), ("cutting_2d", 324),
]
FJSP_CASES = [("fjsp", 305), ("fjsp", 315), ("fjsp_setup", 322)]


def _base(problem_id: int) -> dict:
    record = json.loads((PROBLEM_DIR / f"prob_{problem_id}.json").read_text(encoding="utf-8"))
    return record["instance"]


def _generate(kind: str, problem_id: int, seed: str = "seed-a") -> dict:
    return GENERATORS[kind](random.Random(seed), _base(problem_id))


def _ops(instance: dict) -> list[dict]:
    return [op for job in instance["jobs"] for op in job["operations"]]


def _work_lower_bound(instance: dict) -> float:
    """最短加工時間の合計 / 機械数（メイクスパンの下界で、きつさの指標）。"""
    return sum(min(op["machine_options"].values()) for op in _ops(instance)) / len(
        instance["machines"]
    )


@pytest.mark.parametrize(("kind", "problem_id"), CASES)
def test_generated_instance_keeps_base_shape(kind: str, problem_id: int) -> None:
    base = _base(problem_id)
    for seed in ("seed-a", "seed-b", "seed-c"):
        generated = _generate(kind, problem_id, seed)
        assert shape_signature(generated) == shape_signature(base)


@pytest.mark.parametrize(("kind", "problem_id"), CASES)
def test_generated_instance_is_detected_as_same_kind(kind: str, problem_id: int) -> None:
    detected = find_kind(_generate(kind, problem_id))
    assert detected is not None and detected.name == kind


@pytest.mark.parametrize(("kind", "problem_id"), CASES)
def test_generation_is_deterministic_per_seed(kind: str, problem_id: int) -> None:
    first = _generate(kind, problem_id, "seed-a")
    again = _generate(kind, problem_id, "seed-a")
    other = _generate(kind, problem_id, "seed-b")
    assert first == again
    assert first != other
    assert first != _base(problem_id)


# ---------------------------------------------------------------- FJSP
@pytest.mark.parametrize(("kind", "problem_id"), FJSP_CASES)
def test_fjsp_keeps_total_operations_machines_and_option_counts(
    kind: str, problem_id: int
) -> None:
    base = _base(problem_id)
    generated = _generate(kind, problem_id)
    assert generated["machines"] == base["machines"]
    assert len(_ops(generated)) == len(_ops(base))
    machine_ids = {str(m["id"]) for m in base["machines"]}
    for base_job, job in zip(base["jobs"], generated["jobs"]):
        assert (job["id"], job["name"]) == (base_job["id"], base_job["name"])
        assert [op["op_index"] for op in job["operations"]] == list(
            range(1, len(job["operations"]) + 1)
        )
        # 工程数は base の範囲、選択肢数の集合は base の同じジョブと一致（形の署名が見る）。
        ops_per_job = {len(j["operations"]) for j in base["jobs"]}
        assert min(ops_per_job) <= len(job["operations"]) <= max(ops_per_job)
        assert {len(op["machine_options"]) for op in job["operations"]} == {
            len(op["machine_options"]) for op in base_job["operations"]
        }
        for op in job["operations"]:
            assert set(op["machine_options"]) <= machine_ids
            assert list(op["machine_options"]) == sorted(op["machine_options"], key=int)


@pytest.mark.parametrize(("kind", "problem_id"), FJSP_CASES)
def test_fjsp_durations_and_releases_follow_base_distribution(
    kind: str, problem_id: int
) -> None:
    base = _base(problem_id)
    generated = _generate(kind, problem_id)
    base_durations = [d for op in _ops(base) for d in op["machine_options"].values()]
    durations = [d for op in _ops(generated) for d in op["machine_options"].values()]
    assert min(base_durations) <= min(durations) and max(durations) <= max(base_durations)
    assert mean(durations) == pytest.approx(mean(base_durations), rel=0.05)
    # release_time は base の値の経験分布から引く（305 は {0, 10, 25, 50} の離散値）。
    base_releases = {job["release_time"] for job in base["jobs"]}
    assert {job["release_time"] for job in generated["jobs"]} <= base_releases
    # きつさ: 総最短加工時間 / 機械数 が base の近傍。
    assert _work_lower_bound(generated) == pytest.approx(_work_lower_bound(base), rel=0.1)


def test_fjsp_setup_redraws_setups_and_breakdowns_within_base_ranges() -> None:
    base = _base(322)
    generated = _generate("fjsp_setup", 322)
    assert generated["note"] == base["note"]
    assert generated["setups"].keys() == base["setups"].keys()
    base_setups = [
        v for table in base["setups"].values()
        for a, row in table.items() for b, v in row.items() if a != b
    ]
    for m, table in generated["setups"].items():
        assert table.keys() == base["setups"][m].keys()
        for a, row in table.items():
            assert row[a] == 0
            assert all(min(base_setups) <= v <= max(base_setups) for v in row.values())
    assert generated["setups"] != base["setups"]
    base_rows = [br for rows in base["breakdowns"].values() for br in rows]
    rows = [br for rows in generated["breakdowns"].values() for br in rows]
    assert generated["breakdowns"].keys() == base["breakdowns"].keys()
    assert len(rows) == len(base_rows)
    for key in ("start", "duration"):
        lo, hi = min(br[key] for br in base_rows), max(br[key] for br in base_rows)
        assert all(lo <= br[key] <= hi for br in rows)


# ---------------------------------------------------------------- カッティング
@pytest.mark.parametrize("problem_id", [307, 317])
def test_cutting_1d_keeps_stocks_and_total_demand(problem_id: int) -> None:
    base = _base(problem_id)
    generated = _generate("cutting_1d", problem_id)
    assert generated["stocks"] == base["stocks"]
    assert generated["max_distinct_patterns"] == base["max_distinct_patterns"]
    assert [i["id"] for i in generated["items"]] == [i["id"] for i in base["items"]]
    assert sum(i["demand"] for i in generated["items"]) == sum(i["demand"] for i in base["items"])
    widths = [i["width"] for i in generated["items"]]
    base_widths = [i["width"] for i in base["items"]]
    assert len(set(widths)) == len(widths)
    assert min(base_widths) <= min(widths) and max(widths) <= max(base_widths)
    demands = [i["demand"] for i in generated["items"]]
    base_demands = [i["demand"] for i in base["items"]]
    assert min(base_demands) <= min(demands) and max(demands) <= max(base_demands)
    # きつさ: 必要総長 / 最長原材 が base の近傍。
    longest = max(s["length"] for s in base["stocks"])
    area = sum(i["width"] * i["demand"] for i in generated["items"]) / longest
    base_area = sum(i["width"] * i["demand"] for i in base["items"]) / longest
    assert area == pytest.approx(base_area, rel=0.2)


def test_cutting_2d_keeps_plates_and_draws_sizes_from_base_grid() -> None:
    base = _base(324)
    generated = _generate("cutting_2d", 324)
    for key in ("plates", "max_distinct_patterns", "min_lot", "note"):
        assert generated[key] == base[key]
    assert sum(i["demand"] for i in generated["items"]) == sum(i["demand"] for i in base["items"])
    widths, heights = {i["width"] for i in base["items"]}, {i["height"] for i in base["items"]}
    demands = [i["demand"] for i in base["items"]]
    for item in generated["items"]:
        assert item["width"] in widths and item["height"] in heights
        assert min(demands) <= item["demand"] <= max(demands)
    # きつさ: 必要総面積 / 最小板面積 が base の近傍。
    plate = base["plates"][0]["width"] * base["plates"][0]["height"]
    area = sum(i["width"] * i["height"] * i["demand"] for i in generated["items"]) / plate
    base_area = sum(i["width"] * i["height"] * i["demand"] for i in base["items"]) / plate
    assert area == pytest.approx(base_area, rel=0.15)

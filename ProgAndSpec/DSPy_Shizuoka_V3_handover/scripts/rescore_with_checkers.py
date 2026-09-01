#!/usr/bin/env python3
"""保存済みの生成コードを再実行し、現在のチェッカー構成で採点し直す。

feasibility チェッカーを追加するとスコアの意味が変わるため、過去の実行結果は
そのままでは新しい結果と比較できない。評価結果 JSON には各問題の生成コードが
保存されているので、LLM を呼び直さずに再採点できる。

各 shard ディレクトリを1つの評価単位として扱い、元実行と同じ順序で
BestKnownRegistry を再生する。再実行したコストが保存値と食い違う場合は
非決定な解法とみなして印を付ける。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import best_known as _best_known
from src.data_loader import convert_to_dspy_example, load_v3_data
from src.metrics_v3 import evaluate_algorithm_v3

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"
DEFAULT_DATA_DIR = BASE_DIR / "data" / "problems"
SUBSET_ORDER = ("untouched40", "legacy_test20", "train40", "all100")


@dataclass
class ShardOutcome:
    """1 shard ぶんの再採点結果。"""

    label: str
    records: list[dict] = field(default_factory=list)
    cost_mismatches: list[str] = field(default_factory=list)


def _condition_label(shard_label: str) -> str:
    """`before__qwen3_6_27b__shard01of02` から条件名を取り出す。"""
    marker = "__shard"
    return shard_label.split(marker)[0] if marker in shard_label else shard_label


def _load_examples(data_dir: Path) -> dict[str, dict]:
    return {
        example["instance_id"]: example
        for example in (
            convert_to_dspy_example(record, use_reference=True)
            for record in load_v3_data(str(data_dir))
        )
    }


def _rescore_shard(
    shard_dir: Path,
    examples: dict[str, dict],
    timeout: float,
) -> ShardOutcome:
    payload = json.loads((shard_dir / RESULT_FILENAME).read_text(encoding="utf-8"))
    stored = payload["test"]["results"]
    outcome = ShardOutcome(label=shard_dir.name)

    # 元実行と同じく、この shard の問題だけを参照値で seed したレジストリを使う。
    registry = _best_known.BestKnownRegistry()
    for row in stored:
        example = examples.get(row["instance_id"])
        if example and example.get("reference_value") is not None:
            registry.register(row["instance_id"], example["reference_value"])

    for row in stored:
        instance_id = row["instance_id"]
        example = examples.get(instance_id)
        code = row.get("code")
        if example is None or not code:
            # 生成段階で落ちた記録にはコードがないため、そのまま引き継ぐ。
            outcome.records.append({**row, "rescored": False})
            continue

        result = evaluate_algorithm_v3(
            code=code,
            instance=example["instance"],
            core_type=example["core_type"],
            instance_id=instance_id,
            registry=registry,
            timeout=timeout,
            reference_value=example.get("reference_value"),
            reference_solution=example.get("reference_solution", {}),
            objective_text=example.get("objective", ""),
            use_reference=True,
        )
        old_cost = row.get("cost")
        new_cost = result.get("cost")
        if old_cost is not None and new_cost is not None and abs(old_cost - new_cost) > 1e-6:
            outcome.cost_mismatches.append(
                f"{instance_id}: stored {old_cost:g} -> replayed {new_cost:g}"
            )
        outcome.records.append(
            {
                "instance_id": instance_id,
                "name": example.get("name", row.get("name", "")),
                "core_type": example["core_type"],
                "rescored": True,
                "old_status": row.get("status"),
                "old_score": row.get("score"),
                "old_cost": old_cost,
                "status": result["status"],
                "score": result["score"],
                "cost": new_cost,
                "detail": result.get("detail", ""),
                "feasibility_verified": result.get("feasibility_verified"),
            }
        )
    return outcome


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarise(records: list[dict], subsets: dict[str, list[str]]) -> dict:
    by_id = {row["instance_id"]: row for row in records}
    summary: dict[str, Any] = {}
    for name, ids in subsets.items():
        present = [by_id[i] for i in ids if i in by_id]
        if not present:
            continue
        statuses: dict[str, int] = {}
        for row in present:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        old_statuses: dict[str, int] = {}
        for row in present:
            key = row.get("old_status", row["status"])
            old_statuses[key] = old_statuses.get(key, 0) + 1
        summary[name] = {
            "total_count": len(present),
            "mean_score": _mean([row["score"] for row in present]),
            "old_mean_score": _mean([row.get("old_score", row["score"]) for row in present]),
            "status_counts": statuses,
            "old_status_counts": old_statuses,
        }
    return summary


def _discover_shard_dirs(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / RESULT_FILENAME).exists())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay stored solution code and rescore it with the current checkers."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories holding per-condition shard folders.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--subsets-from",
        type=Path,
        help="factorial_comparison.json to take subset instance ids from.",
    )
    parser.add_argument("--output", type=Path, help="Where to write the rescored JSON.")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    examples = _load_examples(args.data_dir)

    subsets: dict[str, list[str]]
    if args.subsets_from:
        manifest = json.loads(args.subsets_from.read_text(encoding="utf-8"))
        subsets = manifest["benchmark"]["subset_instance_ids"]
    else:
        subsets = {"all": sorted(examples)}

    by_condition: dict[str, list[dict]] = {}
    mismatches: list[str] = []
    for run_dir in args.runs:
        for shard_dir in _discover_shard_dirs(run_dir):
            outcome = _rescore_shard(shard_dir, examples, args.timeout)
            condition = _condition_label(outcome.label)
            by_condition.setdefault(condition, []).extend(outcome.records)
            mismatches.extend(f"{outcome.label} {line}" for line in outcome.cost_mismatches)

    report = {
        "conditions": {
            condition: _summarise(records, subsets)
            for condition, records in sorted(by_condition.items())
        },
        "cost_mismatches": mismatches,
        "records": {condition: rows for condition, rows in sorted(by_condition.items())},
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rescored report: {args.output}")

    names = [s for s in SUBSET_ORDER if s in subsets]
    header = f"{'Condition':26}" + "".join(f"{name:>22}" for name in names)
    print("\nRescored vs original mean score")
    print(header)
    for condition, summary in report["conditions"].items():
        cells = ""
        for name in names:
            entry = summary.get(name)
            cells += (
                f"{entry['old_mean_score']:>10.3f}->{entry['mean_score']:<10.3f}"
                if entry
                else f"{'-':>22}"
            )
        print(f"{condition:26}{cells}")
    if mismatches:
        print(f"\nNon-deterministic replays: {len(mismatches)}")
        for line in mismatches[:10]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

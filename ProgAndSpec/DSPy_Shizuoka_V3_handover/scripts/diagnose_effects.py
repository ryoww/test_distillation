#!/usr/bin/env python3
"""再採点済みの結果から、プロンプト効果を分解して信頼区間つきで出す。

平均スコアの差だけでは、GEPA が反省に使った問題・候補選択に使った問題・
一度も見ていない問題が混ざったままになる。この3者は露出の度合いが違うので、
分けないと「訓練例の記憶なのか汎化なのか」を判別できない。

出力するもの:

- 露出段階別（train27 / val13 / legacy20 / untouched40 / all100）の効果
- paired bootstrap による 95% 信頼区間
- Phase E の demo として焼き込まれた問題を除いた場合の効果

demo は compiled program から requirement 照合で特定する。GPU も LLM も使わない。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_and_split_stratified, load_v3_data

DEFAULT_DATA_DIR = BASE_DIR / "data" / "problems"
DEFAULT_PROGRAM = BASE_DIR / "compiled_program_v3_gepa_phaseE.json"
MODELS = ("qwen3_6_27b", "qwen3_8_27b")


def historical_train_val(data_dir: Path, n_val: int | None = None) -> tuple[list[str], list[str]]:
    """Phase E を最適化したときの train/val 境界を再現する。

    Why not reuse split_train_val_stratified: 現在の分割は層化へ修正済みだが、
    保存済みの結果は修正前の「ソート済み train を末尾で切る」規則で最適化された。
    過去の結果を診断するには当時の境界が要る。
    """
    train, _ = load_and_split_stratified(
        str(data_dir), n_train=40, n_test=20, seed=42, use_reference=True
    )
    ids = [str(example["instance_id"]) for example in train]
    size = n_val if n_val is not None else max(3, len(ids) // 3)
    return ids[:-size], ids[-size:]


def find_demo_ids(program_path: Path, data_dir: Path) -> list[str]:
    """compiled program に焼き込まれた demo の instance_id を特定する。"""
    if not program_path.exists():
        return []
    program = json.loads(program_path.read_text(encoding="utf-8"))
    names = {
        f"prob_{record['id']:03d}": record.get("name", "") for record in load_v3_data(str(data_dir))
    }

    def walk(node: Any):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "demos" and isinstance(value, list):
                    yield from (e for e in value if isinstance(e, dict))
                elif isinstance(value, (dict, list)):
                    yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    found: list[str] = []
    for demo in walk(program):
        requirement = str(demo.get("requirement", ""))
        for instance_id, name in names.items():
            if name and name in requirement and instance_id not in found:
                found.append(instance_id)
    return sorted(found)


def paired_bootstrap(
    deltas: list[float],
    rounds: int,
    seed: int,
) -> tuple[float, float, float]:
    """対応のあるブートストラップで平均差の 95% 区間を出す。"""
    mean = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    size = len(deltas)
    draws = sorted(sum(rng.choice(deltas) for _ in range(size)) / size for _ in range(rounds))
    return mean, draws[int(0.025 * rounds)], draws[int(0.975 * rounds)]


def deltas_for(scores: dict[str, dict[str, float]], model: str, ids: list[str]) -> list[float]:
    before = scores.get(f"before__{model}", {})
    after = scores.get(f"after__{model}", {})
    return [after[i] - before[i] for i in ids if i in before and i in after]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Break the prompt effect down by how much GEPA saw each problem."
    )
    parser.add_argument("rescored", type=Path, help="rescore_with_checkers.py の出力 JSON")
    parser.add_argument("--comparison", type=Path, help="subset_instance_ids を持つ factorial JSON")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = json.loads(args.rescored.read_text(encoding="utf-8"))
    scores = {
        condition: {row["instance_id"]: row["score"] for row in rows}
        for condition, rows in report["records"].items()
    }

    train27, val13 = historical_train_val(args.data_dir)
    groups: dict[str, list[str]] = {"train27 (反省)": train27, "val13 (候補選択)": val13}
    if args.comparison:
        subsets = json.loads(args.comparison.read_text(encoding="utf-8"))["benchmark"][
            "subset_instance_ids"
        ]
        groups["legacy20"] = list(subsets["legacy_test20"])
        groups["untouched40"] = list(subsets["untouched40"])
        groups["all100"] = list(subsets["all100"])

    demos = find_demo_ids(args.program, args.data_dir)
    print(f"Phase E の demo: {demos or '(なし)'}")
    if args.comparison and demos:
        for demo in demos:
            where = [name for name, ids in groups.items() if demo in ids and name != "all100"]
            print(f"  {demo} は {where} に含まれる")

    out: dict[str, Any] = {"demos": demos, "groups": {}}
    header = f"\n{'分割':22}{'n':>4}  " + "".join(f"{m:<30}" for m in MODELS)
    print(header)
    for name, ids in groups.items():
        row = f"{name:22}{len(ids):>4}  "
        entry: dict[str, Any] = {"n": len(ids)}
        for model in MODELS:
            deltas = deltas_for(scores, model, ids)
            if not deltas:
                row += f"{'-':<30}"
                continue
            mean, low, high = paired_bootstrap(deltas, args.bootstrap, args.seed)
            significant = not (low <= 0 <= high)
            row += f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]{'*' if significant else ' '}   "
            entry[model] = {
                "effect": mean,
                "ci_low": low,
                "ci_high": high,
                "excludes_zero": significant,
            }
        kept = [i for i in ids if i not in demos]
        if len(kept) != len(ids):
            entry["without_demos"] = {
                model: (
                    sum(deltas_for(scores, model, kept)) / len(deltas_for(scores, model, kept))
                    if deltas_for(scores, model, kept)
                    else None
                )
                for model in MODELS
            }
        out["groups"][name] = entry
        print(row)
    print("\n* = 95% 信頼区間が 0 を含まない")

    for name, entry in out["groups"].items():
        without = entry.get("without_demos")
        if without:
            pairs = ", ".join(f"{m}: {v:+.3f}" for m, v in without.items() if v is not None)
            print(f"demo 除外後 {name}: {pairs}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n診断 JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

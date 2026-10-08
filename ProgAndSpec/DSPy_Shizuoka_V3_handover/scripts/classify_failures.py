#!/usr/bin/env python3
"""未正解の問題を、保存済みの判定ログ（status、detail、出力 token 数）から失敗原因に分ける。

10 月 7 日の会議の決定事項「未正解問題の生成コード・ログを見て失敗原因を分類する」に当たる。区分は議事録の 5 つ
（難しさ / 初歩的なコード生成の失敗 / 推論の堂々巡り・出力上限 / 定式化の誤り / 制約違反）に、実行時間超過を足したもの。
コードは再実行しない（評価時の判定をそのまま使う）。

  uv run python scripts/classify_failures.py \\
      --run fft12b=outputs/prompt_model_comparisons/gemma4-ft-20261004-test/sft_gemma4_12b_fft_fp32__shard01of01 \\
      --problem-dir data/problems_hard_gen/test --supported-kinds-file prompts/supported_kinds/sft_merged_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.summarize_solver_runs import gap_of, is_correct, load

# 議事録の区分 → このスクリプトの区分（表示順）
CATEGORIES = [
    ("出力上限・生成が終わらない", "truncated"),
    ("初歩的: 構文エラー", "syntax"),
    ("初歩的: データの読み取り（キー・型・添字）", "data_access"),
    ("初歩的: 返り値の形が読めない", "output_shape"),
    ("初歩的: 禁止 import", "import"),
    ("実行時間超過", "timeout"),
    ("その他の実行時エラー", "runtime_other"),
    ("制約違反: 目的値の申告ずれだけ", "declared_only"),
    ("制約違反: 一部（違反 50% 未満）", "violation_partial"),
    ("定式化の誤りの疑い: 制約の大半（50% 以上）に違反", "violation_most"),
    ("定式化・解法の誤りの疑い: 可行だが参照より 50% 超悪い", "gap_large"),
    ("難しさ・解法の弱さ: 可行だが参照より 10〜50% 悪い", "gap_moderate"),
]
DATA_ACCESS = ("KEYERROR", "TYPE", "ATTRIBUTE", "INDEX", "VALUE", "NAME", "ZERODIVISION")
VIOLATION = re.compile(r"constraint violation \((\d+)/(\d+) violated")


def classify(row: dict, max_tokens: int) -> str:
    """1 行の失敗区分。正解の行には呼ばない。"""
    tokens = (row.get("usage") or {}).get("completion_tokens") or 0
    status, detail = row.get("status", ""), row.get("detail", "") or ""
    head = detail.split(":", 1)[0].strip().upper()
    if status == "gen_error" or tokens >= max_tokens - 8:
        return "truncated"
    if status == "exec_error":
        if head == "SYNTAX":
            return "syntax"
        if head == "IMPORT" or "forbidden" in detail.lower():
            return "import"
        if head == "TIMEOUT":
            return "timeout"
        if head in DATA_ACCESS:
            return "data_access"
        return "runtime_other"
    if status in ("invalid_solution", "unverified"):
        return "output_shape"
    if status in ("partial_feasible", "infeasible"):
        listed = detail.split(":", 1)[1] if ":" in detail else ""
        if listed and all("declared" in part for part in listed.split(";") if part.strip()):
            return "declared_only"
        m = VIOLATION.search(detail)
        if m and int(m.group(2)) and int(m.group(1)) / int(m.group(2)) >= 0.5:
            return "violation_most"
        return "violation_partial"
    gap = gap_of(row)
    if gap is not None and gap > 0.5:
        return "gap_large"
    return "gap_moderate"


def kinds_from(dirs: list[Path]) -> dict[str, str]:
    table = {}
    for directory in dirs:
        for path in directory.glob("prob_*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            requirements = record.get("requirements")
            kind = (record.get("provenance") or {}).get("kind") or (
                requirements.get("kind") if isinstance(requirements, dict) else None
            )
            table[f"prob_{record['id']:03d}" if record["id"] < 1000 else f"prob_{record['id']}"] = kind or ""
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=shard_dir|json")
    parser.add_argument("--problem-dir", type=Path, action="append", default=[])
    parser.add_argument("--supported-kinds-file", type=Path, help="学習した種別（対応外を別に数える）")
    parser.add_argument("--max-tokens", type=int, default=8192, help="評価時の出力枠（上限到達の判定）")
    parser.add_argument("--max-gap", type=float, default=0.10)
    parser.add_argument("--examples", type=int, default=0, help="区分ごとに例の問題番号をいくつ出すか")
    args = parser.parse_args()

    kind_of = kinds_from(args.problem_dir)
    supported = (
        set(json.loads(args.supported_kinds_file.read_text(encoding="utf-8"))["kinds"])
        if args.supported_kinds_file
        else None
    )
    runs = [load(spec) for spec in args.run]
    counts: dict[str, Counter] = {}
    examples: dict[str, dict[str, list[str]]] = {}
    for label, rows in runs:
        c, ex = Counter(), defaultdict(list)
        for iid, row in sorted(rows.items()):
            if is_correct(row, args.max_gap):
                c["correct"] += 1
                continue
            kind = kind_of.get(iid, "")
            if supported is not None and kind and kind not in supported:
                c["untrained"] += 1
                ex["untrained"].append(f"{iid}({kind})")
                continue
            category = classify(row, args.max_tokens)
            c[category] += 1
            ex[category].append(f"{iid}({kind})" if kind else iid)
        counts[label], examples[label] = c, ex

    labels = [label for label, _ in runs]
    print("| 区分 | " + " | ".join(labels) + " |")
    print("|---|" + "---:|" * len(labels))
    print("| 正解 | " + " | ".join(str(counts[l]["correct"]) for l in labels) + " |")
    if supported is not None:
        print("| 学習データにない種別 | " + " | ".join(str(counts[l]["untrained"]) for l in labels) + " |")
    for name, key in CATEGORIES:
        if any(counts[l][key] for l in labels):
            print(f"| {name} | " + " | ".join(str(counts[l][key]) for l in labels) + " |")
    print("| 合計 | " + " | ".join(str(sum(counts[l].values())) for l in labels) + " |")
    if args.examples:
        for label in labels:
            print(f"\n{label}:")
            for name, key in [("学習データにない種別", "untrained"), *CATEGORIES]:
                if examples[label][key]:
                    print(f"  {name}: {', '.join(examples[label][key][: args.examples])}")


if __name__ == "__main__":
    main()

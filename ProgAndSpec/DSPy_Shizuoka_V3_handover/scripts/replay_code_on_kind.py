#!/usr/bin/env python3
"""ある run で正解したコードを種別ごとに 1 本選び、別の問題集の同じ種別の問題で実行し採点する（LLM は呼ばない）。

学習済みモデルは種別ごとに教師コードをほぼそのまま書く。元問題 28 で解けないのが「教師の解き方が通じない」
からか「問題文の返り値の形が学習時と違い、モデルが書き換えて壊す」からかを分けるのに使う。
採点は対象の問題の参照解で行う（RESCORE_REPORT 36 章）。

  uv run python scripts/replay_code_on_kind.py \\
      --run outputs/prompt_model_comparisons/gemma4-ft-20261004-test/sft_gemma4_12b_fft_fp32__shard01of01 \\
      --source-dir data/problems_hard_gen/test --target-dir data/problems_hard \\
      --out outputs/prompt_model_comparisons/replay-code-on-kind-20261008.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.replay_failed import _rescore
from scripts.summarize_solver_runs import RESULT_FILENAME, is_correct
from src.agent import detect_kind
from src.data_loader import convert_to_dspy_example, load_v3_data


def _kind(record: dict) -> str:
    return detect_kind(convert_to_dspy_example(record)["core_type"], record.get("instance", {}))


def _run(job: tuple[dict, dict, float]) -> dict:
    row, record, timeout = job
    out = _rescore((row, record, timeout))
    # Why not 元の行の値: 選んだ行は別の instance のもので、参照値もその instance のもの。
    out["reference_value"] = convert_to_dspy_example(record)["reference_value"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="コードを取る shard ディレクトリ")
    parser.add_argument("--source-dir", type=Path, required=True, help="その run の問題ディレクトリ")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-gap", type=float, default=0.10)
    args = parser.parse_args()

    kinds = {f"prob_{r['id']:03d}": _kind(r) for r in load_v3_data(str(args.source_dir))}
    rows = json.loads((args.run / RESULT_FILENAME).read_text(encoding="utf-8"))["test"]["results"]
    code_of: dict[str, dict] = {}
    for row in sorted(rows, key=lambda r: r["instance_id"]):
        kind = kinds.get(row["instance_id"])
        if kind and kind not in code_of and is_correct(row, args.max_gap):
            code_of[kind] = row

    jobs, missing = [], []
    for record in load_v3_data(str(args.target_dir)):
        target_id, kind = f"prob_{record['id']:03d}", _kind(record)
        if kind not in code_of:
            missing.append(f"{target_id}({kind})")
            continue
        source = code_of[kind]
        row = {**source, "instance_id": target_id, "source_id": source["instance_id"]}
        jobs.append((row, record, args.timeout))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_run, jobs))

    correct = 0
    for row in results:
        ok = is_correct(row, args.max_gap)
        correct += ok
        print(f"{row['instance_id']} <- {row['source_id']}: {'ok' if ok else row['status']}")
    print(f"correct {correct}/{len(results)}; no correct code for: {', '.join(missing) or 'none'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

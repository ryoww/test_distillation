#!/usr/bin/env python3
"""student に 1 問あたり複数本の解を書かせ、参照値を使わない検証器で 1 本を選んで採点する。

1 本目は温度 0、残りは --temperature で生成する（単発評価と同じ messages）。各本を `verify_solution` で走らせ、
select_verified_best.pick の順位（可行で目的値あり ＞ 可行 ＞ 形を読めない ＞ 失敗、同順位は温度 0 の本を先）で
選び、選んだ本だけを参照値を使う採点にかける。

  uv run python scripts/sample_and_select.py --data-dir data/problems_hard --run-name samples-20261010 \\
      --label fft12b_v3_n4 --api-base http://127.0.0.1:<port>/v1 --model gemma4-12b --samples 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_solver_model import RESULT_FILENAME, chat
from scripts.evaluate_with_dsh import score
from scripts.run_cascade import verify
from scripts.select_verified_best import pick
from src.data_loader import convert_to_dspy_example, load_v3_data
from src.exec_gate import set_exec_concurrency
from src.modules import AlgorithmGenerator, ensure_parse_helpers, strip_code_fence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=4, help="1 問あたりの本数（1 本目は温度 0）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--extra-body", default="{}")
    parser.add_argument("--solve-limit", type=int, default=300)
    parser.add_argument("--exec-timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--exec-concurrency", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs" / "prompt_model_comparisons"
    )
    return parser.parse_args()


def generate(args: argparse.Namespace, instruction: str, example: dict, temperature: float) -> str | None:
    prompt = convert_to_dspy_example(example["record"], use_reference=False)["requirement"]
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
    content, _ = chat(args.api_base, "local", args.model, messages, args.max_tokens, temperature,
                      1800, json.loads(args.extra_body))
    code = ensure_parse_helpers(strip_code_fence(content))
    return code if "def solve" in code else None


def solve_one(args: argparse.Namespace, instruction: str, example: dict) -> dict:
    started = time.monotonic()
    temperatures = [0.0] + [args.temperature] * (args.samples - 1)
    codes = [generate(args, instruction, example, t) for t in temperatures]
    candidates = [(str(i), *verify(code, example, args.solve_limit)) for i, code in enumerate(codes)]
    chosen = int(pick(candidates))
    row = {
        "instance_id": example["instance_id"],
        "name": example.get("name", ""),
        "core_type": example["core_type"],
        "samples": [{"index": c[0], "temperature": t, "verdict": c[1], "cost": c[2]}
                    for c, t in zip(candidates, temperatures)],
        "distinct_codes": len({c for c in codes if c}),
        "picked_index": chosen,
    }
    if not codes[chosen]:
        result = {"code": None, "status": "gen_error", "score": -0.5, "detail": "no code"}
    else:
        result = score(args, example, codes[chosen])
    # Why not 前の単発評価と比べる: 温度 0 の本も生成し直すので回ごとに揺れる。同じ回の温度 0 の本を基準にする。
    if chosen == 0 or not codes[0]:
        greedy = result if chosen == 0 else {"status": "gen_error", "score": -0.5, "cost": None}
    else:
        greedy = score(args, example, codes[0])
    row.update({"greedy_status": greedy["status"], "greedy_score": greedy["score"],
                "greedy_cost": greedy.get("cost")})
    return {**row, **result, "elapsed": time.monotonic() - started}


def main() -> int:
    args = parse_args()
    set_exec_concurrency(args.exec_concurrency)
    instruction = AlgorithmGenerator().generate.predict.signature.instructions
    examples = []
    for record in load_v3_data(str(args.data_dir)):
        example = convert_to_dspy_example(record, use_reference=True)
        example["record"] = record
        examples.append(example)
    examples.sort(key=lambda e: e["instance_id"])
    shard_dir = args.output_dir / args.run_name / f"{args.label}__shard01of01"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(solve_one, args, instruction, ex) for ex in examples]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{index}/{len(examples)}] {row['instance_id']}: {row['status']} "
                  f"picked={row['picked_index']} distinct={row['distinct_codes']} "
                  f"verdicts={[s['verdict'] for s in row['samples']]} ({row['elapsed']:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["instance_id"])
    payload = {
        "test": {"results": rows, "total_count": len(rows),
                 "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1)},
        "config": {"label": args.label, "data_dir": str(args.data_dir.resolve()), "model": args.model,
                   "samples": args.samples, "temperature": args.temperature,
                   "rule": "verifier pick; greedy sample first on ties"},
    }
    (shard_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"results: {shard_dir / RESULT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

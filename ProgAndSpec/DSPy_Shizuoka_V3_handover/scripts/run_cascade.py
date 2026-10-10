#!/usr/bin/env python3
"""3 月の構成を段階式で通して動かす: student の一発回答 → 検証 → 失敗なら汎用エージェント（dsh）→ 選択 → 採点。

1. 対応表（--supported-kinds-file）にない種別は student を使わず、最初から dsh に解かせる。
2. 対応する種別は student（学習済みモデル、OpenAI 互換 API）に単発評価と同じ messages で解かせ、参照値を使わない
   検証器（`verify_solution`）で確かめる。可行ならそれで終わる。
3. 可行でなければ dsh に一から解かせ、student と dsh の版を検証器の順位（select_verified_best.pick）で選ぶ。
4. 選んだ版だけを、参照値を使う採点（他の章と同じ）にかける。

dsh のモデルは DSH_LOCAL_BASE_URL などの環境変数で指す（evaluate_with_dsh.py と同じプロファイル）。

  DSH_HOME=... DSH_LOCAL_BASE_URL=http://127.0.0.1:<port>/v1 DSH_LOCAL_API_KEY=local DSH_LOCAL_MODEL=fallback \\
  uv run python scripts/run_cascade.py --data-dir data/problems_hard --run-name cascade-20261010 \\
      --label fft_then_dsh --student-api-base http://127.0.0.1:<port>/v1 --student-model gemma4-12b \\
      --supported-kinds-file prompts/supported_kinds/sft_merged_v3.json \\
      --dsh /var/tmp/yy-lab-dsh/node_modules/.bin/dsh --work-root /var/tmp/yy-lab-dsh/ws-cascade
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
from scripts.evaluate_with_dsh import run_dsh, score
from scripts.select_verified_best import pick
from src.agent import detect_kind, load_supported_kinds
from src.data_loader import convert_to_dspy_example, load_v3_data
from src.exec_gate import exec_slot, set_exec_concurrency
from src.modules import AlgorithmGenerator, ensure_parse_helpers, strip_code_fence
from src.utils.feasibility import check_feasibility_detailed
from src.verify_loop import verify_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--student-api-base", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-max-tokens", type=int, default=16384)
    parser.add_argument("--student-extra-body", default="{}")
    parser.add_argument("--supported-kinds-file", type=Path, required=True)
    parser.add_argument("--dsh", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--task-timeout", type=float, default=3600.0)
    parser.add_argument("--solve-limit", type=int, default=300)
    parser.add_argument("--exec-timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--exec-concurrency", type=int, default=4)
    parser.add_argument("--exclude-templated", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs" / "prompt_model_comparisons"
    )
    return parser.parse_args()


def verify(code: str | None, example: dict, timeout: float) -> tuple[str, float | None]:
    """参照値を使わない検証器の判定と、可行なら再計算した目的値。"""
    if not code:
        return "missing", None
    with exec_slot():
        verdict = verify_solution(code, example["instance"], example["core_type"], timeout=timeout)
    if verdict.kind != "feasible" or not isinstance(verdict.solution, dict):
        return verdict.kind, None
    checked = check_feasibility_detailed(example["core_type"], example["instance"], verdict.solution)
    return "feasible", checked.get("cost")


def student_code(args: argparse.Namespace, instruction: str, example: dict) -> str | None:
    prompt = convert_to_dspy_example(example["record"], use_reference=False)["requirement"]
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
    content, _ = chat(
        args.student_api_base, "local", args.student_model, messages, args.student_max_tokens,
        0.0, 1800, json.loads(args.student_extra_body),
    )
    code = ensure_parse_helpers(strip_code_fence(content))
    return code if "def solve" in code else None


def solve_one(args: argparse.Namespace, instruction: str, supported: set[str], example: dict) -> dict:
    started = time.monotonic()
    kind = detect_kind(example["core_type"], example["instance"])
    row = {"instance_id": example["instance_id"], "name": example.get("name", ""),
           "core_type": example["core_type"], "kind": kind, "supported": kind in supported}
    candidates: list[tuple[str, str, float | None]] = []
    codes: dict[str, str | None] = {}
    if row["supported"]:
        codes["student"] = student_code(args, instruction, example)
        candidates.append(("student", *verify(codes["student"], example, args.solve_limit)))
    if not candidates or candidates[0][1] != "feasible":
        workspace = args.work_root / args.label / example["instance_id"]
        codes["dsh"], record = run_dsh(
            args.dsh, workspace, example, args.solve_limit, args.task_timeout
        )
        row.update({k: v for k, v in record.items() if k != "dsh_stderr_tail"})
        candidates.append(("dsh", *verify(codes["dsh"], example, args.solve_limit)))
    chosen = pick(candidates)
    row["stages"] = [{"run": c[0], "verdict": c[1], "cost": c[2]} for c in candidates]
    row["picked_from"] = chosen
    row["elapsed"] = time.monotonic() - started
    if not codes.get(chosen):
        return {**row, "code": None, "status": "gen_error", "score": -0.5, "detail": "no code"}
    return {**row, **score(args, example, codes[chosen])}


def main() -> int:
    args = parse_args()
    set_exec_concurrency(args.exec_concurrency)
    supported = load_supported_kinds(args.supported_kinds_file)
    instruction = AlgorithmGenerator().generate.predict.signature.instructions
    examples = []
    for record in load_v3_data(str(args.data_dir)):
        example = convert_to_dspy_example(record, use_reference=True)
        example["record"] = record
        examples.append(example)
    examples.sort(key=lambda e: e["instance_id"])
    if args.exclude_templated:
        from src.datagen import TEMPLATES

        templated = {f"prob_{k:03d}" for k in TEMPLATES}
        examples = [ex for ex in examples if ex["instance_id"] not in templated]
    shard_dir = args.output_dir / args.run_name / f"{args.label}__shard01of01"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(solve_one, args, instruction, supported, ex) for ex in examples]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{index}/{len(examples)}] {row['instance_id']}: {row['status']} "
                  f"picked={row['picked_from']} stages={row['stages']} ({row['elapsed']:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["instance_id"])
    payload = {
        "test": {"results": rows, "total_count": len(rows),
                 "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1)},
        "config": {"label": args.label, "data_dir": str(args.data_dir.resolve()),
                   "student_model": args.student_model,
                   "supported_kinds_file": str(args.supported_kinds_file),
                   "harness": "student -> verify -> dsh on failure -> verifier pick"},
    }
    (shard_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"results: {shard_dir / RESULT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

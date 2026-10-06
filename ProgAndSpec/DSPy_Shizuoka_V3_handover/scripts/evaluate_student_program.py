#!/usr/bin/env python3
"""生成→検証→修復の 2 段プログラム（src/student_program.StudentRepairSolver）を問題集で採点する。

evaluate_solver_model.py と同じ shard 形式で書くので、summarize_solver_runs.py でそのまま比べられる。
修復段の指示文は GEPA の出力（outputs/<run>/gepa/instruction.md）を `--repair-instruction-file` で渡す。
省略すると既定の修復指示文で、「修復ループだけの効果」が測れる。

  uv run python scripts/evaluate_student_program.py --data-dir data/problems_hard_gen/test \\
      --api-base http://127.0.0.1:7601/v1 --model student --label s0_repair_gepa --run-name bt-repair-eval \\
      --repair-instruction-file outputs/bt-repair-r1/gepa/instruction.md
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

import dspy

from src import best_known as _best_known
from src.data_loader import convert_to_dspy_example, load_v3_data
from src.lm_config import LMConfig, create_lm
from src.metrics_v3 import evaluate_algorithm_v3
from src.student_program import StudentRepairSolver

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:7501/v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs" / "prompt_model_comparisons"
    )
    parser.add_argument("--generate-instruction-file", type=Path)
    parser.add_argument("--repair-instruction-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--exec-timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--exclude-templated", action="store_true", help="雛形化済みの問題を除く（汎化の測定用）"
    )
    return parser.parse_args()


def _read(path: Path | None) -> str | None:
    return path.read_text(encoding="utf-8") if path else None


def score_row(program: StudentRepairSolver, example: dict, exec_timeout: float) -> dict:
    """1 問を 2 段で解いて採点する。参照値は採点にだけ使い、プログラムには渡さない。"""
    started = time.monotonic()
    prompt = convert_to_dspy_example(example["record"], use_reference=False)["requirement"]
    pred = program(
        requirement=prompt, core_type=example["core_type"], problem_instance=example["instance"]
    )
    code = pred.algorithm_code
    base = {
        "instance_id": example["instance_id"],
        "name": example.get("name", ""),
        "core_type": example["core_type"],
        "code": code,
        "first_verdict": pred.first_verdict,
        "repaired": pred.repaired,
    }
    if "def solve" not in code:
        return {**base, "status": "gen_error", "score": -0.5, "detail": "no solve() in response",
                "elapsed": round(time.monotonic() - started, 2)}
    registry = _best_known.BestKnownRegistry()
    if example.get("reference_value") is not None:
        registry.register(example["instance_id"], example["reference_value"])
    result = evaluate_algorithm_v3(
        code=code,
        instance=example["instance"],
        core_type=example["core_type"],
        instance_id=example["instance_id"],
        registry=registry,
        timeout=exec_timeout,
        reference_value=example.get("reference_value"),
        reference_solution=example.get("reference_solution", {}),
        objective_text=example.get("objective", ""),
        use_reference=True,
    )
    return {
        **base,
        "status": result["status"],
        "score": result["score"],
        "cost": result.get("cost"),
        "reference_value": example.get("reference_value"),
        "detail": result.get("detail", ""),
        "feasibility_verified": result.get("feasibility_verified"),
        "elapsed": round(time.monotonic() - started, 2),
    }


def main() -> int:
    args = parse_args()
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
    if args.limit:
        examples = examples[: args.limit]

    dspy.settings.configure(
        lm=create_lm(
            LMConfig(
                model=args.model,
                api_base=args.api_base,
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                enable_thinking=args.thinking == "on",
            )
        )
    )
    program = StudentRepairSolver(
        _read(args.generate_instruction_file), _read(args.repair_instruction_file), args.exec_timeout
    )

    shard_dir = args.output_dir / args.run_name / f"{args.label}__shard01of01"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(score_row, program, ex, args.exec_timeout) for ex in examples]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(examples)}] {row['instance_id']}: {row['status']} "
                f"score={row['score']:.2f} first={row['first_verdict']} repaired={row['repaired']} "
                f"({row['elapsed']:.0f}s)",
                flush=True,
            )
    rows.sort(key=lambda r: r["instance_id"])
    payload = {
        "test": {
            "results": rows,
            "total_count": len(rows),
            "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1),
        },
        "config": {
            "api_base": args.api_base,
            "model": args.model,
            "label": args.label,
            "data_dir": str(args.data_dir.resolve()),
            "program": "StudentRepairSolver",
            "generate_instruction_file": str(args.generate_instruction_file or ""),
            "repair_instruction_file": str(args.repair_instruction_file or ""),
            "thinking": args.thinking,
        },
    }
    (shard_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    repaired = sum(1 for r in rows if r["repaired"])
    print(f"\n{args.label}: mean={payload['test']['mean_score']:.3f} repaired={repaired}/{len(rows)}")
    print(f"results: {shard_dir / RESULT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

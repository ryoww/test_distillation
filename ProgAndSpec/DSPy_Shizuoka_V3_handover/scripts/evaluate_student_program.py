#!/usr/bin/env python3
"""生成→検証→修復（→フォールバック）のエージェント経路（src/agent.OptimizationAgent）を問題集で採点する。

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
from src.agent import OptimizationAgent, load_supported_kinds
from src.data_loader import convert_to_dspy_example, load_v3_data
from src.exec_gate import exec_slot, set_exec_concurrency
from src.lm_config import LMConfig, create_lm
from src.metrics_v3 import evaluate_algorithm_v3

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
    parser.add_argument("--max-repairs", type=int, default=1, help="修復の回数（GEPA の検証と揃えるなら 1）")
    parser.add_argument("--fallback-model", help="student が通らないときに同じ手順で試す大きいモデル")
    parser.add_argument("--fallback-api-base")
    parser.add_argument("--fallback-thinking", choices=("on", "off"), default="off")
    parser.add_argument("--supported-kinds-file", type=Path, help="対応外の種別を unsupported で返す")
    parser.add_argument("--repair-instruction-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--exec-timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--exec-concurrency", type=int, default=0, help="生成コードの同時実行数（0 は無制限）")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--exclude-templated", action="store_true", help="雛形化済みの問題を除く（汎化の測定用）"
    )
    parser.add_argument(
        "--schema-from",
        type=Path,
        help="問題文の返り値の形を、この問題ディレクトリの同じ種別でいちばん多い形に差し替える",
    )
    return parser.parse_args()


def _read(path: Path | None) -> str | None:
    return path.read_text(encoding="utf-8") if path else None


def score_row(agent: OptimizationAgent, example: dict, exec_timeout: float) -> dict:
    """1 問をエージェント経路で解いて採点する。参照値は採点にだけ使い、エージェントには渡さない。"""
    started = time.monotonic()
    record = example.get("prompt_record", example["record"])
    prompt = convert_to_dspy_example(record, use_reference=False)["requirement"]
    result = agent.solve(prompt, example["core_type"], example["instance"])
    code = result.code
    base = {
        "instance_id": example["instance_id"],
        "name": example.get("name", ""),
        "core_type": example["core_type"],
        "code": code,
        "agent_status": result.status,
        "first_verdict": result.attempts[0].verdict if result.attempts else "",
        "repaired": any(a.stage.endswith("-repair") for a in result.attempts),
        "fallback_used": result.fallback_used,
        "attempts": len(result.attempts),
        "kind": result.kind,
        "supported": result.supported,
    }
    if result.status == "unsupported":
        # Why score 0: 対応外と明示して止まるのは誤答（−0.5）より良く、正解でもない。
        return {**base, "status": "unsupported", "score": 0.0, "detail": result.failure_reason,
                "elapsed": round(time.monotonic() - started, 2)}
    if "def solve" not in code:
        return {**base, "status": "gen_error", "score": -0.5, "detail": "no solve() in response",
                "elapsed": round(time.monotonic() - started, 2)}
    registry = _best_known.BestKnownRegistry()
    if example.get("reference_value") is not None:
        registry.register(example["instance_id"], example["reference_value"])
    with exec_slot():
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
    set_exec_concurrency(args.exec_concurrency)
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
    if args.schema_from:
        from scripts.evaluate_solver_model import apply_schema_from

        apply_schema_from(examples, args.schema_from)
    if args.limit:
        examples = examples[: args.limit]

    def lm_for(model, api_base, thinking):
        return create_lm(
            LMConfig(
                model=model,
                api_base=api_base,
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                enable_thinking=thinking == "on",
            )
        )

    student = lm_for(args.model, args.api_base, args.thinking)
    dspy.settings.configure(lm=student)
    fallback = None
    if args.fallback_model:
        fallback = lm_for(
            args.fallback_model, args.fallback_api_base or args.api_base, args.fallback_thinking
        )
    program = OptimizationAgent(
        student,
        fallback_lm=fallback,
        generate_instruction=_read(args.generate_instruction_file),
        repair_instruction=_read(args.repair_instruction_file),
        max_repairs=args.max_repairs,
        exec_timeout=args.exec_timeout,
        supported_kinds=(
            load_supported_kinds(args.supported_kinds_file) if args.supported_kinds_file else None
        ),
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
                f"fallback={row['fallback_used']} "
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
            "program": "OptimizationAgent",
            "max_repairs": args.max_repairs,
            "fallback_model": args.fallback_model or "",
            "generate_instruction_file": str(args.generate_instruction_file or ""),
            "repair_instruction_file": str(args.repair_instruction_file or ""),
            "thinking": args.thinking,
            "schema_from": str(args.schema_from or ""),
        },
    }
    (shard_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    repaired = sum(1 for r in rows if r["repaired"])
    fell_back = sum(1 for r in rows if r["fallback_used"])
    print(
        f"\n{args.label}: mean={payload['test']['mean_score']:.3f} "
        f"repaired={repaired}/{len(rows)} fallback={fell_back}/{len(rows)}"
    )
    print(f"results: {shard_dir / RESULT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

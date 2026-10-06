#!/usr/bin/env python3
"""エージェントの入口: 問題レコード（この repo の JSON 形式）を 1 問渡すと、コードと判定を JSON で返す。

  uv run python scripts/solve_requirement.py data/problems_hard_gen/test/prob_4035.json \\
      --student-model student --student-api-base http://127.0.0.1:7601/v1 \\
      [--fallback-model gemma4-12b --fallback-api-base http://127.0.0.1:7602/v1] [--out result.json]

問題文は参照値を含めずに組み立てる（`convert_to_dspy_example(use_reference=False)`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent import OptimizationAgent
from src.data_loader import convert_to_dspy_example
from src.lm_config import LMConfig, create_lm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-api-base", default="http://127.0.0.1:7501/v1")
    parser.add_argument("--student-thinking", choices=("on", "off"), default="on")
    parser.add_argument("--fallback-model")
    parser.add_argument("--fallback-api-base")
    parser.add_argument("--fallback-thinking", choices=("on", "off"), default="off")
    parser.add_argument("--repair-instruction-file", type=Path)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--exec-timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))
    example = convert_to_dspy_example(record, use_reference=False)
    student = create_lm(
        LMConfig(
            model=args.student_model,
            api_base=args.student_api_base,
            temperature=0.0,
            max_tokens=args.max_tokens,
            timeout=1800,
            enable_thinking=args.student_thinking == "on",
        )
    )
    fallback = None
    if args.fallback_model:
        fallback = create_lm(
            LMConfig(
                model=args.fallback_model,
                api_base=args.fallback_api_base or args.student_api_base,
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout=1800,
                enable_thinking=args.fallback_thinking == "on",
            )
        )
    agent = OptimizationAgent(
        student,
        fallback_lm=fallback,
        repair_instruction=(
            args.repair_instruction_file.read_text(encoding="utf-8")
            if args.repair_instruction_file
            else None
        ),
        max_repairs=args.max_repairs,
        exec_timeout=args.exec_timeout,
    )
    result = agent.solve(example["requirement"], example["core_type"], example["instance"])
    payload = {"instance_id": example["instance_id"], **result.to_dict()}
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

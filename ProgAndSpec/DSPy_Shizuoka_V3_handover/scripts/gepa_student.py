#!/usr/bin/env python3
"""特化学習したモデル（student）の system 指示文を DSPy の GEPA で進化させる。

プロンプト最適化と fine-tuning を交互に回す（BetterTogether 型）ときの「プロンプト側」の 1 段。
student は vLLM で配信し、学習時と同じ形式（`src/student_program.py`）で解かせる。反省は別の LM
（既定は Qwen3.8）が行い、反省プロンプトには一般則だけを書かせるテンプレートを使う。

出力（`--run-dir`）:
- `instruction.md`: 検証集合で最良だった指示文（次の fine-tuning の system と評価に使う）
- `summary.json`: 候補ごとの検証平均と設定
- `gepa_logs/`: GEPA の状態（同じ `--run-dir` で `--resume` すると続きから）

  uv run python scripts/gepa_student.py --run-dir outputs/bt-round1/gepa \
      --student-model gemma4-12b --student-api-base http://127.0.0.1:7501/v1 \
      --reflection-model Qwen/Qwen3.8-27B --reflection-api-base http://127.0.0.1:7502/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import dspy
from dspy.clients._litellm import get_litellm

from src.best_known import init_registry
from src.data_loader import load_split_dirs, prepare_examples
from src.gepa_feedback_v3 import gepa_feedback_v3, set_exec_timeout, set_use_reference
from src.lm_config import LMConfig, create_lm
from src.student_program import (
    StudentRepairSolver,
    StudentSolver,
    default_instruction,
    examples_with_instance,
)
from train_gepa_v3 import TemplateInstructionProposer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--student-model", required=True, help="vLLM の served model name")
    parser.add_argument("--student-api-base", default="http://127.0.0.1:7501/v1")
    parser.add_argument("--student-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--student-thinking",
        choices=("on", "off"),
        default="on",
        help="chat template の enable_thinking。SFT 済みは学習時の描画に合わせて on、素のモデルは評価と同じ off",
    )
    parser.add_argument("--reflection-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--reflection-api-base", default="http://127.0.0.1:7502/v1")
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data" / "problems_hard_gen"))
    parser.add_argument("--per-kind-train", type=int, default=1)
    parser.add_argument("--per-kind-val", type=int, default=1)
    parser.add_argument(
        "--max-train-requirement-chars",
        type=int,
        default=60000,
        help="反省用の train から外す問題文の長さ（検証集合には残す）",
    )
    parser.add_argument("--seed-instruction-file", type=Path, help="省略時は SFT の既定の指示文")
    parser.add_argument(
        "--reflection-template",
        type=Path,
        default=BASE_DIR / "prompts" / "reflection_template_general.md",
    )
    parser.add_argument("--max-full-evals", type=int, default=6)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--exec-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="生成→検証→修復の 2 段にし、GEPA には修復段の指示文だけを進化させる（生成段の指示文は固定）",
    )
    parser.add_argument("--repair-seed-file", type=Path, help="修復段の初期指示文（省略時は既定）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Why not dspy の遅延読み込みに任せる: 並列評価の複数スレッドが同時に litellm を初めて読むと、読み込み途中の
    # モジュールを掴んで例外になり、GEPA が止まった（Slurm 924）。スレッドを立てる前にメインスレッドで読んでおく。
    get_litellm(feature="dspy.LM")
    # Why not run_dir の存在で止める: Slurm ジョブが先に logs/ を作る。上書きを防ぐのは GEPA の状態だけでよい。
    if (args.run_dir / "gepa_logs").exists() and not args.resume:
        raise SystemExit(f"{args.run_dir}/gepa_logs exists; pass --resume or use another --run-dir")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_exec_timeout(args.exec_timeout)
    # 採点は参照解との比で行う（学習データの選別と同じ基準に近づける）。問題文には参照値を入れない。
    set_use_reference(True)
    registry = init_registry(storage_path=args.run_dir / "best_known.json")

    splits = load_split_dirs(
        args.data_dir,
        {"train": args.per_kind_train, "validation": args.per_kind_val, "test": 0},
        use_reference=False,
    )
    for example in splits["train"] + splits["validation"]:
        if example.get("reference_value") is not None:
            registry.register(example["instance_id"], example["reference_value"])
    # Why not 全問を反省に回す: 反省 LM には minibatch 3 問の問題文がまとめて渡る。ポートフォリオは
    # 1 問で約 4 万 token あり、2 問重なると Qwen3.8 の文脈 131k を超える。
    train = [
        ex for ex in splits["train"] if len(ex["requirement"]) <= args.max_train_requirement_chars
    ]
    trainset = prepare_examples(train)
    valset = prepare_examples(splits["validation"])
    if args.repair:
        trainset, valset = examples_with_instance(trainset), examples_with_instance(valset)

    # Why temperature 0: GEPA は同じ minibatch で親と子を比べて採否を決める。温度を上げると
    # 指示文の差より標本の揺れで採否が決まる。enable_thinking は学習時の描画に合わせる。
    student = create_lm(
        LMConfig(
            model=args.student_model,
            api_base=args.student_api_base,
            temperature=0.0,
            max_tokens=args.student_max_tokens,
            timeout=1800,
            enable_thinking=args.student_thinking == "on",
        )
    )
    reflection = create_lm(
        LMConfig(model=args.reflection_model, api_base=args.reflection_api_base)
    )
    dspy.settings.configure(lm=student)

    seed = (
        args.seed_instruction_file.read_text(encoding="utf-8")
        if args.seed_instruction_file
        else default_instruction()
    )
    if args.repair:
        repair_seed = (
            args.repair_seed_file.read_text(encoding="utf-8") if args.repair_seed_file else None
        )
        program = StudentRepairSolver(seed, repair_seed, exec_timeout=args.exec_timeout)
    else:
        program = StudentSolver(seed)
    optimizer = dspy.GEPA(
        metric=gepa_feedback_v3,
        reflection_lm=reflection,
        instruction_proposer=TemplateInstructionProposer(
            args.reflection_template.read_text(encoding="utf-8")
        ),
        max_full_evals=args.max_full_evals,
        track_stats=True,
        log_dir=str(args.run_dir / "gepa_logs"),
        num_threads=args.num_threads,
        seed=0,
    )
    compiled = optimizer.compile(program, trainset=trainset, valset=valset)

    evolved = compiled.repair if args.repair else compiled.generate
    instruction = evolved.signature.instructions
    (args.run_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    results = getattr(compiled, "detailed_results", None)
    summary = {
        "student_model": args.student_model,
        "student_thinking": args.student_thinking,
        "reflection_model": args.reflection_model,
        "train": len(trainset),
        "validation": len(valset),
        "max_full_evals": args.max_full_evals,
        "seed_is_default": args.seed_instruction_file is None,
        "instruction_chars": len(instruction),
        "evolved_component": "repair" if args.repair else "generate",
        "val_aggregate_scores": list(getattr(results, "val_aggregate_scores", []) or []),
        "best_idx": getattr(results, "best_idx", None),
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

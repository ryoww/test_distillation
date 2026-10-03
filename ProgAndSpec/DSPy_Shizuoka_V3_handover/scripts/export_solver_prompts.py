#!/usr/bin/env python3
"""問題ごとの入力（指示文 + requirement）を書き出し、外部で書かれた solve() を集めて結果 JSON にする。

文脈を持たないモデル（Claude のサブエージェントなど）に同じ入力で解かせるための入出力。
`export` は 1 問 1 ファイルの入力を書き、`collect` は同じ名前の出力ファイルから solve() を
読んで、compare の shard 形式（`test.results` に instance_id / code / status）で保存する。
保存した結果は `scripts/rescore_with_checkers.py` でほかの条件と同じ採点系にかける。

  uv run python scripts/export_solver_prompts.py export --data-dir data/problems_hard --out-dir outputs/prompts_hard
  uv run python scripts/export_solver_prompts.py collect --data-dir data/problems_hard \
      --answers-dir outputs/prompts_hard/answers --run-dir outputs/prompt_model_comparisons/hard28-fable/fable__claude__shard01of01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import convert_to_dspy_example, load_v3_data
from src.modules import AlgorithmGenerator

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--data-dir", type=Path, required=True)
    exp.add_argument("--out-dir", type=Path, required=True)
    exp.add_argument(
        "--no-reference",
        action="store_true",
        help="参照値を問題文に入れない（SFT の学習入力と同じ形にする）",
    )
    exp.add_argument("--ids", type=Path, help="対象の instance_id を 1 行 1 つ書いたファイル")
    exp.add_argument(
        "--instances-out",
        type=Path,
        help="instance だけの JSON を書き出す先（解答者が手元で実行して確かめる用。参照解は含めない）",
    )
    col = sub.add_parser("collect")
    col.add_argument("--data-dir", type=Path, required=True)
    col.add_argument("--answers-dir", type=Path, required=True)
    col.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def extract_code(text: str) -> str:
    """```python フェンスがあれば中身、なければ全文をコードとして扱う。"""
    fenced = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    return max(fenced, key=len).strip() if fenced else text.strip()


def main() -> None:
    args = parse_args()
    records = load_v3_data(str(args.data_dir))
    if args.command == "export":
        if args.ids:
            wanted = {line.strip() for line in args.ids.read_text().splitlines() if line.strip()}
            records = [r for r in records if f"prob_{r['id']:03d}" in wanted]
        instruction = AlgorithmGenerator().generate.predict.signature.instructions
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if args.instances_out:
            args.instances_out.mkdir(parents=True, exist_ok=True)
        for record in records:
            example = convert_to_dspy_example(record, use_reference=not args.no_reference)
            if args.instances_out:
                (args.instances_out / f"{example['instance_id']}.json").write_text(
                    json.dumps(record["instance"], ensure_ascii=False), encoding="utf-8"
                )
            text = (
                f"{instruction}\n\n---\n\n{example['requirement']}\n\n---\n\n"
                "Write the complete Python code for solve(instance) and nothing else. "
                "Put the code in a single ```python fence.\n"
            )
            (args.out_dir / f"{example['instance_id']}.md").write_text(text, encoding="utf-8")
        print(f"wrote {len(records)} prompts to {args.out_dir}")
        return
    results = []
    for record in records:
        instance_id = f"prob_{record['id']:03d}"
        answer = args.answers_dir / f"{instance_id}.py"
        if not answer.exists():
            answer = args.answers_dir / f"{instance_id}.md"
        if not answer.exists():
            results.append(
                {
                    "instance_id": instance_id,
                    "status": "gen_error",
                    "score": -0.5,
                    "error": "no answer file",
                }
            )
            continue
        code = extract_code(answer.read_text(encoding="utf-8"))
        results.append(
            {"instance_id": instance_id, "code": code, "status": "pending", "score": 0.0}
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "test": {"results": results},
        "config": {"data_dir": str(args.data_dir.resolve()), "source": "external answers"},
    }
    (args.run_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"collected {sum('code' in r for r in results)} / {len(results)} answers into {args.run_dir}"
    )


if __name__ == "__main__":
    main()

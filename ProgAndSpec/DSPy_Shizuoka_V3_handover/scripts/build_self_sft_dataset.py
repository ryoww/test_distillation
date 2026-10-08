#!/usr/bin/env python3
"""モデル自身が解いて正解した行（評価 shard の exact_match / beat_reference）を SFT の messages にする。

BetterTogether（論文どおり）の w 段で、教師データを使わず「プロンプト最適化したモデルの成功例」だけで学習するための
データ。system は解いたときの指示文、user は参照値を含まない問題文、assistant はそのコード（`build_hard_sft_dataset.py`
と同じ形）。instance id の末尾で validation を分ける（再生はしない）。

  uv run python scripts/build_self_sft_dataset.py \\
      --run outputs/prompt_model_comparisons/btpaper-sample-problems_train/base_ib1__shard01of01 \\
      --data-dir data/sft/problems_train --instruction-file outputs/bt-base-r1/gepa/instruction.md \\
      --output-dir data/sft_self_base_ib1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import convert_to_dspy_example

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"
SOLVED = {"exact_match", "beat_reference"}


def to_rows(results: list[dict], records: dict[str, dict], instruction: str, every: int) -> dict:
    """正解した行を messages にし、id の数字が every で割り切れるものを validation に回す。"""
    out: dict[str, list[dict]] = {"train": [], "validation": []}
    for row in results:
        if row.get("status") not in SOLVED or not row.get("code"):
            continue
        record = records[row["instance_id"]]
        prompt = convert_to_dspy_example(record, use_reference=False)["requirement"]
        split = "validation" if int(record["id"]) % every == 0 else "train"
        out[split].append(
            {
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"```python\n{row['code'].strip()}\n```"},
                ],
                "tools": None,
                "instance_id": row["instance_id"],
                "status": row["status"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="評価 shard のディレクトリ")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-every", type=int, default=20)
    args = parser.parse_args()

    records = {}
    for path in args.data_dir.glob("prob_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[convert_to_dspy_example(record)["instance_id"]] = record
    results = json.loads((args.run / RESULT_FILENAME).read_text(encoding="utf-8"))["test"]["results"]
    rows = to_rows(
        results, records, args.instruction_file.read_text(encoding="utf-8"), args.validation_every
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8"
        )
    stats = {split: dict(Counter(r["status"] for r in items)) for split, items in rows.items()}
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

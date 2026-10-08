#!/usr/bin/env python3
"""種別ごとに解答コードを 1 本選び、`evaluate_solver_model.py --incontext-examples` 用の JSON にする。

解答は `<answers>/<prob_id>.py`（学習用 instance への Opus の解答など）。種別は問題レコードの instance から
`src.agent.detect_kind` で引く。同じ種別に複数あれば問題番号の小さいものを使う。テスト用 instance の解答は渡さないこと。

  uv run python scripts/build_incontext_examples.py --answers outputs/opus_hard/answers \\
      --problem-dir data/problems_hard_gen/train --out outputs/incontext/opus_hard_per_kind.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent import detect_kind
from src.data_loader import core_type_from_v3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--problem-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    for directory in args.problem_dir:
        for path in directory.glob("prob_*.json"):
            records[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, str] = {}
    sources: dict[str, str] = {}
    for path in sorted(args.answers.glob("prob_*.py")):
        record = records.get(path.stem)
        if record is None:
            continue
        kind = detect_kind(core_type_from_v3(record), record["instance"])
        if kind not in table:
            table[kind], sources[kind] = path.read_text(encoding="utf-8"), path.stem
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"kinds": len(table), "from": sources}, ensure_ascii=False))


if __name__ == "__main__":
    main()

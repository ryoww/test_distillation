#!/usr/bin/env python3
"""student が学習した種別の対応表を、学習に使ったデータから書き出す（エージェントの対応外判定に使う）。

種別は `src.agent.detect_kind` と同じ規則で決める: 大規模問題は instance の形から引いた種別名、雛形問題は core_type。
学習対が 0 件の種別や、長さ超過で学習から落ちた種別はデータからは見分けられないので `--exclude-kinds` で外す。

  uv run python scripts/list_supported_kinds.py --template-dir data/sft/problems_train \\
      --hard-sft data/sft_hard_opus --exclude-kinds crew_pairing,portfolio,portfolio_cvar \\
      --out prompts/supported_kinds/sft_merged_v1.json
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


def template_kinds(directory: Path) -> set[str]:
    kinds = set()
    for path in directory.glob("prob_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        kinds.add(detect_kind(core_type_from_v3(record), record.get("instance", {})))
    return kinds


def hard_kinds(sft_dir: Path) -> set[str]:
    kinds = set()
    for line in (sft_dir / "train.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            kinds.add(json.loads(line)["kind"])
    return kinds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-dir", type=Path, action="append", default=[])
    parser.add_argument("--hard-sft", type=Path, action="append", default=[])
    parser.add_argument("--exclude-kinds", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    kinds: set[str] = set()
    for directory in args.template_dir:
        kinds |= template_kinds(directory)
    for directory in args.hard_sft:
        kinds |= hard_kinds(directory)
    excluded = {k for k in args.exclude_kinds.split(",") if k}
    kinds -= excluded
    payload = {
        "kinds": sorted(kinds),
        "template_dirs": [str(d) for d in args.template_dir],
        "hard_sft_dirs": [str(d) for d in args.hard_sft],
        "excluded": sorted(excluded),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(kinds)} kinds -> {args.out}")


if __name__ == "__main__":
    main()

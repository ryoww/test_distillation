#!/usr/bin/env python3
"""複数の SFT データ（train.jsonl / validation.jsonl）を 1 つの学習用ディレクトリにまとめる。

雛形 89 問の `data/sft` と大規模問題の `data/sft_hard_*` は列が違うので、学習に要る
`messages` と `tools` に出典ラベル `source` だけを足して書き出す。`label=dir*N` で train を
N 回繰り返す（件数の少ない集合の比重を上げる）。validation は繰り返さない。

  uv run python scripts/merge_sft_datasets.py --input templates=data/sft \
      --input hard_opus=data/sft_hard_opus*2 --output-dir data/sft_merged
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation")


def parse_input(spec: str) -> tuple[str, Path, int]:
    label, _, rest = spec.partition("=")
    path, _, repeat = rest.partition("*")
    if not label or not path:
        raise SystemExit(f"expected label=dir[*N], got {spec!r}")
    return label, Path(path), int(repeat or 1)


def with_system(messages: list[dict], system: str | None) -> list[dict]:
    """system メッセージを差し替える（GEPA で進化させた指示文で学習し直すとき）。"""
    if system is None:
        return messages
    return [{"role": "system", "content": system}] + [m for m in messages if m["role"] != "system"]


def merge(
    inputs: list[tuple[str, Path, int]],
    seed: int,
    system: str | None = None,
    exclude_kinds: frozenset[str] = frozenset(),
) -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for label, directory, repeat in inputs:
        for split in SPLITS:
            rows = [
                json.loads(line)
                for line in (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            # 種別ホールドアウト: `kind` を持つ行（大規模問題）だけが対象。雛形の行は kind を持たず常に残る。
            rows = [r for r in rows if r.get("kind") not in exclude_kinds]
            times = repeat if split == "train" else 1
            for row in rows * times:
                merged[split].append(
                    {
                        "messages": with_system(row["messages"], system),
                        "tools": row.get("tools"),
                        "source": label,
                    }
                )
    # Why not 出典ごとの連結のまま: Trainer は既定で shuffle するが、検証側の先頭 N 件だけを
    # 使う --max-eval-samples が片方の出典に偏るので、ここで並びを混ぜておく。
    rng = random.Random(seed)
    for rows in merged.values():
        rng.shuffle(rows)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="label=dir[*N]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--system-file", type=Path, help="全行の system をこの指示文に差し替える")
    parser.add_argument(
        "--exclude-kinds", default="", help="学習から外す大規模問題の種別（カンマ区切り、汎化の測定用）"
    )
    args = parser.parse_args()
    exclude = frozenset(k for k in args.exclude_kinds.split(",") if k)

    inputs = [parse_input(spec) for spec in args.input]
    system = args.system_file.read_text(encoding="utf-8") if args.system_file else None
    merged = merge(inputs, args.seed, system, exclude)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {
        "inputs": [f"{l}={p}*{n}" for l, p, n in inputs],
        "system_file": str(args.system_file) if args.system_file else None,
        "exclude_kinds": sorted(exclude),
        "splits": {},
    }
    for split, rows in merged.items():
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        stats["splits"][split] = dict(Counter(r["source"] for r in rows))
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

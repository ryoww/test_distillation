#!/usr/bin/env python3
"""大規模 28 問を元に、種別ごとに新しい instance を作る（参照解はまだ付けない）。

出力は `<output-dir>/instances/prob_4XXX.json`。各 record は元問題の文章・要件を引き継ぎ、
instance だけを差し替える。reference_solution は空で、`reference_meta.status = "pending"`。
参照解は scripts/reference_hard_instances.py が教師コードを走らせて付ける。

  uv run python scripts/generate_hard_instances.py --per-kind 40 --seed 20261002
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.datagen.pipeline import shape_signature
from src.hardgen import GENERATORS
from src.utils.hard import find_kind

START_ID = 4001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, default=BASE_DIR / "data" / "problems_hard")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "problems_hard_gen")
    parser.add_argument("--per-kind", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--splits",
        default="train:30,validation:4,test:6",
        help="種別ごとの分割。合計が --per-kind と一致すること",
    )
    parser.add_argument("--kinds", default="all", help="種別名のカンマ区切り")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_bases(problem_dir: Path) -> dict[str, list[dict]]:
    """種別ごとの元 record。同じ種別が data_hard と data_hard2 にあれば両方使う。"""
    bases: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(problem_dir.glob("prob_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        kind = find_kind(record["instance"])
        if kind is not None:
            bases[kind.name].append(record)
    return bases


def parse_splits(spec: str, per_kind: int) -> list[str]:
    labels: list[str] = []
    for part in spec.split(","):
        name, count = part.split(":")
        labels.extend([name] * int(count))
    if len(labels) != per_kind:
        raise SystemExit(f"splits {spec} sum to {len(labels)}, expected {per_kind}")
    return labels


def main() -> None:
    args = parse_args()
    bases = load_bases(args.problem_dir)
    kinds = sorted(GENERATORS) if args.kinds == "all" else args.kinds.split(",")
    missing = [k for k in kinds if k not in GENERATORS or k not in bases]
    if missing:
        raise SystemExit(f"no generator or base for kinds: {missing}")
    out_dir = args.output_dir / "instances"
    if out_dir.exists() and any(out_dir.glob("prob_*.json")):
        if not args.force:
            raise SystemExit(f"{out_dir} already holds instances; pass --force to replace them")
        for stale in out_dir.glob("prob_*.json"):
            stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_splits(args.splits, args.per_kind)

    next_id = START_ID
    digests = []
    summary: dict[str, int] = {}
    for kind in kinds:
        generate = GENERATORS[kind]
        kind_bases = bases[kind]
        for k in range(args.per_kind):
            # Why not 元問題を順繰りに: 同じ種別に 2 問あるとき、前半を小さい方、後半を大きい方に
            # 割り当てると分割（train/test）が規模で偏る。k を元問題数で割った余りで交互に使う。
            base = kind_bases[k % len(kind_bases)]
            seed_label = f"{args.seed}:{kind}:{base['id']}:{k}"
            rng = random.Random(seed_label)
            instance = generate(rng, base["instance"])
            if shape_signature(instance) != shape_signature(base["instance"]):
                raise SystemExit(f"{kind} #{k}: instance shape differs from prob_{base['id']}")
            detected = find_kind(instance)
            if detected is None or detected.name != kind:
                raise SystemExit(f"{kind} #{k}: generated instance is detected as {detected}")
            record = {
                "id": next_id,
                "name": base["name"],
                "domain": base["domain"],
                "math_type": base["math_type"],
                "difficulty": base.get("difficulty", ""),
                "split": labels[k],
                "description": base["description"],
                "requirements": base["requirements"],
                "instance": instance,
                "reference_solution": {},
                "reference_meta": {"status": "pending"},
                "provenance": {
                    "generator": "src.hardgen",
                    "kind": kind,
                    "base_id": base["id"],
                    "seed": seed_label,
                },
            }
            text = json.dumps(record, ensure_ascii=False) + "\n"
            path = out_dir / f"prob_{next_id}.json"
            path.write_text(text, encoding="utf-8")
            digests.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {path.name}")
            next_id += 1
        summary[kind] = args.per_kind
        print(f"{kind}: {args.per_kind} instances from base(s) {[b['id'] for b in kind_bases]}")
    manifest = {
        "seed": args.seed,
        "per_kind": args.per_kind,
        "splits": args.splits,
        "kinds": summary,
        "files": next_id - START_ID,
        "sha256": hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest(),
    }
    (args.output_dir / "instances_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {next_id - START_ID} instances to {out_dir}")


if __name__ == "__main__":
    main()

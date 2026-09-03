#!/usr/bin/env python3
"""既存問題を雛形にして、検証済み参照解つきの新しい問題を書き出す。

出力先は既存の data/problems とは別のディレクトリにする。既存 100 問と
data_manifest.json は変更しない。生成した問題は同じ雛形の未知 instance なので、
GEPA の候補選択にも demo にも使っていないホールドアウトとして扱える。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.datagen import GENERATOR_VERSION, TEMPLATES, generate_dataset, load_base

DEFAULT_PROBLEM_DIR = BASE_DIR / "data" / "problems"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "problems_generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, default=DEFAULT_PROBLEM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--templates",
        default="all",
        help="雛形問題IDのカンマ区切り（例: 1,55,89）。既定は登録済み全て",
    )
    parser.add_argument("--per-template", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--start-id", type=int, default=1001)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--force",
        action="store_true",
        help="出力先に残っている prob_*.json と manifest.json を消してから書く",
    )
    parser.add_argument("--list", action="store_true", help="登録済みテンプレートを表示して終了")
    return parser.parse_args()


def resolve_templates(spec: str) -> list[int]:
    if spec == "all":
        return sorted(TEMPLATES)
    ids = [int(token) for token in spec.split(",") if token.strip()]
    unknown = [i for i in ids if i not in TEMPLATES]
    if unknown:
        raise SystemExit(f"no template for problem ids: {unknown}")
    return ids


def clear_stale_outputs(output_dir: Path, force: bool) -> None:
    """前回の生成物が混ざると load_v3_data が拾うので、残っていれば止める。"""
    stale = sorted(output_dir.glob("prob_*.json"))
    if not stale:
        return
    if not force:
        raise SystemExit(
            f"{output_dir} already holds {len(stale)} problem files; pass --force to replace them"
        )
    for path in [*stale, output_dir / "manifest.json"]:
        path.unlink(missing_ok=True)


def write_dataset(records: list[dict], output_dir: Path, args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs(output_dir, args.force)
    digests = []
    for record in records:
        path = output_dir / f"prob_{record['id']:04d}.json"
        text = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")
        digests.append(f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  {path.name}")
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": args.seed,
        "per_template": args.per_template,
        "templates": sorted({r["provenance"]["template_id"] for r in records}),
        "files": len(records),
        "split": args.split,
        "algorithm": "sha256 of sorted sha256sum lines",
        "sha256": hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.list:
        for template_id in sorted(TEMPLATES):
            base = load_base(args.problem_dir, template_id)
            print(f"prob_{template_id:03d}  {base['domain']}_{base['math_type']}  {base['name']}")
        return
    records = generate_dataset(
        args.problem_dir,
        template_ids=resolve_templates(args.templates),
        per_template=args.per_template,
        seed=args.seed,
        start_id=args.start_id,
        split=args.split,
    )
    manifest_path = write_dataset(records, args.output_dir, args)
    print(f"wrote {len(records)} problems to {args.output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

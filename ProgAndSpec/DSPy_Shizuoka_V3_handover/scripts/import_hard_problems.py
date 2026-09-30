#!/usr/bin/env python3
"""大規模問題集（data_hard / data_hard2）を prob_XXX.json の形に取り込む。

元の record は reference_solution にヒューリスティック解の目的値（best_objective）と解の
構造（solution）、下界や手法などのメタ情報を一緒に持つ。このリポジトリの採点系は
reference_solution の数値スカラーから目的値を選び、キー集合を返却スキーマとして
プロンプトに載せるので、次の形に組み替える。

- reference_solution = {"objective_value": best_objective, **solution, "note": ...}
- reference_meta = {bound, gap_percent, is_optimal, method, time_limit_sec, naive_milp_result}
- id は 300 + 元の id（data_hard は 1〜10、data_hard2 は 11〜30）。同梱 100 問と衝突させない。
- description の末尾に実行時間の上限を書き足す（生成コードの打ち切り時間と同じ値）。

zip が途中で切れていても（central directory なし）、deflate ストリームを順に読んで
復元できる分だけ取り込む。
"""

from __future__ import annotations

import argparse
import json
import struct
import zipfile
import zlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ID_OFFSET = 300
META_KEYS = ("bound", "gap_percent", "is_optimal", "method", "time_limit_sec", "naive_milp_result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zips", nargs="+", type=Path, help="data_hard.zip などの zip")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "problems_hard")
    parser.add_argument(
        "--exec-timeout", type=int, default=600, help="description に書く実行上限（秒）"
    )
    return parser.parse_args()


def read_zip_members(path: Path) -> dict[str, bytes]:
    """通常の zip はそのまま読み、壊れた zip はローカルヘッダを辿って復元する。"""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {
                name.rsplit("/", 1)[-1]: archive.read(name)
                for name in archive.namelist()
                if name.endswith(".json")
            }
    data = path.read_bytes()
    members: dict[str, bytes] = {}
    pos = 0
    while (start := data.find(b"PK\x03\x04", pos)) >= 0:
        (method,) = struct.unpack("<H", data[start + 8 : start + 10])
        name_len, extra_len = struct.unpack("<HH", data[start + 26 : start + 30])
        name = data[start + 30 : start + 30 + name_len].decode("utf-8")
        body = start + 30 + name_len + extra_len
        if method != 8:
            pos = body + 1
            continue
        stream = zlib.decompressobj(-15)
        payload = stream.decompress(data[body:])
        if not stream.eof:
            print(f"{path.name}: {name} is cut off; skipped")
            break
        if name.endswith(".json"):
            members[name.rsplit("/", 1)[-1]] = payload
        pos = len(data) - len(stream.unused_data)
    return members


def convert(record: dict, exec_timeout: int) -> dict:
    reference = record["reference_solution"]
    solution = reference.get("solution") or {}
    if not isinstance(solution, dict):
        raise TypeError(f"record {record.get('id')} has no solution dict")
    if not isinstance(reference.get("best_objective"), (int, float)):
        raise TypeError(f"record {record.get('id')} has no numeric best_objective")
    converted = {
        "id": ID_OFFSET + int(record["id"]),
        "name": record["name"],
        "domain": record["domain"],
        "math_type": record["math_type"],
        "difficulty": record.get("difficulty", ""),
        "split": "test",
        "description": (
            f"{record['description'].rstrip()} "
            f"solve() の実行時間は {exec_timeout} 秒以内に収めること（超えた場合は失敗扱い）。"
        ),
        "requirements": record["requirements"],
        "instance": record["instance"],
        "reference_solution": {
            "objective_value": reference["best_objective"],
            **{k: v for k, v in solution.items() if k != "note"},
            "note": "ヒューリスティック解（最適性は未証明）。objective_value がその目的値。",
        },
        "reference_meta": {k: reference.get(k) for k in META_KEYS},
        "provenance": {"source": "hard", "original_id": record["id"]},
    }
    return converted


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for zip_path in args.zips:
        for name, payload in sorted(read_zip_members(zip_path).items()):
            record = convert(json.loads(payload), args.exec_timeout)
            out = args.output_dir / f"prob_{record['id']:03d}.json"
            out.write_text(
                json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
            print(f"{zip_path.name}/{name} -> {out.name}  {record['name']}")
            written += 1
    print(f"wrote {written} problems to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""大規模生成問題（data/problems_hard_gen）で、モデルが書いて正解した solve() を SFT データにする。

「正解」は、検証器が解を読めて（verified）、違反 0 で、再計算した目的値が参照解（教師コードの最良）から
`--max-gap` 以内のもの。申告した目的値のずれも違反に数える（学習させたくないので救済しない）。

解答の出典は 2 種類を受ける。
- `--answers <label>=<dir>`: `<dir>/<prob_id>.py` の形で 1 問 1 ファイル（Opus のサブエージェントなど）
- `--runs <label>=<run_dir>`: compare / evaluate_solver_model の shard 形式（GLM など、API 経由の生成）

手順:
1. 各解答を自分の問題で実行・検証する（origin=own）。
2. `--replay` のとき、正解したコードを同じ種別の他の train instance でも実行し、正解したものを加える
   （origin=replay）。コードは種別で汎用に書かれているので、問題文だけ違う学習対が増える。
3. messages 形式（system=既定の指示文、user=参照値なしの問題文、assistant=コード）で書き出す。

  uv run python scripts/build_hard_sft_dataset.py --answers opus=outputs/opus_hard/answers --replay
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import convert_to_dspy_example
from src.modules import AlgorithmGenerator
from src.utils.feasibility import check_feasibility_detailed
from src.utils.safe_exec import safe_run

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"
GEN_DIR = BASE_DIR / "data" / "problems_hard_gen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answers", action="append", default=[], help="label=dir（1 問 1 ファイル）"
    )
    parser.add_argument("--runs", action="append", default=[], help="label=run_dir（shard 形式）")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "sft_hard")
    parser.add_argument(
        "--max-gap", type=float, default=0.10, help="参照解に対する許容 gap（0.10 = +10%%）"
    )
    parser.add_argument(
        "--replay", action="store_true", help="正解コードを同種別の他 instance でも検証"
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--instruction-file", type=Path, help="system に入れる指示文（省略時は既定の指示文）"
    )
    return parser.parse_args()


def _labelled(specs: list[str]) -> list[tuple[str, Path]]:
    pairs = []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not label or not path:
            raise SystemExit(f"expected label=path, got {spec!r}")
        pairs.append((label, Path(path)))
    return pairs


def load_records() -> dict[str, dict]:
    records = {}
    for split in ("train", "validation"):
        for path in sorted((GEN_DIR / split).glob("prob_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            records[f"prob_{record['id']}"] = record
    return records


def collect_answers(args: argparse.Namespace, records: dict[str, dict]) -> list[dict]:
    """(出典, 問題, コード) を集める。対象外の問題（test など）は捨てる。"""
    answers = []
    for label, directory in _labelled(args.answers):
        for path in sorted(directory.glob("prob_*.py")):
            if path.stem in records:
                answers.append(
                    {"source": label, "instance_id": path.stem, "code": path.read_text()}
                )
    for label, run_dir in _labelled(args.runs):
        for shard in sorted(run_dir.glob(f"*/{RESULT_FILENAME}")):
            for row in json.loads(shard.read_text(encoding="utf-8"))["test"]["results"]:
                if row.get("code") and row["instance_id"] in records:
                    answers.append(
                        {"source": label, "instance_id": row["instance_id"], "code": row["code"]}
                    )
    return answers


def _judge(job: tuple[str, str, str, float, float]) -> tuple[str, str, dict]:
    """1 対を実行して「正解」かどうかを返す。子プロセスで動くので素の値だけ受け渡す。"""
    record_path, code, key, timeout, max_gap = job
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    ok, result = safe_run(code, record["instance"], timeout=timeout)
    if not ok:
        return record_path, key, {"correct": False, "why": f"exec: {str(result)[:160]}"}
    core_type = f"{record['domain']}_{record['math_type']}"
    checked = check_feasibility_detailed(core_type, record["instance"], result)
    cost = checked.get("cost")
    if not checked.get("verified"):
        return record_path, key, {"correct": False, "why": "unverified shape"}
    if not checked["feasible"] or checked["violation_count"] or cost is None:
        first = (checked.get("violations") or ["?"])[0]
        return record_path, key, {"correct": False, "why": f"violation: {str(first)[:160]}"}
    reference = record["reference_solution"]["objective_value"]
    # 目的値は最小化の向きで再計算されている（検証器の規約）。gap は参照比の相対差。
    gap = (cost - reference) / max(abs(reference), 1e-9)
    if gap > max_gap:
        return record_path, key, {"correct": False, "why": f"gap {gap:+.3f}", "gap": gap}
    return record_path, key, {"correct": True, "gap": gap, "cost": cost}


def _run(jobs: list[tuple], workers: int) -> list[tuple[str, str, dict]]:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 50 == 0:
                print(f"  {index}/{len(jobs)} judged", flush=True)
    return results


def to_messages(record: dict, code: str, instruction: str, meta: dict) -> dict:
    prompt = convert_to_dspy_example(record, use_reference=False)["requirement"]
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```python\n{code.strip()}\n```"},
        ],
        "tools": None,
        "instance_id": f"prob_{record['id']}",
        "kind": record["provenance"]["kind"],
        **meta,
    }


def main() -> None:
    args = parse_args()
    records = load_records()
    path_of = {iid: str(GEN_DIR / r["split"] / f"{iid}.json") for iid, r in records.items()}
    answers = collect_answers(args, records)
    print(f"{len(answers)} answers from {sorted({a['source'] for a in answers})}")

    code_by_key = {f"{a['source']}::{a['instance_id']}": a for a in answers}
    own = _run(
        [
            (path_of[a["instance_id"]], a["code"], key, args.timeout, args.max_gap)
            for key, a in code_by_key.items()
        ],
        args.workers,
    )
    pairs: list[tuple[str, str, str, dict]] = []  # (instance_id, key, origin, verdict)
    own_verdicts = Counter()
    for record_path, key, verdict in own:
        iid = Path(record_path).stem
        own_verdicts[(code_by_key[key]["source"], verdict["correct"])] += 1
        if verdict["correct"]:
            pairs.append((iid, key, "own", verdict))
    print("own answers:", dict(own_verdicts))

    if args.replay:
        correct_keys = {key for _, key, _, _ in pairs}
        by_kind: dict[str, list[str]] = defaultdict(list)
        for iid, record in records.items():
            by_kind[record["provenance"]["kind"]].append(iid)
        # Why not 正解キーをすべて再生: 解答者は同じ種別の 5 問に同じソルバーを保存することが多く、
        # そのまま再生すると同一コードを同じ instance で何度も走らせ、同じ学習対が重複する。
        # 本文が同じコードは 1 本だけ再生し、すでに自分の問題で正解した instance には走らせない。
        solved_by_code: dict[tuple[str, str], set[str]] = defaultdict(set)
        representative: dict[tuple[str, str], str] = {}
        for iid, key, _, _ in pairs:
            answer = code_by_key[key]
            code_id = (answer["source"], answer["code"].strip())
            solved_by_code[code_id].add(iid)
            representative.setdefault(code_id, key)
        jobs = []
        for code_id, key in sorted(representative.items(), key=lambda item: item[1]):
            answer = code_by_key[key]
            kind = records[answer["instance_id"]]["provenance"]["kind"]
            for iid in by_kind[kind]:
                if iid not in solved_by_code[code_id]:
                    jobs.append((path_of[iid], answer["code"], key, args.timeout, args.max_gap))
        print(
            f"replaying {len(representative)} distinct correct codes "
            f"(of {len(correct_keys)}) on {len(jobs)} other instances"
        )
        replay_verdicts = Counter()
        for record_path, key, verdict in _run(jobs, args.workers):
            replay_verdicts[verdict["correct"]] += 1
            if verdict["correct"]:
                pairs.append((Path(record_path).stem, key, "replay", verdict))
        print("replay:", dict(replay_verdicts))

    instruction = (
        args.instruction_file.read_text(encoding="utf-8")
        if args.instruction_file
        else AlgorithmGenerator().generate.predict.signature.instructions
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"max_gap": args.max_gap, "replay": args.replay, "splits": {}}
    for split in ("train", "validation"):
        rows = []
        for iid, key, origin, verdict in pairs:
            record = records[iid]
            if record["split"] != split:
                continue
            answer = code_by_key[key]
            meta = {
                "source": answer["source"],
                "origin": origin,
                "written_for": answer["instance_id"],
                "gap": round(verdict["gap"], 6),
            }
            rows.append(to_messages(record, answer["code"], instruction, meta))
        rows.sort(key=lambda r: (r["instance_id"], r["source"], r["written_for"]))
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        stats["splits"][split] = {
            "pairs": len(rows),
            "instances": len({r["instance_id"] for r in rows}),
            "by_kind": dict(Counter(r["kind"] for r in rows)),
            "by_source_origin": dict(Counter(f"{r['source']}/{r['origin']}" for r in rows)),
        }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {s: {k: v for k, v in d.items() if k != "by_kind"} for s, d in stats["splits"].items()}
        )
    )


if __name__ == "__main__":
    main()

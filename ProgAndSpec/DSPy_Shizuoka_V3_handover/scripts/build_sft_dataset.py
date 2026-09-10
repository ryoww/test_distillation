#!/usr/bin/env python3
"""この問題集に特化した小型モデルの SFT データを、検証済みの (問題文, コード) 対として作る。

手順:
1. 保存済み評価結果から exact_match だった solve() コードを雛形ごとに集める（Fable、Qwen 各条件）。
2. 雛形生成器で学習用・検証用・テスト用の新しい instance を作る（seed を分ける）。
3. 各 instance に対して候補コードを実際に実行し、exact_match になった対だけを残す。
   instance に依存したコードや、たまたま当たっていたコードはここで落ちる。
4. messages 形式（system=既定の指示文、user=参照値なしの requirement、assistant=コード）で書き出す。

出力は data/sft/ 以下（生成物なのでコミットしない）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import best_known as _best_known
from src.data_loader import convert_to_dspy_example
from src.datagen import TEMPLATES, generate_dataset
from src.metrics_v3 import evaluate_algorithm_v3
from src.modules import AlgorithmGenerator, ensure_parse_helpers

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"
COMPARISONS = BASE_DIR / "outputs" / "prompt_model_comparisons"
PROBLEM_DIR = BASE_DIR / "data" / "problems"
GENERATED_DIR = BASE_DIR / "data" / "problems_generated"

# 正解コードの出典。先に並ぶ出典を優先して雛形ごとの候補を選ぶ。生成 140 問への解のあとに、
# 同梱 100 問への解を並べる。同梱問題は自分自身が雛形なので、雛形化した問題の教師になる。
SOURCES: list[tuple[str, str, str]] = [
    ("fable", "rescored-final-fable-20260907.json", "generated140-fable-20260907"),
    (
        "qwen36-compact",
        "rescored-final-qwen36-20260907.json",
        "generated140-variants-20260906-qwen3_6_27b-compact",
    ),
    ("qwen36-before", "rescored-final-qwen36-20260907.json", "generated140-20260904"),
    ("qwen36-after", "rescored-final-qwen36-20260907.json", "generated140-20260904"),
    (
        "qwen36-before_demos",
        "rescored-final-qwen36-20260907.json",
        "generated140-variants-20260906-qwen3_6_27b-before_demos",
    ),
    (
        "qwen36-modular",
        "rescored-final-qwen36-20260907.json",
        "generated140-variants-20260906-qwen3_6_27b-modular",
    ),
    (
        "qwen38-compact",
        "rescored-final-qwen38-20260907.json",
        "generated140-variants-20260906-qwen3_8_27b-compact",
    ),
    (
        "qwen38-after",
        "rescored-final-qwen38-20260907.json",
        "generated140-qwen38-max64k-20260905-after",
    ),
    (
        "qwen38-before",
        "rescored-final-qwen38-20260907.json",
        "generated140-qwen38-max64k-20260905-before",
    ),
    (
        "shipped-qwen36-before",
        "rescored-shipped100-20260908.json",
        "prompt-model-qwen36-mtp131k-20260901",
    ),
    (
        "shipped-qwen36-after",
        "rescored-shipped100-20260908.json",
        "prompt-model-qwen36-mtp131k-20260901",
    ),
    ("shipped-gemma4", "rescored-candidates-shipped-20260909.json", "candidates-20260908-problems"),
    (
        "shipped-qwen38-after",
        "rescored-shipped100-20260908.json",
        "prompt-model-qwen38-mtp131k-20260901",
    ),
    (
        "shipped-ministral3",
        "rescored-candidates-shipped-20260909.json",
        "candidates-20260908-problems",
    ),
    (
        "shipped-qwen38-before",
        "rescored-shipped100-20260908.json",
        "prompt-model-qwen38-mtp131k-20260901",
    ),
]
CONDITION_OF = {
    "fable": "fable__claude",
    "qwen36-compact": "compact__qwen3_6_27b",
    "qwen36-before": "before__qwen3_6_27b",
    "qwen36-after": "after__qwen3_6_27b",
    "qwen36-before_demos": "before_demos__qwen3_6_27b",
    "qwen36-modular": "modular__qwen3_6_27b",
    "qwen38-compact": "compact__qwen3_8_27b",
    "qwen38-after": "after__qwen3_8_27b",
    "qwen38-before": "before__qwen3_8_27b",
    "shipped-qwen36-before": "before__qwen3_6_27b",
    "shipped-qwen36-after": "after__qwen3_6_27b",
    "shipped-gemma4": "gemma4_12b_nothink",
    "shipped-qwen38-after": "after__qwen3_8_27b",
    "shipped-ministral3": "ministral3_14b_reasoning",
    "shipped-qwen38-before": "before__qwen3_8_27b",
}
# 同梱の参照解が近似解の問題では、正しいコードが beat_reference になる。新 instance の厳密解で
# 再検証するので、候補集めの段階では両方を通す。
TEACHER_STATUSES = {"exact_match", "beat_reference"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "sft")
    parser.add_argument("--codes-per-template", type=int, default=6)
    parser.add_argument("--train-seed", type=int, default=90001)
    parser.add_argument("--train-per-template", type=int, default=40)
    parser.add_argument("--val-seed", type=int, default=90002)
    parser.add_argument("--val-per-template", type=int, default=4)
    parser.add_argument("--test-seed", type=int, default=90003)
    parser.add_argument("--test-per-template", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--templates", default="all", help="雛形IDのカンマ区切り。既定は全部")
    return parser.parse_args()


def template_of_generated() -> dict[str, int]:
    """instance_id → 雛形 id。生成問題は provenance から、同梱問題は自分の番号。"""
    mapping = {f"prob_{template_id:03d}": template_id for template_id in TEMPLATES}
    for path in GENERATED_DIR.glob("prob_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        mapping[f"prob_{record['id']}"] = record["provenance"]["template_id"]
    return mapping


def collect_candidates(codes_per_template: int) -> dict[int, list[dict]]:
    """雛形ごとに、出典の優先順で重複のない exact_match コードを最大 N 件集める。"""
    template_of = template_of_generated()
    verdict: dict[tuple[str, str], str] = {}
    for _, rescored, _ in SOURCES:
        for condition, records in json.loads((COMPARISONS / rescored).read_text(encoding="utf-8"))[
            "records"
        ].items():
            for row in records:
                verdict[(condition, row["instance_id"])] = row["status"]
    pools: dict[int, list[dict]] = defaultdict(list)
    seen: dict[int, set[str]] = defaultdict(set)
    for source, _, run_dir in SOURCES:
        condition = CONDITION_OF[source]
        for shard in sorted((COMPARISONS / run_dir).glob(f"{condition}__shard*/{RESULT_FILENAME}")):
            for row in json.loads(shard.read_text(encoding="utf-8"))["test"]["results"]:
                code = row.get("code")
                if not code or verdict.get((condition, row["instance_id"])) not in TEACHER_STATUSES:
                    continue
                template_id = template_of.get(row["instance_id"])
                if template_id is None:  # 雛形化していない同梱問題
                    continue
                normalized = ensure_parse_helpers(code.strip())
                if normalized in seen[template_id]:
                    continue
                seen[template_id].add(normalized)
                pools[template_id].append(
                    {"code": normalized, "source": source, "origin": row["instance_id"]}
                )
    # Why not 出典を混ぜてから切る: 強いモデルの解を優先しつつ、同じ雛形で複数の書き方を残す。
    rng = random.Random(0)
    selected: dict[int, list[dict]] = {}
    for template_id, pool in pools.items():
        by_source: dict[str, list[dict]] = defaultdict(list)
        for entry in pool:
            by_source[entry["source"]].append(entry)
        picked: list[dict] = []
        for source, _, _ in SOURCES:
            candidates = by_source.get(source, [])
            rng.shuffle(candidates)
            picked.extend(candidates[: max(1, codes_per_template // 3)])
            if len(picked) >= codes_per_template:
                break
        selected[template_id] = picked[:codes_per_template]
    return selected


def _verify(args: tuple[dict, dict, float]) -> tuple[str, str, dict]:
    """1 対を採点する。子プロセスで動くので引数と戻り値は素の dict にする。"""
    example, candidate, timeout = args
    registry = _best_known.BestKnownRegistry()
    if example.get("reference_value") is not None:
        registry.register(example["instance_id"], example["reference_value"])
    result = evaluate_algorithm_v3(
        code=candidate["code"],
        instance=example["instance"],
        core_type=example["core_type"],
        instance_id=example["instance_id"],
        registry=registry,
        timeout=timeout,
        reference_value=example.get("reference_value"),
        reference_solution=example.get("reference_solution", {}),
        objective_text=example.get("objective", ""),
        use_reference=True,
    )
    return (
        example["instance_id"],
        candidate["code"],
        {"status": result["status"], "score": result["score"]},
    )


def _generate_one_template(args: tuple[int, int, int, int, str]) -> list[dict]:
    template_id, per_template, seed, start_id, split = args
    return generate_dataset(
        PROBLEM_DIR,
        template_ids=[template_id],
        per_template=per_template,
        seed=seed,
        start_id=start_id,
        split=split,
    )


def fresh_split(
    name: str,
    seed: int,
    per_template: int,
    template_ids: list[int],
    start_id: int,
    out_dir: Path,
    workers: int = 8,
) -> list[dict]:
    """雛形生成器で新しい問題集を作り、problem ディレクトリとしても書き出す。

    雛形ごとに別プロセスで生成する。id は generate_dataset を一括で呼んだときと同じ並びになる。
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = pool.map(
            _generate_one_template,
            [
                (template_id, per_template, seed, start_id + index * per_template, name)
                for index, template_id in enumerate(template_ids)
            ],
        )
        records = [record for chunk in chunks for record in chunk]
    problems_dir = out_dir / f"problems_{name}"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        (problems_dir / f"prob_{record['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    (problems_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generator_version": 1,
                "seed": seed,
                "per_template": per_template,
                "files": len(records),
                "split": name,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def build_pairs(
    records: list[dict], selected: dict[int, list[dict]], workers: int, timeout: float = 60.0
) -> tuple[list[dict], Counter]:
    """全 (instance, 候補コード) 対を採点し、exact_match の対だけを返す。"""
    examples = {f"prob_{r['id']}": convert_to_dspy_example(r, use_reference=True) for r in records}
    template_of = {f"prob_{r['id']}": r["provenance"]["template_id"] for r in records}
    jobs = []
    for instance_id, example in examples.items():
        for candidate in selected.get(template_of[instance_id], []):
            jobs.append((example, candidate, timeout))
    counts: Counter = Counter()
    kept: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_verify, job) for job in jobs]
        for future in as_completed(futures):
            instance_id, code, result = future.result()
            counts[result["status"]] += 1
            if result["status"] in ("exact_match", "beat_reference"):
                kept.append(
                    {
                        "instance_id": instance_id,
                        "template_id": template_of[instance_id],
                        "code": code,
                    }
                )
    return kept, counts


def to_messages(example: dict, code: str, instruction: str) -> dict:
    prompt = convert_to_dspy_example(example, use_reference=False)["requirement"]
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```python\n{code.strip()}\n```"},
        ],
        "tools": None,
        "instance_id": f"prob_{example['id']}",
        "template_id": example["provenance"]["template_id"],
    }


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    selected = collect_candidates(args.codes_per_template)
    # Why not 候補のある雛形だけ: 教師コードのない雛形もテスト集合には入れて、解けないことを測れるようにする。
    template_ids = (
        sorted(TEMPLATES)
        if args.templates == "all"
        else [int(t) for t in args.templates.split(",")]
    )
    without_teacher = [t for t in template_ids if not selected.get(t)]
    instruction = AlgorithmGenerator().generate.predict.signature.instructions
    print(
        f"candidates: {sum(len(v) for v in selected.values())} codes over {len(selected)} templates"
    )

    if without_teacher:
        print(f"templates without teacher code: {without_teacher}")
    stats: dict = {
        "candidates_per_template": {t: len(selected.get(t, [])) for t in template_ids},
        "templates_without_teacher": without_teacher,
    }
    for name, seed, per_template, start in (
        ("train", args.train_seed, args.train_per_template, 20001),
        ("validation", args.val_seed, args.val_per_template, 30001),
        ("test", args.test_seed, args.test_per_template, 40001),
    ):
        records = fresh_split(name, seed, per_template, template_ids, start, out, args.workers)
        by_id = {r["id"]: r for r in records}
        kept, counts = build_pairs(records, selected, args.workers, args.timeout)
        rows = [
            to_messages(by_id[int(k["instance_id"].split("_")[1])], k["code"], instruction)
            for k in kept
        ]
        rows.sort(key=lambda r: (r["instance_id"], r["messages"][2]["content"]))
        path = out / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        covered = len({r["instance_id"] for r in rows})
        stats[name] = {
            "instances": len(records),
            "pairs_checked": sum(counts.values()),
            "verdicts": dict(counts),
            "pairs_kept": len(rows),
            "instances_with_a_solution": covered,
        }
        print(
            f"{name}: {len(records)} instances, {sum(counts.values())} pairs checked, {len(rows)} kept, {covered} instances covered -> {path}"
        )
    (out / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

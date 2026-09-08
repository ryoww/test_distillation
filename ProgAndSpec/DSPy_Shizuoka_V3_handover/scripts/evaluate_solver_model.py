#!/usr/bin/env python3
"""OpenAI 互換 API で配信中のモデルに solve() を書かせ、既存の採点系で採点する。

DSPy を通さず、SFT データと同じ messages（system=既定の指示文、user=参照値なしの requirement）
を送る。特化学習したモデルを学習時と同じ入力形式で評価するためのもの。結果は
compare_prompt_models.py の子評価と同じ shard 形式で書くので、rescore_with_checkers.py で
再採点できる。
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import best_known as _best_known
from src.data_loader import convert_to_dspy_example, load_v3_data
from src.metrics_v3 import evaluate_algorithm_v3
from src.modules import AlgorithmGenerator, ensure_parse_helpers, strip_code_fence

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:7501/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--label", required=True, help="condition label, e.g. sft__agents_a1_4b")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs" / "prompt_model_comparisons"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, help="先頭 N 問だけ（動作確認用）")
    parser.add_argument(
        "--system-prefix-file",
        type=Path,
        help="system 指示文の前に置くテキスト（例: Ministral Reasoning が思考を開くための推奨 system prompt）",
    )
    parser.add_argument(
        "--exclude-templated",
        action="store_true",
        help="雛形化済み（SFT の教師データに含まれる種別）の問題を除く。汎化の測定用",
    )
    parser.add_argument(
        "--extra-body", default="{}", help="chat completions に足す JSON（例: thinking の無効化）"
    )
    return parser.parse_args()


def chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
    extra: dict,
) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **extra,
    }
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    message = body["choices"][0]["message"]
    return message.get("content") or "", body.get("usage", {})


def solve_one(args: argparse.Namespace, instruction: str, example: dict) -> dict:
    prompt = convert_to_dspy_example(example["record"], use_reference=False)["requirement"]
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
    started = time.monotonic()
    try:
        content, usage = chat(
            args.api_base,
            args.api_key,
            args.model,
            messages,
            args.max_tokens,
            args.temperature,
            args.timeout,
            json.loads(args.extra_body),
        )
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            # Why not gen_error として記録: 4xx は設定ミス（context 超過、モデル名違い）で、
            # 140 問ぶん同じ失敗を保存しても意味がない。すぐ止めて直させる。
            raise RuntimeError(f"HTTP {exc.code} from the server: {exc.read()[:300]!r}") from exc
        return {
            "instance_id": example["instance_id"],
            "code": None,
            "status": "gen_error",
            "score": -0.5,
            "detail": f"request failed: {exc}",
            "elapsed": time.monotonic() - started,
        }
    except (
        urllib.error.URLError,
        OSError,
        http.client.HTTPException,
        TimeoutError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "instance_id": example["instance_id"],
            "code": None,
            "status": "gen_error",
            "score": -0.5,
            "detail": f"request failed: {exc}",
            "elapsed": time.monotonic() - started,
        }
    code = ensure_parse_helpers(strip_code_fence(content))
    if "def solve" not in code:
        return {
            "instance_id": example["instance_id"],
            "code": code,
            "status": "gen_error",
            "score": -0.5,
            "detail": "no solve() in response",
            "usage": usage,
            "elapsed": time.monotonic() - started,
        }
    registry = _best_known.BestKnownRegistry()
    if example.get("reference_value") is not None:
        registry.register(example["instance_id"], example["reference_value"])
    result = evaluate_algorithm_v3(
        code=code,
        instance=example["instance"],
        core_type=example["core_type"],
        instance_id=example["instance_id"],
        registry=registry,
        timeout=60.0,
        reference_value=example.get("reference_value"),
        reference_solution=example.get("reference_solution", {}),
        objective_text=example.get("objective", ""),
        use_reference=True,
    )
    return {
        "instance_id": example["instance_id"],
        "name": example.get("name", ""),
        "core_type": example["core_type"],
        "code": code,
        "status": result["status"],
        "score": result["score"],
        "cost": result.get("cost"),
        "reference_value": example.get("reference_value"),
        "detail": result.get("detail", ""),
        "feasibility_verified": result.get("feasibility_verified"),
        "usage": usage,
        "elapsed": round(time.monotonic() - started, 2),
    }


def main() -> int:
    args = parse_args()
    records = load_v3_data(str(args.data_dir))
    examples = []
    for record in records:
        example = convert_to_dspy_example(record, use_reference=True)
        example["record"] = record
        examples.append(example)
    examples.sort(key=lambda e: e["instance_id"])
    if args.exclude_templated:
        from src.datagen import TEMPLATES

        templated = {f"prob_{k:03d}" for k in TEMPLATES}
        examples = [ex for ex in examples if ex["instance_id"] not in templated]
        print(f"excluded templated problems; {len(examples)} remain", file=sys.stderr)
    if args.limit:
        examples = examples[: args.limit]
    instruction = AlgorithmGenerator().generate.predict.signature.instructions
    if args.system_prefix_file:
        # Why: Ministral 3 Reasoning は system が無いときだけ思考用の既定 system を差し込む。
        # 我々の指示文を system に置くとそれが消えるので、推奨文を前置して思考の条件を保つ。
        instruction = args.system_prefix_file.read_text().strip() + "\n\n" + instruction
    # Why not 環境変数 ROLE: 出力は compare の shard 形式に揃え、rescore と集計を共通にする。
    shard_dir = args.output_dir / args.run_name / f"{args.label}__shard01of01"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(solve_one, args, instruction, ex): ex["instance_id"] for ex in examples
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except RuntimeError as exc:
                print(f"aborting: {exc}", file=sys.stderr)
                pool.shutdown(cancel_futures=True)
                return 2
            rows.append(row)
            print(
                f"[{index}/{len(examples)}] {row['instance_id']}: {row['status']} score={row['score']:.2f} ({row.get('elapsed', 0):.0f}s)",
                flush=True,
            )
    rows.sort(key=lambda r: r["instance_id"])
    payload = {
        "test": {
            "results": rows,
            "total_count": len(rows),
            "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1),
        },
        "config": {
            "api_base": args.api_base,
            "model": args.model,
            "label": args.label,
            "data_dir": str(args.data_dir.resolve()),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "extra_body": json.loads(args.extra_body),
            "prompt": "system=default instruction, user=requirement without reference value",
        },
    }
    (shard_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(f"\n{args.label}: mean={payload['test']['mean_score']:.3f} statuses={statuses}")
    print(f"results: {shard_dir / RESULT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

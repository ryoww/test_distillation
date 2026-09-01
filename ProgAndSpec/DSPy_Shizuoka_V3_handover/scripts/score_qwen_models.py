"""Run the same DSPy evaluation for Qwen3.6 and Qwen3.8 and compare scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = BASE_DIR / "train_gepa_v3.py"
MANIFEST_PATH = BASE_DIR / "model_manifest.json"


@dataclass(frozen=True)
class ModelTarget:
    """One OpenAI-compatible local model endpoint to evaluate."""

    label: str
    model: str
    api_base: str
    revision: str | None = None


def _load_default_targets() -> tuple[ModelTarget, ModelTarget]:
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    generation = manifest["generation"]
    reflection = manifest["reflection"]
    return (
        ModelTarget(
            "qwen3_6_27b",
            generation["model_id"],
            generation["api_base"],
            generation.get("revision"),
        ),
        ModelTarget(
            "qwen3_8_27b",
            reflection["model_id"],
            reflection["api_base"],
            reflection.get("revision"),
        ),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = _load_default_targets()
    parser = argparse.ArgumentParser(
        description="Score Qwen3.6 and Qwen3.8 with one shared DSPy benchmark configuration."
    )
    parser.add_argument("--qwen36-model", default=defaults[0].model)
    parser.add_argument("--qwen36-api-base", default=defaults[0].api_base)
    parser.add_argument("--qwen36-revision", default=defaults[0].revision)
    parser.add_argument("--qwen38-model", default=defaults[1].model)
    parser.add_argument("--qwen38-api-base", default=defaults[1].api_base)
    parser.add_argument("--qwen38-revision", default=defaults[1].revision)
    parser.add_argument("--n-train", type=_nonnegative_int, default=0)
    parser.add_argument("--n-test", type=_positive_int, default=20)
    parser.add_argument("--program-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs" / "model_scores")
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=_positive_int, default=8192)
    parser.add_argument("--lm-timeout", type=_positive_int, default=1800)
    parser.add_argument(
        "--temperature",
        type=_nonnegative_float,
        default=0.0,
        help="Generation temperature shared by both models (default: 0.0).",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Use the historical Phase F reference-free program and reward mode.",
    )
    parser.add_argument(
        "--skip-connection-check",
        action="store_true",
        help="Skip the bounded endpoint probe performed by the child evaluator.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate both endpoints concurrently; use only when GPU capacity permits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print both child commands without connecting or creating output files.",
    )
    return parser.parse_args(argv)


def _program_path(args: argparse.Namespace) -> Path:
    if args.program_path is not None:
        return args.program_path.expanduser().resolve()
    name = (
        "compiled_program_v3_gepa_phaseF_noref.json"
        if args.no_reference
        else "compiled_program_v3_gepa_phaseE.json"
    )
    return BASE_DIR / name


def build_eval_command(target: ModelTarget, args: argparse.Namespace, run_root: Path) -> list[str]:
    """Build one isolated eval-only child invocation."""

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--eval-only",
        "--program-path",
        str(_program_path(args)),
        "--n-train",
        str(args.n_train),
        "--n-test",
        str(args.n_test),
        "--output-dir",
        str(run_root),
        "--run-name",
        target.label,
        "--generation-model",
        target.model,
        "--generation-api-base",
        target.api_base,
        "--reflection-model",
        target.model,
        "--reflection-api-base",
        target.api_base,
        "--max-tokens",
        str(args.max_tokens),
        "--lm-timeout",
        str(args.lm_timeout),
        "--temperature",
        str(args.temperature),
    ]
    if args.no_reference:
        command.append("--no-reference")
    if args.skip_connection_check:
        command.append("--skip-connection-check")
    return command


def _result_path(target: ModelTarget, run_root: Path, *, no_reference: bool) -> Path:
    tag = "phaseF_noref" if no_reference else "phaseE"
    return run_root / target.label / f"evaluation_results_v3_gepa_{tag}.json"


def summarize_result(
    target: ModelTarget, result_path: Path, elapsed_seconds: float
) -> dict[str, Any]:
    """Extract comparable test metrics while retaining the raw result location."""

    with result_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    test = payload["test"]
    total_count = int(test["total_count"])
    valid_count = int(test["valid_count"])
    status_counts = dict(Counter(row.get("status", "unknown") for row in test["results"]))
    return {
        "label": target.label,
        "model": target.model,
        "api_base": target.api_base,
        "expected_revision": target.revision,
        "mean_score": float(test["mean_score"]),
        "valid_count": valid_count,
        "total_count": total_count,
        "valid_rate": valid_count / total_count if total_count else 0.0,
        "beat_reference": int(test.get("beat_reference", 0)),
        "beat_reference_analysis": int(test.get("beat_reference_analysis", 0)),
        "beat_reference_strict": int(test.get("beat_reference_strict", 0)),
        "status_counts": status_counts,
        "instance_ids": [str(row.get("instance_id", "")) for row in test["results"]],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "result_path": str(result_path.resolve()),
    }


def rank_models(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by mean score and assign competition ranks, preserving score ties."""

    ranked: list[dict[str, Any]] = []
    previous_score: float | None = None
    previous_rank = 0
    ordered = sorted(summaries, key=lambda row: (-float(row["mean_score"]), row["label"]))
    for position, summary in enumerate(ordered, start=1):
        score = float(summary["mean_score"])
        rank = previous_rank if previous_score is not None and score == previous_score else position
        ranked.append({**summary, "rank": rank})
        previous_score = score
        previous_rank = rank
    return ranked


def _run_target(
    target: ModelTarget,
    args: argparse.Namespace,
    run_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> tuple[ModelTarget, int | None, float, list[str], str | None, str]:
    command = build_eval_command(target, args, run_root)
    log_path = run_root / "logs" / f"{target.label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = runner(
                command,
                cwd=BASE_DIR,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
    # The injected runner is an external boundary; unknown failures must become comparison data.
    except Exception as exc:  # noqa: BLE001
        return target, None, time.monotonic() - started, command, repr(exc), str(log_path.resolve())
    return (
        target,
        int(completed.returncode),
        time.monotonic() - started,
        command,
        None,
        str(log_path.resolve()),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_manifest_sha256() -> str | None:
    manifest_path = BASE_DIR / "data_manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as file:
            return json.load(file).get("sha256")
    except (OSError, ValueError, TypeError):
        return None


def _print_summary(rows: Sequence[dict[str, Any]]) -> None:
    print("\nModel scores")
    print(f"{'Rank':<6}{'Model':<22}{'Mean':>9}{'Valid':>12}{'Rate':>10}")
    for row in rows:
        rank = "-" if row["rank"] is None else str(row["rank"])
        valid = f"{row['valid_count']}/{row['total_count']}"
        print(
            f"{rank:<6}{row['label']:<22}{row['mean_score']:>9.3f}"
            f"{valid:>12}{row['valid_rate']:>9.1%}"
        )


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    args = parse_args(argv)
    targets = (
        ModelTarget("qwen3_6_27b", args.qwen36_model, args.qwen36_api_base, args.qwen36_revision),
        ModelTarget("qwen3_8_27b", args.qwen38_model, args.qwen38_api_base, args.qwen38_revision),
    )
    run_name = args.run_name or datetime.now().astimezone().strftime("qwen_scores_%Y%m%d_%H%M%S")
    run_root = args.output_dir.expanduser().resolve() / run_name
    program_path = _program_path(args)

    commands = [build_eval_command(target, args, run_root) for target in targets]
    if args.dry_run:
        for target, command in zip(targets, commands, strict=True):
            print(f"[{target.label}] {shlex.join(command)}")
        return 0

    if not program_path.is_file():
        print(f"Compiled program not found: {program_path}", file=sys.stderr)
        return 2
    try:
        run_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"Cannot create comparison output {run_root}: {exc}", file=sys.stderr)
        return 2
    print(f"Comparison output: {run_root}")
    if args.no_reference:
        print("Mode: reference-free (historical Phase F program)")
    else:
        print("Mode: reference-guided Phase E")

    outcomes: list[tuple[ModelTarget, int | None, float, list[str], str | None, str]] = []
    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = {
                executor.submit(_run_target, target, args, run_root, runner): target
                for target in targets
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
    else:
        for target in targets:
            print(f"Running {target.label} at {target.api_base}")
            outcomes.append(_run_target(target, args, run_root, runner))

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for target, returncode, elapsed_seconds, command, execution_error, log_path in outcomes:
        result_path = _result_path(target, run_root, no_reference=args.no_reference)
        if returncode != 0 or not result_path.is_file():
            failures.append(
                {
                    "label": target.label,
                    "model": target.model,
                    "returncode": returncode,
                    "error": execution_error or "result file was not created",
                    "log_path": log_path,
                    "result_path": str(result_path.resolve()),
                    "command": command,
                }
            )
            continue
        try:
            summary = summarize_result(target, result_path, elapsed_seconds)
            summary["log_path"] = log_path
            summaries.append(summary)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(
                {
                    "label": target.label,
                    "model": target.model,
                    "returncode": returncode,
                    "error": f"invalid result JSON: {exc}",
                    "log_path": log_path,
                    "result_path": str(result_path.resolve()),
                    "command": command,
                }
            )

    ids_by_model = {row["label"]: row["instance_ids"] for row in summaries}
    ids_match = (
        len(ids_by_model) == len(targets)
        and len({tuple(ids) for ids in ids_by_model.values()}) == 1
    )
    if len(summaries) == len(targets) and not ids_match:
        failures.append(
            {
                "label": "benchmark",
                "error": "test instance IDs differ between models",
                "instance_ids_by_model": ids_by_model,
            }
        )
    comparable = len(summaries) == len(targets) and not failures and ids_match
    ranked = rank_models(summaries) if comparable else [{**row, "rank": None} for row in summaries]
    winners = [row["label"] for row in ranked if row["rank"] == 1] if comparable else []
    common_instance_ids = next(iter(ids_by_model.values())) if comparable else None
    comparison = {
        "benchmark": {
            "run_name": run_name,
            "mode": "reference_free" if args.no_reference else "reference_guided",
            "program_path": str(program_path),
            "program_sha256": _sha256(program_path),
            "data_manifest_sha256": _data_manifest_sha256(),
            "seed": 42,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "test_instance_ids": common_instance_ids,
            "test_instance_ids_by_model": ids_by_model,
            "comparable": comparable,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "models": ranked,
        "winner": winners,
        "failures": failures,
    }
    comparison_path = run_root / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
        file.write("\n")

    if ranked:
        _print_summary(ranked)
    print(f"Comparison JSON: {comparison_path}")
    if failures:
        print(f"Failed models: {', '.join(row['label'] for row in failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

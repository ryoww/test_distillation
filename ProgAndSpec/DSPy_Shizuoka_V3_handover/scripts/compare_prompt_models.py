"""Compare prompt improvement and model choice across the complete 100-problem corpus."""

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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.score_qwen_models import ModelTarget
from src.data_loader import load_and_split_stratified, load_v3_data

TRAIN_SCRIPT = BASE_DIR / "train_gepa_v3.py"
MODEL_MANIFEST_PATH = BASE_DIR / "model_manifest.json"
DATA_MANIFEST_PATH = BASE_DIR / "data_manifest.json"
IMPROVED_PROGRAM_PATH = BASE_DIR / "compiled_program_v3_gepa_phaseE.json"
DATA_DIR = BASE_DIR / "data" / "problems"
MODEL_LABELS = ("qwen3_6_27b", "qwen3_8_27b")
EXPECTED_CONDITION_KEYS = (
    ("before", "qwen3_6_27b"),
    ("before", "qwen3_8_27b"),
    ("after", "qwen3_6_27b"),
    ("after", "qwen3_8_27b"),
)
MERGE_COMPATIBILITY_FIELDS = (
    "primary_subset",
    "seed",
    "temperature",
    "max_tokens",
    "data_manifest_sha256",
    "model_manifest_sha256",
    "serving",
    "subset_instance_ids",
    "subset_counts",
    "baseline_definition",
    "improved_definition",
    "baseline_program_sha256",
    "improved_program_sha256",
)


@dataclass(frozen=True)
class Condition:
    """One prompt and generation-model pairing in the factorial comparison."""

    label: str
    prompt_label: str
    model_target: ModelTarget
    program_path: Path


@dataclass(frozen=True)
class EvaluationJob:
    """One condition shard executed by a child evaluator."""

    label: str
    condition: Condition
    data_dir: Path
    instance_ids: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def _default_models() -> tuple[ModelTarget, ModelTarget]:
    with MODEL_MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)
    qwen36 = manifest["generation"]
    qwen38 = manifest["reflection"]
    return (
        ModelTarget("qwen3_6_27b", qwen36["model_id"], qwen36["api_base"], qwen36["revision"]),
        ModelTarget("qwen3_8_27b", qwen38["model_id"], qwen38["api_base"], qwen38["revision"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = _default_models()
    parser = argparse.ArgumentParser(
        description="Compare before/after DSPy prompts on Qwen3.6 and Qwen3.8 over 100 problems."
    )
    parser.add_argument("--qwen36-model", default=defaults[0].model)
    parser.add_argument("--qwen36-api-base", default=defaults[0].api_base)
    parser.add_argument("--qwen36-revision", default=defaults[0].revision)
    parser.add_argument("--qwen38-model", default=defaults[1].model)
    parser.add_argument("--qwen38-api-base", default=defaults[1].api_base)
    parser.add_argument("--qwen38-revision", default=defaults[1].revision)
    parser.add_argument("--improved-program-path", type=Path, default=IMPROVED_PROGRAM_PATH)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Problem directory: the shipped 100 (default) or a datagen output with manifest.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs" / "prompt_model_comparisons"
    )
    parser.add_argument("--run-name")
    parser.add_argument("--max-tokens", type=_positive_int, default=32768)
    parser.add_argument("--lm-timeout", type=_positive_int, default=1800)
    parser.add_argument("--temperature", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--only-model",
        choices=MODEL_LABELS,
        help="Evaluate both prompts for only this model so one model can occupy the GPU at a time.",
    )
    parser.add_argument(
        "--only-prompt",
        help="Evaluate only this prompt label (before, after, or an --extra-program label).",
    )
    parser.add_argument(
        "--extra-program",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Add a prompt condition from a saved DSPy program; repeatable.",
    )
    parser.add_argument(
        "--merge-runs",
        nargs="+",
        type=Path,
        metavar="RUN",
        help="Merge completed model-slice run directories without contacting a model server.",
    )
    parser.add_argument(
        "--shards",
        type=_positive_int,
        default=1,
        help="Split each condition into this many disjoint child evaluators.",
    )
    parser.add_argument("--skip-connection-check", action="store_true")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all selected prompt/shard evaluators concurrently.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected child commands without creating the baseline program or output directory.",
    )
    return parser.parse_args(argv)


def create_baseline_program(path: Path) -> Path:
    """Save the current uncompiled AlgorithmGenerator as the before-prompt artifact."""

    from src.modules import AlgorithmGenerator

    path.parent.mkdir(parents=True, exist_ok=True)
    AlgorithmGenerator().save(str(path))
    return path


def _generated_manifest(data_dir: Path) -> dict[str, Any] | None:
    """Return the datagen manifest when data_dir holds generated problems, else None."""

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    return manifest if "generator_version" in manifest else None


def split_instance_ids(data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    """Name the problem subsets whose means the comparison reports.

    Shipped 100-problem corpus: reproduce the historical 40/20 split and identify the
    never-selected 40 problems. Generated holdout (datagen manifest present): every
    problem is untouched by GEPA, so the subsets are the whole set plus one per domain.
    """

    if _generated_manifest(data_dir) is not None:
        records = load_v3_data(str(data_dir))
        all_ids = sorted(f"prob_{int(row['id']):03d}" for row in records)
        by_domain: dict[str, list[str]] = {}
        for row in records:
            by_domain.setdefault(str(row["domain"]), []).append(f"prob_{int(row['id']):03d}")
        subsets = {"all": all_ids}
        for domain in sorted(by_domain):
            subsets[f"domain_{domain}"] = sorted(by_domain[domain])
        return subsets

    train, legacy_test = load_and_split_stratified(
        str(data_dir), n_train=40, n_test=20, seed=42, use_reference=True
    )
    all_ids = sorted(f"prob_{int(row['id']):03d}" for row in load_v3_data(str(data_dir)))
    train_ids = sorted(str(row["instance_id"]) for row in train)
    legacy_test_ids = sorted(str(row["instance_id"]) for row in legacy_test)
    selected = set(train_ids) | set(legacy_test_ids)
    untouched_ids = sorted(set(all_ids) - selected)
    return {
        "train40": train_ids,
        "legacy_test20": legacy_test_ids,
        "untouched40": untouched_ids,
        "all100": all_ids,
    }


def whole_set_key(subsets: dict[str, list[str]]) -> str:
    """The subset that covers every problem: all100 for the shipped corpus, all otherwise."""

    return "all100" if "all100" in subsets else "all"


def primary_subset_key(subsets: dict[str, list[str]]) -> str:
    """The subset headline effects are read from."""

    return "untouched40" if "untouched40" in subsets else whole_set_key(subsets)


def aggregate_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one condition's raw result rows for a named problem subset."""

    scores = [float(row["score"]) for row in rows]
    valid_count = sum(score > 0 for score in scores)
    invalid_statuses = {
        "exec_error",
        "infeasible",
        "invalid_solution",
        "solver_failure",
        "suspicious_zero",
        "partial_feasible",
        "unverified",
        "gen_error",
    }
    strict_reference = 0
    for row in rows:
        cost = row.get("cost")
        reference = row.get("reference_value")
        if (
            cost is not None
            and reference not in (None, 0)
            and row.get("status") not in invalid_statuses
            and float(cost) < float(reference) * (1 - 1e-6)
        ):
            strict_reference += 1
    return {
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "valid_count": valid_count,
        "total_count": len(rows),
        "valid_rate": valid_count / len(rows) if rows else 0.0,
        "beat_reference": sum(
            row.get("status") in {"exact_match", "beat_reference"} for row in rows
        ),
        "beat_reference_analysis": sum(bool(row.get("beat_reference_analysis")) for row in rows),
        "beat_reference_strict": strict_reference,
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
        "instance_ids": [str(row.get("instance_id", "")) for row in rows],
    }


def compute_effects(condition_summaries: Sequence[dict[str, Any]], subset: str) -> dict[str, Any]:
    """Compute prompt, model, and interaction effects for subset mean scores."""

    values = {
        (row["prompt_label"], row["model_label"]): float(row["subsets"][subset]["mean_score"])
        for row in condition_summaries
    }
    before36 = values[("before", "qwen3_6_27b")]
    before38 = values[("before", "qwen3_8_27b")]
    after36 = values[("after", "qwen3_6_27b")]
    after38 = values[("after", "qwen3_8_27b")]
    prompt_effects = {
        "qwen3_6_27b": after36 - before36,
        "qwen3_8_27b": after38 - before38,
    }
    model_effects = {
        "before": before38 - before36,
        "after": after38 - after36,
    }
    return {
        "prompt_effect_by_model": prompt_effects,
        "model_effect_by_prompt": model_effects,
        "interaction": prompt_effects["qwen3_8_27b"] - prompt_effects["qwen3_6_27b"],
    }


def _condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["prompt_label"]), str(row["model_label"])


def _comparison_path(path: Path) -> Path:
    expanded = path.expanduser().resolve()
    return expanded / "factorial_comparison.json" if expanded.is_dir() else expanded


def _validate_merge_source(payload: dict[str, Any], source_path: Path) -> None:
    benchmark = payload.get("benchmark")
    conditions = payload.get("conditions")
    if not isinstance(benchmark, dict) or not isinstance(conditions, list):
        raise TypeError(f"invalid comparison artifact: {source_path}")
    if payload.get("failures"):
        raise ValueError(f"source comparison has failures: {source_path}")
    if not benchmark.get("run_complete"):
        raise ValueError(f"source comparison is incomplete: {source_path}")
    expected_subsets = benchmark.get("subset_instance_ids")
    if not isinstance(expected_subsets, dict):
        raise TypeError(f"source comparison has no subset IDs: {source_path}")
    for condition in conditions:
        subsets = condition.get("subsets", {})
        for subset, expected_ids in expected_subsets.items():
            actual_ids = subsets.get(subset, {}).get("instance_ids")
            if actual_ids != expected_ids:
                key = _condition_key(condition)
                raise ValueError(f"subset IDs differ for {key} in {source_path}")


def merge_comparison_runs(
    source_paths: Sequence[Path], run_root: Path, run_name: str
) -> dict[str, Any]:
    """Merge complete model-slice artifacts into one comparable 2x2 result."""

    if len(source_paths) < 2:
        raise ValueError("at least two source comparison runs are required")
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for source in source_paths:
        comparison_path = _comparison_path(source)
        if not comparison_path.is_file():
            raise ValueError(f"comparison artifact not found: {comparison_path}")
        with comparison_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        _validate_merge_source(payload, comparison_path)
        payloads.append((comparison_path, payload))

    reference_benchmark = payloads[0][1]["benchmark"]
    for comparison_path, payload in payloads[1:]:
        benchmark = payload["benchmark"]
        for field in MERGE_COMPATIBILITY_FIELDS:
            if benchmark.get(field) != reference_benchmark.get(field):
                raise ValueError(f"benchmark field {field!r} differs in {comparison_path}")

    condition_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for comparison_path, payload in payloads:
        for condition in payload["conditions"]:
            key = _condition_key(condition)
            if key in condition_by_key:
                raise ValueError(f"duplicate condition {key} in {comparison_path}")
            condition_by_key[key] = condition
    if set(condition_by_key) != set(EXPECTED_CONDITION_KEYS):
        missing = sorted(set(EXPECTED_CONDITION_KEYS) - set(condition_by_key))
        extra = sorted(set(condition_by_key) - set(EXPECTED_CONDITION_KEYS))
        raise ValueError(f"2x2 condition set mismatch: missing={missing}, extra={extra}")

    conditions = [condition_by_key[key] for key in EXPECTED_CONDITION_KEYS]
    subset_names = list(reference_benchmark["subset_instance_ids"])
    effects = {subset: compute_effects(conditions, subset) for subset in subset_names}
    benchmark = {
        field: reference_benchmark[field]
        for field in MERGE_COMPATIBILITY_FIELDS
        if field in reference_benchmark
    }
    source_shards = {
        payload["benchmark"]["run_name"]: payload["benchmark"].get("shards_per_condition")
        for _, payload in payloads
    }
    benchmark.update(
        {
            "run_name": run_name,
            "design": "2x2 prompt_by_model_factorial",
            "selected_models": list(MODEL_LABELS),
            "partial": False,
            "run_complete": True,
            "comparable": True,
            "source_shards_per_condition": source_shards,
            "merged_from": [str(path) for path, _ in payloads],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    comparison = {
        "benchmark": benchmark,
        "conditions": conditions,
        "effects": effects,
        "failures": [],
    }
    run_root.mkdir(parents=True, exist_ok=False)
    comparison_path = run_root / "factorial_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return comparison


def parse_extra_programs(specs: Sequence[str]) -> list[tuple[str, Path]]:
    """`LABEL=PATH` の並びを (label, path) にする。before / after は予約済み。"""
    programs: list[tuple[str, Path]] = []
    for spec in specs:
        label, sep, raw_path = spec.partition("=")
        if not sep or not label or not raw_path:
            raise ValueError(f"--extra-program expects LABEL=PATH, got {spec!r}")
        if label in ("before", "after") or any(label == seen for seen, _ in programs):
            raise ValueError(f"duplicate or reserved prompt label {label!r}")
        programs.append((label, Path(raw_path).expanduser().resolve()))
    return programs


def _conditions(
    args: argparse.Namespace, baseline_path: Path, improved_path: Path
) -> tuple[Condition, ...]:
    models = (
        ModelTarget("qwen3_6_27b", args.qwen36_model, args.qwen36_api_base, args.qwen36_revision),
        ModelTarget("qwen3_8_27b", args.qwen38_model, args.qwen38_api_base, args.qwen38_revision),
    )
    prompts = [("before", baseline_path), ("after", improved_path)]
    prompts.extend(parse_extra_programs(args.extra_program))
    return tuple(
        Condition(f"{prompt}__{model.label}", prompt, model, program)
        for prompt, program in prompts
        for model in models
    )


def _evaluation_jobs(
    conditions: Sequence[Condition],
    all_ids: Sequence[str],
    run_root: Path,
    shard_count: int,
    *,
    materialize: bool,
    source_dir: Path = DATA_DIR,
) -> tuple[EvaluationJob, ...]:
    shards = [tuple(all_ids[index::shard_count]) for index in range(shard_count)]
    shard_dirs: list[Path] = []
    for index, instance_ids in enumerate(shards, start=1):
        if shard_count == 1:
            shard_dirs.append(source_dir)
            continue
        data_dir = run_root / "shards" / f"shard_{index:02d}_of_{shard_count:02d}"
        shard_dirs.append(data_dir)
        if materialize:
            data_dir.mkdir(parents=True, exist_ok=False)
            for instance_id in instance_ids:
                (data_dir / f"{instance_id}.json").symlink_to(
                    (source_dir / f"{instance_id}.json").resolve()
                )

    jobs = []
    for condition in conditions:
        for index, (instance_ids, data_dir) in enumerate(
            zip(shards, shard_dirs, strict=True), start=1
        ):
            label = (
                condition.label
                if shard_count == 1
                else f"{condition.label}__shard{index:02d}of{shard_count:02d}"
            )
            jobs.append(EvaluationJob(label, condition, data_dir, instance_ids))
    return tuple(jobs)


def _build_command(job: EvaluationJob, args: argparse.Namespace, run_root: Path) -> list[str]:
    condition = job.condition
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--eval-only",
        "--program-path",
        str(condition.program_path),
        "--data-dir",
        str(job.data_dir),
        "--n-train",
        "0",
        "--n-test",
        str(len(job.instance_ids)),
        "--output-dir",
        str(run_root),
        "--run-name",
        job.label,
        "--generation-model",
        condition.model_target.model,
        "--generation-api-base",
        condition.model_target.api_base,
        "--reflection-model",
        condition.model_target.model,
        "--reflection-api-base",
        condition.model_target.api_base,
        "--max-tokens",
        str(args.max_tokens),
        "--lm-timeout",
        str(args.lm_timeout),
        "--temperature",
        str(args.temperature),
    ]
    if args.skip_connection_check:
        command.append("--skip-connection-check")
    return command


def _run_job(
    job: EvaluationJob,
    args: argparse.Namespace,
    run_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> dict[str, Any]:
    command = _build_command(job, args, run_root)
    log_path = run_root / "logs" / f"{job.label}.log"
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
    # The injected runner is an external boundary; every failure belongs in the result artifact.
    except Exception as exc:  # noqa: BLE001
        return {
            "job": job,
            "returncode": None,
            "elapsed_seconds": time.monotonic() - started,
            "command": command,
            "log_path": str(log_path.resolve()),
            "error": repr(exc),
        }
    return {
        "job": job,
        "returncode": int(completed.returncode),
        "elapsed_seconds": time.monotonic() - started,
        "command": command,
        "log_path": str(log_path.resolve()),
        "error": None,
    }


def _result_path(job: EvaluationJob, run_root: Path) -> Path:
    return run_root / job.label / "evaluation_results_v3_gepa_phaseE.json"


def _condition_summary(
    condition: Condition,
    result_paths: Sequence[Path],
    subsets: dict[str, list[str]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        with result_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        rows.extend(payload["test"]["results"])
    rows_by_id = {str(row["instance_id"]): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("duplicate instance IDs in child result")
    expected_ids = set(subsets[whole_set_key(subsets)])
    if set(rows_by_id) != expected_ids:
        missing = sorted(expected_ids - set(rows_by_id))
        extra = sorted(set(rows_by_id) - expected_ids)
        raise ValueError(f"problem set mismatch: missing={missing}, extra={extra}")
    aggregates = {
        name: aggregate_results([rows_by_id[instance_id] for instance_id in ids])
        for name, ids in subsets.items()
    }
    return {
        "label": condition.label,
        "prompt_label": condition.prompt_label,
        "program_path": str(condition.program_path.resolve()),
        "program_sha256": _sha256(condition.program_path),
        "model_label": condition.model_target.label,
        "model": condition.model_target.model,
        "api_base": condition.model_target.api_base,
        "expected_revision": condition.model_target.revision,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "result_paths": [str(path.resolve()) for path in result_paths],
        "subsets": aggregates,
    }


def _print_summary(summaries: Sequence[dict[str, Any]]) -> None:
    print("\nPrompt/model comparison")
    columns = [
        name
        for name in ("untouched40", "legacy_test20", "all100", "all")
        if name in summaries[0]["subsets"]
    ]
    print(f"{'Condition':<31}" + "".join(f"{name:>14}" for name in columns))
    for row in summaries:
        print(
            f"{row['label']:<31}"
            + "".join(f"{row['subsets'][name]['mean_score']:>14.3f}" for name in columns)
        )


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> int:
    args = parse_args(argv)
    run_name = args.run_name or datetime.now().astimezone().strftime("prompt_model_%Y%m%d_%H%M%S")
    run_root = args.output_dir.expanduser().resolve() / run_name
    if args.merge_runs:
        if args.dry_run or args.only_model or args.only_prompt:
            print(
                "--merge-runs cannot be combined with --dry-run, --only-model or --only-prompt",
                file=sys.stderr,
            )
            return 2
        try:
            comparison = merge_comparison_runs(args.merge_runs, run_root, run_name)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"Cannot merge comparison runs: {exc}", file=sys.stderr)
            return 2
        _print_summary(comparison["conditions"])
        print(f"Comparison JSON: {run_root / 'factorial_comparison.json'}")
        return 0

    baseline_path = run_root / "baseline_program_v3.json"
    improved_path = args.improved_program_path.expanduser().resolve()
    conditions = _conditions(args, baseline_path, improved_path)
    if args.only_model:
        conditions = tuple(
            condition for condition in conditions if condition.model_target.label == args.only_model
        )
    if args.only_prompt:
        conditions = tuple(c for c in conditions if c.prompt_label == args.only_prompt)
        if not conditions:
            print(f"Unknown prompt label: {args.only_prompt}", file=sys.stderr)
            return 2
    data_dir = args.data_dir.expanduser().resolve()
    subsets = split_instance_ids(data_dir)
    all_key = whole_set_key(subsets)

    if args.dry_run:
        jobs = _evaluation_jobs(
            conditions,
            subsets[all_key],
            run_root,
            args.shards,
            materialize=False,
            source_dir=data_dir,
        )
        for job in jobs:
            print(f"[{job.label}] {shlex.join(_build_command(job, args, run_root))}")
        return 0

    missing = [
        c.program_path
        for c in conditions
        if c.prompt_label != "before" and not c.program_path.is_file()
    ]
    if missing:
        print(f"Program not found: {missing[0]}", file=sys.stderr)
        return 2
    try:
        run_root.mkdir(parents=True, exist_ok=False)
        create_baseline_program(baseline_path)
        jobs = _evaluation_jobs(
            conditions,
            subsets[all_key],
            run_root,
            args.shards,
            materialize=True,
            source_dir=data_dir,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot initialize comparison output {run_root}: {exc}", file=sys.stderr)
        return 2

    print(f"Factorial comparison output: {run_root}")
    outcomes: list[dict[str, Any]] = []
    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {executor.submit(_run_job, job, args, run_root, runner): job for job in jobs}
            for future in as_completed(futures):
                outcomes.append(future.result())
    else:
        for job in jobs:
            print(f"Running {job.label}")
            outcomes.append(_run_job(job, args, run_root, runner))

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    outcome_by_label = {outcome["job"].label: outcome for outcome in outcomes}
    for condition in conditions:
        condition_jobs = [job for job in jobs if job.condition.label == condition.label]
        condition_outcomes = [outcome_by_label[job.label] for job in condition_jobs]
        result_paths = [_result_path(job, run_root) for job in condition_jobs]
        condition_failed = False
        for job, outcome, result_path in zip(
            condition_jobs, condition_outcomes, result_paths, strict=True
        ):
            if outcome["returncode"] == 0 and result_path.is_file():
                continue
            condition_failed = True
            failures.append(
                {
                    "label": job.label,
                    "condition_label": condition.label,
                    "returncode": outcome["returncode"],
                    "error": outcome["error"] or "result file was not created",
                    "result_path": str(result_path.resolve()),
                    "log_path": outcome["log_path"],
                    "command": outcome["command"],
                }
            )
        if condition_failed:
            continue
        try:
            summary = _condition_summary(
                condition,
                result_paths,
                subsets,
                max(float(outcome["elapsed_seconds"]) for outcome in condition_outcomes),
            )
            summary["log_paths"] = [outcome["log_path"] for outcome in condition_outcomes]
            summaries.append(summary)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(
                {
                    "label": condition.label,
                    "returncode": 0,
                    "error": f"invalid result JSON: {exc}",
                    "result_paths": [str(path.resolve()) for path in result_paths],
                    "log_paths": [outcome["log_path"] for outcome in condition_outcomes],
                    "commands": [outcome["command"] for outcome in condition_outcomes],
                }
            )

    run_complete = len(summaries) == len(conditions) and not failures
    observed_keys = {_condition_key(summary) for summary in summaries}
    comparable = run_complete and observed_keys == set(EXPECTED_CONDITION_KEYS)
    effects = (
        {subset: compute_effects(summaries, subset) for subset in subsets} if comparable else {}
    )
    generated = _generated_manifest(data_dir)
    if generated is not None:
        data_manifest_sha256 = generated.get("sha256")
    else:
        with DATA_MANIFEST_PATH.open(encoding="utf-8") as file:
            data_manifest_sha256 = json.load(file).get("sha256")
    with MODEL_MANIFEST_PATH.open(encoding="utf-8") as file:
        serving = json.load(file).get("serving", {})
    comparison = {
        "benchmark": {
            "run_name": run_name,
            "design": "2x2 prompt_by_model_factorial",
            "primary_subset": primary_subset_key(subsets),
            "data_dir": str(data_dir),
            "seed": 42,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "shards_per_condition": args.shards,
            "data_manifest_sha256": data_manifest_sha256,
            "model_manifest_sha256": _sha256(MODEL_MANIFEST_PATH),
            "serving": serving,
            "subset_instance_ids": subsets,
            "subset_counts": {name: len(ids) for name, ids in subsets.items()},
            "baseline_definition": "current uncompiled AlgorithmGenerator signature, no demos",
            "improved_definition": "saved GEPA Phase E program, including optimized instructions and demos",
            "extra_programs": {
                label: str(path) for label, path in parse_extra_programs(args.extra_program)
            },
            "baseline_program_sha256": _sha256(baseline_path),
            "improved_program_sha256": _sha256(improved_path),
            "selected_models": sorted({condition.model_target.label for condition in conditions}),
            "partial": len(conditions) < len(EXPECTED_CONDITION_KEYS),
            "run_complete": run_complete,
            "comparable": comparable,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "conditions": summaries,
        "effects": effects,
        "failures": failures,
    }
    comparison_path = run_root / "factorial_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
        file.write("\n")

    if summaries:
        _print_summary(summaries)
    print(f"Comparison JSON: {comparison_path}")
    if failures:
        print(f"Failed conditions: {', '.join(row['label'] for row in failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V3 GEPA Training: DSPy+GEPAによる強化学習的な最適化。

目標:
- 現状の解を少しでも良くする
- 参考値（存在するもの）を超えられるかを試す
- 最適化アルゴリズムだけでなく、パーススキームも改良

構成:
- Phase 1: safe getterを使うparse codesをブートストラップ（AST検証付き）
- Phase 2: GEPAでalgorithm_codeの最適化（parseは固定）
- 評価: 参考値との比較 + best_knownの追跡

Python 3.11で実行（gepaパッケージが必要）。
"""

import argparse
import ast
import json
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import dspy

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

from src import best_known as _bk
from src.best_known import init_registry
from src.data_loader import (
    convert_to_dspy_example,
    load_and_split_stratified,
    load_v3_data,
    prepare_examples,
    split_train_val_stratified,
)
from src.gepa_feedback_v3 import gepa_feedback_v3, set_use_reference
from src.lm_config import (
    DEFAULT_GENERATION_API_BASE,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_REFLECTION_API_BASE,
    DEFAULT_REFLECTION_MODEL,
    configure_qwen_pair,
    qwen_configs_from_env,
)
from src.metrics_v3 import evaluate_algorithm_v3
from src.modules import AlgorithmGenerator, default_parse_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_parse_codes_v3(examples):
    """Phase 1: safe getterとAST検証を備えたparse codesを生成。

    各core_typeに対して1回だけparse_codeを生成し、それを固定する。
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Bootstrapping robust parse codes")
    logger.info("=" * 60)

    parse_codes = {}
    seen_types = set()

    for e in examples:
        ct = getattr(e, "core_type", None) or e["core_type"]
        if ct in seen_types:
            continue
        seen_types.add(ct)

        inst = getattr(e, "instance", None) or e["instance"]
        instance_keys = ", ".join(inst.keys()) if inst else ""

        logger.info(f"  Generating robust parse for '{ct}' (keys: {instance_keys[:60]}...)")

        # Use default_parse_code as base (covers ALL keys with type info)
        parse_code = default_parse_code(inst)

        # default_parse_code already uses safe .get() accessors. Validate the
        # complete block instead of wrapping function-body lines independently.
        ast.parse(parse_code)
        parse_codes[ct] = parse_code
        logger.info(f"    -> Robust parse code ({len(parse_code)} chars)")

    logger.info(f"Phase 1 complete: {len(parse_codes)} core_types with robust parse codes")
    return parse_codes


def seed_best_known(raw_examples):
    """best_knownをreference_valueで初期化。"""
    registry = _bk.registry
    for ex in raw_examples:
        iid = ex["instance_id"]
        ref = ex.get("reference_value")
        if ref is not None:
            registry.register(iid, ref)
    logger.info(f"Seeded best_known with {len(raw_examples)} instances")


# Generic parse_code that AlgorithmGenerator.forward() injects (kept in sync).
_GENERIC_PARSE_CODE = (
    "# Generic safe accessors for instance dict.\n"
    "# The requirement text contains the exact keys and structure.\n"
    "def get_list(d, key, default=None):\n"
    "    val = d.get(key, default if default is not None else [])\n"
    "    return val if isinstance(val, list) else (default if default is not None else [])\n"
    "def get_dict(d, key, default=None):\n"
    "    val = d.get(key, default if default is not None else {})\n"
    "    return val if isinstance(val, dict) else (default if default is not None else {})\n"
    "def get_scalar(d, key, default=0):\n"
    "    val = d.get(key, default)\n"
    "    return val if isinstance(val, (int, float)) else default\n"
)


def build_demonstrations(
    data_dir,
    demo_ids=(1, 21),
    max_demos=2,
    results_path=None,
    *,
    use_reference=True,
):
    """既知良コード(beat_reference/exact_match)から少数のfew-shot demoを構築。

    GEPAは instructions を進化させるが demos は触らないため、
    ここで生成器 predictor に付与すると「薄いinstruction + 具体例」の
    両立ができ、exec_error を抑えつつ形式を教えられる。

    Returns list[dspy.Example] with input fields (requirement, core_type,
    parse_code) and output fields (reasoning, algorithm_code).
    """
    if results_path is None:
        tag = "phaseE" if use_reference else "phaseF_noref"
        results_path = str(BASE_DIR / f"evaluation_results_v3_gepa_{tag}.json")
    # Map instance_id -> known-good code from prior results
    code_by_iid = {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for split in ("train", "test"):
            for r in prev.get(split, {}).get("results", []):
                status = r.get("status")
                reference_good = status == "exact_match"
                reference_free_good = r.get("score", 0) > 0 and status not in {
                    "exec_error",
                    "gen_error",
                    "infeasible",
                    "invalid_solution",
                    "solver_failure",
                }
                if r.get("code") and (reference_good if use_reference else reference_free_good):
                    code_by_iid[r["instance_id"]] = r["code"]
    except Exception as e:
        logger.warning(f"Could not load demo source results: {e}")
        return []

    records = {f"prob_{r.get('id', 0):03d}": r for r in load_v3_data(data_dir)}
    demos = []
    preferred = [f"prob_{did:03d}" for did in demo_ids]
    candidate_ids = [iid for iid in preferred if iid in code_by_iid]
    candidate_ids.extend(sorted(set(code_by_iid) - set(candidate_ids)))
    for iid in candidate_ids:
        if iid not in code_by_iid or iid not in records:
            continue
        ex = convert_to_dspy_example(records[iid], use_reference=use_reference)
        target_wording = (
            "the reference solution exactly"
            if use_reference
            else "the required solution schema exactly"
        )
        structure_wording = (
            "the reference solution structure" if use_reference else "the required return schema"
        )
        reasoning = (
            f"This is a {ex['core_type']} problem. I read the exact key names from "
            f"{structure_wording}, choose an exact solver for small "
            f"instances (or a greedy+local-search fallback), wrap everything in "
            f"try/except, and return a NON-EMPTY dict whose top-level keys match "
            f"{target_wording}."
        )
        demo = dspy.Example(
            requirement=ex["requirement"],
            core_type=ex["core_type"],
            parse_code=_GENERIC_PARSE_CODE,
            reasoning=reasoning,
            algorithm_code=code_by_iid[iid],
        ).with_inputs("requirement", "core_type", "parse_code")
        demos.append(demo)
        if len(demos) >= max_demos:
            break
    logger.info("Built %s demonstrations from %s", len(demos), candidate_ids[: len(demos)])
    return demos


def run_gepa_training(
    train_raw,
    val_raw,
    test_raw,
    breadth=3,
    depth=5,
    demos=None,
    out_tag="phaseE",
    reflection_lm=None,
    output_dir=BASE_DIR / "outputs",
):
    """GEPAによる最適化訓練。"""

    # Convert to DSPy examples
    train_examples = prepare_examples(train_raw)
    val_examples = prepare_examples(val_raw)

    # === Phase 1: Bootstrap robust parse codes ===
    all_examples = train_examples + val_examples
    parse_codes = bootstrap_parse_codes_v3(all_examples)

    # === Phase 2: GEPA training ===
    program = AlgorithmGenerator(parse_code_dict=parse_codes)

    # Attach few-shot demonstrations to the generator predictor.
    # GEPA evolves instructions but preserves demos, so this gives
    # "thin instructions + concrete examples" together.
    if demos:
        program.generate.predict.demos = list(demos)
        logger.info(f"Attached {len(demos)} demonstrations to generate predictor")

    n_predictors = len(program.predictors())
    logger.info(f"Predictors: {n_predictors} (parse frozen, generate+improve trainable)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    log_dir = str(output_dir / "gepa_logs" / f"run_v3_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    out_json = str(output_dir / f"compiled_program_v3_gepa_{out_tag}.json")

    logger.info("=" * 60)
    logger.info("Phase 2: GEPA Training")
    logger.info(f"  Breadth: {breadth}, Depth: {depth}")
    logger.info(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")
    logger.info(f"  Robust parse codes for {len(parse_codes)} core_types")
    logger.info("=" * 60)

    start_time = time.time()

    max_evals = breadth * depth

    optimizer = dspy.GEPA(
        metric=gepa_feedback_v3,
        reflection_lm=reflection_lm or dspy.settings.lm,
        max_full_evals=max_evals,
        track_stats=True,
        log_dir=log_dir,
        seed=42,
    )

    compiled = optimizer.compile(
        program,
        trainset=train_examples,
        valset=val_examples if val_examples else train_examples[:1],
    )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.0f}s")

    # Save compiled program
    try:
        compiled.save(out_json)
        logger.info(f"Saved compiled program to {out_json}")
    except Exception as e:
        logger.warning(f"compiled.save() failed: {e}")

    return compiled


def evaluate(compiled, raw_examples, name="Test", use_reference=True):
    """評価: 各インスタンスに対してアルゴリズムを生成してスコアを計算。

    use_reference=False でも、参考値との比較 (beat-reference) は
    事後解析としてコスト比較から必ず計算する (報酬には使わない)。
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{name} Evaluation")
    logger.info(f"{'=' * 60}")

    results = []
    parse_errors = 0
    exec_errors = 0
    beat_reference = 0  # status-based (reference mode only)
    beat_ref_analysis = 0  # post-hoc: cost <= reference (at or better)
    beat_ref_strict = 0  # post-hoc: cost < reference (strictly better)

    for i, ex in enumerate(raw_examples):
        iid = ex["instance_id"]
        requirement = ex["requirement"]
        core_type = ex["core_type"]
        instance = ex["instance"]
        ref = ex.get("reference_value")
        ref_sol = ex.get("reference_solution", {})
        objective_text = ex.get("objective", "")

        # Generate algorithm - only pass inputs defined in with_inputs()
        try:
            pred = compiled(requirement=requirement, core_type=core_type)
            code = pred.algorithm_code
        except Exception as e:
            logger.warning(f"  [{i + 1}/{len(raw_examples)}] {iid}: Generation failed: {e}")
            parse_errors += 1
            results.append(
                {"instance_id": iid, "status": "gen_error", "score": -0.5, "error": str(e)}
            )
            continue

        # Evaluate
        result = evaluate_algorithm_v3(
            code=code,
            instance=instance,
            core_type=core_type,
            instance_id=iid,
            reference_value=ref,
            reference_solution=ref_sol,
            objective_text=objective_text,
            use_reference=use_reference,
        )

        score = result["score"]
        status = result["status"]
        cost = result.get("cost")

        # Count categories
        if result.get("parse_error", False):
            parse_errors += 1
        if status == "exec_error":
            exec_errors += 1
        if status in ("exact_match", "beat_reference"):
            beat_reference += 1

        # Post-hoc reference analysis (independent of scoring mode).
        # Only meaningful for feasible solutions that produced a real cost.
        beat_flag = False
        if (
            cost is not None
            and ref is not None
            and ref != 0
            and status
            not in (
                "exec_error",
                "infeasible",
                "invalid_solution",
                "solver_failure",
                "suspicious_zero",
                "partial_feasible",
                "unverified",
                "gen_error",
            )
        ):
            if cost <= ref * (1 + 1e-6):
                beat_ref_analysis += 1
                beat_flag = True
            if cost < ref * (1 - 1e-6):
                beat_ref_strict += 1

        # Format output
        cost_str = f"{cost:.2f}" if cost is not None else "N/A"
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"
        logger.info(
            f"  [{i + 1}/{len(raw_examples)}] {iid}: score={score:.2f} status={status} cost={cost_str} ref={ref_str} beat={beat_flag}"
        )

        results.append(
            {
                "instance_id": iid,
                "name": ex.get("name", "Unknown"),
                "core_type": core_type,
                "status": status,
                "score": score,
                "cost": cost,
                "reference_value": ref,
                "beat_reference_analysis": beat_flag,
                "detail": result.get("detail", ""),
                "error_category": result.get("error_category", ""),
                "feasibility_verified": result.get("feasibility_verified", False),
                "code": code,
            }
        )

    # Summary
    scores = [r["score"] for r in results]
    valid_scores = [s for s in scores if s > 0]

    logger.info(f"\n{name} Summary:")
    logger.info(f"  Total: {len(results)}")
    logger.info(f"  Valid (score>0): {len(valid_scores)}")
    logger.info(f"  Beat reference (status): {beat_reference}")
    logger.info(f"  Beat reference (analysis, cost<=ref): {beat_ref_analysis}")
    logger.info(f"  Beat reference (analysis, cost<ref strict): {beat_ref_strict}")
    logger.info(f"  Mean score: {sum(scores) / len(scores):.3f}" if scores else "  Mean score: N/A")
    logger.info(
        f"  Mean valid score: {sum(valid_scores) / len(valid_scores):.3f}"
        if valid_scores
        else "  Mean valid score: N/A"
    )
    logger.info(f"  Parse errors: {parse_errors}")
    logger.info(f"  Exec errors: {exec_errors}")

    # Status breakdown
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    logger.info(f"  Status breakdown: {status_counts}")

    # Error category breakdown
    error_cats = {}
    for r in results:
        if r.get("error_category"):
            error_cats[r["error_category"]] = error_cats.get(r["error_category"], 0) + 1
    if error_cats:
        logger.info(f"  Error categories: {error_cats}")

    return {
        "results": results,
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "valid_count": len(valid_scores),
        "total_count": len(results),
        "beat_reference": beat_reference,
        "beat_reference_analysis": beat_ref_analysis,
        "beat_reference_strict": beat_ref_strict,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="V3 GEPA Training")
    parser.add_argument("--breadth", type=int, default=6, help="GEPA breadth")
    parser.add_argument("--depth", type=int, default=8, help="GEPA depth")
    parser.add_argument(
        "--data-dir",
        default=str(BASE_DIR / "data" / "problems"),
        help="データディレクトリ（100問）",
    )
    parser.add_argument("--n-train", type=int, default=40, help="訓練問題数(層化選抜)")
    parser.add_argument("--n-test", type=int, default=20, help="テスト問題数(層化選抜)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="(旧)訓練データの割合")
    parser.add_argument("--no-demos", action="store_true", help="few-shot demoを無効化")
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="参照フリーモード: 報酬・フィードバックに参考値を使わない(解析のみ)",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="既存のコンパイル済みプログラムで評価のみ"
    )
    parser.add_argument("--program-path", help="プログラムの保存/読込パス")
    parser.add_argument("--demo-results", help="few-shot demoを抽出する既存評価結果JSON")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--generation-model",
        default=os.getenv("DSPY_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
    )
    parser.add_argument(
        "--generation-api-base",
        default=os.getenv("DSPY_GENERATION_API_BASE", DEFAULT_GENERATION_API_BASE),
    )
    parser.add_argument(
        "--reflection-model",
        default=os.getenv("DSPY_REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL),
    )
    parser.add_argument(
        "--reflection-api-base",
        default=os.getenv("DSPY_REFLECTION_API_BASE", DEFAULT_REFLECTION_API_BASE),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--lm-timeout", type=int, default=1800)
    parser.add_argument("--skip-connection-check", action="store_true")
    return parser.parse_args(argv)


def check_lm_connection(lm, label):
    """Send one bounded request before a long evaluation or GEPA run."""
    response = lm(messages=[{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=8)
    if not isinstance(response, list) or not response:
        raise RuntimeError(f"{label} LM returned no response")
    logger.info("  %s connection OK: %s", label, str(response[0])[:40])


def main(argv=None):
    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("V3: DSPy+GEPA Optimization Algorithm Generation")
    logger.info("=" * 60)

    use_reference = not args.no_reference
    set_use_reference(use_reference)
    out_tag = "phaseE" if use_reference else "phaseF_noref"
    run_name = args.run_name or f"{out_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir.resolve() / run_name
    program_path = (
        Path(args.program_path)
        if args.program_path
        else (BASE_DIR / f"compiled_program_v3_gepa_{out_tag}.json")
    )
    logger.info(
        f"  Reference mode: {'ON (reference-guided)' if use_reference else 'OFF (reference-free / RL-style)'}"
    )
    logger.info("  Run directory: %s", run_dir)

    env_generation, env_reflection = qwen_configs_from_env()
    generation_lm, reflection_lm = configure_qwen_pair(
        generation=replace(
            env_generation,
            model=args.generation_model,
            api_base=args.generation_api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.lm_timeout,
        ),
        reflection=replace(
            env_reflection,
            model=args.reflection_model,
            api_base=args.reflection_api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.lm_timeout,
        ),
    )
    if not args.skip_connection_check:
        try:
            check_lm_connection(generation_lm, "generation")
            if not args.eval_only:
                check_lm_connection(reflection_lm, "reflection")
        except Exception as exc:
            logger.error("LM connection check failed: %s", exc)
            raise SystemExit(2) from exc
    run_dir.mkdir(parents=True, exist_ok=False)

    state_name = "best_known.jsonl" if use_reference else "best_known_noref.jsonl"
    storage = run_dir / state_name
    init_registry(storage_path=storage)
    logger.info("Fresh best-known registry: %s", storage)

    # Load data (stratified 40/20 across core_types)
    logger.info(f"\nLoading data from {args.data_dir}...")
    train_raw, test_raw = load_and_split_stratified(
        args.data_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=42,
        use_reference=use_reference,
    )
    logger.info(f"Train+Val: {len(train_raw)}, Test: {len(test_raw)}")

    # Split train into train/val (larger val for stronger GEPA signal)
    n_val = max(3, len(train_raw) // 3)
    train_only_raw, val_raw = split_train_val_stratified(train_raw, n_val, seed=42)
    logger.info(f"Train: {len(train_only_raw)}, Val: {n_val}, Test: {len(test_raw)}")

    # Print core_type distribution
    core_types = {}
    for ex in train_only_raw + val_raw + test_raw:
        ct = ex["core_type"]
        core_types[ct] = core_types.get(ct, 0) + 1
    logger.info(f"Core types: {core_types}")

    # Seed best_known with reference values.
    # In reference-free mode we must NOT seed from reference (that would leak
    # the reference into the reward via the best_known anchor). Start empty so
    # the model discovers improvements from scratch (RL-style).
    if use_reference:
        seed_best_known(train_only_raw + val_raw + test_raw)
    else:
        logger.info("Reference-free mode: skipping best_known seeding (no reference leak)")

    demos = []
    if args.eval_only:
        if program_path.exists():
            logger.info(f"Loading compiled program from {program_path}")
            compiled = AlgorithmGenerator()
            compiled.load(str(program_path))

            # Evaluate
            train_result = evaluate(
                compiled, train_only_raw + val_raw, name="Train", use_reference=use_reference
            )
            test_result = evaluate(compiled, test_raw, name="Test", use_reference=use_reference)
        else:
            logger.error(f"No compiled program found at {program_path}. Run without --eval-only.")
            sys.exit(1)
    else:
        # Phase 1+2: GEPA training
        if args.no_demos or (not use_reference and args.demo_results is None):
            demos = []
            if not use_reference and not args.no_demos:
                logger.info("Reference-free mode: demos disabled without a clean demo source")
        else:
            demos = build_demonstrations(
                args.data_dir,
                results_path=args.demo_results,
                use_reference=use_reference,
            )
            if len(demos) < 2:
                raise SystemExit("Two demonstrations are required; pass --no-demos to opt out")
        compiled = run_gepa_training(
            train_only_raw,
            val_raw,
            test_raw,
            breadth=args.breadth,
            depth=args.depth,
            demos=demos,
            out_tag=out_tag,
            reflection_lm=reflection_lm,
            output_dir=run_dir,
        )

        # Evaluate
        train_result = evaluate(
            compiled, train_only_raw + val_raw, name="Train", use_reference=use_reference
        )
        test_result = evaluate(compiled, test_raw, name="Test", use_reference=use_reference)

    results_path = run_dir / f"evaluation_results_v3_gepa_{out_tag}.json"
    results = {
        "train": train_result,
        "test": test_result,
        "config": {
            "data_dir": str(Path(args.data_dir).resolve()),
            "n_train": args.n_train,
            "n_test": args.n_test,
            "breadth": args.breadth,
            "depth": args.depth,
            "demos": 0 if args.no_demos else len(demos or []),
            "use_reference": use_reference,
            "generation_model": args.generation_model,
            "generation_api_base": args.generation_api_base,
            "reflection_model": args.reflection_model,
            "reflection_api_base": args.reflection_api_base,
            "program_path": str(program_path.resolve()) if args.eval_only else None,
        },
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("\nResults saved to %s", results_path)

    # Save best_known
    _bk.registry.save_to_disk()
    logger.info(f"Registry saved to {storage}")


if __name__ == "__main__":
    main()

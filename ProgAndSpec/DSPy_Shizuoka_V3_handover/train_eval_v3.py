"""V3 Train & Eval: DSPyによる最適化アルゴリズム生成。

V2(V8)をV3データフォーマットに移植。
- 80/20 split (train/test)
- V3メトリック (reference_solutionベース)
- 改善: parse_errorペナルティ強化(-0.5), タイムアウトペナルティ(-0.3)
- 改善: core_type別アルゴリズム提案フィードバック
- 改善: best_knownのマルチロール播种
- 改善: 直接的なiterative train-eval-improveループ（GEPA不使用）
"""
import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import dspy

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.lm_config import configure_qwen_default
from src.best_known import init_registry, BestKnownRegistry
from src import best_known as _bk
from src.modules import AlgorithmGenerator, default_parse_code
from src.data_loader import load_v3_data, convert_to_dspy_example, load_and_split
from src.requirement_builder import build_requirement
from src.metrics_v3 import evaluate_algorithm_v3, dspy_metric_v3
from src.gepa_feedback_v3 import gepa_feedback_v3


def prepare_examples(examples: list[dict]) -> list[dspy.Example]:
    """V3例をDSPy Exampleに変換。"""
    dspys = []
    for ex in examples:
        dspys.append(dspy.Example(
            requirement=ex["requirement"],
            core_type=ex["core_type"],
            instance=ex["instance"],
            instance_id=ex["instance_id"],
            reference_value=ex["reference_value"],
            reference_solution=ex.get("reference_solution", {}),
        ).with_inputs("requirement", "core_type", "instance"))
    return dspys


def seed_best_known_multiroll(registry: BestKnownRegistry, examples: list[dict],
                               generator: Optional[AlgorithmGenerator] = None,
                               num_rolls: int = 3) -> None:
    """best_knownをreference_valueで初期化 + マルチロール播种。
    
    generatorが指定された場合、各インスタンスに対して複数回生成して
    最良の解でbest_knownを更新する。
    """
    # First: seed with reference values
    for ex in examples:
        iid = ex["instance_id"]
        ref = ex.get("reference_value")
        if ref is not None:
            registry.register(iid, ref)
    
    # Second: if generator available, try multiple rolls
    if generator is None:
        return
    
    print(f"\nSeeding best_known with {num_rolls} rolls per instance...")
    for i, ex in enumerate(examples):
        iid = ex["instance_id"]
        instance = ex["instance"]
        core_type = ex["core_type"]
        ref = ex.get("reference_value")
        
        best_cost = ref  # Start with reference
        
        for roll in range(num_rolls):
            try:
                pred = generator.forward(ex["requirement"], core_type, instance=instance)
                code = pred.algorithm_code
                
                result = evaluate_algorithm_v3(
                    code=code, instance=instance, core_type=core_type,
                    instance_id=iid, reference_value=ref,
                    registry=registry,
                )
                
                cost = result.get("cost")
                if cost is not None and (best_cost is None or cost < best_cost):
                    best_cost = cost
                    registry.update_if_better(iid, cost)
                    
            except Exception:
                pass
        
        status = "improved" if best_cost is not None and ref is not None and best_cost < ref else "baseline"
        if (i + 1) % 5 == 0 or i == len(examples) - 1:
            print(f"  Seeded {i+1}/{len(examples)} instances ({status})")


def train_iterative(train_examples: list[dspy.Example], val_examples: list[dspy.Example],
                     n_iterations: int = 3, max_bootstrapped: int = 3) -> AlgorithmGenerator:
    """Iterative training: generate → evaluate → improve loop。
    
    GEPAを使わずに直接的なフィードバックループを実装。
    1. Bootstrap: LLMで初期コードを生成
    2. Evaluate: 各コードを実行してスコアを計算
    3. Improve: フィードバックで悪いコードを改善
    4. Repeat
    """
    print("\n" + "="*60)
    print(f"Iterative Training ({n_iterations} iterations)")
    print("="*60)
    
    generator = AlgorithmGenerator()
    
    # === Bootstrap phase ===
    print(f"\nBootstrap: generating {min(max_bootstrapped, len(train_examples))} examples...")
    bootstrapped = []
    for i, ex in enumerate(train_examples[:max_bootstrapped]):
        try:
            pred = generator.forward(ex.requirement, ex.core_type, instance=ex.instance)
            
            # Evaluate the bootstrapped code
            result = evaluate_algorithm_v3(
                code=pred.algorithm_code,
                instance=ex.instance,
                core_type=ex.core_type,
                instance_id=ex.instance_id,
                reference_value=ex.reference_value,
            )
            
            score = result["score"]
            status = result["status"]
            print(f"  [{i+1}] {ex.instance_id}: score={score:.2f} status={status}")
            
            # Only keep valid solutions for bootstrapping
            if score > 0:
                bootstrapped.append(dspy.Example(
                    requirement=ex.requirement,
                    core_type=ex.core_type,
                    instance=ex.instance,
                    algorithm_code=pred.algorithm_code,
                    parse_code=pred.parse_code,
                ).with_inputs("requirement", "core_type", "instance"))
            else:
                print(f"    Skipping (score={score:.2f}, not valid)")
        except Exception as e:
            print(f"  Failed to bootstrap example {i+1}: {e}")
    
    print(f"  Kept {len(bootstrapped)}/{min(max_bootstrapped, len(train_examples))} valid bootstraps")
    
    # === Iterative improve phase ===
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration+1}/{n_iterations} ---")
        
        iteration_scores = []
        error_categories = {}  # track error types
        improved_codes = {}  # instance_id -> improved code
        
        for i, ex in enumerate(train_examples):
            try:
                # Generate or use improved code from previous iteration
                if ex.instance_id in improved_codes:
                    code = improved_codes[ex.instance_id]
                else:
                    pred = generator.forward(ex.requirement, ex.core_type, instance=ex.instance)
                    code = pred.algorithm_code
                
                # Evaluate
                result = evaluate_algorithm_v3(
                    code=code, instance=ex.instance, core_type=ex.core_type,
                    instance_id=ex.instance_id, reference_value=ex.reference_value,
                )
                
                score = result["score"]
                status = result["status"]
                err_cat = result.get("error_category", "")
                iteration_scores.append(score)
                
                # Track error categories
                if status == "exec_error" and err_cat:
                    error_categories[err_cat] = error_categories.get(err_cat, 0) + 1
                
                # If score is poor, try to improve
                if score < 1.0:
                    feedback = gepa_feedback_v3(ex, dspy.Prediction(algorithm_code=code))
                    feedback_text = feedback.feedback
                    
                    # Use improve_forward to get better code
                    try:
                        parse_code = generator.parse_code_dict.get(ex.core_type, default_parse_code(ex.instance))
                        
                        improved_pred = generator.improve_forward(
                            original_code=code,
                            parse_code=parse_code,
                            feedback=feedback_text,
                            core_type=ex.core_type,
                        )
                        improved_codes[ex.instance_id] = improved_pred.algorithm_code
                        
                        # Evaluate improved code
                        improved_result = evaluate_algorithm_v3(
                            code=improved_pred.algorithm_code,
                            instance=ex.instance, core_type=ex.core_type,
                            instance_id=ex.instance_id, reference_value=ex.reference_value,
                        )
                        improved_score = improved_result["score"]
                        improved_err = improved_result.get("error_category", "")
                        
                        if improved_score > score:
                            iteration_scores[-1] = improved_score  # Update score
                            print(f"  [{i+1}] {ex.instance_id}: {score:.2f} -> {improved_score:.2f} (improved) [{err_cat}->{improved_err}]")
                        else:
                            print(f"  [{i+1}] {ex.instance_id}: {score:.2f} -> {improved_score:.2f} (no improvement) [{err_cat}->{improved_err}]")
                            del improved_codes[ex.instance_id]  # Discard worse code
                    except Exception as e:
                        print(f"  [{i+1}] {ex.instance_id}: score={score:.2f} [{err_cat}] (improve failed: {e})")
                else:
                    print(f"  [{i+1}] {ex.instance_id}: score={score:.2f} {status}")
                    
            except Exception as e:
                iteration_scores.append(0.0)
                print(f"  [{i+1}] {ex.instance_id}: error ({e})")
        
        mean_score = sum(iteration_scores) / len(iteration_scores) if iteration_scores else 0
        valid_count = sum(1 for s in iteration_scores if s > 0)
        print(f"\nIteration {iteration+1} summary: mean={mean_score:.3f}, valid={valid_count}/{len(train_examples)}")
        if error_categories:
            print(f"  Error breakdown: {error_categories}")
    
    return generator


def evaluate(generator: AlgorithmGenerator, examples: list[dict], name: str = "Test") -> dict:
    """評価: 各インスタンスに対してアルゴリズムを生成してスコアを計算。"""
    print(f"\n{'='*60}")
    print(f"{name} Evaluation")
    print(f"{'='*60}")
    
    registry = _bk.registry
    results = []
    
    for i, ex in enumerate(examples):
        iid = ex["instance_id"]
        requirement = ex["requirement"]
        core_type = ex["core_type"]
        instance = ex["instance"]
        ref = ex.get("reference_value")
        ref_sol = ex.get("reference_solution", {})
        
        # Generate algorithm
        try:
            pred = generator.forward(requirement, core_type, instance=instance)
            code = pred.algorithm_code
        except Exception as e:
            print(f"  [{i+1}/{len(examples)}] {iid}: Generation failed: {e}")
            results.append({"instance_id": iid, "status": "gen_error", "score": 0.0, "error": str(e)})
            continue
        
        # Evaluate
        result = evaluate_algorithm_v3(
            code=code, instance=instance, core_type=core_type,
            instance_id=iid, reference_value=ref, reference_solution=ref_sol,
        )
        
        score = result["score"]
        status = result["status"]
        cost = result.get("cost")
        
        # Format output
        cost_str = f"{cost:.2f}" if cost is not None else "N/A"
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"
        print(f"  [{i+1}/{len(examples)}] {iid}: score={score:.2f} status={status} cost={cost_str} ref={ref_str}")
        
        results.append({
            "instance_id": iid,
            "name": ex.get("name", "Unknown"),
            "core_type": core_type,
            "status": status,
            "score": score,
            "cost": cost,
            "reference_value": ref,
            "detail": result.get("detail", ""),
            "error_category": result.get("error_category", ""),
            "code": code,
        })
    
    # Summary
    scores = [r["score"] for r in results]
    valid_scores = [s for s in scores if s > 0]
    
    print(f"\n{name} Summary:")
    print(f"  Total: {len(results)}")
    print(f"  Valid (score>0): {len(valid_scores)}")
    print(f"  Mean score: {sum(scores)/len(scores):.3f}" if scores else "  Mean score: N/A")
    print(f"  Mean valid score: {sum(valid_scores)/len(valid_scores):.3f}" if valid_scores else "  Mean valid score: N/A")
    print(f"  Min score: {min(scores):.3f}" if scores else "  Min score: N/A")
    print(f"  Max score: {max(scores):.3f}" if scores else "  Max score: N/A")
    
    # Status breakdown
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  Status breakdown: {status_counts}")
    
    # Error category breakdown
    error_cats = {}
    for r in results:
        if r.get("error_category"):
            error_cats[r["error_category"]] = error_cats.get(r["error_category"], 0) + 1
    if error_cats:
        print(f"  Error categories: {error_cats}")
    
    return {
        "results": results,
        "mean_score": sum(scores)/len(scores) if scores else 0,
        "valid_count": len(valid_scores),
        "total_count": len(results),
    }


def save_program(generator: AlgorithmGenerator, path: str) -> None:
    """コンパイル済みプログラムを保存。"""
    program_data = {"metadata": {}}
    
    # Save parse_code_dict
    if hasattr(generator, "parse_code_dict") and generator.parse_code_dict:
        program_data["metadata"]["parse_code_dict"] = generator.parse_code_dict
    
    # Save predictor texts - ChainOfThought wraps Predict, access via .predictor
    for predictor in generator.predictors():
        pname = id(predictor)  # Use id to avoid duplicates
        # ChainOfThought has .predictor attribute which is the actual Predict
        inner = getattr(predictor, 'predictor', predictor)
        program_data[f"cot_{len(program_data)}"] = {
            "name": predictor.__class__.__name__,
            "lm_signature": getattr(inner, 'lm_signature', None),
        }
        # Save the signature's messages if available
        sig = getattr(inner, 'signature', None)
        if sig:
            program_data[f"cot_{len(program_data)-1}"]["signature_name"] = sig.__name__ if hasattr(sig, '__name__') else str(sig)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(program_data, f, indent=2, ensure_ascii=False)
    print(f"Saved program to {path}")


def load_program(path: str) -> AlgorithmGenerator:
    """保存されたプログラムを読み込む。"""
    with open(path, "r", encoding="utf-8") as f:
        program_data = json.load(f)
    
    parse_code_dict = {}
    if "metadata" in program_data and "parse_code_dict" in program_data["metadata"]:
        parse_code_dict = program_data["metadata"]["parse_code_dict"]
    
    compiled = AlgorithmGenerator(parse_code_dict=parse_code_dict)
    
    print(f"Loaded program from {path}")
    return compiled


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="V3 Train & Eval")
    parser.add_argument("--mode", choices=["train", "eval", "full"], default="full",
                        help="train: 訓練のみ, eval: 評価のみ, full: 訓練+評価")
    parser.add_argument("--data-dir", default=str(BASE_DIR / "data"),
                        help="データディレクトリ")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="訓練データの割合")
    parser.add_argument("--iterations", type=int, default=3,
                        help="訓練の反復回数")
    parser.add_argument("--program-path", default=str(BASE_DIR / "compiled_program_v3.json"),
                        help="プログラムの保存/読込パス")
    parser.add_argument("--bootstrap", type=int, default=3,
                        help="ブートストラップ例数")
    parser.add_argument("--seed-rolls", type=int, default=2,
                        help="best_known播种のロール数")
    
    args = parser.parse_args()
    
    print("="*60)
    print("V3: DSPy Optimization Algorithm Generation")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_raw, test_raw = load_and_split(args.data_dir, train_ratio=args.train_ratio)
    print(f"Train: {len(train_raw)}, Test: {len(test_raw)}")
    
    # Print core_type distribution
    core_types = {}
    for ex in train_raw + test_raw:
        ct = ex["core_type"]
        core_types[ct] = core_types.get(ct, 0) + 1
    print(f"Core types: {core_types}")
    
    # Convert to DSPy examples
    train_examples = prepare_examples(train_raw)
    test_examples = prepare_examples(test_raw)
    
    # Split train into train/val
    n_val = max(1, len(train_examples) // 5)
    val_examples = train_examples[-n_val:]
    train_only = train_examples[:-n_val]
    print(f"Train: {len(train_only)}, Val: {n_val}, Test: {len(test_examples)}")
    
    # Configure LM
    configure_qwen_default()
    
    # Initialize registry
    storage = BASE_DIR / "data" / "best_known.jsonl"
    init_registry(storage_path=storage)
    _bk.registry.load_from_disk()
    
    # Initial seed with reference values
    for ex in train_raw + test_raw:
        iid = ex["instance_id"]
        ref = ex.get("reference_value")
        if ref is not None:
            _bk.registry.register(iid, ref)
    
    if args.mode in ("train", "full"):
        # Phase 1: Initial generation + best_known seeding
        print("\n" + "="*60)
        print("Phase 1: Initial generation + best_known seeding")
        print("="*60)
        
        generator = AlgorithmGenerator()
        
        # Generate initial codes and seed best_known with better values
        seed_best_known_multiroll(_bk.registry, train_raw, generator, num_rolls=args.seed_rolls)
        
        # Phase 2: Iterative training
        generator = train_iterative(train_only, val_examples,
                                     n_iterations=args.iterations,
                                     max_bootstrapped=args.bootstrap)
        
        # Final evaluations
        train_result = evaluate(generator, train_raw, name="Train")
        test_result = evaluate(generator, test_raw, name="Test")
        
        # Save results
        results_path = BASE_DIR / "evaluation_results_v3.json"
        results = {
            "train": train_result,
            "test": test_result,
            "config": {
                "train_ratio": args.train_ratio,
                "iterations": args.iterations,
                "bootstrap": args.bootstrap,
                "seed_rolls": args.seed_rolls,
            }
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {results_path}")
    
    elif args.mode == "eval":
        # Load program if available
        program_path = Path(args.program_path)
        if program_path.exists():
            generator = load_program(str(program_path))
            train_result = evaluate(generator, train_raw, name="Train")
            test_result = evaluate(generator, test_raw, name="Test")
        else:
            print("No compiled program found. Run --mode train first.")


if __name__ == "__main__":
    main()

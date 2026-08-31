"""GEPA学習済みモデルの評価のみを実行。"""
import sys, json, logging, time
from pathlib import Path
from datetime import datetime

import dspy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# Configure LM
from src.lm_config import configure_qwen_default
configure_qwen_default()
dspy.settings.lm.kwargs["max_tokens"] = 16384

# Load modules
from src.data_loader import load_v3_data, prepare_examples, convert_to_dspy_example
from src.modules import AlgorithmGenerator
from src.metrics_v3 import evaluate_algorithm_v3
from src import best_known as _bk_module
_bk = _bk_module.init_registry(str(BASE_DIR / "data" / "best_known.jsonl"))


def seed_best_known(raw_examples, registry):
    """best_knownをreference_valueで初期化。"""
    for ex in raw_examples:
        iid = ex["instance_id"]
        ref = ex.get("reference_value")
        if ref is not None:
            registry.register(iid, ref)
    logger.info(f"Seeded best_known with {len(raw_examples)} instances")


def evaluate(compiled, raw_examples, name="Test"):
    """評価: 各インスタンスに対してアルゴリズムを生成してスコアを計算。"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Evaluation")
    logger.info(f"{'='*60}")
    
    results = []
    parse_errors = 0
    exec_errors = 0
    beat_reference = 0
    
    for i, ex in enumerate(raw_examples):
        iid = ex["instance_id"]
        requirement = ex["requirement"]
        core_type = ex["core_type"]
        instance = ex["instance"]
        ref = ex.get("reference_value")
        ref_sol = ex.get("reference_solution", {})
        
        # Generate algorithm - only pass inputs defined in with_inputs()
        try:
            pred = compiled(requirement=requirement, core_type=core_type)
            code = pred.algorithm_code
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(raw_examples)}] {iid}: Generation failed: {e}")
            parse_errors += 1
            results.append({"instance_id": iid, "status": "gen_error", "score": -0.5, "error": str(e)})
            continue
        
        # Evaluate
        result = evaluate_algorithm_v3(
            code=code, instance=instance, core_type=core_type,
            instance_id=iid, reference_value=ref, reference_solution=ref_sol,
        )
        
        score = result["score"]
        status = result["status"]
        cost = result.get("cost")
        
        # Count categories
        if status == "beat_reference":
            beat_reference += 1
        elif status in ("syntax_error", "exec_error"):
            exec_errors += 1
        
        # Log result
        ref_str = f"{ref:.2f}" if ref else "N/A"
        cost_str = f"{cost:.2f}" if cost else "N/A"
        logger.info(
            f"  [{i+1}/{len(raw_examples)}] {iid}: {status} | score={score:.2f} | "
            f"cost={cost_str} | ref={ref_str}"
        )
        
        results.append({
            "instance_id": iid,
            "status": status,
            "score": score,
            "cost": cost,
            "reference_value": ref,
            "solution": result.get("solution"),
            "error": result.get("error"),
        })
    
    # Summary
    valid_scores = [r["score"] for r in results if r["score"] > 0]
    all_scores = [r["score"] for r in results]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Results Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  Total: {len(results)}")
    logger.info(f"  Valid (score>0): {len(valid_scores)}")
    logger.info(f"  Parse Errors: {parse_errors}")
    logger.info(f"  Exec Errors: {exec_errors}")
    logger.info(f"  Beat reference: {beat_reference}")
    logger.info(f"  Mean score: {sum(all_scores)/len(all_scores):.4f}")
    logger.info(f"  Mean valid score: {sum(valid_scores)/len(valid_scores):.4f}" if valid_scores else "  Mean valid score: N/A")
    logger.info(f"{'='*60}")
    
    return results


def main():
    logger.info("=" * 60)
    logger.info("V3: GEPA Compiled Model Evaluation Only")
    logger.info("=" * 60)
    
    # Load and convert data
    data_path = BASE_DIR / "data"
    logger.info(f"\nLoading data from {data_path}...")
    raw_data = load_v3_data(str(data_path))
    raw_examples = [convert_to_dspy_example(d) for d in raw_data]
    
    # Split: 80% train, 20% test
    train_ratio = 0.8
    n_train = int(len(raw_examples) * train_ratio)
    train_raw = raw_examples[:n_train]
    test_raw = raw_examples[n_train:]
    
    logger.info(f"Train: {len(train_raw)}, Test: {len(test_raw)}")
    
    # Seed best_known with ALL data
    seed_best_known(raw_examples, _bk)
    
    # Load compiled model
    model_path = BASE_DIR / "compiled_program_v3_gepa.json"
    logger.info(f"\nLoading compiled model from {model_path}...")
    
    compiled = AlgorithmGenerator()
    compiled.load(str(model_path))
    logger.info("Model loaded successfully")
    
    # Evaluate
    train_results = evaluate(compiled, train_raw, name="Train")
    test_results = evaluate(compiled, test_raw, name="Test")
    
    # Save results
    results = {
        "train": {
            "results": train_results,
            "mean_score": sum(r["score"] for r in train_results) / len(train_results),
            "valid_count": len([r for r in train_results if r["score"] > 0]),
            "total_count": len(train_results),
            "beat_reference": len([r for r in train_results if r["status"] == "beat_reference"]),
        },
        "test": {
            "results": test_results,
            "mean_score": sum(r["score"] for r in test_results) / len(test_results),
            "valid_count": len([r for r in test_results if r["score"] > 0]),
            "total_count": len(test_results),
            "beat_reference": len([r for r in test_results if r["status"] == "beat_reference"]),
        },
        "config": {
            "train_ratio": train_ratio,
            "evaluated_at": datetime.now().isoformat(),
        },
    }
    
    out_path = BASE_DIR / "evaluation_results_v3_gepa.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out_path}")
    
    # Save best_known registry
    _bk.save()
    logger.info(f"Registry saved to {_bk.path}")


if __name__ == "__main__":
    main()

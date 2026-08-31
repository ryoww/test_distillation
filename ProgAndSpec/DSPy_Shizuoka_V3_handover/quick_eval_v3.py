"""V3 簡易評価: コンパイル済みプログラムなしでLLMが生成したコードを評価。"""
import json
import sys
from pathlib import Path

import dspy

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.lm_config import configure_qwen_default
from src.best_known import init_registry
from src import best_known as _bk
from src.modules import AlgorithmGenerator, strip_code_fence
from src.data_loader import load_and_split, convert_to_dspy_example
from src.metrics_v3 import evaluate_algorithm_v3


def main():
    print("="*60)
    print("V3 Quick Evaluation (no compiled program)")
    print("="*60)
    
    # Load data
    train_raw, test_raw = load_and_split(
        str(BASE_DIR / "data"), 
        train_ratio=0.8
    )
    print(f"Train: {len(train_raw)}, Test: {len(test_raw)}")
    
    # Configure LM
    configure_qwen_default()
    
    # Initialize registry
    storage = BASE_DIR / "data" / "best_known.jsonl"
    init_registry(storage_path=storage)
    
    # Seed best_known
    for ex in train_raw + test_raw:
        iid = ex["instance_id"]
        ref = ex.get("reference_value")
        if ref is not None:
            _bk.registry.register(iid, ref)
    
    # Create a basic generator (no training)
    generator = AlgorithmGenerator()
    
    # Evaluate a subset
    eval_set = train_raw[:5] + test_raw[:3]
    print(f"\nEvaluating {len(eval_set)} instances (untrained generator)...")
    
    results = []
    for i, ex in enumerate(eval_set):
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
            print(f"  [{i+1}] {iid}: Generation failed: {e}")
            results.append({"instance_id": iid, "status": "gen_error", "score": 0.0})
            continue
        
        # Evaluate
        result = evaluate_algorithm_v3(
            code=code, instance=instance, core_type=core_type,
            instance_id=iid, reference_value=ref, reference_solution=ref_sol,
        )
        
        score = result["score"]
        status = result["status"]
        cost = result.get("cost")
        cost_str = f"{cost:.2f}" if cost is not None else "N/A"
        ref_str = f"{ref:.2f}" if ref is not None else "N/A"
        
        print(f"  [{i+1}] {iid}: score={score:.2f} status={status} cost={cost_str} ref={ref_str}")
        if result.get("parse_error"):
            print(f"        PARSE ERROR: {result.get('detail', '')[:100]}")
        
        results.append({
            "instance_id": iid,
            "status": status,
            "score": score,
            "cost": cost,
            "reference_value": ref,
            "detail": result.get("detail", ""),
        })
    
    # Summary
    scores = [r["score"] for r in results]
    print(f"\nSummary:")
    print(f"  Mean score: {sum(scores)/len(scores):.3f}")
    print(f"  Success rate: {sum(1 for s in scores if s > 0)/len(scores):.1%}")
    
    # Status breakdown
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"  Status: {status_counts}")


if __name__ == "__main__":
    main()

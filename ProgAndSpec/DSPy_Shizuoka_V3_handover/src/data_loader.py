"""V3 データローダー: prob_XXX.json → DSPy用例フォーマットに変換。

V3データフォーマット:
  {id, name, domain, math_type, difficulty, split, description, requirements, instance, reference_solution}

V2データフォーマット:
  {problem_id, problem_name, core_type, ..., problem_statement, input_format, output_format, instance, reference_value, reference_solution}

主な違い:
- core_type → domain + math_type の組み合わせ
- problem_statement → description
- input_format/output_format → requirements.constraints + requirements.objective
- reference_value → reference_solution から計算
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Tuple


def load_v3_data(data_dir: str) -> List[Dict[str, Any]]:
    """data_dir 内の prob_*.json をすべて読み込んでリストを返す。"""
    pattern = os.path.join(data_dir, "prob_*.json")
    files = sorted(glob.glob(pattern))
    records = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        records.append(data)
    return records


def core_type_from_v3(record: dict) -> str:
    """V3のdomain + math_typeからcore_typeを生成。"""
    domain = record.get("domain", "unknown")
    math_type = record.get("math_type", "unknown")
    # Normalize to a compact core_type string
    return f"{domain}_{math_type}"


def reference_value_from_solution(record: dict) -> float | None:
    """reference_solutionから数値スコアを抽出。

    V3のreference_solutionは多様な構造:
    - objective_value (スケジューリング系)
    - total_distance (VRP系)
    - num_trains (貨物列車)
    - project_duration (クリティカルパス)
    - その他

    最小化問題なので、これらの値をそのままreference_valueとして使用。
    """
    ref = record.get("reference_solution", {})
    if not isinstance(ref, dict):
        return None
    
    # Try common keys for objective values (minimization)
    for key in ["objective_value", "total_distance", "project_duration", 
                "num_trains", "total_cost", "makespan", "total_delay"]:
        if key in ref and isinstance(ref[key], (int, float)):
            return float(ref[key])
    
    # Fallback: first numeric value found
    for v in ref.values():
        if isinstance(v, (int, float)):
            return float(v)
    
    return None


def convert_to_dspy_example(record: dict) -> dict:
    """V3レコードをDSPy用例に変換。"""
    core_type = core_type_from_v3(record)
    ref_value = reference_value_from_solution(record)
    
    # Build requirement text from V3 format
    req_parts = []
    req_parts.append(f"## Problem: {record.get('name', 'Unknown')}")
    req_parts.append(f"Category: {core_type}")
    req_parts.append(f"Difficulty: {record.get('difficulty', 'N/A')}")
    
    # Description
    desc = record.get("description", "")
    if desc:
        req_parts.append(f"## Description\n{desc}")
    
    # Requirements
    requirements = record.get("requirements", {})
    if requirements:
        req_parts.append("## Requirements")
        obj = requirements.get("objective", "")
        if obj:
            req_parts.append(f"Objective: {obj}")
        constraints = requirements.get("constraints", [])
        if constraints:
            req_parts.append("Constraints:")
            for c in constraints:
                req_parts.append(f"  - {c}")
    
    # Instance summary
    instance = record.get("instance", {})
    if instance:
        req_parts.append("## Instance Data")
        inst_summary = {}
        for k, v in instance.items():
            if isinstance(v, (int, float)):
                inst_summary[k] = v
            elif isinstance(v, str):
                inst_summary[k] = f"str(len={len(v)})"
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    inst_summary[k] = f"list[{len(v)}] of dicts with keys {list(v[0].keys())}"
                elif v and isinstance(v[0], (int, float)):
                    inst_summary[k] = f"list[{len(v)}] of {type(v[0]).__name__}"
                elif v and isinstance(v[0], list):
                    inst_summary[k] = f"list[{len(v)}] of lists (inner len={len(v[0])})"
                else:
                    inst_summary[k] = f"list[{len(v)}]"
            elif isinstance(v, dict):
                inst_summary[k] = f"dict with keys {list(v.keys())}"
            else:
                inst_summary[k] = type(v).__name__
        req_parts.append(json.dumps(inst_summary, indent=2, ensure_ascii=False))
    
    # Reference value
    if ref_value is not None:
        req_parts.append(f"## Reference Value: {ref_value:.2f}")
        req_parts.append("Your algorithm should try to achieve a value close to or better than this reference (lower is better).")
    
    # Return format instruction
    req_parts.append("## Return Format")
    req_parts.append("Your solve() function MUST return the solution as a Python value (list, dict, int, etc.).")
    req_parts.append("Do NOT use print() to output the solution. Use `return solution` at the end.")
    req_parts.append("The returned value will be scored automatically.")
    
    requirement_text = "\n\n".join(req_parts)
    
    return {
        "instance_id": f"prob_{record.get('id', 0):03d}",
        "instance": record.get("instance", {}),
        "core_type": core_type,
        "requirement": requirement_text,
        "reference_value": ref_value,
        "reference_solution": record.get("reference_solution", {}),
        "objective": requirements.get("objective", "") if isinstance(requirements, dict) else "",
        "name": record.get("name", "Unknown"),
        "domain": record.get("domain", "unknown"),
        "math_type": record.get("math_type", "unknown"),
        "difficulty": record.get("difficulty", "N/A"),
        "split": record.get("split", "train"),
    }


def load_and_split(data_dir: str, train_ratio: float = 0.8) -> Tuple[List[dict], List[dict]]:
    """V3データをロードして訓練/テストに分割。
    
    splitフィールドが"train"/"test"の両方がある場合はそれを尊重。
    片方のみの場合はratio-based分割にフォールバック。
    """
    records = load_v3_data(data_dir)
    examples = [convert_to_dspy_example(r) for r in records]
    
    # Check if split field has both train and test
    splits = set(ex.get("split") for ex in examples)
    has_both = "train" in splits and "test" in splits
    
    if has_both:
        # Respect explicit split field
        train = [ex for ex in examples if ex.get("split") == "train"]
        test = [ex for ex in examples if ex.get("split") == "test"]
    else:
        # Ratio-based split (ignore split field if only one value)
        n_train = int(len(examples) * train_ratio)
        train = examples[:n_train]
        test = examples[n_train:]
        # Update split field for consistency
        for ex in train:
            ex["split"] = "train"
        for ex in test:
            ex["split"] = "test"
    
    return train, test


def load_and_split_stratified(
    data_dir: str, n_train: int = 40, n_test: int = 20, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """core_typeで層化して n_train/n_test を選抜（再現性のためseed固定）。

    - 全レコードをcore_typeでグループ化
    - 各グループ内をseed付きで決定的にシャッフル
    - ラウンドロビンでtrain/testを埋め、両方にcore_type多様性を確保
    - train/testは互いに素
    """
    import random

    records = load_v3_data(data_dir)
    examples = [convert_to_dspy_example(r) for r in records]

    groups: Dict[str, List[dict]] = {}
    for ex in examples:
        groups.setdefault(ex["core_type"], []).append(ex)

    rng = random.Random(seed)
    ordered_types = sorted(groups.keys())
    for ct in ordered_types:
        groups[ct].sort(key=lambda e: e["instance_id"])
        rng.shuffle(groups[ct])

    # Round-robin pull: first fill test (rarer coverage), then train
    train: List[dict] = []
    test: List[dict] = []
    # Interleave: alternate assigning one example per core_type
    pools = {ct: list(items) for ct, items in groups.items()}
    # Test first for diversity
    while len(test) < n_test:
        progressed = False
        for ct in ordered_types:
            if len(test) >= n_test:
                break
            if pools[ct]:
                test.append(pools[ct].pop())
                progressed = True
        if not progressed:
            break
    while len(train) < n_train:
        progressed = False
        for ct in ordered_types:
            if len(train) >= n_train:
                break
            if pools[ct]:
                train.append(pools[ct].pop())
                progressed = True
        if not progressed:
            break

    for ex in train:
        ex["split"] = "train"
    for ex in test:
        ex["split"] = "test"

    # Deterministic ordering by instance_id for readability
    train.sort(key=lambda e: e["instance_id"])
    test.sort(key=lambda e: e["instance_id"])
    return train, test


def prepare_examples(examples: list[dict]) -> list:
    """V3例をDSPy Exampleに変換。
    
    instanceはwith_inputsに含めない（forward()の引数ではなく、metricで使用する）。
    """
    import dspy
    dspys = []
    for ex in examples:
        dspys.append(dspy.Example(
            requirement=ex["requirement"],
            core_type=ex["core_type"],
            instance=ex["instance"],
            instance_id=ex["instance_id"],
            reference_value=ex["reference_value"],
            reference_solution=ex.get("reference_solution", {}),
            objective=ex.get("objective", ""),
        ).with_inputs("requirement", "core_type"))
    return dspys

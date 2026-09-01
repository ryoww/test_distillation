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

from .metrics_v3 import _select_objective_field
from .requirement_builder import build_requirement


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

    if isinstance(ref.get("objective_value"), (int, float)) and not isinstance(
        ref.get("objective_value"), bool
    ):
        return float(ref["objective_value"])
    requirements = record.get("requirements", {})
    objective_text = requirements.get("objective", "") if isinstance(requirements, dict) else ""
    _, _, value = _select_objective_field(ref, {}, objective_text=objective_text)
    return value


def convert_to_dspy_example(record: dict, *, use_reference: bool = True) -> dict:
    """V3レコードをDSPy用例に変換。"""
    core_type = core_type_from_v3(record)
    ref_value = reference_value_from_solution(record)
    enriched = dict(record)
    enriched["core_type"] = core_type
    enriched["reference_value"] = ref_value
    requirement_text = build_requirement(
        enriched,
        include_reference_values=use_reference,
    )
    requirements = record.get("requirements", {})

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


def load_and_split(
    data_dir: str,
    train_ratio: float = 0.8,
    *,
    use_reference: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """V3データをロードして訓練/テストに分割。

    splitフィールドが"train"/"test"の両方がある場合はそれを尊重。
    片方のみの場合はratio-based分割にフォールバック。
    """
    records = load_v3_data(data_dir)
    examples = [convert_to_dspy_example(r, use_reference=use_reference) for r in records]

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
    data_dir: str,
    n_train: int = 40,
    n_test: int = 20,
    seed: int = 42,
    *,
    use_reference: bool = True,
) -> Tuple[List[dict], List[dict]]:
    """core_typeで層化して n_train/n_test を選抜（再現性のためseed固定）。

    - 全レコードをcore_typeでグループ化
    - 各グループ内をseed付きで決定的にシャッフル
    - ラウンドロビンでtrain/testを埋め、両方にcore_type多様性を確保
    - train/testは互いに素
    """
    import random

    records = load_v3_data(data_dir)
    examples = [convert_to_dspy_example(r, use_reference=use_reference) for r in records]

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


def split_train_val_stratified(
    examples: list[dict],
    n_val: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """train集合をcore_typeで層化してtrain/valへ分ける。

    Why not slice the tail: load_and_split_stratified は可読性のため戻り値を
    instance_id でソートするため、末尾から切るとcore_typeが偏る。実際に
    40問を末尾13問で切った構成では、val側が3 domainだけになり、train側と
    ほとんど重ならなかった。GEPAはtrainで反省しvalで候補を選ぶので、
    この偏りは候補選択をゆがめる。

    - core_typeでグループ化し、seed付きで決定的にシャッフル
    - ラウンドロビンでvalを埋め、valにcore_type多様性を確保
    - 戻り値はどちらもinstance_id順
    """
    import random

    if n_val <= 0 or n_val >= len(examples):
        return list(examples), []

    groups: dict[str, list[dict]] = {}
    for ex in examples:
        groups.setdefault(ex["core_type"], []).append(ex)

    rng = random.Random(seed)
    ordered_types = sorted(groups)
    pools: dict[str, list[dict]] = {}
    for ct in ordered_types:
        items = sorted(groups[ct], key=lambda e: e["instance_id"])
        rng.shuffle(items)
        pools[ct] = items

    val: list[dict] = []
    while len(val) < n_val:
        progressed = False
        for ct in ordered_types:
            if len(val) >= n_val:
                break
            if pools[ct]:
                val.append(pools[ct].pop())
                progressed = True
        if not progressed:
            break

    val_ids = {ex["instance_id"] for ex in val}
    train_only: list[dict] = [ex for ex in examples if ex["instance_id"] not in val_ids]

    train_only.sort(key=lambda e: e["instance_id"])
    val.sort(key=lambda e: e["instance_id"])
    return train_only, val


def prepare_examples(examples: list[dict]) -> list:
    """V3例をDSPy Exampleに変換。

    instanceはwith_inputsに含めない（forward()の引数ではなく、metricで使用する）。
    """
    import dspy

    dspys = []
    for ex in examples:
        dspys.append(
            dspy.Example(
                requirement=ex["requirement"],
                core_type=ex["core_type"],
                instance=ex["instance"],
                instance_id=ex["instance_id"],
                reference_value=ex["reference_value"],
                reference_solution=ex.get("reference_solution", {}),
                objective=ex.get("objective", ""),
            ).with_inputs("requirement", "core_type")
        )
    return dspys

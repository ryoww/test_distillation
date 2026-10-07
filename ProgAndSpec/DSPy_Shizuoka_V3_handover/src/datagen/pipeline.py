"""雛形から新しい問題レコードを作り、既存チェッカーで検証する。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..utils.feasibility import check_feasibility_detailed
from .base import TEMPLATES, Template

GENERATOR_VERSION = 1


class ValidationError(ValueError):
    """生成した問題が雛形と同じ形になっていない、または参照解が検証を通らない。"""


def shape_signature(value: Any, depth: int = 0) -> Any:
    """型・キー集合と、浅い階層の件数だけを残した形状。値は捨てる。

    件数を見るのは深さ 2 まで（トップレベル dict の値と、その直下の行列の行）。
    さらに深い配列（被覆集合の要素数や先行タスク一覧など）は instance ごとに変わってよい。
    """
    if isinstance(value, dict):
        # Why not キーを比較: 数値キーの辞書はキー自体が乱数で変わるので件数だけを見る。
        if value and all(str(k).lstrip("-").isdigit() for k in value):
            return ("dict#", len(value), shape_signature(next(iter(value.values())), depth + 1))
        return (
            "dict",
            tuple(sorted((str(k), shape_signature(v, depth + 1)) for k, v in value.items())),
        )
    if isinstance(value, list):
        size = len(value) if depth < 3 else None
        # 先頭要素だけでは行列の後続行の長さ違いを見逃すので、全要素の形を集める。
        elements = tuple(sorted({repr(shape_signature(v, depth + 1)) for v in value}))
        return ("list", size, elements)
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def load_base(problem_dir: Path, problem_id: int) -> dict:
    return json.loads((problem_dir / f"prob_{problem_id:03d}.json").read_text(encoding="utf-8"))


def core_type_of(record: dict) -> str:
    return f"{record['domain']}_{record['math_type']}"


def make_problem(
    base: dict,
    template: Template,
    rng: random.Random,
    *,
    new_id: int,
    seed_label: str,
    split: str = "test",
) -> dict:
    """雛形の文章を引き継ぎ、instance と reference_solution だけを差し替える。"""
    instance = template.generate(rng, base["instance"])
    reference = template.solve(instance)
    record = {
        "id": new_id,
        "name": base["name"],
        "domain": base["domain"],
        "math_type": base["math_type"],
        "difficulty": base["difficulty"],
        "split": split,
        "description": base["description"],
        "requirements": base["requirements"],
        "instance": instance,
        "reference_solution": reference,
        "provenance": {
            "generator": "src.datagen",
            "version": GENERATOR_VERSION,
            "template_id": base["id"],
            "seed": seed_label,
        },
    }
    validate_problem(record, base, template)
    return record


def validate_problem(record: dict, base: dict, template: Template) -> None:
    if shape_signature(record["instance"]) != shape_signature(base["instance"]):
        raise ValidationError(f"instance shape differs from prob_{base['id']:03d}")
    reference = record["reference_solution"]
    if set(reference) != set(base["reference_solution"]):
        raise ValidationError("reference_solution keys differ from the template")
    objective = reference[template.objective_key]
    if not isinstance(objective, (int, float)) or isinstance(objective, bool):
        raise ValidationError("objective value is not numeric")
    result = check_feasibility_detailed(core_type_of(record), record["instance"], reference)
    if not result.get("verified"):
        raise ValidationError(f"checker could not verify the reference: {result['violations']}")
    if not result["feasible"] or result["violation_count"]:
        raise ValidationError(f"reference violates constraints: {result['violations']}")
    # Why not チェッカーの cost と比較: cost は目的値ではなく重量や支出を返す
    # チェッカーがある。申告値と再計算値の一致はチェッカーが violations で報告する。


def generate_dataset(
    problem_dir: Path,
    *,
    template_ids: list[int],
    per_template: int,
    seed: int,
    start_id: int,
    split: str = "test",
) -> list[dict]:
    """テンプレートごとに per_template 問を作る。乱数は (seed, template, 通番) で決まる。"""
    records = []
    next_id = start_id
    for template_id in template_ids:
        template = TEMPLATES[template_id]
        base = load_base(problem_dir, template_id)
        for k in range(per_template):
            seed_label = f"{seed}:{template_id}:{k}"
            rng = random.Random(seed_label)
            records.append(
                make_problem(
                    base, template, rng, new_id=next_id, seed_label=seed_label, split=split
                )
            )
            next_id += 1
    return records

"""保存済み Phase E プログラムが、signature 変更後も同じ形で読めることを守る。

`improve` へ `return_schema` を足したとき、`generate` 側の最適化済み instruction と
demo が壊れていないかを確かめる必要があった。ここを崩すと、記録済み4条件の
「改善後」が別物になる。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.modules import AlgorithmGenerator

BASE_DIR = Path(__file__).resolve().parents[1]
PROGRAM = BASE_DIR / "compiled_program_v3_gepa_phaseE.json"


@pytest.fixture(scope="module")
def loaded() -> AlgorithmGenerator:
    if not PROGRAM.exists():
        pytest.skip("compiled Phase E program is not present")
    program = AlgorithmGenerator()
    program.load(str(PROGRAM))
    return program


def test_generate_keeps_its_three_inputs(loaded):
    """生成側の入力契約は据え置き。ここを変えると既存条件と比較できなくなる。"""
    assert list(loaded.generate.predict.signature.input_fields) == [
        "requirement",
        "core_type",
        "parse_code",
    ]


def test_generate_keeps_its_optimized_instructions_and_demos(loaded):
    predict = loaded.generate.predict
    assert len(predict.demos) == 2
    assert len(predict.signature.instructions) > 10_000


def test_improve_now_receives_the_return_schema(loaded):
    assert "return_schema" in loaded.improve.predict.signature.input_fields


def test_improve_keeps_its_other_inputs_and_outputs(loaded):
    signature = loaded.improve.predict.signature
    for name in ("original_code", "parse_code", "feedback", "core_type"):
        assert name in signature.input_fields
    assert list(signature.output_fields) == ["reasoning", "improved_parse_code", "improved_code"]


def test_improve_carries_no_demos_so_the_added_field_breaks_nothing(loaded):
    """demo があれば入力を増やした時点で欠損フィールドが生じる。0件なら安全。"""
    assert len(loaded.improve.predict.demos) == 0

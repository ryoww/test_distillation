"""特化学習したモデル（student）を DSPy のプログラムとして扱う。

student は「system = 指示文、user = 参照値なしの問題文、出力 = コード」の形で学習しているので、
DSPy の ChatAdapter が付ける欄見出し（[[ ## requirement ## ]] など）を入れると学習時と入力が変わる。
PlainChatAdapter は学習データと同じ 2 通のメッセージだけを送り、応答からコードを取り出す。
GEPA が進化させる指示文はそのまま system に入る。
"""

from __future__ import annotations

import dspy

from src.modules import AlgorithmGenerator, ensure_parse_helpers, strip_code_fence


class SolveRequirement(dspy.Signature):
    """Write solve(instance) for the requirement."""

    requirement: str = dspy.InputField()
    algorithm_code: str = dspy.OutputField()


class PlainChatAdapter(dspy.Adapter):
    """学習データと同じ messages を組み、応答の Python を algorithm_code に入れる。"""

    def format(self, signature, demos, inputs):
        return [
            {"role": "system", "content": signature.instructions},
            {"role": "user", "content": inputs["requirement"]},
        ]

    def parse(self, signature, completion):
        return {"algorithm_code": ensure_parse_helpers(strip_code_fence(completion or ""))}


class StudentSolver(dspy.Module):
    def __init__(self, instruction: str | None = None):
        super().__init__()
        instruction = instruction or default_instruction()
        self.generate = dspy.Predict(SolveRequirement.with_instructions(instruction))

    def forward(self, requirement: str, core_type: str | None = None):
        with dspy.context(adapter=PlainChatAdapter()):
            return self.generate(requirement=requirement)


def default_instruction() -> str:
    """SFT データの system に入れた既定の指示文。"""
    return AlgorithmGenerator().generate.predict.signature.instructions

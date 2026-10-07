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


DEFAULT_REPAIR_INSTRUCTION = """Your previous solve(instance) failed the verifier. Rewrite it so it runs on the instance and satisfies every constraint listed in the feedback.
Keep the same problem reading and the same return schema. Change only what the feedback shows is wrong, keep the solver time-limited, and return the complete corrected Python code only."""


class RepairCode(dspy.Signature):
    """Rewrite solve(instance) using the verifier feedback."""

    requirement: str = dspy.InputField()
    previous_code: str = dspy.InputField()
    verifier_feedback: str = dspy.InputField()
    algorithm_code: str = dspy.OutputField()


class PlainRepairAdapter(dspy.Adapter):
    """修復段も学習時と同じ 2 通で送る。user に問題文・前のコード・検証器の指摘を並べる。"""

    def format(self, signature, demos, inputs):
        user = (
            f"{inputs['requirement']}\n\n## Your previous solve()\n```python\n"
            f"{inputs['previous_code']}\n```\n\n## Verifier feedback\n{inputs['verifier_feedback']}"
        )
        return [
            {"role": "system", "content": signature.instructions},
            {"role": "user", "content": user},
        ]

    def parse(self, signature, completion):
        return {"algorithm_code": ensure_parse_helpers(strip_code_fence(completion or ""))}


class StudentRepairSolver(dspy.Module):
    """生成 → 検証器 → 違反があれば 1 回修復、の 2 段。

    生成段は dspy の predictor にせず LM を直接呼ぶ。GEPA は named_predictors() の指示文をすべて進化
    させるので、predictor にすると SFT 済み student が張り付いている生成の指示文まで書き換えられる
    （29・31 章: 採用 0）。進化させるのは修復段の指示文だけにする。
    """

    def __init__(
        self,
        generate_instruction: str | None = None,
        repair_instruction: str | None = None,
        exec_timeout: float = 600.0,
    ):
        super().__init__()
        self.generate_instruction = generate_instruction or default_instruction()
        self.exec_timeout = exec_timeout
        self.repair = dspy.Predict(
            RepairCode.with_instructions(repair_instruction or DEFAULT_REPAIR_INSTRUCTION)
        )

    def generate_code(self, requirement: str) -> str:
        lm = dspy.settings.lm
        output = lm(
            messages=[
                {"role": "system", "content": self.generate_instruction},
                {"role": "user", "content": requirement},
            ]
        )[0]
        text = output["text"] if isinstance(output, dict) else output
        return ensure_parse_helpers(strip_code_fence(text or ""))

    # Why not 引数名 instance: dspy のコールバック包みは第 1 引数を instance（モジュール自身）と呼ぶので、
    # 同名のキーワード引数を渡すと衝突する。
    def forward(self, requirement: str, core_type: str, problem_instance: dict):
        from src.verify_loop import verify_solution

        code = self.generate_code(requirement)
        verdict = verify_solution(code, problem_instance, core_type, timeout=self.exec_timeout)
        repaired = False
        if not verdict.ok:
            with dspy.context(adapter=PlainRepairAdapter()):
                revised = self.repair(
                    requirement=requirement,
                    previous_code=code,
                    verifier_feedback=verdict.feedback,
                ).algorithm_code
            if revised.strip() and revised.strip() != code.strip():
                code, repaired = revised, True
        return dspy.Prediction(
            algorithm_code=code, first_verdict=verdict.kind, repaired=repaired
        )


def examples_with_instance(examples: list) -> list:
    """修復段は instance で検証するので、forward の入力に problem_instance としても渡す（metric は instance を読む）。"""
    return [
        dspy.Example(**ex.toDict(), problem_instance=ex.instance).with_inputs(
            "requirement", "core_type", "problem_instance"
        )
        for ex in examples
    ]

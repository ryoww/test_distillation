import dspy

from src.student_program import PlainChatAdapter, StudentSolver, default_instruction


class _EchoLM(dspy.BaseLM):
    """送られた messages を記録し、固定のコードを返す。"""

    def __init__(self):
        super().__init__(model="echo")
        self.seen = []

    def forward(self, prompt=None, messages=None, **kwargs):
        self.seen.append(messages)
        from types import SimpleNamespace

        message = SimpleNamespace(content="```python\ndef solve(instance):\n    return {}\n```", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
        return SimpleNamespace(choices=[choice], usage={}, model="echo")


def test_student_sends_only_instruction_and_requirement():
    lm = _EchoLM()
    with dspy.context(lm=lm):
        pred = StudentSolver("RULES").forward(requirement="PROBLEM TEXT")
    assert lm.seen[0] == [
        {"role": "system", "content": "RULES"},
        {"role": "user", "content": "PROBLEM TEXT"},
    ]
    assert "def solve(instance)" in pred.algorithm_code
    assert "```" not in pred.algorithm_code


def test_default_instruction_is_the_sft_system_prompt():
    assert StudentSolver().generate.signature.instructions == default_instruction()
    assert default_instruction().startswith("Write one Python function")


def test_adapter_parse_handles_empty_completion():
    assert "algorithm_code" in PlainChatAdapter().parse(None, None)

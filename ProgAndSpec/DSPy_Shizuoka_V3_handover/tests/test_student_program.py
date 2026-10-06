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


from src import student_program as sp
from src.verify_loop import Verdict


class _ScriptedLM(dspy.BaseLM):
    """呼ばれるたびに用意した応答を順に返し、送られた messages を残す。"""

    def __init__(self, replies):
        super().__init__(model="scripted")
        self.replies, self.seen = list(replies), []

    def forward(self, prompt=None, messages=None, **kwargs):
        from types import SimpleNamespace

        self.seen.append(messages)
        message = SimpleNamespace(content=self.replies.pop(0), tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
        return SimpleNamespace(choices=[choice], usage={}, model="scripted")


def test_only_the_repair_stage_is_a_predictor():
    names = [name for name, _ in sp.StudentRepairSolver("GEN", "FIX").named_predictors()]
    assert names == ["repair"]


def test_repair_runs_only_when_the_verifier_rejects(monkeypatch):
    verdicts = iter([Verdict(ok=False, kind="infeasible", feedback="capacity exceeded")])
    monkeypatch.setattr(sp, "verify_solution", lambda *a, **k: next(verdicts), raising=False)
    import src.verify_loop as vl

    monkeypatch.setattr(vl, "verify_solution", lambda *a, **k: next(verdicts))
    lm = _ScriptedLM(["```python\ndef solve(instance):\n    return {}\n```", "def solve(instance):\n    return {'ok': 1}"])
    with dspy.context(lm=lm):
        pred = sp.StudentRepairSolver("GEN", "FIX").forward(requirement="REQ", core_type="x", instance={})
    assert pred.repaired and "'ok': 1" in pred.algorithm_code
    assert lm.seen[0][0] == {"role": "system", "content": "GEN"}
    assert lm.seen[1][0] == {"role": "system", "content": "FIX"}
    assert "capacity exceeded" in lm.seen[1][1]["content"] and "REQ" in lm.seen[1][1]["content"]


def test_no_repair_when_the_verifier_accepts(monkeypatch):
    import src.verify_loop as vl

    monkeypatch.setattr(vl, "verify_solution", lambda *a, **k: Verdict(ok=True, kind="feasible"))
    lm = _ScriptedLM(["def solve(instance):\n    return {'a': 1}"])
    with dspy.context(lm=lm):
        pred = sp.StudentRepairSolver("GEN", "FIX").forward(requirement="REQ", core_type="x", instance={})
    assert not pred.repaired and len(lm.seen) == 1

"""最適化専用エージェントの実行経路: 生成 → 隔離実行 → 検証 → 修復（最大 N 回）→ フォールバック。

3 月の成果物の中核。入力は問題文（参照値なし）と instance、出力はコードと「目的値・制約検査・実行時間・
失敗理由」を添えた判定。検証器は参照値を一切使わない（`src/verify_loop.verify_solution`）ので、
正解の分からない要件にもそのまま使える。student が修復しても通らなければ、指定があればフォールバック LM
（大きいモデル）で同じ手順をもう 1 周し、それでも駄目なら failed として理由を返す。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import dspy

from src.student_program import (
    DEFAULT_REPAIR_INSTRUCTION,
    PlainRepairAdapter,
    RepairCode,
    default_instruction,
)
from src.utils.feasibility import check_feasibility_detailed
from src.verify_loop import verify_solution


@dataclass
class Attempt:
    stage: str  # generate | repair | fallback-generate | fallback-repair
    verdict: str  # feasible | unverified | exec_error | infeasible | empty
    feedback: str
    seconds: float


@dataclass
class AgentResult:
    status: str  # solved | unverified | failed
    code: str
    objective: float | None
    violations: list[str]
    attempts: list[Attempt] = field(default_factory=list)
    fallback_used: bool = False
    seconds: float = 0.0
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _chat(lm, system: str, user: str) -> str:
    output = lm(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])[0]
    return output["text"] if isinstance(output, dict) else output


class OptimizationAgent:
    def __init__(
        self,
        student_lm,
        *,
        fallback_lm=None,
        generate_instruction: str | None = None,
        repair_instruction: str | None = None,
        max_repairs: int = 2,
        exec_timeout: float = 600.0,
    ):
        self.student_lm = student_lm
        self.fallback_lm = fallback_lm
        self.generate_instruction = generate_instruction or default_instruction()
        self.repair = dspy.Predict(
            RepairCode.with_instructions(repair_instruction or DEFAULT_REPAIR_INSTRUCTION)
        )
        self.max_repairs = max_repairs
        self.exec_timeout = exec_timeout

    def _round(self, lm, requirement: str, core_type: str, instance: dict, stage: str, attempts):
        """1 つの LM で生成 → 検証 → 修復を回し、(code, verdict) を返す。"""
        from src.modules import ensure_parse_helpers, strip_code_fence

        started = time.monotonic()
        code = ensure_parse_helpers(strip_code_fence(_chat(lm, self.generate_instruction, requirement) or ""))
        verdict = verify_solution(code, instance, core_type, timeout=self.exec_timeout)
        attempts.append(Attempt(stage, verdict.kind, verdict.feedback, round(time.monotonic() - started, 1)))
        for _ in range(self.max_repairs):
            if verdict.ok:
                break
            started = time.monotonic()
            with dspy.context(lm=lm, adapter=PlainRepairAdapter()):
                revised = self.repair(
                    requirement=requirement, previous_code=code, verifier_feedback=verdict.feedback
                ).algorithm_code
            if not revised.strip() or revised.strip() == code.strip():
                break
            code = revised
            verdict = verify_solution(code, instance, core_type, timeout=self.exec_timeout)
            attempts.append(
                Attempt(f"{stage}-repair", verdict.kind, verdict.feedback, round(time.monotonic() - started, 1))
            )
        return code, verdict

    def solve(self, requirement: str, core_type: str, instance: dict) -> AgentResult:
        started = time.monotonic()
        attempts: list[Attempt] = []
        code, verdict = self._round(self.student_lm, requirement, core_type, instance, "student", attempts)
        fallback_used = False
        if not verdict.ok and self.fallback_lm is not None:
            fallback_used = True
            code, verdict = self._round(self.fallback_lm, requirement, core_type, instance, "fallback", attempts)

        objective, violations = None, list(verdict.violations)
        if verdict.ok and verdict.solution is not None:
            checked = check_feasibility_detailed(core_type, instance, verdict.solution)
            objective = checked.get("cost")
        status = "solved" if verdict.kind == "feasible" else ("unverified" if verdict.ok else "failed")
        return AgentResult(
            status=status,
            code=code,
            objective=objective,
            violations=violations,
            attempts=attempts,
            fallback_used=fallback_used,
            seconds=round(time.monotonic() - started, 1),
            failure_reason="" if verdict.ok else verdict.feedback[:500],
        )

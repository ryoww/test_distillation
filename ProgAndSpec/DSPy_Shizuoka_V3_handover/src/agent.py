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
from src.utils.hard import find_kind
from src.verify_loop import verify_solution


def detect_kind(core_type: str, instance: dict) -> str:
    """大規模問題は instance の形で種別名を引き、それ以外は core_type を種別とみなす。

    Why not core_type だけ: 大規模問題は同梱 100 問と core_type が重なる（`src/utils/hard`）。
    雛形外の問題も雛形と同じ core_type を持つので、この判定では雛形外を見分けられない。
    """
    hard = find_kind(instance)
    return hard.name if hard else core_type


@dataclass
class Attempt:
    stage: str  # generate | repair | fallback-generate | fallback-repair
    verdict: str  # feasible | unverified | exec_error | infeasible | empty
    feedback: str
    seconds: float


@dataclass
class AgentResult:
    status: str  # solved | unverified | failed | unsupported
    code: str
    objective: float | None
    violations: list[str]
    attempts: list[Attempt] = field(default_factory=list)
    fallback_used: bool = False
    seconds: float = 0.0
    failure_reason: str = ""
    kind: str = ""
    supported: bool | None = None  # None は対応表を渡していない
    routed_unsupported: bool = False

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
        supported_kinds: set[str] | None = None,
        unsupported_lm=None,
    ):
        self.student_lm = student_lm
        self.fallback_lm = fallback_lm
        self.supported_kinds = supported_kinds
        self.unsupported_lm = unsupported_lm
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
        kind = detect_kind(core_type, instance)
        supported = None if self.supported_kinds is None else kind in self.supported_kinds
        if supported is False:
            # Why not student に解かせる: 学習していない種別では student も 12B も 0 問だった
            # （RESCORE_REPORT 33.5 節）。黙って誤答を返さず、対応外と明示するか、指定の LM に回す。
            if self.unsupported_lm is None:
                return AgentResult(
                    status="unsupported",
                    code="",
                    objective=None,
                    violations=[],
                    seconds=round(time.monotonic() - started, 1),
                    failure_reason=f"kind '{kind}' is outside the kinds this student was trained on",
                    kind=kind,
                    supported=False,
                )
            code, verdict = self._round(
                self.unsupported_lm, requirement, core_type, instance, "unsupported-route", attempts
            )
            return self._finish(code, verdict, core_type, instance, attempts, started, kind, False,
                                fallback_used=False, routed=True)
        code, verdict = self._round(self.student_lm, requirement, core_type, instance, "student", attempts)
        fallback_used = False
        if not verdict.ok and self.fallback_lm is not None:
            fallback_used = True
            code, verdict = self._round(self.fallback_lm, requirement, core_type, instance, "fallback", attempts)
        return self._finish(code, verdict, core_type, instance, attempts, started, kind, supported,
                            fallback_used=fallback_used, routed=False)

    def _finish(self, code, verdict, core_type, instance, attempts, started, kind, supported, *,
                fallback_used: bool, routed: bool) -> AgentResult:
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
            kind=kind,
            supported=supported,
            routed_unsupported=routed,
        )


def load_supported_kinds(path) -> set[str]:
    """`scripts/list_supported_kinds.py` が書いた対応表を読む。"""
    import json
    from pathlib import Path

    return set(json.loads(Path(path).read_text(encoding="utf-8"))["kinds"])

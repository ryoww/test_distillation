"""提出前に解を検証し、落ちていれば作り直させるループ。

生成した `solve()` は、これまで一度も実行されないまま採点へ渡っていた。
`improve_forward` は実装済みだが評価経路から呼ばれておらず、実行結果を
生成へ戻す経路だけが欠けていた。ここでそれを繋ぐ。

検証に使えるのは instance から導ける情報だけである:

- コードが走るか（safe_exec の traceback）
- 解が制約を満たすか（feasibility チェッカーの違反メッセージ）
- 解が自分で申告した目的値と、構造から再計算した値が一致するか

参照値と参照解は渡さない。渡せば「検証して直した」のではなく
「答えを見て直した」になる。

Why not shell out to a coding agent: 検証信号はすでにこのプロセス内にある。
外部エージェントへ渡すには結局この結果を渡すことになるうえ、共用GPU機で
モデル生成コードをサンドボックス外で実行することになる。reviser は
差し替え可能にしてあるので、必要ならそこへ外部ツールを挿せる。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .utils.feasibility import check_feasibility_detailed
from .utils.safe_exec import safe_run

MAX_FEEDBACK_CHARS = 1200
MAX_VIOLATIONS = 6


class SupportsImprove(Protocol):
    """`improve_forward` を持つ生成器。"""

    def improve_forward(
        self, original_code: str, parse_code: str, feedback: str, core_type: str
    ) -> Any: ...


@dataclass
class Verdict:
    """1回の検証結果。"""

    ok: bool
    kind: str
    feedback: str = ""
    solution: Any = None
    violations: list[str] = field(default_factory=list)


@dataclass
class RepairAttempt:
    """1回の生成と、その検証結果。"""

    index: int
    kind: str
    ok: bool
    feedback: str


def _truncate(text: str, limit: int = MAX_FEEDBACK_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " …(truncated)"


def verify_solution(
    code: str,
    instance: dict,
    core_type: str,
    *,
    timeout: float = 60.0,
) -> Verdict:
    """コードを走らせ、制約充足まで見る。参照値は一切使わない。"""
    ok, result = safe_run(code, instance, timeout=timeout)

    if not ok:
        return Verdict(
            ok=False,
            kind="exec_error",
            feedback=_truncate(
                "Your solve() raised before returning. Fix the code so it runs on this "
                f"instance.\n\n{result}"
            ),
        )

    if isinstance(result, dict) and "error" in result:
        return Verdict(
            ok=False,
            kind="exec_error",
            feedback=_truncate(
                "Your solve() caught an exception internally and returned an error dict. "
                f"Fix the underlying cause instead of swallowing it.\n\n{result.get('error')}"
            ),
            solution=result,
        )

    if result is None or (isinstance(result, (list, dict, str)) and len(result) == 0):
        return Verdict(
            ok=False,
            kind="empty",
            feedback=(
                "Your solve() returned an empty solution. Return every field of the "
                "return schema, filled with an actual solution."
            ),
            solution=result,
        )

    checked = check_feasibility_detailed(core_type, instance, result)
    violations = [str(v) for v in checked.get("violations", [])][:MAX_VIOLATIONS]

    if not checked.get("verified", True):
        # 形状を判定できなかっただけなので、違反として突き返さない。
        return Verdict(ok=True, kind="unverified", solution=result, violations=violations)

    if not checked.get("feasible", True):
        listed = "\n".join(f"- {v}" for v in violations)
        return Verdict(
            ok=False,
            kind="infeasible",
            feedback=_truncate(
                "Your solution runs but breaks the problem's own constraints. "
                "These checks read only the instance you were given:\n\n" + listed
            ),
            solution=result,
            violations=violations,
        )

    return Verdict(ok=True, kind="feasible", solution=result, violations=violations)


def generate_verified(
    program: SupportsImprove,
    requirement: str,
    core_type: str,
    instance: dict,
    *,
    max_attempts: int = 2,
    timeout: float = 60.0,
    return_schema: str = "",
    reviser: Callable[..., Any] | None = None,
) -> tuple[str, str, list[RepairAttempt]]:
    """生成 → 検証 → 必要なら作り直し、を最大 max_attempts 回まで。

    Returns:
        (algorithm_code, parse_code, attempts)

    max_attempts=1 なら従来どおりの1発生成で、検証結果は記録だけする。

    return_schema は呼び出し側が組み立てて渡す。ここで reference_solution を
    受け取らないのは、目的値へ触れる経路をこの関数に作らないため。
    """
    prediction = program(requirement=requirement, core_type=core_type)
    code = prediction.algorithm_code
    parse_code = getattr(prediction, "parse_code", "")
    revise = reviser or program.improve_forward

    attempts: list[RepairAttempt] = []
    for index in range(1, max(max_attempts, 1) + 1):
        verdict = verify_solution(code, instance, core_type, timeout=timeout)
        attempts.append(
            RepairAttempt(index=index, kind=verdict.kind, ok=verdict.ok, feedback=verdict.feedback)
        )
        if verdict.ok or index >= max_attempts:
            break
        revised = revise(
            original_code=code,
            parse_code=parse_code,
            feedback=verdict.feedback,
            core_type=core_type,
            return_schema=return_schema,
        )
        revised_code = getattr(revised, "algorithm_code", "") or ""
        if not revised_code.strip() or revised_code.strip() == code.strip():
            # Why stop: 同じコードが返ったら、もう一周しても同じ結果にしかならない。
            break
        code = revised_code

    return code, parse_code, attempts


def summarize_attempts(attempts: list[RepairAttempt]) -> dict:
    """first-pass と修復後を別々に読める形へまとめる。"""
    if not attempts:
        return {"attempts": 0, "first_pass_ok": None, "final_ok": None, "repaired": False}
    first, last = attempts[0], attempts[-1]
    return {
        "attempts": len(attempts),
        "first_pass_ok": first.ok,
        "first_pass_kind": first.kind,
        "final_ok": last.ok,
        "final_kind": last.kind,
        "repaired": (not first.ok) and last.ok,
    }

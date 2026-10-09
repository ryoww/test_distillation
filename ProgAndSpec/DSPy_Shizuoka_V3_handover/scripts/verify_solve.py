#!/usr/bin/env python3
"""作業ディレクトリの solve.py を instance.json で走らせ、参照値を使わない検証器の判定を表示する。

外部のエージェント（DeepSeek Harness など）に渡す道具。エージェントの実行経路（src/agent.py）と同じ
`verify_solution` を使い、可行かどうか、違反、解から計算し直した目的値だけを返す。参照値は表示しない。

  python scripts/verify_solve.py <作業ディレクトリ>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.feasibility import check_feasibility_detailed
from src.verify_loop import verify_solution


def report(workspace: Path) -> str:
    meta = json.loads((workspace / "meta.json").read_text(encoding="utf-8"))
    instance = json.loads((workspace / "instance.json").read_text(encoding="utf-8"))
    solve_path = workspace / "solve.py"
    if not solve_path.exists():
        return "verdict: missing\nWrite solve.py that defines solve(instance) first."
    verdict = verify_solution(
        solve_path.read_text(encoding="utf-8"),
        instance,
        meta["core_type"],
        timeout=meta.get("timeout", 300),
    )
    lines = [f"verdict: {verdict.kind}"]
    if verdict.feedback:
        lines.append(verdict.feedback)
    if verdict.ok and isinstance(verdict.solution, dict):
        cost = check_feasibility_detailed(meta["core_type"], instance, verdict.solution).get("cost")
        if cost is not None:
            lines.append(f"objective recomputed from your solution: {cost}")
    return "\n".join(lines)


def main() -> int:
    workspace = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(report(workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

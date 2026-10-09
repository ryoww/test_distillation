#!/usr/bin/env python3
"""作業ディレクトリの solve.py を instance.json で走らせ、参照値を使わない検証器の判定を表示する。

外部のエージェント（DeepSeek Harness など）に渡す道具。エージェントの実行経路（src/agent.py）と同じ
`verify_solution` を使い、可行かどうか、違反、解から計算し直した目的値だけを返す。参照値は表示しない。

  python scripts/verify_solve.py <作業ディレクトリ>

solve() は子プロセスで走らせ、時間切れなら子プロセスごと止める。
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import verify_loop
from src.utils.feasibility import check_feasibility_detailed
from src.utils.safe_exec import _validate_ast, _worker
from src.verify_loop import verify_solution


class _Box:
    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)


def _run_inline(code: str, instance: dict, timeout: float = 0.0) -> tuple[bool, object]:
    """safe_run と同じ検査と制限で、この process の中で solve を走らせる（時間切れは親が扱う）。"""
    # Why not safe_run: multiprocessing の Queue は /dev/shm にセマフォを作るが、外部エージェントの
    # サンドボックスはそこへの書き込みを禁じるので、2 回目以降の実行が権限エラーで落ちた。
    code = textwrap.dedent(code)
    ok, message = _validate_ast(code)
    if not ok:
        return False, message
    box = _Box()
    _worker(code, instance, box)
    if not box.items:
        return False, "No result (solve produced nothing)"
    status, payload = box.items[0]
    return status == "ok", payload


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
    if len(sys.argv) > 2 and sys.argv[1] == "--inner":
        verify_loop.safe_run = _run_inline
        print(report(Path(sys.argv[2])))
        return 0
    workspace = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    limit = json.loads((workspace / "meta.json").read_text(encoding="utf-8")).get("timeout", 300)
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--inner", str(workspace)],
            capture_output=True,
            text=True,
            timeout=limit + 30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"verdict: exec_error\nYour solve() did not finish within {limit} s. "
              "Bound its loops and give every solver a time limit.")
        return 0
    print(proc.stdout, end="")
    if proc.returncode != 0 and not proc.stdout.strip():
        print(f"verdict: exec_error\n{proc.stderr[-2000:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

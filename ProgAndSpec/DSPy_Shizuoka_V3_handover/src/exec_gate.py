"""生成したコードの実行（サンドボックス）の同時実行数を、生成の並列度とは別に絞る。

時間制限付きのソルバーは使える CPU 時間で解の質が変わるので、生成を 8 並列で回しても、実行は少数ずつにしないと
同じコードの判定が混み具合で変わる（RESCORE_REPORT 34.2 節）。同じプロセス内のスレッドで共有する。
"""

from __future__ import annotations

import threading
from contextlib import nullcontext

_gate: threading.BoundedSemaphore | None = None


def set_exec_concurrency(limit: int | None) -> None:
    """None または 0 で無制限（従来どおり）。"""
    global _gate
    _gate = threading.BoundedSemaphore(limit) if limit else None


def exec_slot():
    """`with exec_slot():` の中だけで実行する。"""
    return _gate if _gate is not None else nullcontext()

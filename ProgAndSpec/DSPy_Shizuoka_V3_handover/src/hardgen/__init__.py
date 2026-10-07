"""大規模問題集（data/problems_hard）の instance 生成器。

28 問は 1 問 1 instance で厳密解も生成器も付いていない。ここでは各種別について、元 instance と
同じ形（キー構成と件数）で、同じ分布から新しい instance を作る生成器を登録する。参照解は
`scripts/reference_hard_instances.py` が教師コード（検証器を通った既存の solve()）を走らせて付ける。
登録の副作用が必要なので、種別モジュールは読み込むだけでよい。
"""

from __future__ import annotations

import random
from collections.abc import Callable

Generator = Callable[[random.Random, dict], dict]
GENERATORS: dict[str, Generator] = {}


def register(kind: str) -> Callable[[Generator], Generator]:
    """種別名（src.utils.hard の HardKind.name）に生成器を結びつける。"""

    def decorator(fn: Generator) -> Generator:
        GENERATORS[kind] = fn
        return fn

    return decorator


from . import (  # noqa: F401
    facility,
    network_finance,
    production,
    rosters,
    routing,
    shop_cutting,
)

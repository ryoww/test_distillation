"""大規模問題集（data/problems_hard）の検証器。

同梱 100 問と core_type が重なる問題があるので、core_type ではなく instance の形で問題種別を
判定し、種別ごとの検証器（制約の検査と目的値の再計算）へ振り分ける。各検証器は
`register_kind` で登録する。登録の副作用が必要なので、種別モジュールは読み込むだけでよい。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardKind:
    name: str
    detect: Callable[[dict], bool]
    check: Callable[[dict, Any], dict]


KINDS: list[HardKind] = []


def register_kind(
    name: str, detect: Callable[[dict], bool], check: Callable[[dict, Any], dict]
) -> None:
    """種別を登録する。detect は instance だけを見て真偽を返し、他の種別と排他であること。"""
    KINDS.append(HardKind(name, detect, check))


def _matches(kind: HardKind, instance: dict) -> bool:
    try:
        return bool(kind.detect(instance))
    except (KeyError, TypeError, AttributeError, ValueError):
        return False  # 判定の失敗は「該当なし」と同じ扱い


def find_kind(instance: Any) -> HardKind | None:
    if not isinstance(instance, dict):
        return None
    return next((kind for kind in KINDS if _matches(kind, instance)), None)


from . import (  # noqa: F401
    facility,
    network_finance,
    production,
    rosters,
    routing,
    shop_cutting,
)

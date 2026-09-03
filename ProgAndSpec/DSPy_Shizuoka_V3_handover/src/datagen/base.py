"""問題テンプレートの登録と共通ヘルパー。

テンプレートは「既存の prob_XXX.json を雛形にして、同じ形状の instance を乱数で作り、
厳密ソルバーで参照解を付ける」単位である。description と requirements は雛形の
文章をそのまま使うので、文章に書かれている件数や容量は雛形の値を保ち、
文章に書かれていない数値だけを乱数で置き換える。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class SolveError(RuntimeError):
    """ソルバーが最適性を証明できなかった。"""


@dataclass(frozen=True)
class Template:
    problem_id: int
    objective_key: str
    generate: Callable[[random.Random, dict], dict]
    solve: Callable[[dict], dict]


TEMPLATES: dict[int, Template] = {}


def register(problem_id: int, objective_key: str) -> Callable:
    """`(generate, solve)` の組を返す関数をテンプレートとして登録する。"""

    def decorator(factory: Callable[[], tuple[Callable, Callable]]):
        generate, solve = factory()
        TEMPLATES[problem_id] = Template(problem_id, objective_key, generate, solve)
        return factory

    return decorator


def int_matrix(rng: random.Random, rows: int, cols: int, low: int, high: int) -> list[list[int]]:
    return [[rng.randint(low, high) for _ in range(cols)] for _ in range(rows)]


def int_list(rng: random.Random, n: int, low: int, high: int) -> list[int]:
    return [rng.randint(low, high) for _ in range(n)]


def partition(rng: random.Random, total: int, parts: int, minimum: int) -> list[int]:
    """total を parts 個の minimum 以上の整数へ分ける。"""
    if total < parts * minimum:
        raise ValueError("total too small to partition")
    weights = [rng.random() for _ in range(parts)]
    scale = sum(weights)
    spare = total - parts * minimum
    values = [minimum + int(spare * w / scale) for w in weights]
    values[-1] += total - sum(values)
    return values


def retry(rng: random.Random, make: Callable[[], Any], ok: Callable[[Any], bool], limit: int = 200):
    """条件を満たす instance が出るまで生成を繰り返す。"""
    for _ in range(limit):
        candidate = make()
        if ok(candidate):
            return candidate
    raise RuntimeError("could not generate an acceptable instance")


def cp_sat_solver(time_limit: float = 30.0):
    from ortools.sat.python import cp_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    # Why not 並列探索: 同点の最適解が worker のタイミングで入れ替わり、同じ seed でも
    # 別プロセスで違う参照解になる。instance は小さいので 1 worker で十分速い。
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    return cp_model, solver


def require_optimal(cp_model, status: int) -> None:
    if status != cp_model.OPTIMAL:
        raise SolveError(f"CP-SAT did not prove optimality (status={status})")

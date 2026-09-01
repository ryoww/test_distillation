"""制約充足チェックフレームワーク。

V2では40+の問題タイプを扱うため、登録型のフレームワークを採用。
- register_feasibility_check(core_type, fn) でチェック関数を登録
- check_feasibility(core_type, instance, solution) で実行
- 未登録のcore_typeはTrueを返す（LLMの出力を信頼するモード）

NOTE: V2のsolution形式多样（list[list], list[dict], dict, float, str）のため、
厳密な制約チェックは各core_typeの実装次第。未登録時はTrueを返す。
"""

from __future__ import annotations

from typing import Any, Callable

CHECKERS: dict[str, Callable[[dict, Any], bool]] = {}
CHECKERS_DETAILED: dict[str, Callable[[dict, Any], dict]] = {}


def register_feasibility_check(core_type: str, fn: Callable[[dict, Any], bool]) -> None:
    """core_type に対して制約チェック関数を登録する。"""
    CHECKERS[core_type] = fn


def register_feasibility_check_detailed(core_type: str, fn: Callable[[dict, Any], dict]) -> None:
    """core_type に対して詳細制約チェック関数を登録する。

    戻り値: dict with keys:
        - feasible: bool (全体として充足か)
        - partial_score: float (0.0-1.0, 部分的充足度)
        - violation_count: int (違反数)
        - total_constraints: int (総制約数)
        - violations: list (違反の詳細)
        - cost: float | None (コスト値があれば)
    """
    CHECKERS_DETAILED[core_type] = fn


def check_feasibility(core_type: str, instance: dict, solution: Any) -> bool:
    """
    core_type に対応する制約チェックを実行。
    未登録の場合は True を返す（LLMの出力を信頼）。
    """
    fn = CHECKERS.get(core_type)
    if fn is None:
        # V2では未登録のcore_typeが多いので、Trueを返す
        # solutionがNoneや空でないことのみチェック
        if solution is None:
            return False
        if isinstance(solution, (list, str, dict)) and len(solution) == 0:
            return False
        return True
    try:
        return fn(instance, solution)
    except Exception:
        return False


def check_feasibility_detailed(core_type: str, instance: dict, solution: Any) -> dict:
    """
    詳細制約チェック: 部分的充足度も返す。

    Returns:
        dict with keys: feasible, partial_score, violation_count, total_constraints, violations, cost
    """
    fn = CHECKERS_DETAILED.get(core_type)
    if fn is None:
        # Fallback: 基本的なチェックのみ
        if solution is None:
            return {
                "feasible": False,
                "partial_score": 0.0,
                "violation_count": 1,
                "total_constraints": 1,
                "violations": ["solution is None"],
                "cost": None,
            }
        if isinstance(solution, (list, str, dict)) and len(solution) == 0:
            return {
                "feasible": False,
                "partial_score": 0.0,
                "violation_count": 1,
                "total_constraints": 1,
                "violations": ["solution is empty"],
                "cost": None,
            }
        return {
            "feasible": True,
            "verified": False,
            "partial_score": 1.0,
            "violation_count": 0,
            "total_constraints": 0,
            "violations": ["no feasibility checker registered"],
            "cost": None,
        }

    try:
        result = fn(instance, solution)
        # Ensure required keys
        if "feasible" not in result:
            result["feasible"] = True
        if "partial_score" not in result:
            result["partial_score"] = 1.0 if result["feasible"] else 0.0
        if "violation_count" not in result:
            result["violation_count"] = 0
        if "total_constraints" not in result:
            result["total_constraints"] = 0
        if "violations" not in result:
            result["violations"] = []
        result["verified"] = True
        return result
    except Exception as e:
        return {
            "feasible": False,
            "verified": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": [f"feasibility checker failed: {e}"],
            "cost": None,
        }


# --- Built-in checkers for common types ---


def check_tsp(instance: dict, solution: Any) -> bool:
    """TSP: 全都市を1度ずつ訪問。"""
    if not isinstance(solution, (list, tuple)):
        return False
    coords = instance.get("coords") or instance.get("customers")
    if not coords:
        return False
    n = len(coords)
    if len(solution) == n:
        return set(solution) == set(range(n))
    if len(solution) == n + 1 and solution[0] == solution[-1]:
        return set(solution[:-1]) == set(range(n))
    return False


def check_tsp_detailed(instance: dict, solution: Any) -> dict:
    """TSP: 詳細制約チェック - 訪問率を部分スコアとして返す。"""
    violations = []
    total_constraints = 3  # format, visit_all, visit_once

    # Format check
    if not isinstance(solution, (list, tuple)):
        violations.append("solution is not a list/tuple")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": total_constraints,
            "violations": violations,
            "cost": None,
        }

    coords = instance.get("coords") or instance.get("customers")
    if not coords:
        return {
            "feasible": True,
            "partial_score": 1.0,
            "violation_count": 0,
            "total_constraints": total_constraints,
            "violations": [],
            "cost": None,
        }

    n = len(coords)
    visited_nodes = set()

    # Check visit coverage
    if len(solution) == n:
        visited_nodes = set(solution)
    elif len(solution) == n + 1 and solution[0] == solution[-1]:
        visited_nodes = set(solution[:-1])
    else:
        violations.append(f"path length mismatch: got {len(solution)}, expected {n} or {n + 1}")

    # Calculate visit rate
    expected_nodes = set(range(n))
    visit_rate = len(visited_nodes & expected_nodes) / n if n > 0 else 0

    # Check for unvisited nodes
    unvisited = expected_nodes - visited_nodes
    if unvisited:
        violations.append(f"unvisited nodes: {sorted(unvisited)[:5]}... ({len(unvisited)} total)")

    # Check for duplicate visits
    if len(solution) == n:
        duplicates = n - len(set(solution))
        if duplicates > 0:
            violations.append(f"duplicate visits: {duplicates} nodes visited multiple times")

    violation_count = len(violations)
    feasible = violation_count == 0

    # Partial score: weighted by visit rate and no duplicates
    partial_score = visit_rate * 0.8
    if len(solution) == n or (len(solution) == n + 1 and solution[0] == solution[-1]):
        partial_score += 0.1  # format bonus
    if not violations or all("duplicate" not in v for v in violations):
        partial_score += 0.1  # no duplicate bonus

    return {
        "feasible": feasible,
        "partial_score": min(partial_score, 1.0),
        "violation_count": violation_count,
        "total_constraints": total_constraints,
        "violations": violations,
        "cost": None,
    }


def check_cvrp(instance: dict, solution: Any) -> bool:
    """CVRP: 全顧客訪問、容量制約。"""
    if not isinstance(solution, (list, tuple)):
        return False
    customers = instance.get("customers", [])
    demands = instance.get("demands", [])
    capacity = instance.get("capacity", 0)
    n = len(customers)
    if len(demands) != n:
        return False
    visited: list[int] = []
    for route in solution:
        if not isinstance(route, (list, tuple)):
            return False
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            return False
        route_demand = 0
        for node in route[1:-1]:
            if not (1 <= node <= n):
                return False
            route_demand += demands[node - 1]
            visited.append(node)
        if route_demand > capacity:
            return False
    return sorted(visited) == list(range(1, n + 1))


def check_cvrp_detailed(instance: dict, solution: Any) -> dict:
    """CVRP: 詳細制約チェック - 訪問率と容量遵守率を部分スコアとして返す。"""
    violations = []
    total_constraints = 4  # format, visit_all, capacity, route_format

    customers = instance.get("customers", [])
    demands = instance.get("demands", [])
    capacity = instance.get("capacity", 0)
    n = len(customers)

    # Format check
    if not isinstance(solution, (list, tuple)):
        violations.append("solution is not a list/tuple")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": total_constraints,
            "violations": violations,
            "cost": None,
        }

    if len(demands) != n:
        violations.append(f"demand count mismatch: {len(demands)} vs {n} customers")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": total_constraints,
            "violations": violations,
            "cost": None,
        }

    visited = set()
    capacity_violations = 0
    route_violations = 0
    total_routes = len(solution)

    for route in solution:
        if not isinstance(route, (list, tuple)):
            route_violations += 1
            violations.append(f"route is not a list/tuple: {route}")
            continue

        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            route_violations += 1
            violations.append(f"route doesn't start/end at depot: {route[:3]}...")
            continue

        route_demand = 0
        for node in route[1:-1]:
            if not (1 <= node <= n):
                violations.append(f"invalid node in route: {node}")
                continue
            route_demand += demands[node - 1]
            visited.add(node)

        if route_demand > capacity:
            capacity_violations += 1
            violations.append(f"capacity violation: {route_demand} > {capacity}")

    # Visit coverage
    expected = set(range(1, n + 1))
    visit_rate = len(visited & expected) / n if n > 0 else 0

    # Route format compliance
    route_compliance = (total_routes - route_violations) / total_routes if total_routes > 0 else 1.0

    # Capacity compliance
    capacity_compliance = (
        (total_routes - capacity_violations) / total_routes if total_routes > 0 else 1.0
    )

    violation_count = len(violations)
    feasible = violation_count == 0 and visit_rate == 1.0

    # Partial score: weighted combination
    partial_score = visit_rate * 0.4 + route_compliance * 0.3 + capacity_compliance * 0.3

    return {
        "feasible": feasible,
        "partial_score": min(partial_score, 1.0),
        "violation_count": violation_count,
        "total_constraints": total_constraints,
        "violations": violations,
        "cost": None,
    }


# Register built-ins
register_feasibility_check("tsp", check_tsp)
register_feasibility_check("cvrp", check_cvrp)
register_feasibility_check_detailed("tsp", check_tsp_detailed)
register_feasibility_check_detailed("cvrp", check_cvrp_detailed)


# ============================================================
# V3 core_type specific detailed checkers
# ============================================================


def check_scheduling_dp_detailed(instance: dict, solution: any) -> dict:
    """スケジューリング_動的計画法 (single-machine): 全ジョブ割当チェック。"""
    violations = []
    jobs = instance.get("jobs", [])
    if not jobs:
        return {
            "feasible": True,
            "partial_score": 1.0,
            "violation_count": 0,
            "total_constraints": 0,
            "violations": [],
            "cost": None,
        }

    n = len(jobs)

    # Extract sequence
    if isinstance(solution, list):
        seq = solution
    elif isinstance(solution, dict):
        seq = (
            solution.get("sequence")
            or solution.get("optimal_sequence")
            or solution.get("order")
            or []
        )
    else:
        violations.append(f"solution type unknown: {type(solution).__name__}")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    if not seq:
        violations.append("empty sequence (no jobs scheduled)")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    job_ids = {j.get("id", i) for i, j in enumerate(jobs)}
    seq_ids = set(seq)
    coverage = len(seq_ids & job_ids) / n

    if coverage < 1.0:
        violations.append(f"only {len(seq_ids & job_ids)}/{n} jobs scheduled")

    return {
        "feasible": coverage == 1.0,
        "partial_score": coverage,
        "violation_count": len(violations),
        "total_constraints": 1,
        "violations": violations,
        "cost": None,
    }


def check_scheduling_jobshop_detailed(instance: dict, solution: any) -> dict:
    """スケジューリング_(混合)整数計画 (job shop): スケジュール構造チェック。"""
    violations = []
    jobs = instance.get("jobs", [])
    if not jobs:
        return {
            "feasible": True,
            "partial_score": 1.0,
            "violation_count": 0,
            "total_constraints": 0,
            "violations": [],
            "cost": None,
        }

    n = len(jobs)

    if not isinstance(solution, dict):
        violations.append(f"solution is not a dict: {type(solution).__name__}")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    schedule = solution.get("schedule", solution.get("assignments", solution.get("assignment")))
    if not schedule:
        violations.append("empty or missing schedule/assignments field")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    # Count scheduled jobs
    scheduled_count = 0
    if isinstance(schedule, dict):
        scheduled_count = len(schedule)
    elif isinstance(schedule, list):
        scheduled_count = len(schedule)

    if scheduled_count == 0:
        violations.append("no jobs in schedule")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    # Check makespan is present (main objective)
    makespan = solution.get("makespan", solution.get("total_time"))
    if makespan is None:
        violations.append("no makespan field in solution")
    elif makespan == 0:
        violations.append("makespan=0 (suspiciously low)")

    coverage = min(1.0, scheduled_count / n)
    partial = 0.5 + coverage * 0.4
    if makespan is not None and makespan > 0:
        partial += 0.1

    return {
        "feasible": len(violations) == 0 and coverage == 1.0,
        "partial_score": min(partial, 1.0),
        "violation_count": len(violations),
        "total_constraints": 3,
        "violations": violations,
        "cost": None,
    }


def check_scheduling_graph_detailed(instance: dict, solution: any) -> dict:
    """スケジューリング_グラフ最適化 (critical path): project_duration チェック。"""
    violations = []

    if isinstance(solution, (int, float)):
        if solution == 0:
            violations.append("project_duration=0 (suspicious)")
        return {
            "feasible": len(violations) == 0,
            "partial_score": 1.0 if not violations else 0.3,
            "violation_count": len(violations),
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    if not isinstance(solution, dict):
        violations.append(f"solution is not a dict or number: {type(solution).__name__}")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    duration = solution.get("project_duration", solution.get("duration", solution.get("makespan")))
    if duration is None:
        violations.append("no project_duration/duration/makespan field")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }
    if duration == 0:
        violations.append("project_duration=0 (suspicious)")
        return {
            "feasible": False,
            "partial_score": 0.2,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    return {
        "feasible": True,
        "partial_score": 1.0,
        "violation_count": 0,
        "total_constraints": 1,
        "violations": [],
        "cost": None,
    }


def check_vrp_v3_detailed(instance: dict, solution: any) -> dict:
    """配送・輸送_混合整数計画/確率最適化: ルート構造 + 顧客カバー率チェック。"""
    violations = []
    customers = instance.get("customers", [])
    if not customers:
        return {
            "feasible": True,
            "partial_score": 1.0,
            "violation_count": 0,
            "total_constraints": 0,
            "violations": [],
            "cost": None,
        }
    n = len(customers)
    customer_ids = {c.get("id", i) for i, c in enumerate(customers)}

    if not isinstance(solution, dict):
        violations.append(f"solution is not a dict: {type(solution).__name__}")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    routes = solution.get("routes")
    if not routes:
        violations.append("empty or missing routes field")
        return {
            "feasible": False,
            "partial_score": 0.0,
            "violation_count": 1,
            "total_constraints": 1,
            "violations": violations,
            "cost": None,
        }

    # Extract visited customer ids from various route formats
    visited = set()
    route_list = list(routes.values()) if isinstance(routes, dict) else routes
    for r in route_list:
        if isinstance(r, dict):
            path = r.get("route") or r.get("path") or r.get("customers") or r.get("nodes") or []
        elif isinstance(r, list):
            path = r
        else:
            continue
        for node in path:
            if isinstance(node, dict):
                nid = node.get("id")
            else:
                nid = node
            if nid is not None and nid != 0 and nid in customer_ids:
                visited.add(nid)

    coverage = len(visited) / n if n > 0 else 0

    # Check total distance
    total_dist = solution.get("total_distance", solution.get("total_cost", solution.get("cost")))
    if total_dist is None:
        violations.append("no total_distance/total_cost field")
    elif total_dist == 0:
        violations.append("total_distance=0 (suspicious - customers may not be actually visited)")
    elif total_dist == float("inf"):
        violations.append("total_distance=inf (solver failure)")

    if coverage < 1.0:
        violations.append(f"only {len(visited)}/{n} customers visited (coverage={coverage:.1%})")

    feasible = coverage == 1.0 and total_dist is not None and 0 < total_dist < float("inf")
    partial_score = coverage * 0.7
    if total_dist is not None and 0 < total_dist < float("inf"):
        partial_score += 0.3

    return {
        "feasible": feasible,
        "partial_score": min(partial_score, 1.0),
        "violation_count": len(violations),
        "total_constraints": 3,
        "violations": violations,
        "cost": None,
    }


# Register V3 core_type detailed checkers
register_feasibility_check_detailed("スケジューリング_動的計画法", check_scheduling_dp_detailed)
register_feasibility_check_detailed("スケジューリング_貪欲法", check_scheduling_dp_detailed)
register_feasibility_check_detailed("スケジューリング_整数計画", check_scheduling_jobshop_detailed)
register_feasibility_check_detailed(
    "スケジューリング_混合整数計画", check_scheduling_jobshop_detailed
)
register_feasibility_check_detailed(
    "スケジューリング_グラフ最適化", check_scheduling_graph_detailed
)
register_feasibility_check_detailed(
    "スケジューリング_確率最適化", check_scheduling_jobshop_detailed
)
register_feasibility_check_detailed("配送・輸送_混合整数計画", check_vrp_v3_detailed)
register_feasibility_check_detailed("配送・輸送_確率最適化", check_vrp_v3_detailed)

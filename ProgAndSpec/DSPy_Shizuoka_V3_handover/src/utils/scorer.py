"""V3 スコア計算モジュール。

V3のreference_solutionは多様な構造を持つため、core_typeごとに
スコア計算関数を登録するフレームワーク。

V3のcore_type: domain_math_type の組み合わせ
- スケジューリング_動的計画法
- スケジューリング_貪欲法  
- スケジューリング_混合整数計画
- スケジューリング_グラフ最適化
- 配送・輸送_混合整数計画
- 配送・輸送_確率的最適化
"""
from __future__ import annotations

import math
from typing import Any, Callable

SCORERS: dict[str, Callable[[dict, Any], float]] = {}


def register_scorer(core_type: str, fn: Callable[[dict, Any], float]) -> None:
    SCORERS[core_type] = fn


def has_scorer(core_type: str) -> bool:
    return core_type in SCORERS


def compute_score(core_type: str, instance: dict, solution: Any) -> float | None:
    """core_type に対応するスコア計算を実行。未登録の場合は None を返す。"""
    fn = SCORERS.get(core_type)
    if fn is None:
        return None
    try:
        result = fn(instance, solution)
        if result is None:
            return None
        return float(result)
    except Exception:
        return None


# ============================================================
# Generic scorer: reference_solutionとの比較
# ============================================================

def generic_reference_comparison(instance: dict, solution: Any, reference_solution: dict) -> float:
    """
    汎用的なスコア計算: reference_solutionとの比較。
    
    最小化問題: solution_cost / reference_cost の逆数に近いほど良い
    スコア = reference_cost / solution_cost (>=1.0=良好, <=1.0=悪化)
    
    solutionが数値の場合: そのままコストとして使用
    solutionがdictの場合: objective値を抽出
    """
    if isinstance(solution, (int, float)):
        sol_cost = float(solution)
    elif isinstance(solution, dict):
        # Try to extract cost from solution dict
        for key in ["cost", "objective", "value", "total_distance", "total_cost", 
                     "makespan", "total_delay", "project_duration", "num_trains", "score"]:
            if key in solution and isinstance(solution[key], (int, float)):
                sol_cost = float(solution[key])
                break
        else:
            # Fallback: first numeric value
            sol_cost = None
            for v in solution.values():
                if isinstance(v, (int, float)):
                    sol_cost = float(v)
                    break
            if sol_cost is None:
                return None
    else:
        return None
    
    # Get reference cost
    ref_cost = None
    for key in ["objective_value", "total_distance", "project_duration", 
                "num_trains", "total_cost", "makespan", "total_delay"]:
        if key in reference_solution and isinstance(reference_solution[key], (int, float)):
            ref_cost = float(reference_solution[key])
            break
    
    if ref_cost is None or ref_cost == 0:
        return None
    
    if sol_cost == 0:
        if ref_cost > 0:
            return 2.0  # Perfect (lower cost is better)
        return 1.0
    
    # Score: higher is better, 1.0 = same as reference
    return ref_cost / sol_cost


# ============================================================
# スケジューリング系スコアラー
# ============================================================

def score_single_machine_scheduling(instance: dict, solution: Any) -> float:
    """単一機械のジョブ順序決定: 総遅延時間の最小化。自前計算。
    
    空sequenceや全ジョブ未割当ならNoneを返す（cost=None → 評価不可）。
    """
    jobs = instance.get("jobs", [])
    if not jobs:
        return None
    
    # Parse solution sequence
    if isinstance(solution, list):
        sequence = solution
    elif isinstance(solution, dict):
        sequence = solution.get("sequence") or solution.get("optimal_sequence") or solution.get("order") or []
    else:
        return None
    
    if not sequence:
        return None  # Empty sequence - invalid
    
    # Build job lookup by id
    job_map = {j.get("id", i): j for i, j in enumerate(jobs)}
    
    # Validation: all jobs must be assigned
    assigned_ids = [sid for sid in sequence if sid in job_map]
    if len(set(assigned_ids)) < len(jobs):
        return None  # Not all jobs assigned - invalid
    
    # Self-compute total tardiness
    current_time = 0
    total_tardiness = 0.0
    for job_id in sequence:
        if job_id not in job_map:
            continue
        job = job_map[job_id]
        pt = job.get("processing_time", 0)
        dd = job.get("due_date", float("inf"))
        current_time += pt
        tardiness = max(0, current_time - dd)
        total_tardiness += tardiness
    
    # Return negative tardiness (higher score = lower tardiness)
    return -total_tardiness


def score_job_shop_scheduling(instance: dict, solution: Any) -> float:
    """ジョブショップスケジューリング: makespanの最小化。自前計算。
    
    生成コード側のmakespan値は信じず、schedule構造から自前で最大end_timeを求める。
    schedule欠損や空ならNoneを返す。
    """
    jobs = instance.get("jobs", [])
    if not jobs:
        return None
    
    if isinstance(solution, (int, float)):
        # Bare number - not enough info to validate
        val = float(solution)
        if val <= 0:
            return None
        return -val
    
    if not isinstance(solution, dict):
        return None
    
    schedule = solution.get("schedule") or solution.get("assignments") or solution.get("assignment")
    if not schedule:
        return None  # Empty schedule
    
    # Self-compute makespan from schedule
    max_end = 0.0
    assignments_found = 0
    
    if isinstance(schedule, dict):
        # {job_id: [{machine, start, end, ...}, ...]}
        for job_id, ops in schedule.items():
            if isinstance(ops, list):
                for op in ops:
                    if isinstance(op, dict):
                        end = op.get("end") or op.get("end_time") or op.get("finish")
                        if isinstance(end, (int, float)):
                            max_end = max(max_end, float(end))
                            assignments_found += 1
    elif isinstance(schedule, list):
        for item in schedule:
            if isinstance(item, dict):
                end = item.get("end") or item.get("end_time") or item.get("finish")
                if isinstance(end, (int, float)):
                    max_end = max(max_end, float(end))
                    assignments_found += 1
    
    if assignments_found == 0:
        return None  # No valid assignments in schedule
    
    if max_end <= 0:
        return None  # Suspicious makespan
    
    return -max_end


def score_critical_path(instance: dict, solution: Any) -> float:
    """クリティカルパス: プロジェクト最短完了日の算出。自前計算優先。
    
    activityリストがあれば、生成コード側のdurationを検証。
    """
    # Extract claimed duration
    claimed = None
    if isinstance(solution, (int, float)):
        claimed = float(solution)
    elif isinstance(solution, dict):
        claimed = solution.get("project_duration") or solution.get("duration") or solution.get("makespan")
        if claimed is not None:
            claimed = float(claimed)
    
    if claimed is None or claimed <= 0:
        return None  # Invalid or suspicious
    
    # Try to self-verify from activities
    activities = instance.get("activities", [])
    if activities:
        # Forward pass: earliest_finish per activity
        act_map = {a.get("id", i): a for i, a in enumerate(activities)}
        earliest_finish = {}
        
        def compute_ef(aid, visited=None):
            if visited is None:
                visited = set()
            if aid in visited:
                return 0.0
            visited.add(aid)
            if aid in earliest_finish:
                return earliest_finish[aid]
            if aid not in act_map:
                return 0.0
            act = act_map[aid]
            dur = act.get("duration", 0)
            preds = act.get("predecessors") or act.get("dependencies") or []
            if not preds:
                ef = dur
            else:
                ef = dur + max((compute_ef(p, visited) for p in preds), default=0)
            earliest_finish[aid] = ef
            return ef
        
        for aid in list(act_map.keys()):
            compute_ef(aid)
        
        if earliest_finish:
            computed_duration = max(earliest_finish.values())
            # If computed is much larger than claimed, claimed is invalid
            if computed_duration > 0 and claimed < computed_duration * 0.5:
                return -computed_duration  # Force correct value
            return -claimed
    
    return -claimed


# ============================================================
# 配送・輸送系スコアラー
# ============================================================

def _euclid(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def score_vrp(instance: dict, solution: Any) -> float:
    """VRP系: 総移動距離の最小化。自前計算優先。
    
    routes構造から総距離を自前計算。生成コード側のtotal_distanceは検証用。
    空routes/未訪問顧客がある場合はNoneまたは減点。
    """
    customers = instance.get("customers", [])
    if not customers:
        return None
    n = len(customers)
    
    if isinstance(solution, (int, float)):
        val = float(solution)
        if val <= 0 or val == float("inf"):
            return None
        return -val
    
    if not isinstance(solution, dict):
        return None
    
    routes = solution.get("routes")
    if not routes:
        return None  # Empty routes
    
    # Convert routes to list of paths
    route_list = []
    if isinstance(routes, dict):
        for r in routes.values():
            if isinstance(r, dict):
                path = r.get("route") or r.get("path") or r.get("customers") or r.get("nodes") or []
            elif isinstance(r, list):
                path = r
            else:
                continue
            route_list.append(path)
    elif isinstance(routes, list):
        for r in routes:
            if isinstance(r, dict):
                path = r.get("route") or r.get("path") or r.get("customers") or r.get("nodes") or []
            elif isinstance(r, list):
                path = r
            else:
                continue
            route_list.append(path)
    
    if not route_list or all(not p for p in route_list):
        return None
    
    # Get depot coords
    depot = instance.get("depot", {})
    if isinstance(depot, dict):
        depot_xy = (depot.get("x", 0), depot.get("y", 0))
    elif isinstance(depot, list) and depot:
        d0 = depot[0]
        if isinstance(d0, dict):
            depot_xy = (d0.get("x", 0), d0.get("y", 0))
        else:
            depot_xy = (0, 0)
    else:
        depot_xy = (0, 0)
    
    # Customer coords
    customer_map = {}
    for i, c in enumerate(customers):
        cid = c.get("id", i)
        customer_map[cid] = (c.get("x", 0), c.get("y", 0))
    
    # Compute total distance
    total_dist = 0.0
    visited = set()
    for path in route_list:
        if not path:
            continue
        # Extract node IDs
        nodes = []
        for node in path:
            if isinstance(node, dict):
                nid = node.get("id")
            else:
                nid = node
            nodes.append(nid)
        
        # Compute route distance from depot -> ... -> depot
        prev_xy = depot_xy
        for nid in nodes:
            if nid == 0 or nid is None:
                cur_xy = depot_xy
            elif nid in customer_map:
                cur_xy = customer_map[nid]
                visited.add(nid)
            else:
                continue
            total_dist += _euclid(prev_xy, cur_xy)
            prev_xy = cur_xy
        # Return to depot if not already
        if nodes and (nodes[-1] != 0):
            total_dist += _euclid(prev_xy, depot_xy)
    
    if total_dist <= 0:
        return None
    
    # Coverage penalty
    coverage = len(visited) / n if n > 0 else 0
    if coverage < 0.5:
        return None  # Too incomplete
    if coverage < 1.0:
        # Add penalty for missing customers
        total_dist = total_dist / coverage  # inflate cost
    
    return -total_dist


def score_train_composition(instance: dict, solution: Any) -> float:
    """貨物列車編成計画: 列車数の最小化。自前計算。"""
    if isinstance(solution, (int, float)):
        val = float(solution)
        if val <= 0:
            return None
        return -val
    if isinstance(solution, dict):
        trains = solution.get("trains")
        if isinstance(trains, dict) and trains:
            return -len(trains)
        if isinstance(trains, list) and trains:
            return -len(trains)
        num = solution.get("num_trains")
        if num is not None and num > 0:
            return -float(num)
    return None


# ============================================================
# Register all scorers
# ============================================================

# スケジューリング系
register_scorer("スケジューリング_動的計画法", score_single_machine_scheduling)
register_scorer("スケジューリング_整数計画", score_job_shop_scheduling)
register_scorer("スケジューリング_混合整数計画", score_job_shop_scheduling)
register_scorer("スケジューリング_グラフ最適化", score_critical_path)
register_scorer("スケジューリング_確率最適化", score_job_shop_scheduling)

# 配送・輸送系
register_scorer("配送・輸送_混合整数計画", score_vrp)
register_scorer("配送・輸送_確率最適化", score_vrp)

"""V3 メトリック: reference_solutionベースの評価 + Shape Reward。

V2との違い:
- V2: best_knownとの比較 + scorerモジュール
- V3: reference_solutionとの直接比較 + best_knownの追跡

V3ではreference_solutionが各問題の基準値となる。
スコア計算:
  1. solve()を実行してsolutionを取得
  2. scorerでスコアを計算（またはreference_solutionと比較）
  3. best_knownとの比較でShape Rewardを付加

報酬構成:
  1. base_score: reference_cost/solution_cost の線形マッピング
  2. improvement_bonus: best_known更新時 +0.15
  3. exact_match: reference_solutionと完全一致 +0.2
  4. exploration_bonus: 新しいアプローチ試行 +0.03
  5. incremental_improvement: 微小改善(1%+) +0.05×改善率
"""

from __future__ import annotations

from . import best_known as _best_known_module
from .utils.feasibility import check_feasibility_detailed
from .utils.scorer import compute_score, has_scorer

# Objective field name → (is_minimization) mapping heuristic
_MIN_KEYWORDS = [
    "cost",
    "distance",
    "makespan",
    "duration",
    "delay",
    "time",
    "waste",
    "penalty",
    "gap",
    "deviation",
    "flow_time",
    "tardiness",
    "unassigned",
    "unmet",
    "shortage",
    "overflow",
    "num_",
    "count",
]
_MAX_KEYWORDS = [
    "rating",
    "profit",
    "revenue",
    "match",
    "assigned",
    "coverage",
    "satisfaction",
    "utility",
    "preference",
    "score",
    "quality",
    "throughput",
    "capacity_used",
    "priority",
]

# Bilingual (JP objective phrase → English field-name token) map.
# Used to pick the correct objective field when the reference solution
# reports several numeric metrics (e.g. peak_resource_usage vs makespan).
_OBJ_TOKEN_MAP = [
    (["ピーク", "peak"], ["peak"]),
    (
        ["完了", "工期", "完成", "納期", "日数", "makespan", "完工"],
        ["makespan", "duration", "completion", "days", "span"],
    ),
    (["移動距離", "総距離", "距離", "distance", "走行"], ["distance", "travel"]),
    (["費用", "コスト", "cost", "料金", "予算"], ["cost", "price", "budget"]),
    (["利益", "収益", "profit", "利潤"], ["profit", "revenue"]),
    (["遅延", "遅れ", "tardiness", "delay"], ["tardiness", "delay", "late"]),
    (["待機", "待ち", "wait"], ["wait", "delay", "idle"]),
    (["一致", "希望", "マッチ", "match", "preference"], ["match", "preferred", "preference"]),
    (["優先", "priority"], ["priority"]),
    (
        ["本数", "個数", "台数", "数の最小", "rolls", "count"],
        ["rolls", "count", "num", "pieces", "bins"],
    ),
    (["差", "偏差", "difference", "deviation"], ["difference", "deviation", "gap", "diff"]),
    (["流量", "フロー", "flow"], ["flow"]),
    (["カバー", "被覆", "coverage", "カバレッジ", "人口"], ["covered", "coverage", "population"]),
    (["価値", "value", "評価値"], ["value", "worth"]),
    (["リソース", "資源", "資源使用", "resource"], ["resource", "usage", "workers", "equipment"]),
]


def _objective_tokens_from_text(objective_text: str) -> list[str]:
    """Map a (Japanese/English) objective description to English field-name tokens."""
    if not objective_text:
        return []
    text = objective_text.lower()
    tokens: list[str] = []
    for jp_keys, en_tokens in _OBJ_TOKEN_MAP:
        for jk in jp_keys:
            if jk.lower() in text:
                tokens.extend(en_tokens)
                break
    return tokens


def _select_objective_field(
    reference_solution: dict, solution: dict, objective_text: str = ""
) -> tuple[str | None, bool, float | None]:
    """Choose the objective field among reference numeric scalars via scoring.

    Signals (higher = more likely the true objective):
      +5  name starts with 'min_' / 'max_' (explicit optimum marker)
      +4  name contains a token derived from the requirement objective text
      +2  name starts with an aggregate marker (total_/peak_/expected_/best_/optimal_)
      +1  same key present in the solution as a numeric value
      -4  value equals the element-count of the assignment container (structural count)
      -1  looks like a secondary count (*_count / num_* / *assigned / *scheduled / *rate)
      -idx*0.01  small tiebreak preferring earlier-listed fields

    Returns (field_name, is_min, ref_value).
    """
    cands = [
        (k, v)
        for k, v in reference_solution.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k.lower() != "note"
    ]
    if not cands:
        return None, True, None

    # Size of the assignment container (for structural-count demotion)
    container_size = 0
    for k, v in reference_solution.items():
        if isinstance(v, (list, dict)) and len(v) > container_size:
            container_size = len(v)

    obj_tokens = _objective_tokens_from_text(objective_text)
    agg_prefixes = ("total_", "peak_", "expected_", "best_", "optimal_")

    best_field = None
    best_score = float("-inf")
    best_val = None
    for idx, (k, v) in enumerate(cands):
        kl = k.lower()
        score = 0.0
        if kl.startswith("min_") or kl.startswith("max_"):
            score += 5.0
        if obj_tokens and any(tok in kl for tok in obj_tokens):
            score += 4.0
        if kl.startswith(agg_prefixes):
            score += 2.0
        if (
            k in solution
            and isinstance(solution.get(k), (int, float))
            and not isinstance(solution.get(k), bool)
        ):
            score += 1.0
        if container_size and float(v) == float(container_size):
            score -= 4.0
        if (
            kl.endswith("_count")
            or kl.startswith("num_")
            or "assigned" in kl
            or "scheduled" in kl
            or kl.endswith("_rate")
        ):
            score -= 1.0
        score -= idx * 0.01

        if score > best_score:
            best_score = score
            best_field = k
            best_val = float(v)

    # Determine direction for the chosen field
    kl = best_field.lower()
    if kl.startswith("min_"):
        is_min = True
    elif kl.startswith("max_"):
        is_min = False
    else:
        hint = _is_min_objective(best_field)
        # objective text can also flip direction
        if "最大" in (objective_text or "") or "maximize" in (objective_text or "").lower():
            is_min = False
        elif "最小" in (objective_text or "") or "minimize" in (objective_text or "").lower():
            is_min = True
        elif hint is not None:
            is_min = hint
        else:
            is_min = True
    return best_field, is_min, best_val


def _is_min_objective(field_name: str) -> bool | None:
    """Heuristically decide if a field name indicates a minimization objective."""
    fn = field_name.lower()
    for kw in _MIN_KEYWORDS:
        if kw in fn:
            return True
    for kw in _MAX_KEYWORDS:
        if kw in fn:
            return False
    return None


def _is_non_empty(value) -> bool:
    """Check if a solution assignment field has actual content."""
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


def generic_ref_guided_cost(solution, reference_solution, requirements=None, objective_text=""):
    """Dynamically score a solution using reference_solution structure.

    Steps:
    1. Find the "assignment" field in ref (largest non-scalar non-note field)
    2. Verify solution has same/similar field with non-empty content
    3. Select the "objective" field via _select_objective_field (robust to
       multi-metric references like peak_resource_usage vs makespan, using the
       requirement objective text when available)
    4. Return solution[objective_field] as cost (invert if maximization)

    Returns:
        cost (float) if valid, None if solution shape invalid
    """
    if not isinstance(reference_solution, dict) or not reference_solution:
        # Reference-free fallback: identify the objective field from the
        # solution's own numeric fields (works when no reference exists).
        if isinstance(solution, dict) and solution:
            reference_solution = solution
        elif isinstance(solution, (int, float)):
            return float(solution)
        else:
            return None
    if not isinstance(solution, dict):
        # If solution is a number and ref has one numeric field, use it
        if isinstance(solution, (int, float)):
            return float(solution)
        return None
    if objective_text == "" and isinstance(requirements, str):
        objective_text = requirements

    # Identify assignment field: prefer list/dict fields with substantive content
    assignment_field = None
    for k, v in reference_solution.items():
        if k.lower() in ("note", "feasible", "all_scheduled", "all_assigned"):
            continue
        if isinstance(v, (list, dict)) and _is_non_empty(v):
            assignment_field = k
            break

    # Check solution has some non-empty structural field
    # Look for exact match, then try common alternatives
    sol_has_content = False
    if assignment_field:
        # Try exact match first
        if assignment_field in solution and _is_non_empty(solution[assignment_field]):
            sol_has_content = True
        else:
            # Try any list/dict field in solution with content
            for k, v in solution.items():
                if isinstance(v, (list, dict)) and _is_non_empty(v):
                    sol_has_content = True
                    break
    else:
        # Ref has no assignment field (rare) — accept any dict solution
        sol_has_content = True

    if not sol_has_content:
        return None  # Solution has no substantive content

    # Identify objective field via robust scoring selector
    obj_field, is_min, ref_obj = _select_objective_field(
        reference_solution, solution, objective_text=objective_text
    )

    if obj_field is None:
        return None

    # Get value from solution: try exact field, then similar
    sol_val = None
    if (
        obj_field in solution
        and isinstance(solution[obj_field], (int, float))
        and not isinstance(solution[obj_field], bool)
    ):
        sol_val = float(solution[obj_field])
    else:
        # Try common alternatives
        for alt in [
            obj_field,
            obj_field.replace("total_", ""),
            obj_field.replace("min_", ""),
            obj_field.replace("max_", ""),
            "objective_value",
            "objective",
            "cost",
            "value",
        ]:
            if (
                alt in solution
                and isinstance(solution[alt], (int, float))
                and not isinstance(solution[alt], bool)
            ):
                sol_val = float(solution[alt])
                break

    if sol_val is None:
        # A fabricated cost can accidentally earn a reference bonus, especially
        # for maximization objectives. Missing objective fields are invalid.
        return None

    # For maximization: invert to minimization convention
    # (metrics_v3 treats lower cost as better)
    if not is_min and ref_obj:
        # Convert maximization: cost = ref_obj / sol_val * ref_obj (so equal → cost=ref_obj)
        # If sol_val >= ref_obj: cost <= ref_obj (better)
        # If sol_val < ref_obj: cost > ref_obj (worse)
        if sol_val > 0:
            return float(ref_obj) * float(ref_obj) / float(sol_val)
        else:
            return float(ref_obj) * 10.0  # very bad

    return sol_val


def generic_reference_free_cost(solution, objective_text=""):
    """Extract a monotonic cost from the candidate without reading reference values."""
    if not isinstance(solution, dict) or not solution:
        return None
    obj_field, is_min, value = _select_objective_field(
        solution,
        solution,
        objective_text=objective_text,
    )
    if obj_field is None or value is None:
        return None
    if is_min:
        return float(value)
    if value > 0:
        return 1.0 / float(value)
    return None


def classify_exec_error(error_msg: str) -> str:
    """
    exec_errorの詳細分類。エラーメッセージからエラー種別を判別。

    Returns:
        error_category: syntax / no_solve / keyerror / timeout / import / attribute / type / index / runtime
    """
    msg_lower = error_msg.lower()

    # Syntax errors
    if "syntaxerror" in msg_lower or "invalid syntax" in msg_lower or "unexpected EOF" in msg_lower:
        return "syntax"

    # No solve function
    if "no callable" in msg_lower or "'solve'" in msg_lower and "defined" in msg_lower:
        return "no_solve"
    if "no callable 'solve'" in msg_lower:
        return "no_solve"

    # KeyError
    if "keyerror" in msg_lower:
        return "keyerror"

    # Timeout
    if "timeout" in msg_lower:
        return "timeout"

    # Import errors
    if "importerror" in msg_lower or "import of" in msg_lower and "not allowed" in msg_lower:
        return "import"
    if "banned import" in msg_lower:
        return "import"

    # AttributeError
    if "attributeerror" in msg_lower:
        return "attribute"

    # TypeError
    if "typeerror" in msg_lower:
        return "type"

    # IndexError
    if "indexerror" in msg_lower:
        return "index"

    # Recursion errors
    if "recursion" in msg_lower or "maximum recursion" in msg_lower:
        return "runtime"

    return "runtime"


def compute_v3_score(
    cost: float,
    reference_cost: float | None,
    best_known: float | None,
    use_reference: bool = True,
    self_baseline: float | None = None,
) -> tuple[float, str, dict]:
    """
    V3スコア計算: reference_costとbest_knownの両方を考慮。

    Args:
        cost: 現在の解のコスト（小さいほど良い、負のスコア）
        reference_cost: 参考コスト（reference_solutionから）
        best_known: 過去の最良解
        use_reference: False のとき参考値を報酬に一切使わない
            (exact_match/beat_reference ボーナス・参照ベースの suspicious 検出を無効化)。
            RL 的に best_known とセルフベースラインのみで採点する。
        self_baseline: 参照フリー時の正規化アンカー (初回実行可能コスト)。

    Returns:
        (score, status_string, bonus_breakdown)
    """
    bonuses = {}

    # In reference-free mode the external reference value must not influence
    # the reward. Use the self-baseline as the anchor for degenerate-solution
    # detection instead.
    anchor = reference_cost if use_reference else self_baseline

    # Detect suspiciously-low cost (likely invalid/empty solution)
    # If cost is much smaller than the anchor (< 5%), the solution is probably empty/invalid
    suspicious = False
    if anchor is not None and anchor > 0:
        if cost < anchor * 0.05 and cost >= 0:
            suspicious = True
        elif cost < 0 and abs(cost) < anchor * 0.05:
            # Negative small values (e.g., -0.0 from scorer bug) are also suspicious
            suspicious = True

    # Special handling for suspicious cost=0 solutions
    # This typically happens when generated code returns empty/invalid solution
    if suspicious:
        return (
            0.3,  # Small positive score to discourage but not fully reject
            "suspicious_zero",
            {"suspicious_penalty": -1.0},
        )

    # Start with base score from best_known comparison
    if best_known is None:
        # First valid solution
        base = 1.0
        status = "first_valid"
    elif best_known <= 0:
        # best_known is 0 or negative (from previous suspicious solution) — use anchor
        if anchor is not None and anchor > 0 and cost > 0:
            ratio = anchor / cost
            base = min(2.0, max(0.0, 1.0 + (ratio - 1.0)))
            if ratio > 1.01:
                status = "improved"
            elif ratio < 0.99:
                status = "worse"
            else:
                status = "similar"
        else:
            base = 1.0
            status = "first_valid"
    else:
        ratio = best_known / cost if cost > 0 else float("inf")
        base = 2.0 * (ratio - 0.5)
        base = max(0.0, min(2.0, base))

        if ratio > 1.01:
            status = "improved"
        elif ratio < 0.99:
            status = "worse"
        else:
            status = "similar"

    # Bonus 1: Exact match / beating the reference (reference mode only)
    if use_reference and reference_cost is not None and reference_cost > 0:
        if abs(cost - reference_cost) < 1e-6:
            bonuses["exact_match"] = 0.2
            # Guarantee minimum score of 1.5 for exact match (fix for best_known=0 bug)
            base = max(base + 0.2, 1.5)
            status = "exact_match"
        elif 0 < cost < reference_cost:
            # Better than reference! (but not suspicious)
            improvement = (reference_cost - cost) / abs(reference_cost)
            bonuses["beat_reference"] = min(0.5, 0.2 + 0.3 * improvement)
            # Guarantee minimum score of 1.5 for beat_reference
            base = max(base + bonuses["beat_reference"], 1.5)
            status = "beat_reference"

    # Bonus 2: Improvement over best_known
    if best_known is not None and best_known > 0 and 0 < cost < best_known:
        bonuses["improvement"] = 0.15
        base += 0.15

    # Bonus 3: Incremental improvement (1-10%)
    if best_known is not None and best_known > 0 and cost > 0:
        ratio_check = best_known / cost
        if 1.0 < ratio_check <= 1.1:
            incremental = 0.05 * (ratio_check - 1.0) / 0.1
            bonuses["incremental"] = incremental
            base += incremental

    total = min(base, 2.5)
    return total, status, bonuses


def evaluate_algorithm_v3(
    code: str,
    instance: dict,
    core_type: str,
    instance_id: str | None = None,
    registry: _best_known_module.BestKnownRegistry | None = None,
    timeout: float = 60.0,
    reference_value: float | None = None,
    reference_solution: dict | None = None,
    objective_text: str = "",
    use_reference: bool = True,
    # Shape reward parameters
    new_approach: bool = False,
    violation_reduction: int = 0,
    strategy: str = "",
    strategy_history: list = None,
    same_error: bool = True,
) -> dict:
    """
    V3評価: reference_solutionベース + Shape Reward。

    Returns:
        dict with keys: score, status, cost, best_known, reference_value, gap_to_reference, detail, bonuses
    """
    reg = registry or _best_known_module.registry
    if reg is None:
        raise RuntimeError(
            "BestKnownRegistry is not initialized. "
            "Call init_registry() or pass registry= explicitly."
        )

    # Step 1: Safe execution with enhanced error feedback
    from .utils.safe_exec import safe_run

    ok, result = safe_run(code, instance, timeout=timeout)

    # Detect internal errors: code executed but returned an error dict
    if isinstance(result, dict) and "error" in result and ok:
        # Code caught an exception internally and returned error dict
        # Treat as exec_error so GEPA can learn to fix it
        ok = False
        detail_str = result.get("error", "Unknown internal error")
        error_category = "runtime"
    elif not ok:
        detail_str = str(result)
        error_category = classify_exec_error(detail_str)

    if not ok:
        # Enhanced error feedback with actionable suggestions
        error_feedback = {
            "syntax": {
                "score": -0.5,
                "fix": "Fix Python syntax: check colons, indentation, parentheses, f-strings. Try simplifying complex expressions.",
                "parse_error": True,
            },
            "no_solve": {
                "score": -0.5,
                "fix": "Define 'def solve(instance):' function. Example:\n```python\ndef solve(instance):\n    # extract variables\n    jobs = instance.get('jobs', [])\n    # implement algorithm\n    return solution\n```",
                "parse_error": True,
            },
            "keyerror": {
                "score": -0.5,
                "fix": f"Use instance.get('key', default) not instance['key']. Available keys: {', '.join(instance.keys())}",
                "parse_error": True,
            },
            "timeout": {
                "score": -0.3,
                "fix": f"Code exceeded {timeout}s. Use greedy heuristics or OR-Tools with time limit. Avoid O(N³) or exponential algorithms.",
                "parse_error": False,
            },
            "import": {
                "score": -0.5,
                "fix": "Allowed imports: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, ortools, scipy, pulp, networkx, numpy",
                "parse_error": True,
            },
            "attribute": {
                "score": -0.5,
                "fix": "Check object type before accessing attributes. Use isinstance() to verify.",
                "parse_error": False,
            },
            "type": {
                "score": -0.5,
                "fix": "Check types with isinstance() before operations. Convert types explicitly: int(x), float(x), str(x). READ THE PARSE CODE COMMENTS for correct data structure!",
                "parse_error": False,
            },
            "index": {
                "score": -0.5,
                "fix": "Check list length before indexing: if i < len(my_list): my_list[i]",
                "parse_error": False,
            },
            "runtime": {
                "score": -0.2,
                "fix": "Internal runtime error. The code caught an exception but returned an error dict. Fix the underlying issue - likely a type mismatch or incorrect data access. READ THE PARSE CODE COMMENTS for correct data structure!",
                "parse_error": False,
            },
        }

        feedback = error_feedback.get(error_category, error_feedback["runtime"])
        detail_str = f"{error_category.upper()}: {detail_str[:150]}. FIX: {feedback['fix']}"

        return {
            "score": feedback["score"],
            "status": "exec_error",
            "detail": detail_str,
            "cost": None,
            "best_known": None,
            "gap_to_reference": None,
            "bonuses": {},
            "parse_error": feedback["parse_error"],
            "error_category": error_category,
            "error_fix": feedback["fix"],  # Actionable fix for GEPA feedback
        }

    # Step 2: Feasibility check with partial credit
    feasibility_result = check_feasibility_detailed(core_type, instance, result)
    feasibility_verified = feasibility_result.get("verified", True)
    if not feasibility_result.get("feasible", True):
        # Partial credit for partial constraint satisfaction
        partial_score = feasibility_result.get("partial_score", 0.0)
        violation_count = feasibility_result.get("violation_count", 0)
        total_constraints = feasibility_result.get("total_constraints", 0)

        # Give small credit for satisfying some constraints
        if partial_score > 0 and total_constraints > 0:
            return {
                "score": min(partial_score * 0.3, 0.5),  # Cap at 0.5 for partial feasibility
                "status": "partial_feasible",
                "detail": f"constraint violation ({violation_count}/{total_constraints} violated, partial_score={partial_score:.2f})",
                "cost": feasibility_result.get("cost"),
                "best_known": None,
                "gap_to_reference": None,
                "bonuses": {"partial_feasibility": partial_score * 0.3},
                "parse_error": False,
                "violation_count": violation_count,
                "total_constraints": total_constraints,
                "feasibility_verified": feasibility_verified,
            }
        return {
            "score": 0.0,
            "status": "infeasible",
            "detail": f"constraint violation ({violation_count}/{total_constraints})",
            "cost": feasibility_result.get("cost"),
            "best_known": None,
            "gap_to_reference": None,
            "bonuses": {},
            "parse_error": False,
            "violation_count": violation_count,
            "total_constraints": total_constraints,
        }

    # Step 3: Compute cost from solution
    # PRIORITY: registered scorer with self-computation (catches fake cost=0)
    # If scorer returns None, try generic reference-guided scorer
    cost = None
    if has_scorer(core_type):
        score_val = compute_score(core_type, instance, result)
        if score_val is not None:
            cost = -score_val  # Higher score = lower cost (minimization convention)

    # Use a generic scorer only if a registered scorer rejected or is unavailable.
    if cost is None:
        if use_reference:
            generic_cost = generic_ref_guided_cost(
                result,
                reference_solution,
                objective_text=objective_text,
            )
        else:
            generic_cost = generic_reference_free_cost(result, objective_text=objective_text)
        if generic_cost is not None:
            cost = generic_cost

    # No trustworthy objective means the result cannot participate in ranking.
    if cost is None:
        return {
            "score": 0.1,
            "status": "invalid_solution",
            "detail": f"Solution rejected by scorer: solution is empty/incomplete/wrong-shape. Available instance keys: {', '.join(instance.keys())}. Return format check: does your solution include the required fields with non-zero values?",
            "cost": None,
            "best_known": None,
            "gap_to_reference": None,
            "bonuses": {},
            "parse_error": False,
            "error_category": "invalid_solution",
            "error_fix": "Your solution was rejected because it appears empty or invalid. Make sure your solve() function: 1) Returns a dict with the correct top-level fields matching the reference solution structure, 2) All jobs/customers/activities are actually assigned/scheduled (not empty!), 3) Cost fields reflect a real computation, not just 0.",
        }

    # Detect solver failure: cost=inf means the solver couldn't find a solution
    # Treat as exec_error so GEPA can learn to use a different approach
    if cost == float("inf") or cost == float("-inf"):
        return {
            "score": -0.2,
            "status": "solver_failure",
            "detail": f"Solver returned cost=inf (no feasible solution found). Try a simpler heuristic or different solver approach. Available keys: {', '.join(instance.keys())}",
            "cost": cost,
            "best_known": None,
            "gap_to_reference": None,
            "bonuses": {},
            "parse_error": False,
            "error_category": "solver_failure",
            "error_fix": "The solver (OR-Tools/CP-SAT) could not find a feasible solution. Try: 1) Use a simpler greedy heuristic instead of OR-Tools, 2) Relax constraints, 3) Add time limit to solver, 4) Check if your model formulation is correct",
        }

    # Step 4: Get best_known
    best = reg.get(instance_id) if instance_id else None

    # Reference-free mode: record/lookup the self-baseline (first feasible
    # cost) so improvement can be normalized without the external reference.
    self_baseline = None
    if not use_reference and feasibility_verified and instance_id:
        reg.set_baseline_if_absent(instance_id, cost)
        self_baseline = reg.get_baseline(instance_id)

    # Step 5: Compute V3 score
    score, status, bonuses = compute_v3_score(
        cost=cost,
        reference_cost=reference_value,
        best_known=best,
        use_reference=use_reference,
        self_baseline=self_baseline,
    )
    if not feasibility_verified:
        score = min(score, 1.0)
        status = "unverified"
        bonuses["feasibility_unverified"] = -0.5

    # Step 6: Exploration bonus
    if new_approach:
        bonuses["exploration"] = 0.03
        score = min(score + 0.03, 2.5)

    # Step 7: Update best_known if better
    best_before = best
    if feasibility_verified and instance_id and cost < (best or float("inf")):
        reg.update_if_better(instance_id, cost)
        best = cost
        if status not in ("exact_match", "beat_reference"):
            status = "new_best"

    # Step 8: Compute gap relative to reference
    gap_to_reference = None
    if reference_value and reference_value != 0:
        gap_to_reference = (cost - reference_value) / abs(reference_value)

    # Step 9: Compute gap relative to best_known
    gap_to_best = None
    if best_before and best_before != 0:
        gap_to_best = (cost - best_before) / abs(best_before)

    detail_parts = []
    if reference_value is not None:
        detail_parts.append(f"ref={reference_value:.2f}")
    if best_before is not None:
        detail_parts.append(f"best={best_before:.4f}")
    detail_parts.append(f"cost={cost:.4f}")
    if gap_to_reference is not None:
        detail_parts.append(f"gap_to_ref={gap_to_reference * 100:+.1f}%")
    if not feasibility_verified:
        detail_parts.append("feasibility=unverified")

    return {
        "score": min(score, 2.5),
        "status": status,
        "cost": cost,
        "best_known": best,
        "reference_value": reference_value,
        "gap_to_reference": gap_to_reference,
        "gap_to_best": gap_to_best,
        "detail": ", ".join(detail_parts),
        "bonuses": bonuses,
        "parse_error": False,
        "feasibility_verified": feasibility_verified,
    }


def dspy_metric_v3(example, pred, trace=None) -> float:
    """DSPy 互換メトリック v3。"""
    inst = getattr(example, "instance", None) or example["instance"]
    core_type = getattr(example, "core_type", None) or example.get("core_type")
    iid = getattr(example, "instance_id", None) or example.get("instance_id")
    ref = getattr(example, "reference_value", None) or example.get("reference_value")
    ref_sol = getattr(example, "reference_solution", None) or example.get("reference_solution", {})
    code = pred.algorithm_code
    r = evaluate_algorithm_v3(
        code,
        inst,
        core_type,
        instance_id=iid,
        reference_value=ref,
        reference_solution=ref_sol,
    )
    return r["score"]

"""V3 フィードバック関数: reference_solutionベースの鋭いテキストフィードバック。

GEPA に渡すフィードバック。score だけでなく feedback テキストを返し、
reference_solutionとのギャップを明確に伝えることで、LLM が改善方向を理解できるようにする。

改善: core_typeに応じた具体的なアルゴリズム提案を追加。
"""

from __future__ import annotations

import dspy

from . import best_known as _best_known_module
from .metrics_v3 import evaluate_algorithm_v3

# Global toggle: when False, the GEPA reward and feedback text must not use
# the external reference value at all (reference-free / RL-style training).
# The reference is still available to the caller for post-hoc analysis only.
USE_REFERENCE = True


def set_use_reference(flag: bool) -> None:
    """参照フリーモードの切替 (train スクリプトから設定)。"""
    global USE_REFERENCE
    USE_REFERENCE = bool(flag)


# core_typeに応じた改善提案
_ALGORITHM_HINTS = {
    "スケジューリング_動的計画法": (
        "For single-machine scheduling: try Earliest Due Date (EDD) rule, "
        "Moore's algorithm for minimizing late jobs, or dynamic programming for optimal sequencing. "
        "Consider branch-and-bound for small instances."
    ),
    "スケジューリング_整数計画": (
        "For integer programming scheduling: use OR-Tools CP-SAT solver with time limit. "
        "Define binary assignment variables x[j][m] for job j on machine m. "
        "Add constraints for makespan minimization. Try greedy initialization as warm start."
    ),
    "スケジューリング_混合整数計画": (
        "For mixed-integer scheduling: use OR-Tools CP-SAT with time limit (10-30s). "
        "Define decision variables for start times and assignments. "
        "Add no-overlap constraints. Try first-fit-decreasing heuristic as fallback. "
        "For resource leveling: use minimum-time heuristic then smooth peaks."
    ),
    "スケジューリング_グラフ最適化": (
        "For critical path: use topological sort on activity graph. "
        "Calculate earliest/latest start times with forward/backward pass. "
        "Critical path = activities with zero slack (earliest_start == latest_start)."
    ),
    "スケジューリング_確率最適化": (
        "For stochastic scheduling: use scenario-based approach or robust optimization. "
        "Sample scenarios, solve deterministic version for each, take expected value. "
        "Or use OR-Tools with chance constraints."
    ),
    "配送・輸送_混合整数計画": (
        "For VRP: use OR-Tools RoutingSolver with time limit. "
        "Add dimension for capacity constraints. Use routing.MakeVar or routing.AddDimension. "
        "Set first solution heuristic to PATH_CHEAPEST_ARC or AUTOFIRST. "
        "Enable local search operators: LOCAL_SEARCH_CHAIN, CROSS_TRAIL. "
        "For time windows: add time dimension with slack_max=0. "
        "For backhaul: split into linehaul/backhaul routes with precedence constraints."
    ),
    "配送・輸送_確率最適化": (
        "For stochastic VRP: use scenario-based routing. "
        "Solve deterministic VRP first, then add safety margins. "
        "Or use OR-Tools RoutingSolver with probabilistic demand constraints."
    ),
}


def get_algorithm_hint(core_type: str) -> str:
    """core_typeに応じたアルゴリズム提案を返す。"""
    return _ALGORITHM_HINTS.get(
        core_type, "Try: OR-Tools CP-SAT solver, greedy heuristics, or local search improvements."
    )


def summarize_ref_solution(ref_sol, hide_values: bool = False) -> str:
    """Extract a compact hint from reference_solution to guide the LLM.

    When hide_values=True (reference-free mode), the numeric objective TARGET
    is omitted so the reference value never leaks into the prompt; only the
    output SCHEMA (field names / shapes) is exposed, which is part of the
    problem specification rather than the reference answer.
    """
    if not isinstance(ref_sol, dict) or not ref_sol:
        return ""
    fields = list(ref_sol.keys())
    lines = [f"Required output top-level fields: {fields}."]
    if not hide_values:
        for k, v in ref_sol.items():
            if isinstance(v, (int, float)) and k.lower() != "note":
                lines.append(f"  '{k}' = {v} (numeric objective — you should hit or beat this).")
                break
    for k, v in ref_sol.items():
        if isinstance(v, list) and v:
            lines.append(
                f"  '{k}' is a list[{len(v)}] — your solution must have a non-empty list at this key."
            )
            break
        if isinstance(v, dict) and v:
            lines.append(f"  '{k}' is a dict with {len(v)} entries — non-empty required.")
            break
    return " ".join(lines)


def gepa_feedback_v3(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    V3フィードバック: reference_solutionとの比較 + Shape Rewardボーナスを明示。

    Returns:
        dspy.Prediction(score=float, feedback=str)
    """
    inst = getattr(example, "instance", None) or example["instance"]
    core_type = getattr(example, "core_type", None) or example.get("core_type")
    iid = getattr(example, "instance_id", None) or example.get("instance_id")
    ref = getattr(example, "reference_value", None) or example.get("reference_value")
    ref_sol = getattr(example, "reference_solution", None) or example.get("reference_solution", {})
    objective_text = (
        getattr(example, "objective", None)
        or (example.get("objective", "") if hasattr(example, "get") else "")
        or ""
    )

    code = pred.algorithm_code
    reg = _best_known_module.registry

    result = evaluate_algorithm_v3(
        code,
        inst,
        core_type,
        instance_id=iid,
        registry=reg,
        reference_value=ref,
        reference_solution=ref_sol,
        objective_text=objective_text,
        use_reference=USE_REFERENCE,
    )
    score = result["score"]
    cost = result.get("cost")
    best = result.get("best_known")
    status = result["status"]
    gap_ref = result.get("gap_to_reference")
    gap_best = result.get("gap_to_best")
    bonuses = result.get("bonuses", {})
    parse_error = result.get("parse_error", False)

    # In reference-free mode, the reference value must not appear in the
    # feedback text (it would leak into the LLM prompt via GEPA reflection).
    if not USE_REFERENCE:
        ref = None

    # Build bonus summary
    bonus_parts = []
    for bonus_type in [
        "improvement",
        "incremental",
        "exact_match",
        "beat_reference",
        "exploration",
    ]:
        if bonus_type in bonuses and bonuses[bonus_type] > 0:
            bonus_parts.append(f"{bonus_type}=+{bonuses[bonus_type]:.2f}")
    bonus_str = " ".join(bonus_parts) if bonus_parts else ""

    hint = get_algorithm_hint(core_type)
    ref_hint = summarize_ref_solution(ref_sol, hide_values=not USE_REFERENCE)
    error_category = result.get("error_category", "runtime")

    # Build feedback
    if parse_error:
        detail = result.get("detail", "unknown error")
        if error_category == "syntax":
            feedback = (
                f"SYNTAX ERROR (strong penalty!): {detail[:300]}. "
                f"Fix the Python syntax: check for missing colons, indentation, unmatched parentheses, f-string issues. "
                f"Ensure the code is valid Python before returning. "
                f"Score: {score:.2f}"
            )
        elif error_category == "no_solve":
            feedback = (
                f"NO SOLVE FUNCTION (strong penalty!): {detail[:300]}. "
                f"Your code MUST define a callable 'solve(instance)' function at module level. "
                f"Example structure: 'def solve(instance):\n    # parse instance keys\n    # compute solution\n    return solution' "
                f"Score: {score:.2f}"
            )
        elif error_category == "keyerror":
            feedback = (
                f"KEY ERROR (strong penalty!): {detail[:300]}. "
                f"The instance dict does not have the key you referenced. "
                f"CRITICAL: Use instance.keys() to see available keys, then use those EXACT keys. "
                f"Do NOT assume variable names from the problem description match the dict keys. "
                f"Score: {score:.2f}"
            )
        elif error_category == "import":
            feedback = (
                f"IMPORT ERROR (strong penalty!): {detail[:300]}. "
                f"The imported module is not available. Allowed imports: math, random, heapq, itertools, collections, functools, typing, bisect, operator, ortools, scipy, pulp, networkx, numpy. "
                f"Use only these modules in your code. "
                f"Score: {score:.2f}"
            )
        else:
            feedback = (
                f"PARSE ERROR (strong penalty!): {detail[:200]}. "
                f"The generated code cannot parse the instance dict. "
                f"CRITICAL: Inspect instance.keys() first, then use those EXACT keys. "
                f"Do NOT assume variable names from the problem description match the dict keys. "
                f"HINT: {hint} "
                f"Score: {score:.2f}"
            )
    elif status == "exec_error":
        detail = result.get("detail", "unknown error")
        if error_category == "timeout":
            feedback = (
                f"TIMEOUT: {detail[:300]}. "
                f"Your code exceeded the time limit. Use simpler algorithms: greedy heuristics, OR-Tools with time limit, or early termination. "
                f"Avoid deeply nested loops or exponential search. "
                f"Score: {score:.2f}"
            )
        elif error_category == "attribute":
            # Extract the object and attribute from the error
            attr_detail = detail[:300]
            keys_str = ", ".join(str(k) for k in inst.keys())
            feedback = (
                f"ATTRIBUTE ERROR: {attr_detail}. "
                f"You tried to access an attribute/method that doesn't exist. "
                f"FIX: Check the type of the object first with isinstance(obj, dict/list/str). "
                f"For dicts: use .get(key, default) or check key in obj first. "
                f"For lists: check len(my_list) > index before my_list[index]. "
                f"For nested access: chain .get() calls: obj.get(a, {{}}).get(b, default). "
                f"Available instance keys: {keys_str} "
                f"Score: {score:.2f}"
            )
        elif error_category == "type":
            type_detail = detail[:300]
            feedback = (
                f"TYPE ERROR: {type_detail}. "
                f"You performed an operation on incompatible types (e.g., comparing str to int, adding list to int). "
                f"FIX: Use isinstance() to check types before operations. "
                f"Example: if isinstance(x, (int, float)): result = x + y "
                f"When extracting from dicts: val = instance.get(key, 0) ensures numeric default. "
                f"When iterating: for item in my_list: if isinstance(item, dict): ... "
                f"Score: {score:.2f}"
            )
        elif error_category == "index":
            feedback = (
                f"INDEX ERROR: {detail[:300]}. "
                f"List index out of range. Check list length before indexing. Use 'len(my_list)' to verify bounds. "
                f"FIX: Add 'if idx < len(my_list):' check before accessing my_list[idx]. "
                f"Score: {score:.2f}"
            )
        else:
            feedback = (
                f"RUNTIME ERROR: {detail[:300]}. "
                f"Fix the code. Define callable 'solve(instance)' that returns a valid solution. "
                f"HINT: {hint} "
                f"Score: {score:.2f}"
            )
    elif status == "infeasible":
        detail = result.get("detail", "")
        feedback = (
            f"INFEASIBLE ({detail[:150]}). Constraint violation for {core_type}. "
            f"{ref_hint} "
            f"ACTION: Check your constraint handling — capacity, time windows, precedence, coverage. "
            f"Ensure every required item is assigned exactly once. "
            f"HINT: {hint} "
            f"Score: {score:.2f}"
        )
    elif status == "exact_match":
        bonus_part = f" {bonus_str}." if bonus_str else ""
        feedback = (
            f"PERFECT! Cost={cost:.4f} matches reference exactly! "
            f"This is an optimal solution."
            f"{bonus_part} "
            f"Score: {score:.2f}"
        )
    elif status == "beat_reference":
        improvement = (gap_ref or 0) * -100 if gap_ref and gap_ref < 0 else 0
        best_str = f"{best:.4f}" if best else "N/A"
        bonus_part = f" {bonus_str}." if bonus_str else ""
        feedback = (
            f"Excellent! Cost={cost:.4f} beats reference! Improved by {improvement:.1f}% over reference. "
            f"Try to beat best_known={best_str}."
            f"{bonus_part} "
            f"Score: {score:.2f}"
        )
    elif status == "new_best":
        improvement = (gap_best or 0) * -100 if gap_best and gap_best < 0 else 0
        ref_str = f"{ref:.2f}" if ref else "N/A"
        bonus_part = f" {bonus_str}." if bonus_str else ""
        feedback = (
            f"New best! Cost={cost:.4f}. Improved by {improvement:.1f}% over previous best. "
            f"Target: reference={ref_str}. "
            f"Keep pushing with deeper search."
            f"{bonus_part} "
            f"Score: {score:.2f}"
        )
    elif status == "improved":
        improvement = (gap_best or 0) * -100 if gap_best and gap_best < 0 else 0
        ref_str = f"{ref:.2f}" if ref else "N/A"
        best_str = f"{best:.4f}" if best else "N/A"
        bonus_part = f" {bonus_str}." if bonus_str else ""
        feedback = (
            f"Good! Cost={cost:.4f}, Best={best_str}. "
            f"Within {improvement:.1f}% of best. "
            f"Target: reference={ref_str}. "
            f"Try: more iterations, alternative heuristics, solver tuning."
            f"{bonus_part} "
            f"Score: {score:.2f}"
        )
    elif status == "similar":
        gap_pct = (gap_best or 0) * 100 if gap_best else 0
        best_str = f"{best:.4f}" if best else "N/A"
        ref_str = f"{ref:.2f}" if ref else "N/A"
        bonus_part = f" {bonus_str}." if bonus_str else ""
        feedback = (
            f"Cost={cost:.4f}, Best={best_str}. Gap: {gap_pct:+.1f}%. "
            f"Similar to best. Target: reference={ref_str}. "
            f"HINT: {hint}"
            f"{bonus_part} "
            f"Score: {score:.2f}"
        )
    elif status == "worse":
        gap_pct = (gap_best or 0) * 100 if gap_best else 0
        best_str = f"{best:.4f}" if best else "N/A"
        ref_str = f"{ref:.2f}" if ref else "N/A"
        bonus_part = f" {bonus_str}." if bonus_str else ""
        feedback = (
            f"Cost={cost:.4f}, Best={best_str}. Gap: {gap_pct:+.1f}%. "
            f"Worse than best known. Target: reference={ref_str}. "
            f"{ref_hint} "
            f"ACTION: Try a different strategy — if you used greedy, add local search (2-opt/swap). "
            f"If you used a solver with default settings, tune it (increase time, add warm start). "
            f"HINT: {hint}"
            f"{bonus_part} "
            f"Score: {score:.2f}"
        )
    elif status == "infeasible":
        detail = result.get("detail", "")
        feedback = (
            f"INFEASIBLE ({detail[:150]}). Constraint violation for {core_type}. "
            f"{ref_hint} "
            f"ACTION: Check your constraint handling — capacity limits, time windows, precedence, coverage. "
            f"Ensure every required item is assigned exactly once. "
            f"HINT: {hint} "
            f"Score: {score:.2f}"
        )
    elif status == "partial_feasible":
        detail = result.get("detail", "")
        feedback = (
            f"PARTIALLY FEASIBLE (score={score:.2f}, {detail[:150]}). "
            f"Your solution satisfies most but not all constraints. "
            f"{ref_hint} "
            f"ACTION: Identify which constraint failed and add explicit handling — "
            f"e.g., capacity check before adding to route, deadline check before scheduling. "
            f"HINT: {hint} "
            f"Score: {score:.2f}"
        )
    elif status == "suspicious_zero":
        ref_str = f"{ref:.2f}" if ref else "N/A"
        keys_str = ", ".join(str(k) for k in inst.keys())
        feedback = (
            f"SUSPICIOUS SOLUTION (cost={cost:.4f} is way below reference={ref_str}). "
            f"Your code likely returned an EMPTY or INVALID solution (empty list, empty schedule, etc). "
            f"CHECK: Are you actually computing a solution, or just returning empty containers? "
            f"CHECK: Does your code assign ALL jobs/customers? Do you return the correct field name? "
            f"Available instance keys: {keys_str}. "
            f"HINT: {hint} "
            f"Score: {score:.2f}"
        )
    elif status == "invalid_solution":
        ref_str = f"{ref:.2f}" if ref else "N/A"
        keys_str = ", ".join(str(k) for k in inst.keys())
        detail = result.get("detail", "")
        feedback = (
            f"INVALID SOLUTION (score={score:.2f}). "
            f"Your returned dict is empty or lacks the required assignment field. Reference target: {ref_str}. "
            f"{ref_hint} "
            f"ACTION: Populate the assignment field with real content — for each job/customer/activity, "
            f"produce an entry that assigns it to a resource/route/time. "
            f"CHECK: 1) All jobs/customers/activities are assigned. 2) Numeric fields are non-zero. 3) Field names match reference. "
            f"{detail[:200]}. "
            f"Available instance keys: {keys_str}. "
            f"HINT: {hint} "
            f"Score: {score:.2f}"
        )
    elif status == "solver_failure":
        feedback = (
            f"SOLVER FAILURE (cost=inf). The OR-Tools/CP-SAT solver could not find a feasible solution. "
            f"SWITCH TO A GREEDY HEURISTIC! Do NOT try to fix the OR-Tools model - it's too complex. "
            f"For VRP: use nearest-neighbor insertion with capacity check. "
            f"For scheduling: use earliest-due-date or shortest-processing-time rule. "
            f"For assignment: use greedy matching. "
            f"HINT: {hint} "
            f"Score: {score:.2f}"
        )
    elif status == "first_valid":
        ref_str = f"{ref:.2f}" if ref else "N/A"
        feedback = (
            f"First valid solution! Cost={cost:.4f}. "
            f"Target: reference={ref_str}. "
            f"Use this as baseline and keep improving. "
            f"Score: {score:.2f}"
        )
    else:
        feedback = f"Status={status}, score={score:.2f}"

    return dspy.Prediction(score=score, feedback=feedback)

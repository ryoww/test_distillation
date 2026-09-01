"""V3 Requirement builder: V3データ → LLM用の要件テキスト。

V3データフォーマット:
  {id, name, domain, math_type, difficulty, split, description, requirements, instance, reference_solution}

V2との違い:
- problem_statement → description
- input_format/output_format → requirements.objective + requirements.constraints
- core_type → domain + math_type
"""

from __future__ import annotations

import json


def summarize_reference_solution(
    ref_sol: dict,
    instance: dict,
    *,
    include_values: bool = True,
) -> str:
    """reference_solution の構造から特徴を抽出して LLM 向けサマリーを作る。

    LLMが「参考解の構造」を理解できると、同じ形の解を出しやすくなる。
    """
    if not isinstance(ref_sol, dict) or not ref_sol:
        return ""

    lines = []
    if include_values:
        lines.append("The reference solution has the following structure (use this as a template):")
    else:
        lines.append("Required return schema:")

    # Top-level keys hint at the return format
    top_keys = list(ref_sol.keys())
    lines.append(f"- Top-level fields: {top_keys}")
    if not include_values:
        for key, value in ref_sol.items():
            if isinstance(value, bool):
                descriptor = "bool"
            elif isinstance(value, (int, float)):
                descriptor = "numeric"
            elif isinstance(value, list):
                element = type(value[0]).__name__ if value else "unknown"
                descriptor = f"list of {element}"
                if value and isinstance(value[0], dict):
                    descriptor += f" with keys {list(value[0].keys())}"
            elif isinstance(value, dict):
                descriptor = "dict"
            else:
                descriptor = type(value).__name__
            lines.append(f"- '{key}': {descriptor}")

    # Reference-free mode exposes the objective field, never the target value.
    for obj_key in [
        "objective_value",
        "makespan",
        "total_distance",
        "total_cost",
        "project_duration",
        "num_trains",
        "total_delay",
    ]:
        if (
            obj_key in ref_sol
            and isinstance(ref_sol[obj_key], (int, float))
            and not isinstance(ref_sol[obj_key], bool)
        ):
            if include_values:
                lines.append(
                    f"- Objective ({obj_key}) = {ref_sol[obj_key]} (target to match or beat)"
                )
            else:
                lines.append(f"- Objective field: '{obj_key}' (numeric)")
            break

    # Schedule structure
    if "schedule" in ref_sol:
        sched = ref_sol["schedule"]
        if isinstance(sched, dict):
            if include_values:
                lines.append(
                    f"- 'schedule' is a dict with {len(sched)} entries "
                    "(likely job_id → operations list)"
                )
            else:
                lines.append("- 'schedule' is a dict mapping job IDs to operation lists")
            sample_key = next(iter(sched), None)
            if sample_key is not None:
                sample = sched[sample_key]
                if isinstance(sample, list) and sample:
                    if isinstance(sample[0], dict):
                        lines.append(
                            "  Each schedule entry is a list of operations. "
                            f"Operation keys: {list(sample[0].keys())}"
                        )
        elif isinstance(sched, list):
            lines.append(
                f"- 'schedule' is a list of {len(sched)} entries"
                if include_values
                else "- 'schedule' is a list of operation entries"
            )
            if sched and isinstance(sched[0], dict):
                lines.append(f"  Each entry keys: {list(sched[0].keys())}")

    # Routes structure (VRP)
    if "routes" in ref_sol:
        routes = ref_sol["routes"]
        if isinstance(routes, list):
            num_routes = len(routes)
            lines.append(
                f"- 'routes' is a list of {num_routes} routes"
                if include_values
                else "- 'routes' is a list of routes"
            )
            if routes and isinstance(routes[0], dict):
                lines.append(f"  Each route keys: {list(routes[0].keys())}")
            elif routes and isinstance(routes[0], list):
                lines.append("  Each route is a list of node IDs")
                if include_values:
                    lines.append(f"  Sample route: {routes[0][:8]}")
            # Reference vehicle usage
            customers = instance.get("customers", [])
            if customers:
                total_nodes = sum(
                    len(r)
                    if isinstance(r, list)
                    else len(r.get("route", r.get("path", [])))
                    if isinstance(r, dict)
                    else 0
                    for r in routes
                )
                if include_values:
                    lines.append(
                        f"  Reference uses {num_routes} routes to visit {len(customers)} "
                        f"customers (avg {total_nodes / max(num_routes, 1):.1f} stops per route)"
                    )
        elif isinstance(routes, dict):
            lines.append(
                f"- 'routes' is a dict with {len(routes)} entries"
                if include_values
                else "- 'routes' is a dict mapping route IDs to route data"
            )

    # Assignments (integer programming)
    if "assignments" in ref_sol or "assignment" in ref_sol:
        assigns = ref_sol.get("assignments", ref_sol.get("assignment"))
        if isinstance(assigns, list):
            lines.append(
                f"- 'assignments' is a list of {len(assigns)} entries"
                if include_values
                else "- 'assignments' is a list of assignment entries"
            )
        elif isinstance(assigns, dict):
            lines.append(
                f"- 'assignments' is a dict with {len(assigns)} entries"
                if include_values
                else "- 'assignments' is a dict mapping item IDs to assignments"
            )

    # Sequence (single-machine)
    if "sequence" in ref_sol or "optimal_sequence" in ref_sol:
        seq = ref_sol.get("sequence", ref_sol.get("optimal_sequence"))
        if isinstance(seq, list):
            lines.append(
                f"- 'sequence' is a list of {len(seq)} elements (job order)"
                if include_values
                else "- 'sequence' is a list of job IDs in execution order"
            )
            if include_values:
                lines.append(f"  Sample: {seq[:5]}")

    if len(lines) == 1:
        return ""  # No useful info extracted

    return "\n".join(lines)


def build_requirement(record: dict, *, include_reference_values: bool = True) -> str:
    """
    V3レコードからLLM用の要件テキストを構築。

    Args:
        record: V3のJSONレコード

    Returns:
        要件テキスト（description + requirements + instance概要）
    """
    parts = []

    # Problem header
    name = record.get("name", "Unknown")
    domain = record.get("domain", "unknown")
    math_type = record.get("math_type", "unknown")
    difficulty = record.get("difficulty", "N/A")
    parts.append(f"## Problem: {name}")
    parts.append(f"Domain: {domain}")
    parts.append(f"Math Type: {math_type}")
    parts.append(f"Difficulty: {difficulty}")

    # Description (equivalent to problem_statement)
    description = record.get("description", "")
    if description:
        parts.append("## Description")
        parts.append(description)

    # Requirements (equivalent to input/output format)
    requirements = record.get("requirements", {})
    if requirements:
        parts.append("## Requirements")
        objective = requirements.get("objective", "")
        if objective:
            parts.append(f"Objective: {objective}")
        constraints = requirements.get("constraints", [])
        if constraints:
            parts.append("Constraints:")
            for c in constraints:
                parts.append(f"  - {c}")

    # Instance summary
    instance = record.get("instance", {})
    if instance:
        parts.append("## Instance Data (STRUCTURE - MUST READ)")
        # Full structure analysis with sample values and access patterns
        from .modules import analyze_instance_structure

        parts.append(analyze_instance_structure(instance))
        parts.append("")
        parts.append("## Instance Data (RAW JSON - use these EXACT values)")
        # Include raw JSON so LLM sees actual numbers to solve
        # Truncate very large instances to avoid token blow-up
        raw_json = json.dumps(instance, ensure_ascii=False, indent=2)
        if len(raw_json) > 6000:
            raw_json = (
                raw_json[:6000] + "\n... (truncated - use instance dict at runtime for full data)"
            )
        parts.append("```json")
        parts.append(raw_json)
        parts.append("```")

    # Reference value
    ref = record.get("reference_value")
    if include_reference_values and ref is not None:
        parts.append(f"## Reference Value: {ref:.2f}")
        parts.append(
            "Your algorithm should try to achieve a value close to or better than this reference (lower is better)."
        )

    # Reference solution structure hint (NEW)
    ref_sol = record.get("reference_solution", {})
    if ref_sol:
        ref_summary = summarize_reference_solution(
            ref_sol,
            instance,
            include_values=include_reference_values,
        )
        if ref_summary:
            parts.append(
                "## Reference Solution Structure"
                if include_reference_values
                else "## Required Return Schema"
            )
            parts.append(ref_summary)
            if include_reference_values:
                parts.append(
                    "Your solve() function should return a solution with the SAME top-level "
                    "structure. Match the field names (e.g., 'schedule', 'routes', 'makespan') "
                    "exactly."
                )
            else:
                parts.append(
                    "Your solve() function must match these top-level field names and types exactly."
                )

    # Core type hint
    core_type = record.get("core_type", "")
    if core_type:
        parts.append(f"## Problem Category: {core_type}")

    # Return format instruction
    parts.append("## Return Format")
    parts.append(
        "Your solve() function MUST return the solution as a Python value (list, dict, int, etc.)."
    )
    parts.append("Do NOT use print() to output the solution. Use `return solution` at the end.")
    parts.append("The returned value will be scored automatically.")
    parts.append(
        "CRITICAL: An empty solution (empty schedule, empty routes, cost=0) will be REJECTED. "
        "You MUST actually compute a real solution that visits/assigns all items."
    )

    return "\n\n".join(parts)


def build_requirement_compact(record: dict) -> str:
    """
    コンパクト版の要件テキスト。トークン節約用。
    """
    parts = []
    name = record.get("name", "Unknown")
    core_type = record.get("core_type", "?")
    parts.append(f"Problem: {name} (category: {core_type})")

    description = record.get("description", "")
    if description:
        if len(description) > 500:
            description = description[:500] + "... (truncated)"
        parts.append(f"Description: {description}")

    requirements = record.get("requirements", {})
    if requirements:
        obj = requirements.get("objective", "")
        if obj:
            parts.append(f"Objective: {obj}")

    instance = record.get("instance", {})
    if instance:
        inst_summary = {}
        for k, v in instance.items():
            if isinstance(v, (int, float)):
                inst_summary[k] = v
            elif isinstance(v, list):
                inst_summary[k] = f"list[{len(v)}]"
            elif isinstance(v, dict):
                inst_summary[k] = f"dict{{{len(v)} keys}}"
            else:
                inst_summary[k] = type(v).__name__
        parts.append(f"Instance: {json.dumps(inst_summary, ensure_ascii=False)}")

    ref = record.get("reference_value")
    if ref is not None:
        parts.append(f"Reference: {ref:.2f}")

    return "\n".join(parts)

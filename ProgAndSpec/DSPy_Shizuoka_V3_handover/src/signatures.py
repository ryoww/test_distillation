"""DSPy Signature 定義: 問題文 → Python コード。

V1をV2に移植。40+のcore_typeを動的に扱う。
instance_schemaは不要（問題文に含まれる）。
"""

from __future__ import annotations

import dspy


class ParseInstance(dspy.Signature):
    """Given a problem description and the actual instance dictionary with STRUCTURE ANALYSIS,
    output Python code that safely parses the instance dict into typed variables.

    CRITICAL: You MUST create a variable assignment for EVERY key in the instance dict.
    Do NOT omit any keys. Every key must be assigned to a descriptive variable.

    DATA STRUCTURE ANALYSIS (MOST IMPORTANT):
    The instance field includes a STRUCTURE section that tells you the EXACT type of each value.
    Read this carefully to avoid TypeError and AttributeError!

    COMMON PATTERNS:
    - "customers: list[8] of dicts" → Each customer is a DICT with keys like [id, x, y, demand]
      Access: `for c in customers: x = c['x']; demand = c['demand']`
      NOT: `for i in customers: x = customers[i]` (WRONG - customers is list of dicts!)
    - "vehicles: list[4] of dicts" → Each vehicle is a DICT with keys like [id, capacity, fixed_cost]
      Access: `for v in vehicles: cap = v['capacity']`
    - "depot: dict with keys [x, y]" → Access: `depot_x = depot['x']`
    - "jobs: list[10] of dicts" → Each job is a DICT, NOT a scalar
    - "distances: list[20] of floats" → These are scalars, access by index: `distances[i]`

    TYPE SAFETY RULES (to avoid TypeError and AttributeError):
    - ALWAYS use instance.get('key', default) instead of instance['key'] to avoid KeyError
    - For list values: use default=[] (e.g., `jobs = instance.get('jobs', [])`)
    - For dict values: use default={} (e.g., `depot = instance.get('depot', {})`)
    - For scalar values: use default=0 (e.g., `N = instance.get('N', 0)`)
    - For string values: use default='' (e.g., `name = instance.get('name', '')`)
    - Include type comments so the algorithm knows what type each variable is

    Rules:
    - Output variable assignments for ALL keys
    - For list/dict values, include a brief comment about the structure and nested keys
    - Do NOT define functions or use complex logic
    - Use the actual keys from the instance dict, NOT the variable names from the problem

    Example:
    If instance = {"customers": [{"id": 1, "x": 9, "y": -8, "demand": 8}, ...], "depot": {"x": 0, "y": 0}, "vehicles": [{"id": 1, "capacity": 8, "fixed_cost": 296, "cost_per_km": 1.54}]}
    Output:
    customers = instance.get('customers', [])  # list[8] of dicts with keys [id, x, y, demand]
    depot = instance.get('depot', {})  # dict with keys [x, y]
    vehicles = instance.get('vehicles', [])  # list[4] of dicts with keys [id, capacity, fixed_cost, cost_per_km]
    """

    requirement: str = dspy.InputField(desc="Problem description mentioning variable names")
    instance: str = dspy.InputField(
        desc="Instance dict with STRUCTURE ANALYSIS showing exact types and nested keys for each field"
    )
    parse_code: str = dspy.OutputField(
        desc="Python variable assignment statements using .get() with type-safe defaults for EVERY key. Include structure comments describing nested data."
    )


class GenerateOptimizationAlgorithm(dspy.Signature):
    """Write one Python function `def solve(instance):` that solves the optimisation problem in the requirement and RETURNS the solution. Never print it.

    Contract
    1. Match the "Required Return Schema" exactly: the same top-level field names and the same value shapes (a dict keyed by id stays a dict, a list stays a list). Fill the numeric objective field with the value your own solution actually achieves; recompute it from the solution before returning.
    2. Read instance keys from the STRUCTURE section of the requirement and access them with `.get()` and safe defaults. Never invent keys.
    3. Never return an empty or placeholder solution. If the main method fails, fall back to a simple constructive heuristic that still satisfies every constraint.
    4. Finish within 30 seconds. Give every solver a time limit (about 20 seconds).
    5. Allowed imports: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools. Nothing else. Wrap the body in try/except and use the fallback on any failure.
    6. Valid Python only: balanced brackets, real `for` loops, no bare generator expressions as statements.

    Method
    - Instances are small (a few to a few dozen entities). Prefer an exact method: enumerate permutations or subsets when the count is tiny (up to about 8 items or 9 jobs); otherwise model the problem with CP-SAT (`from ortools.sat.python import cp_model`) or as an LP (`from scipy.optimize import linprog`) and take the optimal value from the solver.
    - Before returning, check your solution against every constraint listed in the requirement. Repair it or use the fallback if anything is violated.
    - Keep ids exactly as the instance gives them (they may start at 0 or at 1) and use them the way the schema shows.
    """

    requirement: str = dspy.InputField(
        desc="Full problem description including input/output format, instance data (raw JSON), and required return schema"
    )
    core_type: str = dspy.InputField(
        desc="Problem category (e.g., スケジューリング_混合整数計画, 配送・輸送_混合整数計画)"
    )
    parse_code: str = dspy.InputField(
        desc="Generic safe accessor helpers (get_list, get_dict, get_scalar). Instance keys are in the requirement text."
    )
    algorithm_code: str = dspy.OutputField(
        desc="Complete Python source code defining `solve(instance)`. MUST return a NON-EMPTY solution matching the required schema."
    )


class ImproveAlgorithm(dspy.Signature):
    """Given an existing solve() implementation, its parse code, and diagnostic
    feedback about its failure or suboptimality, output an improved version.

    HARD RULES:
    1. Keep the signature `solve(instance)`.
    2. Address the SPECIFIC issues listed in the feedback (error type, missing
       fields, empty output, timeout, worse-than-current-baseline, etc.).
    3. Never return empty containers or cost=inf as a "solution".
    4. Runtime budget: 30 seconds max.

    ALLOWED IMPORTS: math, random, heapq, itertools, collections, functools,
    typing, bisect, operator, json, copy, re, ortools, scipy, pulp, networkx, numpy.
    FORBIDDEN: os, sys, subprocess, socket, pickle, requests, urllib, importlib, ctypes.

    Use the feedback to decide the next tactic (switch solver, add fallback,
    fix data access, change heuristic, etc.). Explain the reasoning briefly.
    """

    original_code: str = dspy.InputField(desc="Current algorithm code")
    parse_code: str = dspy.InputField(desc="Current parsing helpers")
    feedback: str = dspy.InputField(
        desc="Diagnostic feedback: error category, cost gap vs reference, structural issues"
    )
    core_type: str = dspy.InputField(desc="Problem category")
    return_schema: str = dspy.InputField(
        desc=(
            "Required return schema: the top-level field names and types solve() must "
            "return. Contains no target value."
        )
    )
    improved_parse_code: str = dspy.OutputField(desc="Improved parsing helpers (or keep the same)")
    improved_code: str = dspy.OutputField(
        desc="Improved solve(instance) code that addresses the feedback"
    )

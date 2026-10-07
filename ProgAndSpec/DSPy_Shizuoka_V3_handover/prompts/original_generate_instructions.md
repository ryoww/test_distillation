Given a natural-language description of an optimization problem, output
an executable Python function `solve(instance) -> solution`.

HARD RULES (never violate):
1. Define exactly one top-level function: `def solve(instance):`
2. RETURN the solution (never print). The returned object MUST match the
   "Required Return Schema" section from the requirement — same
   top-level field names and non-empty content.
3. NEVER return an empty schedule / empty routes / cost=0 — such solutions
   are rejected with score=0.1.
4. Runtime budget: 30 seconds max. For OR-Tools, set a solver time limit
   (e.g., `solver.parameters.max_time_in_seconds = 20`).
5. Wrap the algorithm body in try/except. On any failure, fall back to a
   greedy heuristic that produces a NON-EMPTY solution.
6. SYNTAX SAFETY (avoid exec errors): the code must be valid Python.
   Balance every parenthesis/bracket. NEVER write a trailing generator like
   `model.Add(expr) for x in items` — instead use a real loop:
   `for x in items: model.Add(expr)`. Test comprehensions have balanced ().

ALLOWED IMPORTS: math, random, heapq, itertools, collections, functools,
typing, bisect, operator, json, copy, re, ortools, scipy, pulp, networkx, numpy.
FORBIDDEN: os, sys, subprocess, socket, pickle, requests, urllib, importlib, ctypes.

Choose the algorithm and problem-specific tactics yourself. The requirement
text contains the instance data (raw JSON) and required return schema —
read them carefully.

Write one Python function `def solve(instance):` that solves the optimisation problem in the requirement and RETURNS the solution. Never print it.

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

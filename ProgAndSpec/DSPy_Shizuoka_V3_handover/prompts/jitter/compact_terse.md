Return a Python module defining `solve(instance)` that solves the requirement's optimisation problem and returns (not prints) the solution.

- Output must follow the "Required Return Schema" field for field and shape for shape (dicts keyed by id stay dicts, lists stay lists). Set the objective field to the value your solution attains, recomputed from the solution itself.
- Keys come from the requirement's STRUCTURE section only; read them via `.get()` with safe defaults. Do not invent keys.
- No empty or placeholder solutions. On failure of the main method, fall back to a constructive heuristic that satisfies all constraints.
- Total time under 30 s; cap each solver at about 20 s.
- Imports limited to math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools. Wrap the body in try/except with the fallback.
- Code must be valid Python: balanced brackets, explicit `for` loops, no bare generator expressions as statements.
- Instances are small (a few to a few dozen entities): enumerate permutations/subsets for tiny cases (≲8 items or 9 jobs), otherwise use CP-SAT (`from ortools.sat.python import cp_model`) or an LP (`from scipy.optimize import linprog`) and take the solver's optimum.
- Verify every listed constraint before returning; repair or fall back if any fails.
- Keep instance ids as given (0- or 1-based) and use them as the schema shows.

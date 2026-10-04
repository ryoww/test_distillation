Write one Python function `def solve(instance):` that solves the optimisation problem in the requirement and RETURNS the solution. Never print it.

Contract
1. Match the "Required Return Schema" exactly: the same top-level field names and the same value shapes (a dict keyed by id stays a dict, a list stays a list). Fill the numeric objective field with the value your own solution actually achieves; recompute it from the solution before returning.
2. Read instance keys from the STRUCTURE section of the requirement and access them with `.get()` and safe defaults. Never invent keys.
3. Never return an empty or placeholder solution. If the main method fails, fall back to a simple constructive heuristic that still satisfies every constraint.
4. Respect the runtime limit stated in the problem and finish well inside it (aim for about 300 seconds). Put an explicit time limit on every solver call and every search loop.
5. Allowed imports: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools. Nothing else. Wrap the body in try/except and use the fallback on any failure.
6. Valid Python only: balanced brackets, real `for` loops, no bare generator expressions as statements.

Method
- Instances can be large (hundreds to thousands of entities, thousands of binary decisions), so exact enumeration will not finish. Build a feasible solution first with a constructive heuristic, then improve it with a time-limited solver (CP-SAT, HiGHS through scipy.optimize.milp or linprog, OR-Tools routing) or with local search, and return the best feasible solution found. Use an exact model only for parts that are small enough to solve within the limit.
- Before returning, check your solution against every constraint listed in the requirement. Repair it or use the fallback if anything is violated.
- Keep ids exactly as the instance gives them (they may start at 0 or at 1) and use them the way the schema shows.

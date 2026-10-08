Your task: implement `def solve(instance):` in Python for the optimisation problem described in the requirement. The function must RETURN its solution; do not print it.

How to solve
- Instances are small, from a few to a few dozen entities. Use an exact method where you can: brute-force permutations or subsets when there are only a handful (roughly 8 items or 9 jobs at most); for anything larger, build a CP-SAT model (`from ortools.sat.python import cp_model`) or an LP (`from scipy.optimize import linprog`) and report the solver's optimal value.
- Check the finished solution against each constraint in the requirement before returning it; if one is violated, repair the solution or switch to the fallback.
- Use ids exactly as they appear in the instance (they can start at 0 or at 1), in the form the schema shows.

Rules you must follow
1. Return exactly the "Required Return Schema": identical top-level field names and identical value shapes (keep dicts keyed by id as dicts and lists as lists). The numeric objective field must hold the value your solution really achieves, recomputed from the solution just before you return.
2. Take instance keys only from the STRUCTURE section of the requirement, read them with `.get()` and safe defaults, and never make up keys.
3. Never return an empty or placeholder answer. When the main method fails, use a simple constructive heuristic that still meets every constraint.
4. Stay under 30 seconds in total and give each solver a time limit of about 20 seconds.
5. Import only from: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools. Put the body in try/except and fall back on any failure.
6. Write valid Python: balanced brackets, real `for` loops, and no bare generator expressions used as statements.

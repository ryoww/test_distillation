Write one Python function `def solve(instance):` that solves the optimization problem and RETURNS the solution. Never print.

Contract
1. Match the required return schema exactly: same top-level field names and value shapes. Fill every field with real values. Extract a complete solution, not just the objective. Recompute the numeric objective from the returned solution; if multiple objective fields exist, set them to it unless defined otherwise.
2. Read only keys shown in STRUCTURE using `.get()` and safe defaults. Never mutate `instance` or nested values; copy lists/dicts when adjusting. Preserve ids exactly. Do not invent entities/constraints. If data is missing, use the safest schema-valid default, then repair/fallback if infeasible.
3. Never return empty/placeholder solutions. Fallback must be a real feasible or penalty-feasible solution: serve/assign all required items, open enough arcs/facilities, produce/allocate demand, or list unserved only when penalty is allowed.
4. Respect the stated runtime limit; if absent, assume 30 s. Do not import time. Use solver time limits ~80% of it and bounded loops/iteration budgets. Stop at limit and use best feasible solution.
5. Allowed imports only: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools. Wrap the body in try/except and return fallback on any failure.
6. Valid Python only: balanced brackets, real loops, no bare generator expressions as statements, no runaway/repeated comments, every try has except/finally, no infinite loops.

Method
- Choose by size/structure. Tiny: exact enumeration. Small/medium: CP-SAT/LP. Large/hard: constructive heuristic + local search or specialized solver. Do not finish hard fixed-charge, routing, production, packing, or scheduling problems with a naive greedy pass.
- Separate hard constraints from penalty terms. Hard constraints must hold in the returned solution. Penalty terms may be violated only when the objective explicitly prices them.
- Packing/cutting: generate only feasible patterns (geometry, capacities, lot sizes, pattern limits). Then choose pattern counts with CP-SAT/LP or greedy+repair. If pattern count is limited, create diverse feasible patterns; fallback to minimal feasible single-item patterns if needed.
- Scheduling/rostering: construct valid states/blocks or use CP-SAT/rolling horizon. Enforce sequence rules, consecutive limits, weekly limits, rest after blocks, qualifications, and exact one-state-per-day. Do not assign greedily and then patch hard rules.
- Facility/inventory/distribution: choose open facilities/arcs, assign customers/ODs, then compute period-by-period procurement/flow/inventory. Check throughput, storage, safety stock, nonnegative inventory, balances, and capacities. If infeasible, reassign, open more, or repair.
- VRP/routing: use OR-Tools Routing with capacity/time-window dimensions, unserved penalty if allowed, and local search. Otherwise build feasible routes and repair violations.
- Fixed-charge network flow: use marginal/slope costs, successive shortest paths, or path-based CP-SAT, then local search. Ensure no flow on closed arcs and capacities respected.
- CP-SAT uses integers: scale continuous data safely and divide back. Use LP only when integrality is not required or as relaxation.

Validation before return
- Check every stated constraint against the returned structures: demand/flow conservation, capacities, no flow on unopened arcs, route start/end, load, time windows, service times, max route time, fleet limits, production capacity, inventory nonnegative, pattern/lot limits, scheduling sequence rules, and penalty/unserved rules.
- Recompute objective from the returned solution using the problem's cost formula. If any hard constraint is violated, repair (drop flow, open arcs, split/repair routes, adjust production/inventory, reassign customers) or return fallback.
- Ensure ids in routes, flows, plans, rosters, and lists match the instance. Do not return zeros/empty lists when all items must be served/assigned, unless unserved/penalty entries are explicitly allowed.
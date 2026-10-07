import math
from itertools import pairwise


def solve_cvrp(depot, customers, capacity, vehicles):
    """Exact CVRP: Held-Karp over capacity-feasible subsets, then set-partition DP.

    Returns (total distance, list of routes as customer index lists; index i is customers[i-1]).
    """
    points = [depot, *customers]
    n = len(customers)
    inf = math.inf
    dist = [[math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for b in points] for a in points]
    full = 1 << n
    load = [0] * full
    for s in range(1, full):
        low = (s & -s).bit_length() - 1
        load[s] = load[s ^ (1 << low)] + customers[low]["demand"]

    # path[s][j] = shortest depot -> (all of s) path ending at customer j.
    path = [[inf] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for j in range(n):
        path[1 << j][j] = dist[0][j + 1]
    for s in range(1, full):
        if load[s] > capacity:
            continue
        for j in range(n):
            if not s & (1 << j) or path[s][j] == inf:
                continue
            for k in range(n):
                t = s | (1 << k)
                if s & (1 << k) or load[t] > capacity:
                    continue
                cand = path[s][j] + dist[j + 1][k + 1]
                if cand < path[t][k]:
                    path[t][k], parent[t][k] = cand, j

    # tour[s] = shortest closed route over subset s, closed by returning to the depot.
    tour, tour_end = [inf] * full, [-1] * full
    for s in range(1, full):
        for j in range(n):
            if path[s][j] < inf and path[s][j] + dist[j + 1][0] < tour[s]:
                tour[s], tour_end[s] = path[s][j] + dist[j + 1][0], j

    # best[k][s] = min distance covering s with at most k routes; the route holding the
    # lowest-numbered customer of s is fixed so each partition is counted once.
    best = [[inf] * full for _ in range(vehicles + 1)]
    choice = [[0] * full for _ in range(vehicles + 1)]
    for k in range(vehicles + 1):
        best[k][0] = 0.0
    for k in range(1, vehicles + 1):
        for s in range(1, full):
            low_bit, sub = s & -s, s
            while sub:
                if sub & low_bit and tour[sub] < inf and tour[sub] + best[k - 1][s ^ sub] < best[k][s]:
                    best[k][s], choice[k][s] = tour[sub] + best[k - 1][s ^ sub], sub
                sub = (sub - 1) & s
    if best[vehicles][full - 1] == inf:
        raise ValueError("no feasible CVRP solution")

    routes = []
    s, k = full - 1, vehicles
    while s:
        sub = choice[k][s]
        order, j, t = [], tour_end[sub], sub
        while j != -1:
            order.append(j + 1)
            j, t = parent[t][j], t ^ (1 << j)
        routes.append(order[::-1])
        s ^= sub
        k -= 1
    return best[vehicles][full - 1], routes


def solve(instance):
    """Multi-depot VRP with fixed customer-to-depot assignment.

    Because assignments are given, the problem splits into one independent CVRP per depot.
    Route node 0 denotes that route's own depot; distances are unrounded Euclidean.
    """
    routes, total = {}, 0.0
    for depot in instance["depots"]:
        mine = [c for c in instance["customers"] if c["assigned_depot"] == depot["id"]]
        if not mine:
            continue
        _, orders = solve_cvrp(depot, mine, instance["vehicle_capacity"], depot["num_vehicles"])
        points = [depot, *mine]
        for order in orders:
            closed = [0, *order, 0]
            length = sum(
                math.hypot(points[u]["x"] - points[v]["x"], points[u]["y"] - points[v]["y"])
                for u, v in pairwise(closed)
            )
            routes[str(len(routes) + 1)] = {
                "depot": depot["id"],
                "route": [0, *(mine[i - 1]["id"] for i in order), 0],
                "distance": length,
            }
            total += length
    return {"routes": routes, "total_distance": total}

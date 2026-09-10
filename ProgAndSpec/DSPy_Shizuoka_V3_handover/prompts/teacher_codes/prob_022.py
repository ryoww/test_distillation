import math
from itertools import pairwise


def solve(instance):
    """VRP with time windows: exact via route enumeration + set-partition DP.

    Travel time equals Euclidean distance (speed 1); arriving early means waiting until the
    window opens, each customer takes service_time, and every vehicle must be back at the
    depot before its window closes. A depth-first search enumerates all time- and
    capacity-feasible routes and records, per customer subset, the shortest closed route.
    A DP then partitions the customers into at most num_vehicles subsets.
    """
    depot = instance["depot"]
    customers = instance["customers"]
    points = [depot, *customers]
    n = len(customers)
    capacity = instance["vehicle_capacity"]
    vehicles = instance["num_vehicles"]
    service = instance["service_time"]
    open_at, close_at = depot["time_window"]

    def leg(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    dist = [[leg(a, b) for b in points] for a in points]
    windows = [c["time_window"] for c in customers]
    full = 1 << n
    load = [0] * full
    for s in range(1, full):
        low = (s & -s).bit_length() - 1
        load[s] = load[s ^ (1 << low)] + customers[low]["demand"]

    # tour[s] = shortest feasible closed route over subset s; tour_order[s] = its visit order.
    tour = [math.inf] * full
    tour_order = [[] for _ in range(full)]

    def extend(mask, last, clock, length, order):
        if mask:
            back = length + dist[last][0]
            if clock + dist[last][0] <= close_at and back < tour[mask]:
                tour[mask] = back
                tour_order[mask] = list(order)
        for j in range(n):
            bit = 1 << j
            if mask & bit or load[mask | bit] > capacity:
                continue
            arrive = clock + dist[last][j + 1]
            if arrive > windows[j][1]:  # late now means late forever: prune
                continue
            order.append(j + 1)
            extend(mask | bit, j + 1, max(arrive, windows[j][0]) + service,
                   length + dist[last][j + 1], order)
            order.pop()

    extend(0, 0, float(open_at), 0.0, [])

    # best[k][s] = min total distance covering subset s with at most k routes. The route that
    # contains the lowest-numbered customer of s is fixed to avoid counting partitions twice.
    inf = math.inf
    best = [[inf] * full for _ in range(vehicles + 1)]
    choice = [[0] * full for _ in range(vehicles + 1)]
    for k in range(vehicles + 1):
        best[k][0] = 0.0
    for k in range(1, vehicles + 1):
        for s in range(1, full):
            low_bit = s & -s
            sub = s
            while sub:
                if sub & low_bit and tour[sub] < inf:
                    cand = tour[sub] + best[k - 1][s ^ sub]
                    if cand < best[k][s]:
                        best[k][s] = cand
                        choice[k][s] = sub
                sub = (sub - 1) & s
    if best[vehicles][full - 1] == inf:
        raise ValueError("no feasible set of routes")

    routes, total = {}, 0.0
    s, k = full - 1, vehicles
    while s:
        sub = choice[k][s]
        order = tour_order[sub]
        closed = [0, *order, 0]
        length = sum(dist[u][v] for u, v in pairwise(closed))
        routes[str(len(routes) + 1)] = {"route": closed, "distance": length}
        total += length
        s ^= sub
        k -= 1
    return {"routes": routes, "total_distance": total}

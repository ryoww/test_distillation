import itertools


def solve(instance):
    """Single-allocation hub location, enumerated over hub sets.

    Cost = sum over cities of (outbound + inbound flow) x distance to the assigned hub.
    Given a hub set, each city is best served by its nearest hub, so only the hub set
    needs enumeration. The discount only affects inter-hub legs, which are not costed here.
    """
    flow = instance["flow"]
    distance = instance["distance"]
    n = len(instance["nodes"])
    p = instance["num_hubs"]

    volume = [sum(flow[i]) + sum(flow[j][i] for j in range(n)) for i in range(n)]

    best_cost, best_hubs, best_assign = None, None, None
    for hubs in itertools.combinations(range(n), p):
        # Ties on distance go to the lower-numbered hub.
        assign = [min(hubs, key=lambda h: (distance[i][h], h)) for i in range(n)]
        cost = sum(volume[i] * distance[i][assign[i]] for i in range(n))
        if best_cost is None or cost < best_cost:
            best_cost, best_hubs, best_assign = cost, hubs, assign

    return {
        "min_cost": int(best_cost),
        "hubs": list(best_hubs),
        "assignment": {str(i): h for i, h in enumerate(best_assign)},
    }

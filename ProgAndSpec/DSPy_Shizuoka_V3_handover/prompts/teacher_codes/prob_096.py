import math

from ortools.sat.python import cp_model


def solve(instance):
    """Capacitated covering assignment: open centres at minimum fixed cost so that every
    zone is assigned to exactly one open centre within the radius, respecting capacity."""
    zones = instance["zones"]
    centers = instance["centers"]
    radius = instance["radius"]
    demand = instance["demand"]
    capacity = instance["capacity"]
    fixed_cost = instance["fixed_cost"]

    def dist(a, b):
        # Distances in this problem family are Euclidean rounded to integers.
        return round(math.hypot(a["x"] - b["x"], a["y"] - b["y"]))

    model = cp_model.CpModel()
    opened = [model.NewBoolVar(f"y_{k}") for k in range(len(centers))]
    assign = {}
    for z, zone in enumerate(zones):
        reachable = [k for k, c in enumerate(centers) if dist(zone, c) <= radius]
        if not reachable:
            raise ValueError(f"zone {z} is outside every centre's radius")
        for k in reachable:
            assign[z, k] = model.NewBoolVar(f"x_{z}_{k}")
            model.AddImplication(assign[z, k], opened[k])  # only open centres serve
        model.AddExactlyOne(assign[z, k] for k in reachable)

    for k in range(len(centers)):
        load = [demand[z] * assign[z, k] for z in range(len(zones)) if (z, k) in assign]
        model.Add(sum(load) <= capacity[k])

    model.Minimize(sum(fixed_cost[k] * opened[k] for k in range(len(centers))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 8.0
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT did not prove optimality")

    chosen = [k for k in range(len(centers)) if solver.Value(opened[k])]
    return {
        "min_cost": sum(fixed_cost[k] for k in chosen),
        "opened_centers": chosen,
    }

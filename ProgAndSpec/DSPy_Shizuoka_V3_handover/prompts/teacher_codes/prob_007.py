from ortools.sat.python import cp_model

SIZE_RANK = {"small": 0, "medium": 1, "large": 2}


def solve(instance):
    """Gate assignment: minimise the number of flights left without a gate."""
    gates = instance["gates"]
    flights = instance["flights"]
    model = cp_model.CpModel()

    # x[f, g] = 1 if flight f uses gate g. Only size-compatible pairs get a variable.
    x = {}
    for f in flights:
        for g in gates:
            if SIZE_RANK[g["size_class"]] >= SIZE_RANK[f["aircraft_size"]]:
                x[f["id"], g["id"]] = model.NewBoolVar(f"x_{f['id']}_{g['id']}")

    # Each flight uses at most one gate.
    assigned = []
    for f in flights:
        own = [x[f["id"], g["id"]] for g in gates if (f["id"], g["id"]) in x]
        if own:
            model.AddAtMostOne(own)
        assigned.extend(own)

    # Two flights whose stay overlaps in time cannot share a gate.
    for i, a in enumerate(flights):
        for b in flights[i + 1:]:
            if a["arrival_time"] < b["departure_time"] and b["arrival_time"] < a["departure_time"]:
                for g in gates:
                    if (a["id"], g["id"]) in x and (b["id"], g["id"]) in x:
                        model.AddBoolOr([x[a["id"], g["id"]].Not(), x[b["id"], g["id"]].Not()])

    # Primary: unassigned flights. Secondary (tie-break): prefer low gate numbers.
    unassigned = len(flights) - sum(assigned)
    weight = sum(g["id"] for g in gates) * len(flights) + 1
    model.Minimize(unassigned * weight + sum(g * v for (_, g), v in x.items()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 8.0
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT did not prove optimality")

    assignment = {str(f): g for (f, g), v in x.items() if solver.Value(v)}
    return {
        "gate_assignment": assignment,
        "unassigned_count": len(flights) - len(assignment),
    }

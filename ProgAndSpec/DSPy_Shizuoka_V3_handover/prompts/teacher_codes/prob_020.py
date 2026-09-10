from ortools.sat.python import cp_model


def activities_of(instance):
    """Flatten projects into activities with unique ids.

    Some instances repeat one id for every activity of a project (1, 1, 1, ...); then the
    ids are renumbered as first id + position, and predecessors are read as earlier
    activities of the same project. Instances with unique ids are used as-is.
    """
    keys = sorted(instance["projects"], key=int)
    rows = [row for key in keys for row in instance["projects"][key]]
    unique = len({row["id"] for row in rows}) == len(rows)
    activities = []
    for key in keys:
        members = instance["projects"][key]
        ids = [row["id"] if unique else members[0]["id"] + k for k, row in enumerate(members)]
        for k, row in enumerate(members):
            activities.append(
                {
                    "id": ids[k],
                    "project": int(key),
                    "duration": row["duration"],
                    "resource": row["resource_usage"],
                    "predecessors": sorted(set(row["predecessors"]) & set(ids[:k])),
                }
            )
    return activities


def solve(instance):
    """Multi-project resource levelling: minimise the peak of shared resource usage.

    CP-SAT with a cumulative constraint whose capacity is itself the variable to minimise,
    bounded above by the shared resource capacity and below by the largest single usage.
    """
    activities = activities_of(instance)
    horizon = instance["horizon"]
    capacity = instance["resource_capacity"]
    duration = {a["id"]: a["duration"] for a in activities}

    model = cp_model.CpModel()
    starts, intervals = {}, []
    for a in activities:
        start = model.NewIntVar(0, horizon - a["duration"], f"s_{a['id']}")
        starts[a["id"]] = start
        intervals.append(model.NewFixedSizeIntervalVar(start, a["duration"], f"i_{a['id']}"))
    for a in activities:
        for p in a["predecessors"]:
            model.Add(starts[a["id"]] >= starts[p] + duration[p])

    peak = model.NewIntVar(max(a["resource"] for a in activities), capacity, "peak")
    model.AddCumulative(intervals, [a["resource"] for a in activities], peak)
    # Primary: peak usage. Secondary (tie-break): schedule as early as possible.
    model.Minimize((len(activities) * horizon + 1) * peak + sum(starts.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 8.0
    solver.parameters.num_workers = 1
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT did not prove optimality")

    schedule = {}
    for a in activities:
        begin = solver.Value(starts[a["id"]])
        schedule[str(a["id"])] = {
            "start": begin,
            "end": begin + a["duration"],
            "project": a["project"],
        }
    return {
        "schedule": schedule,
        "peak_resource_usage": solver.Value(peak),
        "makespan": max(entry["end"] for entry in schedule.values()),
    }

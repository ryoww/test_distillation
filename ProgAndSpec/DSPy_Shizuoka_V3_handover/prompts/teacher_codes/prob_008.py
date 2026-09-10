from ortools.sat.python import cp_model

DAY_START, DAY_END = 8, 18


def solve(instance):
    """Operating-room schedule: every surgery starts on the hour in the room matching its
    type, rooms and surgeons never double-book, and the sum of start hours is minimised."""
    rooms = instance["operating_rooms"]
    surgeries = instance["surgeries"]
    room_of = {r["type"]: r["id"] for r in rooms}  # one room per type

    model = cp_model.CpModel()
    start = {}
    by_room = {}
    by_surgeon = {}
    for s in surgeries:
        hours = -(-s["duration"] // 60)  # whole hours needed, rounded up
        var = model.NewIntVar(DAY_START, DAY_END - hours, f"start_{s['id']}")
        interval = model.NewIntervalVar(var, hours, var + hours, f"iv_{s['id']}")
        start[s["id"]] = var
        by_room.setdefault(room_of[s["type"]], []).append(interval)
        by_surgeon.setdefault(s["surgeon_id"], []).append(interval)

    for intervals in by_room.values():
        model.AddNoOverlap(intervals)
    for intervals in by_surgeon.values():
        model.AddNoOverlap(intervals)
    model.Minimize(sum(start.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 8.0
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError("CP-SAT did not prove optimality")

    schedule = {}
    for s in surgeries:
        t = solver.Value(start[s["id"]])
        schedule[str(s["id"])] = {
            "room": room_of[s["type"]],
            "start_time": t,
            "end_time": t + s["duration"] / 60,
        }
    return {
        "schedule": schedule,
        "total_start_time": int(solver.ObjectiveValue()),
        "all_scheduled": True,
    }

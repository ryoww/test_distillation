from ortools.sat.python import cp_model


def solve(instance):
    """Event staff roster: cover every (role, slot) minimum with the fewest placements.

    No constraint links different time slots, so each slot is an independent small
    assignment problem: a staff member takes at most one role per slot, and only roles
    whose skill they hold, in slots they are available.
    """
    roles = instance["roles"]
    slots = instance["time_slots"]
    staff = instance["staff"]
    requirements = instance["role_requirements"]

    roster = {str(person["id"]): {} for person in staff}
    total = 0
    for slot in slots:
        model = cp_model.CpModel()
        # x[s, r] = 1 if staff s works role r in this slot (only allowed pairs get a variable).
        x = {}
        for s, person in enumerate(staff):
            if slot["id"] not in person["available_slots"]:
                continue
            for r, role in enumerate(roles):
                if role["required_skill"] in person["skills"]:
                    x[s, r] = model.NewBoolVar(f"x_{s}_{r}")

        for s in range(len(staff)):
            own = [x[s, r] for r in range(len(roles)) if (s, r) in x]
            if own:
                model.AddAtMostOne(own)
        for r, role in enumerate(roles):
            need = requirements[str(role["id"])][str(slot["id"])]
            model.Add(sum(x[s, r] for s in range(len(staff)) if (s, r) in x) >= need)

        # Primary: number of placements. Secondary (tie-break): prefer low staff numbers.
        count = sum(x.values())
        spread = sum(s * var for (s, _), var in x.items())
        model.Minimize((len(staff) * len(x) + 1) * count + spread)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 8.0
        solver.parameters.num_workers = 1
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError("CP-SAT did not prove optimality")

        for (s, r), var in x.items():
            if solver.Value(var):
                roster[str(staff[s]["id"])][slot["time_slot"]] = roles[r]["name"]
                total += 1

    return {"roster": roster, "total_assignments": total, "feasible": True}

import itertools


def solve(instance):
    """Broadcast lineup: choose programs (at most one per slot) maximising total rating.

    A program fits a slot when duration <= max_duration, a pure threshold rule. So once a
    subset of programs is fixed, it is placeable iff matching the longest program to the
    longest slot, second longest to second longest, ... never violates the threshold.
    Only the subset needs enumeration (C(12, <=8) = 3797 candidates).
    """
    slots = sorted(instance["time_slots"], key=lambda s: (-s["max_duration"], s["id"]))
    programs = sorted(instance["programs"], key=lambda p: p["id"])
    minimums = instance["genre_minimums"]

    best_total, best_order = None, []
    for size in range(min(len(slots), len(programs)) + 1):
        for chosen in itertools.combinations(programs, size):
            counts = {}
            for p in chosen:
                counts[p["genre"]] = counts.get(p["genre"], 0) + 1
            if any(counts.get(genre, 0) < need for genre, need in minimums.items()):
                continue
            ordered = sorted(chosen, key=lambda p: (-p["duration"], p["id"]))
            if any(p["duration"] > s["max_duration"] for p, s in zip(ordered, slots)):
                continue
            total = sum(p["expected_rating"] for p in chosen)
            # Ties keep the first subset enumerated (lexicographically smallest ids).
            if best_total is None or total > best_total:
                best_total, best_order = total, ordered

    if best_total is None:
        raise ValueError("no lineup satisfies the genre minimums")

    lineup = {str(p["id"]): s["time_slot"] for p, s in zip(best_order, slots)}
    return {
        "lineup": dict(sorted(lineup.items(), key=lambda kv: int(kv[0]))),
        "total_expected_rating": float(best_total),
    }

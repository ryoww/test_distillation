def solve(instance):
    """Subset sum via reachable-sum DP; picks the sum closest to the target."""
    numbers = instance["numbers"]
    target = instance["target"]
    n = len(numbers)

    # reach[i] = set of sums achievable using numbers[i:], built from the back.
    reach = [set() for _ in range(n + 1)]
    reach[n] = {0}
    for i in range(n - 1, -1, -1):
        reach[i] = reach[i + 1] | {s + numbers[i] for s in reach[i + 1]}

    # Closest sum to the target; ties resolved toward the smaller sum.
    best = min(reach[0], key=lambda s: (abs(s - target), s))

    # Reconstruct greedily from the front: take numbers[i] whenever the rest can finish.
    chosen, remaining = [], best
    for i in range(n):
        if remaining - numbers[i] in reach[i + 1]:
            chosen.append(i)
            remaining -= numbers[i]

    return {
        "min_difference": abs(best - target),
        "achieved_sum": best,
        "chosen_indices": chosen,
    }

from scipy.optimize import linprog

FACE_VALUE = 100  # bonds redeem at par; prices 90-105 and coupons 3-8 are quoted per 100


def solve(instance):
    """Cash-flow matching LP: buy bonds at minimum cost so every year's liability is met.

    Variables: x_b >= 0 units of each bond, s_t >= 0 surplus carried from year t to t+1.
    Year t balance: sum_b cashflow_b(t) * x_b + s_{t-1} - s_t = liability_t, where a bond
    pays its coupon every year up to maturity and the face value at maturity.
    """
    periods = instance["periods"]
    bonds = instance["bonds"]
    liabilities = instance["liabilities"]
    nb = len(bonds)

    a_eq, b_eq = [], []
    for t in range(periods):
        year = t + 1
        row = [0.0] * (nb + periods)
        for b, bond in enumerate(bonds):
            if year <= bond["maturity"]:
                row[b] += bond["coupon"]
            if year == bond["maturity"]:
                row[b] += FACE_VALUE
        if t > 0:
            row[nb + t - 1] = 1.0  # surplus carried in from the previous year
        row[nb + t] = -1.0  # surplus carried out to the next year
        a_eq.append(row)
        b_eq.append(float(liabilities[t]))

    result = linprog(
        c=[float(b["price"]) for b in bonds] + [0.0] * periods,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * (nb + periods),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    holdings = [max(float(v), 0.0) for v in result.x[:nb]]
    return {
        "min_cost": sum(h * b["price"] for h, b in zip(holdings, bonds)),
        "feasible": True,
        "bond_holdings": {str(b["id"]): h for h, b in zip(holdings, bonds)},
    }

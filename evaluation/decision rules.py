import math


def U(p):
    p = min(max(p, 0.0), 1.0)
    if p in (0.0, 1.0):
        h = 0.0
    else:
        h = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)
    m = 1.0 - abs(2.0 * p - 1.0)
    return h + 0.5 * m


def solve_pstar(u80):
    lo, hi = 0.0, 0.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if U(mid) < u80:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def audit_gates(p, qhat, u80, th_spec95, th_sens95):
    if u80 < 0:
        raise ValueError("u80 must be non-negative")
    u_mid = U(0.5)
    if u80 >= u_mid:
        p_star = None
        lo_u = None
        hi_u = float("-inf")
        width_u = 0.0
    else:
        p_star = solve_pstar(u80)
        lo_u = p_star
        hi_u = 1.0 - p_star
        width_u = hi_u - lo_u
    if qhat >= 0.5:
        lo_c = 1.0 - qhat
        hi_c = qhat
        closed = True
    else:
        lo_c = qhat
        hi_c = 1.0 - qhat
        closed = False
    width_c = hi_c - lo_c
    if width_u == 0.0 and width_c == 0.0:
        gate = "no abstention"
    elif width_u == 0.0:
        gate = "conformal only"
    elif width_c == 0.0:
        gate = "uncertainty only"
    elif abs(width_c - width_u) < 1e-12:
        gate = "equivalent up to endpoints"
    elif width_c > width_u:
        gate = "conformal binds"
    else:
        gate = "uncertainty binds"
    hi_gate = max(hi_u, hi_c)
    lo_gate = lo_c if lo_u is None else min(lo_c, lo_u)
    tier_untouched_sufficient = th_spec95 > hi_gate
    out = {
        "p_star": p_star,
        "lo_c": lo_c,
        "hi_c": hi_c,
        "lo_u": lo_u,
        "hi_u": hi_u,
        "lo_gate": lo_gate,
        "hi_gate": hi_gate,
        "gate": gate,
        "conformal_closed": closed,
        "tier_untouched_sufficient": tier_untouched_sufficient,
        "empty_possible": qhat < 0.5,
    }
    if p is not None:
        n_high = n_removed = n_low = n_abst = 0
        for x in p:
            abst_c = (1.0 - qhat <= x <= qhat) if qhat >= 0.5 else (qhat < x < 1.0 - qhat)
            abst_u = U(x) > u80
            neg_only = (x <= qhat) if qhat < 0.5 else (x < 1.0 - qhat)
            low = neg_only and x < th_sens95 and not abst_u
            high = x >= th_spec95
            n_high += int(high)
            n_removed += int(high and (abst_c or abst_u))
            n_low += int(low)
            n_abst += int(abst_c or abst_u)
        out.update({"n_high": n_high, "n_high_removed": n_removed, "n_low": n_low, "n_abstained": n_abst})
    return out

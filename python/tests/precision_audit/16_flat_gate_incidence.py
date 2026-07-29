"""Incidence: which configurations change route if the size margin goes.

The flat multi-attribute dispatchers refuse the Moebius method whenever
any attribute has K_a < r_a + 2. This script sweeps configurations,
records the route chosen with the margin in force and with it stubbed
out, and reports where the two differ.

Removing the margin does not force the Moebius method anywhere: it only
stops forbidding it, leaving the cost model free to choose. The count of
changed cells is therefore the count of cells where the margin, and not
cost, was deciding.
"""
import itertools
import warnings

import numpy as np

import mpt._tensor.dispatch as D
from mpt.tensor import build_exp_tens

P = 1200.0
N = 2


def route(r, K, is_rel, is_per, n_q=500):
    """Route chosen by each flat dispatcher for this configuration."""
    rng = np.random.default_rng(0)
    geom = ([0.025 * P], [r], [is_rel], [is_per], [P])
    dens = build_exp_tens([rng.uniform(0, P, (K, N))],
                          [rng.uniform(0.1, 1.0, (K, N))], *geom,
                          verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ev = D._select_ma_eval(dens, n_q, method="auto")[0]
        except Exception as exc:
            ev = type(exc).__name__
        try:
            ip = D._select_ma_inner_product_method(
                r_vec=np.array([r]), k_vec=np.array([K]), A=1,
                rel_vec=np.array([is_rel]), N_x=N, N_y=N,
                any_per=is_per, any_rel_nonper=is_rel and not is_per,
                any_rel_per=is_rel and is_per,
                sigma_over_P_max=0.025 if (is_rel and is_per) else 0.0,
                user_method="auto", guard_forced_bulger=False)
        except TypeError:
            ip = "n/a"
        except Exception as exc:
            ip = type(exc).__name__
    return ev, ip


def main():
    modes = [("abs_nonper", False, False), ("abs_per", False, True),
             ("rel_nonper", True, False), ("rel_per", True, True)]
    cells = list(itertools.product(modes, range(2, 7), range(0, 4)))
    orig = D._orbit_safe_for_precision

    shipped = {}
    for (name, is_rel, is_per), r, dk in cells:
        shipped[(name, r, dk)] = route(r, r + dk, is_rel, is_per)

    D._orbit_safe_for_precision = lambda r_vec, k_vec: True
    removed = {}
    for (name, is_rel, is_per), r, dk in cells:
        removed[(name, r, dk)] = route(r, r + dk, is_rel, is_per)
    D._orbit_safe_for_precision = orig

    print(f"Flat size-margin incidence over {len(cells)} configurations "
          f"(A=1, N={N}, sigma/P=0.025).\n")
    header = (f"{'mode':<11} {'r':>2} {'K':>2} {'K-r':>3} | "
              f"{'eval: shipped -> margin removed':<34} | "
              f"{'inner product: shipped -> removed':<34}")
    print(header)
    print("-" * len(header))
    n_ev = n_ip = 0
    for (name, is_rel, is_per), r, dk in cells:
        s_ev, s_ip = shipped[(name, r, dk)]
        r_ev, r_ip = removed[(name, r, dk)]
        if s_ev == r_ev and s_ip == r_ip:
            continue
        n_ev += s_ev != r_ev
        n_ip += s_ip != r_ip
        ev_s = f"{s_ev} -> {r_ev}" if s_ev != r_ev else "(unchanged)"
        ip_s = f"{s_ip} -> {r_ip}" if s_ip != r_ip else "(unchanged)"
        print(f"{name:<11} {r:>2} {r + dk:>2} {dk:>3} | {ev_s:<34} | "
              f"{ip_s:<34}")
    print(f"\nChanged: {n_ev} of {len(cells)} on the evaluation path, "
          f"{n_ip} of {len(cells)} on the inner-product path.")


if __name__ == "__main__":
    main()

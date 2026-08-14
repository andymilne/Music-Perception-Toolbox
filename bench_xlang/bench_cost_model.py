"""Cross-language cost-model audit: skip-flag pricing and Möbius
per-matrix costs.

Part 1 (deterministic, no timing): calls the Bulger-vs-Möbius selector
across a (r, K, N) grid with the self-matrix skip flags off (a first
pair: full triple) and on (a later broadcast pair: cross term only),
recording the chosen route and both predicted costs. Rows where the
choice flips are the cells the skip flags exist for.

Part 2 (timing): times ``method='mobius'`` on absolute densities under
three memo regimes -- ``cosine`` fresh (three matrices), ``oneSidedDenom``
fresh (cross + one self), ``cosine`` warm selves (cross only) -- giving
measured per-matrix costs to hold against the pricing's
``n_matrices / 3`` scaling of the fitted whole-triple constants.

Part 3 (timing): at each flip cell from part 1, times both forced
routes in the warm-selves regime and reports whether the flags-on
choice is the faster -- the behavioural claim the skip flags make.

Writes ``bench_cost_model_python.csv``. The MATLAB counterpart is
``bench_cost_model.m``.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import mpt                                                  # noqa: E402
mpt.set_default(show_hints=False)   # dispatch prints would sit inside the timed closures
import warnings                                       # noqa: E402
from mpt._defaults import TruncationDefaultWarning    # noqa: E402
warnings.filterwarnings("ignore", category=TruncationDefaultWarning)
from mpt.tensor import _select_ma_inner_product_method      # noqa: E402
from bench_xlang import adaptive_time                       # noqa: E402


def build_abs(K, N, r, seed_shift=0):
    """Deterministic absolute density: K values per event, N events."""
    j = np.arange(K * N, dtype=float) + 1 + seed_shift
    vals = (2000.0 * np.mod(11.0 * j * j + 5.0 * j, 397.0) / 397.0
            ).reshape(K, N)
    return mpt.build_exp_tens([vals], None, [30.0], [r], [False],
                              [False], [0.0], verbose=False)


def selector(r, K_x, N_x, skip, K_y=None, N_y=None):
    if K_y is None:
        K_y = K_x
    if N_y is None:
        N_y = N_x
    chosen, pw, orbit = _select_ma_inner_product_method(
        r_vec=np.array([r]), k_vec=np.array([K_x]), A=1,
        N_x=N_x, N_y=N_y,
        any_per=False, any_rel_nonper=False, any_rel_per=False,
        sigma_over_P_max=0.0, user_method="auto",
        rel_vec=np.array([False]), k_vec_y=np.array([K_y]),
        return_costs=True,
        pw_skip_xx=skip, pw_skip_yy=skip,
        orbit_skip_xx=skip, orbit_skip_yy=skip,
    )
    return chosen, pw, orbit


def clear_memos(*densities):
    for d in densities:
        d._self_ip_cache.clear()
        if d._pruned_cached is not None and d._pruned_cached is not d:
            d._pruned_cached._self_ip_cache.clear()


def main():
    out_path = os.path.join(os.path.dirname(__file__),
                            "bench_cost_model_python.csv")
    rows = []
    flips = []

    # ---- Part 1: selector audit -----------------------------------
    print("== selector skip-flag audit (abs, A=1, N_x=N_y) ==")
    for r in (2, 3):
        for N in (4, 16):
            for K in (4, 6, 8, 12, 16, 24, 40):
                c_off, pw_off, orb_off = selector(r, K, N, skip=False)
                c_on, pw_on, orb_on = selector(r, K, N, skip=True)
                flip = c_off != c_on
                if flip:
                    flips.append((r, K, N, c_off, c_on))
                rows.append(dict(
                    language="python", part="selector", r=r, K=K, N=N,
                    regime="", chosen_off=c_off, chosen_on=c_on,
                    pw_off=f"{pw_off:.4g}", orbit_off=f"{orb_off:.4g}",
                    pw_on=f"{pw_on:.4g}", orbit_on=f"{orb_on:.4g}",
                    ms=""))
                mark = "  <-- flip" if flip else ""
                print(f"  r={r} K={K:3d} N={N:3d}: off={c_off:6s} "
                      f"on={c_on:6s}{mark}")

    # Asymmetric cells: the broadcast regime. A large shared X against
    # a small fixed query (K_y = 4, N_y = 3): the shared self matrix
    # dominates the full-triple Bulger price, so skipping it (a later
    # broadcast pair) is where the routing genuinely moves.
    print("\n== selector audit, asymmetric (K_y=4, N_y=3) ==")
    for r in (2, 3):
        for N_x in (8, 32):
            for K_x in (6, 8, 12, 16, 24, 40, 64):
                c_off, pw_off, orb_off = selector(r, K_x, N_x, False,
                                                  K_y=4, N_y=3)
                c_on, pw_on, orb_on = selector(r, K_x, N_x, True,
                                               K_y=4, N_y=3)
                flip = c_off != c_on
                if flip:
                    flips.append((r, K_x, N_x, c_off, c_on))
                rows.append(dict(
                    language="python", part="selector_asym", r=r,
                    K=K_x, N=N_x, regime="Ky4_Ny3",
                    chosen_off=c_off, chosen_on=c_on,
                    pw_off=f"{pw_off:.4g}", orbit_off=f"{orb_off:.4g}",
                    pw_on=f"{pw_on:.4g}", orbit_on=f"{orb_on:.4g}",
                    ms=""))
                mark = "  <-- flip" if flip else ""
                print(f"  r={r} Kx={K_x:3d} Nx={N_x:3d}: off={c_off:6s} "
                      f"on={c_on:6s}{mark}")

    # ---- Part 2: Möbius per-matrix timings ------------------------
    print("\n== Möbius per-matrix timings (abs, A=1, K=8) ==")
    for r in (2, 3):
        for N in (20, 60):
            d_x = build_abs(8, N, r)
            d_y = build_abs(8, N, r, seed_shift=137)

            def t_cos_fresh():
                clear_memos(d_x, d_y)
                return mpt.cos_sim_exp_tens(d_x, d_y, method="mobius",
                                            verbose=False)

            def t_osd_fresh():
                clear_memos(d_x, d_y)
                return mpt.cos_sim_exp_tens(
                    d_x, d_y, method="mobius",
                    normalize="oneSidedDenom", verbose=False)

            def t_cos_warm():
                # Memo persists across calls on the same objects: after
                # the first call, both self terms are cached and only
                # the cross matrix is computed.
                return mpt.cos_sim_exp_tens(d_x, d_y, method="mobius",
                                            verbose=False)

            regimes = (("cos_fresh_3mat", t_cos_fresh),
                       ("osd_fresh_2mat", t_osd_fresh),
                       ("cos_warm_1mat", t_cos_warm))
            times = {}
            for name, fn in regimes:
                t, _, _ = adaptive_time(fn)
                times[name] = t * 1000.0
                rows.append(dict(
                    language="python", part="mobius_matrices", r=r,
                    K=8, N=N, regime=name, chosen_off="", chosen_on="",
                    pw_off="", orbit_off="", pw_on="", orbit_on="",
                    ms=f"{t * 1000.0:.5f}"))
            t3, t2, t1 = (times["cos_fresh_3mat"],
                          times["osd_fresh_2mat"],
                          times["cos_warm_1mat"])
            print(f"  r={r} N={N:3d}: 3mat {t3:.3f}  2mat {t2:.3f}  "
                  f"1mat {t1:.3f} ms  ->  xy~{t1:.3f}  "
                  f"yy~{max(t2 - t1, 0):.3f}  xx~{max(t3 - t2, 0):.3f}")

    # ---- Part 3: flip-cell behavioural check ----------------------
    print("\n== flip-cell forced-route timings (warm selves) ==")
    for (r, K, N, c_off, c_on) in flips[:4]:
        d_x = build_abs(K, N, r)
        d_y = build_abs(4, 3, r, seed_shift=137)
        forced = {}
        for meth in ("bulger", "mobius"):
            mpt.cos_sim_exp_tens(d_x, d_y, method=meth, verbose=False)

            def fn(meth=meth):
                return mpt.cos_sim_exp_tens(d_x, d_y, method=meth,
                                            verbose=False)
            t, _, _ = adaptive_time(fn)
            forced[meth] = t * 1000.0
            rows.append(dict(
                language="python", part="flip_check", r=r, K=K, N=N,
                regime=f"forced_{meth}_warm", chosen_off=c_off,
                chosen_on=c_on, pw_off="", orbit_off="", pw_on="",
                orbit_on="", ms=f"{t * 1000.0:.5f}"))
        faster = min(forced, key=forced.get)
        ok = "OK" if faster == c_on else "MISPICK"
        print(f"  r={r} K={K} N={N}: off->{c_off}, on->{c_on}; "
              f"warm bulger {forced['bulger']:.3f} ms, "
              f"mobius {forced['mobius']:.3f} ms; faster={faster} [{ok}]")

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

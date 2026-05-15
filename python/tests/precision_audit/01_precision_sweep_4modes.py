"""Smaller-scope precision sweep — A=1, N=2, fewer seeds, but full mode coverage."""
import numpy as np
from mpt.tensor import (
    build_exp_tens, _cos_sim_exp_tens_ma_orbit, _cos_sim_exp_tens_ma_pairwise,
)
import time
import sys


def precision_cell(seed, r, K, sigma=50.0, is_rel=False, is_per=False, period=0.0,
                    A=1, N=2):
    rng = np.random.default_rng(seed)
    p_x = [rng.uniform(0, 1200, size=(K, N)) for _ in range(A)]
    p_y = [rng.uniform(0, 1200, size=(K, N)) for _ in range(A)]
    w_x = [rng.uniform(0.1, 1.0, size=(K, N)) for _ in range(A)]
    w_y = [rng.uniform(0.1, 1.0, size=(K, N)) for _ in range(A)]
    sigma_vec = [sigma] * A
    r_vec = [r] * A
    groups = list(range(A))
    is_rel_vec = [is_rel] * A
    is_per_vec = [is_per] * A
    period_vec = [period] * A
    dx = build_exp_tens(p_x, w_x, sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec, verbose=False)
    dy = build_exp_tens(p_y, w_y, sigma_vec, r_vec, groups, is_rel_vec, is_per_vec, period_vec, verbose=False)
    o_xy, o_xx, o_yy = _cos_sim_exp_tens_ma_orbit(dx, dy)
    p_xy, p_xx, p_yy = _cos_sim_exp_tens_ma_pairwise(dx, dy, verbose=False)
    cos_o = o_xy / np.sqrt(max(o_xx * o_yy, 1e-300))
    cos_p = p_xy / np.sqrt(max(p_xx * p_yy, 1e-300))
    return abs(cos_o - cos_p) / max(abs(cos_p), 1e-300)


modes = [
    ('abs_nonper', False, False, 0.0),
    ('abs_per',    False, True,  1200.0),
    ('rel_per',    True,  True,  1200.0),
    ('rel_nonper', True,  False, 0.0),
]

print(f"{'mode':<12} {'r':>2} {'K':>2} | {'med':>9} {'p90':>9} {'max':>9}")
t0 = time.time()
for mode_name, is_rel, is_per, period in modes:
    for r in [2, 3, 4, 5]:
        for K in [r, r+1, r+2, r+4]:
            if time.time() - t0 > 500:
                print("TIMEOUT GUARD HIT")
                sys.exit()
            rels = []
            ok = True
            for seed in range(5):
                try:
                    rels.append(precision_cell(
                        seed, r, K, is_rel=is_rel, is_per=is_per, period=period))
                except Exception as ex:
                    ok = False
                    print(f"{mode_name:<12} {r:>2} {K:>2} | EXCEPTION: {type(ex).__name__}: {ex}")
                    break
            if ok and rels:
                mx = np.max(rels)
                tag = '   '
                if mx > 1e-3: tag = '!! '
                elif mx > 1e-10: tag = 'BAD'
                p50 = np.median(rels)
                p90 = np.quantile(rels, 0.9)
                print(f"{mode_name:<12} {r:>2} {K:>2} | {p50:>9.2e} {p90:>9.2e} {mx:>9.2e} {tag}")
            sys.stdout.flush()
        print()

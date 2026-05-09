"""demo_sigma_space.py

Demonstration of the v2.1.0 sigma + sigma_space additions to
sameness, coherence, and n_tuple_entropy.

These three functions previously had no soft (sigma > 0) versions
in v2.0; v2.1 introduces a Gaussian-kernel-based soft path with a
flag controlling how sigma is interpreted:

  sigma_space = 'position' (default)
     sigma is positional uncertainty on each input value p_k.
     Cross-event correlations between intervals derived from a
     shared position set are captured exactly via per-pair
     variance bookkeeping.

  sigma_space = 'interval'
     sigma is independent uncertainty per derived interval.
     Slots are treated as independent draws (V = 2 sigma**2
     uniformly).

The two flags coincide at sigma = 0; only at sigma > 0 does the
distinction matter.
"""
import warnings

import numpy as np

from mpt import coherence, n_tuple_entropy, sameness
from mpt._utils import position_variance


DIATONIC = [0, 2, 4, 5, 7, 9, 11]
PERIOD = 12
SIGMAS = [0, 0.1, 0.25, 0.5, 1.0, 2.0]


# ===== 1. Sameness =====

print("\n=== Sameness on the diatonic scale [0,2,4,5,7,9,11] in 12-EDO ===\n")
print("  At sigma = 0 the v2.0 hard count is recovered exactly:")
print("    one ambiguity (the tritone, size 6 as both 4th and 5th).\n")

print(f"  {'sigma':<10} {'sq (position)':<15} {'sq (interval)':<15}")
print("  " + "-" * 42)
for s in SIGMAS:
    if s == 0:
        sq, _ = sameness(DIATONIC, PERIOD, 0)
        sq_p = sq_i = sq
    else:
        sq_p, _ = sameness(DIATONIC, PERIOD, s, sigma_space="position")
        sq_i, _ = sameness(DIATONIC, PERIOD, s, sigma_space="interval")
    print(f"  {s:<10g} {sq_p:<15.4f} {sq_i:<15.4f}")

print()
print("  Position is more aggressive (lower sq) at typical sigma")
print("  because its per-pair variance for disjoint-endpoint pairs")
print("  is 4*sigma^2 — wider than interval's uniform 2*sigma^2,")
print("  so the soft-match kernel is broader.")


# ===== 2. Coherence =====

print("\n=== Coherence on the diatonic scale ===\n")
print("  v2.0 strict coherence: 1 failure (the tritone), c = 1 - 1/140.")
print("  Soft sigma -> 0+ limit:   tritone counts as 0.5 of a failure")
print("  (the means D2 = D1 give P(D2 <= D1) = 0.5 exactly under any")
print(f"  sigma > 0), so c -> 1 - 0.5/140 = {1 - 0.5/140:.4f}.\n")

print(f"  {'sigma':<10} {'c (position)':<15} {'c (interval)':<15}")
print("  " + "-" * 42)
for s in SIGMAS:
    if s == 0:
        c, _ = coherence(DIATONIC, PERIOD, 0)
        c_p = c_i = c
    else:
        c_p, _ = coherence(DIATONIC, PERIOD, s, sigma_space="position")
        c_i, _ = coherence(DIATONIC, PERIOD, s, sigma_space="interval")
    print(f"  {s:<10g} {c_p:<15.4f} {c_i:<15.4f}")

print()
print("  The discontinuity at sigma = 0 (0.9929 to 0.9964 between")
print("  the v2.0 hard count and the soft path's sigma -> 0+ limit)")
print("  is intentional: the strict flag splits ties as you choose at")
print("  sigma exactly zero, while the soft path averages over them.")


# ===== 3. The tritone diagnostic =====

print("\n=== The tritone tie under positional jitter ===\n")
print("  In the diatonic scale, the fourth F->B (positions 5->11)")
print("  and the fifth B->F (positions 11->5, wrapping) share both")
print("  endpoints. Under positional jitter on F and B, the")
print("  difference D2 - D1 has variance 8*sigma^2 (not 4*sigma^2:")
print("  the shared positions contribute with reinforcing signs).\n")
print("  Both intervals have specific size = 6, so D2 - D1 has")
print("  mean 0. P(D2 <= D1) = Phi(0) = 0.5, regardless of sigma.\n")

V = position_variance([4, 7, 7, 4], [+1, -1, -1, +1], 1.0)
print(f"  position_variance for the tritone pair (sigma=1): V = {V:g}")
print("  (matches expected 8 for the shared-reinforcing case)")


# ===== 4. n_tuple_entropy =====

print("\n=== n_tuple_entropy on the diatonic ===\n")
print("  At n = 1, sigma_space = 'position' is exactly equivalent")
print("  to sigma_space = 'interval' with sigma multiplied by")
print("  sqrt(2) — the marginal-matched relationship that captures")
print("  the variance that one positional jitter induces in")
print("  the derived step sizes.\n")

print("  n = 1:")
print(f"  {'sigma':<10} {'H (position)':<18} {'H (interval)':<18}")
print("  " + "-" * 48)
for s in [0, 0.1, 0.25, 0.5]:
    if s == 0:
        H, _ = n_tuple_entropy(DIATONIC, PERIOD, 1)
        H_p = H_i = H
    else:
        H_p, _ = n_tuple_entropy(
            DIATONIC, PERIOD, 1, sigma=s, sigma_space="position"
        )
        H_i, _ = n_tuple_entropy(
            DIATONIC, PERIOD, 1, sigma=s, sigma_space="interval"
        )
    print(f"  {s:<10g} {H_p:<18.6f} {H_i:<18.6f}")

print("\n  Verifying the n=1 exactness relationship:")
sigma_p = 0.5
H_p, _ = n_tuple_entropy(
    DIATONIC, PERIOD, 1, sigma=sigma_p, sigma_space="position"
)
H_i_match, _ = n_tuple_entropy(
    DIATONIC, PERIOD, 1,
    sigma=sigma_p * np.sqrt(2), sigma_space="interval",
)
print(f"    H(sigma={sigma_p:.4f}, position) = {H_p:.10f}")
print(f"    H(sigma={sigma_p * np.sqrt(2):.4f}, interval) = {H_i_match:.10f}")
print(f"    Difference: {abs(H_p - H_i_match):.2e} (should be ~zero)")

print("\n  At n >= 2, sigma_space = 'position' falls back to a")
print("  marginal-matched approximation that ignores anti-")
print("  correlations between adjacent step slots. A warning")
print("  fires once per call to make this explicit:\n")

with warnings.catch_warnings():
    warnings.simplefilter("always")
    H_n2_p, _ = n_tuple_entropy(
        DIATONIC, PERIOD, 2, sigma=0.3, sigma_space="position"
    )
print(f"    H(diatonic, n=2, sigma=0.3, position) = {H_n2_p:.6f}")

print("\nDone.")

"""demo_bindEvents.py

Demonstrate ``bind_events``: sliding-window binding into n-attribute
super-events. Combined with ``difference_events``, this generalises
the n-tuple entropy of Milne & Dean (2016) to non-zero sigma,
continuous values, non-periodic domains, and per-event weights, and
exposes the underlying *density* of n-tuples (not just its scalar
entropy) for use with the rest of the toolbox.

Sections
--------
1. n-tuple entropy via difference_events + bind_events (vs nTupleEntropy).
2. Smoothed n-tuple entropy: sigma > 0 generalisation.
3. n-tuple density similarity: cosine similarity between scales' n-gram
   distributions, which the scalar nTupleEntropy cannot do.
4. Mixed pipeline: melodic n-grams from a pitch sequence.
"""

import numpy as np
from mpt import (
    n_tuple_entropy,
    entropy_exp_tens,
    build_exp_tens,
    difference_events,
    bind_events,
    cos_sim_exp_tens,
)

np.set_printoptions(precision=4, suppress=True)


def cyclic_steps(p, period):
    """Cyclic step sizes of a scale: K events -> K differences."""
    p_sorted = np.sort(np.mod(p, period))
    return np.mod(np.diff(np.concatenate([p_sorted, [p_sorted[0] + period]])),
                  period)


def n_tuple_entropy_via_bind(p, period, n, sigma=1e-6):
    """Compute n-tuple entropy through the difference + bind pipeline.

    For sigma -> 0 with grid = period, this matches Milne & Dean (2016).
    For sigma > 0, this matches the smoothed extension in Milne (2024).
    """
    diffs = cyclic_steps(p, period).astype(float).reshape(1, -1)
    p_bound, w_bound = bind_events(diffs, None, n, circular=True)
    T = build_exp_tens(
        p_bound, w_bound,
        [sigma],          # one sigma for the single group
        [1] * n,          # r = 1 per attribute
        [0] * n,          # all n attributes in group 0
        [False],          # not relative
        [True],           # periodic
        [period],
        verbose=False,
    )
    return entropy_exp_tens(T, normalize=False, base=2,
                             n_points_per_dim=period)


# ===================================================================
#  1. n-tuple entropy: bind+MAET vs nTupleEntropy
# ===================================================================

print("=== 1. n-tuple entropy via difference_events + bind_events ===")
print("    (matches nTupleEntropy / Milne & Dean 2016 for sigma -> 0)\n")

# Diatonic scale, 12-EDO
p = np.array([0, 2, 4, 5, 7, 9, 11])
P = 12
print(f"    Diatonic scale: {p.tolist()}, period = {P}")
print(f"    Cyclic step sizes: {cyclic_steps(p, P).astype(int).tolist()}\n")

print(f"    {'n':>3}  {'bind+MAET':>12}  {'nTupleEntropy':>15}  {'difference':>11}")
for n in [1, 2, 3]:
    H_bind = n_tuple_entropy_via_bind(p, P, n, sigma=1e-6)
    H_ref, _ = n_tuple_entropy(p, P, n=n, sigma=0.0,
                                normalize=False, base=2)
    print(f"    {n:>3}  {H_bind:>12.6f}  {H_ref:>15.6f}  {H_bind-H_ref:>+11.2e}")


# ===================================================================
#  2. Smoothed extension: sigma > 0
# ===================================================================

print("\n=== 2. Smoothed n-tuple entropy (sigma > 0) ===")
print("    Bind+MAET pipeline extends naturally to non-zero sigma.\n")

print(f"    {'n':>3}  {'sigma':>6}  {'bind+MAET':>12}  {'nTupleEntropy':>15}")
for n in [1, 2, 3]:
    for sigma in [0.5, 1.0]:
        H_bind = n_tuple_entropy_via_bind(p, P, n, sigma=sigma)
        H_ref, _ = n_tuple_entropy(p, P, n=n, sigma=sigma,
                                    normalize=False, base=2)
        print(f"    {n:>3}  {sigma:>6}  {H_bind:>12.6f}  {H_ref:>15.6f}")


# ===================================================================
#  3. Cosine similarity between n-tuple distributions of two scales
# ===================================================================

print("\n=== 3. Cosine similarity between scales' n-tuple distributions ===")
print("    Goes beyond the scalar nTupleEntropy: the bound MAET is a")
print("    full density that can be compared to other densities.\n")

scales = {
    "Major":               np.array([0, 2, 4, 5, 7, 9, 11]),
    "Phrygian":            np.array([0, 1, 3, 5, 7, 8, 10]),
    "Melodic minor (asc)": np.array([0, 2, 3, 5, 7, 9, 11]),
    "Harmonic minor":      np.array([0, 2, 3, 5, 7, 8, 11]),
}

for name, s in scales.items():
    print(f"    {name:24s} step sequence: {cyclic_steps(s, P).astype(int).tolist()}")
print()


def make_T(p, period, n, sigma):
    diffs = cyclic_steps(p, period).astype(float).reshape(1, -1)
    pB, wB = bind_events(diffs, None, n, circular=True)
    return build_exp_tens(pB, wB, [sigma], [1] * n, [0] * n,
                           [False], [True], [period], verbose=False)


sigma = 0.6
print(f"    Cosine similarity at sigma = {sigma}:")
print(f"    {'':24s}  ", end="")
for n in [2, 3]:
    print(f"  n = {n}     ", end="")
print()

names = list(scales.keys())
for i, ni in enumerate(names):
    for nj in names[i+1:]:
        row = f"    sim({ni:6s} | {nj:18s})"
        for n in [2, 3]:
            Ti = make_T(scales[ni], P, n, sigma)
            Tj = make_T(scales[nj], P, n, sigma)
            s = cos_sim_exp_tens(Ti, Tj, verbose=False)
            row += f"  {s:.4f}     "
        print(row)

print("\n    Note: at n = 2 the major-mode rotations all share a step-pair")
print("    multiset, so they are indistinguishable. At n = 3 they begin to")
print("    separate (e.g. melodic and harmonic minor differ from major).\n")


# ===================================================================
#  4. Melodic n-grams: bind on a pitch sequence directly
# ===================================================================

print("=== 4. Melodic n-grams (binding raw pitches, no differencing) ===")
print("    Skipping the difference step gives n-grams of pitches in their")
print("    actual register, rather than transposition-invariant intervals.\n")

# A short melodic phrase
melody = np.array([67, 69, 71, 72, 71, 69, 67, 65, 67],
                  dtype=float).reshape(1, -1)   # (1, N)
print(f"    Phrase (MIDI): {melody.ravel().astype(int).tolist()}")

# Bind into 3-grams (no circular wrap)
pB, wB = bind_events(melody, None, 3, circular=False)
N_prime = pB[0].shape[1]
print(f"    bind_events(n=3, circular=False) -> {len(pB)} attributes "
      f"of shape (1, {N_prime}):")
for j, M in enumerate(pB):
    print(f"      lag {j}: {M.ravel().astype(int).tolist()}")

print()
print("    These three attributes, placed in one group with r = 1 each,")
print("    form a 3-D MAET over (pitch_t, pitch_{t+1}, pitch_{t+2}).")
print("    The resulting density characterises the melodic 3-gram structure")
print("    of the phrase, in absolute pitch register.")

"""demo_rhythm_tensors.py

Expectation tensors of rhythms: the analyses the other demos apply to
pitch, applied to time.

A rhythm here is a multiset of onsets, one element per onset, in a
cycle of 16 pulses (the equally spaced time points of the cycle), with
time periodic at the cycle length: cycle equivalence, the counterpart
of octave equivalence. Only the period changes; every function is the
one used for pitch. The running example is the son clave
[0, 3, 6, 10, 12], compared with five other well-known 16-pulse
timelines. Sigma is 0.5 pulses, except in the limit taken in 4a.

Sections:
  1. Similarity: absolute (r = 1) against relative (r = 2)
  2. Density: the r = 1 and r = 2 relative densities (figure)
  3. Complexity: entropy of the r = 2 relative tensor
  4. Circular differencing and binding, and n-tuple entropy

Uses: sim_maet, build_maet, plot_maet, entropy_maet, flat_specs,
      pack_pre_maet, difference_events, bind_events, n_tuple_entropy,
      set_default.

The MATLAB mirror is demo_rhythmTensors.m.
"""

import numpy as np
import matplotlib.pyplot as plt

import mpt

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Informational hints are switched off, and restored at the end.
prev_defaults = mpt.set_default(show_hints=False)

PERIOD = 16
SIGMA = 0.5
son = [0, 3, 6, 10, 12]
names = ["son clave 2-3", "rumba clave", "bossa nova", "gahu", "shiko",
         "soukous"]
others = np.array([
    [2, 4, 8, 11, 14],     # the son clave with its two halves swapped
    [0, 3, 7, 10, 12],
    [0, 3, 6, 10, 13],
    [0, 3, 6, 10, 14],
    [0, 4, 6, 10, 12],
    [0, 3, 6, 10, 11],
], dtype=float)
son_f = np.array(son, dtype=float)

# ===================================================================
#  1. Similarity: absolute (r = 1) against relative (r = 2)
# ===================================================================

# At r = 1, absolute, the density records where in the cycle the onsets
# fall, so similarity depends on phase. At r = 2, relative, it records
# the interval between every pair of onsets (an inter-onset interval,
# IOI, taken between any two onsets, not only successive ones): the
# rhythmic counterpart of the interval vector, unchanged when the
# rhythm is rotated (every onset shifted by the same number of pulses).
print("\n=== 1. Similarity to the son clave (sigma = 0.5 pulses) ===")
s_abs = mpt.sim_maet(son_f, None, others, None, SIGMA, 1, False, True,
                     PERIOD, verbose=False)
s_rel = mpt.sim_maet(son_f, None, others, None, SIGMA, 2, True, True,
                     PERIOD, verbose=False)
print(f"  {'':15s} r = 1 absolute   r = 2 relative")
for name, a, b in zip(names, s_abs, s_rel):
    print(f"  {name:15s} {a:10.3f} {b:16.3f}")

rotations = (son_f + np.arange(PERIOD)[:, None]) % PERIOD
s_rot = mpt.sim_maet(son_f, None, rotations, None, SIGMA, 1, False, True,
                     PERIOD, verbose=False)
print(f"  r = 1, against its rotations by 0-15 pulses:\n    {s_rot}")
# The 2-3 son clave is the rotation by 8 pulses: absolute similarity
# 0.314, the lowest of the 16 rotations, relative 1. Each of the other
# five timelines is the son clave with one onset displaced, so they
# share most of its IOIs (relative 0.955-0.984); absolute similarity
# separates them more.

# ===================================================================
#  2. Density: the r = 1 and r = 2 relative densities
# ===================================================================

# Evaluated at a point, the r = 1 density says how strongly an onset is
# expected there; the r = 2 relative density, how many pairs of onsets
# lie that far apart. The latter is symmetric about 8 pulses, since an
# IOI of d pulses read backwards around the cycle is one of 16 - d.
print("\n=== 2. Densities of the son clave (figure) ===")
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))
for ax, (r, rel, label) in zip(axes, [(1, False, "onset (pulse)"),
                                         (2, True, "IOI (pulses)")]):
    dens = mpt.build_maet(son_f, None, SIGMA, r, rel, True, PERIOD,
                          verbose=False)
    mpt.plot_maet(dens, method='density', ax=ax)
    ax.set_xticks(np.arange(PERIOD + 1))
    ax.set_xlabel(label)
    ax.set_ylabel("density")
axes[0].set_title("Son clave: r = 1, absolute")
axes[1].set_title("Son clave: r = 2, relative")
fig.tight_layout()
print("  Drawn: r = 1 absolute and r = 2 relative.")

# ===================================================================
#  3. Complexity: entropy of the r = 2 relative tensor
# ===================================================================

# The Renyi-2 entropy (minus the logarithm of the integral of the
# squared density, here in bits, computed in closed form) of the r = 2
# relative density is high when a rhythm's IOIs take many different
# sizes in similar numbers, and low when few sizes recur.
print("\n=== 3. IOI-content entropy (Renyi-2, bits) ===")
rhythms = {
    "four on the floor": [0, 4, 8, 12],
    "bossa nova":        [0, 3, 6, 10, 13],
    "son clave":         son,
    "shiko":             [0, 4, 6, 10, 12],
    "clustered":         [0, 1, 2, 3, 9],
}
for name, rh in rhythms.items():
    H = mpt.entropy_maet(np.asarray(rh, dtype=float), None, SIGMA, 2, True,
                         True, PERIOD, method='renyi2', verbose=False)
    print(f"  {name:17s}: {H:.3f}")
# Four evenly spaced onsets have IOIs of only 4, 8, and 12 pulses, so
# the lowest entropy. Among the five-onset rhythms, the bossa nova, the
# most evenly spread, repeats IOIs most and scores lowest; the clustered
# rhythm is irregular, yet its IOIs crowd into two groups of
# neighbouring sizes (1-3 and 6-8 pulses), which the smoothing partly
# merges, so it scores below the son clave and the shiko. The measure
# reads the variety of IOIs, not irregularity as such.

# ===================================================================
#  4. Circular differencing and binding, and n-tuple entropy
# ===================================================================

# n_tuple_entropy (Milne & Dean, 2016) is the entropy of the n-tuples
# of successive IOIs, the cycle wrapping from the last onset to the
# first. In a pre-MAET each onset is an event holding one element.
# Circular differencing (difference_events, circular = True) replaces
# each onset by the IOI from the onset before it, and circular binding
# (bind_events) nests n successive IOIs into one super-event, whose
# MAET's entropy is n-tuple entropy.
print("\n=== 4. Differencing and binding against n_tuple_entropy ===")


def rhythm_pm(onsets, sigma=None):
    """A rhythm as a pre-MAET: one onset per event."""
    p = [np.asarray(onsets, dtype=float)[None, :]]
    specs = mpt.flat_specs(p, names=['onset'], sigma=sigma,
                           per=[True], period=[PERIOD])
    return mpt.pack_pre_maet(p, None, specs)


# 4a. The sigma -> 0 limit, on the grid of 16 integer pulses: the
# Shannon entropy of the IOI n-tuple histogram, normalized to [0, 1].
# entropy_maet needs sigma > 0, so a vanishing width of 1e-12 stands in
# for 0, as inside n_tuple_entropy.
print("  4a. sigma -> 0, normalized entropy on the 16-pulse grid")
steps = mpt.difference_events(rhythm_pm(son), 1, circular=True)
for n in (1, 2):
    pm_n = mpt.bind_events(steps, n, circular=True)
    dens = mpt.build_maet(pm_n, sigma=[1e-12], verbose=False)
    H_pipe = mpt.entropy_maet(dens, method='normalized',
                              n_points_per_dim=PERIOD, verbose=False)
    H_nte, _ = mpt.n_tuple_entropy(son, PERIOD, n, verbose=False)
    print(f"    n = {n}: pipeline {H_pipe:.12f}, "
          f"n_tuple_entropy {H_nte:.12f}")

# 4b. sigma = 0.5 pulses on each onset, Renyi-2 entropy. A difference of
# two onsets each of width sigma has width sigma * sqrt(2), and
# difference_events rescales the IOIs' sigma accordingly (it announces
# this). Differencing then binding treats successive IOIs as
# independent, which is n_tuple_entropy's sigma_space = 'interval' at
# the rescaled width. Its default, sigma_space = 'position', keeps the
# dependence: successive IOIs share an onset, so each pair covaries by
# -sigma^2. Binding n + 1 onsets and taking the super-event relative
# (rel_outer = True), in place of differencing, captures it exactly.
print("  4b. sigma = 0.5 on each onset, Renyi-2 entropy (bits)")
steps = mpt.difference_events(rhythm_pm(son, [SIGMA]), 1, circular=True)
print(f"    IOI sigma after differencing: {steps['specs'][0]['sigma']:.4f}")
for n in (1, 2):
    H_db = mpt.entropy_maet(mpt.bind_events(steps, n, circular=True),
                            method='renyi2', verbose=False)
    H_br = mpt.entropy_maet(mpt.bind_events(rhythm_pm(son, [SIGMA]), n + 1,
                                            circular=True, rel_outer=True),
                            method='renyi2', verbose=False)
    H_int, _ = mpt.n_tuple_entropy(son, PERIOD, n, sigma=SIGMA * np.sqrt(2),
                                   sigma_space='interval', method='renyi2',
                                   verbose=False)
    H_pos, _ = mpt.n_tuple_entropy(son, PERIOD, n, sigma=SIGMA,
                                   method='renyi2', verbose=False)
    print(f"    n = {n}: difference-bind {H_db:.6f} = 'interval' "
          f"{H_int:.6f};  bind-relative {H_br:.6f} = 'position' "
          f"{H_pos:.6f}")
# Each route equals its n_tuple_entropy mode to machine precision. At
# n = 1 there is no neighbouring IOI to covary with, so the two models
# agree; at n = 2 they differ, so "n-tuple entropy" at sigma > 0 needs
# its sigma_space stated. At sigma -> 0 (4a) both reduce to the
# histogram.

# 4c. The same pre-MAET goes where n_tuple_entropy cannot. A second
# pre-MAET is a query, and sim_maet compares the two rhythms' densities
# of successive-IOI pairs: rotation-invariant, like r = 2 relative in
# section 1, but sensitive to the order of the IOIs. (The pre-MAET
# would equally carry per-onset weights, such as accents, for which
# n_tuple_entropy has no argument.)
print("  4c. Similarity of successive-IOI pairs (bind-relative, n = 2)")
ctx = mpt.bind_events(rhythm_pm(son, [SIGMA]), 3, circular=True,
                      rel_outer=True)
for name, rh in zip(names[1:], others[1:]):
    q = mpt.bind_events(rhythm_pm(rh, [SIGMA]), 3, circular=True,
                        rel_outer=True)
    print(f"    son clave vs {name:12s}: "
          f"{float(mpt.sim_maet(ctx, q, verbose=False)):.3f}")
# Each value is lower than its r = 2 relative counterpart in section 1:
# the IOI content of all pairs of onsets discards which IOIs are
# successive and in what order, and that order separates the rhythms
# further.

mpt.set_default(**prev_defaults)
print("\nDone.")
plt.show()

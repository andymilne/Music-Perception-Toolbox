"""demo_translate_sweep.py

Pre-tensor sliding-comparison sweep with ``translate_attributes`` and the
raw-MA list mode of ``cos_sim_exp_tens``.

Scenario: a 3-note motif (C E G) hidden inside a 7-note melody
(D E F C E G A, one note per second). The motif appears exactly at
reference times 3, 4, 5. At each (pitch transposition, time shift)
offset, the query is translated and compared to the un-shifted
reference with the closed-form cosine similarity of a 2-attribute
MAET (pitch periodic at the octave; time absolute non-periodic). The
sweep should peak at (0 cents, 3 s) where the query aligns with the
embedded C-E-G, and at (1200 cents, 3 s) by octave periodicity.

The workflow is two function calls: one to ``translate_attributes``, one
to ``cos_sim_exp_tens`` (raw-MA scalar-vs-list form, with the
translated ``p_attr`` list as one operand and the reference
``p_attr`` as the other). The build step is internalised: the
reference is built once, each translated query once. The sweep can
be specified in either of two equivalent forms --- a single ``(A, M)``
numeric matrix or a ``{group_index: sweep}`` dict --- and Section 3
shows both with a parity check.

Compare ``demo_maet_windowing`` (post-tensor sliding) and
``demo_windowing_reference`` (reference-point options for
``windowed_similarity``). The pre-tensor route used here returns a
strict cosine similarity (bounded in [0, 1] for non-negative weights)
and does not require choosing a window family; the post-tensor route
returns a magnitude-aware windowed similarity and decouples locality
from the query's own support.

See also
--------
mpt.translate_attributes
mpt.cos_sim_exp_tens
mpt.build_exp_tens
mpt.windowed_similarity
"""

import numpy as np
import matplotlib.pyplot as plt

import mpt


# =====================================================================
# 1. Build the reference melody and the query motif (pre-tensor form)
# =====================================================================

print("=== 1. Pre-tensor inputs ===")

# Reference: D-E-F-C-E-G-A at one note per second. The query C-E-G
# appears exactly at times 3, 4, 5.
ref_midi  = np.array([62, 64, 65, 60, 64, 67, 69])
ref_pitch = mpt.transform_attributes(ref_midi, None, ('midi', 'cents')).reshape(1, -1)
ref_time  = np.arange(7, dtype=float).reshape(1, -1)
ref_pAttr = [ref_pitch, ref_time]

# Query: C-E-G triad, 1-second spacing.
qry_midi  = np.array([60, 64, 67])
qry_pitch = mpt.transform_attributes(qry_midi, None, ('midi', 'cents')).reshape(1, -1)
qry_time  = np.arange(3, dtype=float).reshape(1, -1)
qry_pAttr = [qry_pitch, qry_time]

# Per-group geometry. Two attributes -> two groups (pitch in group 0,
# time in group 1). Pitch is periodic at the octave; time is absolute
# non-periodic.
sigma   = [50.0, 0.3]
r       = [1, 1]
groups  = [0, 1]
is_rel  = [False, False]
is_per  = [True,  False]
periods = [1200.0, 0.0]

print("  reference: D-E-F-C-E-G-A, one note per second")
print("  query    : C-E-G triad, 1-second spacing")
print("  (the motif appears exactly at reference times 3, 4, 5)")
print(f"  sigma    : {sigma[0]:.0f} cents (pitch) / {sigma[1]:.2f} s (time)")
print()


# =====================================================================
# 2. Construct the (pitch, time) offset sweep
# =====================================================================

print("=== 2. Offset sweep grid ===")

# Pitch offsets: 0-1200 cents in 100-cent steps (one octave). The
# expected peak at pitch shift 0 is also visible at 1200 cents because
# the pitch group is octave-periodic.
pitch_grid = np.arange(0, 1201, 100, dtype=float)
# Time offsets: -1 to 5 seconds in 0.25-second steps.
time_grid  = np.arange(-1.0, 5.001, 0.25)

P_mesh, T_mesh = np.meshgrid(pitch_grid, time_grid, indexing="ij")
M = P_mesh.size
# P_mesh and T_mesh are used in Section 3 to build the sweep in either
# of the two equivalent offset forms.

print(f"  pitch grid: {pitch_grid.size} transpositions over one octave "
      f"(100-cent steps)")
print(f"  time  grid: {time_grid.size} positions from t = "
      f"{time_grid.min():.1f} to t = {time_grid.max():.1f} s")
print(f"  total sweep positions: M = {M}")
print()


# =====================================================================
# 3. Pre-tensor translation: two equivalent offset forms
# =====================================================================

print("=== 3. translate_attributes (two equivalent offset forms) ===")

# Form A: numeric matrix. Rows index attributes, columns index sweep
# positions. With A = 2 singleton groups here, row 0 is the pitch
# attribute and row 1 is the time attribute.
offsets_mat = np.vstack([P_mesh.ravel(),     # pitch shifts (attribute 0)
                         T_mesh.ravel()])    # time  shifts (attribute 1)

qry_pAttr_swept_mat = mpt.translate_attributes(
    qry_pAttr, groups, offsets_mat, is_rel, is_per, periods,
)

# Form B: dict keyed by group index, with one group's sweep per entry.
# Each value is a 1-D length-M row, which the orientation grammar
# reads as "broadcast within group, M-position sweep" --- here that
# coincides with per-attribute because each group is a singleton.
# Reads naturally as "sweep pitch (group 0) by these values; sweep
# time (group 1) by these values".
offsets_dict = {
    0: P_mesh.ravel(),   # pitch axis
    1: T_mesh.ravel(),   # time axis
}

qry_pAttr_swept_dict = mpt.translate_attributes(
    qry_pAttr, groups, offsets_dict, is_rel, is_per, periods,
)

# Parity check: the two forms must produce identical translated values.
diff_max_forms = 0.0
for entry_m, entry_d in zip(qry_pAttr_swept_mat, qry_pAttr_swept_dict):
    for a, b in zip(entry_m, entry_d):
        diff_max_forms = max(diff_max_forms, float(np.max(np.abs(a - b))))
print(f"  matrix form vs dict form: max |diff| = {diff_max_forms:.2e}")
assert diff_max_forms == 0.0, "Matrix form and dict form disagree."

# Proceed with the matrix-form output for the downstream computation.
qry_pAttr_swept = qry_pAttr_swept_mat
print(f"  qry_pAttr_swept: {type(qry_pAttr_swept).__name__}, "
      f"length {len(qry_pAttr_swept)}")
print(f"  each entry is a length-{len(qry_pAttr)} list of K_a x N "
      f"value matrices")
print()


# =====================================================================
# 4. Raw-MA scalar-vs-list cosine similarity: one call
# =====================================================================

print("=== 4. cos_sim_exp_tens (raw-MA list mode) ===")

S_flat = mpt.cos_sim_exp_tens(
    ref_pAttr, None, qry_pAttr_swept, None,
    sigma, r, groups, is_rel, is_per, periods,
    verbose=False,
)
S = S_flat.reshape(P_mesh.shape)         # (pitch, time) heatmap

i, j = np.unravel_index(int(np.argmax(S)), S.shape)
print(f"  cosine similarity surface: shape {S.shape} (pitch x time)")
print(f"  max similarity {S.max():.4f} at pitch shift "
      f"{P_mesh[i, j]:.0f} c, time shift {T_mesh[i, j]:.2f} s")
print("  (expected: 0 cents, 3.00 s --- the embedded C-E-G)")
print()


# =====================================================================
# 5. Visualise the sweep
# =====================================================================

print("=== 5. Plot ===")

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(
    S.T,                     # transpose: x-axis pitch, y-axis time
    origin="lower",
    aspect="auto",
    extent=[pitch_grid[0], pitch_grid[-1],
            time_grid[0], time_grid[-1]],
)
fig.colorbar(im, ax=ax, label="cosine similarity")
ax.set_xlabel("Pitch transposition (cents)")
ax.set_ylabel("Time shift (s)")
ax.set_title("Pre-tensor sliding-comparison: cosine similarity\n"
             "Reference: D-E-F-C-E-G-A; query: C-E-G")
ax.set_xticks(np.arange(0, 1201, 200))
ax.set_yticks(np.arange(-1, 5.01, 1))

# Mark the expected peak positions: query aligns with the embedded
# C-E-G at (0 c, 3 s). Octave periodicity reproduces the peak at
# (1200 c, 3 s).
ax.plot([0, 1200], [3, 3], "rx", markersize=12, mew=1.5)
ax.text(40,   3.4, "C-E-G match", color="r",
        fontsize=9, bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none"))
ax.text(1100, 3.4, "octave", color="r",
        fontsize=9, bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none"))

plt.tight_layout()
print("  Figure shows the cosine-similarity surface as a function of")
print("  pitch transposition and time shift. Red x marks the")
print("  expected peaks at (0 c, 3 s) and (1200 c, 3 s), where the")
print("  query aligns with the embedded C-E-G in the reference;")
print("  octave-pitch periodicity makes the two peaks identical.")
print()


# =====================================================================
# 6. Equivalent explicit build loop, for transparency
# =====================================================================

print("=== 6. Equivalent explicit build loop ===")
print("  This is what the raw-MA list mode does internally; spelled")
print("  out here so the relationship between translate_attributes,")
print("  build_exp_tens, and cos_sim_exp_tens is transparent.")
print()

dens_ref = mpt.build_exp_tens(
    ref_pAttr, None, sigma, r, groups,
    is_rel, is_per, periods, verbose=False,
)
S_manual = np.empty(M, dtype=np.float64)
for m, pa in enumerate(qry_pAttr_swept):
    dens_q = mpt.build_exp_tens(
        pa, None, sigma, r, groups, is_rel, is_per, periods,
        verbose=False,
    )
    S_manual[m] = mpt.cos_sim_exp_tens(dens_ref, dens_q, verbose=False)

discrepancy = float(np.max(np.abs(S_flat - S_manual)))
print(f"  max |S_raw - S_manual| = {discrepancy:.2e} "
      f"(floating-point parity)")
assert discrepancy < 1e-12, \
    "Raw-MA list mode disagrees with manual build loop."

print("\n=== Demo complete ===")
plt.show()

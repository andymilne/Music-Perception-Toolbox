"""demo_translate_sweep.py

A sliding comparison by attribute translation: a query translated in
pitch and time across a reference, with the similarity read at every
offset.

Scenario: a 3-note motif (C E G) hidden inside a 7-note melody
(D E F C E G A, one note per second). The motif appears exactly at
reference times 3, 4, 5. At each (pitch transposition, time shift)
offset the query is translated and compared with the reference by the
cosine similarity of a 2-attribute MAET (pitch periodic at the octave;
time absolute and not periodic). The profile should peak at
(0 cents, 3 s), where the query lands on the embedded C-E-G, and at
(1200 cents, 3 s) by octave periodicity. Both sequences start at time
0, so a time offset is the time from the start of the reference to the
start of the query (User Guide §7.3.4).

1. The inputs, as two pre-MAETs.
2. The whole profile in one call: ``windowed_similarity`` with an
   ``offsets`` map naming both attributes and no window.
3. The same call with a window in time, which travels with the query
   and restricts the comparison to the reference's notes near it.
4. The two profiles plotted.
5. What the call in 2 computes: ``sweep_sim_maet`` on the two built
   densities, one pass over the tuple pairs and one evaluation per
   offset, with no translated copy of the query built.
6. The same profile offset by offset, for transparency:
   ``translate_attributes`` builds the translated copies, ``sim_maet``
   compares them in its list mode, and an explicit loop over the
   offsets does the same one comparison at a time.

See also
--------
mpt.windowed_similarity
mpt.sweep_sim_maet
mpt.translate_attributes
mpt.sim_maet
mpt.build_maet

The MATLAB mirror is demo_translateSweep.m.
"""

import numpy as np
import matplotlib.pyplot as plt

import mpt

# Keep the dispatcher's per-call announcements out of the printed
# output (show_hints gates only those); restored at the end.
_prev_defaults = mpt.set_default(show_hints=False)

# =====================================================================
# 1. The reference melody and the query motif, as pre-MAETs
# =====================================================================

print("=== 1. Inputs ===")

# Reference: D-E-F-C-E-G-A at one note per second. The query C-E-G
# appears exactly at times 3, 4, 5.
ref_midi  = np.array([62, 64, 65, 60, 64, 67, 69])
ref_pitch = mpt.transform_attributes(ref_midi, None, ('midi', 'cents')).reshape(1, -1)
ref_time  = np.arange(7, dtype=float).reshape(1, -1)
ref_pAttr = [ref_pitch, ref_time]

# Query: C-E-G, one note per second, starting at time 0 as the
# reference does.
qry_midi  = np.array([60, 64, 67])
qry_pitch = mpt.transform_attributes(qry_midi, None, ('midi', 'cents')).reshape(1, -1)
qry_time  = np.arange(3, dtype=float).reshape(1, -1)
qry_pAttr = [qry_pitch, qry_time]

# Per-attribute geometry, carried by both pre-MAETs' specs: pitch
# (attribute 0) is periodic at the octave; time (attribute 1) is
# absolute and not periodic.
sigma = [50.0, 0.3]
specs = mpt.flat_specs(ref_pAttr, name=['pitch', 'time'], sigma=sigma,
                       is_per=[True, False], period=[1200.0, 0.0])
pm_ref = mpt.pack_pre_maet(ref_pAttr, None, specs)
pm_qry = mpt.pack_pre_maet(qry_pAttr, None, specs)

# The offsets: pitch 0-1200 cents in 100-cent steps (the peak at 0
# recurs at 1200 because pitch is periodic), time -1 to 5 s in 0.25 s
# steps.
pitch_grid = np.arange(0, 1201, 100, dtype=float)
time_grid  = np.arange(-1.0, 5.001, 0.25)

print("  reference: D-E-F-C-E-G-A, one note per second")
print("  query    : C-E-G, one note per second")
print("  (the motif appears exactly at reference times 3, 4, 5)")
print(f"  sigma    : {sigma[0]:.0f} cents (pitch) / {sigma[1]:.2f} s (time)")
print(f"  offsets  : {pitch_grid.size} pitch x {time_grid.size} time")
print()


# =====================================================================
# 2. The whole profile in one call
# =====================================================================

print("=== 2. windowed_similarity with an offsets map ===")

# An offsets map {attribute: offsets} translates each named attribute
# of the query by every combination of its offsets and compares; an
# attribute is windowed only if context_window names it, and here none
# is, so this is attribute translation and nothing else. The output is
# indexed by the offsets, one dimension per attribute: (pitch, time).
S = mpt.windowed_similarity(pm_ref, pm_qry,
                            offsets={0: pitch_grid, 1: time_grid},
                            normalize='cosine')

i, j = np.unravel_index(int(np.argmax(S)), S.shape)
print(f"  cosine similarity surface: shape {S.shape} (pitch x time)")
print(f"  max similarity {S.max():.4f} at pitch shift "
      f"{pitch_grid[i]:.0f} c, time shift {time_grid[j]:.2f} s")
print("  (expected: 0 cents, 3.00 s --- the embedded C-E-G)")
print()


# =====================================================================
# 3. Adding a window in time
# =====================================================================

print("=== 3. The same call with a window in time ===")

# Without a window the query is compared with the whole reference, so
# even at the match the reference's other four notes lower the cosine.
# A window on time, named in context_window, travels with the query
# (centred on its position, the offset plus its mean onset) and
# weights the reference's events by their distance from it, so the
# comparison is local. A rectangle 3 s wide spans the query's three
# notes; at the match it keeps exactly the embedded C-E-G.
S_win = mpt.windowed_similarity(
    pm_ref, pm_qry, offsets={0: pitch_grid, 1: time_grid},
    context_window={1: {'shape': 'rect', 'width': 3.0}},
    normalize='cosine')

i_w, j_w = np.unravel_index(int(np.argmax(S_win)), S_win.shape)
print(f"  max similarity {S_win.max():.4f} at pitch shift "
      f"{pitch_grid[i_w]:.0f} c, time shift {time_grid[j_w]:.2f} s")
print(f"  ({S.max():.4f} without the window, where the reference's other")
print("   four notes dilute the match)")
print()


# =====================================================================
# 4. The two profiles
# =====================================================================

print("=== 4. Plot ===")

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, surf, name in [(axs[0], S, "no window"),
                       (axs[1], S_win, "time window 3 s wide")]:
    im = ax.imshow(surf.T, origin="lower", aspect="auto", vmin=0, vmax=1,
                   extent=[pitch_grid[0] - 50, pitch_grid[-1] + 50,
                           time_grid[0] - 0.125, time_grid[-1] + 0.125])
    ax.set_xlabel("Pitch transposition (cents)")
    ax.set_title(f"Cosine similarity, {name}")
    ax.set_xticks(np.arange(0, 1201, 200))
    # The expected peaks: the query on the embedded C-E-G at (0 c, 3 s),
    # and again at (1200 c, 3 s) by octave periodicity.
    ax.plot([0, 1200], [3, 3], "rx", markersize=12, mew=1.5)
axs[0].set_ylabel("Time shift (s)")
axs[0].set_yticks(np.arange(-1, 5.01, 1))
fig.colorbar(im, ax=axs, label="cosine similarity")
fig.suptitle("Reference: D-E-F-C-E-G-A; query: C-E-G")
print("  Red x marks the expected peaks at (0 c, 3 s) and (1200 c, 3 s).")
print()


# =====================================================================
# 5. What the call in 2 computes: sweep_sim_maet
# =====================================================================

print("=== 5. sweep_sim_maet (one pass, no translated copies) ===")
print("  A uniform translation of the query enters the inner product only")
print("  through the offset, so the whole profile is one pass over the")
print("  tuple pairs and then one evaluation per offset. The pitch")
print("  attribute is periodic, which the mixture route refuses; under")
print("  method='auto' the orbit route carries the sweep instead (the")
print("  wrapped kernel absorbs the periodicity).")

dens_ref = mpt.build_maet(pm_ref, verbose=False)
dens_qry = mpt.build_maet(pm_qry, verbose=False)
# One column per combination of offsets: the (A, M) form.
P_mesh, T_mesh = np.meshgrid(pitch_grid, time_grid, indexing="ij")
offsets_am = np.vstack([P_mesh.ravel(), T_mesh.ravel()])
S_sweep = mpt.sweep_sim_maet(dens_ref, dens_qry, offsets_am,
                             verbose=False).reshape(P_mesh.shape)
d_sweep = float(np.max(np.abs(S - S_sweep)))
print(f"  max |S - S_sweep| = {d_sweep:.2e}")
assert d_sweep < 1e-10, "sweep_sim_maet disagrees with windowed_similarity."
print()


# =====================================================================
# 6. Offset by offset, for transparency
# =====================================================================

print("=== 6. translate_attributes + sim_maet, and an explicit loop ===")

# translate_attributes builds the M translated copies: its offsets are
# a length-A list whose entries are (1, M) rows, one candidate shift
# per column, column m of every entry together defining the m-th copy.
# The result is one pre-MAET holding all M copies on one geometry.
pm_swept = mpt.translate_attributes(
    pm_qry, [P_mesh.reshape(1, -1), T_mesh.reshape(1, -1)])
print(f"  {len(pm_swept['p_attr'])} translated copies of the query")

# sim_maet compares the reference with every copy in its list mode; it
# computes every copy with Bulger's method where the sweep in 5 took the
# orbit route, so the two agree to the truncation floor rather than to
# the last digit.
S_list = np.asarray(mpt.sim_maet(pm_ref, pm_swept, verbose=False)
                    ).reshape(P_mesh.shape)

# And one offset at a time: translate the query by one (pitch, time)
# pair, build it, and compare it with the reference.
S_loop = np.empty(P_mesh.size)
for m, (dp, dt) in enumerate(zip(P_mesh.ravel(), T_mesh.ravel())):
    dens_q = mpt.build_maet(mpt.translate_attributes(pm_qry, [dp, dt]),
                            verbose=False)
    S_loop[m] = mpt.sim_maet(dens_ref, dens_q, verbose=False)
S_loop = S_loop.reshape(P_mesh.shape)

d_list = float(np.max(np.abs(S - S_list)))
d_loop = float(np.max(np.abs(S_list - S_loop)))
print(f"  max |S - S_list|      = {d_list:.2e}")
print(f"  max |S_list - S_loop| = {d_loop:.2e}")
assert d_list < 1e-8 and d_loop < 1e-12, \
    "The offset-by-offset routes disagree."

mpt.set_default(**_prev_defaults)
print("\n=== Demo complete ===")
plt.show()

"""demo_pre_maet_io -- showing, exporting, and importing a pre-MAET.

The pre-MAET is the framework's interface: the event sequence, its
per-event–attribute element multisets, and the per-attribute parameters
(Milne 2026, Def. 2.6). Everything a MAET computes follows from it
mechanically, so being able to read one, write one, and hand one to a
colleague as a spreadsheet is the point of this demo.

Four renderings of the same object:

  1. markdown, for a terminal or a notebook;
  2. LaTeX, in the article's own markup, for a manuscript table;
  3. CSV, written out for Excel or Numbers;
  4. CSV, read back in --- byte for byte the same object.

The cell notation is the article's throughout: braces for an unordered
multiset, parentheses for an ordered one, brackets within brackets for a
nested attribute, and ``60^(0.6)`` for a weighted value. ``showPreMaet``
writes that notation and ``readPreMaet`` reads it, so the two are
inverse and a pre-MAET survives a round trip through a spreadsheet.

Sections
    1. A pre-MAET by hand, shown four ways.
    2. Round trip: write, read, and compare.
    3. Building straight from the file.
    4. Overriding a parameter the pre-MAET carries.
    5. A nested attribute, with weights.
    6. NA: a parameter a preprocessing step could not carry forward.
    7. An anisotropic kernel, carried as its three generating scalars.
    8. A pre-MAET from a score, which supplies what a score determines.

The MATLAB mirror is demos/demo_preMaetIo.m.

See also: show_pre_maet, read_pre_maet, write_pre_maet, build_exp_tens.
"""
import os
import tempfile

import numpy as np

import mpt

mpt.set_default(show_hints=False)
OUT_DIR = tempfile.mkdtemp(prefix="mpt_premaet_")


# ===================================================================
#  1. A pre-MAET by hand, shown four ways
# ===================================================================

print("=== 1. One pre-MAET, four renderings ===\n")

# Three chords of a cadence, with their onsets. Pitch is an unordered
# multiset read at r = 2 (shared pitch pairs) and periodic at the octave;
# time is a single value per event on an unbounded axis.
p_attr = [
    np.array([[62.0, 55.0, 60.0],       # a chord per column, NaN-padded
              [65.0, 59.0, 64.0],
              [69.0, 62.0, 67.0],
              [72.0, 65.0, np.nan]]),
    np.array([[0.0, 1.0, 2.0]]),
]
specs = [
    {"name": "pitch", "r": 2, "rel": False, "sym": True,
     "sigma": 0.15, "is_per": True, "period": 12.0},
    {"name": "onset", "r": 1, "rel": False, "sym": True,
     "sigma": 0.1, "is_per": False, "period": 0.0},
]

# pre_maet holds the three parts in one variable, which every function
# below then takes whole. The specs carry the kernel geometry, so nothing
# further is needed here: the pre-MAET is complete as it stands.
pm = mpt.pre_maet(p_attr, specs=specs)

print("  (a) markdown\n")
mpt.show_pre_maet(pm)

print("\n  (b) LaTeX\n")
mpt.show_pre_maet(pm, format="latex",
                  caption="A cadence as a pre-MAET.", label="tab:cadence")

print("\n  (c) CSV\n")
print(mpt.show_pre_maet(pm, format="csv", verbose=False))


# ===================================================================
#  2. Round trip: write, read, and compare
# ===================================================================

print("=== 2. Round trip through a spreadsheet ===\n")

path = os.path.join(OUT_DIR, "cadence.csv")
written = mpt.write_pre_maet(path, pm)
print(f"  written to {os.path.basename(path)}")

pm_back = mpt.read_pre_maet(path)
again = mpt.write_pre_maet(None, pm_back)

vals_ok = all(np.allclose(p_attr[a], pm_back["p_attr"][a], equal_nan=True)
              for a in range(2))
print(f"  values identical : {vals_ok}")
print(f"  file identical   : {again == written}")
print("  (the NaN pad of the three-note chord survives as a shorter cell.)\n")


# ===================================================================
#  3. Building straight from the file
# ===================================================================

print("=== 3. From file to density, with nothing supplied ===\n")

dens = mpt.build_exp_tens(pm_back, verbose=False)
print(f"  build_exp_tens(pm)                 ->  dim {dens.dim}")
print(f"  self-similarity                    ->  "
      f"{float(mpt.cos_sim_exp_tens(dens, dens, verbose=False)):.4f}")
print("  No sigma, is_per or period passed: the file carried them.\n")


# ===================================================================
#  4. Overriding what the pre-MAET carries
# ===================================================================

print("=== 4. Overriding what the pre-MAET carries ===\n")

# A parameter given at the call wins over the one in the spec, for every
# attribute and without comment: holding a baseline in the pre-MAET and
# sweeping a width past it is the ordinary idiom, so a disagreement is
# intent rather than error. All six per-attribute parameters resolve this
# way --- sigma, is_per and period, and r, rel and sym --- so a sweep over
# any of them is one call per value.
#
# show_pre_maet reads the override too, so the table states what the build
# will use rather than what the file said.
mpt.show_pre_maet(pm, sigma=[0.6, 0.1],
                  title="  with sigma = [0.6, 0.1]:")

# What the width buys is visible against a semitone shift: the wider the
# pitch kernel, the more nearly the shifted cadence matches the original.
pm_up = mpt.pre_maet([pm["p_attr"][0] + 1.0, pm["p_attr"][1]],
                     specs=pm["specs"])
print()
for sigma_pitch in (0.05, 0.15, 0.6, 2.0):
    kw = dict(sigma=[sigma_pitch, 0.1], verbose=False)
    sim = float(mpt.cos_sim_exp_tens(mpt.build_exp_tens(pm, **kw),
                                     mpt.build_exp_tens(pm_up, **kw),
                                     verbose=False))
    print(f"  sigma_pitch = {sigma_pitch:.2f}  ->  vs a semitone up: "
          f"{sim:.4f}")

# The sweep reads the pre-MAET and never writes to it, so the baseline is
# still there afterwards.
print(f"\n  pm's own sigma is still {pm['specs'][0]['sigma']:.2f}.")
print("  A nested attribute's r, rel and sym are per-level, so an override")
print("  is refused there and the spec is the place to change them.\n")


# ===================================================================
#  5. A nested attribute, with weights
# ===================================================================

print("=== 5. A nested attribute, with weights ===\n")

# Bind the three chords into one super-event: an ordered run of unordered
# chords, the shape the cadence prototypes of the article use. The kernel
# geometry crosses to the nested spec intact.
pm_b = mpt.bind_events(pm, [3, 3])
mpt.show_pre_maet(pm_b)
print()
print(mpt.show_pre_maet(pm_b, format="csv", verbose=False))
print("  The brackets are the level structure: readPreMaet rebuilds the")
print("  tags from them, so a file never has to write them down.\n")


# ===================================================================
#  6. NA: a parameter that could not be carried forward
# ===================================================================

print("=== 6. NA, where a step could not carry a parameter ===\n")

# A log is non-linear. A width is still meaningful on the log axis --- it
# expresses a ratio rather than a difference --- but the local scaling
# varies across the range, so no single value is the image of the old
# sigma. NA marks the absence of a canonical choice, and the analyst
# supplies the width the new units call for.
pm_log = mpt.transform_attributes(
    mpt.pre_maet([p_attr[1] + 1.0], specs=[specs[1]]), ["log"])
mpt.show_pre_maet(pm_log)
try:
    mpt.build_exp_tens(pm_log, verbose=False)
except ValueError as err:
    print(f"\n  build refuses it: {err}")
print("\n  Supplying a width resolves it, the analyst having chosen one:")
d_log = mpt.build_exp_tens(pm_log, sigma=[0.05], verbose=False)
print(f"    dim {d_log.dim}\n")


# ===================================================================
#  7. An anisotropic kernel in a spreadsheet
# ===================================================================

print("=== 7. A kernel covariance, as three scalars ===\n")

# A covariance is a matrix, but the covariance an analysis wants is
# generated by three numbers, so those are what the file carries.
cov_csv = (
    "name,sigma,r,rel,per,P,sym,n = 1,n = 2\n"
    'trigram,"cov(sd_position=0.2, sd_interval=0.3, sd_shift=0.5)",'
    '3,0,0,,0,"(60, 62, 64)","(62, 64, 65)"\n'
)
pm_c = mpt.read_pre_maet(cov_csv)
print("  read back as a matrix:")
print(np.round(np.asarray(pm_c["specs"][0]["sigma"]), 4))
mpt.show_pre_maet(pm_c)
d_c = mpt.build_exp_tens(pm_c, verbose=False)
print(f"\n  builds with a kernel covariance: "
      f"{d_c.kernel_cov[0] is not None}")
print(f"  and writes back unchanged      : "
      f"{mpt.write_pre_maet(None, pm_c) == cov_csv}\n")


# ===================================================================
#  8. A pre-MAET from a score
# ===================================================================

print("=== 8. From a score ===\n")

# The chorale the JMM demos analyse, which ships with the demos: a demo
# should not reach into the test tree for its data.
score = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "jmm", "data", "bwv347.musicxml")
if os.path.exists(score):
    pm_s = mpt.events_from_score(
        score, attributes=("pitch", "onset"), chords="bind")
    # A score determines periodicity and not kernel widths, so sigma is
    # left for the analyst; the table shows what is still missing.
    mpt.show_pre_maet(pm_s, max_events=5, max_elements=4)
    try:
        mpt.build_exp_tens(pm_s, verbose=False)
    except ValueError as err:
        print(f"\n  {err}")
    for spec, sig in zip(pm_s["specs"], (0.15, 0.1)):
        spec["sigma"] = sig
    out = os.path.join(OUT_DIR, "from_score.csv")
    mpt.write_pre_maet(out, pm_s)
    print(f"\n  widths chosen and exported to {os.path.basename(out)}")
    print("  -- the analysis is now a spreadsheet a colleague can edit.")
else:
    print("  (score fixture not found; skipping)")

print(f"\nFiles written to {OUT_DIR}")

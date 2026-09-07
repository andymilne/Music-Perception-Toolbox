"""Handling repeated pitches under interval-scale and tempo invariance.

A melodic contour can be matched up to a uniform scaling of its
intervals -- every step multiplied by the same factor -- by carrying
each pitch step as two attributes: its sign, and the logarithm of its
magnitude. A scaling then leaves every sign fixed and adds the same
constant, log(a), to every log-magnitude, so reading the bound
log-magnitude tuple relative to a common shift (rel_outer=True) makes
the match exact, just as log inter-onset intervals read relative make a
match exact across tempo (demo_tempo_invariance).

A repeated pitch breaks this: its step is 0, and 0 has no logarithm.
Something must be done with repetitions before the decomposition, and
the choice decides what a repetition is worth in the comparison. This
demo walks through three treatments of the same melody:

  excise    Drop each zero-step event (its step AND its inter-onset
            interval). A repetition leaves no trace: the melody becomes
            indistinguishable from the same melody with its repetitions
            removed and the rhythm closed up.
  prolong   Gather each run of equal consecutive pitches into one
            event at the run's first onset, before differencing. The
            repetition survives as duration: the gathered event's next
            inter-onset interval spans the whole run. The cost is an
            identification: a re-articulated note and a single held
            note of the same length produce the same events.
  count     Prolong, and add a per-event attribute holding the number
            of onsets gathered into the event. Re-articulation now
            survives as such, and its sigma grades how sharply it is
            distinguished -- wide, and the comparison slides back to
            the prolong treatment's identification.

Each treatment is run twice: once with inter-onset intervals in beats
(tempo-sensitive -- a faster statement does not match), and once with
log inter-onset intervals read relative (tempo-invariant -- it does).
The second run exposes a confound the first hides: where repetition is
uniform (every note repeated equally), closing up the repetitions IS a
tempo change, and only the count attribute keeps the two apart.

Sections:
  1. Material            The melody and five comparison variants.
  2. The three treatments, shown as events (the reference melody).
  3. Inter-onset intervals in beats: which variants match ref under
     which treatment.
  4. Log inter-onset intervals read relative: the same table, with
     tempo quotiented out.
  5. The count attribute's sigma, from sharp to broad.
  6. Uniform repetition: closing up equals a tempo change, and the
     count attribute is what keeps them apart.

Exact invariances only (rel_outer flags); the graded counterparts (the
sd_shift ridge of interval_kernel_cov, tending to rel in the limit) are
the subject of demo_tempo_invariance.
"""

import numpy as np

import mpt
from mpt import (bind_events, build_exp_tens, cos_sim_exp_tens,
                 difference_events)

# Keep the dispatcher's per-call announcements out of the printed
# tables (show_hints gates only those; the one-time truncation notice
# is not gated). The controls are covered in demo_dispatch_and_kernel_controls.
_prev_defaults = mpt.set_default(show_hints=False)

# Kernel widths, one per attribute. The sign attribute is two points a
# unit apart (+1/2 and -1/2), so its narrow sigma makes sign an exact
# comparison; log-magnitude and log-IOI sigmas are in natural-log
# units; the IOI-in-beats sigma is in beats; the count sigma is in
# onsets and is varied in Section 5.
SIGMA_SIGN = 0.05
SIGMA_LOGMAG = 0.15
SIGMA_IOI_BEATS = 0.25
SIGMA_LOGIOI = 0.15
SIGMA_COUNT = 0.25


# ===== 1. Material =====

# The reference melody: the Ode to Joy opening, isochronous, one onset
# per beat. Five of its fifteen notes are repetitions of the note
# before (E E, G G, C C, E E, D D), interleaved with unrepeated notes,
# so the repetition pattern is non-uniform -- Section 6 shows why that
# matters.
PITCHES_REF = np.array([64, 64, 65, 67, 67, 65, 64, 62,
                        60, 60, 62, 64, 64, 62, 62], dtype=float)
ONSETS_REF = np.arange(15, dtype=float)


def gather_repetitions(pitches, onsets):
    """Pool each run of equal consecutive pitches into one event.

    Returns the run-start pitches, the run-start onsets, and the run
    lengths (how many onsets each pooled event gathers).
    """
    pitches = np.asarray(pitches, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    starts = np.flatnonzero(
        np.concatenate([[True], np.abs(np.diff(pitches)) > 1e-9]))
    counts = np.diff(np.concatenate([starts, [pitches.size]]))
    return pitches[starts], onsets[starts], counts.astype(float)


# The five comparison variants, each a (pitches, onsets) pair:
#   held       The gathered reference played as sustained notes: same
#              pitches at the same onsets, but each formerly repeated
#              note now a single held note -- no re-articulations.
#   norep      The repetitions removed and the rhythm closed up: the
#              ten distinct pitches, one per beat.
#   augmented  Every pitch step doubled (contour preserved), rhythm as
#              the reference. Repetitions remain repetitions: a zero
#              step doubles to zero.
#   faster     The reference at double speed.
#   aug+faster Both at once.
_gp, _go, _ = gather_repetitions(PITCHES_REF, ONSETS_REF)
_steps_ref = np.diff(PITCHES_REF)
VARIANTS = [
    ("reference",  PITCHES_REF, ONSETS_REF),
    ("held",       _gp, _go),
    ("norep",      _gp, np.arange(_gp.size, dtype=float)),
    ("augmented",  PITCHES_REF[0] + np.concatenate(
        [[0.0], np.cumsum(2.0 * _steps_ref)]), ONSETS_REF),
    ("faster",     PITCHES_REF, 0.5 * ONSETS_REF),
    ("aug+faster", PITCHES_REF[0] + np.concatenate(
        [[0.0], np.cumsum(2.0 * _steps_ref)]), 0.5 * ONSETS_REF),
]

print("\n=== 1. Material ===\n")
print("  reference : Ode to Joy opening, 15 notes, one per beat;")
print("              5 notes are repetitions of the note before.")
for name, p, t in VARIANTS[1:]:
    print(f"  {name:<10}: {p.size} onsets")
print()


# ===== 2. The three treatments, shown as events =====

def make_events(pitches, onsets, treatment):
    """Differenced events for one variant under one treatment.

    Returns a dict with the per-event arrays: 'sign' (+1/2 or -1/2),
    'logmag' (log of the absolute pitch step), 'ioi' (inter-onset
    interval in beats), and, for the count treatment, 'count' (onsets
    gathered into the event completing the step).
    """
    pitches = np.asarray(pitches, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    if treatment in ("prolong", "count"):
        pitches, onsets, counts = gather_repetitions(pitches, onsets)
        p_diff, _, _ = difference_events(
            [pitches[None, :], onsets[None, :], counts[None, :]],
            None, [1, 1, 0])
        steps = p_diff[0].ravel()
        iois = p_diff[1].ravel()
        counts = p_diff[2].ravel()
    elif treatment == "excise":
        p_diff, _, _ = difference_events(
            [pitches[None, :], onsets[None, :]], None, [1, 1])
        steps = p_diff[0].ravel()
        iois = p_diff[1].ravel()
        keep = np.abs(steps) > 1e-9
        steps, iois = steps[keep], iois[keep]
        counts = None
    else:
        raise ValueError(f"unknown treatment {treatment!r}")
    out = {"sign": np.where(steps > 0, 0.5, -0.5),
           "logmag": np.log(np.abs(steps)),
           "ioi": iois}
    if treatment == "count":
        out["count"] = counts
    return out


def build_density(ev, log_ioi, sigma_count=SIGMA_COUNT):
    """One bound super-event: the whole event sequence as one tuple.

    Attributes: sign, log step magnitude, inter-onset interval, and
    (when present) count. The log-magnitude tuple is read relative
    (rel_outer), quotienting a common shift -- a uniform scaling of
    the pitch steps. With log_ioi, the intervals are taken to
    logarithms and read relative too, quotienting a tempo change; in
    beats they are read absolute, so tempo differences count.
    """
    ioi = np.log(ev["ioi"]) if log_ioi else ev["ioi"]
    p_attr = [ev["sign"][None, :], ev["logmag"][None, :], ioi[None, :]]
    rel = [False, True, bool(log_ioi)]
    sig = [SIGMA_SIGN, SIGMA_LOGMAG,
           SIGMA_LOGIOI if log_ioi else SIGMA_IOI_BEATS]
    if "count" in ev:
        p_attr.append(ev["count"][None, :])
        rel.append(False)
        sig.append(sigma_count)
    L = ev["sign"].size
    p_b, w_b, sp_b = bind_events(p_attr, None, L, rel_outer=rel)
    return build_exp_tens(p_b, w_b, specs=sp_b, sigma=sig,
                          is_per=[False] * len(sig),
                          period=[None] * len(sig), verbose=False)


TREATMENTS = ["excise", "prolong", "count"]

print("=== 2. The reference melody under each treatment ===\n")
for tr in TREATMENTS:
    ev = make_events(PITCHES_REF, ONSETS_REF, tr)
    n = ev["sign"].size
    print(f"  {tr}: {n} events")
    print("    step sign  :", "  ".join(
        "+" if s > 0 else "-" for s in ev["sign"]))
    print("    |step|     :", "  ".join(
        f"{m:.0f}" for m in np.exp(ev["logmag"])))
    print("    IOI (beats):", "  ".join(f"{d:.0f}" for d in ev["ioi"]))
    if "count" in ev:
        print("    onsets     :", "  ".join(
            f"{c:.0f}" for c in ev["count"]))
    print()
print("  The excised and prolonged step sequences coincide; they part")
print("  on the intervals. Excision keeps each surviving event's own")
print("  1-beat interval, so the repetitions' beats are gone; the")
print("  prolonged events' intervals span the gathered runs, so those")
print("  beats survive as duration. The count row records what")
print("  prolongation alone forgets: how many onsets each event held.")
print()


# ===== 3 & 4. Which variants match the reference =====

def similarity_table(log_ioi):
    ref_dens = {tr: build_density(
        make_events(PITCHES_REF, ONSETS_REF, tr), log_ioi)
        for tr in TREATMENTS}
    header = f"  {'variant':<12}" + "".join(f"{tr:>10}" for tr in TREATMENTS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, p, t in VARIANTS[1:]:
        row = f"  {name:<12}"
        for tr in TREATMENTS:
            d = build_density(make_events(p, t, tr), log_ioi)
            s = cos_sim_exp_tens(ref_dens[tr], d, verbose=False)
            row += f"{s:>10.3f}"
        print(row)
    print()


print("=== 3. Inter-onset intervals in beats ===\n")
print("  Cosine similarity of each variant to the reference. The")
print("  log-magnitude tuple is read relative throughout, so the")
print("  augmented variant -- every step doubled, rhythm unchanged --")
print("  matches exactly under every treatment. The intervals are in")
print("  beats, so the faster variants do not. At these narrow kernel")
print("  widths each entry reads as identified (1.000) or separated")
print("  (near 0); widening an attribute's sigma grades its")
print("  separations, as Section 5 does for the count.\n")
similarity_table(log_ioi=False)
print("  Each treatment commits to one identification, visible in its")
print("  column's 1.000: excise cannot tell the reference from norep")
print("  (the repetitions leave no trace), prolong cannot tell it from")
print("  held (re-articulation and prolongation coincide), and count")
print("  distinguishes all four.\n")


print("=== 4. Log inter-onset intervals, read relative ===\n")
print("  The same comparisons with the intervals taken to logarithms")
print("  and the bound tuple read relative: a tempo change is a common")
print("  shift of the log intervals, so the faster variants now match")
print("  exactly -- including aug+faster, scaled in pitch and time at")
print("  once.\n")
similarity_table(log_ioi=True)
print("  The treatments' identifications survive the tempo quotient")
print("  here because the reference's repetitions are non-uniform: its")
print("  gathered intervals (2 1 2 1 1 1 2 1 2) are not a common")
print("  scaling of norep's (1 1 ... 1), so prolongation still")
print("  separates them. Section 6 shows the uniform case, where it")
print("  cannot.\n")


# ===== 5. The count attribute's sigma =====

print("=== 5. Grading the count attribute ===\n")
print("  Under the count treatment, reference vs held differ only on")
print("  the count attribute (2 vs 1 at the five gathered events). Its")
print("  sigma sets how much that difference costs: narrow, the two")
print("  are far apart; broad, the counts blur together and the")
print("  comparison returns to the prolong treatment's identification")
print("  of re-articulated with held.\n")
ev_ref = make_events(PITCHES_REF, ONSETS_REF, "count")
ev_held = make_events(VARIANTS[1][1], VARIANTS[1][2], "count")
for sc in (0.25, 0.75, 1.5, 3.0):
    d_ref = build_density(ev_ref, log_ioi=True, sigma_count=sc)
    d_held = build_density(ev_held, log_ioi=True, sigma_count=sc)
    s = cos_sim_exp_tens(d_ref, d_held, verbose=False)
    print(f"    count sigma = {sc:.2f}: similarity = {s:.3f}")
print()


# ===== 6. Uniform repetition: closing up equals a tempo change =====

print("=== 6. Uniform repetition ===\n")
print("  A figure whose every note is repeated: C C G G A A G G, one")
print("  onset per beat, against the same figure with the repetitions")
print("  removed and closed up: C G A G. Gathering the first gives the")
print("  pitches of the second at intervals (2 2 2) against (1 1 1) --")
print("  exactly a common factor, so once tempo is quotiented out the")
print("  prolong treatment cannot separate them: removing uniform")
print("  repetition IS a tempo change. The counts (2 2 2) against")
print("  (1 1 1) are untouched by either quotient, so the count")
print("  treatment can.\n")
P_UNIF = np.array([60, 60, 67, 67, 69, 69, 67, 67], dtype=float)
T_UNIF = np.arange(8, dtype=float)
P_UNOREP = np.array([60, 67, 69, 67], dtype=float)
T_UNOREP = np.arange(4, dtype=float)
for tr in ("prolong", "count"):
    d1 = build_density(make_events(P_UNIF, T_UNIF, tr), log_ioi=True)
    d2 = build_density(make_events(P_UNOREP, T_UNOREP, tr), log_ioi=True)
    s = cos_sim_exp_tens(d1, d2, verbose=False)
    print(f"    {tr:<8}: doubled figure vs closed-up figure = {s:.3f}")
print()
print("  With intervals in beats the two are already distinct under")
print("  every treatment; the confound is a price of tempo invariance,")
print("  and the count attribute -- dimensionless, so invariant to")
print("  both scalings for free -- is what pays it off.")

mpt.set_default(**_prev_defaults)

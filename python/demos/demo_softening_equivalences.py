"""demo_softening_equivalences.py

Softening an equivalence by pairing attributes.

Each of the three flags imposes an equivalence exactly: [rel] relative
identifies a tuple with its transpositions, [per] periodic identifies a
value with its octaves, and [exch] exchangeable identifies a tuple with
its reorderings. Softening an equivalence makes it hold in part. The
same values are carried twice in one event, on two attributes that
agree in every respect but one: the flagged copy (attribute c) carries
the flag at width sigma_c, and the unflagged copy (attribute f) omits it
at width sigma_f. The unflagged width sets how firmly the equivalence
holds: as sigma_f grows, a pair of element multisets the flag
identifies (a query and a context) runs from cosine similarity 0 (the
identification lost) up to the flagged copy's own value. For every pair
compared here the cosine similarity factorizes as

    (flagged copy alone, at sigma_c) * exp(-|d|^2 / (4 sigma_f^2)),

d being the difference between the two ordered tuples (Online
Supplement, Sec. "Softening an equivalence with paired attributes").

Sections:
  1. Transposition softened: a motif against its transposition and an
     altered transposition, [rel] copy paired with an absolute copy;
     the pairing equals the kernel covariance that kernel_cov builds
     from the supplement's mapping of (sigma_c, sigma_f).
  2. Octave equivalence softened: [per] copy paired with a non-periodic
     copy (demo_helix_blend treats this pairing in depth).
  3. Reordering softened: exchangeable copy paired with an ordered copy.
  4. Figure: similarity against sigma_f for the three, with the limits.

Uses: pack_pre_maet, flat_specs, sim_maet, kernel_cov, set_default.

The MATLAB mirror is demo_softeningEquivalences.m.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

import mpt

# The dispatcher's per-call announcements are switched off for a tidy
# printout, and restored at the end.
prev_defaults = mpt.set_default(show_hints=False)
t_start = time.time()

SIG_C = 30.0                                  # flagged copy's width (cents)
SIG_F = np.logspace(1, 5, 41)                 # unflagged copy's widths (cents)
SIG_F_TABLE = [100.0, 300.0, 1000.0, 3000.0, 10000.0]
MOTIF = np.array([6000.0, 6200.0, 6400.0, 6700.0])   # C D E G (cents)


def pre_maet(x, copies):
    """One event whose element multiset x is carried by one attribute per
    copy; each copy is (sigma, rel, exch, is_per), read whole (r = K)."""
    p = [np.reshape(x, (-1, 1))] * len(copies)
    sig, rel, exch, per = (list(c) for c in zip(*copies))
    return mpt.pack_pre_maet(p, None, mpt.flat_specs(
        p, r=len(x), rel=rel, exch=exch, sigma=sig, is_per=per,
        period=[1200.0 if q else 0.0 for q in per]))


def sim(q, x, copies):
    # truncation_sigmas=inf sums each kernel out to the toolbox's accuracy
    # floor (1e-12) rather than the default 6 sigmas, so the comparisons
    # below also hold for similarities too small for the default to keep.
    return mpt.sim_maet(pre_maet(q, copies), pre_maet(x, copies),
                        truncation_sigmas=np.inf, verbose=False)


# Each case: the flagged copy's (rel, exch, is_per), the query, and two
# contexts that the flag identifies with the query, exactly or nearly.
CASES = [
    ("1. Transposition softened ([rel] copy + absolute copy)",
     (True, False, False), MOTIF,
     [("C D E G up a fifth", MOTIF + 700),
      ("... last note 50 cents sharp", MOTIF + [700, 700, 700, 750])]),
    ("2. Octave equivalence softened ([per] copy + non-periodic copy)",
     (False, False, True), MOTIF[:1],
     [("C one octave up", MOTIF[:1] + 1200),
      ("C two octaves up", MOTIF[:1] + 2400)]),
    ("3. Reordering softened (exchangeable copy + ordered copy)",
     (False, True, False), MOTIF,
     [("D C E G (neighbours swapped)", MOTIF[[1, 0, 2, 3]]),
      ("G D E C (C and G swapped)", MOTIF[[3, 1, 2, 0]])]),
]
UNFLAGGED = (False, False, False)

curves = []
for title, flag, q, contexts in CASES:
    print(f"\n=== {title} ===\n")
    print(f"  Query {q.astype(int).tolist()} cents; sigma_c = {SIG_C:g} cents.")
    print("  " + f"{'context':<30}" + "".join(f"{s:>9g}" for s in SIG_F_TABLE)
          + f"{'flagged':>9}{'|pair - product|':>18}")
    rows = []
    for label, x in contexts:
        def paired(s):
            return sim(q, x, [(SIG_C,) + flag, (s,) + UNFLAGGED])
        pair = np.array([paired(s) for s in SIG_F])
        table = [paired(s) for s in SIG_F_TABLE]
        flagged = sim(q, x, [(SIG_C,) + flag])        # sigma_f -> infinity
        product = flagged * np.exp(-np.sum((x - q) ** 2) / (4 * SIG_F ** 2))
        print(f"  {label:<30}" + "".join(f"{v:9.4f}" for v in table)
              + f"{flagged:9.4f}"
              + f"{np.max(np.abs(pair - product)):18.1e}")
        rows.append((label, pair, flagged))
    curves.append((title, rows))
    print("  (Columns: sigma_f in cents; 'flagged' is the flagged copy alone,")
    print("  the limit as sigma_f grows. As sigma_f shrinks the pair tends to 0,")
    print("  the identification lost; the last column is the largest departure")
    print("  from the factorization over the sigma_f sweep.)")

    if flag[0]:
        # On an ordered, absolute, non-periodic attribute at r = K, the
        # kernel covariance sigma_val^2 I + sigma_shift^2 11^T (sigma_int
        # = 0) is the same kernel as the pairing, under the mapping
        # sigma_val = sigma_c sigma_f / sqrt(sigma_c^2 + sigma_f^2),
        # sigma_shift = sigma_f^2 / sqrt(r (sigma_c^2 + sigma_f^2)).
        r = len(q)
        v = SIG_C ** 2 + SIG_F ** 2
        covs = [mpt.kernel_cov(r, sd_value=SIG_C * s / np.sqrt(vs),
                               sd_shift=s ** 2 / np.sqrt(r * vs),
                               differenced=False)
                for s, vs in zip(SIG_F, v)]
        print("\n  The pair against one absolute attribute with the kernel")
        print("  covariance kernel_cov(r, sd_value, sd_shift) of the mapping")
        print("  (the ridge's condition number grows as sigma_f^2 / sigma_c^2,")
        print("  costing a few digits at the widest sigma_f):")
        for (label, x), (_, pair, _) in zip(contexts, rows):
            ridge = np.array([sim(q, x, [(c, False, False, False)])
                              for c in covs])
            print(f"    {label:<30} max |pair - kernel_cov| = "
                  f"{np.max(np.abs(pair - ridge)):.1e}")


# ===== 4. Figure =====

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (title, rows) in zip(axes, curves):
    for (label, pair, flagged), col in zip(rows, ["C0", "C1"]):
        ax.semilogx(SIG_F, pair, color=col, label=label)
        ax.axhline(flagged, color=col, linestyle="--", linewidth=0.8)
    ax.axvline(SIG_C, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title(title.split(" (")[0], fontsize=10)
    ax.set_xlabel(r"$\sigma_f$ (cents)")
    ax.legend(fontsize=8, loc="center left")
axes[0].set_ylabel("cosine similarity to the query")
fig.suptitle("Softening an equivalence: dashed lines mark the flagged copy "
             r"alone ($\sigma_f \to \infty$); dotted, $\sigma_c$", fontsize=10)
fig.tight_layout()

mpt.set_default(**prev_defaults)
print(f"\nDone in {time.time() - t_start:.1f} s.")
plt.show()

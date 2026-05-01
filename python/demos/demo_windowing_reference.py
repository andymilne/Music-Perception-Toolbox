"""demo_windowing_reference.py

MAET post-tensor windowing: choice of reference point for the offset axis.

Background
==========
`windowed_similarity` returns a windowed-similarity profile as a function of
a user-supplied offset delta on each windowed attribute. The offset
is measured from a reference point to the window centre. Two
reference-point options are provided:

  D (default)  The unweighted column mean of the query's tuple centres
               on each attribute:
                   ref_a^D = (1/N) sum_j c_{a,j}^q
               This is a geometric property of the query's tuple
               centres, independent of the tuple weights.

  F (fixed)    A user-supplied constant reference, one vector per
               attribute, that does not depend on the query.

The choice matters most when a pitch attribute has more than one slot
per event (chords with exchangeable voices, or partials added by
addSpectra). Queries can then differ in:

  - slot count   (e.g., adding partials)
  - slot values  (e.g., stretching partials)
  - slot weights (e.g., changing rolloff)

This demo characterises how the similarity profile responds to each
of these three kinds of between-query variation, under each of the
two reference methods, for two baselines: a canonical harmonic query
(12 integer-harmonic partials), and a non-harmonic query (12 stretched
partials at beta=1.05).

Findings (summary)
==================
Let P* be the absolute pitch at which the similarity profile peaks,
and mu_q the query's unweighted pitch centroid. Then peak offsets
satisfy

    delta^*_D = P* - mu_q        (default)
    delta^*_F = P* - ref_fixed   (fixed reference)

How each of P* and mu_q responds to between-query variation gives the
response of delta^*:

* Slot weights. Neither P* nor mu_q moves (unweighted centroid is
  weight-independent; peak location depends on slot values, not
  weights). Both methods give identical, stable peak offsets.
  Holds for any slot structure.

* Slot values. mu_q moves smoothly with the sweep parameter while
  P* sits on a branch of the similarity profile that may be pinned
  locally in absolute pitch. Within a branch, delta^*_D drifts;
  delta^*_F stays put. Holds for any slot structure.

* Slot count. For harmonic queries (slot values at or close to
  integer-harmonic positions above each fundamental), P* and mu_q
  co-move closely as partials are added, so delta^*_D is stable.
  For non-harmonic queries (e.g., stretched partials), the two move
  by different amounts, so delta^*_D drifts. delta^*_F shifts with
  P* in both cases.

The harmonic case is special for slot-count changes because integer-
harmonic positions on a log-frequency axis are self-similar under
extension: adding slot n+1 at 1200*log2(n+1) cents extends a pattern
whose centroid-shift closely matches the alignment-peak shift against
a similarly harmonic context.

Figures
=======
Figure 1: Profiles at fixed time for four query configurations, under
         both reference methods, as a function of offset. Shows the
         visual character of each reference choice for a small
         catalogue of queries.

Figure 2: Three sweeps (slot count N, stretch beta, rolloff rho) for
         each baseline, rendered as similarity-profile heatmaps
         (rows: sweep parameter, columns: reference method). Rows are
         stacked across the three sweeps.

Figure 3: Peak offsets under each method as a function of sweep
         parameter, overlaid per sweep, for both baselines. Compact
         summary of which drifts and which stays put.

Figure 4: Stretch sweep under F calibrated to the canonical harmonic
         query (ref = unweighted centroid of the 12-partial harmonic
         query). The partial-alignment branch reads at the musical
         transposition interval (+700) across the sweep, while peak
         amplitude reflects the degree of harmonicity mismatch.

Uses: buildExpTens, evalExpTens, cosSimExpTens, windowedCosSim,
      addSpectra, convertPitch.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpt

# ---------------- User-editable parameters ------------------------
PITCH_SIGMA_CENTS = 12.0
TIME_SIGMA_SEC    = 0.030
WIN_SIZE_PITCH_EFF = 400.0   # cents
WIN_SIZE_TIME     = 25       # in TIME_SIGMA_SEC units (~0.75 s)
WIN_MIX           = 0.5

# Context: a monophonic 18-event melody over 8 s.
EVENTS = [(0.00,62),(0.50,64),(1.00,65),(1.50,64),(2.00,64),(2.50,65),
          (3.00,67),(3.25,65),(3.50,64),(4.00,67),(4.50,69),(5.00,70),
          (6.00,72),(6.25,70),(6.50,69),(7.00,71),(7.50,72),(8.00,74)]

# Query fundamentals: first three events of the context (D-E-F).
N_QUERY_EVENTS = 3

# Time offset at which profiles are evaluated (corresponds to the
# motif M4 = A-B-C in the context, a P5 above the query fundamentals).
M4_TIME = 6.5

# Baseline slot configurations for the two query families.
HARMONIC_N_PARTIALS = 12
HARMONIC_ROLLOFF    = 0.5
INHARMONIC_BETA     = 1.05  # stretch factor for the non-harmonic baseline

# Sweep grids
N_SWEEP     = list(range(HARMONIC_N_PARTIALS, 25))       # 12 .. 24
BETA_SWEEP  = np.arange(0.90, 1.1001, 0.01)              # 0.90 .. 1.10
RHO_SWEEP   = np.arange(0.5, 1.51, 0.1)                  # 0.5 .. 1.5

# Offset grid used for the profile heatmaps and peak-picking
OFFSET_GRID = np.arange(-3000, 3001, 50.0)

# ---------------- Utilities ---------------------------------------
ctx_t = np.array([e[0] for e in EVENTS])
ctx_midi = np.array([e[1] for e in EVENTS])
ctx_p = mpt.convert_pitch(ctx_midi.astype(float), 'midi', 'cents')
q_t = ctx_t[:N_QUERY_EVENTS]
q_p = ctx_p[:N_QUERY_EVENTS]


def build_context():
    pf, wf = mpt.add_spectra(ctx_p, None, 'harmonic',
                              HARMONIC_N_PARTIALS, 'powerlaw',
                              HARMONIC_ROLLOFF)
    n = ctx_p.size
    return mpt.build_exp_tens(
        [pf.reshape(n, HARMONIC_N_PARTIALS).T, ctx_t[None, :]],
        [wf.reshape(n, HARMONIC_N_PARTIALS).T, None],
        [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], None,
        [False, False], [False, False], [0., 0.], verbose=False)


def build_query(*, rolloff=HARMONIC_ROLLOFF, beta=1.0,
                n_partials=HARMONIC_N_PARTIALS):
    if beta == 1.0:
        pf, wf = mpt.add_spectra(q_p, None, 'harmonic', n_partials,
                                  'powerlaw', rolloff)
    else:
        pf, wf = mpt.add_spectra(q_p, None, 'stretched', n_partials,
                                  beta, 'powerlaw', rolloff)
    n = q_p.size
    return mpt.build_exp_tens(
        [pf.reshape(n, n_partials).T, q_t[None, :]],
        [wf.reshape(n, n_partials).T, None],
        [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], None,
        [False, False], [False, False], [0., 0.], verbose=False)


def profile(dens_q, dens_c, spec, offsets_1d, reference=None):
    """Return cos-sim profile as a function of offset on the pitch
    attribute, with time fixed at M4_TIME."""
    offs = np.vstack([offsets_1d, np.full_like(offsets_1d, M4_TIME)])
    return mpt.windowed_similarity(dens_q, dens_c, spec, offs,
                                 reference=reference, verbose=False)


def mu_q_pitch(dens_q):
    return float(dens_q.centres[0].mean())


# Build baseline objects
dens_c = build_context()
spec = {'size': [WIN_SIZE_PITCH_EFF / PITCH_SIGMA_CENTS, WIN_SIZE_TIME],
        'mix':  [WIN_MIX, WIN_MIX]}

dens_q_harm = build_query()
dens_q_inh  = build_query(beta=INHARMONIC_BETA)

# F reference calibrated to the harmonic baseline (used throughout
# except for Figure 2's 'F' column where it follows the sweep
# baseline, see that section)
REF_HARM = [dens_q_harm.centres[0].mean(axis=1),
            dens_q_harm.centres[1].mean(axis=1)]

print(f'Context: {len(EVENTS)} events over {ctx_t[-1]} s, all {HARMONIC_N_PARTIALS}-partial harmonic.')
print(f'Query fundamentals: D-E-F (events 1-3 of context).')
print(f'Baseline harmonic query:  12 partials at integer-harmonic values, rho={HARMONIC_ROLLOFF}.')
print(f'Baseline non-harmonic query: 12 stretched partials at beta={INHARMONIC_BETA}, rho={HARMONIC_ROLLOFF}.')
print(f'Time offset at evaluation: t={M4_TIME} s (motif M4 = A-B-C, P5 above query).')
print(f'Harmonic-calibrated fixed reference: mu_q of harmonic baseline = '
      f'{float(REF_HARM[0][0]):.2f} cents.')
print()


# ============================================================
# Figure 1 -- Profile catalogue at fixed time, four scenarios
# ============================================================
print('Rendering Figure 1: scenario catalogue')

scenarios = [
    ('1. Harmonic, N=12, ρ=0.5\n(canonical baseline)',
     dict(beta=1.0, n_partials=12, rolloff=0.5)),
    ('2. Harmonic, N=12, ρ=1.5\n(slot weights changed)',
     dict(beta=1.0, n_partials=12, rolloff=1.5)),
    ('3. Stretched, N=12, ρ=0.5\n(slot values changed)',
     dict(beta=1.05, n_partials=12, rolloff=0.5)),
    ('4. Harmonic, N=24, ρ=0.5\n(slot count changed)',
     dict(beta=1.0, n_partials=24, rolloff=0.5)),
]

fig, axes = plt.subplots(len(scenarios), 2, figsize=(12, 9),
                          sharex=True, sharey=True)

y_max = 0.0
all_profs = []
for label, kw in scenarios:
    dq = build_query(**kw)
    mu = mu_q_pitch(dq)
    p_D = profile(dq, dens_c, spec, OFFSET_GRID)
    p_F = profile(dq, dens_c, spec, OFFSET_GRID, reference=REF_HARM)
    all_profs.append((label, mu, p_D, p_F))
    y_max = max(y_max, p_D.max(), p_F.max())

for i, (label, mu, p_D, p_F) in enumerate(all_profs):
    axD, axF = axes[i, 0], axes[i, 1]
    for ax, p, ref_val, title in (
            (axD, p_D, mu, 'Default (ref = query centroid)'),
            (axF, p_F, float(REF_HARM[0][0]), 'Fixed (ref = harmonic-baseline centroid)')):
        ax.plot(OFFSET_GRID, p, color='C0', lw=1.0)
        ax.axvline(0, color='k', lw=0.5, linestyle=':')
        ax.axvline(700, color='tab:red', lw=0.8, linestyle='--',
                   label='+700 (P5 musical)' if i == 0 else None)
        ax.set_ylim(0, y_max * 1.1)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.98, 0.95, f'ref = {ref_val:.0f} c',
                 transform=ax.transAxes, va='top', ha='right',
                 fontsize=8, color=(0.25, 0.25, 0.25),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='none', alpha=0.8))
        if i == 0:
            ax.set_title(title, fontsize=10)
    # Scenario label as row title, placed as the ylabel
    axD.text(-0.15, 0.5, label, transform=axD.transAxes,
             rotation=90, va='center', ha='right', fontsize=8,
             fontweight='bold')
    if i == len(scenarios) - 1:
        axD.set_xlabel('offset delta (cents)')
        axF.set_xlabel('offset delta (cents)')

axes[0, 0].legend(loc='upper right', fontsize=8)
fig.suptitle('Figure 1. Profiles at t=M4 for four scenarios, under each '
             'reference method', fontsize=11)
fig.subplots_adjust(left=0.18, right=0.97, top=0.93, bottom=0.07,
                     hspace=0.25, wspace=0.08)
fig.savefig(os.path.join(os.path.dirname(__file__),
                          'demo_windowing_reference_fig1.png'), dpi=110)
plt.close(fig)


# ============================================================
# Figure 2 -- Sweep heatmaps (N, beta, rho) x (harmonic, non-harm)
# ============================================================
print('Rendering Figure 2: sweep heatmaps')

def collect_sweep(kw_list, ref_F, labels):
    """For a list of query-keyword dicts, return:
      - profiles under D   shape (len(kw_list), len(OFFSET_GRID))
      - profiles under F   shape (same)
      - peaks under D      shape (len(kw_list),)
      - peaks under F      shape (same)
      - peaks in P*        (absolute pitch)  shape (same)
      - mu_q                shape (same)
    """
    nS = len(kw_list)
    profs_D = np.zeros((nS, len(OFFSET_GRID)))
    profs_F = np.zeros((nS, len(OFFSET_GRID)))
    pk_D = np.zeros(nS); pk_F = np.zeros(nS)
    pk_abs = np.zeros(nS); mus = np.zeros(nS)
    # Coarser abs grid to locate absolute peak position
    abs_grid = np.arange(4000, 13001, 50.0)
    for i, kw in enumerate(kw_list):
        dq = build_query(**kw)
        mus[i] = mu_q_pitch(dq)
        profs_D[i, :] = profile(dq, dens_c, spec, OFFSET_GRID)
        profs_F[i, :] = profile(dq, dens_c, spec, OFFSET_GRID,
                                 reference=ref_F)
        p_abs = profile(dq, dens_c, spec, abs_grid - mus[i])
        pk_abs[i] = abs_grid[int(np.argmax(p_abs))]
        pk_D[i] = OFFSET_GRID[int(np.argmax(profs_D[i, :]))]
        pk_F[i] = OFFSET_GRID[int(np.argmax(profs_F[i, :]))]
    return profs_D, profs_F, pk_D, pk_F, pk_abs, mus

# --- Sweep definitions -------
sweeps = []  # list of (name, kw_list, y_ticks_labels, y_axis_label, ref_F, base_tag)
# All sweeps use the harmonic-calibrated F reference (REF_HARM, the
# unweighted centroid of the 12-partial harmonic baseline). This matches
# the practical guidance: F calibrated to a canonical harmonic query
# keeps the musical-interval reading across harmonic and non-harmonic
# perturbations.

harm_base = dict(beta=1.0, n_partials=HARMONIC_N_PARTIALS,
                  rolloff=HARMONIC_ROLLOFF)
inh_base  = dict(beta=INHARMONIC_BETA, n_partials=HARMONIC_N_PARTIALS,
                  rolloff=HARMONIC_ROLLOFF)

# Harmonic baseline sweeps
sweeps.append(('Harm: slot count N',
                [dict(beta=1.0, n_partials=n, rolloff=HARMONIC_ROLLOFF)
                  for n in N_SWEEP],
                [str(n) for n in N_SWEEP], 'N', REF_HARM, 'harm'))
sweeps.append(('Harm: slot values beta',
                [dict(beta=b, n_partials=HARMONIC_N_PARTIALS,
                      rolloff=HARMONIC_ROLLOFF) for b in BETA_SWEEP],
                [f'{b:.2f}' for b in BETA_SWEEP], 'beta',
                REF_HARM, 'harm'))
sweeps.append(('Harm: slot weights rho',
                [dict(beta=1.0, n_partials=HARMONIC_N_PARTIALS, rolloff=r)
                  for r in RHO_SWEEP],
                [f'{r:.1f}' for r in RHO_SWEEP], 'rho',
                REF_HARM, 'harm'))

# Non-harmonic baseline sweeps
sweeps.append(('Inh: slot count N',
                [dict(beta=INHARMONIC_BETA, n_partials=n, rolloff=HARMONIC_ROLLOFF)
                  for n in N_SWEEP],
                [str(n) for n in N_SWEEP], 'N', REF_HARM, 'inh'))
sweeps.append(('Inh: slot values beta',
                # Centre beta sweep on the inharmonic baseline 1.05
                [dict(beta=b, n_partials=HARMONIC_N_PARTIALS,
                      rolloff=HARMONIC_ROLLOFF)
                  for b in np.arange(0.95, 1.1501, 0.01)],
                [f'{b:.2f}' for b in np.arange(0.95, 1.1501, 0.01)], 'beta',
                REF_HARM, 'inh'))
sweeps.append(('Inh: slot weights rho',
                [dict(beta=INHARMONIC_BETA, n_partials=HARMONIC_N_PARTIALS, rolloff=r)
                  for r in RHO_SWEEP],
                [f'{r:.1f}' for r in RHO_SWEEP], 'rho',
                REF_HARM, 'inh'))

sweep_data = []
for (name, kw_list, labels, axname, ref_F, base_tag) in sweeps:
    data = collect_sweep(kw_list, ref_F, labels)
    sweep_data.append((name, labels, axname, data, base_tag))
    print(f'  {name}: {len(kw_list)} steps done')

# Plot as 6 rows x 2 cols
fig, axes = plt.subplots(6, 2, figsize=(12, 20))
for i, (name, labels, axname, data, base_tag) in enumerate(sweep_data):
    profs_D, profs_F, pk_D, pk_F, pk_abs, mus = data
    axD, axF = axes[i, 0], axes[i, 1]
    for ax, profs, pks, title in ((axD, profs_D, pk_D, 'D (default)'),
                                     (axF, profs_F, pk_F, 'F (fixed)')):
        # Heatmap: x=offset, y=sweep steps
        im = ax.imshow(profs, aspect='auto', origin='lower',
                        extent=(OFFSET_GRID[0], OFFSET_GRID[-1], 0, len(labels)),
                        cmap='viridis')
        # Overlay peak offset
        y_centres = np.arange(len(labels)) + 0.5
        ax.plot(pks, y_centres, 'r.-', lw=0.7, ms=3)
        ax.axvline(0, color='white', lw=0.4, linestyle=':')
        if i == 0:
            ax.set_title(title, fontsize=10)
        # Thin out y-tick labels for legibility
        every = max(1, len(labels) // 12)
        ax.set_yticks(y_centres[::every])
        ax.set_yticklabels(labels[::every], fontsize=7)
        if i == 5:
            ax.set_xlabel('offset delta (cents)')
    axD.set_ylabel(f'{name}\n\n{axname}', fontsize=9)
fig.suptitle('Figure 2. Similarity-profile heatmaps across sweeps. '
              'Rows: sweep steps. Columns: reference method. Red line: peak offset.',
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(os.path.dirname(__file__),
                          'demo_windowing_reference_fig2.png'), dpi=110)
plt.close(fig)


# ============================================================
# Figure 3 -- Decomposing peak offsets: peak_abs, mu_q, and their
# difference under each reference method
# ============================================================
# Peak offset under default = peak_abs - mu_q.
# Peak offset under fixed   = peak_abs - ref_fixed (= REF_HARM's pitch).
# Plotting peak_abs and mu_q side by side makes it visible which
# quantity moves with the sweep parameter and which stays put.
print('Rendering Figure 3: peak_abs and mu_q vs sweep parameter')

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
axis_titles = ['slot count N', 'slot values beta', 'slot weights rho']
ref_fixed_pitch = float(REF_HARM[0][0])

for col, tag in enumerate(['harm', 'inh']):
    for row in range(3):
        idx = (0 if tag == 'harm' else 3) + row
        name, labels, axname, data, base_tag = sweep_data[idx]
        _, _, pk_D, pk_F, pk_abs, mus = data
        ax = axes[row, col]
        xvals = np.arange(len(labels))
        # Plot mu_q and peak_abs as two curves, plus the fixed
        # reference as a horizontal line.
        ax.plot(xvals, mus, 'o-', color='C2', ms=3, label='mu_q (query centroid)')
        ax.plot(xvals, pk_abs, 's-', color='C3', ms=3, label='P* (peak in abs pitch)')
        ax.axhline(ref_fixed_pitch, color='C0', lw=0.8, linestyle='--',
                    label=f'ref_fixed = {ref_fixed_pitch:.0f}')
        # Annotate P* - mu_q at the last point for reference
        every = max(1, len(labels) // 10)
        ax.set_xticks(xvals[::every])
        ax.set_xticklabels(labels[::every], fontsize=7, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row == 0:
            ax.set_title(f'{"Harmonic" if tag == "harm" else "Non-harmonic"} baseline',
                          fontsize=10)
        if col == 0:
            ax.set_ylabel(f'{axis_titles[row]}\n\nabsolute pitch (cents)', fontsize=9)
        if row == 0 and col == 0:
            ax.legend(loc='lower right', fontsize=8)

fig.suptitle('Figure 3. Peak absolute pitch P* and query centroid mu_q vs sweep '
              'parameter.\n'
              'Peak offset under default = P* - mu_q; peak offset under '
              'fixed = P* - ref_fixed.',
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(os.path.dirname(__file__),
                          'demo_windowing_reference_fig3.png'), dpi=110)
plt.close(fig)


# ============================================================
# Figure 4 -- Harmonic-calibrated F on stretched sweep
# ============================================================
print('Rendering Figure 4: stretched sweep under harmonic-calibrated F')

BETA_FIG4 = np.arange(0.95, 1.1501, 0.01)
profs_F_cal = np.zeros((len(BETA_FIG4), len(OFFSET_GRID)))
pk_F_cal = np.zeros(len(BETA_FIG4))
pkval_F_cal = np.zeros(len(BETA_FIG4))
for i, b in enumerate(BETA_FIG4):
    dq = build_query(beta=b)
    p = profile(dq, dens_c, spec, OFFSET_GRID, reference=REF_HARM)
    profs_F_cal[i, :] = p
    j = int(np.argmax(p))
    pk_F_cal[i] = OFFSET_GRID[j]
    pkval_F_cal[i] = p[j]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
im = ax.imshow(profs_F_cal, aspect='auto', origin='lower',
                extent=(OFFSET_GRID[0], OFFSET_GRID[-1], BETA_FIG4[0], BETA_FIG4[-1]),
                cmap='viridis')
ax.set_xlabel('offset delta (cents)')
ax.set_ylabel('stretch parameter beta')
ax.axvline(700, color='red', linestyle='--', lw=1.0,
            label='+700 (P5 musical reference)')
ax.axvline(0, color='white', lw=0.4, linestyle=':')
ax.set_title('Profile heatmap under F calibrated to harmonic baseline', fontsize=10)
ax.legend(loc='upper right', fontsize=8)
fig.colorbar(im, ax=ax)

ax2 = axes[1]
ax2.plot(BETA_FIG4, pk_F_cal, 'o-', color='C0', ms=4, label='peak offset (F)')
ax2.axhline(700, color='red', lw=0.6, linestyle='--', label='+700 (P5)')
ax2.set_xlabel('stretch parameter beta')
ax2.set_ylabel('peak offset (cents)', color='C0')
ax2.tick_params(axis='y', labelcolor='C0')
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)

# On a twin axis, peak amplitude
ax2b = ax2.twinx()
ax2b.plot(BETA_FIG4, pkval_F_cal, 's-', color='C2', ms=3, alpha=0.7,
           label='peak value (F)')
ax2b.set_ylabel('peak similarity value', color='C2')
ax2b.tick_params(axis='y', labelcolor='C2')
ax2b.spines['top'].set_visible(False)

# Combined legend
lines_a, labels_a = ax2.get_legend_handles_labels()
lines_b, labels_b = ax2b.get_legend_handles_labels()
ax2.legend(lines_a + lines_b, labels_a + labels_b, loc='center right', fontsize=8)
ax2.set_title('Peak offset and value vs beta under F (harmonic-calibrated)', fontsize=10)

fig.suptitle(f'Figure 4. Stretched query swept 0.95 to 1.15, scored under F '
              f'calibrated to harmonic baseline (ref = {float(REF_HARM[0][0]):.0f} cents)',
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(os.path.dirname(__file__),
                          'demo_windowing_reference_fig4.png'), dpi=110)
plt.close(fig)

print()
print('Done. Four figures written alongside this script.')

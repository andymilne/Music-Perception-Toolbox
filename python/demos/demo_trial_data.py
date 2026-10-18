"""demo_trial_data.py

Adding toolbox predictors to a table of experimental trials.

Experimental data are usually stored in long format: one row per
observation, here one rating by one participant of one stimulus, with
the columns that identify the participant, describe the stimulus, and
record the response. The analyst wants, on every row, the perceptual
predictors of that row's stimulus, so that the table can go straight
into a statistical model. This demo takes such a table and adds three
predictors to it, as new columns:

  * spcs         the spectral pitch-class similarity (SPCS) of the
                 chord to the scale it was heard in -- a paired
                 measure, of the stimulus's two parts;
  * harmonicity  the chord's template harmonicity (Milne 2013) -- a
                 single-set measure, of the chord alone;
  * roughness    the chord's sensory roughness (Sethares 1993).

It then writes the table out for the statistical analysis.

The data
  A simulated experiment: 60 participants rate how well a chord fits a
  preceding scale, on a 0-100 slider. The stimuli cross three scales
  with six chord types (five triads and a dominant seventh) at twelve
  roots, 216 in all. For each scale and chord type, each participant
  hears a random third of the roots, so 72 stimuli, and hears each of
  them twice, once in each of two blocks: 144 trials. Each participant
  also has a Gold-MSI general sophistication score. The table has
  60 x 144 = 8640 rows. The ratings are generated from a known model
  (section 0), so that anyone who fits the model can check that it
  recovers the parameters. With your own data, section 0 is replaced
  by one line, trials = pd.read_csv('my_data.csv').

Repeated stimuli (section 2)
  Each stimulus occurs on 40 rows on average (twice for each of about
  20 participants), so each of its predictor values is needed many
  times over. The table's columns go straight in nonetheless: every
  toolbox feature that accepts a 2-D pitch matrix (one row per item,
  NaN-padded when the items differ in size) finds the repeated rows
  itself, computes each distinct one once, and returns one value per
  row.

Uses: sim_maet, template_harmonicity, and roughness (each batched,
      with spectral enrichment via its spectrum option), set_default.

Requires: pandas, matplotlib (pip install matplotlib)
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mpt

# The toolbox's one-time informational hints (which route a call took,
# and the like) are switched off for a tidy printout, and restored at
# the end.
prev_defaults = mpt.set_default(show_hints=False)

# ===================================================================
#  User-adjustable parameters
# ===================================================================

# SPCS: sigma = 10 cents, 16 harmonics with power-law rolloff rho = 1;
# r = 1, absolute, periodic at the octave.
sigma = 10
r, rel, per, period = 1, False, True, 1200
spec = ['harmonic', 16, 'powerlaw', 1]

# Reference pitch for roughness (Hz): pitch 0 cents is middle C.
f0 = 261.63

# Where the finished table is written.
out_file = Path(tempfile.gettempdir()) / 'trialData.csv'


# ===================================================================
#  0. Simulating the experiment (a stand-in for reading your data)
#
#  This section only manufactures a plausible data table, and can be
#  skipped on a first reading. The ratings are generated from the
#  model that the analysis would fit,
#
#    rating ~ (spcs + harmonicity + roughness) * goldMsi
#             + (1 + spcs + harmonicity + roughness | participant),
#
#  with every predictor standardized (z-scored), and with these true
#  values (rating points per SD of the predictor):
#
#    fixed effects   intercept 50, spcs 8, harmonicity 4,
#                    roughness -3, goldMsi 0, spcs:goldMsi 3,
#                    harmonicity:goldMsi 1, roughness:goldMsi -1
#    participants    SDs of intercept 8, spcs 3, harmonicity 2,
#                    roughness 2 (uncorrelated)
#    residual        SD 10
#
#  So musically sophisticated listeners are simulated as weighting
#  spectral fit more heavily. These values are illustrative, not
#  empirical. Ratings are clipped to the slider's range, which affects
#  about 0.3% of them.
# ===================================================================

def simulate_experiment(n_participants=60, seed=1):
    rng = np.random.default_rng(seed)

    # --- The stimuli: scale x chord type x root ---------------------
    scales = {'diatonic':       [0, 200, 400, 500, 700, 900, 1100],
              'harmonic minor': [0, 200, 300, 500, 700, 800, 1100],
              'melodic minor':  [0, 200, 300, 500, 700, 900, 1100]}
    chords = {'major': [0, 400, 700],   'minor': [0, 300, 700],
              'dim':   [0, 300, 600],   'aug':   [0, 400, 800],
              'sus4':  [0, 500, 700],   'dom7':  [0, 400, 700, 1000]}
    rows = []
    for s_name, s in scales.items():
        for c_name, c in chords.items():
            for root in range(0, 1200, 100):
                chord = np.full(4, np.nan)          # NaN-padded to 4
                chord[:len(c)] = np.add(c, root)
                rows.append((s_name, c_name, root, s, chord))
    stim = pd.DataFrame(rows, columns=['scaleName', 'chordName', 'root',
                                       'scale', 'chord'])
    scale_mat = np.vstack(stim['scale'])
    chord_mat = np.vstack(stim['chord'])

    # --- The stimuli's predictor values -----------------------------
    x = np.column_stack([
        mpt.sim_maet(scale_mat, None, chord_mat, None, sigma, r, rel,
                     per, period, spectrum=spec, verbose=False),
        mpt.template_harmonicity(chord_mat, None, sigma,
                                 chord_spectrum=spec, verbose=False)[0],
        mpt.roughness(f0 * 2.0 ** (chord_mat / 1200.0), spectrum=spec)])
    z = (x - x.mean(0)) / x.std(0, ddof=1)

    # --- The participants -------------------------------------------
    # Gold-MSI general sophistication (range 18-126; mean and SD near
    # the published norms).
    gold = np.clip(np.round(rng.normal(82, 21, n_participants)), 18, 126)
    gold_z = (gold - gold.mean()) / gold.std(ddof=1)
    b_main = np.array([8.0, 4.0, -3.0])             # main effects
    b_gold = np.array([3.0, 1.0, -1.0])             # x goldMsi
    u0 = rng.normal(0, 8, n_participants)           # random intercepts
    u = rng.normal(0, 1, (n_participants, 3)) * [3, 2, 2]  # random slopes

    # --- One row per rating ------------------------------------------
    # The stimuli come in cells of 12 roots (one cell per scale and
    # chord type, root varying fastest). Each participant hears a
    # random 4 of the 12 roots in every cell, and hears that set twice,
    # in a fresh random order in each of two blocks.
    n_cells = len(stim) // 12
    out = []
    for j in range(n_participants):
        heard = np.concatenate([c * 12 + rng.choice(12, 4, replace=False)
                                for c in range(n_cells)])
        slopes = b_main + b_gold * gold_z[j] + u[j]
        for blk in (1, 2):
            order = rng.permutation(heard)
            n = len(order)
            rating = (50 + u0[j] + z[order] @ slopes
                      + rng.normal(0, 10, n))
            b = stim.iloc[order].reset_index(drop=True)
            b.insert(0, 'trial', (blk - 1) * n + np.arange(1, n + 1))
            b.insert(0, 'block', blk)
            b.insert(0, 'goldMsi', gold[j])
            b.insert(0, 'participant', j + 1)
            b['rating'] = np.clip(np.round(rating, 1), 0, 100)
            out.append(b)
    t = pd.concat(out, ignore_index=True)

    # As exported by most experiment software: one column per note,
    # empty where a chord has fewer notes.
    t[[f'scale_{k}' for k in range(1, 8)]] = np.vstack(t['scale'])
    t[[f'chord_{k}' for k in range(1, 5)]] = np.vstack(t['chord'])
    t['rating'] = t.pop('rating')                   # response last
    return t.drop(columns=['scale', 'chord'])


# ===================================================================
#  1. The data table
# ===================================================================

trials = simulate_experiment()   # with real data: pd.read_csv(...)

print(f"=== 1. The data: {len(trials)} rows, one per rating ===\n")
print(trials.head(6).to_string(index=False))

# The pitch columns, as matrices with one row per trial. Empty cells
# (a triad's fourth note) arrive as NaN, which is the padding the
# toolbox expects.
scale_mat = trials.filter(like='scale_').to_numpy(dtype=float)
chord_mat = trials.filter(like='chord_').to_numpy(dtype=float)

# ===================================================================
#  2. Computing the predictors for every row
# ===================================================================

print("\n=== 2. Predictors for every row ===\n")

# One call per predictor, on all 8640 rows. Each feature computes only
# its distinct rows, and decides what counts as distinct from what its
# value depends on: SPCS by the (scale, chord) pair, harmonicity -- a
# transposition-invariant measure of the chord alone -- by chord type
# regardless of root. So the harmonicity call below computes six
# chords, not 8640.
spcs = mpt.sim_maet(scale_mat, None, chord_mat, None, sigma, r, rel,
                    per, period, spectrum=spec, verbose=False)
harm, _ = mpt.template_harmonicity(chord_mat, None, sigma,
                                   chord_spectrum=spec, verbose=False)

# Roughness takes frequencies in Hz, so the chord columns are converted
# first (pitch 0 cents is middle C; NaN padding stays NaN). Roughness
# depends on absolute pitch, so only repeated chords are merged: the
# call below computes 72 chords (6 types at 12 roots).
chord_hz = f0 * 2.0 ** (chord_mat / 1200.0)
rough = mpt.roughness(chord_hz, spectrum=spec)

# Add the predictors to the table.
trials['spcs'] = spcs
trials['harmonicity'] = harm
trials['roughness'] = rough
print(f"  spcs, harmonicity, and roughness added to all {len(trials)} "
      f"rows.")

# ===================================================================
#  3. Checks before modelling
# ===================================================================

print("\n=== 3. Checks before modelling ===\n")

# The predictors' correlations over the stimuli (the first row of each
# stimulus, so that each counts once). Harmonicity and roughness are
# both properties of the chord, and are expected to be correlated.
pred_names = ['spcs', 'harmonicity', 'roughness']
stim_rows = trials.drop_duplicates(['scaleName', 'chordName', 'root'])
c = stim_rows[pred_names].corr().to_numpy()
print("  Predictor correlations (over stimuli):")
print("    " + " " * 12 + "".join(f"{n:>12s}" for n in pred_names))
for i, n in enumerate(pred_names):
    print(f"    {n:<12s}" + "".join(f"{v:12.2f}" for v in c[i]))

# Per participant: how closely ratings follow SPCS, against Gold-MSI.
# This is descriptive, not the analysis, but it shows the simulated
# interaction: steeper for more sophisticated listeners.
gold, r_spcs = np.array(
    [(d['goldMsi'].iloc[0], np.corrcoef(d['spcs'], d['rating'])[0, 1])
     for _, d in trials.groupby('participant')]).T
print(f"\n  Per-participant r(rating, SPCS) correlates "
      f"{np.corrcoef(gold, r_spcs)[0, 1]:.2f} with Gold-MSI.")

# ===================================================================
#  4. Writing the table for the analysis
#
#  The model itself is fitted in any mixed-effects package. In R, with
#  lme4, and the predictors standardized as in the simulation:
#
#    d <- read.csv("trialData.csv")
#    v <- c("spcs", "harmonicity", "roughness", "goldMsi")
#    d[v] <- scale(d[v])
#    m <- lmer(rating ~ (spcs + harmonicity + roughness) * goldMsi +
#                (1 + spcs + harmonicity + roughness | participant),
#              data = d)
#
#  The fixed effects should come out near the true values in section
#  0. (In Python, statsmodels' MixedLM fits the same model.)
# ===================================================================

trials.to_csv(out_file, index=False)
print(f"\n=== 4. Table written to {out_file} ===")

# ===================================================================
#  Plot
# ===================================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) Mean rating of each stimulus against its SPCS, by chord type.
means = (trials.groupby(['scaleName', 'chordName', 'root'])
         [['spcs', 'rating']].mean().reset_index())
for name, g in means.groupby('chordName', sort=False):
    axes[0].plot(g['spcs'], g['rating'], 'o', ms=4, label=name)
axes[0].set_xlabel('SPCS')
axes[0].set_ylabel('mean rating')
axes[0].set_title('(a) Each stimulus, averaged over participants')
axes[0].legend(fontsize=8)

# (b) Each participant's rating-SPCS correlation against Gold-MSI.
axes[1].plot(gold, r_spcs, 'o')
axes[1].set_xlabel('Gold-MSI general sophistication')
axes[1].set_ylabel('r(rating, SPCS)')
axes[1].set_title('(b) Each participant')
fig.tight_layout()

mpt.set_default(**prev_defaults)
print("\nDone.")
plt.show()

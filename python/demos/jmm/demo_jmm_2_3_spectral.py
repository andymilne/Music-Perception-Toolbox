"""demo_jmm_2_3_spectral.py — Analysis 2.3 (Online Supplement, Section 8.2):
windowed similarity of the motif, with spectral augmentation.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (Acknowledgement read from a MIDI transcription you
supply); the figures stay on screen unless SAVE_FIGURES is set.

Analysis 2.3: windowed similarity of the "A Love Supreme" motif across
Coltrane's *Acknowledgement*, with and without spectral augmentation.

Analyses 2.1 and 2.2 recover the motif from the passage. Here it is
supplied instead, as a query, and slid along the time axis: at each
position it is compared with the passage in the surrounding window.
Query and passage are encoded alike, as bound super-events of four
consecutive notes, with two attributes bound at order 4 — the four-note
pitch pattern and the group's four onsets, that is its rhythm. Onset
time is taken relative, and is then either dropped from the comparison,
in which case it only places the window on the group's first onset, or
compared, demanding the motif's rhythm as well.

Four readings of the same cross-correlation:

  A1, A2  relative pitch, time dropped: a transposition-invariant
          similarity against time. Every literal statement scores 1, in
          whatever key it is played.
  B1, B2  absolute pitch, time dropped: a pitch-offset by time map, each
          statement resolving at the offset of its transposition.

The fundamental readings (A1, B1) carry one pitch per note. The spectral
readings (A2, B2) replace each pitch by its twelve harmonic partials —
the hth at p + 12 log2 h semitones, weight h^-rho — an inner multiset
within the ordered four, which adds graded harmonic affinity between
related transpositions. ``add_spectra`` takes the pre-MAET and expands
the pitch attribute of every event at once, so the two readings differ
by one line of the encoding.

A fifth reading, not in the article's figure, compares the onset pattern
rather than dropping it. The motif's single early statement, long before
the closing run, states its pitches in an unrelated rhythm, so under A1
it scores a full match — a false positive of a pitch-only comparison.
Comparing the rhythm as well sends it to nearly zero while leaving the
closing run alone; this is the same call with
``drop_window_attr=False``.

Pre-MAET structure::

    attribute  order   sigma      rel       per
    ---------  ------  ---------  --------  ---
    pitch      4       0.15 st    0 or 1    no    fundamental
    pitch      (1, 4)  0.15 st    (0, 0/1)  no    spectral
    onset      4       0.125 QN   yes       no    dropped (A1, A2, B1, B2)

    Ordered (exch = 0) at the outer level, the spectral inner multiset
    unordered. Estimator: windowed similarity, rectangular window of 0.6
    QN full support, one-sided normalization.

Data: ``jmm_data.acknowledgement`` (the solo, from your own MIDI
transcription at ``data/AwakeningSolo.mid``). Toolbox:
``pre_maet_from_attr_table``, ``add_spectra``, ``bind_events``,
``windowed_similarity``. Runtime: a few minutes (about four on two
cores), three of them the two spectral panels; the spectral pitch-offset
sweep alone is some 80,000 windowed comparisons of twelve-partial
super-events.
"""
import os
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import PowerNorm
except ImportError:
    plt = None

import mpt
# The kernels here are narrow (0.15 semitones), so the article truncates
# them at four standard deviations rather than the toolbox's six: past
# 0.6 semitones the kernel bears on nothing musical, and the spectral
# sweeps are the heaviest calls in these demos.
_prev_defaults = mpt.set_default(show_hints=False, truncation_sigmas=4.0)
from mpt import (pre_maet_from_attr_table, add_spectra, bind_events,
                 windowed_similarity, show_pre_maet, unpack_pre_maet)

import jmm_data

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

SIGMA_PITCH  = 0.15     # semitones (15 cents): the pitch-matching tolerance
SIGMA_TIME   = 0.125    # QN: the onset-matching tolerance
N_PART       = 12       # harmonic partials per note (spectral readings)
RHO          = 0.67     # power-law roll-off (Milne et al. 2015)
WIN          = 0.6      # full support of the rectangular time window (QN)
Q_ROOT       = 56       # query root (MIDI); offset 0 reads as this root
ALS_IV       = np.array([0, 3, 0, 5])          # the motif, from its root
MOTIF_ONSETS = np.array([0.0, 0.5, 1.5, 2.0])  # its rhythm: (0.5, 1.0, 0.5) QN
CENTRE_STEP  = 0.5      # QN between window centres; below the window's support,
                        # so every position of the passage is covered
OFFSETS      = np.arange(-14.0, 14.0 + 1e-9, 0.5)   # semitones, for B1 and B2
BEATS_PER_BAR = 4.0     # 4/4 throughout

C_FUND = '#1f4eb8'
C_SPEC = '#c25008'


# --- the passage, the query, and the window centres ------------------------
notes = jmm_data.acknowledgement()
onset = notes['onset_beats'].to_numpy(dtype=float)
centres = np.arange(onset.min(), onset.max() + 1e-9, CENTRE_STEP)
print(f'{len(notes)} notes; span {onset.max():.1f} QN; '
      f'{centres.size} window centres')

# The query is a four-note score carrying the motif's own rhythm, and goes
# through the same steps as the passage below. Where onset time is dropped
# from the comparison that rhythm only fixes the window's placement on the
# query's first onset; where it is compared it is the rhythm the passage
# must match.
query_notes = pd.DataFrame({'pitch': float(Q_ROOT) + ALS_IV,
                            'onset_beats': MOTIF_ONSETS})

# One conversion each, pitch and onset, one event per note.
ATTRIBUTES = (dict(column='pitch', name='pitch', sigma=SIGMA_PITCH),
              dict(column='onset', name='onset', sigma=SIGMA_TIME))
passage = pre_maet_from_attr_table(notes, attributes=ATTRIBUTES, time='beats',
                                   chords='separate', weights='ones')
query = pre_maet_from_attr_table(query_notes, attributes=ATTRIBUTES,
                                 time='beats', chords='separate',
                                 weights='ones')

# The spectral reading replaces each pitch by its N_PART harmonic partials,
# an inner multiset within the note. This one call is the whole difference
# between the fundamental panels and the spectral ones.
passage_spectral = add_spectra(passage, 'harmonic', N_PART, 'powerlaw', RHO,
                               attribute='pitch', units=12.0)
query_spectral = add_spectra(query, 'harmonic', N_PART, 'powerlaw', RHO,
                             attribute='pitch', units=12.0)

# Four consecutive notes bound into one super-event, step = 1 advancing the
# group by one note at a time. Both attributes are bound at order 4: the
# four-note pitch pattern, and the group's four onsets, that is its rhythm.
# The relative panels take both attributes relative, so any transposition
# of the motif matches; the absolute panels take pitch absolute, so that
# each statement resolves at the offset of its own transposition, and leave
# onset relative.
RELATIVE = [True, True]        # pitch relative, onset relative
ABSOLUTE_PITCH = [False, True]  # pitch absolute, onset relative

ctx_rel_fund = bind_events(passage, [4, 4], step=1, rel_outer=RELATIVE)
ctx_rel_spec = bind_events(passage_spectral, [4, 4], step=1,
                           rel_outer=RELATIVE)
ctx_abs_fund = bind_events(passage, [4, 4], step=1, rel_outer=ABSOLUTE_PITCH)
ctx_abs_spec = bind_events(passage_spectral, [4, 4], step=1,
                           rel_outer=ABSOLUTE_PITCH)
qry_rel_fund = bind_events(query, [4, 4], step=1, rel_outer=RELATIVE)
qry_rel_spec = bind_events(query_spectral, [4, 4], step=1, rel_outer=RELATIVE)
qry_abs_fund = bind_events(query, [4, 4], step=1, rel_outer=ABSOLUTE_PITCH)
qry_abs_spec = bind_events(query_spectral, [4, 4], step=1,
                           rel_outer=ABSOLUTE_PITCH)

show_pre_maet(qry_rel_fund, max_events=1)
show_pre_maet(qry_rel_spec, max_events=1, decimals=2)

# --- A1 and A2: transposition-invariant similarity against time -----------
# The swept axis is time, attribute 1: a rectangular window of full support
# WIN slides over the passage. Onset time is dropped from the comparison
# (drop_window_attr=True), so it only places the window, on the group's
# first onset (locate='start'), which lands each peak on the statement's
# onset. Pitch is then the sole compared attribute.
print('computing A1 (fundamental, relative) ...')
A1 = np.asarray(windowed_similarity(
    ctx_rel_fund, qry_rel_fund, centres,
    context_window=('rect', WIN), window_attr=1, drop_window_attr=True,
    locate='start', normalize='oneSidedDenom')).ravel()
print('computing A2 (spectral, relative) ...')
A2 = np.asarray(windowed_similarity(
    ctx_rel_spec, qry_rel_spec, centres,
    context_window=('rect', WIN), window_attr=1, drop_window_attr=True,
    locate='start', normalize='oneSidedDenom')).ravel()

# --- B1 and B2: pitch offset by time --------------------------------------
# One multi-axis sweep slides the query over both attributes at once: pitch
# (axis 0) over the transposition offsets, compared (drop=False); time (axis
# 1) over the window centres, carrying the window and dropped, exactly as in
# A1 and A2. The pitch positions start from the query's own pitch centroid,
# so that offset 0 reads as the untransposed query.
q_pitch_fund = float(np.nanmean(unpack_pre_maet(qry_abs_fund)[0][0]))
q_pitch_spec = float(np.nanmean(unpack_pre_maet(qry_abs_spec)[0][0]))
print('computing B1 (fundamental, absolute) ...')
B1 = np.asarray(windowed_similarity(
    ctx_abs_fund, qry_abs_fund,
    sweep={0: q_pitch_fund + OFFSETS, 1: centres},
    drop={0: False, 1: True},
    context_window={1: {'shape': 'rect', 'width': WIN}},
    locate={1: 'start'}, normalize='oneSidedDenom'))
print('computing B2 (spectral, absolute) ...')
B2 = np.asarray(windowed_similarity(
    ctx_abs_spec, qry_abs_spec,
    sweep={0: q_pitch_spec + OFFSETS, 1: centres},
    drop={0: False, 1: True},
    context_window={1: {'shape': 'rect', 'width': WIN}},
    locate={1: 'start'}, normalize='oneSidedDenom'))

for tag, panel in [('A1', A1), ('A2', A2), ('B1', B1), ('B2', B2)]:
    print(f'{tag}: max {panel.max():.3f}; cells above 0.99: '
          f'{int((panel > 0.99).sum())}; above 0.005: {int((panel > 0.005).sum())}')
print(f'A1 vs A2 correlation: {np.corrcoef(A1, A2)[0, 1]:.4f}; '
      f'max|A2 - A1| = {np.max(np.abs(A2 - A1)):.3f}')

# --- the rhythm-aware reading ---------------------------------------------
# A1's call again, with the onset attribute compared rather than dropped
# (drop_window_attr=False): a match must now reproduce the motif's rhythm,
# which the query carries, as well as its pitch pattern.
Aj = np.asarray(windowed_similarity(
    ctx_rel_fund, qry_rel_fund, centres,
    context_window=('rect', WIN), window_attr=1, drop_window_attr=False,
    locate='start', normalize='oneSidedDenom')).ravel()
early = centres < 250.0
at = float(centres[early][A1[early].argmax()])
print(f'\nthe early statement, bar {at / BEATS_PER_BAR:.0f}: pitch-only '
      f'match {A1[early].max():.3f}, match with the rhythm compared '
      f'{Aj[early].max():.3f}')
print(f'whole passage: {int((A1 > 0.99).sum())} unit matches on pitch alone, '
      f'{int((Aj > 0.99).sum())} with the rhythm compared')

# --- figure ---------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    mpt.set_default(**_prev_defaults)
    raise SystemExit(0)

plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
bars = centres / BEATS_PER_BAR
fig = plt.figure(figsize=(13, 6.2))
gs = GridSpec(2, 3, width_ratios=[1, 1, 0.035], height_ratios=[1, 1.45],
              hspace=0.16, wspace=0.07)
axA1 = fig.add_subplot(gs[0, 0])
axA2 = fig.add_subplot(gs[0, 1], sharey=axA1)
axB1 = fig.add_subplot(gs[1, 0], sharex=axA1)
axB2 = fig.add_subplot(gs[1, 1], sharex=axA2, sharey=axB1)
cax = fig.add_subplot(gs[1, 2])

for ax, panel, title, colour in [(axA1, A1, 'A1  fundamental, relative', C_FUND),
                                 (axA2, A2, 'A2  spectral, relative', C_SPEC)]:
    ax.vlines(bars, 0, panel, color=colour, lw=0.8)
    ax.set_ylim(0, max(1.05, 1.05 * max(A1.max(), A2.max())))
    ax.set_title(title)
axA1.set_ylabel('similarity')

# Each statement is a single narrow cell against a several-hundred-bar
# axis, so the absolute panels are drawn as a scatter of the non-zero
# cells; a mesh would render them sub-pixel and invisible. The gamma-0.5
# colour scale lifts the low spectral matches.
TG, OG = np.meshgrid(bars, OFFSETS)
norm = PowerNorm(gamma=0.5, vmin=0, vmax=max(B1.max(), B2.max()))
NARROW = [(-0.5, -1), (0.5, -1), (0.5, 1), (-0.5, 1)]   # half-width marker
for ax, panel, title in [(axB1, B1, 'B1  fundamental, absolute'),
                         (axB2, B2, 'B2  spectral, absolute')]:
    ax.set_facecolor('black')
    nz = panel > 0.005
    sc = ax.scatter(TG[nz], OG[nz], c=panel[nz], cmap='magma', norm=norm,
                    s=11, marker=NARROW, edgecolors='none')
    ax.set_ylim(OFFSETS.min(), OFFSETS.max())
    ax.set_xlim(bars.min(), bars.max())
    ax.set_title(title)
    ax.set_xlabel('bar')
axB1.set_ylabel('pitch offset from query root (st)')
axB1.set_yticks(np.arange(-12, 13, 3))
fig.colorbar(sc, cax=cax)
for ax in (axA1, axA2):
    plt.setp(ax.get_xticklabels(), visible=False)
for ax in (axA2, axB2):
    plt.setp(ax.get_yticklabels(), visible=False)
fig.suptitle('Coltrane, Acknowledgement: windowed similarity of the '
             f'A Love Supreme motif (rho = {RHO}, per = 0)', y=0.97, fontsize=14)

if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'demo_jmm_2_3_spectral.png'),
                dpi=160, bbox_inches='tight')
    print('Saved figures/demo_jmm_2_3_spectral.png')
else:
    plt.show()

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)

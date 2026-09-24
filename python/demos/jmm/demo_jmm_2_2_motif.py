"""demo_jmm_2_2_motif.py — Analysis 2.2 (Online Supplement, Section 8.1):
motif discovery in Acknowledgement from pitch alone.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (Acknowledgement read from a MIDI transcription you
supply); the figures stay on screen unless SAVE_FIGURES is set.

Analysis 2.2: data-driven motif discovery in Coltrane's *Acknowledgement*.

The melody is read as one event per note carrying pitch. The recurring
four-note cell is then recovered as a peak in the density of short
interval patterns, without being supplied in advance. Two routes reach
the same transposition-invariant cell.

Differenced route. Pitch is first-differenced to melodic intervals,
collapsing every transposition of a figure onto the same interval
pattern. Three consecutive intervals are then bound into an ordered
super-event: each is a point in a three-dimensional interval space, and
the density over those points is a smoothed recurrence count. A figure
stated k times contributes k near-coincident points, so its interval
triple stands out as a local maximum. Reading the density at every
observed triple and ranking turns motif discovery into peak finding.

Relative route. The same cell is reached without differencing, by
binding four consecutive pitches into an ordered super-event taken
relative: the common transposition is removed, so transposed statements
again coincide, and three degrees of freedom remain — the dimension of
the differenced route's interval triple. The two densities therefore
describe the same object through different internal metrics. The
differenced form treats consecutive intervals independently; the
relative form couples them through the pitch they share. With the
interval kernel set to sqrt(2) times the per-pitch kernel — an interval
being a difference of two pitches, which is the scaling
``difference_events`` applies of its own accord — the two rank the
motifs identically, and differ only in that coupling: a faint shear in
the relative density's slice, absent from the differenced one.

A relative density is read in translation-reduced coordinates: an
r-tuple minus its first value, so a cell's coordinates are its
cumulative intervals. Both the cells and the slice grid are written that
way for the relative route below.

Pre-MAET structure::

    attribute    order  sigma              rel  per
    ----------   -----  -----------------  ---  ---
    dp           3      sqrt(2) * 0.15 st  no   no     differenced route
    pitch        4      0.15 st            yes  no     relative route

    Ordered (exch = 0) in both. Estimator: the density read at each cell.

Data: ``jmm_data.acknowledgement`` (the solo, from your own MIDI
transcription at ``data/AwakeningSolo.mid``). Toolbox:
``pre_maet_from_attr_table``, ``difference_events``, ``bind_events``,
``build_maet``, ``eval_maet``, ``show_pre_maet``. Runtime: a few seconds.
"""
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
plt.rcParams.update({'font.size': 17, 'axes.titlesize': 19, 'axes.labelsize': 17,
                     'xtick.labelsize': 15, 'ytick.labelsize': 15, 'figure.titlesize': 22,
                     'font.family': 'DejaVu Sans'})

import mpt
_prev_defaults = mpt.set_default(show_hints=False)
from mpt import (pre_maet_from_attr_table, difference_events, bind_events,
                 build_maet, eval_maet, show_pre_maet, unpack_pre_maet)

import jmm_data

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

SIGMA_PITCH = 0.15         # semitones (15 cents): the per-pitch uncertainty
R_DIFF      = 3            # bound interval triples (a four-note cell)
R_REL       = 4            # bound pitch quadruples (the same cell)
TOP         = 6            # motifs shown in the ranking
ALS         = (3, -3, 5)   # +m3, -m3, +P4: the "A Love Supreme" cell

C_ALS   = '#c25008'        # the recurring "A Love Supreme" cell
C_OTHER = '#1f4eb8'        # the surrounding recurring cells


def contour(triple):
    """A cell's interval triple as the scale degrees it traces from 0."""
    return '→'.join(str(int(v)) for v in np.cumsum([0, *triple]))


def rank(intervals, density):
    """Group cells by their interval triple and rank the classes.

    The density at a cell already equals the class's recurrence — every
    member of a class sits at the same point, so each sees all c copies.
    Summing over the members would square that, so the class mean is
    what recovers the count.
    """
    keys, inv = np.unique(np.round(intervals).astype(int), axis=0,
                          return_inverse=True)
    inv = np.asarray(inv).ravel()
    count = np.bincount(inv)
    mean = np.bincount(inv, weights=density) / count
    order = np.argsort(-mean)
    return [(tuple(int(v) for v in keys[i]), int(count[i]), float(mean[i]))
            for i in order]


# --- the melody as a pre-MAET, one event per note -------------------------
notes = jmm_data.acknowledgement()
print(f'{len(notes)} notes; pitch range {notes["pitch"].min():.0f}-'
      f'{notes["pitch"].max():.0f} (MIDI); span '
      f'{notes["onset_beats"].max():.1f} QN')
melody = pre_maet_from_attr_table(
    notes, attributes=(dict(column='pitch', name='pitch', sigma=SIGMA_PITCH),),
    time='beats', chords='separate', weights='ones')

# --- differenced route: intervals bound into ordered triples ---------------
# difference_events widens the kernel by sqrt(2) itself, so the bound
# pre-MAET already carries the interval width and build_maet needs no sigma.
diff_route = bind_events(difference_events(melody, 1), R_DIFF, step=1)
show_pre_maet(diff_route, max_events=3)
cells_d = unpack_pre_maet(diff_route)[0][0]              # (3, nCells) intervals
dens_d = np.asarray(eval_maet(build_maet(diff_route, verbose=False),
                              cells_d, verbose=False))
ranked_d = rank(cells_d.T, dens_d)

# --- relative route: pitches bound into ordered relative quadruples -------
rel_route = bind_events(melody, R_REL, step=1, rel_outer=True)
show_pre_maet(rel_route, max_events=3)
cells_r = unpack_pre_maet(rel_route)[0][0]               # (4, nCells) pitches
iv_r = np.diff(cells_r, axis=0)                          # its interval triples
dens_r = np.asarray(eval_maet(build_maet(rel_route, verbose=False),
                              np.cumsum(iv_r, axis=0), verbose=False))
ranked_r = rank(iv_r.T, dens_r)

# --- report ---------------------------------------------------------------
print(f'\ndifferenced: {cells_d.shape[1]} ordered interval triples, '
      f'{len(ranked_d)} distinct classes')
print(f'relative:    {cells_r.shape[1]} ordered pitch quadruples, '
      f'{len(ranked_r)} distinct classes\n')
print(f'{"rank":>4}  {"interval class":>16}  {"count":>5}  '
      f'{"differenced":>11}  {"relative":>9}   contour')
by_class = {k: d for k, _, d in ranked_r}
for i, (k, c, d) in enumerate(ranked_d[:TOP], 1):
    mark = '  <- A Love Supreme cell' if k == ALS else ''
    print(f'{i:>4}  {str(k):>16}  {c:>5}  {d:>11.1f}  {by_class[k]:>9.1f}   '
          f'{contour(k)}{mark}')
print(f'\ntop-{TOP + 2} interval-class set identical: '
      f'{set(k for k, _, _ in ranked_d[:TOP + 2]) == set(k for k, _, _ in ranked_r[:TOP + 2])}')
print(f'A Love Supreme cell {ALS}: rank '
      f'{1 + [k for k, _, _ in ranked_d].index(ALS)} (differenced), rank '
      f'{1 + [k for k, _, _ in ranked_r].index(ALS)} (relative)')

# --- the (+3, i2, i3) slice of each density -------------------------------
# The plane of cells sharing the leading motif's first interval. The grid
# is written as interval triples for the differenced route and as their
# cumulative sums for the relative one.
step = SIGMA_PITCH * np.sqrt(2.0) / 3.0
axis = np.arange(-7.0, 7.0 + 0.5 * step, step)
I2, I3 = np.meshgrid(axis, axis)
grid_iv = np.vstack([np.full(I2.size, float(ALS[0])), I2.ravel(), I3.ravel()])
slice_d = np.asarray(eval_maet(build_maet(diff_route, verbose=False),
                               grid_iv, verbose=False)).reshape(I2.shape)
slice_r = np.asarray(eval_maet(build_maet(rel_route, verbose=False),
                               np.cumsum(grid_iv, axis=0),
                               verbose=False)).reshape(I2.shape)
print(f'\nslice peak (i2, i3): differenced '
      f'({axis[np.unravel_index(slice_d.argmax(), slice_d.shape)[1]]:.2f}, '
      f'{axis[np.unravel_index(slice_d.argmax(), slice_d.shape)[0]]:.2f}), '
      f'relative '
      f'({axis[np.unravel_index(slice_r.argmax(), slice_r.shape)[1]]:.2f}, '
      f'{axis[np.unravel_index(slice_r.argmax(), slice_r.shape)[0]]:.2f})')

# --- figure ---------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    mpt.set_default(**_prev_defaults)
    raise SystemExit(0)

fig, axes = plt.subplots(2, 2, figsize=(15, 10.7),
                         gridspec_kw={'height_ratios': [3.7, 4.06]})


def draw_bars(ax, ranked, title):
    top = ranked[:TOP][::-1]
    yy = np.arange(len(top))
    ax.barh(yy, [d for _, _, d in top], height=0.56,
            color=[C_ALS if k == ALS else C_OTHER for k, _, _ in top])
    ax.set_yticks(yy)
    ax.set_yticklabels([contour(k) for k, _, _ in top], fontsize=13)
    for y, (_, c, d) in zip(yy, top):
        ax.text(d, y, f'  {d:.1f} ({c})', va='center', fontsize=14, color='#444')
    ax.set_xlabel('density (and count)')
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, max(d for _, _, d in top) * 1.20)
    ax.set_ylim(-0.7, len(top) - 0.3)


def draw_slice(ax, field, ranked, title):
    pc = ax.pcolormesh(axis, axis, field, cmap='magma', shading='gouraud')
    pc.set_rasterized(True)
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04, label='density')
    for k, _, d in ranked[:TOP]:
        if k[0] != ALS[0]:
            continue
        als = k == ALS
        ax.scatter(k[1], k[2], s=130 if als else 60, facecolors='none',
                   edgecolors=C_ALS if als else 'white',
                   linewidths=2.4 if als else 1.4, zorder=3)
        ax.annotate(('A Love Supreme  ' if als else '') + f'{d:.1f}',
                    (k[1], k[2]), textcoords='offset points', xytext=(7, 6),
                    fontsize=14, color=C_ALS if als else 'white',
                    fontweight='bold' if als else 'normal')
    ax.set_xlabel('second interval  $i_2$  (semitones)')
    ax.set_ylabel('third interval  $i_3$  (semitones)')
    ax.set_xticks(np.arange(-6, 7, 2))
    ax.set_yticks(np.arange(-6, 7, 2))
    ax.set_title(title)
    ax.set_aspect('equal')


draw_bars(axes[0, 0], ranked_d, 'Differenced interval triples')
draw_bars(axes[0, 1], ranked_r, 'Relative pitch quadruples')
draw_slice(axes[1, 0], slice_d, ranked_d,
           fr'Density through $(+{ALS[0]},\, i_2,\, i_3)$: differenced')
draw_slice(axes[1, 1], slice_r, ranked_r,
           fr'Density through $(+{ALS[0]},\, i_2,\, i_3)$: relative')
fig.suptitle('Coltrane, Acknowledgement: motif density', y=0.985, fontsize=15)
fig.subplots_adjust(left=0.115, right=0.93, top=0.90, bottom=0.075,
                    hspace=0.275, wspace=0.40)

if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'demo_jmm_2_2_motif.png'),
                dpi=140, bbox_inches='tight')
    print('Saved figures/demo_jmm_2_2_motif.png')
else:
    plt.show()

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)

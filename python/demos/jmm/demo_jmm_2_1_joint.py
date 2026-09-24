"""demo_jmm_2_1_joint.py — Analysis 2.1 (JMM article, Section 4.2.1): the
motif of Acknowledgement as a joint pitch-and-rhythm object.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article; lightly edited from the article's own script. Data come
from jmm_data (Acknowledgement read from a MIDI transcription you
supply); the figures stay on screen unless SAVE_FIGURES is set.

Analysis 2.1: the motif of Coltrane's *Acknowledgement* as a joint
object — the interval pattern together with the rhythm it is set in.

Analysis 2.2 recovers the four-note cell from pitch alone. Adding onset
time as a second attribute asks for agreement in pitch and rhythm at
once: the two attributes are joined by the tensor product, so the joint
density over interval and inter-onset-interval super-events ranks a
motif by both. Both routes of Analysis 2.2 carry over, and the
attributes are handled alike within each:

  differenced route  pitch to melodic intervals and onset time to
                     inter-onset intervals, three consecutive of each
                     bound into ordered super-events in step;
  relative route     four consecutive pitches and four consecutive onset
                     times bound into ordered super-events, each taken
                     relative.

The attributes need not be handled alike — pitch may be differenced
while onset time is taken relative, or the reverse, each attribute
carrying its own choice — but taking them in step keeps the two routes
comparable. The motif is set in one rhythm almost always: eighth,
quarter, eighth (0.5, 1.0, 0.5 QN), in 35 of its 36 statements, the
remaining one holding the first note long before two quick sixteenths
(2.5, 0.25, 0.25 QN). The joint motif is therefore sharply defined and
leads both routes.

Because the attributes are tensored, either may be marginalized — by
omitting it from the density, exactly so where its total mass per event
is constant, as here — or conditioned on, by fixing its coordinates in
the point at which the density is evaluated. The lower panels do both
for rhythm: the inter-onset-interval density with pitch marginalized
away, and the same density conditioned on the motif's intervals.

A relative density is read in translation-reduced coordinates: an
r-tuple minus its first value, so a cell's coordinates are its
cumulative intervals. Both the cells and the slice grids are written
that way for the relative route below.

Pre-MAET structure::

    attribute    order  sigma                rel  per
    ----------   -----  -------------------  ---  ---
    dp           3      sqrt(2) * 0.15 st    no   no    differenced route
    dt           3      sqrt(2) * 0.125 QN   no   no    differenced route
    pitch        4      0.15 st              yes  no    relative route
    onset        4      0.125 QN             yes  no    relative route

    Ordered (exch = 0) throughout; the two attributes are tensored.
    Estimator: the density read at each cell.

Data: ``jmm_data.acknowledgement`` (the solo, from your own MIDI
transcription at ``data/AwakeningSolo.mid``). Toolbox:
``pre_maet_from_attr_table``, ``difference_events``, ``bind_events``,
``select_pre_maet``, ``build_maet``, ``eval_maet``, ``show_pre_maet``.
Runtime: under a minute.
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
                 select_pre_maet, build_maet, eval_maet, show_pre_maet,
                 unpack_pre_maet)

import jmm_data

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

SIGMA_PITCH = 0.15            # semitones (15 cents): the per-pitch uncertainty
SIGMA_TIME  = 0.125           # QN (a thirty-second note): the per-onset one
R_DIFF      = 3               # bound interval triples (a four-note cell)
R_REL       = 4               # bound pitch quadruples (the same cell)
TOP         = 6               # motifs shown in the ranking
ALS_IV      = (3, -3, 5)      # +m3, -m3, +P4: the "A Love Supreme" cell
ALS_RHYTHM  = (0.5, 1.0, 0.5)  # its rhythm: eighth, quarter, eighth
ALS         = (ALS_IV, ALS_RHYTHM)
FIRST_IOI   = 0.5             # the motif's first IOI: the slice's fixed value

C_ALS   = '#c25008'           # the recurring "A Love Supreme" cell
C_OTHER = '#1f4eb8'           # the surrounding recurring cells


def contour(triple):
    """A cell's interval triple as the scale degrees it traces from 0."""
    return '→'.join(str(int(v)) for v in np.cumsum([0, *triple]))


def rhythm(triple):
    """A cell's inter-onset intervals in QN, compactly."""
    return '·'.join('%g' % v for v in triple)


def rank(intervals, iois, density):
    """Group cells by interval triple and rhythm, and rank the classes.

    The density at a cell already equals the class's recurrence — every
    member of a class sits at the same point, so each sees all c copies.
    Summing over the members would square that, so the class mean is
    what recovers the count. Inter-onset intervals are grouped to the
    sixteenth note, the passage's shortest written value.
    """
    labels = np.column_stack([np.round(intervals), np.round(iois * 4.0) / 4.0])
    keys, inv = np.unique(labels, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    count = np.bincount(inv)
    mean = np.bincount(inv, weights=density) / count
    order = np.argsort(-mean)
    return [((tuple(int(v) for v in keys[i][:3]), tuple(keys[i][3:])),
             int(count[i]), float(mean[i])) for i in order]


# --- the melody as a pre-MAET, one event per note -------------------------
notes = jmm_data.acknowledgement()
print(f'{len(notes)} notes; pitch range {notes["pitch"].min():.0f}-'
      f'{notes["pitch"].max():.0f} (MIDI); span '
      f'{notes["onset_beats"].max():.1f} QN')
melody = pre_maet_from_attr_table(
    notes, attributes=(dict(column='pitch', name='pitch', sigma=SIGMA_PITCH),
                       dict(column='onset', name='onset', sigma=SIGMA_TIME)),
    time='beats', chords='separate', weights='ones')

# --- differenced route: intervals and IOIs bound in step ------------------
# One bind_events call takes both attributes at order 3, so the two stay
# in step. difference_events widens each kernel by sqrt(2) itself — a
# difference of two values of width sigma has width sqrt(2) sigma — so the
# bound pre-MAET already carries both interval widths.
diff_route = bind_events(difference_events(melody, [1, 1]), [R_DIFF, R_DIFF],
                         step=1)
show_pre_maet(diff_route, max_events=2, decimals=3)
iv_d, ioi_d = [a for a in unpack_pre_maet(diff_route)[0]]
joint_d = build_maet(diff_route, verbose=False)
dens_d = np.asarray(eval_maet(joint_d, [iv_d, ioi_d], verbose=False))
ranked_d = rank(iv_d.T, ioi_d.T, dens_d)

# --- relative route: pitches and onsets bound, each taken relative --------
rel_route = bind_events(melody, [R_REL, R_REL], step=1, rel_outer=True)
show_pre_maet(rel_route, max_events=2, decimals=3)
pitch_r, onset_r = [a for a in unpack_pre_maet(rel_route)[0]]
iv_r, ioi_r = np.diff(pitch_r, axis=0), np.diff(onset_r, axis=0)
joint_r = build_maet(rel_route, verbose=False)
dens_r = np.asarray(eval_maet(
    joint_r, [np.cumsum(iv_r, axis=0), np.cumsum(ioi_r, axis=0)], verbose=False))
ranked_r = rank(iv_r.T, ioi_r.T, dens_r)

# --- report ---------------------------------------------------------------
print(f'\ndifferenced: {iv_d.shape[1]} cells, {len(ranked_d)} distinct '
      f'(interval, rhythm) classes')
print(f'relative:    {pitch_r.shape[1]} cells, {len(ranked_r)} distinct classes\n')
print(f'{"rank":>4}  {"interval class":>16}  {"rhythm (QN)":>13}  {"count":>5}  '
      f'{"differenced":>11}  {"relative":>9}')
by_class = {k: d for k, _, d in ranked_r}
for i, (k, c, d) in enumerate(ranked_d[:TOP], 1):
    mark = '  <- A Love Supreme motif' if k == ALS else ''
    print(f'{i:>4}  {str(k[0]):>16}  {rhythm(k[1]):>13}  {c:>5}  {d:>11.1f}  '
          f'{by_class[k]:>9.1f}{mark}')
stated = [(k[1], c) for k, c, _ in ranked_d if k[0] == ALS_IV]
print(f'\nthe cell {ALS_IV} is stated {sum(c for _, c in stated)} times, in '
      f'{len(stated)} rhythms: '
      + ', '.join(f'({rhythm(r)}) x {c}' for r, c in sorted(stated, key=lambda rc: -rc[1])))

# --- rhythm with pitch marginalized away, and conditioned on the motif ----
# Marginalizing is dropping the pitch attribute from the density (exact
# here, the mass per event being constant); conditioning is fixing its
# coordinates in the evaluation point.
time_d = build_maet(select_pre_maet(diff_route, attributes=['onset']),
                    verbose=False)
time_r = build_maet(select_pre_maet(rel_route, attributes=['onset']),
                    verbose=False)
step = SIGMA_TIME * np.sqrt(2.0) / 3.0
axis = np.arange(0.0, 2.5 + 0.5 * step, step)
T2, T3 = np.meshgrid(axis, axis)
grid_ioi = np.vstack([np.full(T2.size, FIRST_IOI), T2.ravel(), T3.ravel()])
grid_iv = np.tile(np.array(ALS_IV, dtype=float).reshape(3, 1), (1, T2.size))
marg_d = np.asarray(eval_maet(time_d, grid_ioi, verbose=False)).reshape(T2.shape)
cond_d = np.asarray(eval_maet(joint_d, [grid_iv, grid_ioi],
                              verbose=False)).reshape(T2.shape)
marg_r = np.asarray(eval_maet(time_r, np.cumsum(grid_ioi, axis=0),
                              verbose=False)).reshape(T2.shape)
cond_r = np.asarray(eval_maet(joint_r, [np.cumsum(grid_iv, axis=0),
                                        np.cumsum(grid_ioi, axis=0)],
                              verbose=False)).reshape(T2.shape)
for tag, field in [('marginalized, differenced', marg_d),
                   ('marginalized, relative', marg_r),
                   ('conditioned, differenced', cond_d),
                   ('conditioned, relative', cond_r)]:
    r, c = np.unravel_index(field.argmax(), field.shape)
    print(f'{tag:>26}: peak at (second, third) IOI = '
          f'({axis[c]:.2f}, {axis[r]:.2f}) QN')

# --- figure ---------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    mpt.set_default(**_prev_defaults)
    raise SystemExit(0)

fig, axes = plt.subplots(3, 2, figsize=(15, 15.9),
                         gridspec_kw={'height_ratios': [3.7, 4.06, 4.06]})
fig.subplots_adjust(left=0.135, right=0.93, top=0.925, bottom=0.05,
                    hspace=0.27, wspace=0.40)


def draw_bars(ax, ranked, title):
    top = ranked[:TOP][::-1]
    yy = np.arange(len(top))
    ax.barh(yy, [d for _, _, d in top], height=0.72,
            color=[C_ALS if k == ALS else C_OTHER for k, _, _ in top])
    ax.set_yticks(yy)
    ax.set_yticklabels([contour(k[0]) + '\n' + rhythm(k[1]) for k, _, _ in top],
                       fontsize=12)
    for y, (_, c, d) in zip(yy, top):
        ax.text(d, y, f'  {d:.1f} ({c})', va='center', fontsize=14, color='#444')
    ax.set_xlabel('density (and count)')
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, max(d for _, _, d in top) * 1.20)
    ax.set_ylim(-0.7, len(top) - 0.3)


def draw_slice(ax, field, title):
    pc = ax.pcolormesh(axis, axis, field, cmap='magma', shading='gouraud')
    pc.set_rasterized(True)
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04, label='density')
    ax.scatter(ALS_RHYTHM[1], ALS_RHYTHM[2], s=150, facecolors='none',
               edgecolors=C_ALS, linewidths=2.4, zorder=3)
    ax.set_xlabel('second IOI (QN)')
    ax.set_ylabel('third IOI (QN)')
    ax.set_title(title)
    ax.set_aspect('equal')


draw_bars(axes[0, 0], ranked_d, 'Differenced interval triples')
draw_bars(axes[0, 1], ranked_r, 'Relative pitch quadruples')
draw_slice(axes[1, 0], marg_d, 'Pitch marginalized: differenced')
draw_slice(axes[1, 1], marg_r, 'Pitch marginalized: relative')
draw_slice(axes[2, 0], cond_d, r'Conditioned on $(+3, -3, +5)$: differenced')
draw_slice(axes[2, 1], cond_r, r'Conditioned on $(+3, -3, +5)$: relative')
fig.suptitle('Coltrane, Acknowledgement: joint pitch-and-rhythm motifs '
             '(first IOI fixed at an eighth note)', y=0.978, fontsize=15)

if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'demo_jmm_2_1_joint.png'),
                dpi=140, bbox_inches='tight')
    print('Saved figures/demo_jmm_2_1_joint.png')
else:
    plt.show()

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)

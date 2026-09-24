"""demo_jmm_4_1_parse.py — Analysis 4.1 (Online Supplement, Section 11): a
supplied parse carried as a nested attribute.

A demo of the Music Perception Toolbox reproducing the analysis from the
JMM article's Online Supplement. Data come from jmm_data (the
rule-labelled derivations of Ren, Rammos, and Rohrmeier 2024, which you
supply); the figures stay on screen unless SAVE_FIGURES is set.

Analysis 4.1: an expert harmonic analysis, supplied as input, carried
into the framework, and operated on by it.

The material is a rule-labelled derivation: each surface chord of a tune
is assigned a root-to-leaf path of rule labels, the grammar's account of
how that chord is reached. The framework does not produce the
derivation — the rules that license it do that, outside — and its part
begins once it is given one.

The encoding. Each chord is one event and its path one nested attribute:
the inner level a rule label's simplex coordinates, read whole and in
order, so that two labels are either the same or equally different; the
outer level the positions of the path, in order, so that position
carries depth. Paths differ in length, so the positions of a chord are
bound by ``group_by`` rather than by a fixed window, and the outer
tuple size ``r_outer`` then says how much of a path a comparison takes
at once. At ``r_outer`` = 2 a tuple is an ordered pair of positions,
matched wherever it occurs in order, contiguous or not — which is
depth-shift and elaboration tolerance without a level coordinate.

Four things the framework then does with it:

  retrieval     a configuration is the query, and the one-sided
                similarity counts its occurrences — exactly, since a
                label either matches or does not at this kernel width;
  partial match a wider kernel extends retrieval to labels that match
                only approximately, and on a regular simplex every
                substitution is equally wrong;
  reduction     per-position weights g^(level - 3) grade a path by
                degree, and the graded tune is compared with the hard
                reduction in which every path is truncated at level 3;
  depth         one further inner coordinate, the level scaled by
                s_level, puts a displacement of one level at kernel
                distance s_level, so that depth enters the comparison
                itself rather than being ignored.

Pre-MAET structure::

    attribute  order   sigma            rel  per
    ---------  ------  ---------------  ---  ---
    label      (V-1, 2)  0.1 (or 0.3)   no   no    the nested path
    label      (V, 2)    0.1            no   no    with the level coordinate

    V is the number of rule labels in the alphabet, so a label is a
    point of a regular (V-1)-simplex of unit edge. Ordered (exch = 0) at
    both levels. Estimator: one-sided similarity for retrieval, cosine
    for the reduction.

Data: ``jmm_data.derivations`` (from your own copy of ParseTrees.json at
``data/ParseTrees.json``). Toolbox: ``simplex_vertices``,
``pack_pre_maet``, ``flat_specs``, ``bind_attributes``, ``bind_events``
(``group_by``), ``select_pre_maet``, ``build_maet``, ``sim_maet``,
``show_pre_maet``. Runtime: a few seconds.
"""
import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import mpt
_prev_defaults = mpt.set_default(show_hints=False)
from mpt import (simplex_vertices, pack_pre_maet, flat_specs, bind_attributes,
                 bind_events, select_pre_maet, build_maet, sim_maet,
                 show_pre_maet)

import jmm_data

# Set True to write the figures to a figures/ folder beside this script;
# False shows them instead.
SAVE_FIGURES = False
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

TUNE = '(Valid)Solar'            # Miles Davis, as the corpus names it
CONTROL = '(Valid)Interplay'     # holds no instance of the query
SIGMA_LABEL = 0.1                # a label either matches or does not
SIGMA_WIDE = 0.3                 # wide enough for a substitution to count
R_OUTER = 2                      # an ordered pair of path positions
QUERY = ['V_I', 'Descending5th']         # a dominant prepared by fifths
QUERY_LEVELS = [3.0, 4.0]                # where it first occurs in Solar
SUBSTITUTES = ['IV_V', 'Backdoor_I']     # neither occurs in Solar
REDUCTION_LEVEL = 3              # paths are graded, and cut, beyond this
G_VALUES = [1.0, 0.5, 0.2, 0.0]  # the grading's decay per level
S_RATIOS = [0.0, 0.2, 0.5, 1.0, 3.0]     # s_level / sigma

C_TUNE = '#1f4eb8'
C_QUERY = '#c25008'

# --- the derivations, and the alphabet of rule labels ---------------------
rows = jmm_data.derivations([TUNE, CONTROL])
ALPHABET = sorted(set(rows['label']) | set(QUERY) | set(SUBSTITUTES))
VERTEX = dict(zip(ALPHABET, simplex_vertices(len(ALPHABET))))
DIM = len(ALPHABET) - 1          # coordinates of a unit-edge simplex
print(f'{rows["tune"].nunique()} tunes, {len(rows)} path positions, '
      f'{len(ALPHABET)} rule labels: {", ".join(ALPHABET)}')
print(f'each label is a vertex of a unit-edge {DIM}-simplex')


def path_columns(table, *, level_scale=None, decay=None, truncate=None):
    """The value rows, weights, and names of a table of path positions.

    No toolbox function is called here: this only turns the table into the
    arrays the pre-MAET is packed from, so that the encoding itself stays
    in the open below. ``level_scale`` appends the level as one further
    coordinate, scaled, so that depth enters the comparison; ``decay``
    weights a position by ``decay ** (level - REDUCTION_LEVEL)`` beyond
    that level, grading the path by degree; ``truncate`` drops the
    positions beyond a level outright, which is the hard reduction the
    grading approaches.
    """
    t = table if truncate is None else table[table['level'] <= truncate]
    coords = np.array([VERTEX[label] for label in t['label']]).T
    level = t['level'].to_numpy(dtype=float)[None, :]
    values = [coords[i:i + 1, :] for i in range(DIM)]
    if level_scale is not None:
        values.append(level_scale * level)
    names = [f'coord{i + 1}' for i in range(len(values))] + ['chord']
    values.append(t['chord'].to_numpy(dtype=float)[None, :])
    # A position's weight multiplies into every tuple that reads it; the
    # coordinates of one position share it, so it is carried by the first
    # and the others weigh 1.
    weights = None
    if decay is not None:
        beyond = np.maximum(level - REDUCTION_LEVEL, 0.0)
        weights = [decay ** beyond] + [np.ones_like(level)] * (len(values) - 1)
    return values, weights, names


def query_table(labels, levels):
    """The query as a table of the same shape: one chord, one row per
    position, at the levels where the configuration first occurs."""
    import pandas as pd
    return pd.DataFrame({'tune': 'query', 'chord': 0,
                         'level': list(levels), 'label': list(labels)})


tune_rows = rows[rows['tune'] == TUNE]
control_rows = rows[rows['tune'] == CONTROL]

# --- the encoding ----------------------------------------------------------
# Four calls turn a table of path positions into the pre-MAET, and every
# analysis below makes the same four: pack the columns with their kernel
# parameters; bind the coordinates of a label into one attribute read whole
# and in order; bind the positions of one chord, which the run-length form
# reads from the chord index rather than from a window width; and keep the
# bound attribute, the chord index having done its work.
values, weights, names = path_columns(query_table(QUERY, [1.0, 2.0]))
pm = pack_pre_maet(values, weights,
                   flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                              period=0.0, name=names))
pm = bind_attributes(pm, attributes=names[:-1], name='label',
                     r=len(names) - 1, exch=False)
pm = bind_events(pm, group_by='chord', r_outer=R_OUTER)
query_pm = select_pre_maet(pm, attributes=['label'])
show_pre_maet(query_pm, max_events=1, decimals=2)

contexts = {}
for tune_name, table in ((TUNE, tune_rows), (CONTROL, control_rows)):
    values, weights, names = path_columns(table)
    pm = pack_pre_maet(values, weights,
                       flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                                  period=0.0, name=names))
    pm = bind_attributes(pm, attributes=names[:-1], name='label',
                         r=len(names) - 1, exch=False)
    pm = bind_events(pm, group_by='chord', r_outer=R_OUTER)
    contexts[tune_name] = select_pre_maet(pm, attributes=['label'])

# --- retrieval -------------------------------------------------------------
# One-sided normalization divides by the query's self inner product, so a
# tune scores the number of occurrences of the configuration: at this
# kernel width a label either matches (1) or does not (0), and a tuple's
# weight is the product across its values.
query = build_maet(query_pm, verbose=False)
tune = build_maet(contexts[TUNE], verbose=False)
control = build_maet(contexts[CONTROL], verbose=False)
print(f'\nretrieval of ({", ".join(QUERY)}):')
print(f'  s_one({TUNE:<18}) = '
      f'{sim_maet(tune, query, normalize="oneSidedDenom", verbose=False):.4f}')
print(f'  s_one({CONTROL:<18}) = '
      f'{sim_maet(control, query, normalize="oneSidedDenom", verbose=False):.4f}')
print(f'  cos(tune, tune)    = {sim_maet(tune, tune, verbose=False):.4f}')
print(f'  cos(tune, control) = {sim_maet(tune, control, verbose=False):.4f}')

# --- partial match ---------------------------------------------------------
# On a regular simplex every pair of labels is the same distance apart, so
# every substitution costs the same: the two scores below are equal by
# construction, which is what makes the simplex the neutral coding. Only
# the kernel widens, so the pre-MAETs are the ones already encoded, built
# at a sigma of their own: build_maet's own argument takes precedence over
# the width the spec carries.
wide = build_maet(contexts[TUNE], sigma=SIGMA_WIDE, verbose=False)
print(f'\nat sigma = {SIGMA_WIDE}, substituting the query\'s second label:')
for substitute in SUBSTITUTES:
    values, weights, names = path_columns(
        query_table([QUERY[0], substitute], [1.0, 2.0]))
    pm = pack_pre_maet(values, weights,
                       flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                                  period=0.0, name=names))
    pm = bind_attributes(pm, attributes=names[:-1], name='label',
                         r=len(names) - 1, exch=False)
    pm = bind_events(pm, group_by='chord', r_outer=R_OUTER)
    q = build_maet(select_pre_maet(pm, attributes=['label']),
                   sigma=SIGMA_WIDE, verbose=False)
    print(f'  {substitute:<12} -> '
          f'{sim_maet(wide, q, normalize="oneSidedDenom", verbose=False):.3f}')

# --- reduction by degree ---------------------------------------------------
# The hard reduction cuts every path at the reduction level; the graded
# ones keep the deeper positions at a weight that decays with depth.
values, weights, names = path_columns(tune_rows, truncate=REDUCTION_LEVEL)
pm = pack_pre_maet(values, weights,
                   flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                              period=0.0, name=names))
pm = bind_attributes(pm, attributes=names[:-1], name='label',
                     r=len(names) - 1, exch=False)
pm = bind_events(pm, group_by='chord', r_outer=R_OUTER)
hard = build_maet(select_pre_maet(pm, attributes=['label']), verbose=False)

reduction = []
for g in G_VALUES:
    values, weights, names = path_columns(tune_rows, decay=g)
    pm = pack_pre_maet(values, weights,
                       flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                                  period=0.0, name=names))
    pm = bind_attributes(pm, attributes=names[:-1], name='label',
                         r=len(names) - 1, exch=False)
    pm = bind_events(pm, group_by='chord', r_outer=R_OUTER)
    graded = build_maet(select_pre_maet(pm, attributes=['label']),
                        verbose=False)
    reduction.append(float(sim_maet(graded, hard, verbose=False)))
print(f'\nreduction: the graded tune against the hard cut at level '
      f'{REDUCTION_LEVEL}')
for g, c in zip(G_VALUES, reduction):
    print(f'  g = {g:<4} cos = {c:.4f}')

# --- depth in the comparison ----------------------------------------------
# The level enters as one further inner coordinate, scaled so that a
# displacement of one level sits at kernel distance s_level. Query and
# context take the same scale, so both are encoded inside the sweep.
depth = []
for ratio in S_RATIOS:
    built = []
    for table in (query_table(QUERY, QUERY_LEVELS), tune_rows):
        values, weights, names = path_columns(
            table, level_scale=ratio * SIGMA_LABEL)
        pm = pack_pre_maet(values, weights,
                           flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                                      period=0.0, name=names))
        pm = bind_attributes(pm, attributes=names[:-1], name='label',
                             r=len(names) - 1, exch=False)
        pm = bind_events(pm, group_by='chord', r_outer=R_OUTER)
        built.append(build_maet(select_pre_maet(pm, attributes=['label']),
                                verbose=False))
    depth.append(float(sim_maet(built[1], built[0],
                                normalize='oneSidedDenom', verbose=False)))
print(f'\ndepth in the comparison: the query at levels '
      f'{"/".join(str(int(v)) for v in QUERY_LEVELS)}')
for ratio, s in zip(S_RATIOS, depth):
    print(f'  s_level / sigma = {ratio:<4} s_one = {s:.2f}')

# --- figure ---------------------------------------------------------------
if plt is None:
    print('matplotlib not available; skipping the figure.')
    mpt.set_default(**_prev_defaults)
    raise SystemExit(0)

plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17,
                     'axes.labelsize': 15, 'font.family': 'DejaVu Sans'})
fig, (axR, axD) = plt.subplots(1, 2, figsize=(13, 4.8))
axR.plot(G_VALUES, reduction, 'o-', color=C_TUNE, linewidth=2, markersize=8)
axR.set_xlabel('grading $g$ (weight per level beyond '
               f'{REDUCTION_LEVEL})')
axR.set_ylabel('cosine against the hard cut')
axR.set_title('Reduction by degree')
axR.set_ylim(0, 1.05)
axR.invert_xaxis()
axR.grid(True, alpha=0.3)
axR.spines[['top', 'right']].set_visible(False)

axD.plot(S_RATIOS, depth, 'o-', color=C_QUERY, linewidth=2, markersize=8)
axD.set_xlabel(r'$s_{\mathrm{level}} / \sigma$')
axD.set_ylabel('one-sided similarity')
axD.set_title('Depth in the comparison')
axD.set_ylim(0, max(depth) * 1.1)
axD.grid(True, alpha=0.3)
axD.spines[['top', 'right']].set_visible(False)
fig.suptitle('A supplied parse: reduction by degree, and depth as a '
             'coordinate', y=0.99)
fig.tight_layout()

if SAVE_FIGURES:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, 'demo_jmm_4_1_parse.png'),
                dpi=140, bbox_inches='tight')
    print('Saved figures/demo_jmm_4_1_parse.png')
else:
    plt.show()

# The demo leaves the toolbox as it found it: the defaults it set at the
# top are restored here.
mpt.set_default(**_prev_defaults)

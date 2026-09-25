"""demo_jmm_4_1_parse.py — Analysis 4.1 (Online Supplement, Section 11): a
supplied parse carried as a nested multiset.

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

The nested encoding. Each chord is one event and its path one nested
multiset within one attribute: the inner level a rule label's simplex
coordinates, read whole and in order, so that two labels are either the
same or equally different; the outer level the positions of the path, in
order, so that position carries depth. Paths differ in length, so the
positions of a chord are bound into one super-event by ``group_by``
(consecutive rows sharing a chord index) rather than by a fixed window,
and the outer tuple size ``r_outer`` then says how much of a path a
comparison takes at once. At ``r_outer`` = 2 a tuple is an ordered pair
of positions, matched wherever it occurs in order, contiguous or not, so
that a configuration is found at any depth and across intervening
elaboration without a level coordinate.

Five things the framework then does with it, the last on a second,
unrolled encoding:

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
                itself rather than being ignored;
  marginals     unrolled into one event per (chord, position) across
                the whole corpus, each weighted 1/m with m the number of
                surface chords its node governs, the label marginal
                returns the corpus's rule frequencies, and carrying each
                chord's quality alongside gives the joint distribution
                of rule and surface.

Pre-MAET structure::

    attribute  r (inner, outer)  sigma          rel  per
    ---------  ----------------  -------------  ---  ---
    label      (V-1, 2)          0.1 (or 0.3)   no   no   nested
    label      (V, 2)            0.1            no   no   nested, with level
    label      V-1               0.1            no   no   unrolled
    quality    Q-1               0.1            no   no   unrolled

    V is the number of rule labels in the alphabet, so a label is a
    point of a regular (V-1)-simplex of unit edge; Q is the number of
    chord qualities, likewise. Ordered (exch = 0) at every level.
    Estimator: one-sided similarity for retrieval and the marginals,
    cosine for the reduction.

Data: ``jmm_data.derivations`` (from your own copy of ParseTrees.json at
``data/ParseTrees.json``). Toolbox: ``simplex_vertices``,
``pack_pre_maet``, ``flat_specs``, ``bind_attributes``, ``bind_events``
(``group_by``), ``select_pre_maet``, ``build_maet``, ``sim_maet``,
``show_pre_maet``. Runtime: a few seconds.
"""
import os
import numpy as np
import pandas as pd
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
DOMINANT_SEVENTH = 'Maj Min Min'         # a major third, then two minor thirds

C_TUNE = '#1f4eb8'
C_QUERY = '#c25008'

# --- the derivations, and the alphabet of rule labels ---------------------
rows = jmm_data.derivations([TUNE, CONTROL])
ALPHABET = sorted(set(rows['label']) | set(QUERY) | set(SUBSTITUTES))
print(f'{rows["tune"].nunique()} tunes, {len(rows)} path positions, '
      f'{len(ALPHABET)} rule labels: {", ".join(ALPHABET)}')
print(f'each label is a vertex of a unit-edge {len(ALPHABET) - 1}-simplex')


def simplex_rows(categories, alphabet):
    """Each category as its vertex of the unit-edge regular simplex whose
    vertices are the alphabet, returned as one value row per coordinate.

    No toolbox function beyond simplex_vertices is called here: this only
    turns a column of the table into the arrays a pre-MAET is packed from,
    so that the encoding itself stays in the open.
    """
    vertex = dict(zip(alphabet, simplex_vertices(len(alphabet))))
    coords = np.array([vertex[c] for c in categories]).T
    return [row[None, :] for row in coords]


def encode_paths(table, *, level_scale=None, decay=None, truncate=None):
    """A table of path positions as the nested pre-MAET, one event per chord.

    Four calls make the encoding, and every analysis of the nested form
    makes the same four: pack the columns with their kernel parameters;
    bind the coordinates of a label into one attribute, read whole and in
    order; bind the positions of one chord into one super-event, the group
    read from the chord index (``group_by``) rather than from a window
    width; and keep the bound attribute, the chord index having done its
    work. ``level_scale`` appends the level as one further coordinate,
    scaled, so that depth enters the comparison; ``decay`` weights a
    position by ``decay ** (level - REDUCTION_LEVEL)`` beyond that level,
    grading the path by degree; ``truncate`` drops the positions beyond a
    level outright, which is the hard reduction the grading approaches.
    """
    t = table if truncate is None else table[table['level'] <= truncate]
    level = t['level'].to_numpy(dtype=float)[None, :]
    values = simplex_rows(t['label'], ALPHABET)
    if level_scale is not None:
        values.append(level_scale * level)
    inner = [f'coord{i + 1}' for i in range(len(values))]
    values.append(t['chord'].to_numpy(dtype=float)[None, :])
    # A position's weight multiplies into every tuple that reads it; the
    # coordinates of one position share it, so it is carried by the first
    # and the others weigh 1.
    weights = None
    if decay is not None:
        beyond = np.maximum(level - REDUCTION_LEVEL, 0.0)
        weights = [decay ** beyond] + [np.ones_like(level)] * len(inner)
    pm = pack_pre_maet(values, weights,
                       flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                                  period=0.0, name=inner + ['chord']))
    pm = bind_attributes(pm, attributes=inner, name='label', r=len(inner),
                         exch=False)
    pm = bind_events(pm, group_by='chord', r_outer=R_OUTER)
    return select_pre_maet(pm, attributes=['label'])


def query_table(labels, levels):
    """The query as a table of the same shape: one chord, one row per
    position, at the levels given."""
    return pd.DataFrame({'tune': 'query', 'chord': 0,
                         'level': list(levels), 'label': list(labels)})


tune_rows = rows[rows['tune'] == TUNE]
control_rows = rows[rows['tune'] == CONTROL]

# --- the nested encoding ---------------------------------------------------
query_pm = encode_paths(query_table(QUERY, [1.0, 2.0]))
show_pre_maet(query_pm, max_events=1, decimals=2)
tune_pm = encode_paths(tune_rows)

# --- retrieval -------------------------------------------------------------
# One-sided normalization divides by the query's self inner product, so a
# tune scores the number of occurrences of the configuration: at this
# kernel width a label either matches (1) or does not (0), and a tuple's
# weight is the product across its values.
query = build_maet(query_pm, verbose=False)
tune = build_maet(tune_pm, verbose=False)
control = build_maet(encode_paths(control_rows), verbose=False)
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
wide = build_maet(tune_pm, sigma=SIGMA_WIDE, verbose=False)
print(f'\nat sigma = {SIGMA_WIDE}, substituting the query\'s second label:')
for substitute in SUBSTITUTES:
    q = build_maet(encode_paths(query_table([QUERY[0], substitute],
                                            [1.0, 2.0])),
                   sigma=SIGMA_WIDE, verbose=False)
    print(f'  {substitute:<12} -> '
          f'{sim_maet(wide, q, normalize="oneSidedDenom", verbose=False):.3f}')

# --- reduction by degree ---------------------------------------------------
# The hard reduction cuts every path at the reduction level; the graded
# ones keep the deeper positions at a weight that decays with depth.
hard = build_maet(encode_paths(tune_rows, truncate=REDUCTION_LEVEL),
                  verbose=False)
reduction = [float(sim_maet(build_maet(encode_paths(tune_rows, decay=g),
                                       verbose=False), hard, verbose=False))
             for g in G_VALUES]
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
    scale = ratio * SIGMA_LABEL
    q = build_maet(encode_paths(query_table(QUERY, QUERY_LEVELS),
                                level_scale=scale), verbose=False)
    t = build_maet(encode_paths(tune_rows, level_scale=scale), verbose=False)
    depth.append(float(sim_maet(t, q, normalize='oneSidedDenom',
                                verbose=False)))
print(f'\ndepth in the comparison: the query at levels '
      f'{"/".join(str(int(v)) for v in QUERY_LEVELS)}')
for ratio, s in zip(S_RATIOS, depth):
    print(f'  s_level / sigma = {ratio:<4} s_one = {s:.2f}')

# --- marginals, on the unrolled encoding ----------------------------------
# The unrolled encoding makes one event of each (chord, position) of every
# derivation in the corpus, the label and the chord's quality becoming two
# flat attributes, each holding its simplex coordinates read whole and in
# order. Weighting each event 1/m, with m the number of surface chords its
# node governs, gives every rule application unit total weight, so the
# one-sided similarity of the corpus against a one-event query holding a
# label (retrieval, as above, now of single positions) is that rule's
# frequency in the corpus. At unit weights, the same reading against a
# (label, quality) query, divided by the reading against the label alone,
# is the share of that quality among the chords the rule governs.
corpus = jmm_data.derivations()
RULES = sorted(set(corpus['label']))
QUALITIES = sorted(set(corpus['quality']))


def encode_unrolled(table, weights=None):
    """A table of path positions as the unrolled pre-MAET, one event per
    (chord, position): its label and its chord's quality, each one
    attribute holding a simplex vertex, read whole and in order."""
    label = simplex_rows(table['label'], RULES)
    quality = simplex_rows(table['quality'], QUALITIES)
    names_label = [f'rule{i + 1}' for i in range(len(label))]
    names_quality = [f'quality{i + 1}' for i in range(len(quality))]
    values = label + quality
    w = None
    if weights is not None:
        w = ([np.asarray(weights, dtype=float)[None, :]]
             + [np.ones((1, len(table)))] * (len(values) - 1))
    pm = pack_pre_maet(values, w,
                       flat_specs(values, sigma=SIGMA_LABEL, is_per=False,
                                  period=0.0,
                                  name=names_label + names_quality))
    pm = bind_attributes(pm, attributes=names_label, name='label',
                         r=len(names_label), exch=False)
    return bind_attributes(pm, attributes=names_quality, name='quality',
                           r=len(names_quality), exch=False)


def position(label, quality=DOMINANT_SEVENTH):
    """A one-event query: one position, its label and its chord's quality."""
    return encode_unrolled(pd.DataFrame({'label': [label],
                                         'quality': [quality]}))


def labels_only(pm):
    """The density of the label attribute alone."""
    return build_maet(select_pre_maet(pm, attributes=['label']),
                      verbose=False)


by_rule = labels_only(encode_unrolled(corpus, 1.0 / corpus['governed']))
frequency = {rule: float(sim_maet(by_rule, labels_only(position(rule)),
                                  normalize='oneSidedDenom', verbose=False))
             for rule in RULES}
print(f'\nunrolled: {corpus["tune"].nunique()} derivations, '
      f'{len(corpus)} events')
print('rule frequencies, read from the label marginal:')
for rule in sorted(RULES, key=frequency.get, reverse=True):
    print(f'  {rule:<15} {frequency[rule]:7.1f}')

plain = encode_unrolled(corpus)
joint = build_maet(plain, verbose=False)
by_label = labels_only(plain)
print(f'share of the dominant-seventh quality ({DOMINANT_SEVENTH}) among '
      f'the chords a rule governs:')
for rule in ('V_I', 'Repeat'):
    both = sim_maet(joint, build_maet(position(rule), verbose=False),
                    normalize='oneSidedDenom', verbose=False)
    alone = sim_maet(by_label, labels_only(position(rule)),
                     normalize='oneSidedDenom', verbose=False)
    print(f'  under {rule:<7} {100 * both / alone:.0f}%')

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

"""Bulger's X-side restriction in the centres sub-route is exact.

The Möbius route's centres sub-route restricts the X side of the
(n_jx, n_jy) centre-overlap array to one representative per tuple-symmetry
orbit and scales the sum by the orbit size: ``r_a!`` for a flat symmetric
attribute, and for a nested attribute the order of the iterated wreath
product acting on its leaf positions (each symmetric level permuted
independently at every node of that level). Both are exact identities, not
approximations, so the restricted and unrestricted matrices -- and the
cosines built from them -- must agree to floating point.

The unrestricted form is obtained two ways, which must agree with each
other: by dropping the comb-side bundle from the centres tuple
(``cx[:10]``), and by the module-level ``_COMB_RESTRICTION_ENABLED``
switch, which is what the MATLAB twin (tests/test_centres_comb_restriction.m)
uses.

Twin of the MATLAB tests/test_centres_comb_restriction.m.
"""
import itertools
import math

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens
import mpt._tensor._mobius_inner as _mi

P = 12.0
TOL = 1e-13


@pytest.fixture(autouse=True)
def _quiet_and_at_the_floor():
    prev_t = mpt.get_default('truncation_sigmas')
    mpt.set_default(truncation_sigmas=float('inf'), show_hints=False)
    try:
        yield
    finally:
        mpt.set_default(truncation_sigmas=prev_t)
        _mi._COMB_RESTRICTION_ENABLED = True


def _nested(inner_r, n_chords, sym, rel_outer, is_per, n_events, seed,
            cards=None):
    """A nested single-attribute density; ``cards`` gives per-chord
    cardinalities when the chords are of unequal size."""
    if cards is None:
        cards = [inner_r + 1] * n_chords
    tags = np.concatenate(
        [np.full(c, i) for i, c in enumerate(cards)]).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.5, P - 0.5, size=(len(tags), n_events)), axis=0)
    spec = {"tags": tags, "r": [inner_r, n_chords], "sym": list(sym),
            "rel": [0, 1] if rel_outer else [0, 0]}
    return build_exp_tens([p], None, specs=[spec], sigma=[1.0],
                          is_per=[is_per], period=[P], verbose=False)


def _flat(r_a, is_sym, is_rel, is_per, K, n_events, seed):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.5, P - 0.5, size=(K, n_events)), axis=0)
    return build_exp_tens([p], [np.ones((K, n_events))], [1.0], [r_a],
                          [is_rel], [is_per], [P], [is_sym], verbose=False)


def _matrices(dx, dy, a=0):
    """(restricted, unrestricted) per-attribute inner matrices, plus the
    orbit size the restriction used (1 when it declined)."""
    cx = _mi._closed_form_attr_centres(dx, a)
    cy = _mi._closed_form_attr_centres(dy, a)
    m_r = _mi._closed_form_attr_matrix_from(cx, cy, float('inf'),
                                            'full-image')
    m_u = _mi._closed_form_attr_matrix_from(cx[:10], cy[:10], float('inf'),
                                            'full-image')
    mult = int(cx[10][3]) if cx[10] is not None else 1
    return np.asarray(m_r), np.asarray(m_u), mult


def _max_rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# ---------------------------------------------------------------------
# The orbit size the restriction claims is the one the build materialises
# ---------------------------------------------------------------------

NESTED_CASES = [
    (inner_r, n_chords, sym, rel_outer, is_per)
    for inner_r in (1, 2)
    for n_chords in (2, 3)
    for sym in ([True, True], [True, False], [False, True])
    for rel_outer in (0, 1)
    for is_per in (True, False)
]


@pytest.mark.parametrize("inner_r,n_chords,sym,rel_outer,is_per",
                         NESTED_CASES)
def test_nested_multiplicity_is_the_wreath_product_order(
        inner_r, n_chords, sym, rel_outer, is_per):
    """n_j / n_k equals prod over symmetric levels of r_l! ** (nodes at l).

    The identity the restriction rests on needs the perm side to be the
    free orbit tiling of the comb side; this pins the orbit size
    structurally rather than assuming it.
    """
    d = _nested(inner_r, n_chords, sym, rel_outer, is_per, 2, 5)
    r_levels = [inner_r, n_chords]
    expected = 1
    for lev, s in enumerate(sym):
        if s:
            nodes = int(np.prod(r_levels[lev + 1:])) if lev + 1 < 2 else 1
            expected *= math.factorial(r_levels[lev]) ** nodes
    assert int(d.n_k) > 0
    assert int(d.n_j) == expected * int(d.n_k)
    assert _mi._nested_orbit_mult(r_levels, sym) == expected


def test_wreath_orbits_tile_the_perm_side():
    """The perm side is the disjoint union of free wreath-group orbits of
    the comb columns -- checked on the index arrays themselves."""
    from mpt._tensor.build import _nested_enum_indices

    r_levels, sym = [2, 2], [True, True]
    tags = np.repeat(np.arange(3), 3).reshape(-1, 1)      # 3 groups of 3
    valid = np.arange(9, dtype=np.intp)
    perm, comb = _nested_enum_indices(valid, tags, r_levels, sym)

    def elements():
        for block_order in itertools.permutations(range(2)):
            for within in itertools.product(
                    itertools.permutations(range(2)), repeat=2):
                g = []
                for b in range(2):
                    g.extend(block_order[b] * 2 + np.array(within[b]))
                yield tuple(g)

    group = list(elements())
    assert len(group) == 8
    perm_cols = set(map(tuple, perm.T.tolist()))
    assert len(perm_cols) == perm.shape[1]                # free: no repeats
    covered = set()
    for col in comb.T.tolist():
        orbit = {tuple(np.asarray(col)[list(g)]) for g in group}
        assert len(orbit) == 8                            # free action
        assert orbit <= perm_cols                         # inside the perm side
        assert not (orbit & covered)                      # orbits disjoint
        covered |= orbit
    assert covered == perm_cols                           # and they cover it


# ---------------------------------------------------------------------
# Restricted == unrestricted, at inner-matrix level
# ---------------------------------------------------------------------

@pytest.mark.parametrize("inner_r,n_chords,sym,rel_outer,is_per",
                         NESTED_CASES)
def test_nested_matrix_restricted_equals_unrestricted(
        inner_r, n_chords, sym, rel_outer, is_per):
    dx = _nested(inner_r, n_chords, sym, rel_outer, is_per, 2, 11)
    dy = _nested(inner_r, n_chords, sym, rel_outer, is_per, 2, 22)
    m_r, m_u, mult = _matrices(dx, dy)
    assert m_r.shape == (2, 2)
    assert np.all(m_u > 0)
    assert _max_rel(m_r, m_u) < TOL
    # The restriction is taken exactly when there is an orbit to collapse:
    # some level both symmetric and of read-arity >= 2.
    has_orbit = any(s and r >= 2 for s, r in zip(sym, [inner_r, n_chords]))
    assert (mult > 1) == has_orbit


@pytest.mark.parametrize("cards", [[2, 3, 2], [3, 2]])
def test_nested_unequal_chord_cardinalities(cards):
    kw = dict(inner_r=2, n_chords=len(cards), sym=[True, True],
              rel_outer=1, is_per=True, n_events=2, cards=cards)
    m_r, m_u, mult = _matrices(_nested(seed=11, **kw), _nested(seed=22, **kw))
    assert mult == 8 if len(cards) == 2 else mult == 48
    assert _max_rel(m_r, m_u) < TOL


def test_nested_multi_event():
    kw = dict(inner_r=2, n_chords=2, sym=[True, True], rel_outer=1,
              is_per=True, n_events=3)
    m_r, m_u, mult = _matrices(_nested(seed=3, **kw), _nested(seed=4, **kw))
    assert m_r.shape == (3, 3) and mult == 8
    assert _max_rel(m_r, m_u) < TOL


@pytest.mark.parametrize("r_a,is_sym,is_rel,is_per", [
    (2, True, True, True), (2, True, True, False), (2, True, False, True),
    (3, True, True, True), (3, True, False, False),
    (2, False, True, True), (3, False, True, False),   # ordered: declines
    (1, True, True, True),                             # r < 2: declines
])
def test_flat_matrix_restricted_equals_unrestricted(r_a, is_sym, is_rel,
                                                    is_per):
    dx = _flat(r_a, is_sym, is_rel, is_per, 4, 2, 11)
    dy = _flat(r_a, is_sym, is_rel, is_per, 4, 2, 22)
    m_r, m_u, mult = _matrices(dx, dy)
    assert _max_rel(m_r, m_u) < TOL
    expected_mult = math.factorial(r_a) if (is_sym and r_a >= 2) else 1
    assert mult == expected_mult


def test_ordered_and_degenerate_attributes_decline():
    """No orbit to collapse: the bundle is None and the matrix is the
    unrestricted perm-vs-perm one."""
    d_ord = _flat(3, False, True, True, 4, 2, 7)
    assert _mi._closed_form_attr_centres(d_ord, 0)[10] is None
    d_r1 = _flat(1, True, False, True, 4, 2, 7)
    assert _mi._closed_form_attr_centres(d_r1, 0)[10] is None
    d_all_ordered = _nested(2, 2, [False, False], 1, True, 2, 7)
    assert _mi._closed_form_attr_centres(d_all_ordered, 0)[10] is None


# ---------------------------------------------------------------------
# ... and through the public cosine
# ---------------------------------------------------------------------

@pytest.mark.parametrize("method", ["contract", "auto"])
@pytest.mark.parametrize("inner_r,n_chords,sym,rel_outer,is_per",
                         NESTED_CASES)
def test_cosine_unchanged_by_the_restriction(method, inner_r, n_chords, sym,
                                             rel_outer, is_per):
    vals = {}
    for enabled in (True, False):
        _mi._COMB_RESTRICTION_ENABLED = enabled
        dx = _nested(inner_r, n_chords, sym, rel_outer, is_per, 2, 11)
        dy = _nested(inner_r, n_chords, sym, rel_outer, is_per, 2, 22)
        vals[enabled] = float(cos_sim_exp_tens(dx, dy, method=method))
    _mi._COMB_RESTRICTION_ENABLED = True
    assert 0.0 < vals[False] <= 1.0 + 1e-12
    assert abs(vals[True] - vals[False]) <= TOL * abs(vals[False])


def test_switch_and_dropped_bundle_agree():
    """The two ways of getting the unrestricted matrix agree exactly."""
    kw = dict(inner_r=2, n_chords=3, sym=[True, True], rel_outer=1,
              is_per=True, n_events=2)
    dx, dy = _nested(seed=11, **kw), _nested(seed=22, **kw)
    _, m_drop, _ = _matrices(dx, dy)
    _mi._COMB_RESTRICTION_ENABLED = False
    try:
        dx2, dy2 = _nested(seed=11, **kw), _nested(seed=22, **kw)
        cx, cy = (_mi._closed_form_attr_centres(dx2, 0),
                  _mi._closed_form_attr_centres(dy2, 0))
        assert cx[10] is None
        m_switch = np.asarray(
            _mi._closed_form_attr_matrix_from(cx, cy, float('inf'),
                                              'full-image'))
    finally:
        _mi._COMB_RESTRICTION_ENABLED = True
    assert np.array_equal(m_drop, m_switch)


# ---------------------------------------------------------------------
# Abs-per full-image: the L = 0 single-image short-circuit
# ---------------------------------------------------------------------
#
# When the truncation budget admits no image beyond the nearest one the
# wrapped Gaussian *is* the nearest-image Gaussian, so the centres route
# takes the joint Q-form path (one exp on the summed form) instead of the
# per-position theta product. The gate must not change the numbers, and
# it must not fire once a second image carries weight.

ABS_PER_L0 = [(r_a, sop) for r_a in (2, 3, 4)
              for sop in (0.005, 0.02, 0.04)]


def _abs_per_pair(r_a, sigma, seed_x=31, seed_y=32, K=5, n_events=3):
    dx = _flat(r_a, True, False, True, K, n_events, seed_x)
    dy = _flat(r_a, True, False, True, K, n_events, seed_y)
    dx.sigma[0] = dy.sigma[0] = sigma
    cx = _mi._closed_form_attr_centres(dx, 0)
    cy = _mi._closed_form_attr_centres(dy, 0)
    # The rebuild carries sigma from the density, so patch the bundle too.
    cx = cx[:8] + (sigma,) + cx[9:]
    cy = cy[:8] + (sigma,) + cy[9:]
    return cx, cy


@pytest.mark.parametrize("r_a,sop", ABS_PER_L0)
def test_abs_per_short_circuit_agrees_with_the_image_sum(r_a, sop):
    """Below the image-budget threshold the two forms are the same number.

    The reference forces the image-summed helper by lifting the image
    count to one; the extra image it then carries is by construction
    below the accuracy floor, so any disagreement above rounding would
    be the short-circuit's, not the reference's.
    """
    import mpt._wrapped_kernel as wk

    sigma = sop * P
    assert wk._image_count_L(sigma, P, float('inf'), 4) == 0
    cx, cy = _abs_per_pair(r_a, sigma)
    m_gate = np.asarray(
        _mi._closed_form_attr_matrix_from(cx, cy, float('inf'),
                                          'full-image'))
    orig = wk._image_count_L
    wk._image_count_L = lambda *a, **kw: max(1, orig(*a, **kw))
    try:
        m_img = np.asarray(
            _mi._closed_form_attr_matrix_from(cx, cy, float('inf'),
                                              'full-image'))
    finally:
        wk._image_count_L = orig
    assert m_gate.shape == m_img.shape
    scale = float(np.max(np.abs(m_img)))
    assert scale > 0.0
    assert float(np.max(np.abs(m_gate - m_img))) <= 3e-15 * scale


@pytest.mark.parametrize("sop", (0.1, 0.2))
@pytest.mark.parametrize("r_a", (2, 3))
def test_abs_per_above_the_gate_still_sums_images(r_a, sop):
    """At L >= 1 the route keeps the full-image helper: the matrix is not
    the nearest-image one, and the gap is far above rounding."""
    import mpt._wrapped_kernel as wk

    sigma = sop * P
    assert wk._image_count_L(sigma, P, float('inf'), 4) >= 1
    cx, cy = _abs_per_pair(r_a, sigma)
    m_full = np.asarray(
        _mi._closed_form_attr_matrix_from(cx, cy, float('inf'),
                                          'full-image'))
    m_single = np.asarray(
        _mi._closed_form_attr_matrix_from(cx, cy, float('inf'),
                                          'single-image'))
    scale = float(np.max(np.abs(m_full)))
    assert float(np.max(np.abs(m_full - m_single))) > 1e-9 * scale

"""Translation sweeps reduced to a mixture in the offset.

The reduction replaces one inner product per offset with a single pass
over the tuple pairs followed by M evaluations of a Gaussian mixture in
the offset. Every test here compares it against the per-offset path it
replaces: build the translated query explicitly, call the ordinary
similarity, and require agreement.

Reference route. The reduction reproduces the pairwise (Bulger) inner
product term for term. The orbit (Möbius) route computes the same
quantity through signed orbit weights whose cancellation is more exposed
to the truncation floor, so comparisons that must hold at the parity
floor pin ``method='bulger'`` on the reference; the ``auto`` route is
checked separately at the looser tolerance its own accuracy supports.

Truncation. Both paths apply the same floor to the accumulated
log-kernel, and the reduction's up-front prune drops only components
that cannot rise above that floor at any offset, so agreement holds at
the parity floor with truncation active as well as disabled.
"""
import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, cos_sim_exp_tens, sweep_cos_sim_exp_tens
from mpt._tensor.sweep import sweep_eligibility

PARITY = 1e-12


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------


def _densities(p_x, p_y, sigma, r, is_rel=None, is_per=None, period=None,
               is_sym=None):
    A = len(p_x)
    is_rel = [0] * A if is_rel is None else is_rel
    is_per = [0] * A if is_per is None else is_per
    period = [None] * A if period is None else period
    is_sym = [1] * A if is_sym is None else is_sym
    args = ([sigma] * A, [r] * A, is_rel, is_per, period, is_sym)
    return (build_exp_tens(p_x, None, *args, verbose=False),
            build_exp_tens(p_y, None, *args, verbose=False),
            args)


def _reference(p_x, p_y, offsets, args, *, method="bulger",
               normalize="cosine", truncation_sigmas=None):
    """Similarity offset by offset, translating the query explicitly."""
    dx = build_exp_tens(p_x, None, *args, verbose=False)
    A = len(p_y)
    out = []
    for m in range(offsets.shape[1]):
        p_ym = [p_y[a] + offsets[a, m] for a in range(A)]
        dy = build_exp_tens(p_ym, None, *args, verbose=False)
        out.append(cos_sim_exp_tens(
            dx, dy, method=method, normalize=normalize,
            truncation_sigmas=truncation_sigmas, verbose=False))
    return np.array(out, dtype=np.float64)


def _random_case(K, n_x, n_y, A, seed):
    rng = np.random.default_rng(seed)
    return ([rng.normal(0.0, 3.0, size=(K, n_x)) for _ in range(A)],
            [rng.normal(0.0, 3.0, size=(K, n_y)) for _ in range(A)])


def _offsets(A, swept=(0,)):
    """Offsets including an off-peak column and an exact-match column.

    The zero column matters: an exact match agrees under a wrong
    ``2 sigma^2`` denominator as readily as under the correct
    ``4 sigma^2`` one, because the numerator vanishes there. Only the
    off-peak columns separate them.
    """
    base = np.array([-3.1, -1.4, -0.35, 0.0, 0.8, 2.2, 5.0])
    off = np.zeros((A, base.size))
    for i, a in enumerate(swept):
        off[a] = base * (1.0 if i == 0 else 0.5)
    return off


def _rel_dev(got, ref):
    """Largest deviation as a fraction of the profile's own peak.

    A sweep profile spans many orders of magnitude: at offsets where the
    query matches nothing, the similarity is ~1e-13 and a per-element
    relative error there reports a huge number for an absolute deviation
    at the truncation floor. Scaling by the peak measures the error on
    the scale the profile is actually read at.
    """
    got = np.atleast_1d(np.asarray(got, dtype=np.float64))
    ref = np.atleast_1d(np.asarray(ref, dtype=np.float64))
    scale = float(np.max(np.abs(ref)))
    if scale == 0.0:
        return float(np.max(np.abs(got)))
    return float(np.max(np.abs(got - ref)) / scale)


SHAPES = [(1, 1), (3, 3), (4, 4), (3, 2), (4, 2), (5, 3)]


# -------------------------------------------------------------------
#  Core parity
# -------------------------------------------------------------------


@pytest.mark.parametrize("K,r", SHAPES)
@pytest.mark.parametrize("is_sym", [0, 1])
@pytest.mark.parametrize("A", [1, 2])
def test_sweep_matches_per_offset(K, r, is_sym, A):
    """The reduction reproduces the per-offset sweep at the parity floor."""
    p_x, p_y = _random_case(K, 6, 3, A, seed=100 + K * 10 + r)
    off = _offsets(A, swept=tuple(range(A)))
    dx, dy, args = _densities(p_x, p_y, 0.9, r, is_sym=[is_sym] * A)
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


@pytest.mark.parametrize("K,r", [(1, 1), (3, 3), (4, 2)])
def test_sweep_matches_under_default_truncation(K, r):
    """Agreement holds with the truncation floor active, not only without."""
    p_x, p_y = _random_case(K, 6, 3, 2, seed=7)
    off = _offsets(2, swept=(0, 1))
    dx, dy, args = _densities(p_x, p_y, 0.9, r)
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 verbose=False)
    ref = _reference(p_x, p_y, off, args)
    assert _rel_dev(got, ref) <= PARITY


def test_off_peak_offsets_separate_the_denominator():
    """The 4 sigma^2 denominator is pinned by the off-peak offsets.

    An inner product of two width-sigma kernels is a width-sqrt(2)-sigma
    kernel, so the placement Gaussian carries ``4 sigma^2``. Halving it
    leaves exact matches untouched and everything else wrong, which is
    what this asserts: agreement at the zero offset alone would not
    detect the error.
    """
    p_x, p_y = _random_case(3, 5, 2, 1, seed=11)
    off = _offsets(1)
    dx, dy, args = _densities(p_x, p_y, 0.9, 3)
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    zero_col = int(np.flatnonzero(off[0] == 0.0)[0])
    off_peak = [m for m in range(off.shape[1]) if m != zero_col]
    assert abs(got[zero_col] - ref[zero_col]) <= PARITY
    assert _rel_dev(got[off_peak], ref[off_peak]) <= PARITY


def test_unordered_attribute_uses_the_unnormalised_permutation_sum():
    """A permuted copy of the query matches only when the attribute is
    unordered, and then at the unnormalised permutation sum."""
    p_y = [np.array([[0.0], [4.0], [7.0]])]
    p_x = [p_y[0][[2, 0, 1], :]]           # same values, positions permuted
    off = np.zeros((1, 1))
    for is_sym in (0, 1):
        dx, dy, args = _densities(p_x, p_y, 0.9, 3, is_sym=[is_sym])
        got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                     truncation_sigmas=np.inf, verbose=False)
        ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
        assert abs(got[0] - ref[0]) <= PARITY
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3, is_sym=[1])
    assert sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                  truncation_sigmas=np.inf,
                                  verbose=False)[0] == pytest.approx(1.0,
                                                                     abs=1e-12)


@pytest.mark.parametrize("normalize", ["cosine", "oneSidedDenom"])
def test_both_normalisations(normalize):
    p_x, p_y = _random_case(3, 6, 3, 2, seed=21)
    off = _offsets(2, swept=(0, 1))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3)
    got = sweep_cos_sim_exp_tens(dx, dy, off, normalize=normalize,
                                 method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _reference(p_x, p_y, off, args, normalize=normalize,
                     truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


# -------------------------------------------------------------------
#  Attributes that are not swept
# -------------------------------------------------------------------


def test_unswept_relative_attribute():
    """A relative attribute contributes an offset-independent factor."""
    p_x, p_y = _random_case(3, 6, 3, 2, seed=31)
    off = _offsets(2, swept=(0,))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3, is_rel=[0, 1])
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_unswept_absolute_periodic_attribute():
    """A periodic attribute is supported so long as it is not swept."""
    p_x, p_y = _random_case(3, 6, 3, 2, seed=41)
    off = _offsets(2, swept=(0,))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3, is_per=[0, 1],
                              period=[None, 12.0])
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_no_attribute_swept_is_constant():
    """An all-zero offset column set reduces to a repeated single value."""
    p_x, p_y = _random_case(3, 5, 3, 1, seed=51)
    off = np.zeros((1, 4))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3)
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert np.all(np.abs(got - got[0]) <= PARITY)
    assert _rel_dev(got, ref) <= PARITY


# -------------------------------------------------------------------
#  Refusals
# -------------------------------------------------------------------


def test_swept_relative_attribute_is_refused():
    p_x, p_y = _random_case(3, 4, 2, 1, seed=61)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3, is_rel=[1])
    with pytest.raises(ValueError, match="relative and swept"):
        sweep_cos_sim_exp_tens(dx, dy, np.array([[0.0, 1.5]]), verbose=False)


def test_swept_periodic_attribute_is_refused_by_the_mixture():
    """The split needs the shape term; the wrapped kernel has none.

    The orbit route carries this case instead (see below), so ``'auto'``
    reaches it --- the refusal is specific to the mixture.
    """
    p_x, p_y = _random_case(3, 4, 2, 1, seed=62)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3, is_per=[1], period=[12.0])
    with pytest.raises(ValueError, match="periodic and swept"):
        sweep_cos_sim_exp_tens(dx, dy, np.array([[0.0, 1.5]]),
                               method="mixture", verbose=False)


def test_relative_periodic_above_the_limit_is_refused_by_the_mixture():
    """The mixture computes the single-wrap form; above the sigma/P limit
    that is one of two measures, so it declines unless the wrap names it.

    ``'auto'`` still succeeds, by routing to the orbit route, which
    computes the transposition-average form the default ``full-image``
    wrap asks for --- the same resolution the per-offset dispatcher
    reaches for the same inputs.
    """
    p_x, p_y = _random_case(2, 4, 2, 2, seed=63)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 2, is_rel=[0, 1], is_per=[0, 1],
                           period=[None, 12.0])
    off = np.zeros((2, 2))
    off[0] = [0.0, 1.5]
    with pytest.raises(ValueError, match="relative and periodic at sigma/P"):
        sweep_cos_sim_exp_tens(dx, dy, off, method="mixture", verbose=False)
    out = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
    assert np.all(np.isfinite(out))


def test_relative_periodic_below_the_limit_is_accepted_by_the_mixture():
    """Below the limit the two measures agree inside the accuracy floor."""
    p_x, p_y = _random_case(2, 4, 2, 2, seed=64)
    dx, dy, args = _densities(p_x, p_y, 0.2, 2, is_rel=[0, 1],
                              is_per=[0, 1], period=[None, 12.0])
    off = np.zeros((2, 3))
    off[0] = [0.0, 1.5, -2.0]
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = np.array([
        cos_sim_exp_tens(
            dx, build_exp_tens([p_y[0] + off[0, m], p_y[1]], None, *args,
                               verbose=False),
            method="bulger", truncation_sigmas=np.inf, verbose=False)
        for m in range(off.shape[1])])
    assert np.max(np.abs(got - ref)) <= 1e-12


def test_eligibility_reports_reasons_without_raising():
    p_x, p_y = _random_case(3, 4, 2, 1, seed=64)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3, is_per=[1], period=[12.0])
    ok, reason = sweep_eligibility(dx, dy, np.array([[0.0, 1.5]]))
    assert not ok and "periodic" in reason
    ok, reason = sweep_eligibility(dx, dy, np.zeros((1, 2)))
    assert ok and reason is None


def test_offsets_shape_is_validated():
    p_x, p_y = _random_case(3, 4, 2, 2, seed=65)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3)
    with pytest.raises(ValueError, match=r"\(A, M\) array"):
        sweep_cos_sim_exp_tens(dx, dy, np.zeros((3, 4)), verbose=False)


# -------------------------------------------------------------------
#  Automatic routing from translate_attributes
# -------------------------------------------------------------------


def _tagged(p_y, off):
    A = len(p_y)
    entries, _, _ = mpt.unpack_pre_maet(mpt.translate_attributes(
        p_y, None, [off[a].reshape(1, -1) for a in range(A)]))
    return entries


def test_translate_attributes_tags_its_sweep():
    p_y = [np.zeros((2, 3)), np.zeros((2, 3))]
    off = _offsets(2, swept=(0, 1))
    tagged = _tagged(p_y, off)
    assert isinstance(tagged, list)
    assert isinstance(tagged, mpt.TranslatedSweep)
    assert len(tagged) == off.shape[1]
    assert np.allclose(tagged.sweep_offsets, off)


def test_single_translation_is_untagged():
    """``M = 1`` returns a value-list, as before, with nothing attached."""
    p_y = [np.zeros((2, 3))]
    out, _, _ = mpt.unpack_pre_maet(mpt.translate_attributes(p_y, None, [2.0]))
    assert not isinstance(out, mpt.TranslatedSweep)
    assert isinstance(out, list) and isinstance(out[0], np.ndarray)


@pytest.mark.parametrize("K,r", [(1, 1), (3, 3), (4, 2)])
def test_tagged_sweep_routes_and_agrees(K, r):
    """The tagged list and a plain list of the same entries agree."""
    p_x, p_y = _random_case(K, 6, 3, 2, seed=70 + K)
    off = _offsets(2, swept=(0, 1))
    A = 2
    args = ([0.9] * A, [r] * A, [0] * A, [0] * A, [None] * A, [1] * A)
    tagged = _tagged(p_y, off)
    kw = dict(truncation_sigmas=np.inf, verbose=False)
    fast = cos_sim_exp_tens(p_x, None, tagged, None, *args, **kw)
    slow = cos_sim_exp_tens(p_x, None, list(tagged), None, *args, **kw)
    assert _rel_dev(fast, slow) <= PARITY


def test_tagged_sweep_routes_with_operands_reversed():
    p_x, p_y = _random_case(3, 6, 3, 2, seed=81)
    off = _offsets(2, swept=(0, 1))
    args = ([0.9] * 2, [3] * 2, [0] * 2, [0] * 2, [None] * 2, [1] * 2)
    tagged = _tagged(p_y, off)
    kw = dict(truncation_sigmas=np.inf, verbose=False)
    fast = cos_sim_exp_tens(tagged, None, p_x, None, *args, **kw)
    slow = cos_sim_exp_tens(list(tagged), None, p_x, None, *args, **kw)
    assert _rel_dev(fast, slow) <= PARITY


def test_untagged_list_still_computes():
    """A hand-built sweep is not accelerated, and is still correct.

    Recovering offsets from translated values would need a tolerance,
    and no tolerance both admits every honestly translated sweep and
    preserves the parity floor. Untagged lists therefore take the
    per-offset path.
    """
    p_x, p_y = _random_case(3, 5, 2, 1, seed=91)
    off = _offsets(1)
    hand_built = [[p_y[0] + off[0, m]] for m in range(off.shape[1])]
    args = ([0.9], [3], [0], [0], [None], [1])
    kw = dict(truncation_sigmas=np.inf, verbose=False)
    got = cos_sim_exp_tens(p_x, None, hand_built, None, *args, **kw)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_forced_method_bypasses_the_reduction():
    """An explicit ``method`` names a route; it is honoured, not replaced."""
    p_x, p_y = _random_case(3, 5, 2, 1, seed=92)
    off = _offsets(1)
    args = ([0.9], [3], [0], [0], [None], [1])
    tagged = _tagged(p_y, off)
    kw = dict(truncation_sigmas=np.inf, verbose=False)
    got = cos_sim_exp_tens(p_x, None, tagged, None, *args,
                           method="mobius", **kw)
    ref = _reference(p_x, p_y, off, args, method="mobius",
                     truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_relative_no_op_column_still_agrees():
    """A uniform offset on a relative attribute is a no-op in both paths."""
    p_x, p_y = _random_case(3, 6, 3, 2, seed=93)
    off = _offsets(2, swept=(0, 1))
    args = ([0.9] * 2, [3] * 2, [0, 1], [0] * 2, [None] * 2, [1] * 2)
    with pytest.warns(mpt.TranslateAttributesNoOpWarning):
        tagged, _, _ = mpt.unpack_pre_maet(mpt.translate_attributes(
            p_y, None, [off[a].reshape(1, -1) for a in range(2)],
            specs=[{"rel": False}, {"rel": True}]))
    kw = dict(truncation_sigmas=np.inf, verbose=False)
    fast = cos_sim_exp_tens(p_x, None, tagged, None, *args, **kw)
    slow = cos_sim_exp_tens(p_x, None, list(tagged), None, *args, **kw)
    assert _rel_dev(fast, slow) <= PARITY


# -------------------------------------------------------------------
#  Orbit route
# -------------------------------------------------------------------
#
#  The orbit route evaluates the Möbius decomposition at the shifted
#  values instead of forming the placement/shape split, so its cost
#  scales with the orbit count rather than with the tuple-pair count.
#  It is a different decomposition of the same quantity, so it is
#  compared against the orbit reference at the parity floor and against
#  the mixture at the looser tolerance two decompositions support.

ROUTE_AGREEMENT = 1e-11


def _orbit_reference(p_x, p_y, offsets, args, *, normalize="cosine",
                     truncation_sigmas=None):
    dx = build_exp_tens(p_x, None, *args, verbose=False)
    A = len(p_y)
    out = []
    for m in range(offsets.shape[1]):
        dy = build_exp_tens([p_y[a] + offsets[a, m] for a in range(A)],
                            None, *args, verbose=False)
        out.append(cos_sim_exp_tens(
            dx, dy, method="mobius", normalize=normalize,
            truncation_sigmas=truncation_sigmas, verbose=False))
    return np.array(out, dtype=np.float64)


@pytest.mark.parametrize("K,r", [(3, 2), (4, 3), (4, 4), (5, 3)])
def test_orbit_route_matches_per_offset(K, r):
    p_x, p_y = _random_case(K, 5, 3, 1, seed=200 + K)
    off = _offsets(1)
    dx, dy, args = _densities(p_x, p_y, 0.9, r)
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="orbit",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _orbit_reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


@pytest.mark.parametrize("K,r", [(3, 2), (4, 3), (3, 3)])
def test_orbit_and_mixture_agree(K, r):
    """Two decompositions of one quantity."""
    p_x, p_y = _random_case(K, 5, 3, 2, seed=210 + K)
    off = _offsets(2, swept=(0, 1))
    dx, dy, _ = _densities(p_x, p_y, 0.9, r)
    mix = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    orb = sweep_cos_sim_exp_tens(dx, dy, off, method="orbit",
                                 truncation_sigmas=np.inf, verbose=False)
    assert _rel_dev(orb, mix) <= ROUTE_AGREEMENT


def test_orbit_route_carries_a_swept_periodic_attribute():
    """The mixture refuses this; the orbit route does not need the split."""
    rng = np.random.default_rng(220)
    p_x = [rng.uniform(0, 12, size=(3, 5))]
    p_y = [rng.uniform(0, 12, size=(3, 2))]
    off = _offsets(1)
    args = ([0.6], [3], [0], [1], [12.0], [1])
    dx = build_exp_tens(p_x, None, *args, verbose=False)
    dy = build_exp_tens(p_y, None, *args, verbose=False)
    with pytest.raises(ValueError, match="periodic and swept"):
        sweep_cos_sim_exp_tens(dx, dy, off, method="mixture", verbose=False)
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="orbit",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _orbit_reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_orbit_route_refuses_a_swept_relative_attribute():
    """Untranslated relative attributes ride this route; swept ones cannot."""
    p_x, p_y = _random_case(3, 4, 2, 1, seed=230)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3, is_rel=[1])
    with pytest.raises(ValueError, match="orbit route does not support"):
        sweep_cos_sim_exp_tens(dx, dy, np.array([[0.0, 1.5]]),
                               method="orbit", verbose=False)


def _rel_per_pair(sop, seed, period=12.0, wrap=None):
    """Swept absolute attribute beside an untranslated rel-per one."""
    rng = np.random.default_rng(seed)
    p_x = [rng.normal(0.0, 20.0, size=(3, 10)),
           rng.uniform(0.0, period, size=(3, 10))]
    p_y = [rng.normal(0.0, 20.0, size=(3, 3)),
           rng.uniform(0.0, period, size=(3, 3))]
    args = ([0.9, sop * period], [3, 3], [0, 1], [0, 1], [None, period],
            [1, 1])
    kw = dict(verbose=False)
    if wrap is not None:
        kw["wrap"] = wrap
    dx = build_exp_tens(p_x, None, *args, **kw)
    dy = build_exp_tens(p_y, None, *args, **kw)
    return p_x, p_y, dx, dy, args, kw


@pytest.mark.parametrize("sop", [0.02, 0.05, 0.10, 0.15])
def test_orbit_route_carries_an_unswept_rel_per_attribute(sop):
    """The route computes the transposition-average kernel throughout.

    Judged as an absolute deviation: a cosine lives in [-1, 1], and at
    offsets where the two densities barely overlap its value falls to
    1e-16, where a ratio to that value reports noise rather than error.
    """
    p_x, p_y, dx, dy, args, kw = _rel_per_pair(sop, seed=600)
    off = _offsets(2, swept=(0,))
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="orbit",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = np.array([
        cos_sim_exp_tens(
            dx, build_exp_tens([p_y[0] + off[0, m], p_y[1]], None,
                               *args, **kw),
            method="mobius", truncation_sigmas=np.inf, verbose=False)
        for m in range(off.shape[1])])
    assert np.max(np.abs(got - ref)) <= 1e-12


def test_orbit_declines_single_image_rel_per_above_the_limit():
    """That wrap names the other measure, which this route does not compute."""
    from mpt._tensor.sweep import orbit_sweep_supported

    _, _, dx, dy, _, _ = _rel_per_pair(
        0.10, seed=610, wrap=["full-image", "single-image"])
    off = _offsets(2, swept=(0,))
    assert not orbit_sweep_supported(dx, dy, off, np.inf)
    _, _, dx2, dy2, _, _ = _rel_per_pair(0.02, seed=610,
                                         wrap=["full-image", "single-image"])
    # Below the limit the two measures agree, so the wrap does not bind.
    assert orbit_sweep_supported(dx2, dy2, off, np.inf)


def test_auto_prefers_the_mixture_at_low_tuple_order():
    from mpt._tensor.sweep import (_choose_sweep_route, orbit_sweep_supported,
                                   sweep_eligibility)
    p_x, p_y = _random_case(3, 8, 3, 1, seed=240)
    off = _offsets(1)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 2)
    dx, dy = dx.pruned(), dy.pruned()
    ok, _ = sweep_eligibility(dx, dy, off)
    assert _choose_sweep_route(dx, dy, off, ok,
                               orbit_sweep_supported(dx, dy, off)) == "mixture"


def test_auto_falls_back_to_the_orbit_route_when_the_mixture_is_refused():
    """A swept periodic attribute: only one route can carry it."""
    from mpt._tensor.sweep import (_choose_sweep_route, orbit_sweep_supported,
                                   sweep_eligibility)
    rng = np.random.default_rng(250)
    p_x = [rng.uniform(0, 12, size=(3, 5))]
    p_y = [rng.uniform(0, 12, size=(3, 2))]
    off = _offsets(1)
    args = ([0.6], [3], [0], [1], [12.0], [1])
    dx = build_exp_tens(p_x, None, *args, verbose=False).pruned()
    dy = build_exp_tens(p_y, None, *args, verbose=False).pruned()
    ok, _ = sweep_eligibility(dx, dy, off)
    assert not ok
    assert _choose_sweep_route(dx, dy, off, ok,
                               orbit_sweep_supported(dx, dy, off)) == "orbit"
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
    ref = _orbit_reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_method_is_validated():
    p_x, p_y = _random_case(3, 4, 2, 1, seed=260)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3)
    with pytest.raises(ValueError, match="method must be"):
        sweep_cos_sim_exp_tens(dx, dy, _offsets(1), method="bulger",
                               verbose=False)


# -------------------------------------------------------------------
#  Nested attributes
# -------------------------------------------------------------------
#
#  A nested attribute resolved to an inner or intermediate
#  co-transposition unit has a block-diagonal quadratic form: each block
#  removes its own all-ones. That is the relative case read per block, so
#  it contributes one shape term per block and no placement term, and
#  like a relative attribute it cannot be swept.

_NEST_INNER = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False],
                   rel="innermost")
_NEST_ABS = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False])
_NEST_OUTER = dict(tags=[0, 0, 1, 1], r=[2, 2], sym=[True, False],
                   rel="outermost")


def _nested_pair(spec, seed, n_x=3, n_y=2):
    rng = np.random.default_rng(seed)
    p_x = [np.sort(rng.normal(0, 4, size=(4, n_x)), axis=0)]
    p_y = [np.sort(rng.normal(0, 4, size=(4, n_y)), axis=0)]
    kw = dict(nested=[spec], verbose=False)
    args = ([1.0], [1], [False], [False], [0.0])
    dx = build_exp_tens(p_x, None, *args, **kw)
    dy = build_exp_tens(p_y, None, *args, **kw)
    return p_x, p_y, dx, dy, args, kw


def test_nested_absolute_attribute_splits():
    """An absolute nested attribute is an absolute attribute."""
    p_x, p_y, dx, dy, args, kw = _nested_pair(_NEST_ABS, seed=300)
    off = _offsets(1)
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = np.array([
        cos_sim_exp_tens(
            dx, build_exp_tens([p_y[0] + off[0, m]], None, *args, **kw),
            method="bulger", truncation_sigmas=np.inf, verbose=False)
        for m in range(off.shape[1])])
    assert _rel_dev(got, ref) <= PARITY


def test_nested_inner_attribute_is_supported_unswept():
    """Swept absolute attribute alongside an untranslated nested one."""
    rng = np.random.default_rng(310)
    p_x = [np.sort(rng.normal(0, 4, size=(4, 3)), axis=0),
           rng.normal(0, 3, size=(2, 3))]
    p_y = [np.sort(rng.normal(0, 4, size=(4, 2)), axis=0),
           rng.normal(0, 3, size=(2, 2))]
    args = ([1.0, 0.9], [1, 2], [False, False], [False, False], [0.0, 0.0])
    kw = dict(nested=[_NEST_INNER, None], verbose=False)
    dx = build_exp_tens(p_x, None, *args, **kw)
    dy = build_exp_tens(p_y, None, *args, **kw)
    off = _offsets(2, swept=(1,))
    got = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = np.array([
        cos_sim_exp_tens(
            dx, build_exp_tens([p_y[0], p_y[1] + off[1, m]], None,
                               *args, **kw),
            method="bulger", truncation_sigmas=np.inf, verbose=False)
        for m in range(off.shape[1])])
    assert _rel_dev(got, ref) <= PARITY


def test_nested_inner_attribute_alone_is_supported():
    p_x, p_y, dx, dy, args, kw = _nested_pair(_NEST_INNER, seed=320)
    got = sweep_cos_sim_exp_tens(dx, dy, np.zeros((1, 3)), method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = cos_sim_exp_tens(dx, dy, method="bulger",
                           truncation_sigmas=np.inf, verbose=False)
    assert _rel_dev(got, np.full(3, ref)) <= PARITY


def test_swept_nested_inner_attribute_is_refused():
    """Each block removes its own all-ones, so a shift cancels in all."""
    p_x, p_y, dx, dy, _, _ = _nested_pair(_NEST_INNER, seed=330)
    with pytest.raises(ValueError, match="cancels within"):
        sweep_cos_sim_exp_tens(dx, dy, _offsets(1), method="mixture",
                               verbose=False)


def test_swept_nested_outermost_attribute_is_refused():
    """The outermost unit rides the ordinary relative path."""
    p_x, p_y, dx, dy, _, _ = _nested_pair(_NEST_OUTER, seed=340)
    with pytest.raises(ValueError, match="relative and swept"):
        sweep_cos_sim_exp_tens(dx, dy, _offsets(1), method="mixture",
                               verbose=False)


# -------------------------------------------------------------------
#  Conditioning
# -------------------------------------------------------------------


@pytest.mark.parametrize("origin", [0.0, 1e3, 1e6, 1e9])
def test_accuracy_holds_far_from_the_origin(origin):
    """Far from the origin, the requested offset is not representable.

    At magnitude 1e9 the float spacing is 1.2e-7, so translating the
    query by a requested 0.8 lands on the nearest representable value
    and realises a slightly different shift. The reduction applies the
    offset it was given; the per-offset path applies whatever the grid
    can hold. Comparing them therefore requires the *realised* offset,
    and with it the floor holds at every magnitude --- comparing against
    the requested one instead reports the spacing (1e-7 at 1e9) and
    would look like a failure of the reduction rather than of the
    reference's arithmetic.
    """
    rng = np.random.default_rng(400)
    p_x = [rng.normal(0.0, 5.0, size=(3, 8)) + origin]
    p_y = [rng.normal(0.0, 5.0, size=(3, 3)) + origin]
    off = _offsets(1)
    realised = np.array([[float(np.mean((p_y[0] + off[0, m]) - p_y[0]))
                          for m in range(off.shape[1])]])
    dx, dy, args = _densities(p_x, p_y, 0.35, 3)
    got = sweep_cos_sim_exp_tens(dx, dy, realised, method="mixture",
                                 truncation_sigmas=np.inf, verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


# -------------------------------------------------------------------
#  Gram quadratic form in the shared inner-product path
# -------------------------------------------------------------------


def _difference_form_Q(U, V, block, is_rel):
    from mpt._tensor.dispatch import _compute_Q, _compute_Q_inner_blocks
    D = U[:, :, None] - V[:, None, :]
    if block > 0 and block < U.shape[0]:
        return np.asarray(_compute_Q_inner_blocks(D, block, False, 0.0,
                                                  reduced=False))
    return np.asarray(_compute_Q(D, U.shape[0], is_rel, False, 0.0))


@pytest.mark.parametrize("magnitude", [0.0, 1e3, 1e6, 1e9])
@pytest.mark.parametrize("block,is_rel", [(0, False), (4, True), (2, False)])
def test_gram_quadratic_form_matches_the_difference_form(magnitude, block,
                                                         is_rel):
    """One gemm in place of an (r, nJ, nK) difference array.

    The shared shift is what makes this hold far from the origin: the
    Gram identity cancels two large numbers, and without the shift the
    log-kernel departs by 4e-3 at magnitude 1e6 and 5e-1 at 1e7.
    """
    from mpt._tensor.cosine import _gram_quadratic_form

    rng = np.random.default_rng(500)
    U = rng.normal(0.0, 5.0, size=(4, 120)) + magnitude
    V = rng.normal(0.0, 5.0, size=(4, 80)) + magnitude
    got = _gram_quadratic_form(U, V, block)
    ref = _difference_form_Q(U, V, block, is_rel)
    # Q is consumed as Q / (4 sigma^2), so judge it on its own scale.
    assert np.max(np.abs(got - ref)) <= 1e-12 * max(float(np.max(ref)), 1.0)


def test_gram_is_declined_when_it_would_cost_accuracy():
    """Small sigma against a wide spread falls back to the difference form.

    The Gram rounding is relative to the size of the coordinates, not to
    the distance they encode, so it grows as sigma shrinks against the
    attribute's spread. The guard compares that against the floor
    ``truncation_sigmas`` implies, so asking for accuracy-floor accuracy
    takes the difference form and the default takes the fast one.
    """
    from mpt._tensor.cosine import _gram_is_accurate_enough

    rng = np.random.default_rng(510)
    U = rng.normal(0.0, 5.0, size=(4, 60))
    V = rng.normal(0.0, 5.0, size=(4, 40))
    assert not _gram_is_accurate_enough(U, V, 0.01, np.inf)
    assert _gram_is_accurate_enough(U, V, 0.01, 6.0)
    assert _gram_is_accurate_enough(U, V, 10.0, np.inf)


@pytest.mark.parametrize("is_rel", [False, True])
@pytest.mark.parametrize("A", [1, 2])
def test_ma_log_kernel_is_unchanged_by_the_gram_route(is_rel, A):
    """The routed and unrouted forms agree on the log-kernel itself."""
    from mpt._tensor.cosine import _ma_log_kernel

    rng = np.random.default_rng(520 + A)
    r, n_j, n_k = 3, 200, 150
    u = [rng.normal(0.0, 5.0, size=(r, n_j)) for _ in range(A)]
    v = [rng.normal(0.0, 5.0, size=(r, n_k)) for _ in range(A)]
    args = ([r] * A, [5.0] * A, [is_rel] * A, [False] * A, [0.0] * A)
    fast = _ma_log_kernel(u, v, n_j, n_k, A, *args, truncation_sigmas=6.0)
    exact = _ma_log_kernel(u, v, n_j, n_k, A, *args,
                           truncation_sigmas=np.inf)
    assert np.max(np.abs(fast - exact)) <= 1e-11


# -------------------------------------------------------------------
#  The wrap choice survives density transformations
# -------------------------------------------------------------------


def test_pruning_preserves_the_wrap_choice():
    """A pruned density must keep the measure it was built with.

    ``pruned()`` rebuilds the density, and the constructor defaults
    ``wrap`` to ``'full-image'``, so a density built with
    ``'single-image'`` silently reverted --- changing the *measure*
    rather than the speed. It surfaces only above the sigma/period
    threshold, where the two forms diverge, which is why it went
    unnoticed: below it the two agree to the floor and nothing looks
    wrong.
    """
    p = [np.array([[1.0, 5.0, 9.0, 2.0], [3.0, 7.0, 11.0, 4.0]])]
    w = [np.array([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])]
    dens = build_exp_tens(p, w, [1.2], [2], [1], [1], [12.0], [1],
                          wrap=["single-image"], verbose=False)
    pruned = dens.pruned()
    assert pruned is not dens          # an event really was dropped
    assert pruned.n < dens.n
    assert list(pruned.wrap) == ["single-image"]


def test_pruning_preserves_wrap_for_every_attribute():
    p = [np.array([[0.0, 4.0, 8.0]]), np.array([[1.0, 5.0, 9.0]])]
    w = [np.array([[1.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 1.0]])]
    dens = build_exp_tens(p, w, [1.0, 1.2], [1, 1], [0, 1], [1, 1],
                          [12.0, 12.0], [1, 1],
                          wrap=["single-image", "full-image"], verbose=False)
    assert list(dens.pruned().wrap) == ["single-image", "full-image"]


def test_orbit_route_declines_ordered_attributes():
    """The orbit decomposition is symmetric-only.

    It sums over unordered value subsets with multiplicity, and the
    per-attribute routine it calls takes no symmetry flag. On an ordered
    attribute it therefore computes a *different* quantity rather than
    an approximation of the right one --- measured departures up to 0.22
    --- so the route must decline and ``'auto'`` must fall back.
    """
    from mpt._tensor.sweep import orbit_sweep_supported

    p_x, p_y = _random_case(5, 6, 3, 2, seed=830)
    off = _offsets(2, swept=(0, 1))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3, is_sym=[0, 0])
    dxp, dyp = dx.pruned(), dy.pruned()
    assert not orbit_sweep_supported(dxp, dyp, off, np.inf)
    with pytest.raises(ValueError, match="orbit route does not support"):
        sweep_cos_sim_exp_tens(dx, dy, off, method="orbit",
                               truncation_sigmas=np.inf, verbose=False)
    # 'auto' falls back to the mixture and stays exact.
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


@pytest.mark.parametrize("is_sym", [0, 1])
def test_auto_is_exact_for_both_symmetry_settings(is_sym):
    """Whichever route ``'auto'`` picks, the answer is the same one."""
    p_x, p_y = _random_case(4, 6, 3, 2, seed=840 + is_sym)
    off = _offsets(2, swept=(0, 1))
    dx, dy, args = _densities(p_x, p_y, 0.9, 2, is_sym=[is_sym] * 2)
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


@pytest.mark.parametrize("method", ["mixture", "orbit", "auto"])
def test_default_truncation_reaches_every_route(method):
    """Every other test names ``truncation_sigmas`` explicitly.

    The default therefore never reached the routes' internals, which is
    how a MATLAB-side defect slipped through: the Möbius entry points
    require a scalar and reject an empty value, a failure about the
    argument's *shape* that no accuracy test can find.
    """
    p_x, p_y = _random_case(4, 6, 3, 1, seed=840)
    off = _offsets(1)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3)
    out = sweep_cos_sim_exp_tens(dx, dy, off, method=method, verbose=False)
    assert out.shape == (off.shape[1],)
    assert np.all(np.isfinite(out))


def test_routes_agree_at_the_default_truncation():
    p_x, p_y = _random_case(4, 6, 3, 1, seed=841)
    off = _offsets(1)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3)
    mix = sweep_cos_sim_exp_tens(dx, dy, off, method="mixture", verbose=False)
    orb = sweep_cos_sim_exp_tens(dx, dy, off, method="orbit", verbose=False)
    assert np.max(np.abs(mix - orb)) <= 1e-8

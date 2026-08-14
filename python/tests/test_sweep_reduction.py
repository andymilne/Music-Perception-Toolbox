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
    return np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))


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
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


@pytest.mark.parametrize("K,r", [(1, 1), (3, 3), (4, 2)])
def test_sweep_matches_under_default_truncation(K, r):
    """Agreement holds with the truncation floor active, not only without."""
    p_x, p_y = _random_case(K, 6, 3, 2, seed=7)
    off = _offsets(2, swept=(0, 1))
    dx, dy, args = _densities(p_x, p_y, 0.9, r)
    got = sweep_cos_sim_exp_tens(dx, dy, off, verbose=False)
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
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
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
        got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                     verbose=False)
        ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
        assert abs(got[0] - ref[0]) <= PARITY
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3, is_sym=[1])
    assert sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                  verbose=False)[0] == pytest.approx(1.0,
                                                                     abs=1e-12)


@pytest.mark.parametrize("normalize", ["cosine", "oneSidedDenom"])
def test_both_normalisations(normalize):
    p_x, p_y = _random_case(3, 6, 3, 2, seed=21)
    off = _offsets(2, swept=(0, 1))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3)
    got = sweep_cos_sim_exp_tens(dx, dy, off, normalize=normalize,
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
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_unswept_absolute_periodic_attribute():
    """A periodic attribute is supported so long as it is not swept."""
    p_x, p_y = _random_case(3, 6, 3, 2, seed=41)
    off = _offsets(2, swept=(0,))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3, is_per=[0, 1],
                              period=[None, 12.0])
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
    ref = _reference(p_x, p_y, off, args, truncation_sigmas=np.inf)
    assert _rel_dev(got, ref) <= PARITY


def test_no_attribute_swept_is_constant():
    """An all-zero offset column set reduces to a repeated single value."""
    p_x, p_y = _random_case(3, 5, 3, 1, seed=51)
    off = np.zeros((1, 4))
    dx, dy, args = _densities(p_x, p_y, 0.9, 3)
    got = sweep_cos_sim_exp_tens(dx, dy, off, truncation_sigmas=np.inf,
                                 verbose=False)
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


def test_swept_periodic_attribute_is_refused():
    p_x, p_y = _random_case(3, 4, 2, 1, seed=62)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 3, is_per=[1], period=[12.0])
    with pytest.raises(ValueError, match="periodic and swept"):
        sweep_cos_sim_exp_tens(dx, dy, np.array([[0.0, 1.5]]), verbose=False)


def test_relative_periodic_attribute_is_refused_even_unswept():
    """The single-wrap and transposition-average kernels differ there.

    Above sigma/P ~ 0.03 the pairwise and orbit routes compute genuinely
    different measures on a relative-periodic attribute. The reduction's
    fixed factor would silently commit to one, so it declines instead
    and leaves the choice to the per-offset path's ``method``.
    """
    p_x, p_y = _random_case(2, 4, 2, 2, seed=63)
    dx, dy, _ = _densities(p_x, p_y, 0.9, 2, is_rel=[0, 1], is_per=[0, 1],
                           period=[None, 12.0])
    off = np.zeros((2, 2))
    off[0] = [0.0, 1.5]
    with pytest.raises(ValueError, match="both relative and periodic"):
        sweep_cos_sim_exp_tens(dx, dy, off, verbose=False)


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
    entries, _, _ = mpt.translate_attributes(
        p_y, None, [off[a].reshape(1, -1) for a in range(A)])
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
    out, _, _ = mpt.translate_attributes(p_y, None, [2.0])
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
        tagged, _, _ = mpt.translate_attributes(
            p_y, None, [off[a].reshape(1, -1) for a in range(2)],
            specs=[{"rel": False}, {"rel": True}])
    kw = dict(truncation_sigmas=np.inf, verbose=False)
    fast = cos_sim_exp_tens(p_x, None, tagged, None, *args, **kw)
    slow = cos_sim_exp_tens(p_x, None, list(tagged), None, *args, **kw)
    assert _rel_dev(fast, slow) <= PARITY

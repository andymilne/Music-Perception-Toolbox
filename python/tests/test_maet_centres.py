"""Tests for ``maet_centres``, the public tuple-centre accessor.

``build_maet`` builds the per-tuple fields lazily, so a density that
reached its consumer by the Möbius route carries no centres array. The
accessor is the supported way to ask for one: it materialises the
fields when they are absent and passes them through when they are not.
What is pinned here is the shape and content of what it returns, that
it works on a lazy density, and that it rejects anything that is not a
density. The MATLAB twin is ``tests/test_maet_centres.m``.
"""

import numpy as np
import pytest

import mpt
from mpt.tensor import build_maet


P = 1200.0
TRIAD = [0.0, 400.0, 700.0]


def _dens(r=2, is_rel=False, is_per=False):
    return build_maet(TRIAD, None, 10.0, r, is_rel, is_per, P,
                      verbose=False)


def _dens_exch(r=2, is_rel=False, exch=True):
    """The same density by the specs route, which is where ``exch``
    lives."""
    p_attr = [np.asarray(TRIAD).reshape(-1, 1)]
    specs = mpt.flat_specs(p_attr, r=r, rel=is_rel, exch=exch)
    return mpt.build_maet(p_attr, None, specs=specs, sigma=[10.0],
                          is_per=[False], period=[P], verbose=False)


class TestShapeAndContent:

    def test_returns_one_matrix_per_attribute(self):
        C = mpt.maet_centres(_dens())
        assert isinstance(C, list) and len(C) == 1

    def test_absolute_pairs_are_the_ordered_index_pairs(self):
        """r = 2 absolute on three pitches: the six ordered pairs, one
        column each, in the density's own coordinates."""
        C = mpt.maet_centres(_dens(r=2))[0]
        assert C.shape == (2, 6)
        expected = np.array([[a, b] for a in TRIAD for b in TRIAD
                             if a != b]).T
        got = np.array(sorted(map(tuple, C.T)))
        assert np.allclose(got, np.array(sorted(map(tuple, expected.T))))

    def test_relative_centres_lose_a_dimension(self):
        """A relative attribute is one coordinate shorter than its
        tuple size, the translation having been quotiented out."""
        C = mpt.maet_centres(_dens(r=2, is_rel=True))[0]
        assert C.shape[0] == 1

    def test_ordered_tuples_are_the_index_increasing_subsequences(self):
        """Unordered, the r-tuples are every arrangement of the event's
        elements; ordered, only the index-increasing ones."""
        n_unordered = mpt.maet_centres(_dens_exch(exch=True))[0].shape[1]
        n_ordered = mpt.maet_centres(_dens_exch(exch=False))[0].shape[1]
        assert n_unordered == 6 and n_ordered == 3

    def test_centres_are_where_the_density_peaks(self):
        """Each centre carries a kernel, so the density there is at
        least the weight of that one kernel."""
        dens = _dens(r=2)
        C = mpt.maet_centres(dens)[0]
        vals = mpt.eval_maet(dens, C, verbose=False)
        assert np.all(np.asarray(vals) >= 1.0 - 1e-9)


class TestLaziness:

    def test_materialises_a_lazy_density(self):
        dens = _dens()
        assert dens.materialised is False
        C = mpt.maet_centres(dens)
        assert dens.materialised is True
        assert C[0].shape == (2, 6)

    def test_idempotent_on_an_already_built_density(self):
        dens = _dens()
        first = mpt.maet_centres(dens)
        second = mpt.maet_centres(dens)
        assert first[0] is second[0]


class TestValidation:

    @pytest.mark.parametrize("bad", [TRIAD, np.asarray(TRIAD), None, 3.0])
    def test_non_density_input_raises(self, bad):
        with pytest.raises(TypeError, match="maet_centres"):
            mpt.maet_centres(bad)

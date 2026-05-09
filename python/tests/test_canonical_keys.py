"""Unit tests for the canonical-key primitives.

Tests :func:`mpt.tensor._chord_canonical_key` and
:func:`mpt.tensor._pair_canonical_key` in isolation. Coverage:

- All four (is_rel, is_per) combinations.
- Permutation invariance of the input multiset.
- Translation invariance per side (relative modes).
- Joint co-transposition invariance of the pair (absolute modes).
- Cyclic / period-shift invariance (periodic modes).
- Distinct keys for genuinely different chords.
- Hashability of returned keys.
- Returned canonical arrays match the canonical form recorded in the key.
- Density-determining parameters baked into the key (different sigma /
  r / period produce different keys for the same chord input).
- Precision rounding collapses arithmetic noise.

Tests are independent of `batch_cos_sim_exp_tens`'s integration with the
helpers — that integration is covered by the existing test suite, which
runs unchanged after the refactor.
"""

import numpy as np
import pytest

from mpt.tensor import _chord_canonical_key, _pair_canonical_key


# ---------------------------------------------------------------------
# Test fixtures: a small set of (mode, params) configurations.
# ---------------------------------------------------------------------


MODES = [
    pytest.param(False, False, id="abs-nonper"),
    pytest.param(False, True, id="abs-per"),
    pytest.param(True, False, id="rel-nonper"),
    pytest.param(True, True, id="rel-per"),
]


PARAMS = dict(sigma=15.0, r=2, period=1200.0)


# =====================================================================
# _chord_canonical_key
# =====================================================================


class TestChordCanonicalKey:
    """Single-chord canonical key behaviour."""

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_returns_hashable_key(self, is_rel, is_per):
        p = np.array([0.0, 400.0, 700.0])
        key, _, _ = _chord_canonical_key(
            p, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        # Hashable: usable as a dict key.
        d = {key: 1}
        assert d[key] == 1

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_permutation_invariance(self, is_rel, is_per):
        """Reordering pitches at input gives the same canonical key."""
        p1 = np.array([0.0, 400.0, 700.0])
        p2 = np.array([700.0, 0.0, 400.0])
        p3 = np.array([400.0, 700.0, 0.0])
        k1, _, _ = _chord_canonical_key(p1, None, is_rel=is_rel, is_per=is_per, **PARAMS)
        k2, _, _ = _chord_canonical_key(p2, None, is_rel=is_rel, is_per=is_per, **PARAMS)
        k3, _, _ = _chord_canonical_key(p3, None, is_rel=is_rel, is_per=is_per, **PARAMS)
        assert k1 == k2 == k3

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_weights_tracked(self, is_rel, is_per):
        """Weight permutation matches pitch permutation in the canonical form."""
        p = np.array([700.0, 0.0, 400.0])
        w = np.array([3.0, 1.0, 2.0])
        # In the canonical form, pitches are sorted (or sorted-then-shifted),
        # and weights should follow the same sort.
        k_uniform, _, _ = _chord_canonical_key(p, None, is_rel=is_rel, is_per=is_per, **PARAMS)
        k_weighted, _, _ = _chord_canonical_key(p, w, is_rel=is_rel, is_per=is_per, **PARAMS)
        # Different weights => different keys (uniform-weight is None).
        assert k_uniform != k_weighted

    def test_translation_invariance_relative_nonper(self):
        """In relative non-periodic mode, translating the chord doesn't change the key."""
        p1 = np.array([0.0, 400.0, 700.0])
        p2 = np.array([100.0, 500.0, 800.0])  # +100
        p3 = np.array([-50.0, 350.0, 650.0])  # -50
        k1, _, _ = _chord_canonical_key(p1, None, is_rel=True, is_per=False, **PARAMS)
        k2, _, _ = _chord_canonical_key(p2, None, is_rel=True, is_per=False, **PARAMS)
        k3, _, _ = _chord_canonical_key(p3, None, is_rel=True, is_per=False, **PARAMS)
        assert k1 == k2 == k3

    def test_period_invariance_absolute_per(self):
        """In absolute periodic mode, octave-displaced inputs collapse to the same key."""
        p1 = np.array([0.0, 400.0, 700.0])
        p2 = np.array([1200.0, 1600.0, 1900.0])
        k1, _, _ = _chord_canonical_key(p1, None, is_rel=False, is_per=True, **PARAMS)
        k2, _, _ = _chord_canonical_key(p2, None, is_rel=False, is_per=True, **PARAMS)
        assert k1 == k2

    def test_cyclic_invariance_relative_per(self):
        """In relative periodic mode, cyclic rotations within the period
        produce the same key (the cyclic canonical form collapses them)."""
        p1 = np.array([0.0, 400.0, 700.0])
        p2 = np.array([400.0, 700.0, 1200.0])  # rotation by adding 400 then mod
        # Note: rotation in the cyclic sense means each pitch advances the
        # set boundary; this depends on implementation. The cyclic-canonical
        # form should produce the same key for any starting rotation.
        # Easier test: any p2 = (p1 + c) mod 1200 should give the same key.
        c = 400.0
        p2 = np.mod(p1 + c, 1200.0)
        k1, _, _ = _chord_canonical_key(p1, None, is_rel=True, is_per=True, **PARAMS)
        k2, _, _ = _chord_canonical_key(p2, None, is_rel=True, is_per=True, **PARAMS)
        assert k1 == k2

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_distinct_chords_distinct_keys(self, is_rel, is_per):
        """Genuinely different chords give different keys."""
        p_major_triad = np.array([0.0, 400.0, 700.0])
        p_minor_triad = np.array([0.0, 300.0, 700.0])
        k_maj, _, _ = _chord_canonical_key(p_major_triad, None, is_rel=is_rel, is_per=is_per, **PARAMS)
        k_min, _, _ = _chord_canonical_key(p_minor_triad, None, is_rel=is_rel, is_per=is_per, **PARAMS)
        assert k_maj != k_min

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_canonical_arrays_match_key(self, is_rel, is_per):
        """The returned p_canon and w_canon arrays match the pitch / weight
        components of the key."""
        p = np.array([700.0, 0.0, 400.0])
        w = np.array([3.0, 1.0, 2.0])
        key, p_canon, w_canon = _chord_canonical_key(
            p, w, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        ca_p_in_key = key[0]  # first element is canonical pitch tuple
        ca_w_in_key = key[1]
        assert tuple(p_canon) == ca_p_in_key
        if w_canon is None:
            assert ca_w_in_key is None
        else:
            assert tuple(w_canon) == ca_w_in_key

    def test_distinct_sigma_distinct_keys(self):
        """Different sigma values produce different keys for the same chord."""
        p = np.array([0.0, 400.0, 700.0])
        k1, _, _ = _chord_canonical_key(
            p, None, sigma=10.0, r=2, is_rel=True, is_per=True, period=1200.0,
        )
        k2, _, _ = _chord_canonical_key(
            p, None, sigma=20.0, r=2, is_rel=True, is_per=True, period=1200.0,
        )
        assert k1 != k2

    def test_distinct_r_distinct_keys(self):
        """Different r values produce different keys for the same chord."""
        p = np.array([0.0, 400.0, 700.0])
        k1, _, _ = _chord_canonical_key(
            p, None, sigma=15.0, r=2, is_rel=True, is_per=True, period=1200.0,
        )
        k2, _, _ = _chord_canonical_key(
            p, None, sigma=15.0, r=3, is_rel=True, is_per=True, period=1200.0,
        )
        assert k1 != k2

    def test_distinct_modes_distinct_keys(self):
        """Different (is_rel, is_per) combinations give different keys for
        the same chord — they describe structurally different densities."""
        p = np.array([0.0, 400.0, 700.0])
        keys = set()
        for is_rel, is_per in [(False, False), (False, True), (True, False), (True, True)]:
            k, _, _ = _chord_canonical_key(p, None, is_rel=is_rel, is_per=is_per, **PARAMS)
            keys.add(k)
        assert len(keys) == 4

    def test_precision_rounding(self):
        """Precision rounding collapses near-identical canonical forms."""
        p1 = np.array([0.0, 400.0, 700.0])
        p2 = np.array([0.0, 400.0 + 1e-10, 700.0 - 1e-10])  # FP noise
        k1, _, _ = _chord_canonical_key(
            p1, None, is_rel=False, is_per=False, precision=4, **PARAMS,
        )
        k2, _, _ = _chord_canonical_key(
            p2, None, is_rel=False, is_per=False, precision=4, **PARAMS,
        )
        assert k1 == k2


# =====================================================================
# _pair_canonical_key
# =====================================================================


class TestPairCanonicalKey:
    """Paired canonical-key behaviour."""

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_returns_hashable_keys(self, is_rel, is_per):
        p_a = np.array([0.0, 400.0, 700.0])
        p_b = np.array([0.0, 300.0, 700.0])
        ka, kb, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        d = {(ka, kb): 1}
        assert d[(ka, kb)] == 1

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_permutation_invariance(self, is_rel, is_per):
        """Permuting input order on either side gives the same pair key."""
        p_a = np.array([0.0, 400.0, 700.0])
        p_a_perm = np.array([700.0, 0.0, 400.0])
        p_b = np.array([0.0, 300.0])
        p_b_perm = np.array([300.0, 0.0])

        ka1, kb1, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        ka2, kb2, *_ = _pair_canonical_key(
            p_a_perm, None, p_b_perm, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        assert ka1 == ka2
        assert kb1 == kb2

    def test_independent_translation_relative(self):
        """Relative mode: translating either side independently preserves both keys."""
        p_a = np.array([0.0, 400.0, 700.0])
        p_b = np.array([0.0, 300.0])

        ka1, kb1, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=True, is_per=False, **PARAMS,
        )
        # Translate A by +100, B by -50, independently.
        ka2, kb2, *_ = _pair_canonical_key(
            p_a + 100.0, None, p_b - 50.0, None, is_rel=True, is_per=False, **PARAMS,
        )
        assert ka1 == ka2
        assert kb1 == kb2

    def test_co_transposition_absolute_nonper(self):
        """Absolute non-periodic: jointly translating (A, B) by the same c
        preserves the pair key."""
        p_a = np.array([0.0, 400.0, 700.0])
        p_b = np.array([200.0, 600.0])

        ka1, kb1, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=False, is_per=False, **PARAMS,
        )
        c = 50.0
        ka2, kb2, *_ = _pair_canonical_key(
            p_a + c, None, p_b + c, None, is_rel=False, is_per=False, **PARAMS,
        )
        assert ka1 == ka2
        assert kb1 == kb2

    def test_independent_translation_breaks_absolute(self):
        """Absolute mode: translating only one side gives different keys
        (since absolute encodes A-B relative position)."""
        p_a = np.array([0.0, 400.0, 700.0])
        p_b = np.array([200.0, 600.0])

        ka1, kb1, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=False, is_per=False, **PARAMS,
        )
        # Translate only A. In absolute mode, this changes the (A, B) relationship.
        ka2, kb2, *_ = _pair_canonical_key(
            p_a + 50.0, None, p_b, None, is_rel=False, is_per=False, **PARAMS,
        )
        # In absolute mode, A determines the joint shift, so ka1 == ka2 always
        # (A on its own is canonicalised the same way). But kb should differ.
        assert kb1 != kb2

    def test_co_octave_absolute_per(self):
        """Absolute periodic: jointly shifting (A, B) by an octave preserves both keys."""
        p_a = np.array([0.0, 400.0, 700.0])
        p_b = np.array([200.0, 600.0])
        ka1, kb1, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=False, is_per=True, **PARAMS,
        )
        ka2, kb2, *_ = _pair_canonical_key(
            p_a + 1200.0, None, p_b + 1200.0, None, is_rel=False, is_per=True, **PARAMS,
        )
        assert ka1 == ka2
        assert kb1 == kb2

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_canonical_arrays_match_keys(self, is_rel, is_per):
        """Returned canonical numpy arrays match the corresponding key components."""
        p_a = np.array([700.0, 0.0, 400.0])
        w_a = np.array([3.0, 1.0, 2.0])
        p_b = np.array([300.0, 0.0])

        ka, kb, p_a_canon, w_a_canon, p_b_canon, w_b_canon = _pair_canonical_key(
            p_a, w_a, p_b, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        # First two key components are (canonical_pitch_tuple, canonical_weight_tuple_or_None)
        assert tuple(p_a_canon) == ka[0]
        assert tuple(w_a_canon) == ka[1]
        assert tuple(p_b_canon) == kb[0]
        assert kb[1] is None  # w_b was None
        assert w_b_canon is None

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_distinct_pairs_distinct_keys(self, is_rel, is_per):
        """Genuinely different pairs give different (ka, kb) keys."""
        p_a = np.array([0.0, 400.0, 700.0])
        p_b1 = np.array([0.0, 300.0])
        p_b2 = np.array([0.0, 500.0])
        ka1, kb1, *_ = _pair_canonical_key(
            p_a, None, p_b1, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        ka2, kb2, *_ = _pair_canonical_key(
            p_a, None, p_b2, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        assert ka1 == ka2  # same A
        assert kb1 != kb2  # different B

    @pytest.mark.parametrize("is_rel,is_per", MODES)
    def test_consistency_with_chord_canonical_key_relative(self, is_rel, is_per):
        """In relative mode, _pair_canonical_key on (A, B) should produce the
        same individual keys as _chord_canonical_key on A and B independently."""
        if not is_rel:
            pytest.skip("Independence holds only in relative mode.")
        p_a = np.array([0.0, 400.0, 700.0])
        p_b = np.array([200.0, 600.0])

        ka_chord, _, _ = _chord_canonical_key(
            p_a, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        kb_chord, _, _ = _chord_canonical_key(
            p_b, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        ka_pair, kb_pair, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=is_rel, is_per=is_per, **PARAMS,
        )
        assert ka_chord == ka_pair
        assert kb_chord == kb_pair

    def test_precision_rounding_pair(self):
        """Precision rounding collapses pairs that differ only by FP noise."""
        p_a = np.array([0.0, 400.0, 700.0])
        p_b = np.array([200.0, 600.0])
        ka1, kb1, *_ = _pair_canonical_key(
            p_a, None, p_b, None, is_rel=False, is_per=False,
            precision=4, **PARAMS,
        )
        ka2, kb2, *_ = _pair_canonical_key(
            p_a + 1e-10, None, p_b + 1e-10, None, is_rel=False, is_per=False,
            precision=4, **PARAMS,
        )
        assert ka1 == ka2
        assert kb1 == kb2

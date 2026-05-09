"""Tests for the polymorphic input semantics of :func:`cos_sim_exp_tens`.

Coverage:

- Scalar - vs - scalar (the v2.0 case): output is a Python ``float``,
  numerically identical to pre-polymorphic behaviour.
- Length-1 list collapses to scalar (per spec §3.4): same return shape
  as the scalar case.
- Empty list returns empty array of correct shape.
- Scalar - vs - list / list - vs - scalar (broadcast) returns a 1-D
  array of correct length.
- List - vs - list with ``mode='pairwise'``: equal-length required,
  returns 1-D array.
- List - vs - list with ``mode='cartesian'``: returns 2-D array of
  shape ``(M, N)``.
- ``mode='auto'`` resolves to ``'pairwise'`` for equal lengths,
  otherwise raises with a message suggesting ``'cartesian'``.
- ``dedup=True`` (the default) produces results numerically identical
  to ``dedup=False`` — dedup is correctness-preserving.
- ``dedup=False`` is honoured.
- Output shape checks across all input combinations.
- Mixed density types in a list raises.

Tests use single-attribute (ExpTensDensity) inputs throughout; MA
dedup is bypassed transparently and is exercised in a single
correctness-only check.
"""

import numpy as np
import pytest

from mpt import build_exp_tens, cos_sim_exp_tens


# ---------------------------------------------------------------------
# Fixtures: a small library of densities for cosine similarity testing.
# ---------------------------------------------------------------------


@pytest.fixture
def density_set():
    """Return a small set of distinct SA densities for testing."""
    sigma, r = 15.0, 2
    period = 1200.0
    chords = {
        "major": np.array([0.0, 400.0, 700.0]),
        "minor": np.array([0.0, 300.0, 700.0]),
        "dim":   np.array([0.0, 300.0, 600.0]),
        "aug":   np.array([0.0, 400.0, 800.0]),
        # A transposed major triad — should canonicalise to the same key
        # as 'major' in absolute periodic mode.
        "major_transposed": np.array([200.0, 600.0, 900.0]),
    }
    densities = {
        name: build_exp_tens(
            p, np.ones_like(p), sigma, r, False, True, period,
            verbose=False,
        )
        for name, p in chords.items()
    }
    return densities


# =====================================================================
# Scalar - vs - scalar (the v2.0 case)
# =====================================================================


class TestScalarScalar:
    """The original v2.0 case must be unchanged."""

    def test_returns_python_float(self, density_set):
        s = cos_sim_exp_tens(density_set["major"], density_set["minor"], verbose=False)
        # Spec: scalar-vs-scalar returns scalar (the v2.0 case).
        assert isinstance(s, float)

    def test_self_cosine_is_one(self, density_set):
        s = cos_sim_exp_tens(density_set["major"], density_set["major"], verbose=False)
        assert s == pytest.approx(1.0)

    def test_distinct_chords_below_one(self, density_set):
        s = cos_sim_exp_tens(density_set["major"], density_set["dim"], verbose=False)
        assert 0.0 < s < 1.0

    def test_symmetry(self, density_set):
        s_xy = cos_sim_exp_tens(density_set["major"], density_set["minor"], verbose=False)
        s_yx = cos_sim_exp_tens(density_set["minor"], density_set["major"], verbose=False)
        assert s_xy == pytest.approx(s_yx)


# =====================================================================
# Length-1 list inputs (Option II: strict shape preservation, no collapse)
# =====================================================================


class TestLengthOneNoCollapse:
    """Under Option II, length-1 lists do NOT collapse to scalars.

    A genuine scalar density input produces a scalar return; a length-1
    list (or any list) produces an ndarray return. This pairs naturally
    with the row-vector vs column-vector orientation rule for raw input
    (a row vector means one chord; a 1-row matrix means a length-1
    batch). Strict shape preservation matches NumPy conventions and
    avoids surprising shape-promotion behaviour.
    """

    def test_len1_x_returns_array(self, density_set):
        s = cos_sim_exp_tens(
            [density_set["major"]], density_set["minor"], verbose=False,
        )
        assert isinstance(s, np.ndarray)
        assert s.shape == (1,)

    def test_len1_y_returns_array(self, density_set):
        s = cos_sim_exp_tens(
            density_set["major"], [density_set["minor"]], verbose=False,
        )
        assert isinstance(s, np.ndarray)
        assert s.shape == (1,)

    def test_both_len1_returns_array(self, density_set):
        s = cos_sim_exp_tens(
            [density_set["major"]], [density_set["minor"]], verbose=False,
        )
        assert isinstance(s, np.ndarray)
        assert s.shape == (1,)

    def test_len1_value_matches_scalar(self, density_set):
        """Numerical value (single element of the (1,) array) matches the
        scalar-input result."""
        s_scalar = cos_sim_exp_tens(
            density_set["major"], density_set["minor"], verbose=False,
        )
        s_len1 = cos_sim_exp_tens(
            [density_set["major"]], [density_set["minor"]], verbose=False,
        )
        assert s_len1[0] == pytest.approx(s_scalar)


# =====================================================================
# Empty inputs
# =====================================================================


class TestEmptyInputs:
    """Empty lists return empty arrays of the appropriate shape."""

    def test_empty_x_scalar_y(self, density_set):
        result = cos_sim_exp_tens([], density_set["major"], verbose=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_scalar_x_empty_y(self, density_set):
        result = cos_sim_exp_tens(density_set["major"], [], verbose=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_empty_pairwise(self, density_set):
        result = cos_sim_exp_tens([], [], mode="pairwise", verbose=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_empty_cartesian(self, density_set):
        result = cos_sim_exp_tens([], [], mode="cartesian", verbose=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)

    def test_empty_x_list_y_cartesian(self, density_set):
        result = cos_sim_exp_tens(
            [], [density_set["major"], density_set["minor"]],
            mode="cartesian", verbose=False,
        )
        assert result.shape == (0, 2)

    def test_list_x_empty_y_cartesian(self, density_set):
        result = cos_sim_exp_tens(
            [density_set["major"], density_set["minor"]], [],
            mode="cartesian", verbose=False,
        )
        assert result.shape == (2, 0)


# =====================================================================
# Broadcast (scalar-vs-list, list-vs-scalar)
# =====================================================================


class TestBroadcast:
    """Broadcast: a single density compared against a list."""

    def test_scalar_x_list_y_shape(self, density_set):
        result = cos_sim_exp_tens(
            density_set["major"],
            [density_set["minor"], density_set["dim"], density_set["aug"]],
            verbose=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_list_x_scalar_y_shape(self, density_set):
        result = cos_sim_exp_tens(
            [density_set["major"], density_set["minor"], density_set["dim"]],
            density_set["aug"],
            verbose=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_broadcast_values_match_individual_calls(self, density_set):
        """Values produced by broadcast match values from per-pair scalar calls."""
        targets = ["minor", "dim", "aug"]
        ref = np.array([
            cos_sim_exp_tens(density_set["major"], density_set[t], verbose=False)
            for t in targets
        ])
        result = cos_sim_exp_tens(
            density_set["major"], [density_set[t] for t in targets],
            verbose=False,
        )
        np.testing.assert_allclose(result, ref)

    def test_broadcast_symmetry(self, density_set):
        """Broadcast is symmetric: x vs list-y equals list-y-as-x vs x."""
        targets = ["minor", "dim", "aug"]
        r1 = cos_sim_exp_tens(
            density_set["major"], [density_set[t] for t in targets],
            verbose=False,
        )
        r2 = cos_sim_exp_tens(
            [density_set[t] for t in targets], density_set["major"],
            verbose=False,
        )
        np.testing.assert_allclose(r1, r2)


# =====================================================================
# List-vs-list, mode='pairwise' / 'cartesian' / 'auto'
# =====================================================================


class TestListListPairwise:
    def test_pairwise_shape_equal_lengths(self, density_set):
        a_list = [density_set["major"], density_set["minor"], density_set["dim"]]
        b_list = [density_set["aug"], density_set["dim"], density_set["minor"]]
        result = cos_sim_exp_tens(a_list, b_list, mode="pairwise", verbose=False)
        assert result.shape == (3,)

    def test_pairwise_unequal_lengths_raises(self, density_set):
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"], density_set["minor"]]
        with pytest.raises(ValueError, match="pairwise"):
            cos_sim_exp_tens(a_list, b_list, mode="pairwise", verbose=False)

    def test_pairwise_values_match_scalar_calls(self, density_set):
        a_list = [density_set["major"], density_set["minor"], density_set["dim"]]
        b_list = [density_set["aug"], density_set["dim"], density_set["minor"]]
        result = cos_sim_exp_tens(a_list, b_list, mode="pairwise", verbose=False)
        ref = np.array([
            cos_sim_exp_tens(a, b, verbose=False) for a, b in zip(a_list, b_list)
        ])
        np.testing.assert_allclose(result, ref)


class TestListListCartesian:
    def test_cartesian_shape(self, density_set):
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"], density_set["minor"]]
        result = cos_sim_exp_tens(a_list, b_list, mode="cartesian", verbose=False)
        assert result.shape == (2, 3)

    def test_cartesian_values(self, density_set):
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"], density_set["minor"]]
        result = cos_sim_exp_tens(a_list, b_list, mode="cartesian", verbose=False)
        for i, a in enumerate(a_list):
            for j, b in enumerate(b_list):
                expected = cos_sim_exp_tens(a, b, verbose=False)
                assert result[i, j] == pytest.approx(expected)

    def test_cartesian_equal_length_works(self, density_set):
        """cartesian mode works for equal-length lists too."""
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"]]
        result = cos_sim_exp_tens(a_list, b_list, mode="cartesian", verbose=False)
        assert result.shape == (2, 2)


class TestListListAuto:
    def test_auto_pairwise_equal_lengths(self, density_set):
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"]]
        result = cos_sim_exp_tens(a_list, b_list, verbose=False)  # mode='auto'
        assert result.shape == (2,)

    def test_auto_unequal_lengths_raises(self, density_set):
        # Both lists need length ≥ 2 to avoid length-1 collapse to scalar.
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"], density_set["minor"]]
        with pytest.raises(ValueError, match="cartesian"):
            cos_sim_exp_tens(a_list, b_list, verbose=False)

    def test_auto_with_invalid_mode_string(self, density_set):
        # Need length ≥ 2 lists for the mode kwarg to be inspected;
        # length-1 lists collapse to scalar before mode resolution.
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"]]
        with pytest.raises(ValueError, match="mode must be"):
            cos_sim_exp_tens(a_list, b_list, mode="weird", verbose=False)


# =====================================================================
# Dedup behaviour
# =====================================================================


class TestDedup:
    """Dedup is correctness-preserving: same numerical results with or without."""

    def test_dedup_default_matches_no_dedup(self, density_set):
        """Default (dedup=True) produces identical results to dedup=False."""
        a_list = [density_set["major"], density_set["minor"], density_set["major_transposed"]]
        b_list = [density_set["minor"], density_set["minor"], density_set["minor"]]
        r_dedup = cos_sim_exp_tens(a_list, b_list, mode="pairwise", verbose=False)
        r_no_dedup = cos_sim_exp_tens(
            a_list, b_list, mode="pairwise", dedup=False, verbose=False,
        )
        np.testing.assert_allclose(r_dedup, r_no_dedup)

    def test_dedup_recognises_jointly_transposed_pair(self, density_set):
        """In absolute periodic mode, the cosine of a *jointly* transposed
        pair is identical to the original. With dedup, the canonical
        keys collide and the IP is computed once.

        major-vs-minor and (major+200)-vs-(minor+200) should produce
        identical cosines, since absolute periodic cosine is invariant
        under joint co-transposition. The test verifies this numerical
        identity (the dedup itself is internal, but the equivalence is
        the property that justifies it)."""
        sigma, r, period = 15.0, 2, 1200.0

        major = density_set["major"]
        minor = density_set["minor"]
        # Build a jointly co-transposed pair.
        major_shifted_chord = np.array([200.0, 600.0, 900.0])
        minor_shifted_chord = np.array([200.0, 500.0, 900.0])
        major_shifted = build_exp_tens(
            major_shifted_chord, np.ones(3), sigma, r, False, True, period,
            verbose=False,
        )
        minor_shifted = build_exp_tens(
            minor_shifted_chord, np.ones(3), sigma, r, False, True, period,
            verbose=False,
        )

        s1 = cos_sim_exp_tens(major, minor, verbose=False)
        s2 = cos_sim_exp_tens(major_shifted, minor_shifted, verbose=False)
        assert s1 == pytest.approx(s2)

        # Broadcast call: both pairs canonicalise to the same key, dedup
        # computes once, both result positions get the same value.
        result = cos_sim_exp_tens(
            [major, major_shifted], [minor, minor_shifted],
            mode="pairwise", verbose=False,
        )
        np.testing.assert_allclose(result, [s1, s2])

    def test_dedup_false_explicit(self, density_set):
        """Passing dedup=False bypasses canonical-form dedup but produces
        the same numerical results."""
        a_list = [density_set["major"], density_set["minor"]]
        b_list = [density_set["aug"], density_set["dim"]]
        r1 = cos_sim_exp_tens(
            a_list, b_list, mode="pairwise", dedup=False, verbose=False,
        )
        r2 = cos_sim_exp_tens(
            a_list, b_list, mode="pairwise", dedup=True, verbose=False,
        )
        np.testing.assert_allclose(r1, r2)


# =====================================================================
# Mixed types / errors
# =====================================================================


class TestErrors:
    def test_invalid_density_in_list_raises(self, density_set):
        # Mixed list with one density and one non-density routes to
        # density-list path and raises via _normalize_density_input.
        with pytest.raises(TypeError, match="ExpTensDensity"):
            cos_sim_exp_tens(
                [density_set["major"], "not a density"],
                density_set["minor"], verbose=False,
            )

    def test_invalid_single_first_arg_raises(self, density_set):
        # A string first argument with only 2 positional args fails the
        # arg-count check for raw SA mode (9 expected) — the simplest
        # and clearest error in this case.
        with pytest.raises(TypeError, match="positional"):
            cos_sim_exp_tens(
                "not a density", density_set["minor"], verbose=False,
            )

    def test_invalid_single_first_arg_with_full_args_raises(self, density_set):
        # A string first argument with the right number of positional
        # args fails type coercion and reports the type problem.
        with pytest.raises(TypeError, match="density object"):
            cos_sim_exp_tens(
                "not a density", None, [0.0, 4.0, 7.0], None,
                15.0, 2, False, True, 1200.0,
                verbose=False,
            )

    def test_too_few_args_raises(self):
        with pytest.raises(TypeError, match="at least 2 positional"):
            cos_sim_exp_tens(verbose=False)

    def test_density_mode_with_extra_args_raises(self, density_set):
        # Density input expects exactly 2 positional args.
        with pytest.raises(TypeError, match="2 positional"):
            cos_sim_exp_tens(
                density_set["major"], density_set["minor"], 15.0,
                verbose=False,
            )

    def test_spectrum_with_density_raises(self, density_set):
        with pytest.raises(TypeError, match="spectrum"):
            cos_sim_exp_tens(
                density_set["major"], density_set["minor"],
                spectrum=("harmonic", 6, "geometric", 0.7),
                verbose=False,
            )

    def test_precision_with_density_raises(self, density_set):
        with pytest.raises(TypeError, match="precision"):
            cos_sim_exp_tens(
                density_set["major"], density_set["minor"],
                precision=4, verbose=False,
            )

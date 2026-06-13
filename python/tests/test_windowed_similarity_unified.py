"""Tests for the unified :func:`windowed_tensor_similarity`.

Coverage:

- Scalar-vs-scalar (the v2.0 case) — output shape ``(M,)``.
- Scalar-vs-list and list-vs-scalar broadcast — output ``(N, M)``.
- List-vs-list pairwise (``mode='pairwise'`` or ``'auto'`` with equal
  lengths) — output ``(N, M)``.
- List-vs-list cartesian — output ``(n_q, n_c, M)``.
- Empty lists.
- Length-1 lists (Option II — strict shape preservation).
- Reference: ``None``, shared form (length-``n_attrs`` per-attribute
  arrays), and per-query form (length-``n_q`` of per-attribute lists).
- Errors.

The MaetDensity setup uses a (time, pitch) two-attribute density —
the same pattern as the existing windowed-similarity tests in
test_mpt.py.
"""

import warnings

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, windowed_tensor_similarity


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _make_time_pitch_dens(events, sigma_pitch=10.0, sigma_time=0.05):
    """Build a (pitch, time) MaetDensity from a list of (pitch, time) events.

    Pitch is non-periodic; time is non-periodic. r=1 on each.
    """
    pitches = np.array([[p for p, _ in events]])    # (1, n_events)
    times = np.array([[t for _, t in events]])        # (1, n_events)
    return build_exp_tens(
        [pitches, times], None,
        [sigma_pitch, sigma_time], [1, 1], 
        [False, False], [False, False], [0.0, 0.0],
        verbose=False,
    )


@pytest.fixture
def context_dens():
    """A 4-event context: ascending pitches at unit-spaced times."""
    return _make_time_pitch_dens(
        [(60.0, 0.0), (62.0, 1.0), (64.0, 2.0), (65.0, 3.0)],
    )


@pytest.fixture
def queries_three():
    """Three single-event queries at different pitches."""
    return [
        _make_time_pitch_dens([(60.0, 0.0)]),
        _make_time_pitch_dens([(62.0, 0.0)]),
        _make_time_pitch_dens([(64.0, 0.0)]),
    ]


@pytest.fixture
def offsets_grid():
    """Sweep along time at fixed pitch offset 0; M=11 points."""
    n = 11
    offs = np.zeros((2, n))
    offs[1, :] = np.linspace(0.0, 3.0, n)
    return offs


@pytest.fixture
def window_spec():
    """Narrow Gaussian on time, infinite on pitch."""
    return {"size": [np.inf, 0.3], "mix": [0.0, 0.0]}


# =====================================================================
# Scalar query × scalar context (v2.0 unchanged)
# =====================================================================


class TestScalarScalar:
    def test_returns_1d(self, context_dens, queries_three, offsets_grid, window_spec):
        prof = windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
            verbose=False,
        )
        assert isinstance(prof, np.ndarray)
        assert prof.shape == (11,)

    def test_truncation_and_precision_kwargs_threaded(
        self, queries_three, context_dens, offsets_grid, window_spec,
    ):
        """v2.2.x: truncation_sigmas and kernel_precision are threaded
        through to the per-offset cos_sim_exp_tens calls (replacing
        the v2.2.0 temporary-defaults stop-gap). At
        truncation_sigmas=inf and kernel_precision='double', the result
        must be identical to the default-mode call. At a tight
        truncation (e.g., 6 sigma), the result must match the default
        to numerical precision."""
        prof_default = windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
            verbose=False,
        )
        # Explicit "no-op" kwargs.
        prof_inf = windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
            truncation_sigmas=float("inf"), kernel_precision="double",
            verbose=False,
        )
        np.testing.assert_array_equal(prof_default, prof_inf)
        # Tight truncation: still matches default to high precision.
        prof_trunc = windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
            truncation_sigmas=6.0, verbose=False,
        )
        np.testing.assert_allclose(prof_default, prof_trunc, atol=1e-12)

    def test_peak_location_is_meaningful(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        # Query is pitch 60 at t=0; should peak at t=0 in the context
        # (which has 60 at t=0). With the auto-centroid reference (μ_q
        # for time = 0 here), peak offset ≈ 0.
        prof = windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
            verbose=False,
        )
        peak_idx = int(np.argmax(prof))
        assert offsets_grid[1, peak_idx] < 0.5


# =====================================================================
# Scalar × list (broadcast)
# =====================================================================


class TestScalarVsList:
    def test_scalar_vs_list_contexts(
        self, queries_three, offsets_grid, window_spec,
    ):
        c1 = _make_time_pitch_dens([(60.0, 0.0), (62.0, 1.0)])
        c2 = _make_time_pitch_dens([(60.0, 0.0), (64.0, 2.0)])
        out = windowed_tensor_similarity([c1, c2], queries_three[0], window_spec, offsets_grid,
            verbose=False,
        )
        assert out.shape == (2, 11)

    def test_list_vs_scalar_queries(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        out = windowed_tensor_similarity(context_dens, queries_three, window_spec, offsets_grid,
            verbose=False,
        )
        assert out.shape == (3, 11)

    def test_broadcast_matches_per_pair(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        out = windowed_tensor_similarity(context_dens, queries_three, window_spec, offsets_grid,
            verbose=False,
        )
        for i, q in enumerate(queries_three):
            ref_prof = windowed_tensor_similarity(context_dens, q, window_spec, offsets_grid, verbose=False,
            )
            np.testing.assert_allclose(out[i], ref_prof, atol=1e-12)


# =====================================================================
# List × list (pairwise / cartesian / auto)
# =====================================================================


class TestListVsList:
    def test_pairwise_explicit(
        self, queries_three, offsets_grid, window_spec,
    ):
        contexts = [
            _make_time_pitch_dens([(60.0, 0.0), (62.0, 1.0), (64.0, 2.0)]),
            _make_time_pitch_dens([(62.0, 0.0), (64.0, 1.0), (65.0, 2.0)]),
            _make_time_pitch_dens([(64.0, 0.0), (65.0, 1.0), (67.0, 2.0)]),
        ]
        out = windowed_tensor_similarity(contexts, queries_three, window_spec, offsets_grid,
            mode="pairwise", verbose=False,
        )
        assert out.shape == (3, 11)
        # Each row matches the per-pair call.
        for i in range(3):
            ref = windowed_tensor_similarity(contexts[i], queries_three[i], window_spec, offsets_grid,
                verbose=False,
            )
            np.testing.assert_allclose(out[i], ref, atol=1e-12)

    def test_auto_mode_resolves_pairwise_when_lengths_match(
        self, queries_three, offsets_grid, window_spec,
    ):
        contexts = [_make_time_pitch_dens([(p, 0.0)]) for p in [60.0, 62.0, 64.0]]
        out_auto = windowed_tensor_similarity(contexts, queries_three, window_spec, offsets_grid,
            verbose=False,
        )
        out_pairwise = windowed_tensor_similarity(contexts, queries_three, window_spec, offsets_grid,
            mode="pairwise", verbose=False,
        )
        np.testing.assert_array_equal(out_auto, out_pairwise)

    def test_auto_mode_raises_when_lengths_differ(
        self, queries_three, offsets_grid, window_spec,
    ):
        contexts = [_make_time_pitch_dens([(p, 0.0)]) for p in [60.0, 62.0]]
        with pytest.raises(ValueError, match="cartesian"):
            windowed_tensor_similarity(contexts, queries_three, window_spec, offsets_grid,
                verbose=False,
            )

    def test_cartesian(
        self, queries_three, offsets_grid, window_spec,
    ):
        contexts = [_make_time_pitch_dens([(p, 0.0)]) for p in [60.0, 62.0]]
        out = windowed_tensor_similarity(contexts, queries_three, window_spec, offsets_grid,
            mode="cartesian", verbose=False,
        )
        # New convention: positional arg 1 = context, arg 2 = query;
        # cartesian output shape is (n_c, n_q, M).
        assert out.shape == (2, 3, 11)
        # Each (i, j) row matches the per-pair call with context i,
        # query j.
        for i in range(2):
            for j in range(3):
                ref = windowed_tensor_similarity(contexts[i], queries_three[j], window_spec, offsets_grid,
                    verbose=False,
                )
                np.testing.assert_allclose(out[i, j], ref, atol=1e-12)


# =====================================================================
# Edge cases: length-1 and empty
# =====================================================================


class TestEdgeCases:
    def test_length_1_query_list(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        """Option II: length-1 list returns (1, M), not (M,)."""
        out = windowed_tensor_similarity(context_dens, [queries_three[0]], window_spec, offsets_grid,
            verbose=False,
        )
        assert out.shape == (1, 11)

    def test_length_1_context_list(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        out = windowed_tensor_similarity([context_dens], queries_three[0], window_spec, offsets_grid,
            verbose=False,
        )
        assert out.shape == (1, 11)

    def test_empty_query_list(
        self, context_dens, offsets_grid, window_spec,
    ):
        out = windowed_tensor_similarity(context_dens, [], window_spec, offsets_grid, verbose=False,
        )
        assert out.shape == (0, 11)

    def test_empty_context_list(
        self, queries_three, offsets_grid, window_spec,
    ):
        out = windowed_tensor_similarity([], queries_three[0], window_spec, offsets_grid, verbose=False,
        )
        assert out.shape == (0, 11)


# =====================================================================
# Reference handling
# =====================================================================


class TestReference:
    def test_reference_none_default(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        # Just checks that None default still works in scalar mode.
        out = windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
            reference=None, verbose=False,
        )
        assert out.shape == (11,)

    def test_reference_shared_form(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        # Shared reference: list of n_attrs (= 2) per-attribute arrays.
        ref = [np.array([60.0]), np.array([0.0])]
        out_shared = windowed_tensor_similarity(context_dens, queries_three, window_spec, offsets_grid,
            reference=ref, verbose=False,
        )
        # Compare against per-query calls with the same shared reference.
        for i, q in enumerate(queries_three):
            ref_prof = windowed_tensor_similarity(context_dens, q, window_spec, offsets_grid,
                reference=ref, verbose=False,
            )
            np.testing.assert_allclose(out_shared[i], ref_prof, atol=1e-12)

    def test_reference_per_query_form(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        # Per-query: outer length = n_q (= 3); each element is a list of
        # n_attrs (= 2) 1-D arrays.
        per_query_refs = [
            [np.array([60.0]), np.array([0.0])],
            [np.array([62.0]), np.array([0.0])],
            [np.array([64.0]), np.array([0.0])],
        ]
        out_per_q = windowed_tensor_similarity(context_dens, queries_three, window_spec, offsets_grid,
            reference=per_query_refs, verbose=False,
        )
        for i, (q, ref_q) in enumerate(zip(queries_three, per_query_refs)):
            ref_prof = windowed_tensor_similarity(context_dens, q, window_spec, offsets_grid,
                reference=ref_q, verbose=False,
            )
            np.testing.assert_allclose(out_per_q[i], ref_prof, atol=1e-12)

    def test_reference_shared_in_scalar_mode(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        """The shared (single per-attribute list) form also works in
        scalar-vs-scalar mode, matching the v2.0 reference behaviour."""
        ref = [np.array([60.0]), np.array([0.0])]
        out = windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
            reference=ref, verbose=False,
        )
        assert out.shape == (11,)


# =====================================================================
# Errors
# =====================================================================


class TestErrors:
    def test_invalid_query_type_raises(
        self, context_dens, offsets_grid, window_spec,
    ):
        # ExpTensDensity is not allowed.
        sa = mpt.build_exp_tens(
            [60.0, 62.0, 64.0], None, 10.0, 1, False, True, 12.0,
            verbose=False,
        )
        with pytest.raises(TypeError, match="MaetDensity"):
            windowed_tensor_similarity(context_dens, sa, window_spec, offsets_grid, verbose=False,
            )

    def test_invalid_context_type_raises(
        self, queries_three, offsets_grid, window_spec,
    ):
        sa = mpt.build_exp_tens(
            [60.0, 62.0, 64.0], None, 10.0, 1, False, True, 12.0,
            verbose=False,
        )
        with pytest.raises(TypeError, match="MaetDensity"):
            windowed_tensor_similarity(sa, queries_three[0], window_spec, offsets_grid,
                verbose=False,
            )

    def test_per_query_reference_wrong_length_raises(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        # 2 references for 3 queries.
        bad = [
            [np.array([60.0]), np.array([0.0])],
            [np.array([62.0]), np.array([0.0])],
        ]
        with pytest.raises(ValueError, match="length n_q = 3"):
            windowed_tensor_similarity(context_dens, queries_three, window_spec, offsets_grid,
                reference=bad, verbose=False,
            )

    def test_shared_reference_wrong_length_raises(
        self, context_dens, queries_three, offsets_grid, window_spec,
    ):
        # 1 entry for 2 attributes.
        bad = [np.array([60.0])]
        with pytest.raises(ValueError, match="2 entries"):
            windowed_tensor_similarity(context_dens, queries_three[0], window_spec, offsets_grid,
                reference=bad, verbose=False,
            )

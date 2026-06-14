"""Tests for nested-carrier support in ``windowed_similarity`` /
``windowed_entropy``.

Each nested result is pinned to the explicit composition it stands in for
(``weight_events`` / ``translate_attributes`` -> ``build_exp_tens`` ->
``cos_sim_exp_tens`` / ``entropy_exp_tens`` with ``specs=``), exactly as
``test_windowed_premaet.py`` pins the flat path. To prove the nested
geometry is read from ``specs`` and not from the positional ``r``/``is_rel``,
the calls pass deliberately wrong flat ``r``/``is_rel`` and still match.
"""
import numpy as np
import pytest

from mpt import (
    add_spectra, bind_events,
    windowed_similarity, windowed_entropy,
    weight_events, translate_attributes, build_exp_tens,
    cos_sim_exp_tens, entropy_exp_tens,
)

SIG_P, SIG_T = 0.15, 0.125
AXIS, TARGET = 1, 0          # window on time (1), target the bound pitch (0)


def _carrier(spectral):
    """A short monophonic passage with a clean (+3, -3, +5) statement, bound
    into ordered relative four-note super-events (inner partial multiset when
    spectral), plus a flat onset-time axis. Returns (ctx, w_ctx, specs)."""
    N = 16
    pit = (np.array([0, 2, 4, 5, 7, 5, 4, 2, 0, 3, 0, 5, 7, 9, 7, 5],
                    dtype=float) + 60.0)        # notes 8..11 are (+3, -3, +5)
    on = np.cumsum(np.full(N, 0.5))
    if spectral:
        Kp = 8
        pp, wp = add_spectra(pit, None, 'harmonic', Kp, 'powerlaw', 1.0, units=12.0)
        pitch_attr, w_attr = pp.reshape(N, Kp).T, wp.reshape(N, Kp).T
    else:
        pitch_attr, w_attr = pit.reshape(1, N), None
    pb, wb, sb = bind_events([pitch_attr, on.reshape(1, N)], [w_attr, None],
                             [4, 1], step=1, rel_outer=True)
    return pb, wb, sb


def _ref_locked(ctx, w_ctx, qry, w_qry, specs, centres, q_ext):
    """Inline hand-built locked-sweep reference (the composition the nested
    windowed_similarity replaces)."""
    sigma, is_per, period = [SIG_P, SIG_T], [False, False], [0.0, 0.0]
    mu_q = float(np.nanmean(np.asarray(qry[AXIS], dtype=float)))
    out = np.empty(len(centres))
    for i, c in enumerate(centres):
        pc, wc, sc = weight_events(ctx, w_ctx, AXIS, TARGET, float(c), 1.0,
                                   width=q_ext, is_per=False, period=0.0,
                                   delete_input=False, specs=specs)
        dc = build_exp_tens(pc, wc, sigma=sigma, is_per=is_per, period=period,
                            specs=sc, verbose=False)
        offs = [None, None]
        offs[AXIS] = np.array([[c - mu_q]], dtype=float)
        pq, wq, sq = translate_attributes(qry, w_qry, offs, specs=specs)
        dq = build_exp_tens(pq, wq, sigma=sigma, is_per=is_per, period=period,
                            specs=sq, verbose=False)
        out[i] = float(cos_sim_exp_tens(dc, dq, normalize="oneSidedDenom",
                                        verbose=False))
    return out


@pytest.mark.parametrize("spectral", [False, True])
def test_similarity_nested_locked_matches_handbuilt(spectral):
    ctx, w_ctx, specs = _carrier(spectral)
    # query = the bound super-event at the clean statement (notes 8..11)
    qi = 8
    qry = [ctx[0][:, qi:qi + 1], ctx[1][:, qi:qi + 1]]
    w_qry = [w_ctx[0][:, qi:qi + 1] if w_ctx[0] is not None else None, None]
    width = 0.4                                       # narrow: ~one super-event per centre
    centres = ctx[1].ravel()[:ctx[0].shape[1]]        # one centre per super-event
    ref = _ref_locked(ctx, w_ctx, qry, w_qry, specs, centres, width)

    # Deliberately WRONG flat r/is_rel: in nested mode they must be ignored.
    got = windowed_similarity(
        ctx, w_ctx, qry, w_qry,
        [SIG_P, SIG_T], [1, 1], [False, False], [False, False], [0.0, 0.0],
        centres, context_window=("rect", width),
        normalize="oneSidedDenom", window_attr=AXIS, target_attr=TARGET,
        specs=specs, verbose=False)

    assert got.shape == (len(centres),)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)
    # sanity: the statement's own super-event is the cross-correlation peak
    assert np.argmax(got) == qi
    assert got[qi] > 0.99


def test_similarity_nested_is_transposition_invariant():
    """rel=1 in the specs makes the profile blind to global transposition of
    a context super-event."""
    ctx, w_ctx, specs = _carrier(spectral=False)
    qi = 8
    qry = [ctx[0][:, qi:qi + 1], ctx[1][:, qi:qi + 1]]
    width = 0.4
    centres = ctx[1].ravel()[:ctx[0].shape[1]]
    base = windowed_similarity(ctx, w_ctx, qry, None,
                               [SIG_P, SIG_T], [1, 1], [False, False],
                               [False, False], [0.0, 0.0], centres,
                               context_window=("rect", width),
                               window_attr=AXIS, target_attr=TARGET,
                               specs=specs, verbose=False)
    # transpose the query super-event up a tritone; rel=1 => identical profile
    qry_t = [qry[0] + 6.0, qry[1]]
    shifted = windowed_similarity(ctx, w_ctx, qry_t, None,
                                  [SIG_P, SIG_T], [1, 1], [False, False],
                                  [False, False], [0.0, 0.0], centres,
                                  context_window=("rect", width),
                                  window_attr=AXIS, target_attr=TARGET,
                                  specs=specs, verbose=False)
    assert np.allclose(base, shifted, rtol=1e-9, atol=1e-9)


def test_entropy_nested_matches_handbuilt():
    ctx, w_ctx, specs = _carrier(spectral=False)
    width = 2.0
    centres = np.linspace(ctx[1].min(), ctx[1].max(), 7)
    sigma, is_per, period = [SIG_P, SIG_T], [False, False], [0.0, 0.0]

    ref = np.empty(len(centres))
    for i, c in enumerate(centres):
        pw, ww, sw = weight_events(ctx, w_ctx, AXIS, TARGET, float(c), 1.0,
                                   width=width, is_per=False, period=0.0,
                                   delete_input=False, specs=specs)
        dens = build_exp_tens(pw, ww, sigma=sigma, is_per=is_per,
                              period=period, specs=sw, verbose=False)
        ref[i] = entropy_exp_tens(dens, method="renyi2", verbose=False)

    got = windowed_entropy(ctx, w_ctx, [SIG_P, SIG_T], [1, 1], [False, False],
                           [False, False], [0.0, 0.0], centres,
                           window=(1.0, width), method="renyi2",
                           window_attr=AXIS, target_attr=TARGET,
                           specs=specs, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_flat_path_unchanged_when_specs_none():
    """specs=None must reproduce the pre-existing flat result exactly."""
    rng = np.random.default_rng(0)
    N = 20
    pitch = np.sort(rng.integers(48, 84, size=N).astype(float)).reshape(1, N)
    onset = np.cumsum(rng.uniform(0.4, 0.6, size=N)).reshape(1, N)
    p_attr = [pitch, onset]
    query = [np.array([[60., 64., 67.]]), np.array([[0., 0.5, 1.0]])]
    centres = np.linspace(onset.min(), onset.max(), 9)
    kw = dict(window_attr=1, normalize="oneSidedDenom", verbose=False)
    a = windowed_similarity(p_attr, None, query, None, [0.12, 0.05], [1, 1],
                            [False, False], [False, False], [0.0, 0.0],
                            centres, **kw)
    b = windowed_similarity(p_attr, None, query, None, [0.12, 0.05], [1, 1],
                            [False, False], [False, False], [0.0, 0.0],
                            centres, specs=None, **kw)
    assert np.array_equal(a, b)

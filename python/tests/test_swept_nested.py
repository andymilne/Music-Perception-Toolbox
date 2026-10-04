"""Tests for nested-triple support in ``swept_similarity`` /
``swept_entropy``.

Each nested result is pinned to the explicit composition it stands in for
(``weight_events`` / ``translate_attributes`` -> ``build_maet`` ->
``sim_maet`` / ``entropy_maet`` with ``specs=``), exactly as
``test_swept_premaet.py`` pins the flat path. To prove the nested
geometry is read from ``specs`` and not from the positional ``r``/``rel``,
the calls pass deliberately wrong flat ``r``/``rel`` and still match.
"""
import numpy as np
import pytest

from mpt import (
    unpack_pre_maet,
    add_spectra, bind_events,
    swept_similarity, swept_entropy,
    weight_events, translate_attributes, build_maet,
    sim_maet, entropy_maet,
)

SIG_P, SIG_T = 0.15, 0.125
AXIS, TARGET = 1, 0          # window on time (1), target the bound pitch (0)


def _triple(spectral):
    """A short monophonic passage with a clean (+3, -3, +5) statement, bound
    into ordered relative four-note super-events (inner partial multiset when
    spectral), plus a flat onset-time attribute. Returns (ctx, w_ctx, specs)."""
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
    pb, wb, sb = unpack_pre_maet(bind_events([pitch_attr, on.reshape(1, N)], [w_attr, None],
                             [4, 1], step=1, rel_outer=True))
    return pb, wb, sb


def _ref_locked(ctx, w_ctx, qry, w_qry, specs, sweep_values, q_ext):
    """Inline hand-built locked-sweep reference (the composition the nested
    swept_similarity replaces)."""
    sigma, per, period = [SIG_P, SIG_T], [False, False], [0.0, 0.0]
    mu_q = float(np.nanmean(np.asarray(qry[AXIS], dtype=float)))
    out = np.empty(len(sweep_values))
    for i, c in enumerate(sweep_values):
        pc, wc, sc = unpack_pre_maet(weight_events(ctx, w_ctx, AXIS, TARGET, float(c), 1.0,
                                   width=q_ext, per=False, period=0.0,
                                   drop_input_attr=False, specs=specs))
        dc = build_maet(pc, wc, sigma=sigma, per=per, period=period,
                            specs=sc, verbose=False)
        offs = [None, None]
        offs[AXIS] = np.array([[c - mu_q]], dtype=float)
        pq, wq, sq = unpack_pre_maet(translate_attributes(qry, w_qry, offs, specs=specs))
        dq = build_maet(pq, wq, sigma=sigma, per=per, period=period,
                            specs=sq, verbose=False)
        out[i] = float(sim_maet(dc, dq, normalize="oneSidedDenom",
                                        verbose=False))
    return out


@pytest.mark.parametrize("spectral", [False, True])
def test_similarity_nested_locked_matches_handbuilt(spectral):
    ctx, w_ctx, specs = _triple(spectral)
    # query = the bound super-event at the clean statement (notes 8..11)
    qi = 8
    qry = [ctx[0][:, qi:qi + 1], ctx[1][:, qi:qi + 1]]
    w_qry = [w_ctx[0][:, qi:qi + 1] if w_ctx[0] is not None else None, None]
    width = 0.4                                       # narrow: ~one super-event per centre
    sweep_values = ctx[1].ravel()[:ctx[0].shape[1]]        # one centre per super-event
    ref = _ref_locked(ctx, w_ctx, qry, w_qry, specs, sweep_values, width)

    # Deliberately WRONG flat r/rel: in nested mode they must be ignored.
    got = swept_similarity(
        ctx, w_ctx, qry, w_qry,
        [SIG_P, SIG_T], [1, 1], [False, False], [False, False], [0.0, 0.0],
        sweep={AXIS: sweep_values}, align={AXIS: "both"},
        window={AXIS: {"shape": "rect", "width": width}},
        normalize="oneSidedDenom", target_attr=TARGET,
        specs=specs, verbose=False)

    assert got.shape == (len(sweep_values),)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)
    # sanity: the statement's own super-event is the cross-correlation peak
    assert np.argmax(got) == qi
    assert got[qi] > 0.99


def test_similarity_nested_is_transposition_invariant():
    """rel=1 in the specs makes the profile blind to global transposition of
    a context super-event."""
    ctx, w_ctx, specs = _triple(spectral=False)
    qi = 8
    qry = [ctx[0][:, qi:qi + 1], ctx[1][:, qi:qi + 1]]
    width = 0.4
    sweep_values = ctx[1].ravel()[:ctx[0].shape[1]]
    base = swept_similarity(ctx, w_ctx, qry, None,
                            [SIG_P, SIG_T], [1, 1], [False, False],
                            [False, False], [0.0, 0.0],
                            sweep={AXIS: sweep_values}, align={AXIS: "both"},
                            window={AXIS: {"shape": "rect", "width": width}},
                            target_attr=TARGET, specs=specs, verbose=False)
    # transpose the query super-event up a tritone; rel=1 => identical profile
    qry_t = [qry[0] + 6.0, qry[1]]
    shifted = swept_similarity(ctx, w_ctx, qry_t, None,
                               [SIG_P, SIG_T], [1, 1], [False, False],
                               [False, False], [0.0, 0.0],
                               sweep={AXIS: sweep_values}, align={AXIS: "both"},
                               window={AXIS: {"shape": "rect", "width": width}},
                               target_attr=TARGET, specs=specs,
                               verbose=False)
    assert np.allclose(base, shifted, rtol=1e-9, atol=1e-9)


def test_entropy_nested_matches_handbuilt():
    ctx, w_ctx, specs = _triple(spectral=False)
    width = 2.0
    sweep_values = np.linspace(ctx[1].min(), ctx[1].max(), 7)
    sigma, per, period = [SIG_P, SIG_T], [False, False], [0.0, 0.0]

    ref = np.empty(len(sweep_values))
    for i, c in enumerate(sweep_values):
        pw, ww, sw = unpack_pre_maet(weight_events(ctx, w_ctx, AXIS, TARGET, float(c), 1.0,
                                   width=width, per=False, period=0.0,
                                   drop_input_attr=False, specs=specs))
        dens = build_maet(pw, ww, sigma=sigma, per=per,
                              period=period, specs=sw, verbose=False)
        ref[i] = entropy_maet(dens, method="renyi2", verbose=False)

    got = swept_entropy(ctx, w_ctx, [SIG_P, SIG_T], [1, 1], [False, False],
                        [False, False], [0.0, 0.0], sweep={AXIS: sweep_values},
                        window={AXIS: {"shape": 1.0, "width": width}}, method="renyi2",
                        target_attr=TARGET, specs=specs, verbose=False)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_flat_path_unchanged_when_specs_none():
    """specs=None must reproduce the pre-existing flat result exactly."""
    rng = np.random.default_rng(0)
    N = 20
    pitch = np.sort(rng.integers(48, 84, size=N).astype(float)).reshape(1, N)
    onset = np.cumsum(rng.uniform(0.4, 0.6, size=N)).reshape(1, N)
    p_attr = [pitch, onset]
    query = [np.array([[60., 64., 67.]]), np.array([[0., 0.5, 1.0]])]
    sweep_values = np.linspace(onset.min(), onset.max(), 9)
    kw = dict(sweep={1: sweep_values}, align={1: "both"},
              window={1: {"shape": "rect", "width": 1.5}},
              normalize="oneSidedDenom", verbose=False)
    a = swept_similarity(p_attr, None, query, None, [0.12, 0.05], [1, 1],
                         [False, False], [False, False], [0.0, 0.0], **kw)
    b = swept_similarity(p_attr, None, query, None, [0.12, 0.05], [1, 1],
                         [False, False], [False, False], [0.0, 0.0],
                         specs=None, **kw)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("spectral", [False, True])
def test_similarity_empty_window_scores_zero(spectral):
    """A sweep value whose window catches no super-event scores exactly 0 --
    not NaN, and not an error. The nested (spectral) path must agree with the
    flat path here: an empty windowed triple otherwise reaches the nested
    contraction's value-range scan, which has no identity over an empty
    attribute column. This pins the empty-operand guard in the multi-attribute
    cosine entry."""
    iv = np.array([0., 3., 0., 5.])                     # the (+3, -3, +5) motif
    pit = np.concatenate([60. + iv, 60. + iv])          # two identical statements
    on = np.concatenate([np.arange(4.), 40. + np.arange(4.)])   # a wide rest between
    N = pit.size
    if spectral:
        Kp = 8
        pp, wp = add_spectra(pit, None, 'harmonic', Kp, 'powerlaw', 1.0, units=12.0)
        pitch_attr, w_attr = pp.reshape(N, Kp).T, wp.reshape(N, Kp).T
    else:
        pitch_attr, w_attr = pit.reshape(1, N), None
    ctx, w_ctx, specs = unpack_pre_maet(bind_events([pitch_attr, on.reshape(1, N)], [w_attr, None],
                                    [4, 1], step=1, rel_outer=True))
    qry = [ctx[0][:, 0:1], ctx[1][:, 0:1]]
    w_qry = [w_ctx[0][:, 0:1] if w_ctx[0] is not None else None, None]
    # Each super-event is timed at its span's last onset (end-aligned
    # binding), so the two statements sit at t = 3 and t = 43.
    sweep_values = np.array([3.0, 20.0, 43.0])               # 20.0 falls in the rest
    got = np.asarray(swept_similarity(
        ctx, w_ctx, qry, w_qry,
        [SIG_P, SIG_T], [1, 1], [True, False], [False, False], [0.0, 0.0],
        sweep={AXIS: sweep_values}, align={AXIS: "window"},
        window={AXIS: {"shape": "rect", "width": 0.6}}, drop=[AXIS],
        normalize="oneSidedDenom", specs=specs,
        verbose=False)).ravel()
    assert np.all(np.isfinite(got))
    assert got[1] == 0.0                                # empty window -> exactly zero
    assert got[0] > 0.99 and got[2] > 0.99              # the statements still match

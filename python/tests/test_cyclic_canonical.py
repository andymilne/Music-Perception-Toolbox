"""Regression tests for the cyclic canonical form and batched dedup.

``_cyclic_canonical`` narrows the candidate rotations before comparing
them exactly (it was a rotation-by-rotation search, whose O(n^2 log n)
cost dominated canonicalisation of spectrally-expanded sets). These
tests pin that the narrowed search returns exactly what the
rotation-by-rotation search returns, on sets chosen to stress the
narrowing: symmetric sets (many tied rotations), runs of equal values
with weights, values just below the period, and large sets evaluated
in several blocks.

They also pin that the batched-raw path of ``sim_maet`` deduplicates
pairs on its Phase 1 keys (from the raw rows) and does not
recanonicalise the spectrally-expanded densities, which made the
deduplication cost far exceed the similarities themselves.

Mirror of MATLAB tests/test_cyclic_canonical.m.
"""

import numpy as np
import pytest

import mpt
from mpt._tensor import canonical
from mpt._tensor.canonical import (
    _cyclic_canonical,
    _cyclic_canonical_loop,
)


def _prep(p, w, period):
    p = np.mod(np.asarray(p, dtype=float), period)
    si = np.argsort(p)
    return p[si], (None if w is None else np.asarray(w, dtype=float)[si])


def _assert_same(p, w, period):
    p, w = _prep(p, w, period)
    ref = _cyclic_canonical_loop(p, w, period)
    got = _cyclic_canonical(p, w, period)
    assert got[0] == ref[0]
    assert got[1] == ref[1]
    assert got[2] == ref[2]


PERIODS = [1200.0, 12.0, 1.0, 2 * np.pi, 7.3]


@pytest.mark.parametrize("period", PERIODS)
@pytest.mark.parametrize("edo", [12, 19, 31, 53, 100])
def test_symmetric_sets(period, edo):
    rng = np.random.default_rng(edo)
    base = np.arange(edo) * period / edo
    _assert_same(base, None, period)
    _assert_same(base, rng.integers(1, 3, edo), period)
    _assert_same(np.concatenate([base, base]), rng.random(2 * edo), period)
    sub = rng.choice(edo, size=max(7, edo // 2), replace=False)
    _assert_same(np.concatenate([base[sub], base[sub] + period / 2]),
                 None, period)


@pytest.mark.parametrize("period", PERIODS)
def test_random_sets_with_ties_and_near_period_values(period):
    rng = np.random.default_rng(11)
    for _ in range(150):
        n = int(rng.integers(2, 80))
        near = (period - rng.uniform(0, 1e-9, n) * rng.integers(0, 2, n)
                + rng.uniform(0, period, n) * rng.integers(0, 2, n))
        _assert_same(near, None if rng.random() < 0.5 else rng.random(n),
                     period)
        q = np.round(rng.uniform(0, period, n), 3)
        q = np.concatenate([q, q[: n // 3]])
        _assert_same(q, rng.integers(1, 4, len(q)), period)
        chain = np.mod(np.arange(n) * rng.uniform(0, period), period)
        _assert_same(chain, None, period)


def test_spectral_sets():
    rng = np.random.default_rng(5)
    for _ in range(40):
        x = np.sort(rng.integers(0, 24, 4)) * 100.0 + 3600.0
        p, w = mpt.add_spectra(x, None, "harmonic", 64, "powerlaw", 1)
        _assert_same(p, w, 1200.0)


def test_blocked_evaluation(monkeypatch):
    monkeypatch.setattr(canonical, "_CYCLIC_BLOCK_ELEMS", 50)
    rng = np.random.default_rng(3)
    for n in [40, 500]:
        base = np.arange(n) * 1200.0 / n
        _assert_same(base, None, 1200.0)
        _assert_same(base, rng.integers(1, 3, n), 1200.0)
        _assert_same(rng.uniform(0, 1200.0, n), rng.random(n), 1200.0)


def test_batched_spectrum_does_not_recanonicalise_spectra(monkeypatch):
    """Batched-raw dedup uses the raw rows' keys only."""
    rng = np.random.default_rng(2)
    X = (np.sort(rng.integers(0, 24, (60, 4)), axis=1) * 100.0 + 3600.0)
    Y = (np.sort(rng.integers(0, 24, (60, 4)), axis=1) * 100.0 + 3600.0)
    X = np.vstack([X, X + 1200.0])  # octave-equivalent duplicates
    Y = np.vstack([Y, Y + 1200.0])
    spec = ["harmonic", 16, "powerlaw", 1]

    sizes = []
    orig = canonical._cyclic_canonical

    def spy(p, w, period, **kwargs):
        sizes.append(len(p))
        return orig(p, w, period, **kwargs)

    monkeypatch.setattr(canonical, "_cyclic_canonical", spy)
    s = mpt.sim_maet(X, None, Y, None, 10, 1, False, True, 1200,
                     spectrum=spec, verbose=False)
    assert max(sizes) == 4

    ref = np.array([
        mpt.sim_maet(*mpt.add_spectra(X[i], None, *spec),
                     *mpt.add_spectra(Y[i], None, *spec),
                     10, 1, False, True, 1200, verbose=False)
        for i in range(len(X))
    ])
    np.testing.assert_allclose(s, ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s[:60], s[60:], rtol=0, atol=0)


# ---------------------------------------------------------------------
# Equal divisions of the octave: pitches that are not whole cents
# ---------------------------------------------------------------------


def _edo_cents(steps, n_edo, how):
    """The same EDO pitches by different arithmetic, so transposed copies
    carry different floating-point noise."""
    s = np.asarray(steps, dtype=float)
    step = 1200.0 / n_edo
    if how == 0:
        return s * 1200.0 / n_edo
    if how == 1:
        return s * step
    if how == 2:
        return s / n_edo * 1200.0
    return np.array([np.sign(k) * np.sum(np.full(abs(int(k)), step))
                     for k in s])


def _edo_truth(x, y, n_edo, rel, per):
    """The pair's equivalence class, computed exactly in EDO steps."""
    if per:
        x, y = np.mod(x, n_edo), np.mod(y, n_edo)
        if rel:
            def f(z):
                return min(tuple(sorted(np.mod(z - t, n_edo)))
                           for t in range(n_edo))
            return f(x), f(y)
        return min((tuple(sorted(np.mod(x - t, n_edo))),
                    tuple(sorted(np.mod(y - t, n_edo))))
                   for t in range(n_edo))
    if rel:
        return tuple(sorted(x - x.min())), tuple(sorted(y - y.min()))
    return tuple(sorted(x - x.min())), tuple(sorted(y - x.min()))


@pytest.mark.parametrize("n_edo", [12, 19, 22, 24, 31, 53])
@pytest.mark.parametrize("rel", [False, True])
@pytest.mark.parametrize("per", [False, True])
@pytest.mark.parametrize("precision", [None, 4])
@pytest.mark.parametrize("truncation_sigmas", [6, np.inf])
def test_edo_pairs_share_keys_exactly_when_equivalent(n_edo, rel, per,
                                                      precision,
                                                      truncation_sigmas):
    """Pairs that are not equivalent never share a key, and at the
    factory accuracy (truncation_sigmas = 6) transposed copies of an EDO
    pair always do. Includes symmetric A sets (several rotations give A's
    canonical form), for which B's shift is chosen jointly.

    At truncation_sigmas = inf the canonical forms are rounded finely
    enough to keep the 1e-12 accuracy floor (11 places at sigma = 10),
    which is the scale of the floating-point noise in pitches built by
    long arithmetic (here, repeated addition); copies may then keep
    separate keys unless ``precision`` is given, which is correct at that
    accuracy and costs only repeated computation."""
    from mpt._tensor.canonical import _pair_canonical_key

    rng = np.random.default_rng(n_edo)
    divisors = [d for d in range(2, n_edo) if n_edo % d == 0]
    keys = {}
    classes = set()
    for i in range(20):
        if i % 3 == 0 and divisors:
            d = int(rng.choice(divisors))
            g = n_edo // d
            cell = rng.choice(g, size=int(rng.integers(1, min(g, 3) + 1)),
                              replace=False)
            x = np.sort(np.concatenate([cell + g * j for j in range(d)]))
        else:
            x = np.sort(rng.choice(3 * n_edo, int(rng.integers(2, 6)),
                                   replace=False))
        y = np.sort(rng.choice(3 * n_edo, int(rng.integers(2, 6)),
                               replace=False))
        for _ in range(4):
            t = int(rng.integers(-2 * n_edo, 3 * n_edo))
            xs, ys = x + t, y + t
            if per:
                xs = xs + rng.integers(-1, 2, len(xs)) * n_edo
                ys = ys + rng.integers(-1, 2, len(ys)) * n_edo
            how = int(rng.integers(0, 4))
            ka, kb, *_ = _pair_canonical_key(
                _edo_cents(xs, n_edo, how), None,
                _edo_cents(ys, n_edo, how), None,
                sigma=10, r=2 if rel else 1, rel=rel, per=per,
                period=1200.0, precision=precision,
                truncation_sigmas=truncation_sigmas)
            c = _edo_truth(xs, ys, n_edo, rel, per)
            keys.setdefault((ka, kb), set()).add(c)
            classes.add(c)
    assert all(len(v) == 1 for v in keys.values())
    if truncation_sigmas == 6 or precision is not None:
        assert len(keys) == len(classes)


def test_batched_precision_does_not_split_edo_copies():
    """'precision' rounds canonical forms, not raw pitches, so 22-EDO
    copies still collapse to one computation per class."""
    rng = np.random.default_rng(22)
    X, Y, cls = [], [], []
    for c in range(10):
        x = np.sort(rng.choice(66, 4, replace=False))
        y = np.sort(rng.choice(66, 4, replace=False))
        for _ in range(5):
            t = int(rng.integers(-30, 60))
            how = int(rng.integers(0, 4))
            X.append(_edo_cents(x + t, 22, how))
            Y.append(_edo_cents(y + t, 22, how))
            cls.append(c)
    X, Y = np.array(X), np.array(Y)
    for prec in (None, 4):
        s = mpt.sim_maet(X, None, Y, None, 10, 1, False, True, 1200,
                         spectrum=["harmonic", 8, "powerlaw", 1],
                         precision=prec, truncation_sigmas=6,
                         verbose=False)
        cls_arr = np.array(cls)
        for c in range(10):
            v = s[cls_arr == c]
            assert np.ptp(v) == 0.0

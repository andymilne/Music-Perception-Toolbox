"""Raw entry forms must thread the truncation width.

Regression cover for the parameter-dropping class of bug where a raw
(loose-array) entry form of ``eval_exp_tens`` / ``cos_sim_exp_tens`` failed
to forward ``truncation_sigmas`` (and ``kernel_precision``) to the shared
core. Two consequences were possible:

* the single-multiset centres path read a dropped width as "no truncation"
  and summed every joint tuple densely (a large slowdown), and
* any raw form silently ignored a caller-supplied width, defaulting it.

Both are caught here without relying on wall-clock timing:

* **agreement** --- a raw form must match the prebuilt-density form exactly.
  Were the raw form running dense while the prebuilt form truncates at the
  default width, the two would differ by the (small, non-zero) truncation
  error; exact agreement therefore also witnesses that the raw form
  truncates at the same width (i.e. is not dense).
* **width sensitivity** --- a raw form evaluated at two different finite
  widths must give different results, proving the caller-supplied width is
  honoured rather than dropped to the default.
"""

import numpy as np
import pytest

import mpt
from mpt import build_exp_tens, eval_exp_tens, cos_sim_exp_tens

mpt.set_default(show_hints=False)


def _chord(seed, K, span):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, span, K))
    w = 0.3 + 0.7 * rng.uniform(size=K)
    return p, w


# (r, K, is_rel, is_per, span, period)
_CASES = [
    (3, 12, False, False, 3600.0, 0.0),
    (2, 16, True, False, 3600.0, 0.0),
    (3, 9, True, True, 1200.0, 1200.0),
]

# Regimes where a finite truncation width demonstrably changes the result.
# The relative-periodic case above σ/P = 0.03 uses the all-image
# (transposition-integral) form, whose value does not separate at the two
# probe widths, so it is covered by the agreement test only, not here.
_WIDTH_CASES = [
    (3, 12, False, False, 3600.0, 0.0),
    (2, 16, True, False, 3600.0, 0.0),
]


@pytest.mark.parametrize("r,K,is_rel,is_per,span,period", _CASES)
def test_eval_raw_scalar_matches_prebuilt(r, K, is_rel, is_per, span, period):
    p, w = _chord(0, K, span)
    sigma = 15.0
    dim = r - 1 if is_rel else r
    x = np.random.default_rng(7).uniform(0.0, span, (dim, 40))
    dens = build_exp_tens(
        [p.reshape(K, 1)], [w.reshape(K, 1)], [sigma], [r],
        [is_rel], [is_per], [period], verbose=False,
    )
    v_pre = eval_exp_tens(dens, x, verbose=False)
    v_raw = eval_exp_tens(
        p, w, sigma, r, is_rel, is_per, period, x, verbose=False,
    )
    # Exact agreement also witnesses the raw form is truncated (not dense).
    assert np.max(np.abs(v_raw - v_pre)) == 0.0


@pytest.mark.parametrize("r,K,is_rel,is_per,span,period", _WIDTH_CASES)
def test_eval_raw_scalar_honours_width(r, K, is_rel, is_per, span, period):
    p, w = _chord(1, K, span)
    sigma = 60.0   # wide kernel so the truncation width bites
    dim = r - 1 if is_rel else r
    x = np.random.default_rng(3).uniform(0.0, span, (dim, 40))

    def val(ts):
        return eval_exp_tens(
            p, w, sigma, r, is_rel, is_per, period, x,
            truncation_sigmas=ts, verbose=False,
        )

    assert np.max(np.abs(val(3.0) - val(9.0))) > 0.0


def test_eval_raw_batch_matches_prebuilt_and_honours_width():
    r, K, span, sigma = 3, 12, 3600.0, 60.0
    p1, w1 = _chord(10, K, span)
    p2, w2 = _chord(11, K, span)
    P = np.vstack([p1, p2])
    W = np.vstack([w1, w2])
    x = np.random.default_rng(5).uniform(0.0, span, (r, 30))

    # Agreement with per-row prebuilt densities.
    rows = []
    for p, w in ((p1, w1), (p2, w2)):
        d = build_exp_tens(
            [p.reshape(K, 1)], [w.reshape(K, 1)], [sigma], [r],
            [False], [False], [0.0], verbose=False,
        )
        rows.append(eval_exp_tens(d, x, verbose=False))
    v_pre = np.vstack(rows)
    v_raw = eval_exp_tens(P, W, sigma, r, False, False, 0.0, x, verbose=False)
    assert np.max(np.abs(v_raw - v_pre)) == 0.0

    # Width honoured on the batched path.
    d3 = eval_exp_tens(
        P, W, sigma, r, False, False, 0.0, x,
        truncation_sigmas=3.0, verbose=False,
    )
    d9 = eval_exp_tens(
        P, W, sigma, r, False, False, 0.0, x,
        truncation_sigmas=9.0, verbose=False,
    )
    assert np.max(np.abs(d3 - d9)) > 0.0


def test_eval_raw_ma_honours_method_and_width():
    r, K, span, sigma = 3, 12, 3600.0, 60.0
    p, w = _chord(2, K, span)
    x = np.random.default_rng(9).uniform(0.0, span, (r, 30))
    p_attr = [p.reshape(K, 1)]
    w_attr = [w.reshape(K, 1)]

    def val(ts):
        return eval_exp_tens(
            p_attr, w_attr, [sigma], [r], [False], [False], [0.0], x,
            truncation_sigmas=ts, verbose=False,
        )

    assert np.max(np.abs(val(3.0) - val(9.0))) > 0.0


def test_cos_sim_raw_forms_honour_width():
    r, K, span, sigma = 3, 10, 1200.0, 60.0
    p1, w1 = _chord(20, K, span)
    p2, w2 = _chord(21, K, span)

    def sm(ts):
        return cos_sim_exp_tens(
            p1, w1, p2, w2, sigma, r, True, False, 0.0,
            truncation_sigmas=ts, verbose=False,
        )

    def ma(ts):
        return cos_sim_exp_tens(
            [p1.reshape(K, 1)], [w1.reshape(K, 1)],
            [p2.reshape(K, 1)], [w2.reshape(K, 1)],
            [sigma], [r], [True], [False], [0.0],
            truncation_sigmas=ts, verbose=False,
        )

    assert abs(sm(3.0) - sm(9.0)) > 0.0
    assert abs(ma(3.0) - ma(9.0)) > 0.0

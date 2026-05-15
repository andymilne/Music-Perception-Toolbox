"""Cross-language golden-value tests for v2.2 paths.

Hardcodes outputs of representative v2.2 computations on deterministic
inputs (no RNG). The companion MATLAB file
``matlab/tests/test_cross_language_golden.m`` hardcodes the same
values; running both pins down cross-language numerical agreement to
1e-8 relative on the v2.2 surface (orbit cosine similarity SA + MA
including the safe/unsafe hybrid, Rényi-2 entropy SA + MA, orbit-path
:func:`tensor_harmonicity`, and orbit-path :func:`eval_exp_tens`).

Inputs use ``method='mobius'`` on the cosine cases so the Möbius method's
machinery is genuinely exercised rather than the dispatcher's
cost-model fallback to pairwise. Sigmas are chosen to keep values
well-conditioned (away from FP underflow); a 1e-8 relative tolerance
is the standard used elsewhere in the v22 suite.

To regenerate the golden values (e.g. after a deliberate algorithm
change), run this file as a script: ``python -m
tests.test_cross_language_golden`` prints the freshly computed
values for both languages to mirror.
"""

import numpy as np
import pytest

import mpt
from mpt.tensor import build_exp_tens, cos_sim_exp_tens, eval_exp_tens
from mpt.entropy import entropy_exp_tens
from mpt.harmony import tensor_harmonicity


# Tolerance for cross-language equality. 1e-8 is the standard used
# throughout the v22 suite (orbit-vs-pairwise on shared regimes).
RTOL = 1e-8
ATOL = 1e-12


# ----------------------------------------------------------------------
# Case A: SA cosSim, abs r=3, Möbius method
# ----------------------------------------------------------------------

def test_golden_sa_cossim_abs_r3():
    """Major triad vs minor triad, abs r=3, sigma=80 cents (orbit)."""
    p1 = np.array([0.0, 400.0, 700.0])
    p2 = np.array([0.0, 300.0, 700.0])
    w = np.array([1.0, 1.0, 1.0])
    s = cos_sim_exp_tens(
        p1, w, p2, w, 80.0, 3, False, False, 0.0,
        method='mobius', verbose=False,
    )
    GOLDEN = 0.67614851033133
    assert abs(s - GOLDEN) < RTOL * abs(GOLDEN) + ATOL


# ----------------------------------------------------------------------
# Case B: SA cosSim, rel r=3 periodic, Möbius method
# ----------------------------------------------------------------------

def test_golden_sa_cossim_rel_r3_per():
    """Major triad vs minor triad, rel r=3, period=1200, sigma=80."""
    p1 = np.array([0.0, 400.0, 700.0])
    p2 = np.array([0.0, 300.0, 700.0])
    w = np.array([1.0, 1.0, 1.0])
    s = cos_sim_exp_tens(
        p1, w, p2, w, 80.0, 3, True, True, 1200.0,
        method='mobius', verbose=False,
    )
    GOLDEN = 0.98878587374986
    assert abs(s - GOLDEN) < RTOL * abs(GOLDEN) + ATOL


# ----------------------------------------------------------------------
# Case C: MA cosSim with ragged-K hybrid (the killer cross-language case)
# ----------------------------------------------------------------------

def test_golden_ma_cossim_ragged_k_hybrid():
    """Mixed safe/unsafe ragged-K MA (single attribute, K=8, r=3,
    two unsafe events on each side via NaN padding to K_eff=4).

    Exercises the per-event safe/unsafe partition: safe pairs go
    through the vectorised orbit, unsafe pairs through direct
    enumeration. Identical hybrid logic in both languages.
    """
    P_x = np.array([
        [50.0,  100.0,  200.0,  300.0,  400.0,  500.0],
        [150.0, 250.0,  350.0,  450.0,  550.0,  650.0],
        [350.0, 450.0,  550.0,  650.0,  750.0,  850.0],
        [550.0, 650.0,  750.0,  850.0,  950.0, 1050.0],
        [np.nan, 850.0,  950.0,  np.nan, 1150.0, 1250.0],
        [np.nan, 1050.0, 1150.0, np.nan, 1350.0, 1450.0],
        [np.nan, 1250.0, 1350.0, np.nan, 1550.0, 1650.0],
        [np.nan, 1450.0, 1550.0, np.nan, 1750.0, 1850.0],
    ])  # (8, 6); columns 0 and 3 have K_eff=4 (unsafe at r=3)
    W_x = np.where(np.isnan(P_x), np.nan, 1.0)
    P_y = P_x + 50.0
    W_y = W_x.copy()

    dx = build_exp_tens([P_x], [W_x], [25.0], [3], None,
                        [False], [False], [0.0], verbose=False)
    dy = build_exp_tens([P_y], [W_y], [25.0], [3], None,
                        [False], [False], [0.0], verbose=False)
    s = cos_sim_exp_tens(dx, dy, method='mobius', verbose=False)
    GOLDEN = 0.12066345091832
    assert abs(s - GOLDEN) < RTOL * abs(GOLDEN) + ATOL


# ----------------------------------------------------------------------
# Case D: SA entropy Rényi-2, abs r=2
# ----------------------------------------------------------------------

def test_golden_sa_renyi2_abs_r2():
    """Rényi-2 entropy of a major-triad density, r=2 abs, sigma=20."""
    p = np.array([0.0, 400.0, 700.0])
    w = np.array([1.0, 1.0, 1.0])
    H = entropy_exp_tens(
        p, w, 20.0, 2, False, False, 0.0,
        method='renyi2', normalize=False, base=2,
    )
    GOLDEN = 14.88031481996820
    assert abs(H - GOLDEN) < RTOL * abs(GOLDEN) + ATOL


# ----------------------------------------------------------------------
# Case E: MA entropy Rényi-2 (Möbius method)
# ----------------------------------------------------------------------

def test_golden_ma_renyi2():
    """MA Rényi-2 entropy on a 2-attribute density (pitch + time),
    r=3 on pitch (K=5, safe), r=1 on time."""
    pitch = np.array([
        [0.0,    200.0,  400.0,  600.0],
        [400.0,  600.0,  700.0,  900.0],
        [700.0,  900.0,  1000.0, 1100.0],
        [1000.0, 1200.0, 1300.0, 1400.0],
        [1100.0, 1300.0, 1500.0, 1700.0],
    ])  # (5, 4)
    time = np.array([[0.0, 0.5, 1.0, 1.5]])  # (1, 4)
    H = entropy_exp_tens(
        [pitch, time], None,
        [12.0, 0.05], [3, 1], [0, 1],
        [False, False], [True, False], [1200.0, 0.0],
        method='renyi2', normalize=False, base=2,
    )
    GOLDEN = 21.64284222436801
    assert abs(H - GOLDEN) < RTOL * abs(GOLDEN) + ATOL


# ----------------------------------------------------------------------
# Case F: tensor_harmonicity (orbit-rel path; bypasses build_exp_tens)
# ----------------------------------------------------------------------

def test_golden_tensor_harmonicity_orbit():
    """Tensor harmonicity of a major triad with default duplicate=K=3."""
    chord = np.array([0.0, 400.0, 700.0])
    h = tensor_harmonicity(chord, None, 12.0, verbose=False)
    GOLDEN = 0.17358467740231
    assert abs(h - GOLDEN) < RTOL * abs(GOLDEN) + ATOL


# ----------------------------------------------------------------------
# Case G: eval_exp_tens at a query point, orbit-rel
# ----------------------------------------------------------------------

def test_golden_eval_exp_tens_rel():
    """Evaluate a 4-partial harmonic-template tensor at a single query
    (sigma=80 to keep value above FP underflow)."""
    tp = np.array([0.0, 1200.0, 1902.0, 2400.0])
    tw = np.array([1.0, 0.5, 0.333, 0.25])
    X = np.array([[400.0], [700.0]])  # 2 x 1 (r=3 rel -> dim=2)
    v = eval_exp_tens(tp, tw, 80.0, 3, True, False, 1200.0, X,
                      verbose=False)
    GOLDEN = 2.07507623760499e-06
    assert abs(v[0] - GOLDEN) < RTOL * abs(GOLDEN) + ATOL


if __name__ == '__main__':
    import sys
    print("=" * 60)
    print("Generating golden values (Python side)")
    print("=" * 60)
    for name in [n for n in sorted(dir()) if n.startswith('test_golden_')]:
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}", file=sys.stderr)

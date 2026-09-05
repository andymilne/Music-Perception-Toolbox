"""Standard test regimes for MPT numerical robustness.

This module codifies the parameter sets across which every numerical
quantity in MPT (cosine similarity, Rényi-2 entropy, Shannon entropy,
point-wise density evaluation, windowed similarity, harmonicity,
virtual pitches, ...) should be exercised before release. The intent
is twofold:

* Every new quantity gets the same battery of tests, so we don't ship
  fixes for one quantity that accidentally regress another.
* When a new quantity exposes a bug in a previously-uncovered regime,
  we add the regime here once and every other quantity inherits the
  coverage.

The regimes consolidate (a) the earlier audit scripts in
``tests/precision_audit/`` and (b) the seven-regime self-IP
investigation that motivated the v3 MA-diagnostic simplification.

Usage
-----
A "quantity-of-interest" is a callable ``compute(params) -> value``
together with an optional ``reference(params) -> value``. The runner
``run_regime_assessment`` iterates the regimes, calls ``compute`` and
``reference`` on each cell, and reports per-cell flags and per-regime
summaries.

Schema
------
Each regime is a dict with at least:

* ``name``: short identifier (e.g. ``'R2_high_r_single_multiset'``).
* ``description``: human-readable rationale.
* ``cells``: a list of dicts, each describing one parameter point. The
  cell schema depends on the regime, but always carries a ``seed``
  (or ``seeds`` list) plus enough information to construct the
  density via ``build_exp_tens``.

Reading a regime list does not by itself construct any densities;
the caller materialises them from the cell parameters.
"""
from __future__ import annotations

import numpy as np


# Standard period in cents (one octave, the canonical pitch period).
P_DEFAULT = 1200.0

# Standard seed list. Use 5 for routine sweeps; expand to 20+ for
# release-blocking confirmation runs.
SEEDS_QUICK = [0, 1, 2, 3, 4]
SEEDS_THOROUGH = list(range(20))


# ===================================================================
#  Single-attribute (single-multiset) regimes — pitch on the octave
# ===================================================================

def regime_sm_sigma_to_zero(seeds=SEEDS_QUICK, P=P_DEFAULT):
    """single-multiset: σ → 0 catastrophic regime.

    At very small σ/P, the orbit-Möbius alternating sum can lose all
    significant digits. Earlier audits established that:

    * abs_per: the diagnostic remains reliable.
    * rel_per: the diagnostic over-fires (fixed in v3).
    * abs_nonper: depends on K-r margin (see R3_abs_nonper).

    Parameters explored: r ∈ {2, 3, 4}, K ∈ {r+1, r+2, r+5},
    σ ∈ {33, 10, 3, 1, 0.1, 0.01, 1e-3, 1e-5} cents.
    """
    cells = []
    for r in (2, 3, 4):
        for K in (r + 1, r + 2, r + 5):
            for sigma in (33.0, 10.0, 3.0, 1.0, 0.1, 0.01, 1e-3, 1e-5):
                for seed in seeds:
                    cells.append(dict(
                        kind='single-multiset', r=r, K=K, sigma=sigma,
                        is_rel=False, is_per=True, period=P,
                        seed=seed,
                    ))
    return dict(
        name='R0_sigma_to_zero',
        description='single-multiset σ → 0 catastrophic cancellation regime',
        cells=cells,
    )


def regime_sm_high_r(seeds=SEEDS_QUICK, P=P_DEFAULT):
    """single-multiset: r ∈ {5, 6} with K barely above r.

    Stress-tests the orbit table at the largest available r and
    smallest K-r margin. Bell numbers grow rapidly here (B_5 = 52,
    B_6 = 203) so the alternating sum has many more terms.
    """
    cells = []
    for r in (5, 6):
        for K in (r + 1, r + 2, r + 4):
            for sigma in (33.0, 10.0, 3.0, 1.0, 0.1):
                for seed in seeds:
                    cells.append(dict(
                        kind='single-multiset', r=r, K=K, sigma=sigma,
                        is_rel=False, is_per=True, period=P,
                        seed=seed,
                    ))
    return dict(
        name='R2_high_r_single_multiset',
        description='single-multiset r ∈ {5, 6} with K-r margin ≤ 4',
        cells=cells,
    )


def regime_sm_abs_nonper(seeds=SEEDS_QUICK):
    """single-multiset: absolute non-periodic mode at sharp σ.

    Geometry differs from periodic mode: no period parameter, density
    falls off at infinity. Pitches drawn from a typical-musical range
    (-300 to +300 cents).
    """
    cells = []
    for r in (2, 3, 4, 5):
        for K in (r + 1, r + 3, r + 6):
            for sigma in (33.0, 10.0, 3.0, 1.0):
                for seed in seeds:
                    cells.append(dict(
                        kind='single-multiset', r=r, K=K, sigma=sigma,
                        is_rel=False, is_per=False, period=0.0,
                        pitch_range=(-300.0, 300.0),
                        seed=seed,
                    ))
    return dict(
        name='R3_abs_nonper_single_multiset',
        description='single-multiset absolute non-periodic mode at sharp σ',
        cells=cells,
    )


def regime_sm_rel_per(seeds=SEEDS_QUICK, P=P_DEFAULT):
    """single-multiset: relative-periodic mode.

    Relative-mode densities depend only on within-tuple differences.
    Geometry is on a (r-1)-dimensional torus quotient. Per-u-point
    cancellation ratio is over-conservative here (false alarms);
    earlier audit established this and the fix shipped in v3.
    """
    cells = []
    for r in (2, 3, 4):
        for K in (r + 1, r + 2, r + 5):
            for sigma in (33.0, 10.0, 3.0, 1.0, 0.1):
                for seed in seeds:
                    cells.append(dict(
                        kind='single-multiset', r=r, K=K, sigma=sigma,
                        is_rel=True, is_per=True, period=P,
                        seed=seed,
                    ))
    return dict(
        name='R0b_rel_per_single_multiset',
        description='single-multiset relative-periodic mode at sharp σ',
        cells=cells,
    )


def regime_sm_high_K(seeds=(0,), P=P_DEFAULT):
    """single-multiset: high K with low-to-moderate r.

    Spectral analyses use K of 24 to 100 partials with r=2 or 3.
    Different cancellation profile from low-K cases. K=64 is the
    upper end of typical spectral enrichment in perceptual studies.
    """
    cells = []
    for r, K_list in ((2, (50, 100)), (3, (24, 50, 64))):
        for K in K_list:
            for sigma in (33.0, 10.0, 3.0, 1.0):
                for seed in seeds:
                    cells.append(dict(
                        kind='single-multiset', r=r, K=K, sigma=sigma,
                        is_rel=False, is_per=True, period=P,
                        seed=seed,
                    ))
    return dict(
        name='R7_high_K_single_multiset',
        description='single-multiset high K (up to 100) with r ∈ {2, 3}',
        cells=cells,
    )


# ===================================================================
#  Adversarial pitch arrangements (single-multiset only — generalise as needed)
# ===================================================================

def regime_sm_adversarial(P=P_DEFAULT):
    """Structured pitch configurations that random uniform sampling
    misses: equal-tempered scale degrees, near-degenerate pairs,
    tight clusters, just-intonation triads.
    """
    configs = [
        ('12-TET pentachord',
         np.array([0.0, 200.0, 400.0, 700.0, 900.0]),
         np.ones(5)),
        ('near-degenerate 1ct',
         np.array([0.0, 1.0, 400.0, 700.0]),
         np.ones(4)),
        ('near-degenerate 0.1ct',
         np.array([0.0, 0.1, 400.0, 700.0]),
         np.ones(4)),
        ('tight cluster',
         np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
         np.ones(5)),
        ('just-intonation triad',
         1200.0 * np.log2(np.array([1.0, 5/4, 3/2])),
         np.ones(3)),
    ]
    cells = []
    for desc, p, w in configs:
        for r in (2, 3):
            if r > len(p):
                continue
            for sigma in (33.0, 10.0, 3.0, 1.0):
                cells.append(dict(
                    kind='single-multiset', r=r, K=len(p), sigma=sigma,
                    is_rel=False, is_per=True, period=P,
                    pitches=p.copy(), weights=w.copy(),
                    description=desc, seed=None,
                ))
    return dict(
        name='R4_adversarial_single_multiset',
        description='Structured single-multiset pitch configurations',
        cells=cells,
    )


# ===================================================================
#  Spectral weight distributions
# ===================================================================

def regime_sm_spectral_weights(seeds=SEEDS_QUICK, P=P_DEFAULT):
    """Realistic spectral inputs: harmonic series with rolloff α.

    Real timbres have weights spanning orders of magnitude, not the
    near-uniform distribution of the random sweep. Alpha ranges:

    * α=1: standard 1/k (typical violin, voice)
    * α=2: steep rolloff (clarinet-like)
    * α=0.5: shallow rolloff (sawtooth-like)

    Partial counts cover the realistic range. Spectral enrichment in
    perceptual studies can use up to 64 partials per tone (and rarely
    more), so the upper end of K is set there. Below K=8 the spectrum
    is too sparse to stress the orbit machinery.
    """
    cells = []
    for desc, K, alpha in (
        ('harmonic α=1, K=8', 8, 1.0),
        ('harmonic α=1, K=24', 24, 1.0),
        ('harmonic α=1, K=48', 48, 1.0),
        ('harmonic α=1, K=64', 64, 1.0),
        ('steep α=2, K=12', 12, 2.0),
        ('steep α=2, K=24', 24, 2.0),
        ('steep α=2, K=64', 64, 2.0),
        ('shallow α=0.5, K=24', 24, 0.5),
        ('shallow α=0.5, K=64', 64, 0.5),
    ):
        partials = np.arange(1, K + 1, dtype=float)
        p_base = 1200.0 * np.log2(partials)
        w = 1.0 / partials ** alpha
        for r in (2, 3):
            for sigma in (33.0, 10.0, 3.0, 1.0):
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    fund = rng.uniform(0, P)
                    cells.append(dict(
                        kind='single-multiset', r=r, K=K, sigma=sigma,
                        is_rel=False, is_per=True, period=P,
                        pitches=((p_base + fund) % P).copy(),
                        weights=w.copy(),
                        description=desc, seed=seed,
                    ))
    return dict(
        name='R5_spectral_weights_single_multiset',
        description='Harmonic spectra K up to 64 with realistic rolloff',
        cells=cells,
    )


# ===================================================================
#  Multi-attribute (MA) regimes
# ===================================================================

def regime_ma_self_ip(seeds=SEEDS_QUICK, P=P_DEFAULT):
    """MA: per-attribute self-IP across realistic configurations.

    The v3 sweep established that the per-(n,m) cancellation
    ratio diagnostic over-fires here in 100% of typical musical
    regimes — values are correct to FP precision. This regime is
    retained as a regression check against any future per-cell
    diagnostic resurfacing.
    """
    cells = []
    layouts = [
        ('2-attr r=[2,1] K=[3,1]', 2, [2, 1], [3, 1]),
        ('2-attr r=[3,1] K=[5,1]', 2, [3, 1], [5, 1]),
        ('2-attr r=[2,2] K=[3,2]', 2, [2, 2], [3, 2]),
        ('3-attr r=[2,1,1] K=[3,1,1]', 3, [2, 1, 1], [3, 1, 1]),
    ]
    for desc, A, r_vec, K_vec in layouts:
        for sigma_pitch in (33.0, 10.0, 3.0, 1.0):
            for seed in seeds:
                cells.append(dict(
                    kind='MA', A=A, r=list(r_vec), K=list(K_vec), N=5,
                    sigma=[sigma_pitch] + [0.05] * (A - 1),
                    is_rel=[False] * A,
                    is_per=[True] * A,
                    period=[P] + [1.0] * (A - 1),
                    description=desc, seed=seed,
                ))
    return dict(
        name='R1_ma_self_ip',
        description='MA per-attribute self-IP regression check',
        cells=cells,
    )


# ===================================================================
#  Standard bundle
# ===================================================================

def standard_regimes(thorough=False, P=P_DEFAULT):
    """Return the full standard test regime as a list of regime dicts.

    Parameters
    ----------
    thorough : bool, default False
        If True, use 20 seeds per cell instead of 5. Use for
        release-blocking confirmation runs.
    P : float, default 1200.0
        Pitch period in cents.
    """
    seeds = SEEDS_THOROUGH if thorough else SEEDS_QUICK
    return [
        regime_sm_sigma_to_zero(seeds=seeds, P=P),
        regime_sm_rel_per(seeds=seeds, P=P),
        regime_ma_self_ip(seeds=seeds, P=P),
        regime_sm_high_r(seeds=seeds, P=P),
        regime_sm_abs_nonper(seeds=seeds),
        regime_sm_adversarial(P=P),
        regime_sm_spectral_weights(seeds=seeds, P=P),
        regime_sm_high_K(seeds=(0,), P=P),
    ]


# ===================================================================
#  Cell materialisation helpers
# ===================================================================

def materialise_cell(cell, default_pitch_range=None):
    """Turn a cell parameter dict into the positional argument tuple
    expected by build_exp_tens (single-multiset or MA, depending on cell['kind']).

    single-multiset call: ``build_exp_tens(p, w, sigma, r, is_rel, is_per, period)``
    MA call: ``build_exp_tens(p_attr, w, sigma, r, is_rel, is_per, period)``
    """
    if cell['kind'] == 'single-multiset':
        K = cell['K']
        if 'pitches' in cell:
            p = cell['pitches']
            w = cell['weights']
        else:
            seed = cell['seed']
            rng = np.random.default_rng(seed)
            if cell['is_per']:
                p = rng.uniform(0.0, cell['period'], K)
            else:
                lo, hi = cell.get('pitch_range', default_pitch_range or (-300, 300))
                p = rng.uniform(lo, hi, K)
            w = rng.uniform(0.5, 1.5, K)
        return (p, w, cell['sigma'], cell['r'],
                cell['is_rel'], cell['is_per'], cell['period'])
    elif cell['kind'] == 'MA':
        A = cell['A']
        K_vec = cell['K']
        N = cell['N']
        period_vec = cell['period']
        seed = cell['seed']
        rng = np.random.default_rng(seed)
        p_attr = []
        w = []
        for a in range(A):
            p_attr.append(rng.uniform(0.0, period_vec[a], (K_vec[a], N)))
            w.append(rng.uniform(0.5, 1.5, (K_vec[a], N)))
        return (p_attr, w, cell['sigma'], cell['r'],
                cell['is_rel'], cell['is_per'], cell['period'])
    else:
        raise ValueError(f"Unknown cell kind: {cell['kind']!r}")


# ===================================================================
#  Generic assessor
# ===================================================================

def run_regime_assessment(
    quantity_name,
    compute_fn,
    reference_fn=None,
    regimes=None,
    rel_tol=1e-10,
    abs_tol=1e-12,
    print_per_cell=False,
):
    """Apply ``compute_fn`` (and optional ``reference_fn``) to every
    cell of every regime, collect per-regime statistics.

    ``compute_fn(cell) -> float`` is the quantity under test.
    ``reference_fn(cell) -> float`` is an independent reference. If
    omitted, only finiteness and sign sanity are checked.

    A cell passes when ``|value - ref| <= max(abs_tol, rel_tol * |ref|)``
    (np.isclose semantics). The absolute floor is essential when the
    quantity itself is near zero — e.g., cosine similarity between
    near-orthogonal densities can be 1e-10 or smaller, where pure
    relative error gives spurious failures from FP noise.

    Cells where both ``compute_fn`` and ``reference_fn`` return NaN
    are counted as **skipped** rather than failed.

    Returns a list of per-regime summary dicts.
    """
    if regimes is None:
        regimes = standard_regimes()

    summaries = []
    for regime in regimes:
        n_cells = len(regime['cells'])
        n_failed = 0
        n_skipped = 0
        n_finite = 0
        max_rel_err = 0.0
        max_abs_err = 0.0
        first_failure = None
        for cell in regime['cells']:
            try:
                value = compute_fn(cell)
            except Exception as e:
                n_failed += 1
                if first_failure is None:
                    first_failure = (cell, f'compute raised: {e}')
                continue
            value_nan = not np.isfinite(value)
            if reference_fn is not None:
                try:
                    ref = reference_fn(cell)
                except Exception as e:
                    n_failed += 1
                    if first_failure is None:
                        first_failure = (cell, f'reference raised: {e}')
                    continue
                ref_nan = not np.isfinite(ref)
                if value_nan and ref_nan:
                    n_skipped += 1
                    continue
                if value_nan or ref_nan:
                    n_failed += 1
                    if first_failure is None:
                        first_failure = (
                            cell,
                            f'one-sided NaN: value={value}, ref={ref}'
                        )
                    continue
                n_finite += 1
                abs_err = abs(value - ref)
                rel_err = abs_err / max(abs(ref), 1e-300)
                max_abs_err = max(max_abs_err, abs_err)
                max_rel_err = max(max_rel_err, rel_err)
                tolerance = max(abs_tol, rel_tol * abs(ref))
                if abs_err > tolerance:
                    n_failed += 1
                    if first_failure is None:
                        first_failure = (
                            cell,
                            f'abs_err {abs_err:.2e} > tol '
                            f'{tolerance:.2e} (rel {rel_err:.2e})'
                        )
            else:
                if value_nan:
                    n_failed += 1
                    if first_failure is None:
                        first_failure = (cell, f'non-finite value: {value}')
                    continue
                n_finite += 1
            if print_per_cell:
                print(f'  {regime["name"]} cell={cell}: value={value}')
        summary = dict(
            quantity=quantity_name,
            regime=regime['name'],
            n_cells=n_cells,
            n_failed=n_failed,
            n_skipped=n_skipped,
            n_finite=n_finite,
            max_rel_err=max_rel_err,
            max_abs_err=max_abs_err,
            first_failure=first_failure,
        )
        summaries.append(summary)
    return summaries


def print_summary(summaries):
    """Tabular summary of an assessment."""
    print(f'{"quantity":>20} {"regime":>27} {"cells":>6} {"failed":>7} '
          f'{"skipped":>8} {"max_abs_err":>12} {"max_rel_err":>12}')
    for s in summaries:
        print(f'{s["quantity"]:>20} {s["regime"]:>27} {s["n_cells"]:>6d} '
              f'{s["n_failed"]:>7d} {s.get("n_skipped", 0):>8d} '
              f'{s.get("max_abs_err", 0.0):>12.2e} '
              f'{s["max_rel_err"]:>12.2e}')
        if s['first_failure'] is not None:
            cell, why = s['first_failure']
            print(f'{"":>20} {"":>27}   first failure: {why}')

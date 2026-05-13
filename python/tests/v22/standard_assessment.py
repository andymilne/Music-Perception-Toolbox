"""Apply the standard test regime to each v2.2 quantity of interest.

This is the canonical "release-blocking" assessment: every shipping
numerical quantity must pass every regime in
``tests/v22/standard_regimes.standard_regimes()``.

When a new quantity is added (e.g., windowed similarity, Möbius-based
``eval_exp_tens``, harmonicity orbit path), add it here as another
quantity entry.
"""
import math
import sys
import numpy as np

# Allow running directly without installation.
sys.path.insert(0, '.')

from mpt.tensor import build_exp_tens, cos_sim_exp_tens, eval_exp_tens
from mpt.entropy import entropy_exp_tens
from tests.v22.standard_regimes import (
    standard_regimes,
    materialise_cell,
    run_regime_assessment,
    print_summary,
)


# Maximum n_j for which pairwise <T_x, T_y> is run inline. Pairwise
# scales as O(n_j^2) memory; at n_j=5000 the kernel matrix alone is
# ~200 MB. Cells exceeding this threshold are skipped (NaN reference).
_PAIRWISE_N_J_MAX = 5000


def _n_j_sa(r, K):
    """Number of permutation tuples K! / (K-r)! for SA mode."""
    return math.perm(K, r)


def _cell_pairwise_feasible(cell):
    """True if pairwise on this cell stays under the n_j threshold."""
    if cell['kind'] == 'SA':
        return _n_j_sa(cell['r'], cell['K']) <= _PAIRWISE_N_J_MAX
    elif cell['kind'] == 'MA':
        # Per-attribute n_j_a = K_a! / (K_a - r_a)! and the MA pairwise
        # kernel scales with the product across attributes.
        prod = 1
        for r_a, K_a in zip(cell['r'], cell['K']):
            prod *= _n_j_sa(r_a, K_a)
            if prod > _PAIRWISE_N_J_MAX:
                return False
        return True
    return False


def _build_dens(cell):
    args = materialise_cell(cell)
    return build_exp_tens(*args, verbose=False)


# -----------------------------------------------------------------
# Quantity: cosine self-similarity (must equal 1).
# -----------------------------------------------------------------

def cos_self(cell):
    d = _build_dens(cell)
    return cos_sim_exp_tens(d, d, verbose=False)

def cos_self_ref(cell):
    return 1.0


# -----------------------------------------------------------------
# Quantity: cosine cross-similarity orbit-vs-pairwise consistency.
# Compute via auto routing and via forced pairwise; demand FP-precision
# agreement. Cells where pairwise would OOM are skipped (NaN reference).
# -----------------------------------------------------------------

def _build_y_dens(cell):
    """Build a 'y' density that differs from the cell's primary
    density. For seedable cells, offset the seed; for adversarial
    cells (no seed), perturb pitches by a small offset.
    """
    if cell.get('seed') is not None:
        return _build_dens({**cell, 'seed': cell['seed'] + 1000})
    # Adversarial cell with fixed pitches: perturb to make a distinct
    # density. cos_sim between identical densities is 1 trivially and
    # would not exercise the cross-IP code path.
    perturbed = {**cell, 'pitches': cell['pitches'] + 7.3}
    return _build_dens(perturbed)


def cos_cross_auto(cell):
    if not _cell_pairwise_feasible(cell):
        return float('nan')
    d_x = _build_dens(cell)
    d_y = _build_y_dens(cell)
    return cos_sim_exp_tens(d_x, d_y, method='auto', verbose=False)


def cos_cross_pairwise(cell):
    if not _cell_pairwise_feasible(cell):
        return float('nan')
    d_x = _build_dens(cell)
    d_y = _build_y_dens(cell)
    return cos_sim_exp_tens(d_x, d_y, method='bulger', verbose=False)


# -----------------------------------------------------------------
# Quantity: Rényi-2 entropy (finite + positive).
# -----------------------------------------------------------------

def renyi2(cell):
    d = _build_dens(cell)
    return entropy_exp_tens(d, method='renyi2', normalize=False)


# -----------------------------------------------------------------
# Quantity: eval_exp_tens orbit-vs-centres consistency.
# Compute density at a small batch of query points via auto routing
# and via forced centres; demand FP-precision agreement. SA only —
# the v2.2 orbit eval covers SA; an MA orbit eval is on the roadmap.
# -----------------------------------------------------------------

_EVAL_N_Q = 16  # query batch size; small to keep centres path tractable
                # at high r/K cells of the regime.


def _eval_query_pts(cell):
    """Generate r-1 (rel) or r (abs) dimensional query columns for a
    cell. Periodic queries live in [0, P); non-periodic queries live
    in a typical-musical band around the source range."""
    if cell['kind'] != 'SA':
        return None
    seed = cell.get('seed') if cell.get('seed') is not None else 12345
    rng = np.random.default_rng(seed + 9000)
    r = cell['r']
    is_rel = cell['is_rel']
    is_per = cell['is_per']
    period = cell['period']
    dim = r - 1 if is_rel else r
    if is_per:
        return rng.uniform(0.0, period, (dim, _EVAL_N_Q))
    return rng.uniform(-300.0, 300.0, (dim, _EVAL_N_Q))


def eval_auto(cell):
    if cell['kind'] != 'SA':
        return float('nan')
    d = _build_dens(cell)
    x = _eval_query_pts(cell)
    vals = eval_exp_tens(d, x, method='auto', verbose=False)
    # Reduce to a scalar so the assessor's rel/abs comparison works.
    # Mean is invariant to query-point ordering and exposes any
    # discrepancy uniformly.
    return float(np.mean(vals))


def eval_centres(cell):
    if cell['kind'] != 'SA':
        return float('nan')
    d = _build_dens(cell)
    x = _eval_query_pts(cell)
    vals = eval_exp_tens(d, x, method='centres', verbose=False)
    return float(np.mean(vals))


# -----------------------------------------------------------------
# Run.
# -----------------------------------------------------------------

if __name__ == '__main__':
    regimes = standard_regimes(thorough=False)

    print('\n=== Quantity: cosine self-similarity (must equal 1) ===')
    s1 = run_regime_assessment(
        'cos_sim self', cos_self, cos_self_ref,
        regimes=regimes, rel_tol=1e-10,
    )
    print_summary(s1)

    print('\n=== Quantity: cosine cross-similarity (auto vs pairwise) ===')
    s2 = run_regime_assessment(
        'cos_sim auto vs pw', cos_cross_auto, cos_cross_pairwise,
        regimes=regimes, rel_tol=1e-10,
    )
    print_summary(s2)

    print('\n=== Quantity: Rényi-2 entropy (finite + positive) ===')
    s3 = run_regime_assessment(
        'renyi2 entropy', renyi2, None,
        regimes=regimes,
    )
    print_summary(s3)

    print('\n=== Quantity: eval_exp_tens (auto vs centres, SA only) ===')
    s4 = run_regime_assessment(
        'eval auto vs centres', eval_auto, eval_centres,
        regimes=regimes, rel_tol=1e-10,
    )
    print_summary(s4)

    n_failed_total = sum(s['n_failed'] for s in (s1 + s2 + s3 + s4))
    print(f'\nTotal failed cells across all assessments: {n_failed_total}')
    sys.exit(1 if n_failed_total > 0 else 0)

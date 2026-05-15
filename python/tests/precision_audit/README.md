# v2.2 precision audit

Standalone investigation scripts and reference outputs for the v2.2 Möbius
release's precision behaviour. These are not pytest tests — they are
exploratory sweeps producing tabular output for inspection. Each script has a
companion `*.out.txt` capturing its reference output at the time of last run.

## Scripts

### `01_precision_sweep_4modes.py`
Comprehensive orbit-vs-pairwise comparison across all four modes
(abs/rel × per/nonper), `r ∈ {2, 3, 4, 5}`, `K ∈ {r, r+1, r+2, r+4}`, at A=1
N=2 with 5 random seeds. σ=50, P=1200 (so σ/P ≈ 0.042 in the periodic cases).

Each row reports median, 90th-percentile, and maximum relative discrepancy
between orbit and pairwise cosine.

Markers:
- `!!`  catastrophic blow-up (cancellation when both inner products are
  near zero — Issue 2).
- `BAD` discrepancy above 1e-12 but not catastrophic.
- blank cells are at floating-point precision.

### `02_orbit_inner_rel_window_size.py`
Sweeps the integration-window half-width parameter in `_orbit_inner_rel`
from 4σ to 16σ at `r=2`, K=4 N=4 A=2 rel_nonper, where the default 4σ window
shows ~5e-7 truncation error. Demonstrates that 8σ recovers FP precision
(median 9e-16) and further widening gives no improvement. Also shows that
sps=5 vs sps=10 trapezoidal density doesn't matter once the window is wide
enough — the issue is truncation tail, not under-sampling.

### `03_window_fix_verify_sa.py`
Same window-widening study at the SA path (`_cos_sim_exp_tens_sa`),
sweeping (r, K, window_sigma) at K ∈ {r, r+2}. Confirms the truncation issue
is shared between SA and MA paths. Also documents the K=r cancellation
regime separately (those rows stay bad regardless of window width — that is
Issue 2, not Issue 1).

### `04_rel_per_sigma_scaling.py`
SA rel_per discrepancy as σ/P decreases from ~0.04 to ~0.001.
Demonstrates that orbit and pairwise differ at σ/P ≳ 0.03 and converge to
floating point at smaller σ/P, confirming the regime boundary in
`v22_specification.md` line 17. Note: this is **not** a bug — the two
formulas are different mathematical objects (per-pair-wrap closed form vs
integration over circle translates) that coincide at small σ/P. See
`V22_DEV_LOG.md` for the full discussion.

### `05_orbit_self_consistency_high_sp.py`
Probes whether orbit's trapezoidal quadrature is itself converged at the
default sps=10 across σ/P ∈ {0.04, 0.08, 0.17, 0.33, 0.67, 1.0}, by
comparing against sps=50. Confirms trapezoid is converged to FP at the
endpoints and to ~1e-6 at intermediate σ/P. This is the raw integral form,
unrelated to the pairwise closed form.

## How to run

From the repo's `python/` directory:

```
python tests/precision_audit/01_precision_sweep_4modes.py
```

Each script prints results directly to stdout. Total runtime for all five
is well under a minute.

## Limitations of the current audit

- All sweeps fix σ=50, P=1200 except `04_*`. σ/P only crossed in script 04
  (and only at SA path). σ alone (independent of P) and (σ, P) interaction
  effects on the orbit precision are **not** swept.
- A is restricted to A∈{1, 2}, with most sweeps at A=1.
- N is restricted to N∈{2, 4}, with most sweeps at N=2.
- Random seeding is 5–10 seeds, not enough to guarantee tail behaviour.
- The K-vs-r boundary distinguishing "safe orbit" from "cancellation-risk
  orbit" is characterised by pattern but not pinned to a precise rule.

These gaps are tracked in `V22_DEV_LOG.md` at the repository root.

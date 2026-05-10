# v2.2 development work log

Internal tracking document for the v2.2 (Möbius release) work in progress.
Not user-facing; delete or fold into CHANGELOG when v2.2 ships. Updated at
each session boundary.

The spec is `v22_specification.md` (kept in the project knowledge area, not
in this repo). This file is the *implementation* counterpart, recording
what's been done, what's been verified, and what's outstanding.

---

## Audit summary (2026-05-05)

The May 2026 precision-and-performance audit covers the dispatcher's
behaviour across (mode, r, K, A, N, σ, P). Empirical sweeps live in
`python/tests/v22/precision_audit/01_*` through `12_*` with reference
outputs.

| # | Issue                            | Status                            |
|---|----------------------------------|-----------------------------------|
| 1 | `_orbit_inner_rel` 4σ truncation | CLOSED (live code is 8σ)          |
| 2 | K-vs-r Möbius cancellation       | GUARDED (`K ≥ r + 2`, sufficient for r ≤ 4 across σ/P) |
| 3 | rel_per σ/P definitional gap     | DOCS UPDATED (orbit & pairwise are different formulas; dispatcher routes correctly) |
| 4 | Sharp-Gaussian abs cancellation  | GUARDED (runtime cancellation-ratio diagnostic; threshold 1e-10) |
| 5 | σ/P-blind cost model in rel modes | DOCUMENTED (one minor misroute at rel_per r=2 K=14 σ/P=0.01) |
| 6 | σ→0 orbit overflow (music-theoretical regime) | GUARDED (post-hoc sanity check; auto = pairwise in σ/P ≤ 1e-3 regime) |
| 7 | MA per-(n,m) cancellation diagnostic over-fires on self-IP | RESOLVED — diagnostic removed; see "Diagnostic redesign" section below |

Coverage in this audit:
- All four modes
- r ∈ {2, 3, 4, 5}
- K ∈ {r, r+1, r+2, r+3, r+4} for the boundary determination; up to
  K=100 for the high-K spectral sweep
- A ∈ {1, 2}, N ∈ {2, 4, 8} where pairwise can run
- σ/P ∈ {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.025, 0.042,
  0.05, 0.1, 0.2, 0.5, 1.0}, P ∈ {1200, 12000}
- Half-shared p1 / p2 in σ→0 sweep (so cosine is non-degenerate)
- 3 to 30 seeds depending on the sweep

Use-case coverage by audit sweep:
- Perceptual chord analysis (σ/P ≈ 0.025–0.05, K ≤ 10): 01_*, 09_*, 10_*.
- Spectral similarity (σ/P ≈ 0.005–0.04, K = 50–500): 12_*.
- Music-theoretical exact-match (σ→0, K ≤ 12): 11_*.

What remains at the boundary of the audit:
- σ/P-aware orbit-cost factor for routing (low-priority — narrow misroute zone).

**Note on "OOM" framing in the original sweep notes:**
The audit was run on a 4 GB container, where pairwise was killed by
the OS at high K combinations. This was misread as "pairwise infeasible".
In fact `_ip_core_ma` has memory-aware chunking with a 4 GB default
budget; on a normal workstation (≥ 16 GB RAM) pairwise chunks and runs
correctly even at K=64 r=3. It is correctly-but-slowly tractable, not
infeasible. The relevant comparison is therefore "orbit much faster
than chunked pairwise" rather than "orbit only feasible option".
Nothing about the dispatcher's correctness changes, but the framing in
the high-K sweep (12_*) and the σ→0 sweep (11_*) is misleading. To be
re-stated: at high K orbit is decisively faster (10× to 1000×) and is
the correct dispatcher choice; pairwise remains a valid fallback path
on adequately-resourced systems and is preserved as the always-available
correctness backstop.

---

## Phase 1 status (single- and multi-attribute orbit dispatcher)

**Last updated:** 2026-05-05.

### Done

#### Step 1: orbit table generation
- `mpt/_mobius.py` — `inner_product_orbit_grid`, `inner_product_orbit_pw_batched`, orbit-table caching.
- Direct enumeration via integer partitions × contingency tables × within-size-group canonicalisation.
- Cross-checked against direct enumeration at small (r, n) and against pairwise at moderate (r, n).

#### Step 2: cancellation guard
- `cancellation_threshold=1e-12` in `cos_sim_exp_tens`.
- Triggers when the cross-inner-product is near zero relative to either norm.
- Note: one-sided — does NOT catch corruption in auto-IPs <A,A> or <B,B>. The K-vs-r margin in `_orbit_safe_for_precision` is the primary protection there.
- Tested in `tests/v22/test_cancellation_guard.py`.

#### Step 3: SA orbit dispatcher
- `_select_inner_product_method` (SA path).
- Cost-model constants calibrated empirically.
- Tests in `tests/v22/test_dispatcher.py`, `tests/v22/test_method_keyword.py`.

#### Step 4: MA orbit dispatcher
- `_select_ma_inner_product_method` at `mpt/tensor.py:1939`.
- `_ma_per_attr_inner_matrix` (abs paths), `_ma_per_attr_inner_matrix_rel_per` (rel-per branch).
- `_orbit_inner_rel` (rel-non-per branch), per-pair loop using shared `_inner_product_orbit_grid`.
- Cost-model fit completed across abs/rel × per/nonper × A∈{1,2} × r∈{2,3,4} (277 cells, σ=50, P=1200).
  - 94% of measured cells within 5% of optimal route.
  - 0 OOM-zone misroutes.
- Precision audit (May 2026) covers (mode, r, K, A, N, σ, P) — see audit summary above.
- Tests in `tests/v22/test_ma_orbit.py`.

### Issue details

#### Issue 1: `_orbit_inner_rel` truncation window — CLOSED
- Investigation: 4σ buffer leaves integrand tail ~5e-10, producing
  ~5e-7 cosine errors at rel_nonper r=2 K=4 N=4 A=2.
- Live code in `mpt/tensor.py:2587` `_orbit_inner_rel` already uses 8σ.
  Verified at 30 seeds A=2 N=4 r=2 K=4 rel_nonper: median 1.40e-15,
  max 5.77e-15. The previous handoff summary's claim that the fix was
  "in /tmp scripts only" was stale.
- Reference: `precision_audit/02_*`, `03_*`.

#### Issue 2: K-vs-r Möbius cancellation — GUARDED with `K ≥ r + 2`
- Comprehensive sweep at σ/P=0.025 (sharp), 30 seeds, all four modes,
  r ∈ {2, 3, 4, 5}, K ∈ {r, r+1, r+2, r+3} (`precision_audit/10_*`).
- K=r: catastrophic in all 4 modes for all r (pure cancellation).
- K=r+1: BAD in abs modes for r ≥ 3 (max 1e-3 to 1e+8 wild). BAD in
  rel_nonper for r ≥ 4. Safe in rel_per (FP at r ≤ 3, WARN 1e-12 at r=5).
- K=r+2: FP precision for r ≤ 4 in all modes. At r=5, abs modes show
  occasional 1e-6 outliers under sharp Gaussians (σ/P=0.025) — this is
  Issue 4 amplification (see below), not pure cancellation.
- The current `_orbit_safe_for_precision` predicate enforces `K ≥ r + 2`
  uniformly, which is sufficient for r ≤ 4 across the σ/P range and for
  r ≥ 5 at moderate σ/P. The compound failure mode (high r + sharp σ/P
  in abs modes) is the Issue 4 territory.
- Reference: `precision_audit/01_*`, `10_*`.

#### Issue 3: rel_per orbit vs pairwise discrepancy at σ/P > 0.03 — DOCS UPDATED
- **NOT a bug.** Pairwise (`_compute_Q` per-pair-wrap form) is the closed
  form from JMM article §3.3 Eq. 3.4 — the v2.1 toolbox value, by
  definition. Orbit (1D quadrature over circle translates) computes a
  related but distinct integral. They coincide to FP at σ/P ≲ 0.03 and
  diverge by Theta-function tail terms at higher σ/P.
- Current dispatcher routing direction is correct: orbit at σ/P ≤ 0.03,
  pairwise above (preserves v2.1 toolbox value exactly).
- σ/P sweep (`precision_audit/06_*`) characterises the magnitude of the
  divergence at higher σ/P (1e-7 at σ/P=0.05; 1e-2 at σ/P=0.2).
- Identical magnitude at P=1200 vs P=12000 for matched σ/P confirms σ/P
  is the only controlling parameter.
- **Documentation softening done** in `tensor.py` lines 2455–2470:
  removed the "analytically exact for any sigma/P in periodic-relative
  mode" phrasing in favour of explicit reference to JMM Eq. 3.4 as the
  definitional value.
- Reference: `precision_audit/04_*`, `05_*`, `06_*`.

#### Issue 4: sharp-Gaussian Möbius cancellation in absolute modes — GUARDED via runtime cancellation diagnostic
Found during the σ/P precision sweep at A=1 N=2:

| mode       | r | σ/P  | max rel error | K-r=2 | K-r=4 |
|------------|---|------|---------------|-------|-------|
| abs_nonper | 3 | 0.01 | 1e-4 to 4e-4  | BAD   | FP    |
| abs_nonper | 4 | 0.01 | 1.5e-2        | BAD   | 2e-9 BAD |
| abs_per    | 3 | 0.01 | 1e-4 to 4e-4  | BAD   | FP    |
| abs_per    | 4 | 0.01 | 4e-4 to 6e-4  | BAD   | 2.7e-9 BAD |
| rel_nonper | 4 | 0.01 | 2e-12         | WARN  | (rel orbit averaging damps it) |
| rel_per    | r | 0.01 | FP            | OK    | (orbit and pairwise both agree below σ/P=0.03 threshold) |

- **Mechanism:** the orbit Möbius identity expands the inner product as
  an alternating sum over partition lattice. When the kernel matrix
  entries are small (sharp Gaussians, σ ≪ data range), the alternating
  sum's terms have comparable magnitudes and partially cancel;
  floating-point loss amplifies relative error. K-vs-r margin reduces
  the effect (compare K-r=2 vs K-r=4 columns) but does not eliminate
  it at extreme σ/P.
- **Independent of P** (verified): identical errors at P=1200 vs P=12000
  for matched σ/P. The controlling parameter is σ/data_range; σ/P is a
  conservative proxy because users sample `p` within `[0, P]`.
- **rel modes are largely immune** because the orbit path averages over
  the relative orbit (1D quadrature in `_orbit_inner_rel`), which damps
  cancellation magnitude.
- **Practical exposure:** typical music data has σ ≈ 33 cents on
  P=1200 (σ/P ≈ 0.028) and chord spans < 1 octave (σ/data_range > σ/P).
  Realistic analyses sit in the borderline zone; users running finer
  σ analyses (perceptual fine-grained models, near-just-intonation
  studies, microtonal analyses) are exposed.
- **Fix in place:** runtime cancellation-ratio diagnostic. The orbit
  helpers (`inner_product_orbit`, `inner_product_orbit_grid`,
  `inner_product_orbit_pw_batched`) now optionally return
  ``cancellation_ratio = |total| / max(|term|)`` from the alternating
  sum. The cosine wrappers (`_cos_sim_exp_tens_*_orbit`) aggregate the
  worst-case ratio across all per-attribute matrices and three IPs and
  return it as a fourth value. The dispatcher checks this against
  ``_ORBIT_CANCELLATION_RATIO_MIN = 1e-10`` and falls back to pairwise
  if the orbit alternating sum has lost too many significant digits.
  Threshold rationale: at 1e-10 the result has ~6 surviving decimal
  digits, sufficient for 1e-6 user tolerance with margin.
- **Verified:** at r=4 K=6 σ/P=0.01 abs modes (the worst-case cell
  from the σ/P sweep), all 10 seeds × {abs_nonper, abs_per} now match
  pairwise to FP precision under method='auto'. Direct orbit calls in
  the same cells show ratios well below 1e-10 (typically 1e-16 to
  1e-17) and silent relative errors up to 5e-3 — confirming the
  diagnostic is doing real work.
- **Cost of the diagnostic:** one extra `max` reduction per orbit
  class in the alternating sum. Negligible overhead in benchmark
  (~1% of orbit cost).
- Reference: `precision_audit/06_*`,
  `tests/v22/test_sigma_to_zero_fallback.py`.

#### Issue 5: σ/P-blind cost model in rel modes — DOCUMENTED
- The cost model in `_predict_orbit_cost_ms` does not account for σ/P.
  In rel modes, orbit's grid size scales as ~1/σ at fixed period
  (or fixed data range for non-periodic), so actual orbit cost varies
  ~10× across σ/P at fixed (r, K).
- **Practical impact:** one misroute observed in the routing test at
  rel_per r=2 K=14 σ/P=0.01 — cost model predicts orbit 9.1 ms,
  measured orbit 21.1 ms vs pairwise 9.1 ms. Dispatcher picks orbit;
  call is ~2× slower than optimal. No correctness impact.
- **Roadmap:** add σ/P scaling factor to `_predict_orbit_cost_ms` for
  the rel-mode constants. Low priority — the misroute zone is narrow,
  the penalty is bounded at ~2×, and most users operate at moderate σ/P
  where the model is accurate.
- Reference: `precision_audit/07_*`, `08_*`.

#### Issue 6: σ→0 orbit overflow in music-theoretical regime — GUARDED via post-hoc sanity check
- σ→0 (or σ very small) is the music-theoretical exact-match regime: a
  legitimate, common use case in academic music analysis where users
  want exact pitch coincidence rather than perceptual approximation.
- Pairwise handles σ→0 cleanly: kernel entries become {0, 1}, inner
  product becomes a count of matched tuples, cosine is the exact-match
  proportion. No numerical issue.
- Orbit at σ→0 with r ≥ 3 in abs modes: the Möbius alternating sum
  involves products of kernel entries that have collapsed to 0 or 1,
  so cancellation is severe and the result can overflow to magnitudes
  like 1e+283 (verified empirically).
- **Failure regime, abs modes** (precision_audit/11_*, partial sweep):
    - K = r+1: BLOW at σ/P ≤ 1e-3 for all r ∈ {2, 3, 4, 5}.
    - K = r+2: FP at σ/P=1e-7 only for r=2; BLOW at σ/P ≤ 1e-3 for r ≥ 3.
    - K = r+4: FP at σ/P=1e-7 for r=2, 3; r=4 still BAD/WARN; r=5 untested.
- **Failure regime, rel modes:** orbit's trapezoidal u-grid is
  `period/σ × samples_per_σ` points — at σ/P=1e-7 that is 1.2e8
  points, infeasible for any practical run. Effectively orbit cannot
  run in this regime even before precision concerns.
- **Fix in place:** `_orbit_ips_look_corrupted` helper called after
  `_cos_sim_exp_tens_*_orbit` returns. Triggers on:
    - non-finite (NaN/Inf) IPs,
    - negative auto-IPs (sign flip on a Gram-matrix diagonal),
    - cosine magnitude > 1 + 1e-6.
  When triggered, the dispatcher falls back to pairwise.
- **Verified:** at σ/P ∈ {1e-3, 1e-4, 1e-5, 1e-7} for r=4 K=6 in both
  abs_nonper and abs_per, `cos_sim_exp_tens(method='auto')` now
  matches `cos_sim_exp_tens(method='pairwise')` to FP precision.
  Regression test in `tests/v22/test_sigma_to_zero_fallback.py` (10
  cases, all passing). Full suite: 462 passed.
- **What this does NOT catch:** Issue 4's quiet-corruption regime
  (orbit returns finite-looking values that are wrong by 1e-4 to 1e-2).
  That requires the runtime cancellation diagnostic on the roadmap.
- Reference: `precision_audit/11_*`, `tests/v22/test_sigma_to_zero_fallback.py`.

### Pending — to complete before commit

#### Apply Issue 3 documentation softening — DONE in this commit.

#### Audit gaps closed in this pass
- σ and P swept across {0.01, 0.025, 0.042, 0.05, 0.1, 0.2, 0.5, 1.0}
  × {1200, 12000} for both precision and performance.
- A and N: A ∈ {1, 2} × N ∈ {2, 4, 8} for precision (A=4 not testable —
  pairwise OOM zone exactly where orbit dominates).
- K-vs-r boundary characterised at 30 seeds across r ∈ {2..5}.

#### Roadmap items (not blockers for v2.2)

1. **Issue 4 runtime cancellation diagnostic.** Modify
   `inner_product_orbit` (`mpt/_mobius.py`) to compute and return the
   cancellation ratio. Wire fallback through SA and MA orbit cosine
   wrappers. Add regression tests.
2. **σ/P-aware orbit cost factor.** Modify `_predict_orbit_cost_ms` for
   rel modes to scale by `1/σ` or `period/σ` as appropriate. Re-run cost
   benchmark to refit constants.
3. **A=4 orbit-only benchmark.** Verify orbit dispatcher correctness
   in the OOM zone where pairwise cannot run. Test against direct
   enumeration at small (r, n) where direct is feasible.

### Backlog (deferred to later phases)

- Step 5: `_cos_sim_exp_tens_windowed` with K → K_W substitution.
- Step 6 — **DONE**: `entropy_exp_tens` Rényi-2 branch (see below).
- Step 7: `tensor_harmonicity` orbit point-evaluator path; remove K>3
  duplication warning.
- Step 8: `windowed_similarity` updates.
- Pre-built orbit table pickles for r ∈ {2..6}.
- MATLAB port (Phase 4).
- r ∈ {7, 8} tables (Phase 5).

### Step 6 — entropy_exp_tens Rényi-2 branch

`entropy_exp_tens` now accepts `method='shannon'|'renyi2'`, default
`'shannon'` (unchanged behaviour for existing callers). The
`'renyi2'` path computes the analytical continuous Rényi-2 entropy
H_2 = -log_b(<T,T> / Z²) using:

- Orbit-Möbius IP for `<T,T>` (via `_orbit_inner_abs` /
  `_orbit_inner_rel` for SA, via `_ma_per_attr_inner_matrix` per
  attribute and per-event-pair matrix product for MA), inheriting the
  runtime cancellation diagnostic added in the previous commit. A
  RuntimeWarning is raised if the worst-case ratio drops below 1e-10.
- Closed-form `total_mass_abs` / `total_mass_rel` for Z (SA), and
  `Z = Σ_n Π_a Z_a^(n)` for MA (per-event-per-attribute total mass).
- Direct r=1 abs branch (single matrix multiply and total_mass),
  bypassing the orbit table which is undefined at r=1. r=1 rel is
  degenerate (0-D relative space) and returns 0 by convention.

`normalize=True` with `method='renyi2'` raises NotImplementedError —
the natural normaliser log_b(V) yields a (-∞, 1] range rather than
Shannon's [0, 1], so a direct uniform normaliser would be misleading.
Users wanting normalised values can divide by an appropriate
log_b(V) externally; resolving the normaliser API is left for a
follow-up commit.

Verified against direct grid integration of (T/Z)² across all four
SA modes and several MA configurations to FP precision in periodic
modes and to grid-quadrature precision in non-periodic. The
analytical path scales to high r where the existing Shannon grid path
strains: r=6 SA K=12 takes ~100 ms in closed form (Shannon would need
a 6-D grid).

Limitation: `WindowedMaetDensity` is not yet supported on the
`renyi2` path (the windowed-IP machinery is Step 5; entropy will
follow once that lands). Calling `entropy_exp_tens(windowed,
method='renyi2', normalize=False)` raises NotImplementedError.

Tests in `python/tests/v22/test_renyi2_entropy.py` (18 cases): SA
all-modes-vs-grid, MA two-attribute and mixed-r-and-K cases, MA
relative-attribute, API-contract checks (method validation,
normalize=True rejection, default-is-Shannon, Renyi-2 ≤ Shannon
inequality on r=1).

### Integration-technique notes for the rel-mode u-grid (v2.3+ wishlist)

The orbit method in relative mode integrates the Möbius kernel sum over
a translation parameter u — over [0, P) for periodic, over a Gaussian-
bounded window for non-periodic. The current implementation uses
trapezoidal quadrature with `period/σ × samples_per_σ` points (and
analogously in the non-periodic case). At small σ/P this becomes
infeasibly large (~1e8 points at σ/P=1e-7), and below σ/P ~ 1e-3 the
grid is the binding constraint on rel-mode orbit usability.

Note that integration techniques only help in rel modes. The abs-mode
orbit failures (catastrophic overflow at σ→0, sharp-Gaussian quiet
corruption) are pure floating-point cancellation in the alternating
Möbius sum across orbit classes, with no integral involved.

Three approaches worth considering when this regime needs to ship:

1. **Adaptive quadrature** (Gauss–Kronrod, scipy.integrate.quad). The
   integrand F(u) is a sum of Gaussian-like spikes of width ~σ at
   pitch-pair-alignment translations; adaptive methods refine only
   near the spikes. For K=10 at σ/P=1e-3 this is roughly 1000
   evaluations vs 100,000 trapezoid points — about 100×. Drop-in via
   scipy. Modest dev cost. Worth piloting first.

2. **Gauss–Hermite quadrature** per spike. Designed for ∫ f(x) exp(-x²) dx;
   converges faster than adaptive trapezoid for known-Gaussian
   structure, but requires per-orbit identification of spike centres.
   Likely overkill given option 1 should suffice.

3. **Closed-form Theta-function evaluation per orbit.** The most
   principled. Each orbit's integrand is a product of Gaussians in u;
   products of Gaussians are Gaussians; a Gaussian integrated over
   [0, P) on the torus is a Theta function value. So the per-orbit
   integral can be evaluated analytically with no grid. This is
   exactly what the v2.1 pairwise path already does for r=2 (each
   pair's u-integral is closed-form); generalising to higher-r orbit
   terms would unify orbit and pairwise into one analytical framework.
   Substantial work — v2.4+ architectural project.

Caveat: even with perfect integration, the orbit Möbius alternating
sum across orbit classes can still cancel catastrophically at small σ
in any mode. Theta-orbit might improve conditioning (cleaner
per-orbit values feed into the same alternating sum) but does not
guarantee it. Empirical investigation needed before committing.

Practical scoping: not blocking for v2.2 because the dispatcher's
σ/P ≈ 0.03 split between the two formulas in rel_per, plus the
catastrophic-overflow fallback in abs r ≥ 3, already keep users out
of the regime where integration would matter. Add to v2.3+ wishlist
if/when orbit needs to handle σ → 0 in rel-periodic mode without
falling back to pairwise.

---

## Diagnostic redesign (2026-05-05)

### Background

The original v2.2 orbit dispatcher gated a fallback to pairwise on a
per-orbit-class cancellation ratio (`worst_ratio`) returned alongside
each inner product. The threshold was `_ORBIT_CANCELLATION_RATIO_MIN
= 1e-10`. The intent was to catch the "sub-catastrophic" regime —
finite-looking IPs that have nevertheless lost most of their
significant digits.

### What the seven-regime sweep found

Comprehensive sweeps (`python/sweep_self_ip.py`, see also
`tests/v22/standard_regimes.py`) across:

1. MA self-IP across realistic configurations
2. SA r ∈ {5, 6} abs_per
3. SA abs_nonper at sharp σ
4. Adversarial pitch arrangements (12-TET, near-degenerate,
   clusters, just intonation)
5. Realistic spectral weights (harmonic α=1, steep α=2,
   shallow α=0.5; K up to 64)
6. Multiple seeds per cell (5 minimum)
7. High K + high r (K up to 100)

revealed two distinct findings:

* **MA path**: the `worst_ratio` aggregation across the
  `(N_x × N_y)` per-attribute IP matrix produces a 100 % false-alarm
  rate on self-IPs at typical musical sigmas. The off-diagonal
  entries (cross-IP between distinct events) can have low per-entry
  cancellation ratios while the diagonal entries — which dominate
  the sum `Σ_{n,m} P_xx[n,m]` — are clean. The cosine and entropy
  consume only the sum, so per-entry cancellation in small-magnitude
  off-diagonal entries is irrelevant. Grid integration of `T(x)²`
  in flagged cells confirmed the orbit values match to 4×10⁻¹⁶.

* **SA renyi-2**: the per-orbit-class ratio never dropped below
  ≈ 0.13 in any tested cell — well above the 10⁻¹⁰ threshold. The
  diagnostic was vestigial.

### Resolution

The MA `worst_ratio` aggregation in `_cos_sim_exp_tens_ma_orbit`
and the parallel diagnostic in `_renyi2_exp_tens_ma` /
`_renyi2_exp_tens_sa` were removed in favour of:

* the existing structural K-r ≥ 2 guard at the dispatcher level
  (the load-bearing protection),
* the existing σ/P > 0.03 guard for periodic-relative,
* the existing σ → 0 fallback,
* the cross-cancellation guard (cosine path only),
* a post-hoc finite/positive check that raises
  `FloatingPointError` (entropy) or falls back to pairwise (cosine)
  on non-finite, non-positive, or sign-flipped IPs.

The SA cosine `cancellation_too_severe` path was kept — it remains
empirically reliable in SA mode and the SA cross-IP can genuinely
cancel.

### Validation

Standard test regime (`tests/v22/standard_assessment.py`) over 1475
cells across all eight regimes:

| Quantity                       | Cells | Failed | Skipped (pairwise infeasible) |
|--------------------------------|-------|--------|-------------------------------|
| `cos_self` (= 1)               | 1475  | 0      | 0                             |
| `cos_cross` (auto vs pairwise) | 1475  | 0      | 316                           |
| `renyi2` (finite + positive)   | 1475  | 0      | 0                             |

The skipped cells in `cos_cross` are those where pairwise OOMs
under the standard 5000-tuple memory budget; orbit ran for those
cells and `cos_self` confirmed self-similarity = 1 to FP precision.

### Residual gap (v2.3 deliverable)

A regime that produces a finite, positive, but slightly inaccurate
self-IP from sub-catastrophic Möbius cancellation in the orbit
alternating sum. None observed across the 1475 cells of the
standard regime, but "no observed instance" is not a proof.

The right closure is a **sum-level** cancellation diagnostic: track
partial sums during the orbit Möbius accumulation; if the magnitude
of the partial sum at any intermediate step is many orders of
magnitude larger than the final sum, that signals a real loss of
significant digits in the SUM (not the per-entry false-alarm trap
the original diagnostic fell into). The implementation requires
plumbing partial-sum tracking through `_orbit_inner_abs`,
`_orbit_inner_rel`, and `_ma_per_attr_inner_matrix`; the threshold
calibration would re-use the per-orbit `_ORBIT_CANCELLATION_RATIO_MIN`
discipline.

Until v2.3 ships this, the precision envelope is documented in the
public docstrings of `cos_sim_exp_tens` and `entropy_exp_tens` so
users have an honest characterisation of what the orbit path
guarantees and what it does not.

### Standard test regime as a permanent artifact

`tests/v22/standard_regimes.py` consolidates the eight regimes
(R0 σ→0, R0b rel-per, R1 MA self-IP, R2 high r, R3 abs-nonper,
R4 adversarial, R5 spectral, R7 high K) into a parameterised cell
list. `tests/v22/standard_assessment.py` registers each
quantity-of-interest as a `compute_fn` plus optional `reference_fn`.
When future quantities arrive (windowed similarity, Möbius-based
`eval_exp_tens`, harmonicity orbit path, salience read-out), they
are added by registering one new function — the same battery of
1475 cells (or 5900 with `thorough=True`) applies automatically.

---

## Conventions

- British spelling, -ize endings.
- Unspaced em-dashes (—).
- camelCase MATLAB / snake_case Python.
- All public API preserves v2.1 cosine to FP precision; orbit method is
  internal and routed only where it agrees.

## File map for v2.2 work

- `python/mpt/_mobius.py` — orbit table generator and low-level inner-product helpers.
- `python/mpt/tensor.py` — dispatcher and pairwise/orbit branches in
  `_select_*_inner_product_method`, `_ma_per_attr_inner_matrix*`,
  `_cos_sim_exp_tens_*_orbit` / `_pairwise`.
- `python/tests/v22/` — pytest unit tests.
- `python/tests/v22/precision_audit/` — exploratory sweeps and reference outputs.

---

## Phase 1 complete (2026-05-10)

Commits 1–7 closed. Cross-language parity established across the v2.2 surface:

- **Commits 1–5.** Foundation: `+mobius` package (MATLAB) and `_mobius` module (Python) with orbit table builder, partition / set-partition / contingency-table enumerators, contraction helper, pre-built tables for r ∈ {2..6}, lazy density-struct refactor in `buildExpTens` + `ensureExpTensExpensive` helper.

- **Commit 6.** Single-attribute (a–b) and multi-attribute (c) cosine similarity and point-evaluation dispatchers with the orbit method. Three-layer guard for SA (cross-cancellation, corruption, severe ratio); simpler heuristic for MA. Renyi-2 entropy (e) for SA and MA via orbit IP and analytical total mass. `tensorHarmonicity` rewrite (d) bypassing `buildExpTens` entirely, routing through `mobius.evalOrbitRel` with per-template caching.

- **Ragged-K hybrid.** Per-event safe/unsafe partition in MA per-attribute IP wrapper. Safe×safe pairs flow through vectorised batched orbit; pairs involving any unsafe event flow through direct enumeration (no Möbius alternating sum, no cancellation). Cross-language parity: same threshold (`_ORBIT_K_MINUS_R_MIN = 2`), same partition logic, same dispatch. Replaces the earlier zero-pad-everything fallback.

- **Commit 7.** Cross-language equivalence tests (hardcoded golden values; 7 cases covering SA / MA cosine, Rényi-2, tensorHarmonicity, evalExpTens). Speed-comparison spot-checks (`bench_orbit_xlang.{m,py}`) at five (r, K) configurations.

  *First iteration:* MATLAB 14–32× slower than Python on orbit IP/eval (median 22×). Profile localised the cost: `mobius.contract`'s greedy pair-picker spent ~86% of wall time in `intersect` / `setdiff` / `unique` / `ismember` calls (~700k set-operation calls per single r=5 IP), with `pagemtimes` and arithmetic not even appearing in the top 20.

  *Stage A (orbit contraction-graph precomputation, both languages):* moved the contraction-graph construction from runtime to table-build time. **MATLAB**: each orbit struct gains three precomputed recipes (`recipeIP`, `recipeGrid`, `recipeBatched`); runtime executes precomputed permutations via `mobius.executeRecipe` — zero set operations. **Python**: each `OrbitEntry` gains three precomputed `np.einsum_path` outputs (and one pre-built `einsum_str_pw_batched` replacing per-call string-patching); runtime passes `optimize=path` to `np.einsum`. Backward-compat paths in `mobius.getOrbitTable` (MATLAB) and `_mobius._ensure_paths` (Python) augment old `.mat` / `.pkl` files on first load. Tests: 9 MATLAB cases in `test_recipe_equivalence.m`, 11 Python cases in `test_orbit_path_equivalence.py`. **Result**: MATLAB median speedup 32× (range 12–74×); MATLAB / Python ratio dropped from 22× to 0.53× (MATLAB now generally faster). Python median speedup 2.2× (range 1.6–3.9×) on top of its already-fast baseline. Stage B (`tensorprod` substitution) was prepared but skipped — the gating concern was crushed by Stage A alone.

  Documentation updates across CHANGELOG, MIGRATION, USER_GUIDE.

Test count at Phase 1 close: **683 MATLAB / 340 Python v22**.

### Deferred to v2.3+

- Adding raw-array overload to `windowedSimilarity` (the only similarity-and-evaluation function still requiring a pre-built density). Would unblock `demo_helixBlend`, `demo_maetWindowing`, `demo_windowingReference` from their explicit `buildExpTens` calls. TODO note co-located in source.
- r ∈ {7, 8} orbit tables. Cost-model crossover currently unfavourable beyond r=6 in standard regimes; deferred until a use case requires it.
- Tightening the K-vs-r precision guard (`_ORBIT_K_MINUS_R_MIN = 2`) — the empirical-calibration constant. Could be replaced with a dynamic per-call cancellation prediction once enough audit data accumulates.
- Test redundancy audit on the v22 corpus (~130 tests). Lazy-density tests and Möbius-vs-toolbox cross-validation tests are the prime candidates. Roughly 1–2 hours of focused work; defer until v2.3 stabilises.

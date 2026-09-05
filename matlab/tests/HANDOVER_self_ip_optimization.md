# Handover: self-IP memoisation, oneSidedDenom skip, and the r = 1 route

## What shipped (both languages, v3)

Origin: `HANDOVER_sweep_optimization.md` measured ~42 ms/offset on the
point-set-sweep shape (N = 1200 context, 3-event query,
`oneSidedDenom`). Investigation confirmed its hypothesis 1 (the context
self term `<X,X>` recomputed — and under `oneSidedDenom` discarded —
once per pair), refuted hypothesis 2's mechanism (no dispatch-path
switch; the superquadratic break is the quadratic self term compounded
by cache-throughput falloff, 76 → 28 kernel evals/µs as the (A, N, N)
temporaries leave cache), and mostly refuted hypothesis 3 (warm-cache
per-pair overhead is small; the apparent overhead was one-off
calibration).

Four changes, mirrored across Python and MATLAB:

1. **`oneSidedDenom` no longer computes `<X,X>`.** Both triple routines
   accept a need-xx flag; the guard and finalisation tolerate the
   absent value (`None` in Python, `[]` in MATLAB), and the finaliser
   raises if a skipped `<X,X>` ever reaches `'cosine'`.
2. **Self inner products are memoised**, keyed by route + resolved
   `truncationSigmas` + kernel precision + (Möbius) the per-attribute
   closed-form-vs-grid choices, hoisted before the orbit loop because
   the closed form drops a per-attribute prefactor that cancels only
   within one route's triple. Möbius entries are purged if the
   post-hoc guard trips. Python: cache dict on `MaetDensity`, plus
   memoised `pruned()` (essential — the dispatcher prunes per pair, and
   the shared pruned object is what carries the cache across a sweep).
   MATLAB: value-semantics cache struct (`keys`/`vals`) threaded
   through the sweep loops in-call, plus optional cache-carrying
   outputs `[s, densX, densY] = cosSimExpTens(densX, densY, ...)`
   (`selfIP` field) for manual scalar loops; refused with
   `cosSimExpTens:selfIpOutputsUnavailable` outside that form.
3. **Direct all-r = 1 inner-product route** (`_ip_r1_direct` /
   `ipR1Direct` inside the MA IP core): one accumulator, in-place
   arithmetic, plain exponential; same per-attribute terms in the same
   accumulation order, same truncation threshold including the
   `-log(nTerms)` tightening, same chunking heuristic. Relative
   attributes at r = 1 contribute exactly zero, as the generic
   quadratic form evaluates them. Bit-identical to the generic path in
   Python validation; twelve-digit cross-language parity (below).
4. **Cost models follow the computed work.** `estimate_comp_time` /
   `estimateCompTime` receive only the pairs actually evaluated; both
   selectors (`_select_ma_inner_product_method`,
   `internal.selectMaInnerProductMethod`) gained per-route skip flags
   threaded into both pricing formulae. Defaults reproduce legacy
   pricing exactly. *(Superseded: the flags are now shared by the two
   routes' prices --- per-route flags priced the first route to run at
   one matrix and its rival at three, which locked that first choice in.
   The memoised values stay route-keyed; only the pricing is shared. See
   `cosine._self_ip_memoised` / `internal.selfIpMemoised` and
   `tests/test_self_ip_memo_sharing.*`.)* The absolute-attribute Möbius constants and the
   grid term are scaled by n_matrices/3 under skips — an admitted
   approximation, biased toward Bulger (cheap-to-mispick side).

## Measurements (Python, container, single core)

Handover shape, `oneSidedDenom`: 42.4 → 0.90 ms/offset at N = 1200
(~47×), linear in N restored; `cosine` now 1.15 ms/offset (self term
paid once). Inner-product work is 0.24 ms/offset; the residual ~0.6 ms
is per-query density build (`_ma_build_perm_arrays` ~0.5 ms) plus
dispatch — the agreed second-pass target, not kernel work. The raw
reference remains ~0.06 ms/offset.

## Verification status

- Python: full suite 1,981 passed / 14 skipped / 0 failed (three
  slices); new `python/tests/test_self_ip_cache.py` (10 tests).
  Broadcast == fresh scalars to exactly zero difference on both routes,
  both normalisations, including auto-dispatch Möbius at r = 3.
- MATLAB: **runs under Octave 8.4 for all r = 1 paths** — this session
  executed rather than only hand-traced. Sweep == fresh scalars (0
  diff, both normalisations); threaded loop via 2nd output == sweep (0
  diff) with `selfIP` populated; cross-language parity on identical
  fixed inputs exact to all 12 printed digits across nonper / perA /
  perAB, both normalisations. New
  `matlab/tests/test_self_ip_cache.m` registered in `test_mpt.m`;
  its Octave-runnable subset passes 15/15 under the suite baseline.
  Golden values embedded in the test are Python-computed on
  formula-based (RNG-free) inputs **at `truncation_sigmas = inf`** ---
  the un-truncated accuracy-floor path `mptTestIsolateDefaults` pins as
  the suite baseline. An earlier revision computed them at the factory
  default (6) and failed six golden checks on Andrew's MATLAB run
  (the internal-consistency checks all passed, correctly isolating the
  fault to the constants); perAB coincidentally passed because every
  wrapped distance there falls inside the effective truncation window.
  Note the deliberate `reshape(...).'` to match NumPy row-major order.
- **NOT yet verified: the MATLAB Möbius (r >= 2) path.** Octave cannot
  reach it — two pre-existing incompatibilities, untouched by this
  work: `max(..., [], 'omitnan')` in `localCosSimMA`'s `nuVecSel`
  computation, and `arguments` blocks throughout `+mobius`. Andrew
  runs: `clear cosSimExpTens; clear internal.selectMaInnerProductMethod;
  rehash; test_mpt`. The new test's Möbius section (repeat-vs-fresh
  equality both normalisations, plus abs r = 3 goldens
  cosine 1.566062218777721e-01 / oneSidedDenom 1.497935199140709e-01,
  1e-9 rel) covers exactly the unverified edits. Andrew's first
  test_mpt run already confirmed the Möbius repeat-vs-fresh equality
  checks pass on real MATLAB; the golden checks await a re-run with
  the corrected constants.

## Decisions settled this session (do not re-open)

- MATLAB memoisation design: in-call threading + optional
  cache-carrying outputs; **not** a handle-class conversion (reference
  semantics would silently change assignment/copy meaning toolbox-wide)
  and **not** returned-struct-only (in-call covers the standard
  `'cosine'`-normalised sweep with no calling-convention change).
- r = 2 relative K = 2 collapse to an r = 1 absolute density on the
  difference values (σ' = σ√2; same identity as interval-mode
  sameness, V = 2σ²): **too niche to implement** — that case is already
  fast. Future-directions entry text is in this handover (below).
- Chunking granularity deliberately not in the memo key (perturbs only
  accumulation order, within the ≤ 1e-12 parity discipline).
- `method='factored'` and the nested-contract route keep full-triple
  computation (explicit/diagnostic routes; out of scope).
- `batchCosSimExpTens` / batched-raw single-multiset path untouched
  (its dedup already avoids recomputation across duplicate rows).

## Open items, in order

1. Andrew: run `test_mpt` (MATLAB), esp. the Möbius section of
   `test_self_ip_cache.m`.
2. Second pass: per-query overhead. Two targets: (a) an r = 1 shortcut
   in the lazy build (`_ma_build_perm_arrays` — at r = 1, K = 1 the
   perm/comb arrays are views of the inputs); (b) hoist structural
   validation + method selection out of the broadcast loop (shared
   geometry across pairs). Expected: sweep to ~0.1–0.3 ms/offset.
3. Re-benchmark both languages for the cost models (after 1 and 2):
   verify near-crossover routing with skip flags active (first pair
   full-triple, later pairs cross-only), and consider refitting the
   n_matrices/3 scaling with measured per-matrix constants. Repeated
   timings, not single container runs; MATLAB half on Andrew's machine
   (`tools/calibrateRelIpCost.m` with `'check', true` first).
4. Paste into `future_directions.md` (Drive; no in-place update from
   here): "**r = 2 relative non-periodic K = 2 fast route.** Such a
   density is an r = 1 absolute density on the within-tuple difference
   values with σ' = σ√2 (the identity behind interval-mode sameness as
   a MAET inner product, V = 2σ²), so it could be routed to the direct
   all-r = 1 inner-product path. Deferred: the case is already fast,
   the collapse covers only K = 2, and the constant prefactors and
   [sym] handling would need careful verification."

## Recurring-error notes for future sessions

- The near-miss pattern this session was **altering predictor/pricing
  defaults while adding flags** (twice caught mid-edit: the A == 0
  branch of `_predict_pairwise_kernel_size`, and self-matrix pair
  counts in `_predict_orbit_cost_ms`). Rule: when adding optional
  behaviour, first diff the default path against the original
  expression symbolically.
- Octave is now a partial MATLAB verification channel in the container
  (`apt-get update && apt-get install octave`; parse-check via
  `@fnName`, run r = 1 paths directly). Its limits: no `arguments`
  blocks (all of `+mobius`), no `'omitnan'` flag in `max`/`min`.
- **Cross-language goldens must be computed under the settings the
  consuming test runs with.** The suite baseline is
  `truncationSigmas = Inf` (see `mptTestIsolateDefaults`), not the
  factory default of 6; goldens computed at the factory default fail
  1e-12 checks by construction wherever truncation bites. Modes whose
  wrapped distances all fit inside the effective truncation window
  coincide at either setting and can mask the mismatch.
- MATLAB list-mode previously recursed through the public entry point
  and forwarded only `normalize`/`verbose` (method / truncationSigmas /
  cancellationThreshold silently re-defaulted). The new
  `localScalarPairDispatch` preserves that behaviour deliberately —
  it is a pre-existing limitation, not a new one.

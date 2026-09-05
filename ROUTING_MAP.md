# Routing map — Music Perception Toolbox (Python `mpt` and MATLAB)

> Supporting material in `docs/routing/`: the per-language maps this document was reconciled from (`routing_map_python.md`, `routing_map_matlab.md`), the decision-by-decision parity report with full evidence (`routing_parity.md`), the specification the mappers followed (`routing_map_spec.md`), and the decision trees as TikZ/forest figures (`routing_trees.tex`, compiled to `routing_trees.pdf`) for the user guide or the online supplement.

> **Status (rounds 1–3, September 2026).** Every finding in §10 has since been resolved: A-2, A-3, A-4 and the `K_a` item in the first pass; A-1, A-6, A-7, A-9, A-10, A-11, A-17, B-1, B-3, B-5, B-8 to B-14 in round 1; A-5, A-8, A-12, A-13 (removed), A-14, A-15, A-16, B-6, B-7 in round 2; B-16, B-17, all C items, the D orphans (deleted, or moved under `tests/` as reference oracles), and `explain_dispatch`'s fidelity in round 3. The trees and tables below describe the code before those rounds where a `[Py]`/`[M]` tag marks a difference; after them the two languages share every rule. Terminology: what earlier drafts called the "SA path" is the single-multiset corner (A = N = 1) of the one density type, a speed branch of the general path.

## 0. Preamble

### Purpose

This document is the single, language-neutral map of every routing decision in the toolbox: which computational route each public entry point takes for a given input, which guards and overrides bear on that choice, which leaf routine finally does the arithmetic, and under which measure. It serves two ends:

1. **Verification of routing and parity.** Each decision was traced in both implementations (`/home/claude/mpt/python/mpt` and `/home/claude/mpt/matlab`) and the two traces were aligned by meaning. Where the languages differ, the node is tagged and cross-referenced to a finding in §10.
2. **Reference for the user guide and the online supplement.** The trees in §1–§6 and the tables in §7–§9 are the authoritative description of what `method=`, `wrap=`, `truncation_sigmas=`, and the global defaults actually do.

The per-language maps (`routing_map_python.md`, `routing_map_matlab.md`) and the parity report (`routing_parity.md`) carry the full file:line evidence; this document condenses them. Where the two maps disagreed, the source was re-read and the result is stated inline. The parity report's 98-row decision table and its findings A/B/C/D are authoritative where the maps disagree.

### How to read the trees

Each entry point is a nested decision tree. A line reads

```
IF <predicate> → NODE | LEAF | ERROR  (Python name / MATLAB name)   [tag]
```

- **NODE** – a further decision point (expanded beneath it or in a numbered sub-section).
- **LEAF** – a routine that performs arithmetic and returns a number.
- **ERROR** – an exception is raised; the Python class and the MATLAB error identifier are named.
- **GUARD** – an admissibility or feasibility check that can redirect or raise.
- **OVERRIDE** – a user value that bypasses one or more decisions; each OVERRIDE line says what it bypasses and which guards still apply.
- **MEMO** – the self-inner-product cache key read or written at a leaf (Python tuple form / MATLAB string form `route|ts|extra`).
- Predicates are in neutral notation: `r_a` is the tuple order of attribute `a`, `K_a` its distinct-value count, `N` the event count, `A` the attribute count, `σ` the kernel width, `P` the period, `sop` the largest `σ/P` over relative-periodic attributes, `ts` the resolved truncation width (default 6; `inf` resolves to the accuracy floor ≈ 7.4338), `thr(ts)` the relative-periodic `σ/P` admissibility threshold (0.03 at the default width; capped at 0.05 by positive-definiteness for `ts ≤ 4`), `L(σ, P, ts, e)` the wrapped-kernel image count with exponent denominator `e`, and `wrap_a ∈ {'full-image', 'single-image'}` the per-attribute measure declaration (default `'full-image'`).
- Rules within a selector are numbered and fire in order; the first that returns ends the selector.

### Tags

- `[both]` – predicate and existence identical in both languages (fitted constants may differ).
- `[Py]` – exists only in Python, or the predicate differs in Python; cross-reference `A-n`/`B-n` follows.
- `[M]` – exists only in MATLAB, or the predicate differs in MATLAB; cross-reference follows.
- Untagged lines are `[both]`.

### Constants

Constants are named, not valued, in the trees; values are collected in §9. The only exceptions are thresholds that determine a branch (`8`, `10`, `200 000`, `0.20`, `4e6`, `1100`, `16e6`, `1e6`, `64`, `2.0`, `1.5`, `256 MiB`, `4 GiB`), which are quoted inline because they are the predicate. Fitted cost-law coefficients are per-language by design; the rule structure that consumes them is shared.

### Measures

- **A – minimum-image**: nearest-image reduction of every difference, or the pairwise-wrapped quadratic form on relative-periodic attributes.
- **C – all-image**: full image sum or its Fourier form (the wrapped Gaussian `θ`), the relative-periodic translation grid with image sum, or the spectral (mode-grid) Gram matrix.
- **non-periodic**: no wrapping; the plain Gaussian.

Which of A and C is computed on an absolute-periodic attribute is governed by `wrap_a` (and the `L = 0` short-circuit, which makes A and C coincide numerically). Which is computed on a relative-periodic attribute is governed by the route (Bulger and centres arms compute A; the Möbius grid, spectral, and nested `taugrid` routes compute C) and hence by the wrap/measure rules that choose the route.

---

## 1. Cosine similarity — `cos_sim_exp_tens` / `cosSimExpTens`

Python: `py:_tensor/cosine.py:159`. MATLAB: `m:cosSimExpTens.m:1`.

### 1.0 Parsing

- `normalize ∈ {'cosine', 'oneSidedDenom'}` (case-insensitive; `normalise` alias). ERROR on other values (`ValueError` / `cosSimExpTens:badNormalize`); Python additionally raises `TypeError` when both spellings are given. `need_xx = (normalize == 'cosine')`.
- `method` accepted set: `{'auto', 'bulger', 'centres', 'mobius', 'contract'}` `[both]` plus `'factored'` `[Py]` (A-13, D-10 – accepted, undocumented). No retired synonyms in either language (`'direct'` → error). Python validates inside the flat dispatcher (`cosine.py:1467`); MATLAB at parse (`:300-308`, `cosSimExpTens:badMethod`).
- `cancellation_threshold` / `cancellationThreshold`: parsed, threaded, never read in either language (D-2 for the stale MATLAB docstring).
- `kernel_precision`: `[Py]` honoured on the single-attribute helper route of the kernel core (§1.5a) and carried in the memo key; `[M]` captured and forwarded only into the batched-raw recursion, read by no cosine kernel (A-9).
- `spectrum`, `precision`, `dedup`: batched-raw only (MATLAB raises `*NotApplicable` otherwise; Python raises `TypeError` on the density and raw-MA forms but accepts `dedup` on the density path, where it is speed-only).
- ERROR `TypeError` / `cosSimExpTens:wrongArgCount` when the argument count is not 2, 9, or 10.

### 1.1 Input-form dispatch

```
ENTRY cos_sim_exp_tens(a, b, …)                                (py:cosine.py:335 / m:cosSimExpTens.m:389)
├─ IF 2 arguments and both are densities (or Python: a is a density, a density list, an empty list,
│  or an object array)                                            → NODE density path (§1.2)
│    ERROR when spectrum/precision are given (py TypeError / m *NotApplicable).
├─ IF 9/10 arguments and either p-argument is a list of attribute sets  → NODE raw multi-attribute (§1.1a)
├─ IF 9 arguments, numeric, and P1 or P2 is 2-D                  → NODE raw single-multiset batched (§1.1b)
├─ IF 9/10 arguments, numeric, both 1-D                          → raw single-multiset scalar:
│    IF sigma is a kernel covariance → whiten both operands, σ := 1 (spectrum with covariance → ERROR)
│    build_exp_tens ×2 → NODE flat MA dispatcher (§1.3)           (py:_cos_sim_raw_single_multiset_scalar / m::714-717, 741)
└─ ELSE → ERROR TypeError / cosSimExpTens:badPairTypes
```

#### 1.1a Raw multi-attribute form

```
NODE raw multi-attribute                                         (py:cosine.py:383 / m:cosSimExpTens.m:556)
├─ ERROR if spectrum/precision given, or (py) mode != 'auto'; ERROR if the second operand is not MA.
├─ IF both p-arguments are lists → ERROR TypeError / cosSimExpTens:listVsListNotSupported   [both]
├─ IF sigma carries a kernel covariance → whiten both operands (arithmetic input only; no routing change)
├─ IF neither is a list → build_exp_tens ×2 → NODE flat MA dispatcher (§1.3)
│                                            (py:_cos_sim_raw_ma_scalar / m::584-591)
└─ ELSE (exactly one list) → build the scalar side once; then
   ├─ [Py] NODE _try_sweep_reduction (cosine.py:1333)                                  (A-14)
   │    declines (→ per-entry loop) unless: the list is a TranslatedSweep; method == 'auto'
   │    (OVERRIDE: any forced method bypasses the reduction); offsets finite and non-empty;
   │    (scalar side first) or normalize == 'cosine'; sweep_eligibility(x, y, offsets) holds.
   │    → ENTRY 2 sweep_cos_sim_exp_tens(…, method='auto') (§2)
   ├─ [M] IF method ∈ {'auto','bulger'} → NODE r1_broadcast_fast (§1.2b)  (m::616-624)   (A-14)
   │    MATLAB has no TranslatedSweep; Python does not try the r = 1 fast path on this form.
   └─ per-entry loop: build_exp_tens + flat MA dispatcher per entry, the scalar side's memo threaded
      (py:_cos_sim_raw_ma_broadcast:1304-1329 / m::627-643)
```

#### 1.1b Raw single-multiset batched form (2-D `P1` or `P2`)

```
NODE batched raw                                                 (py:_cos_sim_raw_single_multiset_batch:3919 / m:localCosSimBatchedRaw:692)
├─ [Py] IF sigma is a kernel covariance → whiten, σ := 1, continue                     (A-8)
├─ [M]  IF isKernelCov(sigma) → ERROR mpt:aniso:batchedUnsupported                     (A-8)
├─ broadcast a single-row operand; ERROR ValueError / batchedRowMismatch when M1 != M2 and neither is 1
├─ ERROR NotImplementedError / batchedOrderedUnsupported when is_sym is given, any attribute is ordered, and r > 1
├─ rows with fewer than r valid values → NaN (skipped)
├─ canonical-key dedup of the individual sets (pair_canonical_key); build_exp_tens per unique set
│  (add_spectra when spectrum is given)
├─ IF no valid row → all-NaN
└─ ELSE → recursive ENTRY 1 on density lists, forwarding method, truncation_sigmas, kernel_precision
   (py mode='pairwise', cosine.py:4125 / m::2165-2178) → §1.2
Deprecated shims: py cos_sim_exp_tens_raw, batch_cos_sim_exp_tens (method='auto'); m batchCosSimExpTens.
```

### 1.2 Density path

```
NODE density path                                                (py:_cos_sim_density_path:1047 / m:cosSimExpTens.m:492-545)
├─ IF both scalar densities
│    py → _cos_sim_pair_core → §1.3
│    m  → seed memo from a.selfIP/b.selfIP; IF both single-multiset → prune, fall through to §1.3; ELSE localCosSimMA → §1.3
├─ IF list vs list
│    [Py] shape by mode ∈ {'auto','pairwise','cartesian'} (_resolve_list_list_mode); 'pairwise' with m != n → ValueError;
│         empty lists → cartesian-shaped empty (the mode='auto' mismatch error is swallowed there);
│         each pair keeps the caller's method, truncation_sigmas, kernel_precision            (A-17 cosmetic)
│    [M]  elementwise only; recursive cosSimExpTens(a{i}, b{i}, 'normalize', …, 'verbose', …) —
│         method, truncationSigmas, kernelPrecision DROPPED → every pair runs method='auto', default ts   (A-10, B-12)
├─ IF one scalar and one list (broadcast)
│    ├─ [Py] IF method ∈ {'auto','bulger'} → NODE r1_broadcast_fast (§1.2b) ; None → loop
│    │    OVERRIDE: method='centres'/'mobius'/'contract'/'factored' skips the fast path.
│    ├─ [M]  always attempt r1_broadcast_fast (method already dropped by the list form)  (A-10)
│    └─ per-entry loop:
│         [Py] IF dedup and every pair is single-multiset → _compute_pair_results_with_dedup
│              (canonical-key dedup; when verbose and n_unique >= 2, a calibration probe of up to 5 timed
│              _cos_sim_pair_core calls seeds the memos) ELSE _compute_pair_results_no_dedup;
│              each pair → _cos_sim_pair_core with the caller's method/ts/kp                (A-16 speed)
│         [M]  localScalarPairDispatch → localCosSimMA(dx, dy, 'auto', normalize, 1e-12, verbose, [])
│              — method, ts DROPPED                                                          (A-10)
└─ py _cos_sim_pair_core (cosine.py:988): ERROR TypeError on Windowed; ERROR ValueError when kernel
   covariances are incompatible (m: mpt:aniso:covMismatch inside localCosSimMA); ERROR TypeError on type
   mismatch → §1.3
```

#### 1.2b All-r = 1 broadcast fast path

```
NODE r1_broadcast_fast                                           (py:_r1_broadcast_fast:638 / m:localR1BroadcastFast:1801)
├─ declines (→ per-entry loop) unless: every density is a flat MaetDensity, no kernel covariance,
│  no nested attribute, A > 0, every r_a == 1, every inner unit == 0, and every entry matches the shared
│  operand in (A, r, σ, is_rel, is_per, period on periodic attributes, wrap)                  [both]
├─ LEAF one kernel pass over the concatenated comb-side columns (py cosine.py:759-787 / m::2547-2670):
│    per attribute: rel → skipped (vanishing form); abs-per full-image → wrapped_gaussian_1d(…, e = 4)
│    log θ (measure C); abs-per single-image → nearest image, −d²/(4σ²) (measure A); else −d²/(4σ²).
│    Threshold −ts²/2 − log(n_terms).
└─ MEMO ('bulger', ts, kp, None) / 'bulger|ts|': read for the shared operand and every entry; written
   when computed (py writes to both the original and the pruned object). <X,X> skipped when
   normalize != 'cosine' and X is the shared operand.
   [M] truncationSigmas not passed → default width on this path (A-10).
```

### 1.3 Flat MA dispatcher — `_cos_sim_exp_tens_ma` / `localCosSimMA`

Python `cosine.py:1406`; MATLAB `cosSimExpTens.m:821`.

```
NODE flat MA dispatcher
├─ prune both densities (memoised on the density); all memo traffic below is on the pruned objects
├─ IF N_x == 0 or N_y == 0 → return 0.0 (no arithmetic)                                   [both]
├─ GUARD structural: A, r, σ, is_rel, is_per equal; period equal on periodic attributes;
│  kernel covariances compatible → ERROR ValueError / cosSimExpTens:*Mismatch, mpt:aniso:covMismatch
├─ [Py] validate method; IF method == 'factored'                                           (A-13)
│    GUARD _ma_factored_ip_supported: any kernel covariance or any (is_rel and is_per) attribute → ERROR ValueError
│    → LEAF _cos_sim_exp_tens_ma_factored (§1.5d)  — bypasses everything below; no memo.
├─ pre-computation for the selector: sop; any_per / any_rel_nonper / any_rel_per; k_vec, k_vec_y; rel_vec;
│  nu_vec[a] = auto_ntau_default(P, σ) (rel-per) or max(64, ceil(max(span, 1)/σ · 10)) (rel-nonper) with
│  span = range_x + range_y + 2·rel_window_margin(GLOBAL ts)·σ                              (B-11)
├─ nested_any = any nested attribute on either density
├─ wrap_vec = dens_x.wrap only (dens_y.wrap never consulted)                               (B-10)
├─ skip_xx = (not need_xx) or self_ip_memoised(x); skip_yy = self_ip_memoised(y)
│  (self_ip_memoised: any memo key whose route ∈ {bulger, centres, mobius, contract, contract_ma})
├─ NODE chosen = flat selector (§1.4) with guard_forced_bulger = not nested_any, sym_vec = dens_x.is_sym
├─ IF ordered_any (any attribute with is_sym == false and r_a > 1 on either side) → chosen = 'bulger'
│    OVERRIDE (silent): discards the selector's answer AND an explicit method='mobius'/'centres'   [both] (Py-2/M-8)
├─ IF method == 'contract' and not nested_any → ERROR ValueError / cosSimExpTens:contractUnavailable
├─ IF nested_any
│    ├─ IF method ∈ {'auto','contract','mobius','centres'}
│    │    → NODE try_nested_contract (§1.8) with force = (method != 'auto'),
│    │      force_route = 'centres' iff method == 'centres'
│    │      [M] the per-call truncationSigmas is passed; [Py] it is not (A-1, B-1)
│    │      triple returned → finalise normalisation and return
│    │      None (only under method='auto') → chosen = 'bulger'
│    └─ ELSE (method == 'bulger') → chosen = 'bulger' (the plan is never consulted)
│    (The selector's output is discarded on nested densities; its only live effect there is the
│     mixed-wrap ERROR of rule 5.)
├─ dispatch on chosen
│    ├─ 'mobius' → LEAF Möbius arm (§1.5c)
│    │    POST-HOC GUARD (default post_hoc_guards == true): impossible value = non-finite IP, or
│    │    <X,X> < 0, or <Y,Y> < 0, or |<X,Y>| > 1.000001·sqrt(<X,X>·<Y,Y>) → RuntimeWarning /
│    │    mpt:cosSimExpTens:impossibleValue; purge every 'mobius' memo key on both densities;
│    │    FALLBACK → Bulger arm.  OVERRIDE post_hoc_guards=false skips the check.   [both]
│    ├─ 'centres' → LEAF centres arm (§1.5b)
│    └─ else → LEAF Bulger arm (§1.5a)
└─ finalise: 'cosine' → <X,Y>/sqrt(max(<X,X>·<Y,Y>, 0)); 'oneSidedDenom' → <X,Y>/<Y,Y>;
   denominator 0 → NaN; missing <X,X> under 'cosine' → ERROR.
```

### 1.4 Flat selector — `_select_ma_inner_product_method` / `selectMaInnerProductMethod`

Python `dispatch.py:513`; MATLAB `+internal/selectMaInnerProductMethod.m:1`. Rules fire in order.

```
Rule 1  OVERRIDE  IF method != 'auto' → return method unchanged            (py:609 / m:100)   [both]
        Bypasses rules 2–6, including the forced-Bulger feasibility guard and the wrap/measure rule.
        ('contract' and 'factored' never reach here on the flat path in practice — handled earlier.)
Rule 2  r_max = max(r_vec) (1 when A == 0). IF r_max <= 1 → 'bulger'      (py:613 / m:109)   [both]
Rule 3  IF r_max > 8 (ORBIT_R_MAX_SHIPPED)                                 (py:616 / m:112)   [both]
          IF guard_forced_bulger → GUARD guard_forced_bulger_feasible: nJ_x · nJ_y · 8 bytes > budget
             → ERROR SingleImageInfeasibleError (MemoryError subclass) / mpt:dispatch:singleImageInfeasible
             nJ side = N · Π_a K_a!/(K_a − r_a)! (÷ r_a! when ordered), saturating at 1e18.
             budget: [Py] fixed 4 GiB (_CENTRES_PROBE_MEM_BUDGET); [M] clamp(availableMemory/2, 1 GiB, 4 GiB)
             — constants-only.
          → 'bulger'
Rule 4  [Py] working-set guard (dispatch.py:642-666)                                          (A-12)
          IF A > 0 and not (any_rel_per and sop > thr(ts)):
             n_J_max = max(N_x·tuples_x, N_y·tuples_y), tuples = Π_a f_a·C(K_a, r_a), f_a = r_a! if sym else 1;
             IF n_J_max · 2·max(Σ r_a, 1) · 8 bytes > 256 MiB → 'mobius'
        [M] absent — the code proceeds from rule 3 to rule 5 (the MATLAB docstring's "K-vs-r precision
            guard" rule 4 does not exist in either language, D-3).
Rule 5  wrap/measure rule                                                  (py:675-695 / m:148-172)   [both]
          IF wrap_vec given and any_rel_per and rel_vec given:
             wants_single = any RELATIVE attribute with wrap == 'single-image';
             wants_full   = any RELATIVE attribute with wrap == 'full-image'   (scan is over all rel
             attributes, not only rel-per: B-9)
             IF wants_single and wants_full → ERROR ValueError "Mixed rel-per wrap" / mpt:mixedRelPerWrap
             IF sop > thr(ts): wants_single → 'bulger' (measure A); wants_full → 'mobius' (measure C)
Rule 6  cost race                                                          (py:707-741 / m:174-192)   [both]
          pw_size = predict_pairwise_kernel_size (inf if any K_a < r_a; perm_x·comb_y + perm_x·comb_x
          unless skip_xx + perm_y·comb_y unless skip_yy);
          pw_cost = rel_route_cost_ms('bulger', r_max, pw_size)  [law row min(max(r, 2), 4)];
          orbit_cost = predict_orbit_cost_ms(…, centres_ok = sop <= thr(ts), skip flags) (§1.4a);
          chosen = 'bulger' if pw_cost <= orbit_cost else 'mobius' (ties → bulger).
        No timing probe exists on this path in either language (D-1).
```

#### 1.4a `predict_orbit_cost_ms` / `predictOrbitCostMs`

`n_matrices = 1 + [not skip_xx] + [not skip_yy]`; `pairs = N_x·N_y`. Per attribute: IF `rel and r_a >= 2` → `per_pair = cost('grid', r_a, pairs·nu_a·max(K_a, K_y_a)·n_matrices/3)`; IF `centres_ok and K_a >= r_a and K_y_a >= r_a` → `min(per_pair, cost('centres', r_a, pairs·(m_x·m_y [+ m_x² unless skip_xx] [+ m_y² unless skip_yy])))` with `m = r_a!·C(K, r_a)`; floored by `ORBIT_REL_FLOOR_MS[r]` as `f + pm·n_matrices`. ELIF `r_a >= 2` (absolute) → `ORBIT_ABS_PER_ATTR_MS[r_a]·n_matrices/3`. `[both]`; coefficient values differ (constants-only). MATLAB's local `GRID_OP`/`CENTRES_OP`/`REL_BASE` are assigned and unused (B-17).

### 1.5 The three arms and their kernel cores

#### 1.5a Bulger arm — `_cos_sim_exp_tens_ma_pairwise` / `localCosSimMA` pairwise branch

`<X,Y> = ip_core_ma(U_perm_x, w_x, V_comb_y, w_y)` (permutation side × combination side). `<X,X>` computed iff `need_xx` and not memoised; `<Y,Y>` iff not memoised. MEMO `('bulger', ts, kp, None)` / `'bulger|ts|'`, written when computed. `[both]`

**Kernel core** — `_ip_core_ma` (`cosine.py:1744`) / `ipCoreMA` (`cosSimExpTens.m:1333`), shared by the Bulger and centres arms:

```
├─ [Py] IF A == 1 and inner unit == 0 and not (is_rel and is_per)                              (A-15)
│    → LEAF _ip_via_helper → gaussian_kernel_sum(V, w_v, U, σ√2, …, wrap, n_terms = |U|·|V|, kp) (§3.6)
│      (the single-attribute inner-product path; direct/culled/circular/exact chosen inside the helper;
│       the only cosine route that honours kernel_precision, A-9)
│    [M] absent: ipCoreMA goes straight to the next rule.
├─ IF every r_a == 1 and every inner unit == 0 → LEAF ip_r1_direct (py:1857 / m:localIpR1Direct:2386)
│    per attribute as in §1.2b; threshold −ts²/2 − log(n_j·n_k); chunked by kernel_chunk_bytes.
├─ IF (max_r + 2)·n_j·n_k·8 bytes <= kernel_chunk_bytes_resolved() → LEAF ip_full_ma (single chunk)
└─ ELSE chunk the comb side; same arithmetic.
   ma_log_kernel per attribute (py:2024 / m:1405-1496):
   ├─ IF not is_per and gram_is_accurate_enough (eps·s²/(4σ²) <= 0.1·truncation_floor(ts), s = max |value − first|)
   │    → LEAF gram_quadratic_form (block = inner unit if > 0, r_a if rel, else 0)
   ├─ ELIF inner unit > 0 → compute_Q_inner_blocks (block-diagonal; per-block pairwise wrap if periodic)
   ├─ ELIF is_per and not is_rel and wrap_a == 'full-image' and L(σ, P, ts, 4) > 0
   │    → wrapped_gaussian_1d(D, σ, P, ts, 4), log-summed over coordinates (measure C)
   └─ ELSE: abs-per (single-image, or full-image at L == 0) → nearest-image reduce;
            compute_Q: rel-per → pairwise-wrapped Σ_{i<j} wrap(d_i − d_j)²/r (measure A);
                       rel-nonper → Σd² − (Σd)²/r; abs → Σd².
   trunc_log_kernel_exp: threshold −ts²/2 − log(n_terms) when n_terms > 1.
```

#### 1.5b Centres arm — `_cos_sim_exp_tens_ma_centres` / `localCosSimMA` centres branch

Same kernel core with the **permutation side on both operands** (unrestricted enumeration, O(K^{2r})). MEMO `('centres', ts, kp, None)` / `'centres|ts|'`; `<X,X>` read/computed only if `need_xx`. Reached from: rule 1 with `method='centres'` on a flat density (then subject to the ordered override). `[both]`

#### 1.5c Möbius arm — `_cos_sim_exp_tens_ma_orbit` / `localCosSimMAOrbit`

Python `cosine.py:2255`; MATLAB `cosSimExpTens.m:1564`.

```
├─ choices[a] = (flat attribute) and NODE rel_attr_prefers_centres(P_x, P_y, σ, r_a, is_rel, is_per, P, ts,
│                user_forced_mobius = (method == 'mobius'))  (§1.6.1)
├─ MEMO ('mobius', ts, None, choices) / 'mobius|ts|<choice bits>': compute <X,X> iff need_xx and not
│  cached; <Y,Y> iff not cached; written after summing; purged by the post-hoc guard.
├─ per attribute a:
│    ├─ IF choices[a] → LEAF closed_form_attr_matrix_from(closed_form_attr_centres(x, a),
│    │                                                   closed_form_attr_centres(y, a), ts, wrap_a)  (§1.6.3)
│    │    [Py] passes the per-call ts; [M] omits it → the closed form resolves the global default
│    │    (A-6 latent, B-3: no numeric effect because this gate admits only relative attributes, whose
│    │     closed form never reads ts; the memo key nevertheless carries the per-call width).
│    └─ ELSE → NODE ma_per_attr_inner_matrix(P_x, W_x, P_y, W_y, σ, r_a, is_rel, is_per, P, ts, wrap_a) (§1.6.2)
└─ <X,Y> = Σ over event pairs of Π_a M_a
```

#### 1.5d `[Py]` Factored route — `_cos_sim_exp_tens_ma_factored` (A-13)

`_ma_ip_factored` ×3 (xy, xx, yy; `<X,X>` always computed; no memo). Per attribute `kind`: flat (`spec is None` and not rel-per) → `_ip_via_helper` → `gaussian_kernel_sum` with `wrap_a`; nested and cullable (`L >= 2` levels and no relative unit, or `rel_unit == 0` and not periodic) → `_ma_ip_factor_nested_culled` (leaf IPs via the helper; upper levels via `_combine_pair` with orbit eligibility on both sides, §1.8.5); otherwise → `_ma_ip_factor_dense` (`compute_Q_inner_blocks` when the inner unit > 0; abs-per single-image nearest-image `compute_Q`; abs-per full-image `wrapped_gaussian_1d(…, 4).prod`; else `compute_Q`). Early exit per event pair when the running product is 0. Bypasses the selector, the ordered rule, the nested plan, the post-hoc guard, and all memoisation; still subject to the structural guards and `_ma_factored_ip_supported`.

### 1.6 Möbius per-attribute matrix sub-tree

#### 1.6.1 Centres-vs-grid gate — `_ma_rel_attr_prefers_centres` / `maRelAttrPrefersCentres`

Python `_mobius_inner.py:1508`; MATLAB `+mobius/maRelAttrPrefersCentres.m:1`. `[both]`

```
├─ IF not is_rel or r_a < 2 → false (grid)
├─ blocked_by_measure = is_per and σ/P > thr(ts); empty_tuple_set = K_x < r_a or K_y < r_a
├─ forced = default rel_attr_route ('auto')
│    ├─ IF forced == 'auto' and user_forced_mobius → false  (OVERRIDE method='mobius' pins the grid)
│    ├─ IF forced == 'mobius' (py also 'grid', D-12 cosmetic) → false
│    └─ IF forced == 'centres': blocked_by_measure or empty_tuple_set → ERROR ValueError /
│         mpt:relAttrRouteBlocked; else → true
├─ IF blocked_by_measure or empty_tuple_set → false
└─ cost: c_wall = (CENTRES_NS_BASE + CENTRES_NS_LIN·(r−1) [+ CENTRES_NS_WRAP·(r−1)(r−2) if periodic])
               · (M_x·M_y + M_x² + M_y²), M = r!·C(K, r);
         g_wall = GRID_NS_FLOOR + GRID_NS_PER_OP[r]·n_u·K_x, n_u = auto_ntau_default(P, σ) (periodic;
               GLOBAL ts, B-11) or max(64, ceil(max(span, 1)/σ · 10)) (non-periodic);
         → true iff c_wall < g_wall
```

#### 1.6.2 Grid branch — `_ma_per_attr_inner_matrix` / `maPerAttrInnerMatrix`

Python `_mobius_inner.py:219`; MATLAB `+mobius/maPerAttrInnerMatrix.m:1`. `[both]`

```
├─ IF prune_zero_weight_events and N_x, N_y > 0: drop events whose weight column is all zero/NaN;
│  a side that empties → zero matrix; else recurse without pruning and scatter back
├─ IF r == 1 → LEAF r1 kernel sum (py::322-385 / m:localR1ZeroPad:233)
│    abs-per full-image (is_per and wrap == 'full-image') → wrapped_gaussian_1d(d, σ, P, ts, 4) (measure C);
│    else nearest image (if periodic) + trunc_kernel_exp (measure A); prefactor σ√π.
│    (wrap is consulted and the absolute kernel is used even when is_rel — shared quirk Py-14/M; the
│     Bulger/centres cores skip rel r = 1 instead; constant factor, cancels in the cosine.)
├─ IF is_rel (r >= 2) → NODE rel_inner_batched (§1.6.4); wrap is NOT passed (relative-periodic here is
│  always the all-image reading, measure C)
└─ ELSE (absolute, r >= 2)
   ├─ SPARSE GATE: not is_per and K_x·K_y >= 200 000 and nnz(K_0) <= 0.20·K_x·K_y (K_0 = sparse kernel
   │  of the first event pair, radius √2·ts·σ) → LEAF orbit sparse (per event pair:
   │  build_sparse_kernel_abs → inner_product_orbit_sparse, prefactor (σ√π)^r)
   └─ ELSE dense: chunk N_x by memory; kernel = wrapped_gaussian_1d(…, 4) when is_per and
      wrap == 'full-image' (C), else nearest image + trunc_kernel_exp (A);
      → LEAF inner_product_orbit_pw_batched(K_pairs, w_x, w_y, r, prefactor (σ√π)^r)
   (The "safe/unsafe" partition described in both docstrings is dead: every event is safe; B-16, D-3.)
```

#### 1.6.3 Tuple-centres closed form — `_closed_form_attr_centres` + `_closed_form_attr_matrix_from` / `closedFormAttrCentres` + `closedFormAttrMatrixFrom`

Python `_mobius_inner.py:1785, 1843`; MATLAB `+mobius/closedFormAttrCentres.m:52`, `closedFormAttrMatrixFrom.m:70`. `[both]`

```
├─ centres bundle: the attribute rebuilt as a one-attribute density (nested spec forwarded with
│  is_rel = false); py caches the bundle on the density (_nested_centres_cache, A-16 speed).
├─ COMB RESTRICTION (comb_side_restriction / localCombRestriction): declined when
│  not COMB_RESTRICTION_ENABLED, or flat r_a < 2, or nested orbit multiplicity < 2, or
│  n_k <= 0 or n_j != mult·n_k (ordered attribute or unexpected tiling);
│  declined at use when the Y bundle has no comb, a different multiplicity, or a mismatched tiling
│  → unrestricted perm × perm. When applied: X restricted to comb representatives, W_x ×= mult.
├─ per chunk of X tuples (chunk = max(1, floor(16e6/(n_jy·d)))):
│    ├─ IF inner block size >= 2 → compute_Q_inner_blocks (reduced) — no shipped caller reaches this (B-16)
│    ├─ ELIF is_per and not is_rel:
│    │    IF wrap_a == 'single-image' or L(σ, P, ts, 4) == 0 → nearest image + compute_Q (measure A;
│    │       L = 0 short-circuit gives the same number as C)
│    │    ELSE → wrapped_gaussian_1d(D, σ, P, ts, 4).prod over coordinates (measure C)
│    └─ ELSE → compute_Q(D, r_a, is_rel, is_per, P, reduced = is_rel) (rel-per → pairwise wrap, measure A)
│    No truncation of the exponential is applied in this leaf (either language).
└─ aggregation by incidence products GX · (overlap · GY)
   (py: local L_abs_per is computed and never read — dead variable, B-16.)
```

#### 1.6.4 Translation grid / spectral — `_rel_inner_batched` / `relInnerBatched`

Python `_mobius_inner.py:1071`; MATLAB `+mobius/relInnerBatched.m:1`. `[both]`

```
├─ SPECTRAL GATE: IF SPECTRAL_IP_ENABLED and 2 <= r <= 4 and not return_cancellation_ratio
│    → NODE spectral_rel_inner_matrix (py::894 / m:+mobius/spectralRelInnerMatrix.m)
│       L = P (periodic) or span_x + span_y + 2·(SPECTRAL_IP_MODE_SIGMAS + 2)·σ (non-periodic; decline
│       when not finite or <= 0); M = ceil(MODE_SIGMAS/√2 · L/(2πσ)) + 2; grid = (2M + 1)^(r−1)
│       ├─ GUARD grid > 4 000 000 (SPECTRAL_IP_MAX_POINTS) → decline (never bypassed)
│       ├─ COST GATE: not SPECTRAL_IP_FORCE and grid > 1100·K_x²·N_x·N_y → decline
│       │    OVERRIDE SPECTRAL_IP_FORCE (module flag / internal.spectralIpForce) bypasses this gate only.
│       └─ ELSE → LEAF spectral Gram matrix (set partitions with Möbius signs on a Hermitian-halved mode
│            grid; measure C; self-IP shortcut when the two sides are the same arrays)
│    decline → fall through to the grid
├─ grid nodes: spp = resolve_samples_per_sigma(None, r, ts); shared_w = all weight columns equal
│    periodic → N_u = auto_ntau_default(P, σ) (GLOBAL ts, B-11), u ∈ [0, P) uniform
│    non-periodic → per-pair mid-range centres; span = spread_x + spread_y + 2·rel_window_margin(ts)·σ;
│                   N_u = max(64, ceil(max(span, 1)/σ · spp))
├─ SPARSE GATE (periodic): is_per and r >= 2 and K_x·K_y >= 200 000 and 2√cutoff < P·(1 − 1e-8)
│  (cutoff = 2(ts·σ)²) and nnz(K_0) <= 0.20·K_x·K_y → LEAF rel_per_inner_sparse (per u-node sparse
│  kernel → inner_product_orbit_sparse)
├─ dense slabs (ORBIT_GRID_SLAB_ELEMS): per u-chunk; periodic → nearest image, then image sum over
│  2·rel_per_image_count(σ, P, ts) + 1 truncated Gaussians (measure C; a single image when the count is 0);
│  non-periodic → trunc_kernel_exp
│    ├─ IF shared_w → LEAF inner_product_orbit_grid(K_uc, W_x[:, 0], W_y[:, 0], r)
│    └─ ELSE → LEAF inner_product_orbit_pw_batched(K_uc, w_A, w_B, r, prefactor 1)
└─ scale (σ√π)^r / (σ·sqrt(2π/r))² · du (Riemann sum)
```

### 1.7 Wrapped Gaussian — `wrapped_gaussian_1d` / `internal.wrappedGaussian1d`

Python `_wrapped_kernel.py:103`; MATLAB `+internal/wrappedGaussian1d.m:1`. `[both]`

```
├─ tol = truncation_floor(ts) (= 1e-12 exactly at the accuracy-floor width, else exp(−ts²/2))
├─ L = ceil((σ/P)·sqrt(−e·ln tol) − ½) (0 when <= 0); M = ceil((P/(πσ))·sqrt(−ln tol / e)) (min 1)
├─ IF M < 2L + 1 (prefer_fourier) → LEAF Poisson (Fourier) series with M modes
├─ ELSE d_red = nearest image;
│    IF L == 0 → single Gaussian exp(−d_red²/(e·σ²))
│    ELSE → image sum over n = −L … L
At the default width: L(e = 4) > 0 iff σ/P > 0.0589; L(e = 2) > 0 iff σ/P > 0.0833; Fourier (e = 4) from σ/P ≈ 0.2.
```

### 1.8 Nested sub-tree — `_try_nested_contract` / `internal.nestedContract`

Python `cosine.py:2798`; MATLAB `+internal/nestedContract.m:1`.

```
NODE try_nested_contract(dens_x, dens_y, normalize, force, force_route, method_name)
│   [M] also receives the per-call truncationSigmas and resolves ts = accuracyFloor('resolve', …);
│   [Py] has no ts parameter: threshold, quadrature tolerance, kernel cutoffs, centres θ, memo key, and
│        enumeration admissibility all use the GLOBAL default                                   (A-1, B-1)
├─ decline(reason): IF force → ERROR ValueError "method=… is not available here" /
│  cosSimExpTens:contractUnavailable; ELSE → None (caller runs the Bulger enumeration)
├─ 1.8.1 DECLINE CONDITIONS: normalize ∉ {cosine, oneSidedDenom}; A differs; IF A != 1 → NODE nested-MA
│  (§1.9); no nested spec on either side; inner/intermediate [rel] unit != 0 on either side;
│  [r]/[sym] level vectors differ                                                                [both]
├─ skip_xx, skip_yy = nested_self_ip_skip_flags (same rule as the flat path; pricing only)
├─ 1.8.2 NODE nested_attr_plan(x, y, a = 0, force_route, skip flags) → (route, taus)
│    ├─ ADMISSIBLE ROUTES (measure rule; reads dens_x.wrap only, B-10):
│    │    not rel → ['contract']; rel non-per → ['centres', 'contract_relnonper'];
│    │    rel-per with P > 0 and σ/P > thr(ts) → ['centres'] if wrap_a == 'single-image' else ['taugrid'];
│    │    rel-per otherwise → ['centres', 'taugrid']
│    ├─ ROUTE (nested_attr_route / nestedAttrRoute):
│    │    [M] localCheckForcedMode: a forced name with no carrier for (rel, per) → ERROR cosSimExpTens:centresUnavailable
│    │    IF not rel → 'centres' iff force_route == 'centres' else 'contract' (no cost model for absolute attributes)
│    │    IF periodic and admissible == ['taugrid'] and force_route == 'centres' → ERROR ValueError /
│    │       centresUnavailable ("cannot be honoured"); ELSE IF admissible == ['taugrid'] → 'taugrid'
│    │    IF force_route == 'centres' → 'centres'  (OVERRIDE of the cost race only)
│    │    IF one admissible route → it
│    │    ELSE → price_nested_attr(x, y, a, admissible, skip flags):
│    │       cost(route) = max(exp(a)·max(term, 1)^b, f + pm·n_matrices), law row keyed by
│    │       nested_cost_key(total order) ∈ {2, 3, 4, 6};
│    │       MEMORY GUARD: route == 'centres' and >1 admissible and working set
│    │       (max(nJ_x, nJ_y)·2·dim·8 bytes) > 256 MiB → cost = inf;
│    │       best = argmin (ties → first listed)
│    └─ QUADRATURE (inline in py _nested_attr_plan / m makeQuadrature):
│         'centres', 'contract' → none; 'taugrid' → τ ∈ [0, P) uniform, auto_ntau_default(P, σ) nodes
│         (GLOBAL ts in both, B-11); 'contract_relnonper' → auto_taus_line: step σ/4,
│         pad (6 + 0.5·max(0, −log10 tol))σ, n = max(64, ceil(2·hi/step))
├─ 1.8.3 PLAN VS ENUMERATION: IF not force and nested_prefers_enumeration({a: route}) → None (Bulger)
│    select_nested_method: plan_ms = Σ_a price + flat_companion_cost; enum_ms = cost('bulger', r_max,
│    joint tuple-pair size with skip flags) if enumeration_admissible else inf;
│    enumeration_admissible = no rel-per attribute with wrap != 'single-image' and σ/P > thr(ts);
│    chosen = 'bulger' iff enum_ms · 2.0 (NESTED_ENUM_SAFETY) < plan_ms. Costs recorded for diagnostics.
├─ MEMO ('contract', ts, None, (route, (ntau, τ_0, τ_end) or None)) / 'contract|ts|route[|ntau|t1|tend]'
│  for <X,X> and <Y,Y>; <X,X> is ALWAYS computed when absent, even under oneSidedDenom (B-14)
│  [Py] ts in the key is the global default (A-1)
└─ 1.8.4 LEAF nested_attr_matrix(x, y, a, route, taus) ×3
     ├─ 'centres' → closed_form_attr_centres ×2 + closed_form_attr_matrix_from(cx, cy, ts, wrap_a) (§1.6.3;
     │    wreath-product comb restriction)  [M passes ts; Py passes the global default]
     └─ else → build_recipe per level (use_orbit = sym and 2 <= r <= 8) → NODE contraction (§1.8.5) with
          'taugrid' → periodic τ, reduce 'mean'; 'contract_relnonper' → is_per = false, reduce 'sum';
          'contract' → no τ, wrap_a = dens_x.wrap[a]
```

#### 1.8.5 Contraction leaves — `nested_attr_matrix` → `_contract` → `_combine_pair` / `nestedAttrInnerMatrix` → `contractNode` → `combinePair`

Python `_nested_contraction.py:930, 549, 291`; MATLAB `+internal/nestedContract.m:1303, 1711`. `[both]`

```
├─ orbit_guard_scope(ts): floor = truncation_floor(ts); enabled = post_hoc_guards
├─ SHARED-TEMPLATE closed form (rel-nonper only): IF τ given, non-periodic, reduce 'sum', both recipes
│  ordered at the outer level, r == number of children, every event a shared leaf template, equal cell
│  lengths → LEAF shared_template_matrix / ipRelNonperFactored (template cross-correlation); else generic
├─ kernel per pair:
│    no τ: is_per and wrap_a != 'single-image' → wrapped_gaussian_1d(d, σ, P, ts, 4) (C);
│          else (periodic → nearest image) exp(−d²/(4σ²)) (A / non-periodic); truncate (< exp(−ts²/2) → 0)
│    τ given: non-periodic → per-pair τ window (tau_window: none when T < 3, non-uniform grid, no
│          truncation, or W >= T); periodic and wrap_a != 'single-image' → wrapped_gaussian_1d (C);
│          else nearest image + exp; reduce 'mean' (÷ full T) or 'sum'
├─ contraction: r == 1 leaf → einsum; uniform siblings → one batched combine_pair; ragged → per pair
└─ combine_pair(M, r, sym, use_orbit):
     ├─ use_orbit = x.use_orbit and y.use_orbit, re-decided by orbit_cost_model(r, max(g_x, g_y), Q):
     │    false if r > K or r < 2; K_eff = max(r, K − 1) when r <= MARGIN_R_MAX (3);
     │    log_ratio = orbit_cost_intercept + 1.0708·ln|Ω_r| + 1.4033·ln K − 0.3608·ln B − 0.8188·ln(T_x·T_y)
     │                − 0.2261·ln r; orbit iff log_ratio < 0
     ├─ IF use_orbit → LEAF combine_orbit → inner_product_orbit_grid(…, term masses);
     │    bound = eps·max(term-mass sum)/r!
     │    ACCURACY GUARD (post_hoc_guards): IF bound > floor:
     │       IF enum_work = Q·C(g_x, r)·r!·C(g_y, r) <= 16e6 → RuntimeWarning / mpt:nestedOrbitCost,
     │          FALLBACK → combine_chunked (enumeration)
     │       ELSE → RuntimeWarning / mpt:nestedOrbitAccuracy, keep the orbit value
     │    OVERRIDE post_hoc_guards = false → keep the orbit value unconditionally
     └─ ELSE → LEAF combine_chunked (enumerated perm × comb product, chunks of 16e6 elements)
```

### 1.9 Nested multi-attribute — `_try_nested_contract_ma` / `nestedContractMA`

Python `cosine.py:2911`; MATLAB `nestedContract.m:997`.

```
├─ pass 1 per attribute:
│    nested → declines (force → ERROR) when nested on one side only, inner [rel] unit != 0, or [r]/[sym]
│             differ; else nested_attr_plan → ('nested', a, route, taus)                        [both]
│    ordered flat (not nested, is_sym == false, r_a > 1) → ('ordered', a)
│    else → ('flat', a)
├─ PLAN VS ENUMERATION as §1.8.3 with {a: route for nested attributes}; None → Bulger
├─ MEMO ('contract_ma', ts, None, ((kind, a, route, tau_sig), …)) / 'contract_ma|ts|kind:a:sig,…'
│  [Py] ts = global default (A-1)
└─ pass 2 per attribute:
     'flat'    → ma_per_attr_inner_matrix (§1.6.2)
                 [M] with 'truncationSigmas', ts, 'wrap', wrap_a
                 [Py] WITHOUT wrap and WITHOUT ts → 'full-image' and the global default      (A-2, B-2 — fixed in this round)
     'nested'  → nested_attr_matrix (§1.8.4)
     'ordered' → closed_form_attr_matrix_from(cx, cy, ts, wrap_a) (§1.6.3)  [Py ts = global default, A-1]
```

### 1.10 Overrides on ENTRY 1 (narrative; table in §8)

- `method='bulger'`: selector rule 1 (rules 2–6 skipped, including the feasibility guard and the wrap rule); on nested densities the plan is never consulted → joint-tuple enumeration. Still subject to the structural guards, the empty-density short-circuit, and the ordered rule (a no-op).
- `method='mobius'`: rule 1 → Möbius arm (wrap rule skipped: a rel-per `single-image` attribute above threshold runs the all-image grid anyway); inside the arm `user_forced_mobius` pins the grid unless `rel_attr_route` is set explicitly. Still subject to: the ordered override (silent → Bulger), the post-hoc impossible-value guard (→ Bulger), and on nested densities the plan with `force = true` (raises on uncovered cases).
- `method='centres'`: flat → unrestricted centres arm; nested → plan with `force_route='centres'` (raises above threshold on rel-per full-image). Still subject to the ordered override.
- `method='contract'`: nested only (else ERROR); same as `'mobius'` on nested densities.
- `method='factored'` `[Py]`: separate route, no guards beyond the structural ones and its own support predicate.
- `normalize='oneSidedDenom'`: `<X,X>` skipped on the Bulger, centres, and Möbius arms; **not** skipped on the nested routes (B-14), the factored route, or the sweep.
- `rel_attr_route` default (`'centres'`/`'mobius'`): pins the per-attribute route inside the Möbius arm; raises when inadmissible.
- `post_hoc_guards=false`: skips the impossible-value fallback and the nested orbit accuracy guard.
- Module/test levers: `SPECTRAL_IP_ENABLED`, `SPECTRAL_IP_FORCE`, `COMB_RESTRICTION_ENABLED`, `_FOURIER_ENABLED` (py) / `internal.spectralIpEnabled`, `spectralIpForce`, `combRestrictionEnabled` (m).
- Per-call `truncation_sigmas`: honoured on the flat Bulger/centres/Möbius-grid paths in both languages; `[Py]` ignored by the entire nested path (A-1); `[M]` ignored by the flat Möbius centres branch (A-6, latent); ignored in both by `auto_ntau_default`, the selector's `nu_vec` margin, and the centres-vs-grid grid estimate (B-11), and dropped by MATLAB's list forms (A-10).

---

## 2. Sweep — `sweep_cos_sim_exp_tens` / `sweepCosSimExpTens`

Python `_tensor/sweep.py:833`; MATLAB `sweepCosSimExpTens.m:1`. `[both]` throughout unless tagged.

```
├─ [parse] method ∈ {'auto', 'mixture', 'orbit'}; normalize ∈ {cosine, oneSidedDenom}; offsets (A × M),
│  finite; ERROR ValueError / sweepCosSimExpTens:* otherwise
├─ prune both; ts resolved
├─ mixture_ok = sweep_eligibility: false when not both flat MaetDensity; bad shape; non-finite offsets;
│  an UNSWEPT rel-per attribute with σ/P > thr(ts) and wrap != 'single-image'; a swept attribute that is
│  relative, periodic, or has an inner unit > 0; a swept attribute with a kernel covariance; n_j == 0 or n_k == 0
│  (m: localCheckEligible errors relativePeriodic / periodicAttribute / sweptNested / sweptRelative / anisotropicKernel)
├─ orbit_ok = orbit_sweep_supported: false when any attribute is ordered; any inner unit > 0; a swept
│  attribute is relative; a rel-per attribute with σ/P > thr(ts) and wrap != 'full-image'; any kernel covariance
├─ IF method == 'auto' → NODE choose_sweep_route:
│    not orbit_ok → mixture; not mixture_ok → orbit; any swept attribute with r_a < 2 → mixture;
│    orbit_work == 0 (nothing swept) → mixture;
│    mixture_bytes = n_pairs·(n_swept + 2)·8 > kernel_chunk_bytes_resolved() → orbit;
│    n_pairs < 1e6 (ORBIT_MIN_PAIRS) → mixture;
│    orbit_total < 64 (ORBIT_WORK_RATIO)·n_pairs → orbit ELSE mixture
│    (orbit_total = M·N_x·N_y·Σ_swept n_orb(r)·K_x·K_y·r)
│  ELSE chosen = method (OVERRIDE of the chooser only)
├─ ERROR ValueError / sweepCosSimExpTens:orbitUnsupported (or the eligibility error rethrown) when the
│  chosen route is not supported — never bypassed
├─ IF chosen == 'orbit' → LEAF orbit sweep:
│    untranslated attributes → ma_per_attr_inner_matrix(…, ts, wrap_a) (§1.6.2) broadcast over M;
│    swept attributes → orbit_attr_matrix_sweep: kernel wrapped_gaussian_1d(…, 4) if is_per and
│    wrap == 'full-image' else nearest image + trunc_kernel_exp; r >= 2 → inner_product_orbit_pw_batched;
│    r == 1 → weighted kernel sum. Denominators via ma_per_attr_inner_matrix — NO memo (either language);
│    <X,X> only when normalize == 'cosine'. [Py] forwards the raw (unresolved) ts; an explicit inf reaches
│    the kernels unresolved (A-17 cosmetic).
└─ ELSE → LEAF mixture:
     build_mixture: splittable attributes (non-periodic, no covariance) → mixture axes (swept) or folded
     placement term; fixed attributes (periodic / anisotropic) → ma_log_kernel (§1.5a) with their wrap;
     threshold −ts²/2 − log(n_j·n_k); components below it dropped;
     evaluate_mixture: S == 0 → constant; P <= mean_slice + 512 → evaluate_dense; else culled per-offset loop.
     Denominators: [Py] MEMO ('sweep', ts, kp) read/written (this key is deliberately excluded from the
     pricing flag); [M] recomputed each call, no memo (A-16 speed; D-8 stale MATLAB comment).
```

---

## 3. Point evaluation — `eval_exp_tens` / `evalExpTens`

Python `_tensor/eval.py:62`; MATLAB `evalExpTens.m:1`.

### 3.1 Parsing and input-form dispatch

```
├─ [parse] method ∈ {'auto', 'centres', 'mobius'} (py validated inside the selector; m at parse,
│  evalExpTens:badMethod); normalize ∈ {'none', 'gaussian', 'pdf'} — [M] validated; [Py] unvalidated,
│  any other string behaves like 'gaussian' (A-17 cosmetic)
├─ ts resolved at entry
├─ density scalar → NODE scalar (§3.2)
├─ density list → per entry:
│    [Py] _eval_exp_tens_density_list: dedup over single-multiset entries by canonical key; each entry keeps
│         method, ts, kp
│    [M]  recursive evalExpTens(dens{i}, X_i, normalize, 'verbose') — method, ts, kp DROPPED   (A-11, B-12)
├─ raw MA (8/9/10 args) → build_exp_tens (whitened when the σ vector carries a covariance) → scalar
├─ raw single-multiset 1-D → (add_spectra when spectrum) build → scalar; covariance: py NotImplementedError when 2-D
├─ raw single-multiset 2-D (batched) → per row: ordered and r > 1 → ERROR NotImplementedError / batchedOrderedUnsupported;
│    rows with < r valid → NaN; canonical-key dedup; each unique density → scalar
│    [Py] forwards method; [M] forwards ts, kp but DROPS method                                  (A-11)
└─ ELSE → ERROR TypeError
```

### 3.2 Scalar density — `_eval_exp_tens_scalar` / `evalExpTens` struct branch

```
├─ IF kernel covariance → whiten the query; after evaluation × exp(−½·logdet) when normalize != 'none'
├─ [M] IF single-multiset → singleMultisetView → single-multiset dispatch (§3.3 single-multiset); ELSE localMaSkinnyDispatch (§3.3 MA)
└─ [Py] → _eval_exp_tens_ma (§3.3)
```

### 3.3 Route selection and the centres branch shapes

```
NODE eval_ma (py:_eval_exp_tens_ma:825 / m:localMaSkinnyDispatch:1241, localEvalMA:817, single-multiset dispatch :508-599)
├─ [M] skinny: kernel covariance → not handled → joint (localEvalMA); n_q == 0 → zeros
├─ chosen = NODE select_ma_eval (§3.4) (m single-multiset corner: method='centres' → centres; method='mobius' →
│  ERROR mpt:evalExpTens:orderedMobius when the density has an ordered attribute, else mobius; else selector)
├─ IF chosen == 'mobius' → LEAF eval_ma_orbit (§3.5): per event, per attribute rel → eval_orbit_rel,
│  abs → eval_orbit_abs (wrap forwarded for absolute attributes only); product over attributes, sum over events
│    [Py] kernel_precision NOT forwarded (A-9, B-13); [M] forwarded
│    [M] single-multiset corner POST-HOC GUARD: not all(isfinite(vals)) → warning evalExpTens:mobiusNonFiniteFallback,
│        chosen = 'centres' (single-multiset only; the MA path has no such guard in either language)   (A-7, B-8)
│    [Py] no non-finite guard anywhere (docstring claims one, D-5)
└─ ELSE (centres) — shape rules:
     ├─ IF single multiset (A == 1, N == 1, not nested) → prune zero-weight tuples; no tuples → zeros;
     │    → LEAF single-multiset centres: py _eval_core(…, ts, wrap = dens.wrap[0]) (§3.6a);
     │      m gaussianKernelSum(Centres, w, X, σ, …, ts, kp) (§3.6)  [Py drops kp here, A-9]
     ├─ ELSE NODE ma_eval_factored (py:1077 / m:localMaEvalFactored:1408): declines (None / []) when any
     │    r_a < 2, or a kernel covariance, or an attribute has fewer ever-valid values than r_a;
     │    else per event × attribute: inner unit > 0 → dense block-diagonal exp(−Q) (no truncation);
     │    flat → LEAF gaussian_kernel_sum / _eval_core(c, w_tuple, X_a, σ, …, ts) called WITHOUT wrap →
     │    'full-image' regardless of the declaration (shared defect B-5); product over attributes, sum over events
     └─ ELSE joint materialisation → LEAF ma_eval_full / maetEvalFull: prune w == 0; query validation;
          n_q == 0 → zeros; value tables for abs-per full-image attributes when tuple_values_repeat
          (n_q >= 100, entries >= 256, distinct·4 <= entries); chunk on (2·max_dim + 2)·n_j·8·n_q bytes vs
          kernel_chunk_bytes_resolved(); per attribute: abs-per full-image → tabulated or dense
          wrapped_gaussian_1d(…, 2) product (C); inner unit > 0 → block-diagonal reduced Q; abs-per
          single-image → nearest image + compute_Q (A); rel-per → pairwise-wrapped Q/r (A); rel-nonper
          → Σd² − (Σd)²/r; abs → Σd²; truncation mask q_total <= ts²/2; × abs-per factor; w · e.
          [Py] kernel_precision = 'single' → float32 accumulator (the only Python eval route honouring kp).
     [M] localEvalMA retries localMaEvalFactored — always [] there (dead in practice, D-list).
normalise: 'none' → raw; else × Π_a 1/gaussian_mass_const(σ_a, d_a, det(r_a, inner, rel)); 'pdf' → ÷ Σw.
```

### 3.4 Eval selector — `_select_ma_eval` / `selectMaEval`

Python `dispatch.py:2004`; MATLAB `+internal/selectMaEval.m:1`. Rules in order. `[both]` unless tagged.

```
Rule 1  OVERRIDE method == 'centres' → 'centres' (skips every rule below)
Rule 2  OVERRIDE method == 'mobius':
          [Py] GUARD reject_ordered_for_mobius (any flat attribute with is_sym == false and r > 1) → ERROR
               ValueError — for every density shape
          [M]  the guard lives in the single-multiset caller only (evalExpTens.m:517-525); the MA path (skinny and joint)
               honours 'mobius' with no ordered check → evalMaOrbit silently symmetrises   (A-4, B-4 — fixed in this round)
          else → 'mobius' (skips the measure rule and the cost model; NOT subject to the r > 10 rule)
Rule 3  method ∉ {'auto', 'centres', 'mobius'} → ERROR (py here; m at parse)
Rule 4  any ordered attribute → 'centres'
Rule 5  any nested attribute → 'centres'
Rule 6  all r_a <= 1 → 'centres'
Rule 7  any r_a > 10 (ORBIT_R_MAX_FEASIBLE) → GUARD estimate_ma_joint_working_set_bytes(r, K, is_rel, is_sym)
          > budget → ERROR SingleImageInfeasibleError / mpt:dispatch:singleImageInfeasible; else 'centres'
          (budget: py 4 GiB fixed; m dispatchMemBudget() — constants-only). No "K − r precision guard" exists
          despite both docstrings (D-3).
Rule 8  measure rule (does not return early): first rel-per attribute with P > 0 and σ/P > thr(ts):
          wrap_a == 'single-image' → forces 'centres' (A); else forces 'mobius' (C)
Rule 9  cost model: centres_ms, mobius_ms = ma_eval_costs_ms(dens, n_q) (§3.4a);
          IF measure forces centres → 'centres'; ELIF forces mobius → 'mobius';
          ELSE safety = 1.5 (MA_MOBIUS_SAFETY) if joint working set > 256 MiB else 1.0;
               'mobius' iff mobius_ms < centres_ms·safety, else 'centres'
```

#### 3.4a Cost model — `_ma_eval_costs_ms` / `maEvalCostsMs`

Centres priced as factored per-attribute sums iff `A > 1 and all r_a >= 2 and no kernel covariance`, else as joint materialisation with a culled-fraction product. Möbius priced per attribute with `r_a >= 2`: Bell-number set-up plus per-query ops `(2^r − 1)·r·K_a` (× u-grid nodes for relative attributes); the spectral price is used when `2 <= r <= 4`, `n_q >= (16, 32, 64)[r]`, `K_a >= (2, 8, 16)[r]`, and `(2M + 1)^(r−1) <= 4e6` — a `MAX_POINTS` decline that the evaluator itself never performs (D-6). `[both]`, constants-only; MATLAB's `K_a = kVec(a)` is now indexed per attribute (fixed in this round).

### 3.5 Möbius evaluator strategies — `_mobius.eval_orbit_abs` / `evalOrbitAbs`, `eval_orbit_rel` / `evalOrbitRel`

`[both]`

```
eval_orbit_abs
├─ per_helper_global = is_per and P > 2·√2·ts·σ
├─ per distinct partition block: non-periodic → use_reduction = true;
│  periodic → use_reduction = per_helper_global and span_ok (block circular offsets span < P/2)
├─ use_reduction → LEAF gaussian_kernel_sum(p, w^m, mean_x, σ/√m, ts, wrap, [is_per, P]) × exp(−var/(2σ²))
└─ else → LEAF direct broadcast: is_per and wrap == 'full-image' → wrapped_gaussian_1d(…, 2).prod (C);
   else nearest image + exp (A; no truncation). Combined by the Möbius partition sum.

eval_orbit_rel
├─ r < 2 → constant Σw (no arithmetic)
├─ u-grid: periodic → N_u = max(64, ceil(P/σ · spp)); non-periodic → window [p_min − max(0, x_max) − 8σ,
│  p_max − min(0, x_min) + 8σ], N_u = max(64, ceil(max(width, 1)/σ · spp)); spp = resolve_samples_per_sigma(None, r, ts)
├─ SPECTRAL GATE: FOURIER_ENABLED and 2 <= r <= 4 and factored unset and not cancellation ratio and
│  n_q >= (16, 32, 64)[r] and K >= (2, 8, 16)[r] and (not is_per or (every query span < P/2 and
│  P > 2·√2·ts·σ)) → LEAF eval_orbit_rel_fourier (C on the circle). NO MAX_POINTS guard here (D-6).
├─ periodic: factored_valid = P > 2·√2·ts·σ and max spread < P/2; factored forced on and invalid → ERROR;
│  auto → use_factored = valid and factored_worthwhile(K, r, n_q, N_u, n_fine)
│  (K·n_fine + 10·B_r·r·N_u·n_q < B_r·r·K·N_u·n_q); non-periodic: auto → factored_worthwhile
├─ use_factored → LEAF tabulated S_m via gaussian_kernel_sum + Lagrange-6 read-back
└─ ELSE → LEAF direct: eval_orbit_abs at every u-node with the default wrap = 'full-image'; integrate
   (periodic → rectangle; non-periodic → trapezoid).
(The 'factored' option is not reachable from the entry point in either language; py docstring's
 "cross-correlation strategy at r = 2" does not exist, D-9.)
```

### 3.6 Kernel-sum chunks — `gaussian_kernel_sum` / `internal.gaussianKernelSum`

Python `_kernel.py:27`; MATLAB `+internal/gaussianKernelSum.m:1`. `[both]`

```
├─ ts resolved; IF n_terms > 1 → k := sqrt(k² + 2·ln n_terms) (the inner-product widening; equivalent to
│  the −k²/2 − log(n_terms) log-threshold of the MATLAB core, A-15)
├─ ERROR on bad kernel_precision, ts <= 0, shape mismatch, is_rel and r < 2, is_per and P <= 0
├─ use_truncation = isfinite(k) and k > 0 and n_j > 0 and n_q > 0 (always true after resolution for
│  non-empty inputs)
├─ IF use_truncation and not is_per:
│    dim == 1 and not is_rel → LEAF truncated 1-D vectorised/sorted-window sum (dense fallback when the
│    window covers everything); else → LEAF truncated bucket-grid sum (relative → whitened by I − 11ᵀ/r)
├─ ELIF use_truncation and is_per and dim == 1 and not is_rel:
│    IF 2·k·σ < P → LEAF truncated 1-D circular sum (nearest copy only — coincides exactly with the
│    L(σ, P, k, 2) == 0 regime, so measures A and C agree; shared fragile coupling M-17); ELSE → exact
└─ ELSE → exact: chunk on (2·dim + 2)·n_j·n_q·itemsize vs kernel_chunk_bytes_resolved() → LEAF eval_chunk:
     IF is_per and not is_rel and wrap != 'single-image' and (L(σ, P, k, 2) > 0 [or m: tabulated]) →
        per-coordinate wrapped_gaussian_1d(…, 2) product (C; m tabulates inside the chunk, py tabulates in
        _eval_core / _ma_eval_full — same numbers)
     ELSE abs-per → nearest image; compute_Q(reduced = is_rel); exp(−Q/(2σ²)) — no cutoff in the chunk.
```

#### 3.6a `[Py]` `_eval_core` (eval.py:1543) — the single-multiset centres leaf and factored-MA factor

`ts finite and not is_per` → `_truncated_kernel_sum_culled` (bucket cull); else value table when `is_per and not is_rel and wrap == 'full-image'`, then `_eval_full` (single chunk if it fits): abs-per full-image → tabulated or dense `wrapped_gaussian_1d(…, 2)` product with floor `E > truncation_floor(ts)`; abs-per single-image → nearest image + `compute_Q`; else `compute_Q(reduced = is_rel)`; mask `q_total <= ts²/2`. MATLAB reaches the same numbers through `gaussianKernelSum` directly (row 82 of the parity table).

---

## 4. Entropy — `entropy_exp_tens` / `entropyExpTens`

Python `entropy.py:352`; MATLAB `entropyExpTens.m:1`. `[both]` unless tagged.

```
├─ [parse] method ∈ {'differential', 'shannon' (default), 'normalized' ('normalised' alias), 'renyi2'};
│  legacy normalize= → ERROR TypeError / entropyExpTens:normalizeRemoved
├─ 'shannon' / 'normalized' → NODE shannon dispatch (normalize flag = method == 'normalized')
│    input forms: density scalar (explicit n_points_per_dim required → ERROR otherwise; bounds and
│    grid_limit errors); density list (py: dedup over single-multiset entries by canonical key;
│    [Py] ts/kp NOT forwarded to the scalar call (A-9); [M] every name-value pair except normalize/isSym
│    forwarded, including ts and kp); raw MA (py: no ts/kp forwarded, A-9); raw single-multiset 1-D (add_spectra when
│    spectrum; ts, kp forwarded); raw single-multiset 2-D batched
│    core (py _entropy_exp_tens_ma:1657 / m localEntropySingleMultiset:461, localEntropyMA:569):
│    ├─ dim == 0 → 0.0; n_points^dim > grid_limit → ERROR
│    ├─ IF not windowed and no relative attribute → LEAF cell masses (erf per axis; periodic axis =
│    │    MINIMUM-IMAGE erf, wrap and ts ignored — shared B-6; measure A regardless of wrap);
│    │    py streams D <= 2 in DIFF_CELL_BLOCK blocks
│    └─ ELSE → ENTRY 3 eval_exp_tens(dens, X_grid, method = 'auto', ts, kp) on an nd-grid (§3)
│    H = −Σ q log_b q; normalized → ÷ log_b(N_cells)
├─ 'differential' → NODE differential adaptive (py:896 / m:localDifferentialAdaptive:1568):
│    ERROR NotImplementedError / *NotSupported on lists, object arrays, 2-D raw, windowed; precision/dedup
│    given → ERROR; ts resolved; tol = max(exp(−ts²/2), 1e-12); grid doubling up to 10 levels
│    (N^dim > effective grid limit → ERROR); each level = the Shannon core above; h = H + Σ log δ;
│    converged when |Δh| < tol, or Richardson |ΔR| < tol, or Richardson differences stop shrinking
│    (dR >= 0.95·dR_prev); kernel covariance → + ½·logdet/log(base)
└─ 'renyi2' → NODE renyi2 dispatch: ERROR on lists / 2-D raw / windowed; ERROR on precision/dedup (py);
     σ == 0 → ERROR
     ├─ [M] IF single multiset → NODE renyi2_single_multiset (entropyExpTens.m:1982):
     │    ordered (is_sym false, r > 1) → LEAF renyi2_per_attr_numerical;
     │    r == 1 and is_rel → H = 0 by convention                                           (A-5, B-7)
     │    r == 1 abs → LEAF direct pairwise (full-image → wrapped_gaussian_1d(…, default ts, 4); else
     │                 nearest image + exp; no truncation) + total_mass_abs;
     │    r >= 2 rel → LEAF orbitInnerRelSingleMultiset (→ relInnerBatched §1.6.4, default ts) + total_mass_rel;
     │    r >= 2 abs → LEAF orbitInnerAbsSingleMultiset (→ innerProductOrbit, wrap forwarded, default ts) + total_mass_abs
     │    [Py] no single-multiset special case: single-multiset densities take the MA loop below (finite value at r = 1 rel)
     └─ NODE renyi2_ma (py:_renyi2_exp_tens_ma:1539 / m::2090): A == 0 → 0; N == 0 → NaN; per attribute:
          nested or ordered flat → LEAF renyi2_per_attr_numerical (rebuild the attribute; overlap via
             inner-block Q, or abs-per: wrap == 'single-image' → nearest image compute_Q (A) else
             wrapped_gaussian_1d(D, σ, P, default ts, 4).prod (C), else compute_Q; no truncation);
          else → LEAF ma_per_attr_inner_matrix(P_a, W_a, P_a, W_a, σ, r_a, is_rel, is_per, P) (§1.6.2)
             [M] with 'wrap', wrap_a (default ts); [Py] WITHOUT wrap → 'full-image' regardless   (A-3, B-2 — fixed in this round)
          + Z_a[n] = total_mass_rel/abs closed form
          GUARD finalise: non-finite, ip_xx <= 0, or Z <= 0 → NaN; else −log_b(ip_xx/Z²); covariance → + ½·logdet/log(base)
Delegation summary: renyi2 → the Möbius per-attribute matrix (never the flat selector, never Bulger,
never the tuple-centres gate) + closed-form total mass; shannon/normalized → erf cell masses (absolute)
or ENTRY 3 grid (relative or windowed); differential → repeated shannon on refined grids.
Memo: none on the entropy path. Per-call ts honoured only on the grid/differential paths (renyi2 uses the global default in both).
```

---

## 5. Windowed similarity

### 5.1 `windowed_similarity` / `windowedSimilarity`

Python `_tensor/windowed.py:319`; MATLAB `windowedSimilarity.m`. `[both]`

```
├─ sweep form (sweep given): ERROR when drop is missing or query_centres/query_window given → multi-position
├─ single form: ERROR when drop_window_attr is missing → single-position
└─ per position: window and translate the raw events (preprocessing), then
     IF nested (specs given) → build_exp_tens ×2 → ENTRY 1 cos_sim_exp_tens(dc, dq, normalize, verbose=false)
     ELSE → ENTRY 1 cos_sim_exp_tens(raw 9/10-argument form, normalize, verbose=false)
   No method, truncation_sigmas, or kernel_precision is exposed: every position routes through §1.3 with
   method = 'auto' and the default width.
   [Py] windowed_entropy → ENTRY 4 entropy_exp_tens(dens, method, base) per position (default 'differential').
```

---

## 6. `explain_dispatch` / `explainDispatch`

Python `_tensor/explain.py:120`; MATLAB `explainDispatch.m`. Reports; does not route. `[both]`

- **eval (one density)**: re-runs the eval selector (§3.4) with `n_q or 200` and the cost model — identical to the real decision for a plain `MaetDensity` passed directly (the real path uses the actual `n_q`); a user `method` overrides the reported choice.
- **cosine, flat**: calls the flat selector (§1.4) **without `wrap_vec`** (Python omits it; MATLAB passes `{}`), so rule 5 (the wrap/measure rule and its mixed-wrap error) is never reported; **without `nu_vec`** (default 2000 nodes per relative attribute instead of the geometry-derived count), so rule 6 prices differently; **without the memo-derived skip flags**; and without the `ordered_any` override, the `N == 0` short-circuit, `method='factored'` `[Py]`, or the post-hoc guard. The report can therefore name a route the call does not take: verified in Python for `r = 3`, `K = 4 vs 5`, rel-per, `σ/P = 0.5`, `wrap = 'full-image'` — explain reports `bulger (cost model)`, the call chooses `mobius` by rule 5; its `measure` line then names the opposite reading. Identically unfaithful in both languages (parity row 97).
- **cosine, nested**: re-runs the route choice (§1.8.2) without skip flags, `enumeration_admissible`, `select_nested_method`, and the safety factor — the same decision as the real plan for `method='auto'` modulo the skip flags; `method='bulger'` reported as forced; `'mobius'`/`'contract'`/`'centres'` reported as "contract".
- **periodicity**: `thr(ts)`, with "positive-definiteness" named as the limiting factor iff `thr >= 0.05`.

---

## 7. Leaves

Column "Measure": A = minimum-image, C = all-image, NP = non-periodic (no wrapping); "A/C by wrap" = the leaf branches on `wrap_a` (with the `L = 0` short-circuit making A and C coincide); "A (rel-per)" = the pairwise-wrapped quadratic form.

### Leaves of ENTRY 1 (cosine)

| Leaf (Python / MATLAB) | Computes | Measure | Reached from |
|---|---|---|---|
| `_r1_broadcast_fast` kernel loop / `localR1BroadcastFast` (`cosine.py:759` / `cosSimExpTens.m:2547`) | all-r = 1 cross terms against a shared operand, one kernel pass | A/C by wrap; NP | §1.2b (density broadcast; `[M]` also raw-MA broadcast) |
| `_ip_via_helper` → `gaussian_kernel_sum` / — (`cosine.py:3713`, `_kernel.py:27`) | single-attribute inner product (A = 1, not rel-per); factored flat factor | A/C by wrap; NP | §1.5a core `[Py]` (A-15); §1.5d `[Py]` |
| `_ip_r1_direct` / `localIpR1Direct` (`cosine.py:1857` / `:2386`) | all-r = 1 multi-attribute inner product | A/C by wrap; NP (rel skipped) | §1.5a core (Bulger and centres arms) |
| `_ip_full_ma` + chunked `_ma_log_kernel` / `ipFullMA` + `maLogKernel` (`cosine.py:1923, 2024` / `:1398, 1405`) | Bulger perm × comb, or centres perm × perm, truncated log-kernel product | A/C by wrap on abs-per; A (rel-per); NP | §1.5a, §1.5b |
| `_gram_quadratic_form` / `localGramQuadraticForm` (`cosine.py:1978` / `:2269`) | non-periodic Q via the Gram identity | NP | `ma_log_kernel` gate |
| `_compute_Q`, `_compute_Q_inner_blocks` / `computeQaMA`, `qInnerBlocks` (`dispatch.py:848, 745` / `:1526, 1498`) | quadratic forms (rel-per pairwise wrap) | A (rel-per); A after nearest image (abs-per); NP | every log-kernel and closed-form leaf |
| `wrapped_gaussian_1d` / `internal.wrappedGaussian1d` (`_wrapped_kernel.py:103` / `+internal/wrappedGaussian1d.m`) | abs-per θ (image sum or Fourier) | C | every abs-per full-image branch |
| `_ma_per_attr_inner_matrix` r = 1 sum / `localR1ZeroPad` (`_mobius_inner.py:322` / `maPerAttrInnerMatrix.m:233`) | r = 1 per-attribute kernel sum | A/C by wrap; NP | §1.6.2 |
| `inner_product_orbit_pw_batched` (dense abs) / `localSafeSafeOrbit` → `innerProductOrbitPwBatched` (`_mobius.py:851` / `maPerAttrInnerMatrix.m:292`) | absolute r ≥ 2 Möbius, dense | A/C by wrap; NP | §1.6.2 |
| `_orbit_safe_submatrix_sparse` → `inner_product_orbit_sparse` / `localSafeSafeOrbitSparse` → `innerProductOrbitSparse` (`_mobius_inner.py:180`, `_mobius.py:1056` / `maPerAttrInnerMatrix.m:378`) | absolute r ≥ 2 Möbius, sparse | NP (gate requires non-periodic) | §1.6.2 sparse gate |
| `_spectral_rel_inner_matrix` / `mobius.spectralRelInnerMatrix` (`_mobius_inner.py:894` / `+mobius/spectralRelInnerMatrix.m`) | relative r = 2..4 Fourier Gram matrix | C; NP | §1.6.4 spectral gate |
| `_rel_inner_batched` grid → `inner_product_orbit_grid` / `_pw_batched` / `relInnerBatched` → `innerProductOrbitGrid` / `innerProductOrbitPwBatched` (`_mobius_inner.py:1244` / `relInnerBatched.m:246-270`) | relative translation grid with image sum | C; NP | §1.6.4 |
| `_rel_per_inner_sparse` / `localRelPerInnerSparse` (`_mobius_inner.py:689` / `relInnerBatched.m:388`) | rel-per sparse per-node orbit | C | §1.6.4 sparse gate |
| `_closed_form_attr_matrix_from` / `mobius.closedFormAttrMatrixFrom` (`_mobius_inner.py:1843` / `+mobius/closedFormAttrMatrixFrom.m`) | tuple-centres Gaussian overlap, comb-restricted | A (rel-per); A/C by wrap (abs-per, nested only); NP | §1.5c `choices` (flat rel); §1.8.4 `'centres'`; §1.9 `'ordered'` |
| `nested_attr_matrix` → `_combine_orbit` / `_combine_chunked` / `nestedAttrInnerMatrix` → `combineOrbit` / `combineChunked` (`_nested_contraction.py:930, 402, 277` / `nestedContract.m:1303, 1822, 1638`) | nested contraction (`contract`, `taugrid`, `contract_relnonper`) | A/C by wrap (abs-per); C (`taugrid`); NP | §1.8.4, §1.8.5 |
| `_shared_template_matrix` / `ipRelNonperFactored` (`_nested_contraction.py:803` / `nestedContract.m:1963`) | rel-nonper shared-template closed form | NP | §1.8.5 |
| `_ma_ip_factored` (+ `_ma_ip_factor_dense`, `_ma_ip_factor_nested_culled`) / — (`cosine.py:3579`) | `method='factored'` | A/C by wrap; NP | §1.5d `[Py]` |
| `sweep_cos_sim_exp_tens` / — (`sweep.py:833`) | tagged-sweep reduction | as ENTRY 2 | §1.1a `[Py]` |

### Leaves of ENTRY 2 (sweep)

| Leaf | Computes | Measure | Reached from |
|---|---|---|---|
| `_orbit_attr_matrix_sweep` / `localOrbitAttrMatrixSweep` (`sweep.py:603` / `sweepCosSimExpTens.m:482`) | swept-attribute Möbius (r ≥ 2) or kernel sum (r = 1) per offset | A/C by wrap; NP | orbit route |
| `_ma_per_attr_inner_matrix` / `mobius.maPerAttrInnerMatrix` | untranslated attributes and orbit denominators | as §1.6.2 | orbit route |
| `_build_mixture` → `_evaluate_mixture` / `_evaluate_dense` / `localBuildMixture` → `localEvaluateMixture` / `localEvaluateDense` (`sweep.py:249, 406, 481` / `:702, 895`) | Gaussian-mixture sweep over offsets; fixed attributes via `ma_log_kernel` | A/C by wrap (fixed); NP (swept) | mixture route |
| `_self_ip` / `localSelfIp` (`sweep.py:1005` / `:1063`) | mixture denominators at zero offset | as above | mixture route (`[Py]` memo `('sweep', ts, kp)`) |

### Leaves of ENTRY 3 (point evaluation)

| Leaf | Computes | Measure | Reached from |
|---|---|---|---|
| `eval_ma_orbit` → `eval_orbit_abs` / `mobius.evalMaOrbit` → `evalOrbitAbs` (`_ma_eval_orbit.py:30`, `_mobius.py:1287` / `+mobius/evalMaOrbit.m:94`, `evalOrbitAbs.m:100`) | factored Möbius, absolute attribute | A/C by wrap (direct branch); reduction branch via `gaussian_kernel_sum` | §3.3 `'mobius'` |
| `eval_orbit_rel` → `_eval_orbit_rel_fourier` / factored tabulation / direct u-grid (`_mobius.py:1873, 1731` / `evalOrbitRel.m:113`) | factored Möbius, relative attribute | C (Fourier); direct/tabulated with default `wrap='full-image'` | §3.3 `'mobius'` |
| `gaussian_kernel_sum` → truncated 1-D / bucket / circular / `_eval_chunk` (`_kernel.py:27` / `+internal/gaussianKernelSum.m`) | block kernel sums | NP; A (circular); A/C by wrap (`eval_chunk`) | single-multiset centres `[M]`; factored MA (both, without wrap — B-5); `eval_orbit_abs` reduction; `eval_orbit_rel` tabulation |
| `_eval_core` → `_truncated_kernel_sum_culled` / `_eval_full` / — (`eval.py:1543, 1422, 1595`) | single-multiset centres (with wrap) and factored-MA factors (without wrap) | NP; A/C by wrap | §3.3 centres `[Py]` |
| `_ma_eval_factored` / `localMaEvalFactored` (`eval.py:1077` / `evalExpTens.m:1408`) | factored centres (A ≥ 2, all r ≥ 2) | C forced on abs-per (B-5) | §3.3 centres |
| `_ma_eval_full` / `maetEvalFull` (`eval.py:1264` / `:999`) | joint-centres materialisation | A/C by wrap; A (rel-per); NP | §3.3 centres |
| `wrapped_gaussian_1d` / `internal.wrappedGaussian1d` | abs-per θ (e = 2) | C | every abs-per full-image eval branch |
| `_compute_Q` / `_compute_Q_inner_blocks` / `computeQ*` | quadratic forms | A / NP | every eval leaf |

### Leaves of ENTRY 4 (entropy)

| Leaf | Computes | Measure | Reached from |
|---|---|---|---|
| `_cell_masses_ma_absolute` / `cell_masses_*_absolute` (`entropy.py:288` / `entropyExpTens.m:517, 666`) | erf cell masses per axis | A (minimum-image erf regardless of wrap, B-6); NP | shannon/normalized on absolute densities; differential levels |
| ENTRY 3 grid (`eval_exp_tens(method='auto')`) | density on an nd-grid | as ENTRY 3 | shannon/normalized on relative or windowed densities |
| `_differential_adaptive` / `localDifferentialAdaptive` (`entropy.py:896` / `:1568`) | Richardson-extrapolated differential entropy | as the Shannon core | `'differential'` |
| `_renyi2_per_attr_numerical` / `renyi2_per_attr_numerical` (`entropy.py:1432` / `:2003, 2160`) | overlap of a rebuilt nested or ordered attribute | A/C by wrap; NP | renyi2 nested/ordered attributes |
| `_ma_per_attr_inner_matrix` / `maPerAttrInnerMatrix` (§1.6.2) | flat symmetric attribute self-overlap | `[M]` A/C by wrap; `[Py]` C forced (A-3, fixed in this round) | renyi2 flat attributes |
| — / `orbitInnerAbsSingleMultiset` → `innerProductOrbit`, `orbitInnerRelSingleMultiset` → `relInnerBatched` (`entropyExpTens.m:2066-2082`) | single-multiset r ≥ 2 self-overlap | A/C by wrap (abs); C (rel) | renyi2 single-multiset `[M]` |
| — / single-multiset r = 1 direct pairwise (`entropyExpTens.m:2020-2052`) | single-multiset r = 1 absolute self-overlap | A/C by wrap | renyi2 single-multiset `[M]` |
| `total_mass_abs` / `total_mass_rel` (`_mobius.py:1098, 1142` / `+mobius/totalMassAbs.m`, `totalMassRel.m`) | closed-form normaliser Z | — | renyi2 |

### Leaves of ENTRY 5 (windowed)

| Leaf | Computes | Measure | Reached from |
|---|---|---|---|
| ENTRY 1 per position (`method='auto'`, default ts) | windowed cosine | as ENTRY 1 | `windowed_similarity` |

---

## 8. Overrides

| `method` value | Entry point | Forces | Bypasses | Still guarded by | Errors it can raise |
|---|---|---|---|---|---|
| `'auto'` | all | nothing | nothing | every rule | as the rules |
| `'bulger'` | cosine, flat | Bulger arm | selector rules 2–6 (feasibility guard, working-set rule `[Py]`, wrap/measure rule, cost race) | structural guards; empty → 0; ordered override (no-op) | structural `ValueError`/`*Mismatch` |
| `'bulger'` | cosine, nested | joint-tuple enumeration | the nested plan entirely | as above | as above |
| `'centres'` | cosine, flat | unrestricted centres arm (perm × perm) | all selector rules | ordered override → Bulger (silent); structural guards | structural only |
| `'centres'` | cosine, nested | plan with `force_route='centres'`, `force=true` | cost race and enumeration race | measure rule (rel-per full-image above threshold → ERROR); decline conditions → ERROR | `ValueError` "cannot be honoured" / `centresUnavailable`; "not available here" / `contractUnavailable` |
| `'mobius'` | cosine, flat | Möbius arm; grid pinned per relative attribute (unless `rel_attr_route` explicit) | all selector rules (including the wrap/measure rule) | ordered override → Bulger (silent); post-hoc impossible-value guard → Bulger; structural guards | structural; `rel_attr_route='centres'` block error |
| `'mobius'` | cosine, nested | plan with `force=true` | cost races | measure rule; decline conditions → ERROR | `contractUnavailable` |
| `'contract'` | cosine | as `'mobius'` on nested densities | cost races | `nested_any` required | `ValueError` / `contractUnavailable` on non-nested |
| `'factored'` `[Py]` | cosine | `_ma_ip_factored` | selector, ordered rule, nested plan, post-hoc guard, memo | structural guards; `_ma_factored_ip_supported` (no covariance, no rel-per) | `ValueError` |
| `'mixture'` | sweep | mixture route | `_choose_sweep_route` | `sweep_eligibility` | eligibility errors |
| `'orbit'` | sweep | orbit route | `_choose_sweep_route` | `orbit_sweep_supported` | `ValueError` / `orbitUnsupported` |
| `'centres'` | eval | centres branch | every selector rule (feasibility and memory guards included) | shape rules inside the branch (single-multiset / factored / joint) | none from the selector |
| `'mobius'` | eval | Möbius evaluator | measure rule, cost model, `r > 10` rule | `[Py]` ordered guard for all shapes; `[M]` ordered guard on the single-multiset corner only (A-4, fixed in this round); `[M]` single-multiset non-finite fallback | `ValueError` / `mpt:evalExpTens:orderedMobius` |
| `'differential'`, `'shannon'`, `'normalized'`, `'renyi2'` | entropy | the named estimator | nothing (no routing beneath the method) | form restrictions (lists, batched, windowed) | `NotImplementedError` / `*NotSupported`, grid errors |
| (none exposed) | windowed | — | — | — | — |

Global defaults and levers (both languages unless tagged): `rel_attr_route` (`'centres'`/`'mobius'`; pins the Möbius arm's per-attribute route, raises when inadmissible); `post_hoc_guards=false` (skips the impossible-value fallback and the nested orbit accuracy guard); `SPECTRAL_IP_ENABLED`/`SPECTRAL_IP_FORCE` (the latter bypasses only the spectral cost gate, never the 4e6 memory guard); `COMB_RESTRICTION_ENABLED`; `_FOURIER_ENABLED` `[Py]`; `nestedCostOverride` `[M]` (test hook). Forms that drop the user's `method`: Windowed eval (both); MATLAB cosine list forms, eval list and batched-raw forms (A-10, A-11).

---

## 9. Constants and thresholds

| Constant | Python value (file) | MATLAB value (file) | Role |
|---|---|---|---|
| `truncation_sigmas` factory default | 6.0 (`_defaults.py:75`) | 6 (`mptDefaults.m:191`) | kernel cutoff width |
| accuracy-floor eps / width | 1e-12 → 7.4338 (`_defaults.py:105`) | 1e-12 → ≈ 7.43 (`+internal/accuracyFloor.m:36, 53`) | `inf` resolution; `truncation_floor` |
| `thr(ts)` (rel-per σ/P) | departure table 0.020…0.100 ↦ 1.67e-16…4.07e-2; PD ceiling 0.05; 0.03 at ts = 6 (`dispatch.py:996-1030`) | same table and ceiling (`+internal/relPerSigmaOverPThreshold.m:60-84`) | flat rule 5, eval rule 8, nested measure rule, sweep, centres gate |
| abs-per σ/P warning threshold | 0.04 (`dispatch.py:1103`) | 0.04 (`+internal/absPerSigmaOverPThreshold.m:40`) | build-time diagnostic only; nothing routes on it |
| `ORBIT_R_MAX_SHIPPED` | 8 (`dispatch.py:976`) | 8 (`selectMaInnerProductMethod.m:112`, `nestedContract.m:1429`, `getOrbitTable.m:115`; hard cap 12) | flat rule 3; orbit eligibility |
| `ORBIT_R_MAX_FEASIBLE` | 10 (`dispatch.py:1220`) | 10 (`+internal/selectMaEval.m:70`) | eval rule 7 |
| forced-Bulger / forced-centres memory budget | 4 GiB fixed (`dispatch.py:1192`) | clamp(availableMemory/2, 1 GiB, 4 GiB) (`+internal/dispatchMemBudget.m:17-20`) | flat rule 3 guard; eval rule 7 guard |
| `CENTRES_WORKING_SET_SOFT_BUDGET` | 256 MiB (`dispatch.py:1213`) | 256 MiB (`selectMaEval.m:92-94`, `nestedCost.m`) | flat rule 4 `[Py]`; eval safety factor; nested centres guard |
| `MA_MOBIUS_SAFETY` / `_SMALL` | 1.5 / 1.0 (`dispatch.py:1460-1461`) | 1.5 / 1.0 (`selectMaEval.m:92-94`) | eval rule 9 |
| impossible-cosine tolerance | 1.000001 (`dispatch.py:169`) | 1.000001 (`cosSimExpTens.m:809`) | post-hoc guard |
| `post_hoc_guards` default | True (`_defaults.py:79`) | true (`mptDefaults.m:194`) | post-hoc and accuracy guards |
| `rel_attr_route` default | `'auto'` (`_defaults.py:88`) | `'auto'` (`mptDefaults.m:197`) | centres-vs-grid gate |
| `orbit_cost_intercept` | 3.8536 (`_defaults.py:80`) | 3.8536 (`mptDefaults.m:196`) | per-level orbit vs enumeration |
| `orbit_cost_model` coefficients; `MARGIN_R_MAX` | 1.0708, 1.4033, −0.3608, −0.8188, −0.2261; 3 (`_orbit_cost.py:33-43`) | same (`+internal/orbitCostModel.m:77-99`) | per-level orbit vs enumeration |
| `ORBIT_ENUM_MAX_WORK` / `ORBIT_ENUM_MAX_ELEMS` | 16e6 / 16e6 (`_nested_contraction.py:229-230`) | 16e6 / 16e6 (`nestedContract.m:1755, 1642`) | accuracy-guard fallback; chunking |
| `NESTED_ENUM_SAFETY` | 2.0 (`_nested_cost.py:195`) | 2.0 (`+internal/nestedCost.m`) | plan vs enumeration |
| `NESTED_LAW_KEYS` | (2, 3, 4, 6) (`_nested_cost.py:91`) | [2, 3, 4, 6] (`nestedCost.m:319-350`) | nested cost law row |
| `NESTED_COST_LAW`, `NESTED_FLOOR_MS` | fitted (`_nested_cost.py:147, 165`) | fitted, different values (`nestedCost.m:174-236`) | nested pricing — per-language by design |
| `REL_COST_LAW` (bulger / centres / grid, rows r = 2, 3, ≥ 4) | fitted (`dispatch.py:340`) | fitted, different values (`+internal/relRouteCostMs.m:43-53`) | flat rule 6 — per-language by design |
| `ORBIT_REL_FLOOR_MS`, `ORBIT_ABS_PER_ATTR_MS` | fitted (`dispatch.py:286, 237`) | fitted; ABS row identical {3, 11.2, 45, 150, 500, 1500, 4500} (`+internal/predictOrbitCostMs.m:34-70`) | flat rule 6 |
| eval cost constants (centres/Möbius set-up and per-op, cull constant, Bell numbers 1..10) | `dispatch.py:1226-1432` | `+internal/maEvalCostsMs.m:47-174, 300-330` | eval rule 9 — per-language by design |
| spectral eval gate `n_q` / `K` minima | (16, 32, 64) / (2, 8, 16) for r = 2, 3, 4 (`_mobius.py:2046`; `dispatch.py:1885, 1912`) | same (`evalOrbitRel.m:185-186`; `maEvalCostsMs.m`) | `eval_orbit_rel` spectral gate; cost model |
| `FOURIER_MODE_SIGMAS` / `SPECTRAL_IP_MODE_SIGMAS` | 8.6 / 8.6 (`_mobius.py:1728`; `_mobius_inner.py:767`) | 8.6 (`spectralRelInnerMatrix.m:100`; `maEvalCostsMs.m`) | mode counts |
| `SPECTRAL_IP_MAX_POINTS` / `SPECTRAL_IP_COST_C` | 4 000 000 / 1100 (`_mobius_inner.py:773, 852`) | 4e6 / 1100 (`spectralRelInnerMatrix.m:100-103`) | spectral IP memory guard / cost gate |
| `SPECTRAL_IP_ENABLED` / `_FORCE` | True / False (`_mobius_inner.py:746, 759`) | `internal.spectralIpEnabled()` / `spectralIpForce()` | spectral IP gate levers |
| `COMB_RESTRICTION_ENABLED` | True (`_mobius_inner.py:1617`) | `internal.combRestrictionEnabled()` default true (`closedFormAttrCentres.m:203`) | comb-side restriction |
| sparse-orbit gate `MIN_KERNEL` / `MAX_DENSITY` | 200 000 / 0.20 (`_mobius_inner.py:81, 84`) | 200000 / 0.20 (`maPerAttrInnerMatrix.m:373-374`; `relInnerBatched.m:318-319`) | abs and rel-per sparse gates |
| `ORBIT_GRID_SLAB_ELEMS` | 2^19 (`_mobius_inner.py:40`) | 2^19 (`relInnerBatched.m:74`) | grid slabs |
| `CENTRES_NS_BASE` / `_LIN` / `_WRAP`; `GRID_NS_FLOOR`; `GRID_NS_PER_OP` | 15, 5, 10/3; 1e6 ns; {2: 30, 3: 700, 4: 2000, r ≥ 5: 2000·3^(r−4)} (`_mobius_inner.py:1440-1452`) | identical (`maRelAttrPrefersCentres.m:64-67, 168-181`) | centres-vs-grid gate |
| closed-form chunk | 16 000 000 ÷ (n_jy·d) (`_mobius_inner.py:1944`) | same (`closedFormAttrMatrixFrom.m:138`) | tuple-centres chunking |
| `resolve_samples_per_sigma` | `max(2, ceil(sqrt(r·ln(1/ε))/(π√2)) + 1)` → 3 (r ≤ 4), 4 (r = 5, 6) at ts = 6 (`_defaults.py:197`) | `max(2, ceil(ts·√r/(2π)) + 1)` — algebraically the same (`+internal/resolveSamplesPerSigma.m:157-159`) | grid node density |
| `rel_window_margin` | `min(√2·ts + 0.1, 8)`; 8 when non-finite (`_mobius_inner.py:1360`) | same (`+internal/relWindowMargin.m:113-117`) | rel-nonper grid window |
| `auto_ntau_default` | `max(64, ceil(2πP/σ·(1 + 0.5·max(0, −log10 tol)/12)))`, tol from the GLOBAL ts (`_nested_contraction.py:646`) | same (`+internal/autoNtauDefault.m:76-84`) | τ grids, `nu_vec`, centres gate |
| `auto_taus_line` / rel-nonper quadrature | step σ/4; pad (6 + 0.5·max(0, −log10 tol))σ; ≥ 64 (`_nested_contraction.py:681`) | same (`nestedContract.m:2102-2106`) | nested `contract_relnonper` |
| `gram_is_accurate_enough` factor | `eps·s²/(4σ²) <= 0.1·floor` (`cosine.py:1975`) | same (`cosSimExpTens.m:2264-2265`) | Gram gate |
| `tuple_values_repeat` | 100 / 256 / 4 (`eval.py:1200`) | 100 / 256 / 4 (`+internal/tupleValuesRepeat.m`) | value tables |
| image count `L(e)` at ts = 6 | L(4) > 0 iff σ/P > 0.0589; L(2) > 0 iff σ/P > 0.0833; Fourier(4) from ≈ 0.2 (`_wrapped_kernel.py:44, 88`) | same rule (`+internal/wrappedKernelImageCount.m:22-37`, `wrappedKernelFourierCount.m:22-38`) | θ branch and L = 0 short-circuits |
| `rel_per_image_count` | `ceil(2(σ/P)·sqrt(ln 1/tol) − ½)` (`_mobius_inner.py:855`) | same (`relInnerBatched.m:220-239`) | rel-per grid image sum |
| `eval_orbit_abs` reduction gate | `P > 2·√2·ts·σ` (`_mobius.py:1287`) | same (`evalOrbitAbs.m:116-117`) | per-block helper reduction |
| `eval_orbit_rel` non-per window margin; factored constants | 8σ; `EPS_CEIL` 1e-3, `CALIB_A6` 1600, `SPP` 8/512, `READBACK_COST` 10 (`_mobius.py:1585-1602`) | same (`evalOrbitRel.m:157-158, 437-469`) | u-grid; factored gate |
| sweep `ORBIT_WORK_RATIO` / `ORBIT_MIN_PAIRS` / dense-vs-culled overhead | 64 / 1e6 / 512 (`sweep.py:523, 532, 462`) | 64 / 1e6 / 512 (`sweepCosSimExpTens.m:345-346, 955`) | sweep chooser |
| r = 1 broadcast cache cap | `4 000 000 // (n_j·8)` columns (`cosine.py:755`) | `floor(4e6/(nJ·8))` (`cosSimExpTens.m:2618`) | fast path |
| entropy `DEFAULT_GRID_LIMIT` / `DIFF_CELL_BLOCK` | 1e8 / 8e6 (`entropy.py:30, 38`) | gridLimit option / — | entropy grid guard, streaming |
| differential `tol`, `max_iter`, stall factor | `max(exp(−ts²/2), 1e-12)`, 10, 0.95 (`entropy.py:915-1003`) | same (`entropyExpTens.m:1568-1700`) | differential convergence |
| `innerProductOrbitSparse` density threshold | — | 0.34 (`+mobius/innerProductOrbitSparse.m:45`) | sparse contraction internals |
| `kernel_chunk_bytes` | resolved from memory (`kernel_chunk_bytes_resolved()`) | `mptDefaults('kernelChunkBytes')` = 'auto' (availableMemory/2) (`mptDefaults.m:195`) | chunk decisions |
| vestigial: `_PROBE_*`, `_PRESCREEN_IP_DOMINANCE`, `_ORBIT_IP_FIXED_OVERHEAD`, `_PW_PER_ENTRY_MS_*` | `dispatch.py:217-223, 1188, 2221-2266` | `GRID_OP`, `CENTRES_OP`, `REL_BASE` (`predictOrbitCostMs.m:39-43`) | unused (B-17) |

---

## 10. Parity

Condensed from `routing_parity.md` §2; the ids are kept so that the full evidence can be looked up there. Rank: [1] changes the number returned (or error vs value); [2] changes the route or measure for the same input; [3] speed only; [4] cosmetic. "Fix follows" names the side whose behaviour the other should adopt.

### A — Parity gaps (behaviour differs for the same input)

| Id | Rank | Gap | Python | MATLAB | Fix follows | Status |
|---|---|---|---|---|---|---|
| A-1 | [1] | Per-call `truncation_sigmas` ignored by the entire Python nested path (threshold, quadrature tolerance, kernel cutoffs, centres θ, memo key, enumeration admissibility all use the global default); MATLAB threads it. | `cosine.py:2798` (no ts parameter), `:1649-1652`, `:2523`, `:2622`, `:2750-2752`, `:2769`, `:2892-2895`, `:2911-2920`, `:3020-3028` | `cosSimExpTens.m:1105-1107`; `nestedContract.m:220, 272, 293-305, 409, 604, 2090-2106` | MATLAB (thread ts) | open |
| A-2 | [1] | Nested-MA flat attribute: Python drops `wrap` and ts → a `single-image` abs-per attribute is computed full-image when it shares a density with a nested attribute. | `cosine.py:3043-3050` | `nestedContract.m:1145-1157` | MATLAB (pass wrap and ts) | fixed in this round |
| A-3 | [1] | Rényi-2 flat symmetric attribute: Python drops `wrap`. | `entropy.py:1605-1607` | `entropyExpTens.m:2171-2178` | MATLAB (pass wrap) | fixed in this round |
| A-4 | [1] | `eval(method='mobius')` on an MA density with an ordered attribute: Python raises; MATLAB evaluates the symmetrised density. | `dispatch.py:1987-1995, 2065-2074` | `evalExpTens.m:1273-1276, 895-904` (single-multiset guard at `:517-525`) | Python (guard ordered attributes on every shape) | fixed in this round |
| A-5 | [1] | Rényi-2 of a single-multiset relative density at r = 1: MATLAB returns 0 by convention; Python returns the finite value from the r = 1 absolute kernel — which MATLAB's own MA loop also returns (intra-MATLAB inconsistency, B-7). | `entropy.py:1596-1625`; `_mobius_inner.py:322-385`; `_mobius.py:1142-1145` | `entropyExpTens.m:2015-2018` vs `:2160-2191` | design decision; the finite value (Python, and MATLAB's MA loop) is the internally consistent one | open |
| A-6 | [4] | Flat Möbius tuple-centres branch: MATLAB omits per-call ts (4th argument); no numeric effect because the flat gate admits only relative attributes, whose closed form never reads ts; the memo key nonetheless carries the per-call width (B-3). | `cosine.py:2339-2346`; `_mobius_inner.py:1971-1987` | `cosSimExpTens.m:1666-1671`; `closedFormAttrMatrixFrom.m:73-75, 118-120`; key `:1615-1616` | Python (pass ts) | open, latent |
| A-7 | [1] | Eval post-hoc non-finite fallback: MATLAB single-multiset corner diverts non-finite Möbius output to centres; Python never checks (its docstring promises one, D-5). MA path unguarded in both (B-8). | `eval.py:893-898`, docstring `:158-162` | `evalExpTens.m:557-571` | MATLAB (add the guard; ideally on the MA path too in both) | open |
| A-8 | [1] | Batched-raw single-multiset with a kernel covariance: Python whitens and computes; MATLAB errors. | `cosine.py:539-562` | `cosSimExpTens.m:651-657` | Python (whiten) — a scope decision | open |
| A-9 | [1] | `kernel_precision` / per-call ts on secondary forms: Python honours kp on the A = 1 cosine helper and the joint eval route only, and drops ts/kp on the Shannon list and raw-MA forms; MATLAB honours kp on the Möbius and single-multiset centres eval routes (no cosine route) and forwards both on the entropy list form. | `cosine.py:1786-1797`; `eval.py:893-898, 926-937, 1035-1039`; `entropy.py:1123-1175, 646-660` | `cosSimExpTens.m:323-334`; `evalExpTens.m:557, 574-599, 1290-1303, 1525-1533`; `entropyExpTens.m:784-787, 936-964` | MATLAB for eval kp and entropy list forwarding; Python for the cosine helper (union: forward everywhere) | open |
| A-10 | [2] | MATLAB density-list cosine forms drop `method` and `truncationSigmas` (cell–cell and scalar–cell); Python forwards them. | `cosine.py:1080-1145` | `cosSimExpTens.m:1763-1766, 1801-1802, 1863-1864` | Python (forward) | open |
| A-11 | [2] | MATLAB eval list and batched-raw forms drop `method`; Python forwards it. | `eval.py:452-470, 500-505, 557-660` | `evalExpTens.m:1785-1792, 1845-1868` | Python (forward) | open |
| A-12 | [2] | Python-only flat selector rule 4 (working-set guard → Möbius); MATLAB races cost. Same measure. | `dispatch.py:642-666` | `selectMaInnerProductMethod.m:112-174` (absent) | Python (port the memory safeguard) or document the asymmetry | open |
| A-13 | [2] | `method='factored'` exists only in Python (undocumented, D-10). | `cosine.py:1467-1492` | `cosSimExpTens.m:300-308` rejects it | document (or port) — not a correctness issue | open |
| A-14 | [3] | Raw-MA scalar-vs-list: Python tries the `TranslatedSweep` reduction, never the r = 1 fast path; MATLAB the reverse. Same numbers. | `cosine.py:1289-1332` | `cosSimExpTens.m:616-624` | either (speed) | open |
| A-15 | [3] | single-multiset inner-product leaf: Python routes A = 1 (non rel-per) through `gaussian_kernel_sum`; MATLAB always uses the log-kernel form. Same cutoff, numbers agree within the floor. | `cosine.py:1781-1797` | `cosSimExpTens.m:1353-1386` | either (speed) | open |
| A-16 | [3] | Python-only memos: sweep mixture `'sweep'` key, nested centres bundle cache, density-list dedup and calibration probe. | `sweep.py:1016-1023`; `_mobius_inner.py:1814-1841`; `cosine.py:850-960` | `sweepCosSimExpTens.m:224-227, 1063`; `nestedContract.m:280-292`; `cosSimExpTens.m:1747-1767` | either (speed) | open |
| A-17 | [4] | Eval `normalize` unvalidated in Python; `mode='cartesian'` Python-only; Python sweep orbit route forwards an unresolved `inf`. | `eval.py:791`; `cosine.py:1080-1097`; `sweep.py:752-760` | `evalExpTens.m:245-250`; `cosSimExpTens.m:1747-1753`; `sweepCosSimExpTens.m:183-200` | MATLAB (validate); cosmetic | open |

### B — Implementation defects

| Id | Rank | Defect | Language | Evidence |
|---|---|---|---|---|
| B-1 | [1] | Dropped argument: ts never reaches the nested contraction (A-1); memo key blind to the per-call width. | Python | `cosine.py:1649-1652, 2798, 2523, 2622, 2769, 2892` |
| B-2 | [1] | Dropped `wrap` (and ts): nested-MA flat branch and Rényi-2 flat branch (A-2, A-3). | Python | `cosine.py:3043-3050`; `entropy.py:1605-1607` — fixed in this round |
| B-3 | [4] | Dropped ts in the flat Möbius centres branch (latent; A-6). | MATLAB | `cosSimExpTens.m:1666-1671` vs `:1615-1616`; `closedFormAttrMatrixFrom.m:118-120` |
| B-4 | [1] | Missing guard: MA eval honours `'mobius'` on an ordered attribute (A-4). | MATLAB | `evalExpTens.m:1273-1276, 895-904` — fixed in this round |
| B-5 | [1] | Shared: factored MA eval route drops `wrap` → single-image opt-in evaluated full-image; the joint and single-multiset routes honour it, so the measure depends on the shape/memory branch. | both | `eval.py:1189-1194` vs `:926-937, 1366-1385`; `evalExpTens.m:1515-1533` vs `:1047-1113` |
| B-6 | [1] | Shared: Shannon/normalized entropy on absolute-periodic densities uses the minimum-image erf regardless of `wrap`; its ts argument is unused. | both | `entropy.py:173-188`; `entropyExpTens.m:1011-1036` |
| B-7 | [1] | Rényi-2 single-multiset r = 1 relative convention (H = 0) contradicts MATLAB's own MA loop (A-5). | MATLAB (intra-language) | `entropyExpTens.m:2015-2018` vs `:2160-2191` |
| B-8 | [2] | Missing eval non-finite guard: Python none (docstring promises one); MATLAB single-multiset corner only. | Python; MATLAB MA path | `eval.py:158-162, 893-898`; `evalExpTens.m:557-571, 1282-1303` |
| B-9 | [2] | Shared: `wants_single`/`wants_full` scan every relative attribute, not only rel-per, so a rel-nonper attribute at the default wrap plus a rel-per `single-image` attribute raises "Mixed rel-per wrap". | both | `dispatch.py:676-683`; `selectMaInnerProductMethod.m:150-157` |
| B-10 | [2] | Shared: only `dens_x.wrap` is consulted by the flat selector, the nested measure rule, and the enumeration rule — operand-order dependent. | both | `cosine.py:1575, 2518-2520, 2623-2625`; `cosSimExpTens.m:928-932`; `nestedContract.m:380-433, 590-618` |
| B-11 | [2] | Shared: `auto_ntau_default`, the selector's `nu_vec` margin, and the centres-vs-grid grid estimate read the global ts, so a per-call ts changes the kernel cutoff but not the τ-grid density or the route price. | both | `_nested_contraction.py:658-670`; `cosine.py:1550-1551`; `_mobius_inner.py:1490-1493`; `autoNtauDefault.m:76-84`; `cosSimExpTens.m:987-992`; `maRelAttrPrefersCentres.m:150-152` |
| B-12 | [3] | Dropped arguments in MATLAB list forms (A-10, A-11); docstrings do not say so (D-11). | MATLAB | `cosSimExpTens.m:1763-1766, 1801-1802, 1863-1864`; `evalExpTens.m:1785-1792, 1845-1868` |
| B-13 | [3] | Dropped `kernel_precision` on the Python eval Möbius and single-multiset centres routes (A-9). | Python | `eval.py:893-898, 926-937` |
| B-14 | [3] | Shared: nested routes compute `<X,X>` under `oneSidedDenom` whenever it is not memoised; flat routes skip it. | both | `cosine.py:2896-2907`; `nestedContract.m:319-324, 1148-1152` |
| B-16 | [4] | Shared dead code: the "safe/unsafe" partition; the block-Q form of the closed form (`bs >= 2`); Python's unused `L_abs_per`. | both | `maPerAttrInnerMatrix.m:161-166, 471-540`; `_mobius_inner.py:555, 579, 1928-1937`; `closedFormAttrMatrixFrom.m:152-153, 180`; `cosine.py:2847-2848` |
| B-17 | [4] | Shared vestigial constants (probe/prescreen in Python; `GRID_OP`/`CENTRES_OP`/`REL_BASE` in MATLAB). | both | `dispatch.py:217-223, 1188, 2221-2266`; `predictOrbitCostMs.m:39-43, 81, 88` |
| K_a | — | `maEvalCostsMs.m` Möbius pricing loop carried a stale `K_a` from the previous loop; now `K_a = kVec(a)` in all three loops (`:199, 227, 279`). | MATLAB | fixed in this round (verified in source) |

### C — Documentation mismatches

| Id | Claim vs code | Where |
|---|---|---|
| D-1 | Python `dispatch.py:4-16` describes a pre-screen / cost-model / timing-probe / cancellation-guard flow and names `_probe_eval_path`/`_probe_ip_path`, which do not exist; no probe runs anywhere. | Python only (`selectMaEval.m:11` correctly says "No probe") |
| D-2 | MATLAB `cosSimExpTens.m:204-210` documents a cancellation-ratio fallback; the value is never read (`:278`). | MATLAB only (`cosine.py:284-287` says it is inert) |
| D-3 | Both docstrings promise a "K − r precision guard" (`dispatch.py:2036-2043`, `:541-548`; `selectMaInnerProductMethod.m:15-18`; `maPerAttrInnerMatrix.m:58-66`); none exists, and the safe/unsafe partition is disabled. | both |
| D-4 | `dispatch.py:536-537` says rel-per above threshold "warns and routes to Bulger"; the code routes by `wrap` (full-image → Möbius) with no warning. | Python only |
| D-5 | `eval.py:158-162` claims a non-finite Möbius fallback; none exists. | Python only |
| D-6 | Both eval cost models price a spectral `MAX_POINTS` decline that the evaluator never performs (`dispatch.py:1897-1903` vs `_mobius.py:2041-2060`; `maEvalCostsMs.m:323-338` vs `evalOrbitRel.m:187-203`). | both |
| D-8 | `selfIpMemoised.m:15-18` mentions a `'sweep'` key that only Python writes. | MATLAB only |
| D-9 | `_mobius.py:1960-1963` describes a "cross-correlation strategy at r = 2" that does not exist. | Python only |
| D-10 | `method='factored'` accepted (`cosine.py:1467`) but absent from the docstring (`:243`). | Python only |
| D-11 | MATLAB list/batched forms silently drop `method`/ts/kp; the usage text does not say so. | MATLAB only |
| D-12 | `_mobius_inner.py:1568` still tests `rel_attr_route in ('mobius', 'grid')` though `set_default` normalises `'grid'`; harmless. | cosmetic |

### D — Orphans and dead code (summary; full table in §11)

Python: `_rel_contract_cheaper`; `_orbit_inner_abs`/`_orbit_inner_rel` (re-export only — their MATLAB twins are live on the Rényi-2 single-multiset corner); `_inner_product_direct_abs`, `_build_ordered_r_tuples`, `_ma_has_nan`, `_attr_value_range`; `_batched_direct_enum_abs`, `_pack_nan_top`; `dispatch._orbit_ips_impossible`, `_estimate_centres_array_bytes`, `_pw_per_entry_ms`, `_warn_rel_per_all_image`, `_format_time`, the `_PROBE_*` constants; `_nested_contraction.make_quadrature`, `nested_ip` and its `_ip_*` leaves, `_theta_truncation_L` (tests only); `_centres_inner.py` (tests only); `sweep.SweepNotEligible`; `L_abs_per`. MATLAB: `mobius.innerProductDirectAbsSingleMultiset` (tests only); `localFillDirectEnumGroups`, `localPackNanTop`, `localBatchedDirectEnumAbsSingleMultiset` (dead); `mobius.contract` (tests only); `internal.timeRepeated`, `nestedCostOverride`, `absPerSigmaOverPThreshold` (bench / hook / diagnostic); the second `localMaEvalFactored` call in `localEvalMA` (always `[]`); `localComputeQBlocks` and the inner-unit branch of `localReducedCentresFromValues`; `evalOrbitRel` `'factored'` ∈ {on, off}. None of these is a parity gap: every orphan either has an orphaned twin or its live twin's work is done by a differently placed routine with the same measure (the one exception in substance is A-5).

---

## 11. Reachability

"Reached" = on a path traced in §1–§6 from a public entry point. "Twin" = the other language's counterpart and its own status. Routines are grouped by routing module; helper arithmetic beneath a listed leaf (partition tables, einsum kernels) is not enumerated separately.

### 11.1 Python — `_tensor/cosine.py`

| Routine | Reached from | Twin (MATLAB) |
|---|---|---|
| `cos_sim_exp_tens`, `_canonical_normalize`, `_finalise_normalisation` | entry | `cosSimExpTens` — reached |
| `_cos_sim_density_path`, `_normalize_density_input`, `_resolve_list_list_mode`, `_compute_pair_results_with_dedup`, `_compute_pair_results_no_dedup`, `_pair_canonical_key` | §1.2 | list branch of `cosSimExpTens`, `localScalarPairDispatch`, `internal.pairCanonicalKey` — reached |
| `_r1_broadcast_fast`, `_self_ip_cache_key`, `_self_ip_memoised` | §1.2b, §1.3 | `localR1BroadcastFast`, `internal.selfIpKey`, `internal.selfIpMemoised` — reached |
| `_cos_sim_raw_ma_scalar`, `_cos_sim_raw_ma_broadcast`, `_try_sweep_reduction`, `_cos_sim_raw_single_multiset_scalar`, `_cos_sim_raw_single_multiset_batch`, `_cos_sim_pair_core` | §1.1 | raw branches of `cosSimExpTens`, `localCosSimBatchedRaw` — reached; no `_try_sweep_reduction` twin (A-14) |
| `cos_sim_exp_tens_raw`, `batch_cos_sim_exp_tens` | deprecated shims (reached when called) | `batchCosSimExpTens` — shim |
| `_cos_sim_exp_tens_ma` | §1.3 | `localCosSimMA` — reached |
| `_cos_sim_exp_tens_ma_pairwise`, `_cos_sim_exp_tens_ma_centres`, `_ip_core_ma`, `_ip_r1_direct`, `_ip_full_ma`, `_ma_log_kernel`, `_trunc_log_kernel_exp`, `_gram_is_accurate_enough`, `_gram_quadratic_form`, `_ip_via_helper` | §1.5a, §1.5b | `ipCoreMA`, `localIpR1Direct`, `ipFullMA`, `maLogKernel`, `localGramAccurateEnough`, `localGramQuadraticForm` — reached; `_ip_via_helper` has no twin on the cosine path (A-15) |
| `_cos_sim_exp_tens_ma_orbit` | §1.5c | `localCosSimMAOrbit` — reached |
| `_cos_sim_exp_tens_ma_factored`, `_ma_factored_ip_supported`, `_ma_ip_factored`, `_ma_ip_factor_dense`, `_ma_ip_factor_nested_culled`, `_nested_factor_cullable` | §1.5d | none (A-13) |
| `_try_nested_contract`, `_try_nested_contract_ma`, `_nested_attr_plan`, `_nested_attr_route`, `_nested_admissible_routes`, `_nested_prefers_enumeration`, `_nested_enumeration_admissible`, `_nested_self_ip_skip_flags`, `_nested_attr_matrix` | §1.8, §1.9 | `internal.nestedContract` (+ `nestedContractMA`, `nestedAttrPlan`, `nestedAttrRoute`, `nestedAdmissibleRoutes`, `nestedPrefersEnumeration`, `nestedEnumerationAdmissible`, `selfIpSkipFlags`, `nestedAttrMatrices`, `makeQuadrature`, `localCheckForcedMode`) — reached |
| `_rel_contract_cheaper` | **orphan** (docstring "Superseded") | none — not a gap |
| `_orbit_inner_abs`, `_orbit_inner_rel` | **orphan** (re-export only) | `mobius.orbitInnerAbsSingleMultiset` / `orbitInnerRelSingleMultiset` — **live** on Rényi-2 single-multiset; Python routes single-multiset Rényi-2 through `_ma_per_attr_inner_matrix` (same measure except A-5) |
| `_inner_product_direct_abs`, `_build_ordered_r_tuples`, `_ma_has_nan`, `_attr_value_range` | **orphan** | `mobius.innerProductDirectAbsSingleMultiset` — tests only; not a gap |

### 11.2 Python — `_tensor/dispatch.py`

| Routine | Reached from | Twin |
|---|---|---|
| `_select_ma_inner_product_method`, `_predict_pairwise_kernel_size`, `_rel_route_cost_ms`, `_predict_orbit_cost_ms`, `_guard_forced_bulger_feasible_ma`, `_orbit_sigma_over_p_threshold` | §1.4 | `internal.selectMaInnerProductMethod`, `predictPairwiseKernelSize`, `relRouteCostMs`, `predictOrbitCostMs`, `guardForcedBulgerFeasible`, `relPerSigmaOverPThreshold` — reached |
| `_impossible_value_reason` | §1.3 post-hoc guard | `localOrbitIPsImpossible` — reached |
| `_compute_Q`, `_compute_Q_inner_blocks` | every log-kernel / closed-form leaf | `computeQaMA`, `qInnerBlocks`, `localComputeQFlat` — reached |
| `_select_ma_eval`, `_reject_ordered_for_mobius`, `_has_ordered_attr`, `_ma_eval_costs_ms`, `_ma_cost_constants`, `_centres_query_slope`, `_estimate_ma_joint_working_set_bytes` | §3.4 | `internal.selectMaEval`, `hasOrderedAttr`, `maEvalCostsMs`, `estimateMaJointWorkingSetBytes` — reached |
| `SingleImageInfeasibleError` | §1.4 rule 3, §3.4 rule 7 | `mpt:dispatch:singleImageInfeasible` — reached |
| `_ABS_PER_SIGMA_OVER_P_THRESHOLD` | build-time warning only (`density.py:80`) | `internal.absPerSigmaOverPThreshold` — diagnostic only; not a gap |
| `_orbit_ips_impossible`, `_estimate_centres_array_bytes`, `_pw_per_entry_ms` (+ `_PW_PER_ENTRY_MS_*`), `_warn_rel_per_all_image`, `_format_time`, `_PROBE_MIN_N_Q`, `_PROBE_K_IP_TARGET`, `_PROBE_TIME_CACHE`, `_PROBE_IP_MOBIUS_DECISION_MARGIN`, `_PRESCREEN_IP_DOMINANCE`, `_ORBIT_IP_FIXED_OVERHEAD` | **orphan** / vestigial (B-17, D-1) | `internal.warnRelPerAllImage`, `timeRepeated` — dead / bench only; not a gap |
| `_probe_eval_path`, `_probe_ip_path` | named in the docstring, **not defined** | none |

### 11.3 Python — `_tensor/_mobius_inner.py`, `_mobius.py`, `_wrapped_kernel.py`, `_kernel.py`, `_ma_eval_orbit.py`

| Routine | Reached from | Twin |
|---|---|---|
| `_ma_per_attr_inner_matrix`, `_orbit_safe_submatrix_sparse`, `_build_sparse_kernel_abs`, `_trunc_kernel_exp` | §1.6.2 | `mobius.maPerAttrInnerMatrix`, `localSafeSafeOrbit`, `localSafeSafeOrbitSparse`, `localBuildSparseKernelAbs`, `localR1ZeroPad` — reached |
| `_rel_inner_batched`, `_spectral_rel_inner_matrix`, `_rel_per_inner_sparse`, `_rel_per_image_count`, `_rel_window_margin` | §1.6.4 | `mobius.relInnerBatched`, `spectralRelInnerMatrix`, `localRelPerInnerSparse`, `internal.relPerImageCount`, `relWindowMargin` — reached |
| `_ma_rel_attr_prefers_centres`, `_predicted_centres_wall_ns`, `_predicted_grid_wall_ns` | §1.6.1 | `mobius.maRelAttrPrefersCentres` — reached |
| `_closed_form_attr_centres`, `_closed_form_attr_matrix_from`, `_comb_side_restriction` | §1.6.3 | `mobius.closedFormAttrCentres`, `closedFormAttrMatrixFrom`, `localCombRestriction` — reached |
| `_batched_direct_enum_abs`, `_pack_nan_top` | **orphan** (re-export only) | `localBatchedDirectEnumAbsSingleMultiset`, `localPackNanTop`, `localFillDirectEnumGroups` — dead (B-16); shared |
| `inner_product_orbit_grid`, `inner_product_orbit_pw_batched`, `inner_product_orbit_sparse`, `get_set_partitions_with_mobius`, `mobius_partition_combine` | §1.6, §1.8.5, §3.5 | `mobius.innerProductOrbitGrid`, `innerProductOrbitPwBatched`, `innerProductOrbitSparse`, `mobiusPartitionCombine` — reached |
| `eval_orbit_abs`, `eval_orbit_rel`, `_eval_orbit_rel_fourier`, `_factored_worthwhile`, `_lagrange6_*`, `total_mass_abs`, `total_mass_rel` | §3.5, §4 | `mobius.evalOrbitAbs`, `evalOrbitRel`, `localEvalOrbitRelFourier`, `factoredWorthwhile`, `totalMassAbs`, `totalMassRel` — reached |
| `eval_orbit_rel(factored=…)` explicit values | internal only (never passed by the entry point) | `evalOrbitRel` `'factored'` on/off — never passed; shared |
| `eval_ma_orbit`, `_ma_join_query` | §3.3 | `mobius.evalMaOrbit` — reached |
| `wrapped_gaussian_1d`, `_image_count_L`, `_fourier_count_M`, `_prefer_fourier` | every abs-per full-image branch | `internal.wrappedGaussian1d`, `wrappedKernelImageCount`, `wrappedKernelFourierCount`, `wrappedKernelPreferFourier` — reached |
| `gaussian_kernel_sum`, `_truncated_kernel_sum_1d_vectorised`, `_truncated_kernel_sum`, `_truncated_kernel_sum_1d_circular`, `_exact_kernel_sum`, `_eval_chunk`, `truncation_radius`, `truncation_ip_sqdist` | §1.5a `[Py]`, §3.5, §3.6 | `internal.gaussianKernelSum`, `localTruncatedKernelSum1D`, `localTruncatedKernelSum`, `localTruncatedKernelSum1DCircular`, `localExactKernelSum`, `evalChunk` — reached |
| `_mobius.inner_product_orbit` (dense per-pair) | not on any cosine path | `mobius.innerProductOrbit` — reached only from Rényi-2 single-multiset abs (`orbitInnerAbsSingleMultiset`) |

### 11.4 Python — `_tensor/_nested_contraction.py`, `_nested_cost.py`, `_orbit_cost.py`

| Routine | Reached from | Twin |
|---|---|---|
| `nested_attr_matrix`, `_nested_attr_matrix_impl`, `orbit_guard_scope`, `_shared_template_matrix`, `_all_shared_templates`, `_tau_window`, `_contract`, `_subtree_overlaps`, `_leaf_overlaps`, `_combine_pair`, `_combine_orbit`, `_combine_chunked`, `_combine`, `_enum_work`, `build_recipe`, `_orbit_eligible`, `auto_ntau_default`, `auto_taus_line` | §1.8 | `nestedAttrInnerMatrix`, `tripSum`, `pairValuesBatched`, `orbitGuard`, `ipRelNonperFactored`, `sharedLeafTemplate`, `tauWindow`, `executeRecipe`, `contractNode`, `combinePair`, `combineOrbit`, `combineChunked`, `combine`, `buildRecipe`, `orbitEligible`, `internal.autoNtauDefault`, `makeQuadrature` — reached |
| `make_quadrature`, `nested_ip`, `_ip_absolute`, `_ip_rel_nonper`, `_ip_rel_nonper_factored`, `_ip_rel_nonper_generic`, `_theta_truncation_L` | **orphan** (tests only) | `makeQuadrature` — **live**; `nestedIp`/`ipRelNonperFactored` — **live** as the per-pair fallback when `batchableMode` is false; same measure — a leaf-placement difference, not a gap |
| `price_nested_attr`, `nested_attr_terms`, `nested_route_cost_ms`, `_nested_cost_key`, `nested_centres_working_set_bytes`, `select_nested_method`, `_flat_companion_cost_ms`, `predict_nested_pairwise_kernel_size` | §1.8.2, §1.8.3 | `internal.nestedCost` (`priceNestedAttr`, `localPriceAttr`, `selectNestedMethod`, `localSelect`), `internal.lastNestedCosts` — reached |
| `orbit_cost_model`, `orbit_cost_log_ratio` | §1.8.5 | `internal.orbitCostModel` — reached |
| `_tensor/_centres_inner.py` (`centres_inner_product`) | **orphan** module (tests only) | none — not a gap |

### 11.5 Python — `_tensor/sweep.py`, `eval.py`, `entropy.py`, `windowed.py`, `explain.py`

| Routine | Reached from | Twin |
|---|---|---|
| `sweep_cos_sim_exp_tens`, `sweep_eligibility`, `_rel_per_measure_admissible`, `orbit_sweep_supported`, `_choose_sweep_route`, `_finalise_orbit_sweep`, `_orbit_sweep`, `_orbit_attr_matrix_sweep`, `_orbit_self_ip`, `_build_mixture`, `_splittable`, `_evaluate_mixture`, `_evaluate_dense`, `_self_ip`, `TranslatedSweep` | §2, §1.1a | `sweepCosSimExpTens`, `localCheckEligible`, `localOrbitSweepSupported`, `localChooseSweepRoute`, `localOrbitSweep`, `localOrbitAttrMatrixSweep`, `localOrbitSelfIp`, `localBuildMixture`, `localEvaluateMixture`, `localEvaluateDense`, `localSelfIp` — reached; `TranslatedSweep` has no twin |
| `SweepNotEligible` | **orphan** (never raised) | MATLAB raises `sweepCosSimExpTens:*` directly — cosmetic |
| `eval_exp_tens`, `_eval_exp_tens_scalar`, `_eval_exp_tens_density_list`, `_eval_exp_tens_raw_ma_scalar`, `_eval_exp_tens_raw_single_multiset_scalar`, `_eval_exp_tens_raw_single_multiset_batch`, `_eval_exp_tens_ma`, `_ma_eval_factored`, `_ma_eval_full`, `_ma_value_tables`, `_tuple_values_repeat`, `_eval_core`, `_truncated_kernel_sum_culled`, `_eval_full`, `_distinct_value_table`, `_ma_eval_normalize`, `_maybe_warn_eval_time`, `eval_exp_tens_raw` (shim) | §3 | `evalExpTens`, `localMaSkinnyDispatch`, `localEvalMA`, `localMaEvalFactored`, `maetEvalFull`, `localMaNormaliseSkinny`, `localEvalBatchedRaw`, `internal.tupleValuesRepeat` — reached; `_eval_core` has no direct twin (MATLAB uses `gaussianKernelSum`) |
| `entropy_exp_tens`, `_canonicalize_method`, `_entropy_exp_tens_shannon_dispatch`, `_entropy_exp_tens_scalar`, `_entropy_exp_tens_density_list`, `_entropy_exp_tens_raw_single_multiset_batch`, `_entropy_exp_tens_ma`, `_cell_masses_ma_absolute`, `_phi_diff_axis`, `_phi_diff_axis_periodic`, `_contract_cell_axes`, `_entropy_exp_tens_differential_dispatch`, `_differential_adaptive`, `_diff_spans_ma`, `_entropy_exp_tens_renyi2_dispatch`, `_renyi2_exp_tens_ma`, `_renyi2_per_attr_numerical`, `_renyi2_finalise`, `_resolve_density`, `_raise_if_any_sigma_zero`, `_looks_like_ma_p` | §4 | `entropyExpTens`, `localEntropySingleMultiset`, `localEntropyMA`, `localPhiDiffAxisPeriodic`, `localDifferentialAdaptive`, `localRenyi2SingleMultiset`, `localRenyi2MA`, `renyi2PerAttrNumerical`, `localCanonicalizeMethod` — reached; `localRenyi2SingleMultiset` has no Python twin (A-5) |
| `windowed_similarity`, `_ws_multi`, `_ws_single`, `windowed_entropy` | §5.1 | `windowedSimilarity` — reached (no `windowedEntropy` twin noted) |
| `explain_dispatch` | §6 | `explainDispatch` — reached |

### 11.6 MATLAB — orphans and dead code with twin status

| Routine | Status | Python twin |
|---|---|---|
| `mobius.innerProductDirectAbsSingleMultiset` | tests only | `_inner_product_direct_abs` — orphan; not a gap |
| `localFillDirectEnumGroups`, `localPackNanTop`, `localBatchedDirectEnumAbsSingleMultiset` (`maPerAttrInnerMatrix.m:471-540`) | dead (`safe_x_mask = true(1, N)`) | `_batched_direct_enum_abs`, `_pack_nan_top` — orphans; shared B-16 |
| `mobius.contract` | tests only (runtime uses `executeRecipe`/`contractNode`) | `_contract` — live; naming only |
| `internal.timeRepeated` | bench only | none (no probe in either) |
| `internal.nestedCostOverride` | test hook, read at `nestedCost.m:341` | none |
| `internal.absPerSigmaOverPThreshold` | diagnostic (`maybeWarnAbsPerSingleImage`) | `_ABS_PER_SIGMA_OVER_P_THRESHOLD` — build-time warning only |
| `mobius.innerProductOrbit`, `orbitInnerAbsSingleMultiset`, `orbitInnerRelSingleMultiset` | live on Rényi-2 single-multiset only (`entropyExpTens.m:2066-2082`); not on any cosine path | `_orbit_inner_abs`/`_orbit_inner_rel` — orphans; Python takes the MA loop |
| second `localMaEvalFactored` call in `localEvalMA` (`evalExpTens.m:940`) | always `[]` (predicate already failed upstream, or kernel covariance) | Python's joint branch has no second attempt — cosmetic |
| `localComputeQBlocks` (`closedFormAttrMatrixFrom.m:180`); inner-unit branch of `localReducedCentresFromValues` (`closedFormAttrCentres.m:257-273`) | no shipped caller builds a centres bundle with block size ≥ 2 | `bs >= 2` branch of `_closed_form_attr_matrix_from` — equally unreachable; shared |
| `mobius.evalOrbitRel` `'factored'` ∈ {on, off} | never passed by `evalExpTens`/`evalMaOrbit` | `factored=` — internal only; shared |
| `internal.relRouteCostMs('centres')` | reached inside `predictOrbitCostMs` — not an orphan (listed for completeness) | `_rel_route_cost_ms('centres')` — reached |
| `localMaEvalFactored` retry, `evalMaOrbit` on the joint MA path (`evalExpTens.m:908-930`) | the joint-path Möbius branch is reachable only with a kernel covariance; whether the whitened query matches the stored attributes is unverified (M-16) | same open question in Python (`eval.py:421`) — shared, unverified |

### 11.7 Routines with no twin at all

| Routine | Language | Nature |
|---|---|---|
| `_try_sweep_reduction`, `TranslatedSweep`, the `'sweep'` memo | Python | speed (A-14, A-16) |
| `_cos_sim_exp_tens_ma_factored` and its `_ma_ip_factor*` family | Python | route (A-13) |
| flat selector rule 4 (working-set guard) | Python | route (A-12) |
| `_nested_centres_cache`, density-list dedup and calibration probe | Python | speed (A-16) |
| `_eval_core` (`_truncated_kernel_sum_culled`, `_eval_full`) | Python | leaf placement only (MATLAB reaches the same numbers through `gaussianKernelSum`) |
| `localRenyi2SingleMultiset` and the r = 1 relative convention | MATLAB | value (A-5) |
| single-multiset eval non-finite fallback (`evalExpTens.m:557-571`) | MATLAB | guard (A-7) |
| `localR1BroadcastFast` on the raw-MA broadcast form | MATLAB | speed (A-14) |
| `dispatchMemBudget` (memory-dependent budget) | MATLAB | constants-only (Python uses a fixed 4 GiB) |

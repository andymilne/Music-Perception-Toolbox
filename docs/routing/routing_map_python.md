# Routing map — Python half of the Music Perception Toolbox (`mpt`)

Source root: `/home/claude/mpt/python/mpt`. All `file:line` references are relative to that root and were read from the source on 2026-09-04. Predicates are quoted as written in the code (variable names as in the code). "ts" = `truncation_sigmas` after `resolve_truncation_sigmas` (`None` → default 6.0; `inf` → `accuracy_floor_sigmas()` = 7.4338). `thr(ts)` = `_orbit_sigma_over_p_threshold(ts)` (dispatch.py:1030; = 0.03 at the default, 0.05 ceiling at ts ≤ 4).

Legend: `NODE` = decision point, `LEAF` = routine that does arithmetic, `OVERRIDE` = user value that bypasses a decision, `ERROR` = raise, `GUARD` = admissibility/feasibility check, `MEMO` = self-inner-product cache read/write.

---

## 1. ENTRY `cos_sim_exp_tens(*args, mode, dedup, spectrum, precision, method, normalize/normalise, cancellation_threshold, truncation_sigmas, kernel_precision, verbose)` — `_tensor/cosine.py:159`

- [parse] `normalize` ∈ {`'cosine'`, `'oneSidedDenom'`} (case-insensitive; `'normalise'` keyword alias) via `_canonical_normalize` (cosine.py:93). ERROR `TypeError` when both `normalize` and `normalise` given (cosine.py:326); `ValueError` on any other value (cosine.py:110).
- [parse] `method`: not validated at the entry; validated in `_cos_sim_exp_tens_ma` (cosine.py:1467) as ∈ {`'auto'`, `'bulger'`, `'centres'`, `'mobius'`, `'contract'`, `'factored'`}. Retired synonyms: `'direct'` (mentioned in dispatch.py:963 comment as retired; **not accepted** — raises `ValueError`). `'factored'` is accepted by the parser but is **undocumented** in the docstring (cosine.py:243). `cancellation_threshold` is accepted and ignored (cosine.py:284; it is threaded but never read on any route).
- ERROR `TypeError` when `len(args) < 2` (cosine.py:320).

### 1.1 Input-form dispatch (cosine.py:335–634)

- IF `isinstance(a, (MaetDensity, WindowedMaetDensity))` or `a` is a list/tuple whose `a[0]` is a density, or an empty list, or an object-dtype ndarray → NODE **density path** (cosine.py:354)
  - ERROR `TypeError` if `len(args) != 2`, or `spectrum is not None`, or `precision is not None` (cosine.py:355–367).
  - → `_cos_sim_density_path` (cosine.py:1047) — see §1.2.
- ELIF `_looks_like_multi_attr(a)` (build.py:428) → NODE **raw multi-attribute** (cosine.py:383)
  - ERROR `TypeError` if `spectrum`/`precision` given or `mode != "auto"` (cosine.py:384–396); if second operand not MA (cosine.py:406); if both operands are lists (`a_is_list and b_is_list`, cosine.py:419).
  - IF `sigma_vec_has_kernel_cov(args[4])` → whiten both operands with `_resolve_aniso_ma`/`whiten_p_attr`, replace `sigma_vec` by the resolved one (isotropic 1) (cosine.py:433–458). Not a routing change downstream; noted because it changes the arithmetic's inputs.
  - IF `not a_is_list and not b_is_list` → ERROR if `len(args) not in (9, 10)` → `_cos_sim_raw_ma_scalar` (cosine.py:1208): `build_exp_tens` ×2 → `_cos_sim_pair_core` (§1.3).
  - ELSE (exactly one list) → `_cos_sim_raw_ma_broadcast` (cosine.py:1239):
    - builds the scalar side once; → NODE `_try_sweep_reduction` (cosine.py:1333):
      - IF `not isinstance(list_pAttr, TranslatedSweep)` → `None` (fall through).
      - IF `method not in ("auto",)` → `None` (OVERRIDE: any forced method bypasses the sweep reduction, cosine.py:1354).
      - IF `off.size == 0 or not np.all(np.isfinite(off))` → `None`.
      - IF `not scalar_first and normalize != "cosine"` → `None` (cosine.py:1372).
      - IF `sweep_eligibility(dens_x, dens_y, off_use)` not ok → `None`.
      - ELSE → **ENTRY 2** `sweep_cos_sim_exp_tens(dens_x, dens_y, off_use, normalize, truncation_sigmas, kernel_precision)` with `method='auto'` (cosine.py:1379).
    - fallback: per-entry loop `build_exp_tens` + `_cos_sim_pair_core` (cosine.py:1304–1329).
- ELSE → NODE **raw single-multiset** (cosine.py:504)
  - ERROR `TypeError` if `len(args) not in (9, 10)`; if `P1`/`P2` not numeric; if `ndim > 2` (cosine.py:504–533).
  - IF `is_kernel_cov(args[4])` → ERROR if `spectrum`; `check_aniso_constraints`; whiten; `sigma := 1.0` (cosine.py:539–562).
  - IF `a_arr.ndim == 2 or b_arr.ndim == 2` → **batched** (cosine.py:565): broadcast a single-row operand; ERROR `ValueError` when `M1 != M2` and neither is 1 (cosine.py:595) → `_cos_sim_raw_single_multiset_batch` (cosine.py:3919):
    - ERROR `NotImplementedError` when `(is_sym is not None) and (not np.all(is_sym)) and r > 1` (ordered batched dedup, cosine.py:3992).
    - Phase 1: rows with `len(pa_valid) < r or len(pb_valid) < r` → NaN output (skipped) (cosine.py:4049); `_pair_canonical_key` dedup of individual sets.
    - Phase 2: `build_exp_tens` per unique canonical set (with `add_spectra` if `spectrum`).
    - Phase 3: IF `n_valid == 0` → all-NaN; ELSE recursive `cos_sim_exp_tens(list_a, list_b, mode="pairwise", dedup=dedup, method=method, ...)` (cosine.py:4125) → §1.2.
  - ELSE (both 1-D) → ERROR if `precision`/`mode != "auto"` → `_cos_sim_raw_single_multiset_scalar` (cosine.py:1156): `add_spectra` if `spectrum`; `build_exp_tens` ×2 → `_cos_sim_pair_core`.
- Deprecated shims: `cos_sim_exp_tens_raw` (cosine.py:3359) → `cos_sim_exp_tens(...)`; `batch_cos_sim_exp_tens` (cosine.py:4153) → `_cos_sim_raw_single_multiset_batch` directly with `method='auto'`.

### 1.2 `_cos_sim_density_path(dens_x, dens_y, mode, dedup, method, ...)` — cosine.py:1047

- `_normalize_density_input` on each side (dispatch.py:42; ERROR `TypeError` on non-density elements).
- IF `is_x_scalar and is_y_scalar` → `_cos_sim_pair_core` (§1.3).
- ELIF one side scalar → pairs = broadcast; empty list → `np.empty((0,))`.
- ELSE list-vs-list: empty → shape by `_resolve_list_list_mode` (dispatch.py:78; `mode='auto'` with `m != n` → `ValueError`, caught and treated as cartesian only in the empty case, cosine.py:1091–1097); `'pairwise'` requires `m == n` else `ValueError`.
- IF `(is_x_scalar != is_y_scalar) and method in ("auto", "bulger")` → NODE `_r1_broadcast_fast` (cosine.py:638) — **all-r = 1 fast path**:
  - returns `None` (→ ordinary loops) unless every density is a flat `MaetDensity` (not Windowed), has no kernel covariance, no nested attribute, `A > 0`, all `int(r) == 1`, all `_inner_r_vec == 0`, and every entry matches the shared operand in `n_attrs, r, sigma, is_rel, is_per, period[per], wrap` (cosine.py:672–704).
  - LEAF: one kernel pass over concatenated comb-side columns (cosine.py:759–787): per attribute `is_rel[a]` → skipped (vanishing form); `is_per[a] and wrap_a == 'full-image'` → `wrapped_gaussian_1d(..., exponent_denominator=4)` (log θ); `is_per[a]` (single-image) → nearest-image wrap then `-d²/(4σ²)`; else `-d²/(4σ²)`. Per-column threshold `-ts²/2 - log(n_terms)`.
  - MEMO key `_self_ip_cache_key("bulger", truncation_sigmas, kernel_precision)` = `('bulger', ts, kp, None)`: read for shared and each entry; written (on both original and pruned objects) when computed via `_ip_core_ma` (cosine.py:795–828). `<X,X>` skipped when `normalize != 'cosine'` and X is the shared operand (`need_xx_shared`), or when X is an entry (`need_self_entry`).
  - OVERRIDE: `method='centres'/'mobius'/'contract'/'factored'` skips this fast path entirely.
- IF `dedup and _all_single_multiset_pairs(pairs)` → `_compute_pair_results_with_dedup` (cosine.py:850): `_pair_canonical_key` dedup; calibration probe (warm-up + ≤5 timed `_cos_sim_pair_core` calls when `verbose and n_unique >= 2`, cosine.py:901–938 — these are real computations that seed memos); then `_cos_sim_pair_core` per unique pair.
- ELSE → `_compute_pair_results_no_dedup` (cosine.py:960): `_cos_sim_pair_core` per pair.

### 1.3 `_cos_sim_pair_core(dens_x, dens_y, method, ...)` — cosine.py:988

- ERROR `TypeError` if either is `WindowedMaetDensity` (cosine.py:1006) — directs to `windowed_tensor_similarity`.
- ERROR `ValueError` if `density_has_kernel_cov` on either and `not density_kernel_covs_compatible` (cosine.py:1016).
- ERROR `TypeError` if types mismatch / not `MaetDensity` (cosine.py:1024, 1039).
- → `_cos_sim_exp_tens_ma` (§1.4).

### 1.4 `_cos_sim_exp_tens_ma(dens_x, dens_y, method, normalize, ...)` — cosine.py:1406 (the flat MA dispatcher)

- `dens_x = dens_x.pruned(); dens_y = dens_y.pruned()` (cosine.py:1435; memoised on the density, density.py:288). All memo reads/writes below are on the **pruned** objects.
- IF `dens_x.n == 0 or dens_y.n == 0` → return `0.0` (cosine.py:1445).
- GUARDS (ERROR `ValueError`): `n_attrs`, `r`, `sigma`, `is_rel`, `is_per` must be equal; `period` equal on `is_per` attributes (cosine.py:1449–1465).
- [parse] `method not in ("auto","bulger","centres","mobius","contract","factored")` → ERROR `ValueError` (cosine.py:1467).
- OVERRIDE `method == "factored"` (cosine.py:1479):
  - GUARD `_ma_factored_ip_supported` (cosine.py:3672): False if any kernel cov or any `is_rel[a] and is_per[a]` → ERROR `ValueError`.
  - → LEAF `_cos_sim_exp_tens_ma_factored` (cosine.py:3691) → `_ma_ip_factored` ×3 (xy, xx, yy; **no memo, `<X,X>` always computed**) (cosine.py:3579):
    - per attribute `kind[a]`: `spec is None` → `"dense"` if `is_rel[a] and is_per[a]` else `"flat"`; nested and `_nested_factor_cullable(spec, is_per, a)` (cosine.py:3447: `L >= 2` and (`rel_unit is None` or (`rel_unit == 0 and not is_per`))) → `"nested_cull"`; else `"dense"`.
    - `"flat"` → LEAF `_ip_via_helper` (cosine.py:3713) → `gaussian_kernel_sum` (§3.6) with `n_terms = M_u * M_v`, `wrap=wrap_a`.
    - `"nested_cull"` → LEAF `_ma_ip_factor_nested_culled` (cosine.py:3471): leaf IPs via `_ip_via_helper`; upper levels via `_combine_pair(..., use_orbit=_orbit_eligible(...) both sides)` (§1.9).
    - `"dense"` → LEAF `_ma_ip_factor_dense` (cosine.py:3537): `r_in > 0` → `_compute_Q_inner_blocks`; abs-per: `wrap_a == 'single-image'` → nearest-image `_compute_Q`, else `wrapped_gaussian_1d(..., 4).prod`; else `_compute_Q`.
    - early exit `if prod == 0.0: break` per event pair (cosine.py:3666).
  - Bypasses: the selector, the ordered-attribute rule, the nested contraction, the post-hoc guard, and all memoisation. Still subject to: structural guards above and `_ma_factored_ip_supported`.
- Pre-computation for the selector (cosine.py:1495–1558): `sop_max` = max σ/P over attributes with `is_rel[a] and is_per[a] and period > 0`; `any_per`, `any_rel_nonper`, `any_rel_per`; `k_vec`, `k_vec_y` = per-attribute value-row counts; `rel_vec`; `nu_vec[a]` = 1 for non-rel or `r < 2`; `auto_ntau_default(period, sigma)` (_nested_contraction.py:658) for rel-per; `max(64, ceil(max(span,1)/sigma*10))` with `span = (max−min)_x + (max−min)_y + 2*_rel_window_margin(default ts)*sigma` for rel-nonper.
- `nested_any` = any `nested[a] is not None` on either density (cosine.py:1564).
- `wrap_vec_x = dens_x.wrap` (authoritative; y's wrap ignored, cosine.py:1575).
- skip flags (cosine.py:1585): `need_xx = (normalize == "cosine")`; `skip_xx = (not need_xx) or _self_ip_memoised(dens_x)`; `skip_yy = _self_ip_memoised(dens_y)` — `_self_ip_memoised` (cosine.py:3116) is True iff any cache key with `k[0] in _SELF_IP_ROUTES = ("bulger","centres","mobius","contract","contract_ma")` exists (the `'sweep'` key is deliberately excluded, cosine.py:3113).
- NODE `chosen = _select_ma_inner_product_method(...)` (dispatch.py:513) — §1.5; called with `guard_forced_bulger = not nested_any`, `sym_vec = dens_x.is_sym`.
- IF `ordered_any` = any `(~is_sym & (r_vec > 1))` on either density → `chosen = "bulger"` (cosine.py:1612–1617). **This overrides the user's `method='mobius'`/`'centres'` silently on the flat path** (see Suspicious).
- IF `method == "contract" and not nested_any` → ERROR `ValueError` (cosine.py:1626).
- IF `nested_any` (cosine.py:1631):
  - IF `method in ("auto", "contract", "mobius", "centres")` → NODE `_try_nested_contract(dens_x, dens_y, normalize, verbose, force=(method != "auto"), method_name=method, force_route=("centres" if method == "centres" else None))` (§1.8):
    - triple returned → `_finalise_normalisation` and return.
    - `None` (only possible under `method == 'auto'`) → `chosen = "bulger"`.
  - ELSE (`method == 'bulger'`) → `chosen = "bulger"`.
  - Note: on a nested density the selector's output (`mobius`/`centres`/`bulger`) is discarded; the selector still ran (its working-set guard and wrap override cannot raise here except the mixed-wrap `ValueError`).
- Dispatch on `chosen` (cosine.py:1681–1738):
  - `"mobius"` → LEAF `_cos_sim_exp_tens_ma_orbit(dens_x, dens_y, user_forced_mobius=(method == "mobius"), truncation_sigmas, need_xx)` (§1.6).
    - POST-HOC GUARD: IF `get_default("post_hoc_guards")` → `_impossible_value_reason(ip_xy, ip_xx, ip_yy)` (dispatch.py:138): non-finite IP; negative `<X,X>`/`<Y,Y>`; `abs(ip_xy) > 1.000001 * sqrt(ip_xx*ip_yy)`. IF not None → `RuntimeWarning`; purge every `_self_ip_cache` key with `k[0] == "mobius"` on both densities; fallback → `_cos_sim_exp_tens_ma_pairwise` (cosine.py:1701–1724). Direction of fallback: Möbius → Bulger. OVERRIDE `post_hoc_guards=False` skips the check.
  - `"centres"` → LEAF `_cos_sim_exp_tens_ma_centres` (cosine.py:3200): `_ip_core_ma(u_perm_x, w_j_x, n_jx, u_perm_y, w_j_y, n_jy, ...)` — perm side on **both** sides (unrestricted enumeration). MEMO key `('centres', ts, kp, None)`: read for xx (if `need_xx`) and yy; written when computed.
  - else (`"bulger"`) → LEAF `_cos_sim_exp_tens_ma_pairwise` (cosine.py:3269): `_ip_core_ma(u_perm_x, w_j_x, n_jx, v_comb_y, wv_comb_y, n_ky, ...)` perm × comb. MEMO key `('bulger', ts, kp, None)`: read for xx (if cached) / computed if `need_xx`; read/computed for yy; written when computed.
- `_finalise_normalisation` (cosine.py:115): `'cosine'` → `ip_xy / sqrt(max(ip_xx*ip_yy, 0))`; `'oneSidedDenom'` → `ip_xy / ip_yy`; denominator 0 → NaN; ERROR `ValueError` if `ip_xx is None` under `'cosine'`.

### 1.5 `_select_ma_inner_product_method(r_vec, k_vec, A, N_x, N_y, any_per, any_rel_nonper, any_rel_per, sigma_over_P_max, user_method, rel_vec, nu_vec, guard_forced_bulger, wrap_vec, k_vec_y, sym_vec, truncation_sigmas, return_costs, skip_xx, skip_yy)` — dispatch.py:513 (flat MA selector)

Rules in order (each `return` ends the selector):
1. OVERRIDE `user_method != 'auto'` → return `user_method` unchanged (dispatch.py:609). Bypasses every guard below, including the `_guard_forced_bulger_feasible_ma` memory guard and the wrap/measure override. (`'centres'`, `'mobius'`, `'bulger'`, `'contract'`, `'factored'` all pass through; `'contract'`/`'factored'` never reach here on the flat path in practice because they are handled earlier.)
2. `r_max = max(r_vec)` (1 if `A == 0`). IF `r_max <= 1` → `'bulger'` (dispatch.py:613).
3. IF `r_max > _ORBIT_R_MAX_SHIPPED` (= 8, dispatch.py:976) → IF `guard_forced_bulger` → GUARD `_guard_forced_bulger_feasible_ma` (dispatch.py:1480): `pair_bytes = nj_x * nj_y * 8 > _CENTRES_PROBE_MEM_BUDGET` (4 GiB, dispatch.py:1192) → ERROR `SingleImageInfeasibleError` (subclass of `MemoryError`, dispatch.py:1158). Then → `'bulger'` (dispatch.py:616–624).
4. Working-set guard (dispatch.py:642–666): IF `A > 0 and not (any_rel_per and sigma_over_P_max > thr(ts))`: `n_J_max = max(N_x*tuples_x, N_y*tuples_y)` with `tuples = ∏_a f_a*C(K_a, r_a)`, `f_a = r_a!` if `sym[a]` else 1; IF `n_J_max * (2*max(dim_sum,1)) * 8 > _CENTRES_WORKING_SET_SOFT_BUDGET` (256 MiB, dispatch.py:1213) → `'mobius'`.
5. Wrap override (dispatch.py:675–695): IF `wrap_vec is not None and any_rel_per and rel_vec is not None`: `wants_single` = any rel attribute with `wrap == 'single-image'`; `wants_full` = any rel attribute with `wrap == 'full-image'` (note: computed over **all** rel attributes, not only rel-per ones). IF `wants_single and wants_full` → ERROR `ValueError("Mixed rel-per wrap ...")`. IF `sigma_over_P_max > thr(ts)`: `wants_single` → `'bulger'`; `wants_full` → `'mobius'`.
6. Cost comparison (dispatch.py:707–741): `pw_size = _predict_pairwise_kernel_size(...)` (dispatch.py:356; `inf` if any `K < r`; sums `perm_x*comb_y [+ perm_x*comb_x unless skip_xx] [+ perm_y*comb_y unless skip_yy]`); `pw_cost_ms = _rel_route_cost_ms("bulger", r_max, pw_size)` (dispatch.py:350: `exp(a)*max(term,1)^b` with `_REL_COST_LAW["bulger"][min(max(r,2),4)]`); `rel_vec`/`nu_vec` defaults if `None` (`nu = 2000`); `orbit_cost_ms = _predict_orbit_cost_ms(..., centres_ok=(sigma_over_P_max <= thr(ts)), ...)` (§1.5a); `chosen = 'bulger' if pw_cost_ms <= orbit_cost_ms else 'mobius'` (ties → bulger).
- There is **no timing probe** on this path (module docstring dispatch.py:15–20 and constants `_PROBE_*`, `_PRESCREEN_IP_DOMINANCE`, `_ORBIT_IP_FIXED_OVERHEAD` are vestigial — see Orphans).

#### 1.5a `_predict_orbit_cost_ms(r_vec, k_vec, A, N_x, N_y, rel_vec, nu_vec, centres_ok, k_vec_y, skip_xx, skip_yy)` — dispatch.py:417
- `n_matrices = 1 + [not skip_xx] + [not skip_yy]`; `pairs = N_x*N_y`.
- per attribute: IF `rel_vec[a] and r_a >= 2`: `per_pair = _rel_route_cost_ms("grid", r_a, pairs*nu_a*max(K_a,K_y_a)*(n_matrices/3))`; IF `centres_ok and K_a >= r_a and K_y_a >= r_a` → `per_pair = min(per_pair, _rel_route_cost_ms("centres", r_a, centres_size))` with `centres_size = pairs*(m_x*m_y [+ m_x² unless skip_xx] [+ m_y² unless skip_yy])`, `m = r!*C(K,r)`; floor `per_pair = max(per_pair, f + pm*n_matrices)` from `_ORBIT_REL_FLOOR_MS[min(max(r,2),4)]`.
  ELIF `r_a >= 2` → `_ORBIT_ABS_PER_ATTR_MS[r_a] * (n_matrices/3)`.
- Terms priced: tuple-pair entries (bulger, centres), node count × larger value count (grid).

### 1.6 `_cos_sim_exp_tens_ma_orbit(dens_x, dens_y, truncation_sigmas, need_xx, user_forced_mobius)` — cosine.py:2255 (Möbius route)

- `choices[a] = _ma_rel_attr_prefers_centres(Px, Py, sigma, r, is_rel, is_per, period, truncation_sigmas, user_forced_mobius)` for every attribute (cosine.py:2292) — §1.6a.
- MEMO key `('mobius', ts, None, choices)`; `compute_xx = need_xx and not xx_cached`; yy computed unless cached (cosine.py:2303–2310); values written after summing (cosine.py:2366–2378).
- per attribute: IF `choices[a]` → LEAF `_closed_form_attr_centres(dens, a)` (_mobius_inner.py:1785; memoised in `dens._nested_centres_cache[a]`) + `_closed_form_attr_matrix_from(cx, cy, truncation_sigmas, wrap_a)` (§1.6c). ELSE → LEAF `_ma_per_attr_inner_matrix(Px, Wx, Py, Wy, sigma, r_a, is_rel, is_per, period, truncation_sigmas, wrap=wrap_a)` (§1.6b).

#### 1.6a `_ma_rel_attr_prefers_centres(Px, Py, sigma, r_a, is_rel, is_per, period, truncation_sigmas, user_forced_mobius)` — _mobius_inner.py:1508
- IF `not is_rel or r_a < 2` → False.
- `blocked_by_measure = is_per and (sigma/period) > thr(ts)`; `empty_tuple_set = K_x < r_a or K_y < r_a`.
- `forced = get_default("rel_attr_route")` (default `'auto'`; _defaults.py:88):
  - IF `forced == "auto" and user_forced_mobius` → False (OVERRIDE `method='mobius'` pins the grid/Möbius route, _mobius_inner.py:1566).
  - IF `forced in ("mobius", "grid")` → False (`'grid'` = deprecated alias).
  - IF `forced == "centres"`: `blocked_by_measure` → ERROR `ValueError`; `empty_tuple_set` → ERROR `ValueError`; else True.
- IF `blocked_by_measure or empty_tuple_set` → False.
- cost: `span_or_period` = period (per) or `(max−min)_x + (max−min)_y`; `c_wall_ns = _predicted_centres_wall_ns(K_x, K_y, r_a, is_per)` (_mobius_inner.py:1466; `per_el = _CENTRES_NS_BASE + _CENTRES_NS_LIN*(r-1) [+ _CENTRES_NS_WRAP*(r-1)(r-2) if per]` × `(M_x*M_y + M_x² + M_y²)`); `g_wall_ns = _predicted_grid_wall_ns(K_x, r_a, sigma, span_or_period, is_per)` (_mobius_inner.py:1485; `_GRID_NS_FLOOR + g_op*n_u*K_x`, `n_u = auto_ntau_default` (per) or `max(64, ceil(max(span+2*margin*sigma,1)/sigma*10))`; `g_op = _GRID_NS_PER_OP[r]` or `_GRID_NS_PER_OP[4]*3^(r-4)`); return `c_wall_ns < g_wall_ns`.

#### 1.6b `_ma_per_attr_inner_matrix(Px, Wx, Py, Wy, sigma, r, is_rel, is_per, period, return_cancellation_ratio, truncation_sigmas, prune_zero_weight_events, wrap)` — _mobius_inner.py:219
- IF `prune_zero_weight_events and N_x > 0 and N_y > 0`: drop events whose weight column is all-zero/NaN; IF none survive on a side → zero matrix; else recurse with `prune_zero_weight_events=False` and scatter back (_mobius_inner.py:284–319).
- IF `r == 1` → LEAF direct kernel sum (_mobius_inner.py:322–385): `abs_per_full_image = is_per and str(wrap) == 'full-image'` → `wrapped_gaussian_1d(diffs, sigma, period, trunc_eff, exponent_denominator=4)`; else (per single-image) nearest-image wrap then `_trunc_kernel_exp(diffs², sigma, ts)`; chunked in `N_x` by `kernel_chunk_bytes_resolved()`; prefactor `sigma*sqrt(pi)`. (Note: `wrap` is consulted here even for `is_rel` at r = 1 — harmless, since the rel form vanishes elsewhere; here the abs kernel is used for rel r = 1 too.)
- IF `is_rel` (r ≥ 2) → `_zero_pad_nan` → LEAF `_rel_inner_batched` (§1.6d). **`wrap` is not passed** (relative-periodic here is always the all-image reading).
- ELSE (abs, r ≥ 2) (_mobius_inner.py:396–552):
  - sparse gate: `use_sparse = (not is_per) and r >= 2 and K_x_max*K_y_max >= _ORBIT_SPARSE_MIN_KERNEL (200_000)` and probe `K0.nnz <= _ORBIT_SPARSE_MAX_DENSITY (0.20) * K_x_max*K_y_max` (probe = first event pair via `_build_sparse_kernel_abs`) → LEAF `_orbit_safe_submatrix_sparse` → `_mobius.inner_product_orbit_sparse` per event pair.
  - else dense: chunk `N_xs` by memory; kernel = `wrapped_gaussian_1d(..., 4)` when `is_per and str(wrap) == 'full-image'`, else nearest-image (if per) + `_trunc_kernel_exp`; → LEAF `_mobius.inner_product_orbit_pw_batched(K_pairs, w_A, w_B, r, prefactor=(sigma*sqrt(pi))**r)`.

#### 1.6c `_closed_form_attr_matrix_from(cx, cy, truncation_sigmas, wrap_a)` — _mobius_inner.py:1843 (tuple-centres route)
- comb restriction (`cx[10]` from `_comb_side_restriction`, _mobius_inner.py:1680): declined (`None`) when `not _COMB_RESTRICTION_ENABLED` (True, :1617); flat `r_a < 2`; nested `_nested_orbit_mult < 2`; `n_k <= 0 or n_j != mult*n_k` (ordered attribute or unexpected tiling). Also declined at use when `comb_y is None or comb_y[3] != comb[3] or Cy.shape[1] != comb[3]*comb_y[0].shape[1]` (:1906). When applied: X restricted to comb side, `Wx *= mult`.
- `L_abs_per` computed (:1928–1937) but **never used** (dead variable; see Suspicious).
- per chunk of X tuples (`chunk = max(1, min(njx, 16_000_000 // (njy*max(d,1))))`):
  - IF `bs >= 2` (inner co-transposition block) → `_compute_Q_inner_blocks(D, bs, is_per, period, reduced=True)`.
  - ELIF `is_per and not is_rel`: IF `wrap_a == 'single-image' or _image_count_L(sigma, period, ts, 4) == 0` → nearest-image reduce + `_compute_Q` (L = 0 short-circuit); ELSE `wrapped_gaussian_1d(D, sigma, period, ts, 4).prod(axis=0)`.
  - ELSE → `_compute_Q(D, r_a, is_rel, is_per, period, reduced=is_rel)` (rel-per = minimum-image pairwise wrap, dispatch.py:885–920).
- aggregation by incidence matmuls `GX @ (ov @ GY)`.

#### 1.6d `_rel_inner_batched(Px, Wx, Py, Wy, sigma, r, is_per, period, return_cancellation_ratio, truncation_sigmas, samples_per_sigma)` — _mobius_inner.py:1071 (translation-grid / spectral)
- SPECTRAL GATE (:1133): IF `_SPECTRAL_IP_ENABLED (True) and 2 <= r <= 4 and not return_cancellation_ratio` → `_spectral_rel_inner_matrix` (:894):
  - `L = period` (per) or `span_x + span_y + 2*(_SPECTRAL_IP_MODE_SIGMAS + 2)*sigma` (non-per; `None` if not finite/≤0).
  - `M = ceil(_SPECTRAL_IP_MODE_SIGMAS/sqrt(2) * L/(2π sigma)) + 2`; `grid_size = (2M+1)^(r-1)`.
  - GUARD `grid_size > _SPECTRAL_IP_MAX_POINTS (4_000_000)` → `None` (never bypassed).
  - COST GATE `(not _SPECTRAL_IP_FORCE) and grid_size > _SPECTRAL_IP_COST_C (1100) * k_slots² * n_pairs` → `None`. OVERRIDE `_SPECTRAL_IP_FORCE=True` (module flag) bypasses only this gate.
  - else → LEAF spectral Gram matrix (`get_set_partitions_with_mobius`, Hermitian-halved mode grid). Self-IP shortcut when `Px is Py and Wx is Wy`.
  - `None` → fall through to the grid.
- grid: `samples_per_sigma = resolve_samples_per_sigma(None, r, ts)` (_defaults.py:197; 3 at r ≤ 4, 4 at r = 5,6 at ts = 6); `shared_w = all columns of Wx equal and of Wy equal`.
  - IF `is_per`: `N_u = auto_ntau_default(period, sigma)`, `u_grid = linspace(0, P, N_u, endpoint=False)`.
  - ELSE: per-pair mid-range centres; `span = max spread_x + max spread_y + 2*_rel_window_margin(ts)*sigma` (:1360: `8.0` if ts non-finite else `min(sqrt(2)*ts + 0.1, 8.0)`); `N_u = max(64, ceil(max(span,1)/sigma*samples_per_sigma))`.
  - SPARSE GATE (periodic): `is_per and r >= 2 and K_x*K_y >= 200_000` and `2*sqrt(cutoff) < period*(1-1e-8)` (`cutoff = truncation_ip_sqdist(ts, sigma) = 2(kσ)²`) and probe `K0.nnz <= 0.20*K_x*K_y` → LEAF `_rel_per_inner_sparse` (:689) → `inner_product_orbit_sparse` per (pair, u-node).
  - dense slabs (`_ORBIT_GRID_SLAB_ELEMS = 2^19`): per u-chunk, IF `is_per`: nearest-image wrap; `n_img = _rel_per_image_count(sigma, period, ts)` (:855; `ceil(2(σ/P)sqrt(ln 1/tol) − 1/2)`, 0 when ≤ 0); IF `n_img > 0` → image sum `Σ_{l=-n..n} _trunc_kernel_exp((diffs ± l·P)²)` (full-image measure C); ELSE `_trunc_kernel_exp(diffs²)`. Non-per → `_trunc_kernel_exp(diffs²)`.
  - IF `shared_w` → LEAF `_mobius.inner_product_orbit_grid(K_uc, Wx[:,0], Wy[:,0], r)`; ELSE LEAF `_mobius.inner_product_orbit_pw_batched(K_uc, w_A_uc, w_B_uc, r, prefactor=1.0)`.
  - result `× (σ√π)^r / (σ sqrt(2π/r))² × du` (Riemann sum).

### 1.7 `_ip_core_ma(u_cell, w_u, n_j, v_cell, w_v, n_k, A, r_vec, sigma, is_rel, is_per, period, truncation_sigmas, kernel_precision, inner_r, wrap)` — cosine.py:1744 (Bulger/centres kernel core)
- `truncation_sigmas = resolve_truncation_sigmas(...)`.
- IF `single_attr_helper_ok = (A == 1 and (inner_r is None or inner_r[0] == 0) and not (is_rel[0] and is_per[0]))` → LEAF `_ip_via_helper` (cosine.py:3713) → `gaussian_kernel_sum(V, w_v, U, sigma*sqrt(2), is_rel, r, is_per, period, wrap, n_terms=|U|*|V|, truncation_sigmas, kernel_precision)` (§3.6). This is the **single-multiset inner-product path** (direct/culled/circular/grid selection lives in `gaussian_kernel_sum`).
- ELIF `all_r1 = A >= 1 and all r_a == 1 and all inner_r == 0` → LEAF `_ip_r1_direct` (cosine.py:1857): per attribute rel → skip; abs-per full-image → `wrapped_gaussian_1d(..., 4)` log; abs-per single-image → nearest wrap; else `-d²/(4σ²)`; threshold `-ts²/2 − log(n_terms)`; chunked by `3*n_j*8` bytes/column.
- ELIF `bytes_needed = (max_r+2)*n_j*8*n_k <= kernel_chunk_bytes_resolved()` → LEAF `_ip_full_ma` (cosine.py:1923) = `_ma_log_kernel` + `_trunc_log_kernel_exp(n_terms=n_j*n_k)` + `w_u @ (E @ w_v)`.
- ELSE chunked comb side with `_ma_log_kernel` + `_trunc_log_kernel_exp` (cosine.py:1836–1853).
- `_ma_log_kernel` (cosine.py:2024), per attribute:
  - IF `not is_per[a] and _gram_is_accurate_enough(U, V, sigma, ts)` (cosine.py:1951: `eps*s²/(4σ²) <= 0.1*truncation_floor(ts)`, `s` = max |value − first value|) → LEAF `_gram_quadratic_form` (cosine.py:1978; block = `r_in` if > 0 else `r_a` if rel else 0).
  - ELIF `r_in > 0` → `_compute_Q_inner_blocks(D, r_in, is_per, period, reduced=False)` (dispatch.py:745).
  - ELIF `is_per[a] and not is_rel[a] and wrap_a == 'full-image'` and `_image_count_L(sigma, period, ts_a, 4) > 0` → `wrapped_gaussian_1d(D, ..., 4)` summed log θ (cosine.py:2102–2109).
  - ELSE: abs-per (single-image or L = 0) → nearest-image wrap; `_compute_Q(D, r_a, is_rel, is_per, period)` (rel-per → minimum-image pairwise-wrap form, measure A).
- `_trunc_log_kernel_exp` (cosine.py:2170): threshold `-ts²/2 − log(n_terms)` (if `n_terms > 1`).

### 1.8 Nested path — `_try_nested_contract(dens_x, dens_y, normalize, verbose, force, force_route, method_name)` — cosine.py:2798
- `_decline(reason)`: IF `force` → ERROR `ValueError("method=... is not available here: ...")`; else `None` (→ Bulger enumeration).
- declines: `normalize not in ("cosine","oneSidedDenom")`; `n_attrs` mismatch; IF `dens_x.n_attrs != 1` → `_try_nested_contract_ma` (below); `spec is None or spec_y is None`; `_inner_r_vec != 0` on either side ("inner/intermediate [rel] unit not yet covered"); `[r]`/`[sym]` levels differ.
- `skip_xx, skip_yy = _nested_self_ip_skip_flags` (cosine.py:2592; same rule as flat).
- NODE `route, taus = _nested_attr_plan(dens_x, dens_y, 0, force_route, skip_xx, skip_yy)` (cosine.py:2530):
  - `route = _nested_attr_route(...)` (cosine.py:2660):
    - IF `not is_rel` → `"centres"` if `force_route == "centres"` else `"contract"`.
    - `admissible = _nested_admissible_routes(dens_x, dens_y, a)` (cosine.py:2481) — the **measure rule**: not rel → `["contract"]`; rel non-per → `["centres", "contract_relnonper"]`; rel-per with `period > 0 and sigma/period > thr(default ts)` → `["centres"]` if `wrap_a == 'single-image'` else `["taugrid"]`; rel-per otherwise → `["centres", "taugrid"]`. (Uses `get_default("truncation_sigmas")`, **not** the call's `truncation_sigmas`.)
    - IF `is_per and admissible == ["taugrid"]`: `force_route == "centres"` → ERROR `ValueError("method='centres' cannot be honoured ...")`; else `"taugrid"`.
    - IF `force_route == "centres"` → `"centres"` (OVERRIDE of the cost race only).
    - IF `len(admissible) == 1` → that route.
    - ELSE → `price_nested_attr(dens_x, dens_y, a, admissible, skip_xx, skip_yy)[0]` (_nested_cost.py:382): terms from `nested_attr_terms` (:291); `cost = nested_route_cost_ms(route, total_order=R, term, n_matrices)` (:198: `max(exp(a)*max(term,1)^b, f + pm*n_matrices)` with `_NESTED_COST_LAW`/`_NESTED_FLOOR_MS` keyed by `_nested_cost_key(R)` ∈ {2,3,4,6}); MEMORY GUARD: `route == "centres" and len(admissible) > 1 and nested_centres_working_set_bytes > _CENTRES_WORKING_SET_SOFT_BUDGET` → `cost = inf`; `best = min(admissible, key=prices)` (ties → first listed).
  - grid: `"centres"`/`"contract"` → `taus=None`; `"taugrid"` → `linspace(0, P, auto_ntau_default(P, σ), endpoint=False)`; `"contract_relnonper"` → `auto_taus_line(allv, allv, sigma, truncation_floor(default ts))` (_nested_contraction.py:681).
- NODE plan vs enumeration (cosine.py:2876): IF `not force and _nested_prefers_enumeration(dens_x, dens_y, {0: route}, skip_xx, skip_yy)` → `_LAST_NESTED_ROUTES = []`; return `None` → Bulger. `_nested_prefers_enumeration` (cosine.py:2637) → `select_nested_method` (_nested_cost.py:501): `plan_ms = Σ price_nested_attr(...) + _flat_companion_cost_ms(...)`; `enum_ms = nested_route_cost_ms("bulger", r_max, predict_nested_pairwise_kernel_size(...), n_matrices)` if `enumeration_ok` else `inf`; `enumeration_ok = _nested_enumeration_admissible` (cosine.py:2608: False iff some rel-per attribute with `wrap != 'single-image'` has `sigma/period > thr(default ts)`); `chosen = "bulger" if enum_ms * _NESTED_ENUM_SAFETY (2.0) < plan_ms else "contract"`. Records `_LAST_NESTED_COSTS` (diagnostic only).
- LEAF `_nested_attr_matrix(dens, dens, 0, route, taus)` (cosine.py:2745) ×3:
  - `"centres"` → `_closed_form_attr_centres` + `_closed_form_attr_matrix_from(cx, cy, ts, wrap_a)` (§1.6c; nested wreath-product restriction).
  - else `build_recipe(r_levels, sym_levels, tags, is_rel, is_per)` (_nested_contraction.py:129; per level `use_orbit = _orbit_eligible(n, r, sym, ...)` = `sym and 2 <= r <= _ORBIT_R_MAX_SHIPPED`) → `nested_attr_matrix(rx, ry, PX, PY, WX, WY, sigma, is_per, period, ts, taus, periodic_taus, taus_reduce, wrap_a)` (§1.9): `"taugrid"` → `periodic_taus=True, taus_reduce="mean"`; `"contract_relnonper"` → `is_per=False, periodic_taus=False, taus_reduce="sum"`; `"contract"` → `taus=None, wrap_a=dens_x.wrap[a]`.
- MEMO key `('contract', default ts, None, (route, (len(taus), taus[0], taus[-1]) or None))` read/written for xx and yy (cosine.py:2892–2907). Note `<X,X>` is **always** computed (no `need_xx` skip) on this route.
- `_try_nested_contract_ma` (cosine.py:2911) — MA with ≥ 1 nested attribute:
  - Pass 1 per attribute: `is_nested` → declines (one-sided nesting; inner `[rel]`; `[r]/[sym]` mismatch) → `_nested_attr_plan(...)` → `("nested", a, route, taus)`; `ordered_flat = not nested and not is_sym_x[a] and r_a > 1` → `("ordered", a)`; else `("flat", a)`.
  - NODE plan vs enumeration as above with `{a: route for nested}` (cosine.py:3006) → `None` → Bulger.
  - MEMO key `('contract_ma', default ts, None, tuple((kind, a, route, tau_sig)))`.
  - Pass 2: `"flat"` → `_ma_per_attr_inner_matrix(Pxa, Wxa, Pya, Wya, sigma, r_a, is_rel, is_per, period)` (**no `wrap`, no `truncation_sigmas` passed** — defaults `'full-image'` and global default); `"nested"` → `_nested_attr_matrix`; `"ordered"` → `_closed_form_attr_matrix_from(cx, cy, _ts_ma, wrap_a)`.

### 1.9 `nested_attr_matrix` / `_contract` / `_combine_pair` — _nested_contraction.py:930, 549, 291
- `nested_attr_matrix` wraps `_nested_attr_matrix_impl` in `orbit_guard_scope(truncation_sigmas)` (:233; binds `floor = truncation_floor(ts)`, `enabled = get_default("post_hoc_guards")`).
- `_nested_attr_matrix_impl` (:981):
  - IF `taus is not None and not periodic_taus and taus_reduce == "sum"` → `_shared_template_matrix` (:803): requires both recipes ordered at the outer level (`not recipe.sym`), `r == len(children)`, every event a shared leaf template (`_all_shared_templates`), equal cell lengths → LEAF closed-form template cross-correlation; else `None` → generic.
  - NaN padding → zero weight (:1003).
  - IF `taus is None`: `is_per and wrap_a != 'single-image'` → `wrapped_gaussian_1d(d, σ, P, ts, 4)`; else (`is_per` → `_wrap`) `exp(-d²/(4σ²))`; `_trunc`; `_contract`.
  - ELSE: `_tau_window` (:879) when `not periodic_taus` (per-pair window slice; `None` when `T < 3`, non-uniform grid, no truncation, or `W >= T`); `periodic_taus and wrap_a != 'single-image'` → `wrapped_gaussian_1d(..., 4)`; else (`periodic_taus` → `_wrap`) `exp`; `_trunc`; `_contract`; reduce `mean` (÷ full `T`) or `sum`.
- `_contract` (:549) → `_subtree_overlaps` (:513) / `_leaf_overlaps` (:478): `r == 1` leaf → einsum; uniform siblings → one batched `_combine_pair`; ragged → per-pair `_combine_pair`. `use_orbit = xnode.use_orbit and ynode.use_orbit`.
- `_combine_pair(M, r, sym, use_orbit, cost_check=True)` (:291):
  - IF `use_orbit and cost_check` → `use_orbit = orbit_cost_model(r, max(gx, gy), Q)[0]` (_orbit_cost.py:71: False if `r > K or r < 2`; `k_eff = max(r, K-1)` if `r <= _MARGIN_R_MAX (3)`; `log_ratio = orbit_cost_intercept (3.8536) + 1.0708·ln n_orbits(r) + 1.4033·ln K − 0.3608·ln B − 0.8188·log_tuple_pairs − 0.2261·ln r`; orbit iff `< 0`).
  - IF `use_orbit` → LEAF `_combine_orbit` (:402) → `inner_product_orbit_grid(..., return_term_mass=True)`, `bound = eps*max(mass_sum)/r!`; ACCURACY GUARD: `budget is None` → vals; `not budget["enabled"]` → vals (OVERRIDE `post_hoc_guards=False`); `bound <= floor` → vals; `work = _enum_work(Q, gx, gy, r) <= _ORBIT_ENUM_MAX_WORK (16e6)` → `RuntimeWarning` (once per scope) + LEAF `_combine_chunked` (enumeration; fallback Möbius → enumeration); else `RuntimeWarning` + vals (kept).
  - ELSE → LEAF `_combine_chunked(M, xtup, ytup, _ORBIT_ENUM_MAX_ELEMS (16e6))` (:277) → `_combine`.

### 1.10 Leaves reached from ENTRY 1

| Leaf | File:line | Role |
|---|---|---|
| `_r1_broadcast_fast` kernel loop | cosine.py:759 | all-r=1 broadcast cross terms (bulger memo) |
| `_ip_via_helper` → `gaussian_kernel_sum` | cosine.py:3713 / _kernel.py:27 | single-multiset inner product (A=1, not rel-per) and factored flat factor |
| `_ip_r1_direct` | cosine.py:1857 | all-r=1 MA inner product |
| `_ip_full_ma` / chunked `_ma_log_kernel` | cosine.py:1923 / 2024 | Bulger perm×comb and centres perm×perm log-kernel |
| `_gram_quadratic_form` | cosine.py:1978 | non-periodic Q via Gram identity |
| `_compute_Q` / `_compute_Q_inner_blocks` | dispatch.py:848 / 745 | quadratic forms (rel-per pairwise-wrap = measure A) |
| `wrapped_gaussian_1d` | _wrapped_kernel.py:103 | abs-per full-image θ (image-sum or Fourier) |
| `_ma_per_attr_inner_matrix` r=1 kernel sum | _mobius_inner.py:322 | Möbius route, r=1 attribute |
| `inner_product_orbit_pw_batched` (via dense abs) | _mobius.py:851 | Möbius abs r≥2 |
| `_orbit_safe_submatrix_sparse` → `inner_product_orbit_sparse` | _mobius_inner.py:180 / _mobius.py:1056 | Möbius abs sparse |
| `_spectral_rel_inner_matrix` | _mobius_inner.py:894 | rel r=2..4 spectral (Fourier) matrix |
| `_rel_inner_batched` grid → `inner_product_orbit_grid` / `_pw_batched` | _mobius_inner.py:1244 | rel translation grid (all-image, measure C) |
| `_rel_per_inner_sparse` | _mobius_inner.py:689 | rel-per sparse grid |
| `_closed_form_attr_matrix_from` | _mobius_inner.py:1843 | tuple-centres closed form (flat rel via `choices`, nested `centres`, ordered flat) |
| `nested_attr_matrix` → `_contract` → `_combine_orbit` / `_combine_chunked` | _nested_contraction.py:930/402/277 | nested contraction (taugrid / relnonper / contract) |
| `_shared_template_matrix` | _nested_contraction.py:803 | spectral-cell rel-nonper closed form |
| `_ma_ip_factored` (+ `_ma_ip_factor_dense`, `_ma_ip_factor_nested_culled`) | cosine.py:3579 | `method='factored'` |
| `sweep_cos_sim_exp_tens` | sweep.py:833 | tagged sweep (ENTRY 2) |

### 1.11 Overrides (ENTRY 1)
- `method='bulger'`: selector returns immediately (skips memory guard, wrap override, cost model; `_guard_forced_bulger_feasible_ma` not applied — dispatch.py:1497). On nested densities → joint-tuple enumeration without the nested-contraction plan. Still subject to: structural guards, `ordered_any` (no-op since it sets bulger), empty-density short-circuit.
- `method='mobius'`: selector returns `'mobius'` (skips wrap/measure override → on rel-per with `wrap='single-image'` above threshold the Möbius all-image route runs anyway); inside the orbit route `user_forced_mobius=True` pins the grid (`_ma_rel_attr_prefers_centres` → False unless `rel_attr_route` is set explicitly). Still subject to: `ordered_any` → bulger (silent), post-hoc impossible-value guard (may divert to bulger), nested → contraction plan (`force=True`, raises on uncovered cases).
- `method='centres'`: flat → unrestricted centres enumeration (`_cos_sim_exp_tens_ma_centres`); nested → contraction with `force_route='centres'` (raises above threshold on rel-per full-image). Still subject to `ordered_any` → bulger.
- `method='contract'`: nested only (else `ValueError`); same plan as `'mobius'` on nested; `force=True`.
- `method='factored'`: separate route; bypasses everything after the structural guards.
- `normalize='oneSidedDenom'`: `need_xx=False` → `<X,X>` skipped on bulger/centres/mobius routes (not on factored, nested-contract, sweep).
- `dedup=False`: skips canonical dedup (does not change per-pair arithmetic).
- `mpt.set_default(rel_attr_route='centres'|'mobius'|'grid')`: pins the per-attribute rel route inside the Möbius route (raises when the pin is inadmissible).
- `mpt.set_default(post_hoc_guards=False)`: skips the impossible-value fallback and the nested orbit accuracy guard.
- `_mobius_inner._SPECTRAL_IP_ENABLED/_SPECTRAL_IP_FORCE`, `_COMB_RESTRICTION_ENABLED`, `_mobius._FOURIER_ENABLED`: module-level test levers.

### 1.12 Guards and fallbacks (ENTRY 1)
- `dens.n == 0` → 0.0 (no arithmetic).
- `r_max > 8` under auto → Bulger, with `SingleImageInfeasibleError` if the pair kernel exceeds 4 GiB.
- Working-set > 256 MiB (not blocked by rel-per σ/P) → Möbius (direction: Bulger → Möbius).
- σ/P > thr with `wrap`: single-image → Bulger; full-image → Möbius (measure, not cost).
- Post-hoc impossible value on Möbius → Bulger (purges Möbius memo).
- Nested orbit accuracy bound > floor → enumeration if work ≤ 16e6 else keep value + warn.
- Nested plan priced vs enumeration: enumeration if `enum_ms*2 < plan_ms` (direction: contraction → Bulger); enumeration inadmissible (`inf`) for rel-per full-image above threshold.
- Nested centres route diverted to `inf` when working set > 256 MiB and another admissible route exists.
- Spectral branch declines (`None`) → translation grid; sparse gates decline → dense.
- `_gram_is_accurate_enough` False → difference-tensor `_compute_Q`.
- Sweep reduction declines → per-entry loop.
- `_r1_broadcast_fast` declines → per-pair loop.

### 1.13 Memo keys (ENTRY 1)
| Route | Key | Read | Written |
|---|---|---|---|
| Bulger pairwise, r1 broadcast | `('bulger', ts, kp, None)` | xx (if needed), yy | when computed |
| centres | `('centres', ts, kp, None)` | xx (if `need_xx`), yy | when computed |
| Möbius | `('mobius', ts, None, choices_tuple)` | xx, yy | when computed; purged on post-hoc failure |
| nested single-attr | `('contract', default_ts, None, (route, tau_sig))` | xx, yy | always computed if absent |
| nested MA | `('contract_ma', default_ts, None, plans_sig)` | xx, yy | always computed if absent |
| factored | none | — | — |
| pricing flag | any key with `k[0] in _SELF_IP_ROUTES` → `_self_ip_memoised` (shared by both sides of every comparison) | | |

---

## 2. ENTRY `sweep_cos_sim_exp_tens(dens_x, dens_y, offsets, method, normalize, truncation_sigmas, kernel_precision, verbose)` — `_tensor/sweep.py:833`

- [parse] `normalize ∈ {'cosine','oneSidedDenom'}` else `ValueError` (:910). `method ∈ {'auto','mixture','orbit'}` else `ValueError` (:923). `offsets` 1-D → `(1, M)`; ERROR if `ndim != 2`, `shape[0] != n_attrs`, or non-finite (:918–936).
- `dx, dy = pruned()`; `mixture_ok, reason = sweep_eligibility(dx, dy, off, truncation_sigmas)` (:81); `orbit_ok = orbit_sweep_supported(dx, dy, off, truncation_sigmas)` (:541).
  - `sweep_eligibility` False when: not both `MaetDensity`; bad shape; non-finite; for an **unswept** rel-per attribute `_rel_per_measure_admissible` (:169) fails (`sop > thr(ts)` and `wrap != 'single-image'`); a swept attribute is `is_rel`, or `is_per`, or `inner_r > 0`; a swept attribute carries `kernel_cov`; `n_j == 0 or n_k == 0`.
  - `orbit_sweep_supported` False when: any `is_sym` False; any `inner_r > 0`; a swept attribute is rel; a rel-per attribute with `sop > thr(ts)` and `wrap != 'full-image'`; any kernel cov on either density.
- IF `method == "auto"` → NODE `_choose_sweep_route(dx, dy, off, mixture_ok, orbit_ok)` (:768): `not orbit_ok` → mixture; `not mixture_ok` → orbit; any swept attribute with `r_a < 2` → mixture; `orbit_work <= 0` → mixture; `mixture_bytes = n_pairs*(n_swept+2)*8 > kernel_chunk_bytes_resolved()` → orbit; `n_pairs < _ORBIT_MIN_PAIRS (1e6)` → mixture; `orbit_total < _ORBIT_WORK_RATIO (64) * n_pairs` → orbit else mixture (`orbit_total = M*N_x*N_y*Σ_swept n_orb(r)*K_x*K_y*r`).
  ELSE `chosen = method` (OVERRIDE).
- ERROR `ValueError` if `chosen == "mixture" and not mixture_ok` (:949) or `chosen == "orbit" and not orbit_ok` (:955).
- IF `chosen == "orbit"` → LEAF `_finalise_orbit_sweep` (:752): numerator `_orbit_sweep` (:682): per attribute untranslated → `_ma_per_attr_inner_matrix(..., truncation_sigmas, wrap=wrap_a)` (§1.6b) broadcast over M; swept → `_orbit_attr_matrix_sweep` (:603): `use_orbit = r >= 2` → `inner_product_orbit_pw_batched`; `r == 1` → einsum kernel sum; kernel = `wrapped_gaussian_1d(..., 4)` if `is_per and wrap == 'full-image'` else nearest-image + `_trunc_kernel_exp`. Denominators `_orbit_self_ip` (:722) via `_ma_per_attr_inner_matrix` — **no memo**; `ip_xx` skipped unless `normalize == 'cosine'`. (`ts`, `kernel_precision`, `verbose` unused on this route.)
- ELSE mixture → LEAF `_build_mixture(dx, dy, swept, truncation_sigmas=ts)` (:249): `split_idx` = attributes with `_splittable` (:226: `not is_per` and no kernel cov); `has_placement = not is_rel and block == 0`; swept attributes → mixture axes; still splittable → folded placement term; `fixed_idx` (periodic / anisotropic) → `_ma_log_kernel(...)` (§1.7) with `wrap_fixed`; threshold `-ts²/2 − log(n_j*n_k)`; components with `log_fixed < threshold` dropped. → `_evaluate_mixture` (:406): `S == 0` → constant; `P <= mean_slice + 512` → `_evaluate_dense` (:481) else culled per-offset loop. Denominators `_self_ip` (:1005): MEMO key `('sweep', float(ts), str(kernel_precision))` read/written; `ip_xx` only when `normalize == 'cosine'`.
- Overrides: `method='mixture'`/`'orbit'` bypass `_choose_sweep_route` but not the eligibility guards.

---

## 3. ENTRY `eval_exp_tens(*args, normalize, dedup, spectrum, precision, method, truncation_sigmas, kernel_precision, verbose)` — `_tensor/eval.py:62`

- [parse] `normalize ∈ {'none','gaussian','pdf'}` (applied in `_ma_eval_normalize`, eval.py:791; any other string acts as `'gaussian'`-like: multiplies by the Gaussian constant and skips the pdf division — no validation). `method` validated in `_select_ma_eval` (dispatch.py:2065–2074) as ∈ {`'auto'`, `'centres'`, `'mobius'`} else `ValueError`. No retired synonyms accepted.
- `truncation_sigmas = resolve_truncation_sigmas(...)` at entry (eval.py:190).
- ERROR `TypeError` when `len(args) < 2` (eval.py:176).

### 3.1 Input-form dispatch (eval.py:192–381)
- density scalar/list → ERROR if `len(args) not in (2, 3)`, `spectrum`, `precision`; scalar → `_eval_exp_tens_scalar` (§3.2); list → `_eval_exp_tens_density_list` (eval.py:452): empty → empty; `use_dedup = dedup and all(is_single_multiset)` → `_chord_canonical_key` cache → `_eval_exp_tens_scalar` once per key; else per density.
- `_looks_like_multi_attr(a)` → 8/9/10 args; IF `sigma_vec_has_kernel_cov` → `build_exp_tens` → `_eval_exp_tens_scalar`; ERROR on `spectrum`/`precision`; → `_eval_exp_tens_raw_ma_scalar` (eval.py:677) → build → `_eval_exp_tens_scalar`.
- raw single-multiset: 8/9/10 args; IF `is_kernel_cov(sigma)`: ERROR on `spectrum`; `NotImplementedError` if `a_arr.ndim != 1`; build → scalar. `ndim == 1` → `_eval_exp_tens_raw_single_multiset_scalar` (eval.py:531; `add_spectra` if `spectrum`; build) → scalar. `ndim == 2` → `_eval_exp_tens_raw_single_multiset_batch` (eval.py:557): `NotImplementedError` when ordered and `r > 1` (eval.py:583); rows with `< r` valid → NaN; `_chord_canonical_key` dedup; `n_valid == 0` → all-NaN; each unique density → `_eval_exp_tens_scalar`. Else `TypeError`.
- Deprecated `eval_exp_tens_raw` (eval.py:1691) → `eval_exp_tens`.

### 3.2 `_eval_exp_tens_scalar(dens, x, normalize, method, ...)` — eval.py:385
- IF `WindowedMaetDensity`: `NotImplementedError` if the underlying density has kernel cov; `underlying = _eval_exp_tens_ma(dens.dens, x, normalize, ...)` — **`method` is not forwarded** (defaults to `'auto'`); `× _evaluate_window_on_query` (windowing.py:257).
- IF `density_has_kernel_cov(dens)` → `x = whiten_query(dens, x)`; after evaluation `× exp(-0.5*density_logdet_sum)` when `normalize != 'none'`.
- IF `MaetDensity` → `_eval_exp_tens_ma` (§3.3). ELSE `TypeError`.

### 3.3 `_eval_exp_tens_ma(dens, x, normalize, method, truncation_sigmas, kernel_precision, verbose, prune_zero_weight_events)` — eval.py:825
- `n_q_hint` from `x`; NODE `chosen, reason = _select_ma_eval(dens, n_q_hint, method, truncation_sigmas)` (§3.4). `_maybe_warn_eval_time` (timing warning only).
- IF `chosen == "mobius"` → LEAF `eval_ma_orbit(dens, _ma_join_query(dens, x), truncation_sigmas)` (_ma_eval_orbit.py:30; **`kernel_precision` not forwarded**) → per event `n`, per attribute `a`: live values; `is_rel[a]` → `_mobius.eval_orbit_rel` else `_mobius.eval_orbit_abs`; `wrap` forwarded only for abs attributes (§3.5). → `_ma_eval_normalize`.
- ELSE (centres):
  - IF `is_single_multiset(dens)` (density.py:643: `n_attrs == 1 and n == 1 and nested[0] is None`) → prune zero-weight tuples; `xs.shape[1] == 0` → zeros; → LEAF `_eval_core(c0, wj0, ..., truncation_sigmas, wrap=dens.wrap[0])` (§3.6a). (**`kernel_precision` not forwarded.**)
  - ELSE NODE `_ma_eval_factored(dens, x, truncation_sigmas, prune_zero_weight_events)` (eval.py:1077): returns `None` when any `r_a < 2`, or `kernel_cov is not None`, or some attribute has fewer ever-valid values than `r_a`; else per event/attribute: `r_in > 0` → dense block form `_compute_Q_inner_blocks(..., reduced=True)`; else `_eval_core(c, w_tuple, ..., truncation_sigmas=ts)` — **`wrap` not passed (defaults `'full-image'`)**; product over attributes, sum over events.
  - ELSE joint materialisation (eval.py:949–1073): prune `w_j == 0`; query shape validation (`ValueError`s); `n_q == 0` → zeros; `_ma_value_tables` (eval.py:1240: abs-per full-image attributes with `_tuple_values_repeat` — `n_q >= 100 and n_entries >= 256 and n_distinct*4 <= n_entries`); IF `bytes_needed = (2*max_dim+2)*n_j*8*n_q <= kernel_chunk_bytes_resolved()` → LEAF `_ma_eval_full` (eval.py:1264) single chunk, else chunked.
  - `_ma_eval_full` per attribute: `da == 0` skip; abs-per full-image with table → tabulated `wrapped_gaussian_1d(..., 2)` product (`abs_per_factor`); `r_in > 0` → `_compute_Q_inner_blocks(reduced=True)`; abs-per full-image (no table) → `wrapped_gaussian_1d(d_a, ..., 2).prod(axis=0)` into `abs_per_factor`; abs-per single-image → nearest-image wrap then `_compute_Q`; else `_compute_Q(reduced=is_rel)`; truncation mask `q_total <= ts²/2`; `e *= abs_per_factor`; `w_j @ e`. `kernel_precision='single'` → float32 accumulator.
- `_ma_eval_normalize` (eval.py:791): `'none'` → raw; else `× ∏_a 1/_gaussian_mass_const(σ_a, d_a, _quadratic_form_det(r_a, inner_r_a, is_rel_a))`; `'pdf'` → `/ Σ w_j` (warn if zero).

### 3.4 `_select_ma_eval(dens, n_q, method, truncation_sigmas)` — dispatch.py:2004 (eval selector)
1. OVERRIDE `method == "centres"` → `("centres", "user override")`.
2. OVERRIDE `method == "mobius"` → GUARD `_reject_ordered_for_mobius` (dispatch.py:1987: `_has_ordered_attr` = any flat attribute with `not is_sym and r > 1`) → ERROR `ValueError`; else `("mobius", "user override")`.
3. `method != "auto"` → ERROR `ValueError`.
4. `_has_ordered_attr(dens)` → centres ("ordered ([sym]=0) attribute").
5. any `nested[a] is not None` → centres ("nested attribute").
6. `all(r_vec[a] <= 1)` → centres ("all r <= 1").
7. any `r_a >= 2 and r_a > _ORBIT_R_MAX_FEASIBLE (10)` → `force_centres_reason`; GUARD `_estimate_ma_joint_working_set_bytes(r_vec, k_vec, is_rel, sym_vec) > _CENTRES_PROBE_MEM_BUDGET (4 GiB)` → ERROR `SingleImageInfeasibleError`; else centres. (There is **no precision guard** on `K_a − r_a` despite the docstring at dispatch.py:2040–2043.)
8. rel-per measure rule: first attribute with `is_rel and is_per and period > 0 and sigma/period > thr(ts)`: `wrap_a == 'single-image'` → centres ("rel-per single-image measure"); else → mobius ("rel-per full-image measure"). Precedes the cost model.
9. cost model: `centres_ms, mobius_ms = _ma_eval_costs_ms(dens, n_q)` (dispatch.py:1725; terms: factored centres (`A > 1 and all r_a >= 2 and no kernel_cov`) = setup + Σ_a call-per-joint·T_a + n_q·(factored base + per-kernel base + slope·T_q·cull_a); joint centres otherwise; Möbius = setup + Σ_a [B_r setup + per-query ops `(2^r−1)·r·K` (× u-grid nodes for rel; spectral branch priced when `r ∈ 2..4`, `n_q >= (16,32,64)[r-2]`, `K >= (2,8,16)[r-2]`, and the mode grid fits `_SPECTRAL_IP_MAX_POINTS`)]); `joint_ws = _estimate_ma_joint_working_set_bytes(...)`; `safety = _MA_MOBIUS_SAFETY (1.5) if joint_ws > _CENTRES_WORKING_SET_SOFT_BUDGET else _MA_MOBIUS_SAFETY_SMALL (1.0)`; `mobius_ms < centres_ms * safety` → mobius else centres.

### 3.5 Möbius point evaluators — `_mobius.py`
- `eval_orbit_abs(p, w, sigma, r, x, is_per, period, return_cancellation_ratio, truncation_sigmas, kernel_precision, wrap)` (:1287): `per_helper_global = period > 2*sqrt(2)*ts*sigma` if `is_per` else False; per distinct block B: non-per → `use_reduction=True`; per → `use_reduction = per_helper_global and span_ok` (`span < 0.5*period` over the block's circular offsets). `use_reduction` → LEAF `gaussian_kernel_sum(p, w^m, mean_x, sigma/sqrt(m), truncation_sigmas, wrap, [is_per, period])` (§3.6) × `exp(-var/2σ²)`; else LEAF direct broadcast: `is_per and wrap == 'full-image'` → `wrapped_gaussian_1d(diffs, σ, P, ts, 2).prod`; else nearest-image wrap + `exp`. Combined by `mobius_partition_combine`.
- `eval_orbit_rel(p, w, sigma, r, x_rel, is_per, period, samples_per_sigma, return_cancellation_ratio, truncation_sigmas, kernel_precision, factored)` (:1873):
  - `r < 2` → constant `Σw` (no arithmetic).
  - u-grid: per → `N_u = max(64, ceil(P/σ·spp))`, non-per → window `[p.min − max(0,x_max) − 8σ, p.max − min(0,x_min) + 8σ]`, `N_u = max(64, ceil(max(width,1)/σ·spp))`, `spp = resolve_samples_per_sigma(None, r, ts)`.
  - SPECTRAL GATE (:2041): `_FOURIER_ENABLED (True) and 2 <= r <= 4 and factored is None and not return_cancellation_ratio and n_q >= (16,32,64)[r-2] and K >= (2,8,16)[r-2]` and (`not is_per` or (`_span_ok` = every query's position span `< 0.5*period` and `period > 2*sqrt(2)*ts*sigma`)) → LEAF `_eval_orbit_rel_fourier` (:1731) (all-image on the circle).
  - periodic: `factored_per_valid = period > 2*sqrt(2)*ts*σ and max_spread < 0.5*period`; `factored is True and not valid` → ERROR `ValueError`; `factored is None` → `use_factored = valid and _factored_worthwhile(K, r, n_q, N_u, n_fine_total)` (:1698: `K*n_fine + 10*B_r*r*N_u*n_q < B_r*r*K*N_u*n_q`); else `bool(factored) and valid`.
  - non-periodic: `factored is None` → `_factored_worthwhile(...)`; else `bool(factored)`.
  - `use_factored` → LEAF tabulated `S_m` via `gaussian_kernel_sum` + `_lagrange6_circular`/`_lagrange6_uniform` read-back (:2118–2205). ELSE LEAF direct: `eval_orbit_abs` at each u-node with default `wrap='full-image'` (:2207–2248). Integral: per → `F.sum*du`; non-per → `np.trapezoid`.
  - (The docstring's "cross-correlation strategy at r = 2" (:1960–1963) does not exist in the code.)

### 3.6 Kernel evaluation — `gaussian_kernel_sum(C, wJ, X, sigma, is_rel, r, is_per, period, truncation_sigmas, kernel_precision, wrap, n_terms)` — _kernel.py:27
- `truncation_sigmas` resolved; IF `n_terms > 1` → widened to `sqrt(k² + 2 ln n_terms)`. ERROR `ValueError` on bad `kernel_precision`, `truncation_sigmas <= 0`, shape mismatches, `is_rel and r < 2`, `is_per and period <= 0`.
- `use_truncation = isfinite(k) and k > 0 and nJ > 0 and nQ > 0` (always true after resolution unless empty).
- IF `use_truncation and not is_per`: `dim == 1 and not is_rel` → LEAF `_truncated_kernel_sum_1d_vectorised` (:466; `max_win >= nJ` → dense fallback inside); else LEAF `_truncated_kernel_sum` (:302; bucket grid, whitening for rel).
- ELIF `use_truncation and is_per and dim == 1 and not is_rel`: `2*truncation_radius(k, σ) < period` → LEAF `_truncated_kernel_sum_1d_circular` (:535; nearest copy only — coincides with `L == 0` of `_image_count_L(σ, P, k, 2)` exactly, since both reduce to `σ/P < 1/(2k)`); else `_exact_kernel_sum`.
- ELSE → `_exact_kernel_sum` (:208): chunk by `(2*dim+2)*nJ*nQ*itemsize <= kernel_chunk_bytes_resolved()` → LEAF `_eval_chunk` (:236):
  - IF `is_per and not is_rel and wrap != 'single-image' and _image_count_L(σ, P, k, 2) > 0` → per-coordinate `wrapped_gaussian_1d(..., 2)` product (abs-per full-image).
  - ELSE: abs-per (single-image or L = 0) → nearest-image wrap; `_compute_Q(D, r, is_rel, is_per, period, reduced=is_rel)`; `exp(-Q/(2σ²))`.
- `wrapped_gaussian_1d(d, σ, P, ts, exponent_denominator)` (_wrapped_kernel.py:103): `_prefer_fourier` (:88: `M < 2L+1` with `L = _image_count_L`, `M = _fourier_count_M`) → Fourier (Poisson) series; else image sum with `L` (L = 0 → single Gaussian). At the default ts=6: L(4) > 0 iff σ/P > 0.0589; L(2) > 0 iff σ/P > 0.0833; Fourier(4) from σ/P ≈ 0.2.

### 3.6a `_eval_core(centres, w_j, n_j, x, n_q, dim, sigma, r, is_rel, is_per, period, truncation_sigmas, wrap)` — eval.py:1543 (single-multiset centres evaluation)
- IF `ts finite > 0 and not is_per` → LEAF `_truncated_kernel_sum_culled` (eval.py:1422; bucket cull; `np.add.at`).
- ELSE: `value_table = _distinct_value_table(centres, n_q)` when `is_per and not is_rel and wrap == 'full-image'`; `bytes_needed <= mem_limit` → LEAF `_eval_full` (eval.py:1595) else chunked `_eval_full`.
- `_eval_full`: abs-per full-image → tabulated or dense `wrapped_gaussian_1d(..., 2)` product, floor `E > truncation_floor(ts)`; abs-per single-image → nearest-image wrap → `_compute_Q`; else `_compute_Q(reduced=is_rel)`; mask `q_total <= ts²/2`.

### 3.7 Leaves reached from ENTRY 3
| Leaf | File:line | Role |
|---|---|---|
| `eval_ma_orbit` → `eval_orbit_abs` | _ma_eval_orbit.py:30 / _mobius.py:1287 | factored Möbius, absolute attribute |
| `eval_orbit_rel` → `_eval_orbit_rel_fourier` / factored tabulation / direct u-grid | _mobius.py:1873 / 1731 | factored Möbius, relative attribute |
| `gaussian_kernel_sum` → `_truncated_kernel_sum_1d_vectorised` / `_truncated_kernel_sum` / `_truncated_kernel_sum_1d_circular` / `_eval_chunk` | _kernel.py:27 | block kernel sums |
| `_eval_core` → `_truncated_kernel_sum_culled` / `_eval_full` | eval.py:1543 | single-multiset centres and factored-MA per-attribute factors |
| `_ma_eval_factored` | eval.py:1077 | factored centres (A ≥ 2, all r ≥ 2) |
| `_ma_eval_full` | eval.py:1264 | joint-centres materialisation |
| `wrapped_gaussian_1d` | _wrapped_kernel.py:103 | abs-per full-image θ |
| `_compute_Q` / `_compute_Q_inner_blocks` | dispatch.py:848 / 745 | quadratic forms |

### 3.8 Overrides, guards, memo (ENTRY 3)
- `method='centres'`: skips every rule in `_select_ma_eval`; still subject to the shape rules inside the centres branch (single-multiset → `_eval_core`; `_ma_eval_factored` support predicate).
- `method='mobius'`: skips the cost model and the rel-per measure rule; subject to `_reject_ordered_for_mobius` (raises); **not** subject to the `r > 10` feasibility rule (would hit `_BELL_NUMBERS.get → inf` only in the cost model, which is skipped; `get_orbit_table` limits apply downstream).
- `factored=` on `eval_orbit_rel` is internal only (not reachable from `eval_exp_tens`).
- Guards: `SingleImageInfeasibleError` (forced centres > 4 GiB); Windowed with kernel cov (`NotImplementedError`); batched ordered r > 1 (`NotImplementedError`).
- No self-IP memo on the eval path (no cache).

---

## 4. ENTRY `entropy_exp_tens(p_or_dens, *args, spectrum, method, precision, dedup, base, n_points_per_dim, x_min, x_max, grid_limit, truncation_sigmas, kernel_precision, verbose, **legacy_kwargs)` — `entropy.py:352`

- [parse] `method` via `_canonicalize_method` (entropy.py:69): ∈ {`'differential'`, `'shannon'`, `'normalized'`, `'renyi2'`}; alias `'normalised'` → `'normalized'`; else `ValueError`; non-str → `TypeError`. Legacy `normalize=` kwarg → `TypeError` with migration message (:518); other unknown kwargs → `TypeError` (:520).
- IF `method == "differential"` → `_entropy_exp_tens_differential_dispatch` (:1012):
  - ERROR `NotImplementedError` on density lists / object arrays / 2-D raw single-multiset; `TypeError` if `precision is not None or dedup is not True`; `_resolve_density` (:1340); `_raise_if_any_sigma_zero`; Windowed → `NotImplementedError`.
  - `ts = resolve_truncation_sigmas`; LEAF `_differential_adaptive` (:896): `tol = max(exp(-ts²/2), 1e-12)`; spans from `_diff_spans_ma`; `eff_grid_limit = min(grid_limit, kernel_chunk_bytes//(16*(dim+1)))`; loop `N` doubling ≤ 10: `N**dim > eff_grid_limit` → ERROR `ValueError`; `H_disc = _entropy_exp_tens_ma(dens, normalize=False, n_points_per_dim=N, ..., truncation_sigmas=ts)`; `h_hat = H + Σ log δ`; converged when `|Δh| < tol`, or Richardson `|ΔR| < tol`, or Richardson differences stop shrinking (`dR >= 0.95*dR_prev`); else last Richardson/raw value. Delegates to §4.2 → cell masses (abs) or `eval_exp_tens` (rel).
  - kernel cov → `+ 0.5*logdet/log(base)`.
- IF `method in ("normalized", "shannon")` → `_entropy_exp_tens_shannon_dispatch(..., normalize=(method == "normalized"))` (:579):
  - density scalar (`len(args) > 0` → `TypeError`; `spectrum`/`precision` → `TypeError`; `_require_explicit_grid` → `TypeError` if `n_points_per_dim is None`) → `_entropy_exp_tens_scalar` (:1092) → `_entropy_exp_tens_ma` (§4.2).
  - density list → `_entropy_exp_tens_density_list` (:1123): `use_dedup = dedup and all single-multiset` → `_chord_canonical_key` cache; **`truncation_sigmas`/`kernel_precision` are not forwarded** on the list path.
  - raw MA (`_looks_like_ma_p`, :1637) → 6/7 args else `ValueError`; build → `_entropy_exp_tens_ma` (**no ts/kp forwarded**).
  - raw single-multiset 1-D → `add_spectra` if `spectrum`; build → `_entropy_exp_tens_ma(..., truncation_sigmas, kernel_precision)`. 2-D → `_entropy_exp_tens_raw_single_multiset_batch` (:1178).
- ELSE (`renyi2`) → `_entropy_exp_tens_renyi2_dispatch` (:743): `NotImplementedError` on lists / 2-D raw; `TypeError` on `precision`/`dedup`; `_resolve_density`; `_raise_if_any_sigma_zero`; dispatch message `('entropy_exp_tens','mobius','renyi2')`; LEAF `_renyi2_exp_tens_ma(dens, base)` (:1539) (§4.3); kernel cov → `+ 0.5*logdet/log(base)`.

### 4.2 `_entropy_exp_tens_ma(dens, normalize, base, n_points_per_dim, x_min, x_max, grid_limit, truncation_sigmas, kernel_precision)` — entropy.py:1657 (shannon/normalized grid)
- `dim == 0` → `0.0`. Bounds validation (`ValueError`). `n_points_per_dim**dim > grid_limit` → ERROR `ValueError`.
- IF `not is_windowed and not any(is_rel)` → LEAF `_cell_masses_ma_absolute(base_dens, axes, truncation_sigmas=ts or 6.0)` (:288): prune `w_j > 0`; per axis `_phi_diff_axis` (non-per erf) or `_phi_diff_axis_periodic` (:173 — **minimum-image erf; ignores `wrap` and the full-image measure**; `truncation_sigmas` unused); `_contract_cell_axes` (:225): `D <= 2` streamed in blocks of `_DIFF_CELL_BLOCK (8e6)`, else whole einsum.
- ELSE → **ENTRY 3** `eval_exp_tens(dens, X, verbose=False, truncation_sigmas, kernel_precision)` with `method='auto'` (:1763) — routes through `_select_ma_eval`.
- `H = -Σ q log_b q`; `normalize` → `/ log_b(N_cells)`.

### 4.3 `_renyi2_exp_tens_ma(dens_or_windowed, base)` — entropy.py:1539
- Windowed → `NotImplementedError`. `dens = pruned()`; `A == 0` → `0.0`; `N == 0` → NaN.
- per attribute: IF `nested[a] is not None or ordered_flat` (`not is_sym[a] and r > 1`) → LEAF `_renyi2_per_attr_numerical(dens, a)` (:1432): rebuild the attribute (`build_exp_tens` with `specs` or `is_sym=False`); overlap: `block_size >= 2` → `_compute_Q_inner_blocks(reduced=True)`; abs-per: `wrap_a == 'single-image'` → nearest-image `_compute_Q`; else `wrapped_gaussian_1d(D, σ, P, default ts, 4).prod`; else `_compute_Q(reduced=is_rel)`; mass `Z_a` from `_gaussian_mass_const`.
  ELSE → LEAF `_ma_per_attr_inner_matrix(Pa, Wa, Pa, Wa, sigma, r_a, is_rel, is_per, period)` (§1.6b; **no `wrap`, no `truncation_sigmas`** — global default and `'full-image'`) and `Z_a[n] = total_mass_rel/abs(pv, wv, sigma, r_a)` (_mobius.py:1142/1098).
- `ip_xx = Σ ∏_a I_a`; `Z = Σ_n ∏_a Z_a`; `_renyi2_finalise` (:1412): non-finite or ≤ 0 → NaN; else `-log_b(ip_xx/Z²)`.
- Delegation summary: `renyi2` → inner product (Möbius per-attribute matrix, never Bulger, never the flat selector, never the tuple-centres `choices` gate) + closed-form total mass; `shannon`/`normalized` → erf cell masses (absolute) or `eval_exp_tens` grid (relative/windowed); `differential` → repeated `shannon` on refined grids.
- Memo: none on the entropy path.

---

## 5. ENTRY `windowed_similarity(p_context, w_context, p_query, w_query, sigma, r, is_rel, is_per, period, centres, is_sym, start, stop, step, query_centres, context_window, query_window, window_attr, drop_window_attr, sweep, drop, locate, target_attr, normalize, specs, verbose)` — `_tensor/windowed.py:319`

- IF `sweep is not None` → ERROR if `drop is None`; ERROR if `query_centres`/`query_window` given → `_ws_multi` (:380). ELSE ERROR if `drop_window_attr is None` → `_ws_single` (:426).
- Both per position: window and translate the raw events (`_apply_windows`, `translate_attributes` — preprocessing, not traced), then:
  - IF `nested` (`specs is not None`) → `build_exp_tens(... specs=...)` ×2 → **ENTRY 1** `cos_sim_exp_tens(dc, dq, normalize=normalize, verbose=False)` (density path, `method='auto'`).
  - ELSE → **ENTRY 1** `cos_sim_exp_tens(pc, wc, pq, wq, sg, rr, rl, pr, pd, [sy], normalize=normalize, verbose=False)` (raw MA scalar path, `method='auto'`).
- No `method`, `truncation_sigmas`, or `kernel_precision` are exposed; every position routes through §1.4 with defaults. `_ws_single`: `drop_window_attr or rel_axis` → query not translated.
- `windowed_entropy` (:520) → **ENTRY 4** `entropy_exp_tens(dens, method=method, base=base)` per position (`method` default `'differential'`).

### 5.1 `windowed_tensor_similarity(dens_context, dens_query, window_spec, offsets, reference, mode, normalize/normalise, truncation_sigmas, kernel_precision, verbose)` — `_tensor/windowing.py:593` (routes differently: single closed-form path)
- `normalize` default `'oneSidedDenom'`; `_canonical_normalize`. → `_windowed_similarity_core` (:739): ERROR `NotImplementedError` on any kernel cov; `TypeError` if any operand is not a plain `MaetDensity`; empty lists → empty; list modes via `_resolve_list_list_mode`; per pair → `_windowed_similarity_pair` (:456) → per offset `window_tensor` + LEAF `_windowed_inner_product(dens_query, wmd, normalize)` (:899):
  - two-sided → `NotImplementedError`; `_check_ma_compatibility` (`ValueError`s).
  - `ip_qq = _cos_sim_numerator_ma(dens_q, dens_q, windowed_c=None)` (computed once per pair, :522 — not the `_self_ip_cache`).
  - `ip_qc = _cos_sim_numerator_ma(dens_q, dens_c, windowed_c=wmd)` (:1060): non-uniform windowed centres → average over within-attribute permutations (recursion with `_skip_symmetrisation=True`); per attribute Bulger-style perm×comb log-kernel: abs-per `wrap == 'full-image'` → `wrapped_gaussian_1d(D, σ, P, default ts, 4)`; abs-per single-image → nearest-image + `_compute_Q`; else `_compute_Q`; window contributions: periodic → `_periodic_image_sum_contribution` (:1405; `is_rel and d_g > 1` → falls to line-case `_windowed_group_contribution`; else per-axis image sum with `_IMAGE_SUM_TOL_DOUBLE = 1e-12`, cap 100 image pairs + `RuntimeWarning`); non-periodic → `_windowed_group_contribution` (:1545; `is_multi_rel and rho > 0` → `NotImplementedError`; 1-D or multi-abs → `_windowed_contribution_factorisable`; multi-rel Gaussian → `_windowed_contribution_gaussian_multi_rel`). **No truncation is applied** to this log-kernel (dense `exp`).
  - `'oneSidedDenom'` → `ip_qc/ip_qq` (0 → NaN); `'cosine'` → `_window_squared` (`mix ∉ {0,1}` → `ValueError`) → `ip_cc_h`; `/sqrt(ip_cc_h*ip_qq)`.
  - `truncation_sigmas` and `kernel_precision` are accepted but **never forwarded** to `_windowed_inner_product` (see Suspicious).
- No method routing, no memo.

---

## 6. `explain_dispatch(dens, other=None, n_q=None, method='auto', truncation_sigmas=None)` — `_tensor/explain.py:120`

Reports (does not route):
- eval (one density): re-runs `_select_ma_eval(dens, n_q or 200, method, truncation_sigmas)` (§3.4) and `_ma_eval_costs_ms` — identical decision to the real path for a plain `MaetDensity` passed directly (the real path uses the actual `n_q`).
- cosine, flat: calls `_select_ma_inner_product_method` (§1.5) with `rel_vec`, `k_vec_y`, `sym_vec`, `truncation_sigmas=ts`, `return_costs=True` — but **without `wrap_vec`** (so the wrap/measure override of rule 5 is never reported), **without `nu_vec`** (default 2000 per rel attribute instead of the real `auto_ntau_default`/span estimate), **without `skip_xx/skip_yy`** (memo state ignored), **without `guard_forced_bulger`** effects visible, and it does not apply the `ordered_any → bulger` rule, the `n == 0` short-circuit, `method='factored'`, or the post-hoc guard. Verified: for `r=3, K=4 vs 5, rel-per, σ/P=0.5, wrap='full-image'` explain reports `bulger (cost model)` while `cos_sim_exp_tens` chooses `mobius` (dispatch message observed).
- cosine, nested: re-runs `_nested_attr_route` (without skip flags), `_nested_enumeration_admissible`, `select_nested_method` and `_NESTED_ENUM_SAFETY` — same decision as `_try_nested_contract` for `method='auto'` (modulo skip flags); `method='bulger'` → reported as forced; `method='mobius'/'contract'/'centres'` → "contract".

---

## Orphans (routines in the routing modules with no caller on any traced path)

Evidence: `grep -rn --include=*.py "\b<name>\b" mpt` (definition + re-export lines only), plus the script `/tmp/.../scratchpad/orphans.py` (counts references package-wide). "re-export only" = the only non-definition reference is an import in `mpt/tensor.py` or `mpt/entropy.py`.

- `cosine._rel_contract_cheaper` (cosine.py:2407) — docstring says "Superseded"; referenced only in docstrings.
- `cosine._orbit_inner_abs` (cosine.py:3743), `cosine._orbit_inner_rel` (cosine.py:3881) — re-export only (tensor.py:92, entropy.py:21–22 imports, never called).
- `cosine._inner_product_direct_abs` (cosine.py:3790), `cosine._build_ordered_r_tuples` (cosine.py:3849; called only by `_inner_product_direct_abs` and the orphan module `_centres_inner`), `cosine._ma_has_nan` (cosine.py:2161), `cosine._attr_value_range` (cosine.py:2397) — no callers.
- `_mobius_inner._batched_direct_enum_abs` (:579), `_mobius_inner._pack_nan_top` (:555) — re-export only (tensor.py:93).
- `dispatch._orbit_ips_impossible` (:103), `dispatch._estimate_centres_array_bytes` (:1465), `dispatch._pw_per_entry_ms` (:223) with `_PW_PER_ENTRY_MS_NONPER/_PER` (:217/219), `dispatch._warn_rel_per_all_image` (:1147, no-op), `dispatch._format_time` (:1175) — re-export only / no callers.
- `dispatch._PROBE_MIN_N_Q` (:1188), `_CENTRES_PROBE_MEM_BUDGET` is used (guards) but `_PROBE_K_IP_TARGET` (:2221), `_PROBE_TIME_CACHE` (:2223), `_PROBE_IP_MOBIUS_DECISION_MARGIN` (:2234), `_PRESCREEN_IP_DOMINANCE` (:2248), `_ORBIT_IP_FIXED_OVERHEAD` (:2266) — no probe exists; `_probe_eval_path`/`_probe_ip_path` named in the module docstring (:16) are **not defined anywhere**.
- `_nested_contraction.make_quadrature` (:1205), `nested_ip` (:1227) and its private leaves `_ip_absolute` (:604), `_ip_rel_nonper`, `_ip_rel_nonper_factored`, `_ip_rel_nonper_generic` (:869/732/860), `_theta_truncation_L` (:572) — reachable only from tests (`nested_ip` referenced in 6 test files), never from `cos_sim_exp_tens`.
- `_tensor/_centres_inner.py` (`centres_inner_product`, :97) — whole module is test-only.
- `sweep.SweepNotEligible` (sweep.py:72) — defined, never raised or caught.
- `_mobius_inner._closed_form_attr_matrix_from` local `L_abs_per` (:1928–1937) — computed, never read.
- `_mobius._factored_worthwhile`'s partner docstring "cross-correlation strategy" — no such code path.

## Suspicious

1. **`explain_dispatch` diverges from the real flat cosine routing** (explain.py:305–314): omits `wrap_vec`, `nu_vec`, `skip_xx/skip_yy`, `ordered_any`. Reproduced: rel-per σ/P = 0.5 → explain says `bulger (cost model)`, actual = `mobius` via the wrap override (dispatch.py:693). Also its `measure` line reads "wrapped-difference approximation" for that case, which is the opposite of what runs.
2. **`ordered_any` silently overrides `method='mobius'`/`'centres'`** to Bulger on the flat path (cosine.py:1616), while the eval path raises `ValueError` for the same request (`_reject_ordered_for_mobius`). Inconsistent between entry points; `_maybe_show_dispatch_msg` announces "bulger (direct on ordered attributes)" but the returned method name differs from the requested one.
3. **`_ma_eval_factored` drops `wrap`** (eval.py:1189–1194 calls `_eval_core` without `wrap=`), so an abs-per attribute opted into `wrap='single-image'` is evaluated full-image on the factored MA centres route (A ≥ 2, all r ≥ 2), but single-image on the joint route (`_ma_eval_full`, eval.py:1366–1385) and on the single-multiset route (`_eval_core(..., wrap=dens.wrap[0])`, eval.py:931). Which measure runs therefore depends on the memory/shape branch.
4. **`_try_nested_contract_ma` flat branch drops `wrap` and `truncation_sigmas`** (cosine.py:3043–3050): `_ma_per_attr_inner_matrix` is called with defaults, so a flat abs-per single-image attribute tensored with a nested one is computed full-image with the global default truncation, unlike the same attribute on the flat Möbius route (cosine.py:2348).
5. **`_renyi2_exp_tens_ma` flat branch drops `wrap`** (entropy.py:1608) — same measure mismatch: single-image opt-in ignored for flat symmetric abs-per attributes, honoured for nested/ordered ones (:1511–1526).
6. **Shannon/normalized entropy on absolute-periodic densities uses the minimum-image erf** (`_phi_diff_axis_periodic`, entropy.py:173) regardless of `wrap`, i.e. measure "single-image", while the v3 default everywhere else is full-image. Its `truncation_sigmas` parameter is unused (:188). The `ts` passed to `_cell_masses_ma_absolute` falls back to a literal `6.0` when `None` (:1759) instead of `resolve_truncation_sigmas`.
7. **`windowed_tensor_similarity` accepts `truncation_sigmas` and `kernel_precision` but never uses them** (windowing.py:597–598, 822–827 → `_windowed_similarity_pair` never forwards them; `_windowed_inner_product` has no such parameters). Docstring at :747–750 claims they are forwarded. The windowed log-kernel is also never truncated.
8. **`_eval_exp_tens_scalar` on a `WindowedMaetDensity` ignores `method`** (eval.py:406–411): always `'auto'`. `kernel_precision` is not forwarded from `_eval_exp_tens_ma` to `eval_ma_orbit` (eval.py:898) nor to the single-multiset `_eval_core` fast path (eval.py:926).
9. **`_select_ma_eval` docstring promises a `K_a − r_a` precision guard** (dispatch.py:2040–2043) that does not exist in the code; the `eval_exp_tens` docstring (eval.py:160) promises a fallback "when the Möbius output contains non-finite values" — no such post-hoc guard exists on the eval path.
10. **`_select_ma_inner_product_method` docstring rule 3** ("rel + per with σ/P beyond ... warns and routes to Bulger", dispatch.py:536) is not what the code does: above the threshold the wrap override decides (full-image → Möbius); no warning is emitted.
11. **`wants_single`/`wants_full` scan all `rel_vec` attributes, not only rel-per** (dispatch.py:676–683). A density with one rel-nonper attribute (`wrap` default `'full-image'`) and one rel-per attribute with `wrap='single-image'` raises "Mixed rel-per wrap" although only one attribute is rel-per.
12. **Nested measure rule reads `get_default("truncation_sigmas")`**, not the call's `truncation_sigmas` (cosine.py:2523, 2622, 2769; `_nested_attr_matrix` uses the default `ts` for the contraction kernels and `_nested_attr_matrix` centres route uses the passed one when not `None` — but `_try_nested_contract` never passes one, so all nested arithmetic uses the global default regardless of the per-call `truncation_sigmas`). The flat path honours the per-call value. The nested memo key likewise uses the default.
13. **`_r1_broadcast_fast` requires `method in ("auto","bulger")`** but uses the `'bulger'` memo key while the per-pair `'auto'` route at r = 1 also resolves to bulger (selector rule 2) — consistent; however when the entries are `is_single_multiset` and `dedup=True`, the fast path pre-empts dedup (no canonical-key dedup on that shape). Behavioural, not wrong.
14. **`_ma_per_attr_inner_matrix` r = 1 branch consults `wrap` for `is_rel` attributes** (it evaluates the absolute kernel for rel r = 1, _mobius_inner.py:336), whereas `_ip_r1_direct`/`_r1_broadcast_fast` skip rel r = 1 attributes entirely (vanishing form). The two routes therefore differ by a constant per-attribute factor at r = 1 rel (cancels in the cosine, but not if a memo from one route were consumed by the other — it is not, keys differ).
15. **`_nested_enumeration_admissible` and `_nested_admissible_routes` ignore `dens_y.wrap`** (only `dens_x.wrap` is read), as does the flat selector (`wrap_vec_x`, cosine.py:1575) — asymmetric in the operands.
16. `sweep._finalise_orbit_sweep` receives `ts` but passes the raw `truncation_sigmas` (sweep.py:964–967); `kernel_precision`/`verbose` unused on the orbit route; `_orbit_self_ip` has no memo while `_self_ip` (mixture) does, so `method='orbit'` sweeps recompute both self IPs every call.
17. `eval_orbit_rel` docstring (mobius.py:1960) describes a "cross-correlation strategy" at r = 2 that does not exist.
18. `_cos_sim_density_path` empty-list handling swallows the `mode='auto'` mismatch `ValueError` and returns a cartesian-shaped empty array (cosine.py:1090–1097).
19. `dispatch.py` module docstring (lines 15–20) describes probe functions `_probe_eval_path`/`_probe_ip_path` that do not exist; the "pre-screen / cost-model / timing-probe / cancellation-guard" flow described there is not the implemented flow.

## Constants table

| Constant | Value | Defined | Used by |
|---|---|---|---|
| `truncation_sigmas` factory default | 6.0 | _defaults.py:75 | everywhere |
| `_ACCURACY_FLOOR_EPS` | 1e-12 (→ `accuracy_floor_sigmas()` = 7.4338) | _defaults.py:105 | `resolve_truncation_sigmas` |
| `post_hoc_guards` default | True | _defaults.py:79 | cosine.py:1702; _nested_contraction.py:248 |
| `orbit_cost_intercept` default | 3.8536 | _defaults.py:80 | `_orbit_cost.orbit_cost_log_ratio` |
| `rel_attr_route` default | `'auto'` | _defaults.py:88 | `_ma_rel_attr_prefers_centres` |
| `resolve_samples_per_sigma` | `max(2, ceil(sqrt(r ln(1/eps))/(π√2)) + 1)` → 3 (r ≤ 4), 4 (r = 5,6) at ts = 6 | _defaults.py:197 | `_rel_inner_batched`, `eval_orbit_rel`, `_ma_eval_costs_ms` |
| `_ORBIT_R_MAX_SHIPPED` | 8 | dispatch.py:976 | selector rule 3; `_orbit_eligible` |
| `_ORBIT_R_MAX_FEASIBLE` | 10 | dispatch.py:1220 | `_select_ma_eval` |
| `_REL_PER_DEPARTURE` | table (0.020…0.100 ↦ 1.67e-16…4.07e-2) | dispatch.py:996 | `_orbit_sigma_over_p_threshold` |
| `_REL_PER_PD_CEILING` | 0.05 | dispatch.py:1027 | `_orbit_sigma_over_p_threshold` |
| `thr(ts)` result | 0.03 at ts=6 (accuracy); 0.05 at ts ≤ 4 (PD ceiling); 0.03 at inf | dispatch.py:1030 | flat selector, eval selector, nested measure rule, sweep, `_ma_rel_attr_prefers_centres` |
| `_ABS_PER_SIGMA_OVER_P_THRESHOLD` | 0.04 | dispatch.py:1103 | build-time warning only (density.py:80) — nothing routes on it |
| `_CENTRES_PROBE_MEM_BUDGET` | 4 GiB | dispatch.py:1192 | `_guard_forced_bulger_feasible_ma`, `_select_ma_eval` |
| `_CENTRES_WORKING_SET_SOFT_BUDGET` | 256 MiB | dispatch.py:1213 | selector rule 4, eval safety factor, nested centres guard |
| `_REL_COST_LAW` | bulger {2:(−8.5858,0.8245),3:(−8.1774,0.7890),4:(−6.7951,0.7278)}; centres {2:(−8.8001,0.8195),3:(−10.0569,0.9269),4:(−9.7867,0.9251)}; grid {2:(−5.0440,0.4817),3:(−3.6893,0.5881),4:(−3.2639,0.7894)} | dispatch.py:340 | `_rel_route_cost_ms` |
| `_ORBIT_REL_FLOOR_MS` | {2:(0.05,0.067),3:(0.09,0.17),4:(0.0,2.65)} | dispatch.py:286 | `_predict_orbit_cost_ms` |
| `_ORBIT_ABS_PER_ATTR_MS` | {2:3.0,3:11.2,4:45,5:150,6:500,7:1500,8:4500} | dispatch.py:237 | `_predict_orbit_cost_ms` |
| `_PW_PER_ENTRY_MS_NONPER/_PER` | {2:1e-4/3,3:1.4e-4/3} / {2:1.1e-4/3,3:1.6e-4/3} | dispatch.py:217/219 | orphan (`_pw_per_entry_ms`) |
| `_MA_COST_CENTRES_SETUP_MS` … (22 linear constants) | 0.03843, 2.399e-05, 6.691e-4, 1.806e-05, 1.03e-4, 2.191e-06, 1.599e-06, 0.0015, 0.08384, 0.02509, 1.269e-06, 0, 3.538e-06, 2.406e-06, 5.52e-05, 1.296e-06, 1.801e-05, 3.324e-4, 2.476e-3, 4.321e-05, 1.388e-4, 0 | dispatch.py:1290–1432 | `_ma_eval_costs_ms` (via `_ma_cost_constants`) |
| `_MA_COST_CENTRES_CULL_C` | 14.32 | dispatch.py:1385 | `_ma_eval_costs_ms` |
| `_MA_COST_CENTRES_QUERY_JOINT_EXP_PER` / `_REL_PER` | 1.0 / 1.2 | dispatch.py:1352/1354 | `_centres_query_slope` |
| `_MA_MOBIUS_SAFETY` / `_SMALL` | 1.5 / 1.0 | dispatch.py:1460/1461 | `_select_ma_eval` |
| `_BELL_NUMBERS` (dispatch) | 1..115975 for r=1..10 | dispatch.py:1226 | `_ma_eval_costs_ms` |
| spectral eval gate thresholds | `n_q >= (16,32,64)`, `K >= (2,8,16)` for r=2,3,4 | _mobius.py:2046; dispatch.py:1885,1912 | `eval_orbit_rel`, cost model |
| `_FOURIER_MODE_SIGMAS` | 8.6 | _mobius.py:1728 | `_eval_orbit_rel_fourier` |
| `_FACTORED_EPS_CEIL`, `_FACTORED_CALIB_A6`, `_FACTORED_SPP_MIN/MAX`, `_FACTORED_READBACK_COST` | 1e-3, 1600, 8/512, 10.0 | _mobius.py:1585–1602 | factored strategy gate/tabulation |
| `_ORBIT_GRID_SLAB_ELEMS` | 2^19 | _mobius_inner.py:40 | `_rel_inner_batched` |
| `_ORBIT_SPARSE_MIN_KERNEL` / `_MAX_DENSITY` | 200_000 / 0.20 | _mobius_inner.py:81/84 | sparse gates (abs, rel-per) |
| `_SPECTRAL_IP_ENABLED` / `_FORCE` | True / False | _mobius_inner.py:746/759 | spectral IP gate |
| `_SPECTRAL_IP_MODE_SIGMAS` | 8.6 | _mobius_inner.py:767 | mode count |
| `_SPECTRAL_IP_MAX_POINTS` | 4_000_000 | _mobius_inner.py:773 | memory guard (never bypassed) |
| `_SPECTRAL_IP_COST_C` | 1100.0 | _mobius_inner.py:852 | spectral cost gate |
| `_CENTRES_NS_BASE/_LIN/_WRAP` | 45/3, 15/3, 10/3 | _mobius_inner.py:1440–1446 | `_predicted_centres_wall_ns` |
| `_GRID_NS_FLOOR`, `_GRID_NS_PER_OP` | 1e6 ns; {2:30,3:700,4:2000} | _mobius_inner.py:1449/1452 | `_predicted_grid_wall_ns` |
| `_rel_window_margin` | `min(√2·ts + 0.1, 8)` (8 if non-finite) | _mobius_inner.py:1360 | rel-nonper grid window |
| `_COMB_RESTRICTION_ENABLED` | True | _mobius_inner.py:1617 | centres X-side restriction |
| closed-form chunk | `16_000_000 // (njy*d)` | _mobius_inner.py:1944 | `_closed_form_attr_matrix_from` |
| `_orbit_cost` coefficients | 1.0708, 1.4033, −0.3608, −0.8188, −0.2261; `_MARGIN_R_MAX`=3 | _orbit_cost.py:33–43 | per-level orbit vs enumeration |
| `_ORBIT_ENUM_MAX_WORK` / `_ORBIT_ENUM_MAX_ELEMS` | 16e6 / 16e6 | _nested_contraction.py:229/230 | accuracy-guard fallback feasibility, chunking |
| `auto_ntau` | `max(64, ceil(2πP/σ·(1 + 0.5·max(0,−log10 tol)/12)))` | _nested_contraction.py:646 | tau grids (flat rel-per grid, nested taugrid, cost models) |
| `auto_taus_line` | step σ/4, pad `(6 + 0.5·max(0,−log10 tol))σ`, ≥ 64 | _nested_contraction.py:681 | rel-nonper line grid |
| `nested_attr_matrix` `mem_budget` | 16_000_000 | _nested_contraction.py:934 | chunking |
| `_NESTED_LAW_KEYS` | (2,3,4,6) | _nested_cost.py:91 | `_nested_cost_key` |
| `_NESTED_COST_LAW` | centres {2:(−2.9601,0.2919),3:(−5.0127,0.5157),4:(−3.6949,0.4074),6:(−4.0540,0.4886)}; taugrid (−7.3893,0.6125) all keys; contract_relnonper (−7.2386,0.5984) all; contract {2:(−1.4719,0.0472),3:(−1.2901,0.0001),4:(−2.3945,0.1239),6:(−2.2457,0.0969)}; bulger {2:(−4.1758,0.4428),3:(−6.3699,0.6668),4:(−3.6142,0.4489),6:(−2.6820,0.4636)} | _nested_cost.py:147 | `nested_route_cost_ms` |
| `_NESTED_FLOOR_MS` | centres {2:0.08577,3:0.1201,4:0.0861,6:0.1289} per matrix; taugrid 0.1594; contract_relnonper 0.1548; contract {0.08933,0.0895,0.0604,0.0691}; bulger {0.05703,0.09857,0.05533,0.1006} (fixed part 0) | _nested_cost.py:165 | `nested_route_cost_ms` |
| `_NESTED_ENUM_SAFETY` | 2.0 | _nested_cost.py:195 | `select_nested_method` |
| `_ORBIT_WORK_RATIO` / `_ORBIT_MIN_PAIRS` | 64.0 / 1e6 | sweep.py:523/532 | `_choose_sweep_route` |
| `_OFFSET_OVERHEAD_IN_COMPONENTS` | 512 | sweep.py:462 | `_evaluate_mixture` dense-vs-culled |
| r1 broadcast cache cap | `4_000_000 // (n_j*8)` columns | cosine.py:755 | `_r1_broadcast_fast` |
| `_gram_is_accurate_enough` factor | `eps·s²/(4σ²) <= 0.1·floor` | cosine.py:1975 | `_ma_log_kernel` |
| `_tuple_values_repeat` | `min_queries=100, min_entries=256, min_saving=4` | eval.py:1200 | value tables |
| `_image_count_L(e_d)` | `ceil((σ/P)·sqrt(−e_d ln floor) − ½)`: L>0 iff σ/P > 0.0589 (e_d=4) / 0.0833 (e_d=2) at ts=6 | _wrapped_kernel.py:44 | all abs-per full-image branches, L=0 short-circuits |
| `_prefer_fourier` | `M < 2L+1` (≈ σ/P > 0.2 at e_d=4) | _wrapped_kernel.py:88 | `wrapped_gaussian_1d` |
| `_rel_per_image_count` | `ceil(2(σ/P)·sqrt(ln 1/tol) − ½)` | _mobius_inner.py:855 | flat rel-per grid image sum |
| `_IMAGE_SUM_TOL_DOUBLE` | 1e-12; cap 100 image pairs | windowing.py:55, 1501 | windowed periodic image sum |
| `_DEFAULT_GRID_LIMIT` / `_DIFF_CELL_BLOCK` | 1e8 / 8e6 | entropy.py:30/38 | entropy grid guard / streaming |
| differential `tol` | `max(exp(−ts²/2), 1e-12)`; `max_iter=10`; Richardson stall `0.95` | entropy.py:915–1003 | `_differential_adaptive` |
| impossible-value cosine tolerance | `1.000001` | dispatch.py:169 | post-hoc guard |

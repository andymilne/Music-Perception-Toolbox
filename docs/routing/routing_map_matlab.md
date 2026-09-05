# Routing map — MATLAB half of the Music Perception Toolbox

Source root: `/home/claude/mpt/matlab`. All `file:line` cites are relative to
that root. Read from source (no MATLAB execution); reachability established
with `grep -rn` over `*.m` excluding `tests/`, `tools/`, `demos/`.

Conventions: `ts` = resolved truncationSigmas
(`internal.accuracyFloor('resolve', …)`, `+internal/accuracyFloor.m:76-88`:
`[]` → `mptDefaults('truncationSigmas')` (factory 6, `mptDefaults.m:191`),
`Inf` → `sqrt(-2 ln 1e-12)` ≈ 7.43); `threshold(ts)` =
`internal.relPerSigmaOverPThreshold(ts)` (table lookup, PD ceiling 0.05,
`+internal/relPerSigmaOverPThreshold.m:60-84`); `wrap(a)` = the density's
per-attribute wrap cell, `'full-image'` when absent.

---

## 1. ENTRY `cosSimExpTens(varargin)` — `cosSimExpTens.m`

```
- ENTRY cos_sim_exp_tens(x, y, …)                                   (cosSimExpTens.m:1)
  - [parse] method ∈ {'auto','bulger','centres','mobius','contract'}   (:300-308)
      ERROR cosSimExpTens:badMethod otherwise. No retired synonyms accepted.
      'cancellationThreshold' parsed (:313-322) but INERT: threaded into
      localCosSimMA and never read (comment at :278 "accepted for compatibility; inert").
      'normalize'/'normalise' ∈ {'cosine','oneSidedDenom'} (:335-347); needXX = normalize=='cosine'.
      'truncationSigmas','kernelPrecision' captured (:323-334); kernelPrecision is
      forwarded only on the batched-raw path (:2171-2173) — every other cosine path ignores it.
      'spectrum','precision','dedup' only in batched-raw (:418-433, ERROR *NotApplicable otherwise).
  - IF nArgs == 10 → pop trailing isSym → symArgs                                    (:389-398)
  - willBatch = nArgs==9 && isnumeric(p1)&&isnumeric(p3) && (P1 is 2-D || P2 is 2-D)  (:412-417)
  - ERROR cosSimExpTens:windowedNotSupported IF either struct tag == 'WindowedMaetDensity' (:441-451)
  - IF nArgs == 2
    - IF both structs, both tag 'MaetDensity'                                        (:496-501)
      - seed cacheX/cacheY from a.selfIP / b.selfIP (localSelfIpFromStruct :2224)
      - IF isSingleMultiset(a) && isSingleMultiset(b) → prune both, FALL THROUGH to shared MA (:507-512)
      - ELSE → NODE cos_sim_exp_tens_ma (localCosSimMA, :514) and return
    - ELIF iscell(a) || iscell(b) → NODE cos_sim_density_list (:540)
      - (cell,cell) → recursive cosSimExpTens(a{i}, b{i}, normalize, verbose) per pair (:1763)
        (method/truncationSigmas/cancellationThreshold NOT forwarded → per-pair method='auto')
      - (struct,cell)/(cell,struct) → NODE r1_broadcast_fast (localR1BroadcastFast :1801)
        - IF ok → LEAF ip_r1_direct batched (see §1.4) and return
        - ELSE per entry → localScalarPairDispatch (:1815/:1819) → localCosSimMA with
          method='auto', cancellationThreshold=1e-12, truncationSigmas=[] (:1863) — user
          method/truncationSigmas are NOT forwarded in list mode.
    - ELSE ERROR cosSimExpTens:badPairTypes
  - ELIF nArgs == 9
    - IF iscell(a)||iscell(c)  (MA raw)
      - ERROR listVsListNotSupported IF both are lists of pAttr (:577)
      - IF neither is a list → buildExpTens×2 → localCosSimMA (:584-591)
      - ELSE scalar-vs-list: build once + per-entry;
        - IF method ∈ {'auto','bulger'} → try r1_broadcast_fast (:616-624)
        - ELSE / not ok → per-entry localCosSimMA with the scalar side's cache threaded (:627-643)
    - ELIF willBatch → NODE cos_sim_batched_raw (localCosSimBatchedRaw :692)
      - ERROR mpt:aniso:batchedUnsupported IF isKernelCov(sigma) (:651)
      - ERROR batchedRowMismatch unless rows match or one side is a single row (:684)
      - ERROR batchedOrderedUnsupported IF isSym given && any(~isSym) && r > 1 (:1917)
      - canonicalise+dedup rows (internal.pairCanonicalKey :2023) → unique (A,B) pairs
      - per unique pair → recursive cosSimExpTens(dA, dB, 'method', method,
        'cancellationThreshold', ct, 'normalize', nz, 'verbose', false
        [, 'truncationSigmas'][, 'kernelPrecision']) (:2165-2178) → localCosSimMA.
        (batchCosSimExpTens.m is a deprecated shim onto this path.)
    - ELSE single-multiset raw: buildExpTens×2 → FALL THROUGH (:714-717)
  - ELSE ERROR cosSimExpTens:wrongArgCount
  - shared tail: localCosSimMA(maet_x, maet_y, method, normalize, ct, verbose, ts, cacheX, cacheY) (:741)
```

### 1.1 NODE `cos_sim_exp_tens_ma` — `localCosSimMA` (`cosSimExpTens.m:821`)

```
- ERROR mpt:aniso:covMismatch IF ~kernelCovsCompatible(dens_x, dens_y)            (:866)
- prune both (internal.prunedExpTens)                                              (:873-874)
- IF dens_x.N == 0 || dens_y.N == 0 → s = 0 (return; no arithmetic)                (:883-886)
- ERROR nAttrs/r/sigma/isRel/isPer/periodMismatch                                  (:888-912)
- wrapG = dens_x.wrap or all 'full-image'                                          (:928-932)
- innerR(a) = prod(spec.r(1:relUnit)) for nested attrs with proj ∈ {inner,intermediate}, else 0 (:938-948)
- kVec/kVecY = per-attribute value counts; anyPer/anyRelNonper/anyRelPer/sigmaOverPMax (:954-973)
- nuVecSel(a): rel&&r_a>=2: periodic → internal.autoNtauDefault(P, sigma) (:987);
  non-periodic → max(64, ceil(max(span,1)/sigma*10)) with span = rangeX+rangeY+2*relWindowMargin(mptDefaults ts)*sigma (:989-999)
- skipXX = ~needXX || selfIpMemoised(cacheX); skipYY = selfIpMemoised(cacheY)      (:1016-1017)
- nestedAny = any nonempty dens_x.nested or dens_y.nested                           (:1024-1027)
- chosen = NODE select_ma_inner_product_method(…, method, …, wrapG, ts, skipXX, skipYY, symVecSel, guardForcedBulger=~nestedAny) (:1033)
- OVERRIDE (silent) IF any ordered attribute (isSym(a)==false && r_a>1) on either side → chosen='bulger' (:1044-1056)
    — overrides the selector's output AND an explicit method='mobius'/'centres'.
- ERROR cosSimExpTens:contractUnavailable IF method=='contract' && ~nestedAny        (:1067)
- IF nestedAny
  - IF method ∈ {'auto','contract','mobius','centres'} → NODE try_nested_contract
      (internal.nestedContract(dens_x, dens_y, normalize, ts, force=~(method=='auto'), opts{methodName, caches, forceRoute='centres' iff method=='centres'})) (:1091-1107)
  - chosen = 'bulger' regardless (used only if contractTriple is empty)             (:1116)
  - (method=='bulger' never consults the plan.)
- announce via internal.maybeShowDispatchMsg                                        (:1160)
- IF contractTriple nonempty → triple taken from the plan; skip both arms          (:1163-1168)
- ELIF chosen == 'centres' → LEAF cos_sim_exp_tens_ma_centres                       (:1170-1211)
    ensureExpTensExpensive; memo key selfIpKey('centres', ts, ''); ip_xy = ipCoreMA(U_perm_x, U_perm_y)
    (perm-vs-perm, unrestricted O(K^{2r})); ip_xx only if needXX (memo read/write); ip_yy (memo read/write)
- ELIF chosen == 'mobius' → NODE cos_sim_exp_tens_ma_orbit (localCosSimMAOrbit, userForcedMobius = method=='mobius') (:1213)
  - post-hoc GUARD orbit_ips_impossible IF mptDefaults('postHocGuards')             (:1225-1228)
      impossible := any non-finite | ip_xx<0 | ip_yy<0 | |ip_xy| > 1.000001*sqrt(ip_xx*ip_yy)  (:752-814)
    - IF impossible → warning mpt:cosSimExpTens:impossibleValue; purge 'mobius|…' memo entries (:1239-1240); chosen='bulger' → FALLBACK to pairwise
- IF ~ranOrbit → LEAF cos_sim_exp_tens_ma_bulger (pairwise, :1250-1305)
    ensureExpTensExpensive; memo key selfIpKey('bulger', ts, ''); ip_xy = ipCoreMA(U_perm_x, V_comb_y);
    ip_xx = ipCoreMA(U_perm_x, V_comb_x) iff needXX && ~hit; ip_yy likewise.
- normalise: 'cosine' → denom = sqrt(max(ip_xx*ip_yy,0)) (ERROR missingSelfIp if ip_xx empty); 'oneSidedDenom' → ip_yy; denom==0 → NaN (:1311-1326)
```

#### 1.1a Pairwise kernel core `ipCoreMA` (`cosSimExpTens.m:1333`) — shared by the bulger and centres arms

```
- IF all(rVec == 1) && all(innerR == 0) → LEAF ip_r1_direct (localIpR1Direct :2386)   (:1353-1357)
    per attribute: rel → skipped (vanishing form); abs-per full-image → wrapped_gaussian_1d(d, σ, P, ts, 4) log-accumulated;
    abs-per single-image → nearest-image reduce then -d²/(4σ²); abs non-per → -d²/(4σ²);
    threshold = -0.5 ts² - log(nJ*nK) (nTerms>1); chunk by kernelChunkBytesResolved (:2406-2408)
- ELSE bytesNeeded = (max r + 2)*nJ*nK*8 vs internal.kernelChunkBytesResolved()      (:1360-1365)
  - IF fits → LEAF ip_full_ma (single truncLogKernelExp on maLogKernel)               (:1366, :1398-1403)
  - ELSE chunk comb side, same arithmetic                                             (:1368-1386)
- maLogKernel per attribute (:1405-1496):
  - IF ~isPer(a) && gram_is_accurate_enough(U,V,σ,ts) → LEAF gram_quadratic_form (:1418-1431)
      predicate: eps * s² / (4σ²) <= 0.1 * truncationFloor(ts), s = max |coordinate - U(1)| (:2240-2266)
      blockSize = innerR(a) if >0, r_a if isRel, else 0 (:1420-1426)
  - ELIF innerR(a) > 0 → qInnerBlocks (block-diagonal, per-block pairwise wrap if per) (:1436-1443)
  - ELIF isPer && ~isRel && wrap(a)=='full-image' && wrappedKernelImageCount(σ,P,ts,4) > 0
        → wrapped_gaussian_1d(D, σ, P, ts, 4), log-summed over coordinates (:1451-1478)
        (L == 0 short-circuit: falls through to the nearest-image Q form, same number)
  - ELSE: IF isPer && ~isRel → nearest-image reduce (:1488-1491); Qa = computeQaMA:
      rel&&per → pairwise-wrapped Σ_{i<j} wrap(d_i-d_j)² / r_a; rel&&~per → Σd² - (Σd)²/r_a; abs → Σd² (:1526-1553)
```

#### 1.1b NODE `select_ma_inner_product_method` — `+internal/selectMaInnerProductMethod.m:1`

Rules in order (returns as soon as one fires):

```
1. IF ~strcmp(userMethod,'auto') → chosen = userMethod (pwCost/orbitCost = NaN)          (:92-95)
     OVERRIDE: bypasses r_max rules, the feasibility guard, the wrap rule, and the cost model.
2. r_max = max(rVec) (1 if A==0). IF r_max <= 1 → 'bulger'                                 (:96-103)
3. IF r_max > 8 (_ORBIT_R_MAX_SHIPPED)                                                      (:104-115)
     IF guardForcedBulger → GUARD guard_forced_bulger_feasible(kVec, rVec, Nx, Ny, kVecY, symVec)
        ERROR mpt:dispatch:singleImageInfeasible IF nJ_x*nJ_y*8 > internal.dispatchMemBudget()
        (nJ side = N*Π K!/(K-r)! unordered, /r! ordered, saturating at 1e18; budget =
        clamp(availableMemory/2, 1 GB, 4 GB), +internal/dispatchMemBudget.m:17-20)
     → 'bulger'
4. (Docstring lists a "K-vs-r precision guard" rule (4); NO such code exists — see Suspicious.)
5. Rel-per wrap rule (only IF ~isempty(wrapVec) && anyRelPer && ~isempty(relVec)):          (:150-177)
     wantsSingle/wantsFull over relative attributes; ERROR mpt:mixedRelPerWrap IF both.
     IF sigmaOverPMax > threshold(ts): wantsSingle → 'bulger'; wantsFull → 'mobius'.
6. Cost race                                                                                (:179-192)
     pwSize = predictPairwiseKernelSize (nJ^X·nK^Y + [nJ^X·nK^X unless skipXX] + [nJ^Y·nK^Y unless skipYY]; Inf if any K < r)
     pwCost = relRouteCostMs('bulger', r_max, pwSize)   (exp(a_r)·term^b_r, rows r=2,3,≥4)
     centresOk = sigmaOverPMax <= threshold(ts)
     orbitCost = predictOrbitCostMs(rVec,kVec,A,Nx,Ny,relVec,nuVec,centresOk,kVecY,skipXX,skipYY)
     IF pwCost <= orbitCost → 'bulger' ELSE 'mobius'   (ties → bulger)
```

`predictOrbitCostMs` (`+internal/predictOrbitCostMs.m`): per attribute, rel && r_a>=2 →
`perPair = relRouteCostMs('grid', r_a, pairs·nu_a·max(K_a,K_aY)·(nMatrices/3))`; if
`centresOk && K_a>=r_a && K_aY>=r_a` also `min(…, relRouteCostMs('centres', r_a, pairs·(mX·mY [+mX² unless skipXX][+mY² unless skipYY])))`
with `m = r_a!·C(K,r_a)`; then `max(perPair, floor(r_a))` with ORBIT_REL_FLOOR_MS; abs r_a>=2 →
`ABS(r_a)·nMatrices/3` (:53-113). Terms priced: event-pair count × grid nodes × larger value count
(grid), tuple-pair entries (centres/bulger).

#### 1.1c NODE `cos_sim_exp_tens_ma_orbit` — `localCosSimMAOrbit` (`cosSimExpTens.m:1564`)

```
- truncResolved = ts or mptDefaults                                                (:1598-1602)
- choices(a) = flatAttr(a) && NODE ma_rel_attr_prefers_centres(Px,Py,σ,r_a,isRel,isPer,P,ts,userForcedMobius) (:1606-1614)
- memo key selfIpKey('mobius', ts, char('0'+choices)) — per-attribute route bits keyed (:1615-1616)
- per attribute:
  - IF choices(a) → LEAF closed_form_attr_matrix_from(closedFormAttrCentres(x,a), closedFormAttrCentres(y,a), wrapA)  (:1656-1672)
      NOTE: truncationSigmas is NOT passed here (4th arg omitted) → the closed form resolves mptDefaults, ignoring a per-call override.
  - ELSE → NODE mobius_inner (mobius.maPerAttrInnerMatrix(…,'truncationSigmas',truncResolved,'wrap',wrapA)) (:1682-1694)
  - P_xx / P_yy formed only when not memoised (and P_xx only if needXX)
- ip = sum of elementwise products; self terms written to memo                     (:1698-1712)
```

#### 1.1d NODE `ma_rel_attr_prefers_centres` — `+mobius/maRelAttrPrefersCentres.m:1`

```
- IF ~isRel || r_a < 2 → false                                                     (:81-83)
- blockedByMeasure = isPer && sigma/period > threshold(ts); emptyTupleSet = K<r_a || K_y<r_a (:84-87)
- forced = mptDefaults('relAttrRoute') ('auto' default)
- IF forced=='auto' && userForcedMobius → false (explicit method='mobius' keeps the grid)  (:92-103)
- switch forced: 'mobius' → false; 'centres' → ERROR mpt:relAttrRouteBlocked IF blockedByMeasure or emptyTupleSet, else true (:104-125)
- IF blockedByMeasure || emptyTupleSet → false                                     (:127-129)
- cost: cWallNs = (15 + 5(r_a-1) [+ (10/3)(r_a-1)(r_a-2) if isPer]) · (M_x M_y + M_x² + M_y²), M = r_a!·C(K,r_a) (:138-145)
        gWallNs = 1e6 + gOp(r_a)·n_u·K   with gOp = {2:30, 3:700, 4:2000, ≥5: 2000·3^(r_a-4)};
        n_u = autoNtauDefault(P,σ) (per) | max(64, ceil(max(span,1)/σ·10)) (non-per, margin from mptDefaults ts) (:147-182)
- true IF cWallNs < gWallNs                                                          (:184)
```

#### 1.1e LEAF chain `closed_form_attr_centres` / `closed_form_attr_matrix_from`

- `+mobius/closedFormAttrCentres.m:52-110`: rebuilds the attribute as a one-attribute density
  (`buildExpTens … 'lazy', false`, nested spec forwarded with isRel=false :85-92); `bundle.comb`
  = comb-side restriction (`localCombRestriction` :130-233): declined ([]) IF
  `~internal.combRestrictionEnabled()` (:203; default true) | flat r_a<2 (:213) | nested
  `nestedOrbitMult < 2` (:219) | `nK<=0 || nJ ~= mult*nK` (ordered attribute / unexpected tiling :224).
- `+mobius/closedFormAttrMatrixFrom.m:70-163`: IF combX nonempty AND combY has the same mult and
  `size(Cy,2) == mult*size(combY.Centres,2)` → X side restricted to comb reps ×mult (:87-99), else
  unrestricted perm-vs-perm. `absPerFullImage = isPer && ~isRel && wrapA=='full-image' && bs<2 &&
  wrappedKernelImageCount(σ,P,ts,4) > 0` (:133-136; L==0 short-circuit → nearest-image Q).
  Kernel: full-image → `prod(wrapped_gaussian_1d(D,σ,P,ts,4),1)`; `bs>=2` → block Q
  (`localComputeQBlocks`, no shipped caller reaches it :48-55); else `localComputeQFlat`
  (rel-per pairwise wrap /r, rel-nonper Σd²-(Σd)²/r, abs Σd² with nearest-image wrap) (:145-158).
  No truncation of the exp (no cutoff applied). Chunk = floor(16e6/(njy·d)) (:138).

#### 1.1f NODE `mobius_inner` — `+mobius/maPerAttrInnerMatrix.m:1`

```
- IF pruneZeroWeightEvents && any zero-weight column → drop, recurse, scatter (all-zero if a side empties) (:107-130)
- IF r == 1 → LEAF r1_zero_pad (localR1ZeroPad :233)                               (:136-140)
    absPerFullImage = isPer && wrap=='full-image' → wrapped_gaussian_1d(diffs,σ,P,ts,4); else nearest-image wrap (if per) + truncKernelExp; ×σ√π
- IF isRel → NODE rel_inner_batched (mobius.relInnerBatched, 'truncationSigmas')   (:143-147)
- ELSE (r>=2 abs): safe_x_mask = true(1,Nx) ALWAYS (:161-162) → every event "safe"
  - sparse gate: IF ~isPer && r>=2 && KxMax*KyMax >= 200000 && nnz(K0) <= 0.20·KxMax·KyMax
      (K0 = sparse kernel of the first event pair, radius √2·ts·σ) → LEAF safe_safe_orbit_sparse (:180-194)
        per pair: localBuildSparseKernelAbs → mobius.innerProductOrbitSparse(Ksp,w,w,r,'prefactor',(σ√π)^r)
  - ELSE → LEAF safe_safe_orbit (localSafeSafeOrbit :292): dense (Kx,nc,Ky,Ny) kernel,
      full-image → wrapped_gaussian_1d(diffs,σ,P,ts,4) else nearest-image + truncKernelExp; chunk on
      kernelChunkBytesResolved; → mobius.innerProductOrbitPwBatched(K_pairs,Wx,Wy,r,'prefactor',(σ√π)^r)
  - DEAD: unsafe_x_idx/unsafe_y_idx are always empty (:164-166) → localFillDirectEnumGroups (:471),
      localPackNanTop (:515), localBatchedDirectEnumAbsSingleMultiset (:540) are unreachable.
```

#### 1.1g NODE `rel_inner_batched` — `+mobius/relInnerBatched.m:1`

```
- samplesPerSigma = resolveSamplesPerSigma([], r, ts) = max(2, ceil(ts√r/(2π))+1)   (:69-70)
- NaN → zero weight; sharedW = every event column has identical weights            (:79-90)
- IF internal.spectralIpEnabled() && 2<=r<=4 && ~wantRatio → NODE spectral_rel_inner_matrix (:99-107)
    (mobius.spectralRelInnerMatrix(…, forceBranch = internal.spectralIpForce()))
    - L = period (per) | spanX+spanY+2(8.6+2)σ (non-per; decline [] if ~finite or <=0) (:114-123)
    - M = ceil(8.6/√2 · L/(2πσ)) + 2; gridSize = (2M+1)^(r-1); IF gridSize > 4e6 → [] (memory guard, never bypassed) (:126-130)
    - IF ~forceBranch && gridSize > COST_C·Kx²·Nx·Ny, COST_C = 1100 → [] (cost gate)  (:136-140)
    - ELSE → LEAF spectral Gram matrix (full-image measure)
    - IF nonempty → return
- grid: per → N_u = autoNtauDefault(P,σ), uGrid over [0,P); non-per → per-pair midrange-centred window,
    N_u = max(64, ceil(max(span,1)/σ·spp)), span = spreadX+spreadY+2·relWindowMargin(ts)·σ (:109-142)
- IF isPer && r>=2 && Kx*Ky >= 200000 && 2√cutoff < P(1-1e-8) && nnz(K0) <= 0.20·Kx·Ky
    (cutoff = 2(ts σ)²) → LEAF rel_per_inner_sparse (per u-node sparse kernel → innerProductOrbitSparse) (:155-176)
- ELSE slab loop (SLAB_ELEMS = 2^19): per u-chunk, per → nearest-image wrap then full-image sum of
    2·relPerImageCount(σ,P,ts)+1 truncKernelExp images (:220-239); non-per → truncKernelExp (:241)
  - IF sharedW → LEAF inner_product_orbit_grid(K_uc, Wx(:,1), Wy(:,1), r)            (:246-255)
  - ELSE → LEAF inner_product_orbit_pw_batched(K_uc, wA_uc, wB_uc, r)                 (:256-270)
- I = (σ√π)^r · I / (σ√(2π/r))²                                                      (:289-290)
```

#### 1.1h LEAF `wrapped_gaussian_1d` — `+internal/wrappedGaussian1d.m:1`

```
- IF wrappedKernelPreferFourier(σ,P,ts,e_d)  [M < 2L+1]  → Fourier (Poisson) sum with M = wrappedKernelFourierCount (:62-79)
- ELSE L = wrappedKernelImageCount; dRed = nearest image;
  - IF L == 0 → exp(-dRed²/(e_d σ²)) (single term)                                  (:87-90)
  - ELSE image sum over n = -L..L                                                    (:91-95)
- L = ceil((σ/P)√(-e_d ln tol) - ½) (0 if <=0); M = ceil((P/(πσ))√(-ln tol / e_d)) (min 1);
  tol = eps floor (1e-12) if ts == accuracyFloor('sigmas') else exp(-ts²/2)  (+internal/wrappedKernelImageCount.m:22-37, wrappedKernelFourierCount.m:22-38)
```

#### 1.1i NODE `try_nested_contract` — `+internal/nestedContract.m:1`

```
- IF opts.termsOnly → localNestedTerms (cost harness only; not a route)             (:122-130)
- decline (return [] ; ERROR cosSimExpTens:contractUnavailable IF force) when:
    normalize ∉ {cosine, oneSidedDenom} (:131) | nAttrs differ (:136) |
    [A==1:] no nested cell (:152) | spec empty (:160) | localInnerR ≠ 0 (inner/intermediate [rel] unit) (:165) |
    [r]/[sym] differ (:201)
- IF nAttrs ~= 1 → NODE nested_contract_ma (:141-151)
- ts = accuracyFloor('resolve'); orbitGuard('begin', ts) (postHocGuards read here :1684)
- [skipXX, skipYY] = selfIpSkipFlags (needXX / selfIpMemoised)                      (:240)
- [route, quad] = NODE nested_attr_plan(densX, densY, 1, forceRoute, ts, skipXX, skipYY) (:241)
- IF ~force && ~routesOnly && NODE nested_prefers_enumeration({route}) → return [] (caller runs Bulger) (:250-256)
- memo key selfIpKey('contract', ts, routeSignature(route, quad)) (route + ntau + tau endpoints) (:272)
- IF route == 'centres' → LEAF closed_form_attr_matrix_from(cxB, cyB, wrapA, ts) summed (ts IS passed here) (:276-305)
- ELSE recipes (buildRecipe) → tripSum(recipeX, recipeY, …, quad, sym) (:311-330)
    ip_xx computed whenever not memoised, even under oneSidedDenom (skipXX affects pricing only)
```

`nested_contract_ma` (`:997-1212`): per attribute kinds ∈ {nested, ordered, flat}; declines
(force → ERROR) IF nested on one side only (:1056) | inner unit (:1064) | [r]/[sym] differ (:1069).
Nested → `nested_attr_plan`; then `nested_prefers_enumeration(attrRoutes)` (:1094) may return []
to the enumeration. Memo key `selfIpKey('contract_ma', ts, 'kind:a:routeSig,…')` (:1118).
Pass 2: flat → `mobius_inner` with `'truncationSigmas', ts, 'wrap', wrapA` (:1145-1157);
ordered (isSym=false, r>1, not nested) → `closed_form_attr_matrix_from(cxB, cyB, wrapA, ts)`
(:1167-1178); nested → `nestedAttrMatrices` (:1181) → centres bundle or `nestedAttrInnerMatrix`.

`nested_attr_plan` (`:341-377`): route = `nested_attr_route`; quad = [] for centres else
`makeQuadrature(isRel, isPer, σ, P, vmin, vmax, ts, wrap)` (:2080-2108): abs → single-node
`{mode 'abs', isPer, wrap}`; rel-per → taus = `linspace(0,P,autoNtauDefault(P,σ)+1)(1:end-1)`
(ntau from the GLOBAL mptDefaults ts, not the per-call ts); rel-nonper → `n = max(64,
ceil(2·hi/(σ/4)))`, hi = spread + (6 + 0.5·max(0,-log10 tol))σ, tol = max(exp(-ts²/2), 1e-12).

`nested_admissible_routes` (`:380-433`) — the measure rule:
- `~isRel` → {'contract'}; `isRel && ~isPer` → {'centres','contract_relnonper'};
- rel-per: IF `period > 0 && σ/P > threshold(ts)` → {'centres'} if wrap=='single-image' else {'taugrid'}; ELSE {'centres','taugrid'}.

`nested_attr_route` (`:436-531`):
- forced route mode check `localCheckForcedMode` (ERROR cosSimExpTens:centresUnavailable IF the forced name has no carrier for (rel,per); unknown name → same id) (:468-471, :534-567)
- `~isRel` → 'centres' iff forced=='centres' else 'contract' (cost model never consulted for absolute attributes) (:472-479)
- rel-per above threshold (1 admissible): admissible 'taugrid' && forced=='centres' → ERROR centresUnavailable; admissible 'centres' && forced ∉ {'', 'centres'} → ERROR; else route = the sole carrier (:483-513)
- forced=='centres' → 'centres'; any other forced name → honoured as given (:514-524)
- 1 admissible → it; ELSE → `internal.nestedCost('priceNestedAttr', …)` (:525-530)

`nested_cost` (`+internal/nestedCost.m`): `priceNestedAttr` → `localPriceAttr` (:375-411):
per admissible route `cost = exp(a)·max(term,1)^b` floored by `f + pm·nMatrices`, law row =
largest key in [2,3,4,6] ≤ total order (:319-350); MEMORY GUARD: `route=='centres' &&
numel(admissible)>1 && workingSetBytes > 256 MiB` → cost = Inf (:398-400), workingSet =
max(nJx,nJy)·2·dim·8 (:354-371); test-only override `internal.nestedCostOverride` (:341-345).
`selectNestedMethod` → `localSelect` (:415-528): planMs = Σ nested attribute prices + flat
companions (`predictOrbitCostMs` on flat-symmetric attrs, 'centres' law on ordered attrs); enumMs
= 'bulger' law on the joint tuple-pair size (skip flags applied) IF `enumOk` (from
`nested_enumeration_admissible`: false IF any rel-per full-image attribute has σ/P > threshold,
:590-618) ELSE Inf; chosen = 'bulger' IF `enumMs·2.0 < planMs` (NESTED_ENUM_SAFETY = 2.0) else
'contract'. Prices recorded in `internal.lastNestedCosts`.

`tripSum` / `nestedAttrInnerMatrix` (`:714-740`, `:1303-1338`): IF `batchableMode` (abs and
relper always; relnonper unless every cell admits the spectral shortcut: ordered recipes,
r == #children, `sharedLeafTemplate` non-empty for every cell :880-915) → `pairValuesBatched`
(:786-876): relnonper → per-pair tau window (`tauWindow`, :919-966) + exp kernel; relper →
`relPerKernel` (wrapped_gaussian_1d(d,σ,P,ts,4) unless wrap=='single-image' → nearest-image
exp) (:2047-2066); abs → `absKernel` (quad.isPer && wrap≠single → wrapped_gaussian_1d; per
single → nearest-image; non-per → exp) (:2019-2044); each `truncK` (K < exp(-ts²/2) → 0, weights
included) → `contractNode`. ELSE per-pair `nestedIp` (:1879-1917): relnonper first tries
`ipRelNonperFactored` (:1963-2004; [] unless ordered whole-cell shared-template) then generic.

`contractNode` → `combinePair(M, r, sym, useOrbit)` (`:1711-1819`):
- useOrbit (recipe: `orbitEligible = sym && 2<=r<=8`, :1414-1431) is re-decided by
  `internal.orbitCostModel(r, max(gx,gy), Q)` (:1730): log-ratio model, `Keff = max(r, K-1)` for
  r<=3, coefficients [mptDefaults('orbitCostIntercept')=3.8536, 1.0708, 1.4033, -0.3608, -0.8188,
  -0.2261] on log|Ω_r|, log K, log B, log(TxTy), log r; true iff logRatio < 0
  (`+internal/orbitCostModel.m:83-118`); r>K or r<2 → false.
- IF useOrbit → LEAF combine_orbit (`innerProductOrbitGrid(M, 1, 1, r)/r!`, bound = eps·max|termMassSum|/r!) (:1822-1856)
  - post-hoc GUARD orbit_guard (only if postHocGuards): IF bound > exp(-ts²/2):
      IF enumWork = Q·C(gx,r)·r!·C(gy,r) <= 16e6 → warning mpt:nestedOrbitCost, FALLBACK → combine_chunked enumeration (:1750-1789)
      ELSE warning mpt:nestedOrbitAccuracy, keep the orbit value (:1790-1814)
- ELSE → LEAF combine_chunked (enumerated perm×comb product, chunks of 16e6 elements) (:1816-1818, :1638-1655)
```

### 1.2 Leaves reached from `cosSimExpTens`

| LEAF id | routine | file | role |
|---|---|---|---|
| ip_r1_direct | localIpR1Direct | cosSimExpTens.m:2386 | all-r=1 direct cross-correlation (Bulger arm / centres arm / r1 broadcast) |
| ip_full_ma / ip_core_ma | ipFullMA, ipCoreMA chunk loop | cosSimExpTens.m:1398, :1359-1386 | Bulger perm×comb (or centres perm×perm) truncated log-kernel product |
| gram_quadratic_form | localGramQuadraticForm | cosSimExpTens.m:2269 | non-periodic Q via Gram identity |
| wrapped_gaussian_1d | internal.wrappedGaussian1d | +internal/wrappedGaussian1d.m | abs-per theta: image-sum or Fourier |
| closed_form_attr_matrix_from | mobius.closedFormAttrMatrixFrom | +mobius/closedFormAttrMatrixFrom.m:1 | tuple-centres Gaussian overlap, comb-side restricted |
| r1_zero_pad | localR1ZeroPad | +mobius/maPerAttrInnerMatrix.m:233 | r=1 per-attribute kernel sum |
| safe_safe_orbit | localSafeSafeOrbit → innerProductOrbitPwBatched | +mobius/maPerAttrInnerMatrix.m:292, +mobius/innerProductOrbitPwBatched.m | abs r>=2 dense batched Möbius |
| safe_safe_orbit_sparse | localSafeSafeOrbitSparse → innerProductOrbitSparse | +mobius/maPerAttrInnerMatrix.m:378, +mobius/innerProductOrbitSparse.m | abs r>=2 sparse per-pair Möbius |
| spectral_rel_inner_matrix | mobius.spectralRelInnerMatrix | +mobius/spectralRelInnerMatrix.m | rel r=2..4 Fourier Gram matrix |
| rel_per_inner_sparse | localRelPerInnerSparse → innerProductOrbitSparse | +mobius/relInnerBatched.m:388 | rel-per sparse per-node orbit |
| inner_product_orbit_grid | mobius.innerProductOrbitGrid | +mobius/innerProductOrbitGrid.m | shared-weight batched orbit contraction |
| inner_product_orbit_pw_batched | mobius.innerProductOrbitPwBatched | +mobius/innerProductOrbitPwBatched.m | per-pair-weight batched orbit contraction |
| combine_orbit | combineOrbit → innerProductOrbitGrid | +internal/nestedContract.m:1822 | nested level Möbius reduction |
| combine_chunked | combineChunked/combine | +internal/nestedContract.m:1638, :1859 | nested level enumerated perm×comb |
| ip_rel_nonper_factored | ipRelNonperFactored | +internal/nestedContract.m:1963 | spectral-template closed form (rel-nonper nested) |

### 1.3 Overrides (cosSimExpTens)

| override | bypasses | still subject to |
|---|---|---|
| `method='bulger'` | selector rules 2–6 (incl. the forced-Bulger feasibility guard and the wrap measure rule), nested plan (never consulted) | ordered-attr rule (no-op), impossible-value guard n/a |
| `method='centres'` (flat) | all selector rules | silent ordered-attr override → 'bulger' (:1053-1055) |
| `method='centres'` (nested) | nested cost race and enumeration race (`force=true`) | measure rule: ERROR centresUnavailable on rel-per full-image above threshold; decline → ERROR contractUnavailable |
| `method='mobius'` (flat) | all selector rules; `ma_rel_attr_prefers_centres` returns false (grid kept) unless `mptDefaults('relAttrRoute')` is explicit | post-hoc impossible-value guard (fallback to bulger); silent ordered-attr override |
| `method='mobius'`/`'contract'` (nested) | cost races (`force=true`) | measure rule + decline → ERROR contractUnavailable; `'contract'` on non-nested → ERROR |
| `mptDefaults('relAttrRoute')` = 'mobius'/'centres' | per-attribute centres-vs-grid cost gate | measure block (ERROR mpt:relAttrRouteBlocked) |
| `internal.spectralIpEnabled(false)` / `spectralIpForce(true)` | spectral branch / its cost gate | MAX_POINTS memory guard never bypassed |
| `internal.combRestrictionEnabled(false)` | comb-side restriction | — |
| `mptDefaults('postHocGuards')=false` | impossible-value guard, nested orbit accuracy guard | — |
| `'truncationSigmas'` per call | mptDefaults ts on the Bulger/centres/grid/nested paths | NOT honoured by closedFormAttrMatrixFrom on the flat Möbius path (:1666-1671), by `autoNtauDefault`, by `nuVecSel`'s margin, by `maRelAttrPrefersCentres`' grid estimate margin |

### 1.4 Guards and fallbacks (cosSimExpTens)

- `N == 0` on either operand → s = 0 (:883).
- `guard_forced_bulger_feasible` (auto, r_max > 8, non-nested) → ERROR, no fallback.
- rel-per wrap rule (auto) → forces bulger/mobius by measure above threshold.
- ordered attribute → forced bulger (silent).
- `orbit_ips_impossible` (postHocGuards) → mobius → bulger, memo purged.
- nested plan declines (auto) → bulger enumeration; forced → ERROR.
- nested plan loses `nested_prefers_enumeration` (auto) → bulger.
- nested `orbit_guard` bound > floor → enumeration if work <= 16e6, else keep + warn.
- `spectralRelInnerMatrix` returns [] (gridSize > 4e6 or cost gate) → translation grid.
- `ma_rel_attr_prefers_centres` blocked by measure / empty tuple set → grid.
- `r1_broadcast_fast` structural mismatch → per-pair loop.
- `gram_is_accurate_enough` false → difference-form Q.
- `wrappedKernelImageCount == 0` → nearest-image Q form (same number).
- `closedFormAttrMatrixFrom` comb mismatch between sides → unrestricted form.
- denom == 0 → NaN.

### 1.5 Memo keys (cosSimExpTens) — `internal.selfIpKey(route, ts, extra)` = `'route|ts|extra'`

| route | key | read/written by |
|---|---|---|
| bulger | `bulger|<ts>|` | pairwise arm (:1276-1303), r1 broadcast fast (:2547-2670) |
| centres | `centres|<ts>|` | unrestricted centres arm (:1190-1210) |
| mobius | `mobius|<ts>|<choices bits>` | localCosSimMAOrbit (:1616-1712); purged on impossible-value guard |
| contract | `contract|<ts>|<route[|ntau|t1|tend]>` | nestedContract single-attribute (:272-333) |
| contract_ma | `contract_ma|<ts>|<kind:a:sig,…>` | nestedContractMA (:1118-1200) |

Shared pricing flag `internal.selfIpMemoised(cache)`: true if any key starts with one of
`{'bulger','centres','mobius','contract','contract_ma'}|` (`+internal/selfIpMemoised.m:36-46`);
feeds skipXX/skipYY for BOTH routes' prices (selector and nested cost).

---

## 2. ENTRY `sweepCosSimExpTens(densX, densY, offsets, …)` — `sweepCosSimExpTens.m`

```
- [parse] method ∈ {'auto','mixture','orbit'} (arguments block :117-118); normalize ∈ {cosine, oneSidedDenom} (:124)
- prune both; tsResolved = accuracyFloor('resolve', ts or mptDefaults)              (:146-154)
- orbitOk = NODE orbit_sweep_supported                                              (:157, :418-479)
    false IF any ~isSym | any nested inner block > 0 | rel && swept | rel-per with σ/P > threshold(tsRaw) and wrap ≠ 'full-image' | any kernelCov
- mixtureOk = ~error(localCheckEligible)                                            (:160-164, :246-323)
    ERRORS: relativePeriodic (rel-per σ/P > threshold and wrap ≠ 'single-image'), periodicAttribute (swept && per),
            sweptNested, sweptRelative, anisotropicKernel
- IF method == 'auto' → NODE choose_sweep_route                                     (:166-171, :326-415)
    ~orbitOk → mixture; ~mixtureOk → orbit;
    any swept attr with r_a < 2 → mixture; orbitWork == 0 (nothing swept) → mixture;
    mixtureBytes = nPairs·(nSwept+2)·8 > kernelChunkBytesResolved() → orbit;
    nPairs < 1e6 (ORBIT_MIN_PAIRS) → mixture;
    orbitTotal < 64·nPairs (ORBIT_WORK_RATIO) → orbit ELSE mixture
- ELSE chosen = method; IF mixture && ~mixtureOk → rethrow; IF orbit && ~orbitOk → ERROR sweepCosSimExpTens:orbitUnsupported (:172-181)
- IF orbit → LEAF orbit_sweep (:183-200)
    numerator: untranslated attrs → mobius.maPerAttrInnerMatrix(…,'truncationSigmas',ts,'wrap') (§1.1f);
               swept attrs → localOrbitAttrMatrixSweep (:482-556): full-image → wrapped_gaussian_1d(…,4) else nearest-image+truncKernelExp;
               r>=2 → innerProductOrbitPwBatched; r==1 → direct weighted kernel sum
    denominators: localOrbitSelfIp via maPerAttrInnerMatrix (no memo)
- ELSE → LEAF sweep_mixture (:205-237): ensureExpTensExpensive; localBuildMixture (:702) with
    threshold = -0.5 ts² - log(nJ nK); abs-per full-image → wrapped_gaussian_1d(…,4);
    localEvaluateMixture (:895): IF P <= meanSlice + 512 → localEvaluateDense ELSE culled per-offset loop;
    self terms via the same mixture at zero offset (localSelfIp :1063) — NOT memoised, no selfIP read
- denom == 0 → NaN
```

Leaves: `orbit_attr_matrix_sweep`, `mobius_inner`, `evaluate_mixture`/`evaluate_dense`.
Overrides: `method` bypasses `choose_sweep_route` only; support/eligibility checks still apply.
Memo: none (the `'sweep'` key mentioned in `selfIpMemoised.m:15-18` does not exist in code).

---

## 3. ENTRY `evalExpTens(varargin)` — `evalExpTens.m`

```
- [parse] method ∈ {'auto','centres','mobius'} (:214-223; ERROR evalExpTens:badMethod); normalize ∈ {none,gaussian,pdf} trailing (:245-250)
- nArgs == 9 → pop isSym (:270-276)
- IF struct with tag
  - 'MaetDensity' && isSingleMultiset → dens = singleMultisetView → FALL THROUGH to single-multiset dispatch (:312-318)
  - 'MaetDensity' general → NODE ma_skinny_dispatch (localMaSkinnyDispatch :1241)          (:320)
      - IF ~handled → ensureExpTensExpensive; whiten if kernelCov; NODE eval_ma_joint (localEvalMA :817) (:323-334)
  - 'WindowedMaetDensity' → ERROR mpt:aniso:windowedEval if kernelCov; skinny dispatch with method='auto' (user method DROPPED :345-347); ~handled → localEvalMA (no method); × window factor (:353)
- ELIF cell of structs → LIST: recursive evalExpTens(dens{i}, X_i, normalize, 'verbose') — method/ts/kernelPrecision DROPPED (:1790)
- ELIF cell of numerics (MA raw, 8 args) → buildExpTens → skinny → localEvalMA (:371-403)
- ELIF numeric 2-D → BATCHED-RAW (localEvalBatchedRaw :1796): ERROR batchedOrderedUnsupported (isSym false, r>1);
    per row evalExpTens(p, w, …, X, normalize, 'verbose'[, ts][, kernelPrecision]) — method DROPPED (:1845-1868)
- ELIF numeric vector → buildExpTens → singleMultisetView → FALL THROUGH
- single-multiset dispatch:
  - IF method=='centres' → chosen='centres'                                              (:508-509)
  - ELIF method=='mobius' → ERROR mpt:evalExpTens:orderedMobius IF hasOrderedAttr(maet); chosen='mobius' (:510-526)
  - ELSE [chosen, reason] = NODE select_ma_eval(maet, nQ, ts)                            (:535)
  - IF chosen=='mobius' → NODE eval_single_multiset_orbit (:557)
      zero-weight prune; rel → mobius.evalOrbitRel(p,w,σ,r,X,'is_per','period'[,ts][,kp]); abs → mobius.evalOrbitAbs(… ,'wrap') (:748-754)
      post-hoc GUARD: IF ~all(isfinite(vals)) → warning evalExpTens:mobiusNonFiniteFallback; chosen='centres' (:562-571)
  - IF ~ranOrbit → LEAF eval_single_multiset_centres: ensureExpTensExpensive → internal.gaussianKernelSum(Centres, wJ, X, σ, isRel/r, isPer/period, ts, kp) (:574-599, :760-815)
  - normalise (gaussian / pdf), aniso logdet correction (:627-663)
```

### 3.1 NODE `ma_skinny_dispatch` (`evalExpTens.m:1241`) and `eval_ma_joint` (`:817`)

```
- skinny: IF densityHasKernelCov → handled=false (→ joint)                          (:1254)
  - nQ == 0 → zeros, handled                                                        (:1266-1270)
  - method centres/mobius → forced; else NODE select_ma_eval(dens, nQ, ts)          (:1273-1280)
  - IF mobius → LEAF eval_ma_orbit (mobius.evalMaOrbit(dens, Xjoint[, ts][, kp])) → localMaNormaliseSkinny (:1282-1303)
  - ELSE → NODE ma_eval_factored (localMaEvalFactored :1408): [] IF any r_a < 2 | kernelCov | #ever-valid < r_a
      per event × attribute: nested (innerR>0) → dense block-diagonal exp(-Q) (NO truncation, :1495-1513);
      flat → LEAF gaussian_kernel_sum(c, wTuple, X_a, σ_a, isRel/r, isPer/period[, ts][, kp]) (:1533)
    - IF [] → handled=false (→ joint)
- joint (localEvalMA): prune zero-weight joint tuples; method forced or select_ma_eval again (:895-904)
  - mobius → eval_ma_orbit (:908-930) [reachable only with kernelCov, since skinny already served the non-cov case]
  - localMaEvalFactored retried (:940) — always [] here (same predicate already failed upstream): DEAD in practice
  - LEAF ma_eval_full (maetEvalFull :999): chunk on (2·maxDim+2)·N_J·8 vs kernelChunkBytesResolved (:961-988)
      per attribute: abs-per full-image (innerR==0) →
         IF tupleValuesRepeat(Ca, nQ) [nQ>=100, numel>=256, unique·4 <= numel] → tabulated wrapped_gaussian_1d(uVals - X, σ, P, ts, 2) gathered (:1047-1084)
         ELSE prod(wrapped_gaussian_1d(D_a, σ, P, ts, 2), 1) (:1098-1111)
      innerR>0 → block-diagonal reduced Q (:1087-1093); abs-per single-image → nearest-image Q (:1113);
      rel-per → pairwise-wrap Q/r; rel-nonper → Σd²-(Σd)²/r; abs → Σd² (:1115-1136)
      post-filter truncation Q_total > ts²/2 → 0; × absPerFactor (:1142-1148)
```

### 3.2 NODE `select_ma_eval` — `+internal/selectMaEval.m:1`

```
- ERROR mpt:selectMaEval:staleCallForm IF islogical(truncationSigmas)                (:61-65)
- IF hasOrderedAttr(dens) → 'centres' ("ordered ([sym]=0) attribute")                 (:111-115)
- IF any nested attribute → 'centres'                                                 (:118-126)
- IF all(r <= 1) → 'centres'                                                          (:129-133)
- IF any r_a > 10 (ORBIT_R_MAX_FEASIBLE) → GUARD: estimateMaJointWorkingSetBytes(r,K,isRel,isSym) > dispatchMemBudget()
      → ERROR mpt:dispatch:singleImageInfeasible; else 'centres' (forced)              (:137-166)
- measure rule (does NOT return early): first rel-per attr with σ/P > threshold(ts): wrap=='single-image' → forces centres, else forces mobius (:184-201)
- [centresMs, mobiusMs] = NODE ma_eval_costs_ms(dens, nQ)                              (:207)
- IF measureForcesCentres → 'centres'; ELIF measureForcesMobius → 'mobius'             (:211-216)
- ELSE safety = 1.5 (MA_MOBIUS_SAFETY) IF estimateMaJointWorkingSetBytes > 256 MiB ELSE 1.0; 'mobius' IF mobiusMs < centresMs·safety ELSE 'centres' (:221-239)
- (Docstring's "precision guard" does not exist in code; only the r > 10 feasibility rule.)
```

`ma_eval_costs_ms` (`+internal/maEvalCostsMs.m`): centres priced as factored per-attribute
sums (IF `A>1 && all r>=2 && ~kernelCov`, :216-245) else joint materialisation with culled
fraction product (:246-275); Möbius priced per attribute r_a>=2 with Bell-number setup +
per-query ops `(2^r-1)·r·K_a` (:277-290); rel attributes: IF `2<=r<=4 && nQ >= [16,32,64](r) &&
K_a >= [2,8,16](r) && (2M+1)^(r-1) <= 4e6` → spectral price (:339-351) ELSE node-grid price with
`N_u = max(64, ceil(spp·window/σ))` (:352-375). Constants at :50-174, :300-310.

### 3.3 Möbius evaluators

`eval_ma_orbit` (`+mobius/evalMaOrbit.m:94-142`): per event × attribute, rel →
`evalOrbitRel`, abs → `evalOrbitAbs` (wrap forwarded for abs only); product over attributes,
sum over events.

`eval_orbit_abs` (`+mobius/evalOrbitAbs.m:100-239`): `perHelperGlobal = is_per && period >
2·√2·ts·σ` (:115-120). Per unique partition block: non-per → `useReduction = true`; per →
`useReduction = perHelperGlobal && spanOk` (block offsets span < P/2, :161-166). useReduction →
LEAF `gaussian_kernel_sum(p, w^m, mean_x, σ/√m, [isPer, period, wrap])` × exp(-var/2σ²)
(:168-181); ELSE direct (m,N,nq) broadcast: wrap=='full-image' → `prod(wrapped_gaussian_1d(diffs,
σ, P, ts, 2))` else nearest-image exp (no truncation) (:182-207). Combine via
`mobiusPartitionCombine` (alternating sum).

`eval_orbit_rel` (`+mobius/evalOrbitRel.m:113-425`): r<2 → closed form sum(w). u-grid:
per → `N_u = max(64, ceil(P/σ·spp))`; non-per → window ±8σ, `N_u = max(64, ceil(max(u_max-u_min,1)/σ·spp))`
with spp = resolveSamplesPerSigma([], r, ts) (:139-160).
- Spectral: IF `2<=r<=4 && factored=='auto' && ~returnCancellationRatio && n_q >= [16,32,64](r) && K >= [2,8,16](r)`
  and `spanOk` (per: query span < P/2) and `(~is_per || P > 2√2·kRes·σ)` → LEAF `eval_orbit_rel_fourier` (:187-203). NO MAX_POINTS guard here.
- per: `factoredPerValid = P > 2·√2·truncEff·σ && maxSpread < P/2`; `'on'` && ~valid → ERROR mobius:evalOrbitRel:factoredPeriodic;
  'auto' → useFactored = valid && `factoredWorthwhile(K,r,n_q,N_u,nFineTotal)` (fact = K·nFine + 10·B_r·r·N_u·n_q < direct = B_r·r·K·N_u·n_q, :461-474); 'off' → direct (:206-245)
- non-per: 'auto' → factoredWorthwhile; 'on' → factored; 'off' → direct (:246-263)
- useFactored → LEAF factored tabulation: per m, `gaussian_kernel_sum(p, w^m, grid_m, σ/√m[, isPer, period])` + Lagrange-6 read-back (:266-359)
- ELSE LEAF direct: `evalOrbitAbs(p, w, σ, r, x_full, 'is_per', 'period', ts, kp)` at every u-node (wrap not forwarded → full-image), query-chunked on kernelChunkBytesResolved (:360-408)
- integrate: per → rectangle, non-per → trapz; /Z_t (:410-417)

### 3.4 LEAF `gaussian_kernel_sum` — `+internal/gaussianKernelSum.m:1`

```
- ts resolved; IF nTerms > 1 → ts = sqrt(ts² + 2 log nTerms)                        (:107-121)
- useTruncation = isfinite(ts) && ts > 0 && nJ>0 && nQ>0 (always true after resolve for non-empty inputs) (:125-127)
- 'single' precision casts                                                           (:129-141)
- IF useTruncation && ~isPer:
    dim==1 && ~isRel → LEAF truncated_kernel_sum_1d (sorted window)                 (:147-149)
    ELSE → LEAF truncated_kernel_sum (grid-bucket index; rel → whitened by M = I - 11'/r) (:150-152)
- ELIF useTruncation && isPer && dim==1 && ~isRel:
    IF 2·ts·σ < period → LEAF truncated_kernel_sum_1d_circular (:159-162)
    ELSE → exact path
- ELSE → localExactKernelSum → evalChunk (chunk on (2dim+2)·nJ·nQ·bytes vs kernelChunkBytesResolved) (:168-172, :186-231)
- evalChunk (:233-330):
    IF isPer && ~isRel:
      IF wrap=='full-image' && (tabulated (uInv nonempty, from tupleValuesRepeat(C,nQ)) || wrappedKernelImageCount(σ,P,ts,2) > 0)
         → tabulated theta gather or prod(wrapped_gaussian_1d(D,σ,P,ts,2),1) (:252-291)
      ELSE (single-image opt-in, or full-image at L==0) → nearest-image reduce, fall through (:292-296)
    Q: rel-per pairwise wrap /r; rel-nonper Σd²-(Σd)²/r; abs Σd²; exp(-Q/(2σ²)) with NO cutoff (:300-329)
```

Note: the 1-D circular truncated path (`localTruncatedKernelSum1DCircular`) computes the
nearest-image kernel regardless of `wrap` (single-image measure); only the exact path honours
full-image. It fires when `2·ts·σ < period` — i.e. exactly where `wrappedKernelImageCount(…,2)`
is 0 at that ts, so the two coincide; but the `nTerms` widening (:118-121) can push ts up
without moving the L computation (which is on the widened ts too — consistent).

### 3.5 Overrides / guards (evalExpTens)

- `method='centres'`: bypasses select_ma_eval entirely (no feasibility guard, no memory guard).
- `method='mobius'`: bypasses select_ma_eval; ERROR orderedMobius on the single-multiset corner only —
  on the MA path (`localMaSkinnyDispatch`) an ordered attribute is NOT rejected and `evalMaOrbit`
  runs (silently symmetrising) — see Suspicious.
- `method` dropped in LIST, BATCHED-RAW and Windowed forms.
- post-hoc: non-finite Möbius values → centres (single-multiset corner only; MA path has no such guard).
- `factored` option of evalOrbitRel is not reachable from evalExpTens (defaults 'auto').

Memo: none on the eval path.

---

## 4. ENTRY `entropyExpTens(varargin)` — `entropyExpTens.m`

```
- [parse] method ∈ {'shannon' (default),'normalized'/'normalised','differential','renyi2'} (localCanonicalizeMethod :1375-1398; ERROR badMethod);
    ERROR entropyExpTens:normalizeRemoved IF 'normalize' passed (:214)
- switch method                                                                       (:274-291)
  - 'normalized' / 'shannon' → NODE entropy_shannon_dispatch (:300)  [normalize flag = method=='normalized']
      input forms: struct (MaetDensity single-multiset → localEntropySingleMultiset :461; MA/Windowed → localEntropyMA :569), LIST (recursive), MA raw, BATCHED-RAW, single-multiset raw (+spectrum)
      - localEntropySingleMultiset: requires explicit nPointsPerDim (ERROR gridRequired); bounds errors; gridLimit ERROR;
          IF ~isRel → LEAF cell_masses_single_multiset_absolute (erf per-axis cell masses; periodic axis = MINIMUM-IMAGE erf, localPhiDiffAxisPeriodic :1011-1036, ignores wrap and truncationSigmas) (:517-527)
          ELSE → LEAF grid via evalExpTens(maet, X[, ts][, kp]) (method default 'auto' → §3) (:544-551)
      - localEntropyMA: IF ~isWindowed && isAbs (no rel attr) → LEAF cell_masses_ma_absolute (:666-672) ELSE evalExpTens(dens, X, …) on an ndgrid (:673-690)
  - 'differential' → NODE entropy_differential_adaptive (localDifferentialAdaptive :1568): nested grids N, N·2, … with Richardson extrapolation;
      each level calls the Shannon core (localEntropySingleMultiset / localEntropyMA with nvSub) (:1633-1654); ERROR differentialGridLimit when uncertified (:1625);
      LIST/batched/windowed → ERROR *NotSupported (:1474-1500)
  - 'renyi2' → NODE entropy_renyi2_dispatch (:1829): LIST/batched → ERROR; Windowed → ERROR; sigma==0 → ERROR sigmaZeroNotSupported
      - single-multiset → NODE renyi2_single_multiset (:1982):
          IF isSym false && r > 1 → LEAF renyi2_per_attr_numerical (ordered rebuild, direct pairwise overlap) (:2003-2009)
          ELIF r==1 && isRel → H = 0 (:2015)
          ELIF r==1 → LEAF direct pairwise: full-image → wrapped_gaussian_1d(diffs,σ,P,accuracyFloor('resolve',[]),4) else nearest-image exp (no truncation); Z = totalMassAbs (:2020-2052)
          ELSE rel → LEAF orbit_inner_rel (mobius.orbitInnerRelSingleMultiset → relInnerBatched §1.1g, default ts) + totalMassRel (:2066-2068)
               abs → LEAF orbit_inner_abs (mobius.orbitInnerAbsSingleMultiset → innerProductOrbit, wrap forwarded, default ts) + totalMassAbs (:2080-2082)
      - MA → NODE renyi2_ma (:2090): A==0 → 0; N==0 → NaN; per attribute:
          nested || (ordered flat r>1) → LEAF renyi2_per_attr_numerical (:2160-2162; abs-per full-image → wrapped_gaussian_1d(…,4) product, else block-metric Q, no truncation :2278-2288)
          ELSE → mobius_inner (maPerAttrInnerMatrix(Pa,Wa,Pa,Wa,…,'wrap',wrapA) — default truncationSigmas) + totalMassRel/Abs per event (:2177-2191)
      - GUARD renyi2_finalise: NaN IF ~finite or ip_xx<=0 or Z<=0 (:1944-1950); aniso logdet correction (:1965)
```

Leaves: cell_masses_*_absolute (erf), eval grid (→ §3), renyi2_per_attr_numerical,
orbit_inner_abs (→ inner_product_orbit), orbit_inner_rel (→ rel_inner_batched),
mobius_inner, total_mass_abs/rel (closed form, `+mobius/totalMassAbs.m`, `totalMassRel.m`).
Overrides: none beyond `method`; `truncationSigmas` nv is honoured only on the grid/differential
paths (renyi2 uses the global default). Memo: none.

---

## 5. ENTRY `windowedSimilarity` / `windowedTensorSimilarity`

- `windowedSimilarity.m`: builds windowed context/query per sweep point and calls
  `cosSimExpTens(dc, dq, 'normalize', …, 'verbose', false)` (nested/specs, :184, :265) or the
  9-arg raw form (:186, :267). `method` is not exposed → every point routes through §1 with
  method='auto' and default truncationSigmas. No separate routing.
- `windowedTensorSimilarity.m`: single algorithmic path (`maybeShowDispatchMsg
  'closed-form'`, :318-319) → `internal.windowedInnerProduct(densQuery, wmd, false, ipQQcache,
  normalize)` per offset (:381). `+internal/windowedInnerProduct.m`: ERROR twoSidedWindowing
  (:84); ip_qq cached once (:76-79, :117-121); `localCosSimNumeratorMACore` (:329) = Bulger
  perm-vs-comb with per-attribute log-kernel: abs-per full-image →
  `wrapped_gaussian_1d(D, σ, P, accuracyFloor('resolve',[]), 4)` (:438-450, ignores per-call ts),
  single-image → nearest-image Q (:452-456); window factor: periodic → image-summed erf
  (`localPeriodicImageSumContribution`, tol 1e-12, multi-D rel defers to line formula :598-601),
  non-periodic → erf window (:488-503); `exp(log_kernel)` with NO truncation (:507). 'cosine'
  normalisation squares the window (`localWindowSquared`, ERROR unsupportedNormalize for mix ∉
  {0,1} :183-195). No cost model, no method, no memo beyond the per-call ip_qq.

---

## 6. `explainDispatch(dens, other/nQ, …)` — `explainDispatch.m`

Reports (does not route):
- cosine, flat: calls `internal.selectMaInnerProductMethod(rVec, kVec, A, nX, nY, anyPer,
  anyRelNonper, anyRelPer, sop, method, false, isRel, [], kVecY, {}, ts, [], [], symVecEx)`
  (:151-155) — with `nuVec = []` (→ 2000 default) and `wrapVec = {}` (→ wrap rule skipped) and
  cold skip flags, so its verdict can differ from `cosSimExpTens`' (which passes the computed
  `nuVecSel`, `wrapG`, and memo-derived skip flags). Reports routeMs [pw, orbit], `priced` =
  both finite.
- cosine, nested: `internal.nestedContract(…, 'routesOnly')` for the plan routes (:208),
  `internal.nestedCost('selectNestedMethod')` / `('priceNestedAttr')` for prices (:235-248),
  `enumSafety` for the verdict (:281).
- eval: `internal.selectMaEval(dens, nQ, ts)` + `internal.maEvalCostsMs` (:105-114); user
  `method` overrides the reported choice.
- periodicity: `relPerSigmaOverPThreshold(ts)`; limitSetBy = 'positive-definiteness' iff limit >= 0.05 (:90-96).

---

## Orphans (routing-module routines with no caller on any traced path)

Evidence: `grep -rn "<name>(" --include=*.m . | grep -v tests/ | grep -v tools/ | grep -v demos/`.

| routine | evidence |
|---|---|
| `mobius.innerProductDirectAbsSingleMultiset` | only `tests/test_ma_per_attr_hybrid.m`; the comments in `maPerAttrInnerMatrix.m:204,545` are the only in-package mentions |
| `localFillDirectEnumGroups`, `localPackNanTop`, `localBatchedDirectEnumAbsSingleMultiset` (`+mobius/maPerAttrInnerMatrix.m:471,515,540`) | reachable only via `unsafe_x_idx`/`unsafe_y_idx`, which are `find(~true(1,N))` = always empty (:161-166) — dead code |
| `mobius.contract` | only `tests/test_recipe_equivalence.m`; runtime uses `mobius.executeRecipe` |
| `internal.timeRepeated` | only `tests/bench_*.m` (no timing probe exists in any dispatcher) |
| `internal.nestedCostOverride` (installer) | read by `nestedCost.m:341`; nothing in the package installs one (test hook) |
| `internal.absPerSigmaOverPThreshold` | only `internal.maybeWarnAbsPerSingleImage` (a diagnostic; nothing routes on it) |
| `mobius.innerProductOrbit` | reachable only from `orbitInnerAbsSingleMultiset` ← `entropyExpTens` renyi2 single-multiset abs; not on any cosine path |
| `mobius.orbitInnerAbsSingleMultiset`, `mobius.orbitInnerRelSingleMultiset` | `entropyExpTens.m:2067,2080` only |
| `localMaEvalFactored` call inside `localEvalMA` (`evalExpTens.m:940`) | always returns [] there (skinny dispatch already failed the same predicate, or kernelCov) — dead in practice |
| `localComputeQBlocks` (`closedFormAttrMatrixFrom.m:180`) and the inner/intermediate branch of `localReducedCentresFromValues` (`closedFormAttrCentres.m:257-273`) | no shipped caller builds a centres bundle with innerBlockSize >= 2 (documented at :48-55 / :249-254) |
| `mobius.evalOrbitRel` option `'factored'` ∈ {'on','off'} | never passed by `evalExpTens`/`evalMaOrbit`; test/benchmark only |
| `internal.relRouteCostMs('centres')` on the flat selector | used only inside `predictOrbitCostMs` (reachable) — not orphan; listed for completeness |

## Suspicious

1. **`maEvalCostsMs.m:278-377` stale `K_a`**: the Möbius pricing loop sets `r_a = rVec(a)` but never
   re-assigns `K_a`; it carries over from the last iteration of the earlier loop (:199-204 or
   :227), i.e. `kVec(A)` for every attribute. For A > 1 with unequal value counts the Möbius price
   (`ops`, the spectral K gate at :341, the tabulation term at :373) is computed with the wrong K.
2. **`selectMaInnerProductMethod.m:15-21` docstring** lists rule "(4) K-vs-r precision guard"; the
   code has no such rule (comment :120-123 says accuracy is governed by truncationSigmas). Same for
   `selectMaEval.m`'s absent "precision guard" (spec item); only the r > 10 feasibility rule exists.
3. **`cancellationThreshold`** is parsed, validated (`cosSimExpTens.m:313-322`), forwarded to every
   `localCosSimMA` call and the batched-raw recursion, but never read (:278 says "inert"). The
   docstring (:204-210) still describes a cancellation-ratio fallback that does not exist.
4. **Per-call `truncationSigmas` not honoured on the flat Möbius centres route**:
   `cosSimExpTens.m:1666-1671` calls `closedFormAttrMatrixFrom(cxB, cyB, wrapA)` without the 4th
   argument → `mptDefaults('truncationSigmas')` is used (`closedFormAttrMatrixFrom.m:118-120`),
   while the nested path passes `ts` (:293, :1169, :1245). The memo key, however, is built from
   the per-call ts (:1615-1616) — a value computed at the default width is stored under the
   override's key.
5. **`explainDispatch.m:151-155`** passes `nuVec = []` and `wrapVec = {}` to the flat selector,
   whereas `cosSimExpTens.m:1033-1037` passes computed `nuVecSel` and `wrapG`. The report can
   therefore name a different route than the one the call takes (the rel-per wrap rule is never
   exercised by the report; the grid price uses 2000 nodes instead of the geometry-derived count).
6. **`maEvalCostsMs.m:323-338`** prices a spectral MAX_POINTS decline ("mirror the spectral
   branch's own MAX_POINTS decline") that `evalOrbitRel.m:187-203` does not perform — the eval-side
   Fourier branch has no `MAX_POINTS` guard (the guard lives only in `spectralRelInnerMatrix`, an
   inner-product routine). The price and the route disagree at r = 4 / small σ/P.
7. **Ordered attribute + `method='mobius'` on the MA eval path**: `localMaSkinnyDispatch`
   (`evalExpTens.m:1275-1276`) honours `method='mobius'` with no `hasOrderedAttr` check (the single-multiset corner
   raises `mpt:evalExpTens:orderedMobius`, :517-525); `evalMaOrbit` then symmetrises the ordered
   attribute silently.
8. **Silent override on the cosine path**: an ordered attribute forces `chosen='bulger'`
   (`cosSimExpTens.m:1044-1056`) even under explicit `method='mobius'`/`'centres'`; no warning.
9. **`method` silently dropped**: `evalExpTens` LIST (:1790), BATCHED-RAW (:1845-1868) and
   Windowed (:345-347) forms; `cosSimExpTens` LIST (cell,cell) (:1763) and broadcast (:1863) forms
   also drop `method`, `truncationSigmas`, `cancellationThreshold` (only `normalize`, `verbose`
   forwarded) — the docstring at :36-48 does not say so.
10. **Measure inconsistency in Shannon entropy**: the absolute cell-mass path uses the minimum-image
    erf (`localPhiDiffAxisPeriodic`, `entropyExpTens.m:1011-1036`, "truncationSigmas … unused")
    regardless of the density's `wrap`, whereas the default measure elsewhere is full-image.
11. **`selfIpMemoised.m:15-18`** refers to a `'sweep'` memo key "deliberately not among them";
    no code writes a `'sweep|…'` key (`sweepCosSimExpTens` has no memo at all).
12. **Dead "hybrid safe/unsafe" partition** in `maPerAttrInnerMatrix.m:58-66, 149-225`: the
    docstring describes a safe/unsafe split; `safe_x_mask = true(1, Nx)` makes the direct-enum
    branches unreachable (see Orphans).
13. **`nestedContract` computes `ip_xx` under `oneSidedDenom`** whenever it is not memoised
    (`:319-324`, `:1148-1152`, `:1171-1174`); `skipXX` affects only pricing. The flat routes skip it.
14. **`autoNtauDefault` / `relWindowMargin(mptDefaults('truncationSigmas'))`** (used by
    `nuVecSel`, `maRelAttrPrefersCentres`, `relInnerBatched` periodic grid, `makeQuadrature`
    rel-per) read the GLOBAL default, not the per-call `truncationSigmas`; the same call's kernel
    truncation uses the per-call value.
15. **`windowedInnerProduct.m:444` and `entropyExpTens.m:2043,2282`** resolve `accuracyFloor('resolve', [])`
    (global default) for the wrapped Gaussian even when the caller supplied `truncationSigmas`.
16. **`evalExpTens` MA joint path Möbius branch** (`:908-930`) is reachable only with a kernel
    covariance (skinny dispatch serves every other case), where `X` has been whitened but
    `evalMaOrbit` reads the density's stored (whitened, σ=1) attributes — consistent only if the
    build stores whitened values; not verified here.
17. **`gaussianKernelSum` truncated 1-D circular path** computes the nearest-image kernel whatever
    `wrap` says (`localTruncatedKernelSum1DCircular`); it is selected iff `2·ts·σ < period`, which is
    the L = 0 regime (equal numbers), but the equivalence depends on the widened `ts` after the
    `nTerms` adjustment being the same one `wrappedKernelImageCount` would see — it is (both use the
    widened opts value), so this is a fragile-but-currently-consistent coupling.

## Constants table

| constant | value | file |
|---|---|---|
| `mptDefaults('truncationSigmas')` factory | 6 | mptDefaults.m:191 |
| accuracy-floor eps / sigmas | 1e-12 / sqrt(-2 ln 1e-12) ≈ 7.43 | +internal/accuracyFloor.m:36, :53 |
| `relPerSigmaOverPThreshold` DEPARTURE table | (0.020,1.67e-16) … (0.100,4.07e-02); PD_CEILING 0.05 | +internal/relPerSigmaOverPThreshold.m:60-77 |
| `absPerSigmaOverPThreshold` (diagnostic only) | 0.04 | +internal/absPerSigmaOverPThreshold.m:40 |
| `_ORBIT_R_MAX_SHIPPED` (cosine selector, orbitEligible, getOrbitTable SHIPPED_MAX) | 8 | selectMaInnerProductMethod.m:104; nestedContract.m:1429; getOrbitTable.m:115 |
| `getOrbitTable` R_HARD_CAP | 12 | +mobius/getOrbitTable.m:27 |
| `ORBIT_R_MAX_FEASIBLE` (eval) | 10 | +internal/selectMaEval.m:70 |
| `MA_MOBIUS_SAFETY` / `_SMALL` / `CENTRES_WORKING_SET_SOFT_BUDGET` | 1.5 / 1.0 / 256 MiB | +internal/selectMaEval.m:92-94 |
| `dispatchMemBudget` | clamp(availableMemory/2, 1 GiB, 4 GiB) | +internal/dispatchMemBudget.m:17-20 |
| `relRouteCostMs` laws A/B (bulger, centres, grid; rows r=2,3,≥4) | bulger A=[-7.2149,-8.0168,-7.7822] B=[0.6970,0.7493,0.7500]; centres A=[-7.7429,-8.6263,-8.8269] B=[0.6800,0.7781,0.8029]; grid A=[-4.1388,-2.1985,-0.5409] B=[0.3916,0.5021,0.5768] | +internal/relRouteCostMs.m:43-53 |
| `predictOrbitCostMs` ABS / GRID_OP / CENTRES_OP / REL_BASE / ORBIT_REL_FLOOR_MS | ABS=[NaN,3,11.2,45,150,500,1500,4500]; GRID_OP=[NaN,3e-5,5e-5]; CENTRES_OP=[NaN,4e-5,9e-5]; 0; [0.137 0.086; 0.280 0.353; 0.497 7.956] | +internal/predictOrbitCostMs.m:34-70 |
| `nuVecSel` default / non-per node rule | 2000 / max(64, ceil(max(span,1)/σ·10)) | cosSimExpTens.m:981, :998-999 |
| `maRelAttrPrefersCentres` CENTRES_NS_BASE/LIN/WRAP, GRID_NS_FLOOR, gOp | 15, 5, 10/3, 1e6, {30, 700, 2000, 2000·3^(r-4)} | +mobius/maRelAttrPrefersCentres.m:64-67, :168-181 |
| `mptDefaults('relAttrRoute')` | 'auto' | mptDefaults.m:197 |
| `spectralRelInnerMatrix` MODE_SIGMAS / MAX_POINTS / COST_C / ENV_FLOOR | 8.6 / 4e6 / 1100 / 1e-18 | +mobius/spectralRelInnerMatrix.m:100-103 |
| `relInnerBatched` SLAB_ELEMS | 2^19 | +mobius/relInnerBatched.m:74 |
| sparse-orbit thresholds minKernel / maxDensity | 200000 / 0.20 | +mobius/maPerAttrInnerMatrix.m:373-374; +mobius/relInnerBatched.m:318-319 |
| `innerProductOrbitSparse` densityThresh | 0.34 | +mobius/innerProductOrbitSparse.m:45 |
| `resolveSamplesPerSigma` | max(2, ceil(ts√r/(2π))+1) | +internal/resolveSamplesPerSigma.m:157-159 |
| `relWindowMargin` | min(√2·ts+0.1, 8); 8 if Inf | +internal/relWindowMargin.m:113-117 |
| `autoNtauDefault` | max(64, ceil(2πP/σ · (1 + 0.5·max(0,-log10 tol)/12))), tol = max(exp(-ts²/2),1e-12) from global ts | +internal/autoNtauDefault.m:76-84 |
| `gram_is_accurate_enough` factor | eps·s²/(4σ²) <= 0.1·truncationFloor(ts) | cosSimExpTens.m:2264-2265 |
| `closedFormAttrMatrixFrom` chunk | 16e6 elements | +mobius/closedFormAttrMatrixFrom.m:138 |
| `tupleValuesRepeat` minQueries/minEntries/minSaving | 100 / 256 / 4 | +internal/tupleValuesRepeat.m |
| `orbitCostModel` coefficients | [mptDefaults('orbitCostIntercept')=3.8536, 1.0708, 1.4033, -0.3608, -0.8188, -0.2261]; Keff = max(r,K-1) for r<=3 | +internal/orbitCostModel.m:94-99, :77-79; mptDefaults.m:196 |
| nested orbit guard enumeration cap | enumWork <= 16e6 | +internal/nestedContract.m:1755 |
| nested combine chunk / pair batch | 16e6 elements | nestedContract.m:1642, :816 |
| nested rel-nonper grid | n = max(64, ceil(2·hi/(σ/4))), pad = (6 + 0.5·max(0,-log10 tol))σ | nestedContract.m:2102-2106 |
| `nestedCost` law keys / A,B per route / floors / NESTED_ENUM_SAFETY / soft budget | keys [2,3,4,6]; see nestedCost.m:174-211; 2.0; 256 MiB | +internal/nestedCost.m:174-236 |
| `maEvalCostsMs` constants | :50-174 (centres/mobius setup & per-op), FOUR_PER_MODE [1.509e-5, 2.111e-4, 1.757e-3], FOUR_PERIODIC_K_MS [3.272e-5, 6.648e-5, 0], FOUR_MIN_Q [16,32,64], FOUR_MIN_K [2,8,16], MODE_SIGMAS 8.6, MAX_POINTS 4e6, BELL B_1..B_10 | +internal/maEvalCostsMs.m:47-174, :300-330 |
| `evalOrbitRel` fourMinQ / fourMinK / READBACK_COST / CALIB_A6 / SPP_MIN,MAX / EPS_CEIL | [16,32,64] / [2,8,16] / 10 / 1600 / 8,512 / 1e-3 | +mobius/evalOrbitRel.m:185-186, :437-459, :469 |
| `evalOrbitRel` non-per window margin | 8σ | +mobius/evalOrbitRel.m:157-158 |
| `evalOrbitAbs` perHelperGlobal | period > 2·√2·ts·σ | +mobius/evalOrbitAbs.m:116-117 |
| `sweepCosSimExpTens` ORBIT_WORK_RATIO / ORBIT_MIN_PAIRS / offsetOverheadInComponents | 64 / 1e6 / 512 | sweepCosSimExpTens.m:345-346, :955 |
| `windowedInnerProduct` periodic image tolerance | 1e-12 | +internal/windowedInnerProduct.m:499 |
| `cosSimExpTens` impossible-cosine tolerance | |ip_xy| > 1.000001·denom | cosSimExpTens.m:809 |
| `r1_broadcast_fast` cacheCap | floor(4e6/(nJ·8)) | cosSimExpTens.m:2618 |
| `mptDefaults('postHocGuards')`, `'kernelChunkBytes'` | true, 'auto' (= availableMemory/2) | mptDefaults.m:194-195 |

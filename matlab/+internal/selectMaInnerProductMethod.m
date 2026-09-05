function [chosen, pwCostOut, orbitCostOut] = selectMaInnerProductMethod( ...
        rVec, kVec, A, Nx, Ny, ...
        anyPer, anyRelNonper, anyRelPer, sigmaOverPMax, userMethod, ...
        verbose, relVec, nuVec, kVecY, wrapVec, truncationSigmas, ...
        skipXX, skipYY, symVec, guardForcedBulger, perVec)
%   [CHOSEN, PWCOST, ORBITCOST] = ... also returns the two predicted
%   wall times in milliseconds that the comparison rests on. They are
%   NaN on the early returns that decide without pricing (an explicit
%   userMethod, r_max <= 1, r above the shipped orbit order, and the
%   memory guard), so a caller can tell a priced decision from a
%   structural one. Exposed for calibration: fitting the cost model
%   needs the prediction beside the measurement.
%SELECTMAINNERPRODUCTMETHOD  Pick the MA inner-product method (cost model).
%   Mirror of Python dispatch._select_ma_inner_product_method. Routing
%   rules, in order: (1) userMethod override; (2) r_max <= 1 -> Bulger;
%   (3) r_max > _ORBIT_R_MAX_SHIPPED -> Bulger (with the forced-Bulger
%   feasibility guard); (4) working-set guard: either side's perm-side
%   working set above CENTRES_WORKING_SET_SOFT_BUDGET, unless a
%   relative-periodic attribute sits above the sigma/P threshold ->
%   Möbius; (5) the rel-per wrap rule above that threshold; (6) predict
%   both wall times (ms) and take the faster path (ties favour Bulger).
%   Above the rel+per sigma/P threshold the two methods compute
%   different measures rather than the same one at different speeds, so
%   the wrap axis decides which is wanted; below it they agree and cost
%   decides.
%
%   The Möbius side is priced per attribute: absolute r_a >= 2
%   attributes cost the vectorised-batch constant for their order;
%   relative attributes cost three (Nx, Ny) matrices -- the cross
%   matrix and one self matrix per density -- at the cheaper of the two
%   per-pair routes the orchestrator itself chooses between: the
%   tuple-centres closed form (MOBIUS.CLOSEDFORMATTRMATRIXFROM, at
%   M_x*M_y + M_x^2 + M_y^2 ops per pair with M = r_a!*C(K_a, r_a) read
%   from each density's own value count) or the batched
%   translation-grid contraction (MOBIUS.MAPERATTRINNERMATRIX,
%   nu_a * K_a^2 ops per matrix). The centres route is measure-blocked
%   above the sigma/P threshold (the orchestrator keeps the all-image
%   grid there), so above it the grid route is priced alone.
%
%   KVEC and KVECY are the two densities' per-attribute value counts.
%   They need not agree, and both Bulger's tuple-pair size and the
%   Möbius side's centres term are products over the two; KVECY omitted
%   means they agree. The grid term reads KVEC alone, matching
%   MOBIUS.MARELATTRPREFERSCENTRES, the gate this function predicts the
%   outcome of.
%
%   RELVEC (logical, per attribute) and NUVEC (grid node estimates,
%   per attribute) are optional; omitted, every r_a >= 2 attribute is
%   treated as relative whenever either rel flag is set (over-pricing
%   the Möbius side -> near-crossover bias toward Bulger's method,
%   the cheap-to-mispick side) with a representative node count.
%
%   SKIPXX and SKIPYY say whether <X,X> and <Y,Y> cost this call
%   nothing --- because some route has already memoised the value on the
%   density, or (for <X,X>) because the requested normalisation does not
%   consume it. Both routes are priced with the same pair of flags; see
%   INTERNAL.SELFIPMEMOISED.
%
%   The cost laws are per-language (INTERNAL.RELROUTECOSTMS, refit with
%   tools/calibrateRelIpCost.m); the absolute-attribute constants and
%   the relative setup floor live in INTERNAL.PREDICTORBITCOSTMS.
    if nargin < 11; verbose = true; end
    if nargin < 12 || isempty(relVec)
        relVec = (anyRelNonper || anyRelPer) & (rVec(:).' >= 2);
    end
    if nargin < 13 || isempty(nuVec)
        nuVec = 2000 * ones(1, max(A, 1));
        nuVec = nuVec(1:A);
    end
    if nargin < 14 || isempty(kVecY)
        kVecY = kVec;
    end
    if nargin < 15
        wrapVec = {};
    end
    if nargin < 16
        truncationSigmas = [];
    end
    % A self inner product that is memoised on its density, or that the
    % requested normalisation does not consume, costs nothing at call
    % time; skipXX / skipYY exclude it from *both* routes' prices. The
    % flags are shared rather than per route: the memoised values are
    % route-keyed (the routes' scales are related in closed form but
    % their truncated numbers are not the same number --- see
    % localSelfIpKey in cosSimExpTens), yet pricing each route against
    % its own memo would decide the comparison on which route ran first
    % rather than on what the routes cost, and would lock that first
    % choice in. Twin of the Python selector's skip_xx / skip_yy.
    % Defaults false reproduce the full-triple pricing exactly.
    if nargin < 17 || isempty(skipXX); skipXX = false; end
    if nargin < 18 || isempty(skipYY); skipYY = false; end
    % Per-attribute [sym] flags for the forced-Bulger feasibility guard
    % (empty -> every attribute treated as unordered, the conservative
    % count), and the guard flag itself (false for nested densities,
    % which route through the hierarchical contraction instead of the
    % flat Bulger pairwise path). Twins of the Python selector's
    % sym_vec and guard_forced_bulger.
    if nargin < 19 || isempty(symVec);            symVec = [];             end
    if nargin < 20 || isempty(guardForcedBulger); guardForcedBulger = true; end
    % Per-attribute isPer flags for the wrap rule below: only a
    % relative-PERIODIC attribute's wrap declares a measure. Empty (older
    % callers) treats every relative attribute as periodic, the pre-fix
    % reading. Twin of the Python selector's per_vec.
    if nargin < 21; perVec = []; end
    % NaN until the priced comparison sets them, so a caller can tell a
    % structural decision from a costed one.
    pwCostOut = NaN;
    orbitCostOut = NaN;
    if ~strcmp(userMethod, 'auto')
        chosen = userMethod;
        return;
    end
    if A > 0
        r_max = max(rVec);
    else
        r_max = 1;
    end
    if r_max <= 1
        chosen = 'bulger'; return;
    end
    if r_max > 8                        % _ORBIT_R_MAX_SHIPPED
        % No orbit table ships above r = 8, so Bulger is forced with no
        % cheaper all-image substitute: guard against an infeasible
        % tuple-pair kernel rather than let it exhaust memory. Twin of
        % the Python selector's forced-Bulger guard.
        if guardForcedBulger
            internal.guardForcedBulgerFeasible(kVec, rVec, Nx, Ny, ...
                'r above the shipped orbit order', kVecY, symVec);
        end
        chosen = 'bulger'; return;
    end
    % Accuracy is governed by truncationSigmas, not by the collection
    % size: the Mobius method's agreement with enumeration tracks the
    % truncation budget and is closest at K_a = r_a. The route is
    % therefore chosen on cost alone from here on.
    % ---- Memory-safety guard (explicit invariant) ----
    % Bulger's MA IP materialises each side's joint perm-side working
    % set n_J = N * prod_a r_a! * C(K_a, r_a) (lazy, built on first
    % access); the Möbius MA IP is n_J-free (per-event, per-attribute
    % additive work). The cost race below already routes large workloads
    % to Möbius, because its Bulger cost keys on the tuple-pair size
    % n_J^X * n_J^Y --- the square of the per-side working set --- so any
    % density big enough to blow memory is diverted on cost alone. This
    % guard makes that invariant explicit: if either side's perm-side
    % working set exceeds the soft budget and Möbius is convention-safe
    % (no relative-periodic attribute above the sigma/P threshold, where
    % the wrap rule below owns the decision), take Möbius now. Same
    % position in the rule order, same working-set formula
    % n_J_max * 2 * max(sum r_a, 1) * 8 (perm + centres + index arrays),
    % and same budget as the Python selector and INTERNAL.SELECTMAEVAL.
    CENTRES_WORKING_SET_SOFT_BUDGET = 256 * 1024^2;   % bytes; Python twin
    if A > 0 && ~(anyRelPer && sigmaOverPMax > ...
                  internal.relPerSigmaOverPThreshold(truncationSigmas))
        if isempty(symVec)
            symGuard = true(1, A);
        else
            symGuard = logical(symVec(:).');
        end
        tuplesX = 1; tuplesY = 1; dimSum = 0;
        for a = 1:A
            ra = rVec(a);
            % Enumerated tuple count: r_a! * C on an unordered attribute,
            % C alone on an ordered one (perm side = comb side).
            if symGuard(a)
                fa = factorial(ra);
            else
                fa = 1;
            end
            tuplesX = tuplesX * fa * combCount(kVec(a), ra);
            tuplesY = tuplesY * fa * combCount(kVecY(a), ra);
            dimSum = dimSum + ra;
        end
        % "Either side" is meant literally: each density's working set is
        % its own event count times its own tuple count, and the two
        % densities need not carry the same number of values.
        nJMax = max(Nx * tuplesX, Ny * tuplesY);
        if nJMax * (2 * max(dimSum, 1)) * 8 > CENTRES_WORKING_SET_SOFT_BUDGET
            chosen = 'mobius'; return;
        end
    end
    % Relative-periodic measure note. The relative periodic density is
    % defined as the transposition average of the absolute periodic
    % density, which the Möbius method computes at every sigma/P.
    % Bulger's method evaluates the kernel that wraps the pairwise
    % component differences instead: an approximation that coincides
    % with the transposition average as sigma/P -> 0 and is cheaper, so
    % the toolbox uses it in that regime. The two diverge above
    % the resolved threshold, and from around 0.06 the wrapped-difference form
    % stops being positive-definite --- its cosine similarity exceeds 1
    % for some density pairs --- so above the threshold the choice is a
    % choice of measure and the wrap axis makes it.
    % Relative-periodic wrap. Above the sigma/P threshold the two methods
    % compute different measures: Bulger's is the single-image
    % (nearest-image) reduction, the Mobius method's the all-image
    % transposition average. A density built with 'single-image' is
    % asking for the former and 'full-image' for the latter, so above the
    % threshold the wrap picks the method rather than the cost model
    % doing so. Below it the two agree numerically and either will do.
    % Callers without a wrap vector see the pre-v3 order unchanged.
    % Mirrors the Python rule in _select_ma_inner_product_method.
    if ~isempty(wrapVec) && anyRelPer && ~isempty(relVec)
        wantsSingle = false;
        wantsFull = false;
        for a = 1:min(numel(wrapVec), numel(relVec))
            if ~relVec(a), continue; end
            % A relative-non-periodic attribute at the default wrap must
            % not turn a single-image rel-per attribute into a "mixed"
            % declaration: its wrap axis has no meaning.
            if ~isempty(perVec) && a <= numel(perVec) && ~perVec(a)
                continue;
            end
            if strcmp(char(wrapVec{a}), 'single-image')
                wantsSingle = true;
            elseif strcmp(char(wrapVec{a}), 'full-image')
                wantsFull = true;
            end
        end
        if wantsSingle && wantsFull
            error('mpt:mixedRelPerWrap', ...
                ['Mixed rel-per wrap on a single density is not yet ' ...
                 'supported; all rel-per attributes must share a wrap ' ...
                 'value.']);
        end
        if sigmaOverPMax > ...
                internal.relPerSigmaOverPThreshold(truncationSigmas)
            if wantsSingle
                chosen = 'bulger'; return;
            elseif wantsFull
                chosen = 'mobius'; return;
            end
        end
    end

    pwSize = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny, kVecY, ...
                                       skipXX, skipYY);
    % Priced by the same fitted law. The per-entry form this replaces
    % assumed a fixed cost per kernel entry; measurement contradicts
    % that, the per-entry cost falling as the arrays grow, which is what
    % the fitted exponent below 1 carries.
    pwCost = internal.relRouteCostMs('bulger', r_max, pwSize);
    centresOk = sigmaOverPMax <= ...
        internal.relPerSigmaOverPThreshold(truncationSigmas);
    orbitCost = internal.predictOrbitCostMs(rVec, kVec, A, Nx, Ny, ...
                                   relVec, nuVec, centresOk, kVecY, ...
                                   skipXX, skipYY);
    pwCostOut = pwCost;
    orbitCostOut = orbitCost;
    if pwCost <= orbitCost
        chosen = 'bulger';
    else
        chosen = 'mobius';
    end
end


function sz = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny, kVecY, ...
                                        skipXX, skipYY)
    % Tuple-pair kernel entries summed over the matrices the inner
    % product will compute. Each density contributes a permutation-side
    % and a combination-side tuple count,
    %
    %   n_J = N * prod_a r_a! * C(K_a, r_a),  n_K = N * prod_a C(K_a, r_a),
    %
    % and the three matrices are the cross matrix at n_J^X * n_K^Y and
    % one self matrix per density at n_J^X * n_K^X and n_J^Y * n_K^Y.
    % Where the two densities carry the same counts the three coincide
    % and the total is three times the cross term; where they do not, the
    % larger density's self matrix dominates. For a five-value density
    % against an 80-value one at r = 2 the second self matrix holds
    % 19971200 of the 20034600 entries, so pricing the cross matrix alone
    % understates the work by 317. SKIPXX / SKIPYY exclude a self matrix
    % that is memoised or not consumed by the requested normalisation;
    % pricing it anyway would steer near-crossover routing away from
    % Bulger's method on exactly the repeated-context sweeps where
    % Bulger's marginal cost is lowest.
    if nargin < 6 || isempty(kVecY)
        kVecY = kVec;
    end
    if nargin < 7 || isempty(skipXX); skipXX = false; end
    if nargin < 8 || isempty(skipYY); skipYY = false; end
    if A == 0
        sz = Nx * Ny; return;
    end
    permX = Nx; combX = Nx;
    permY = Ny; combY = Ny;
    for a = 1:A
        ra = rVec(a); Ka = kVec(a); KaY = kVecY(a);
        if Ka < ra || KaY < ra
            sz = Inf; return;
        end
        fa = factorial(ra);
        cx = combCount(Ka, ra);
        cy = combCount(KaY, ra);
        permX = permX * fa * cx;  combX = combX * cx;
        permY = permY * fa * cy;  combY = combY * cy;
    end
    sz = permX * combY;
    if ~skipXX
        sz = sz + permX * combX;
    end
    if ~skipYY
        sz = sz + permY * combY;
    end
end


function c = combCount(nn, kk)
    if kk < 0 || kk > nn
        c = 0; return;
    end
    c = 1;
    for ii = 0:kk - 1
        c = c * (nn - ii) / (ii + 1);
    end
    c = round(c);
end

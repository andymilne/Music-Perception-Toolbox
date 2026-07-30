function chosen = selectMaInnerProductMethod(rVec, kVec, A, Nx, Ny, ...
        anyPer, anyRelNonper, anyRelPer, sigmaOverPMax, userMethod, ...
        verbose, relVec, nuVec, kVecY)
%SELECTMAINNERPRODUCTMETHOD  Pick the MA inner-product method (cost model).
%   Mirror of Python dispatch._select_ma_inner_product_method. Routing
%   rules, in order: (1) userMethod override; (2) r_max <= 1 -> Bulger;
%   (3) r_max > _ORBIT_R_MAX_SHIPPED -> Bulger; (4) K-vs-r precision guard
%   (accuracy is governed by truncationSigmas, so cost decides) ->
%   Bulger; (5) predict both wall times (ms) and take the faster path
%   (ties favour Bulger); when that path is the all-image Möbius method
%   and rel+per sigma/P exceeds 0.03, warn that it differs from the
%   canonical single-wrap measure and point to method='bulger'.
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
%   Constants are provisional Python-shape values pending MATLAB
%   calibration from matlab/tests/bench_ma_dispatch.m.
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
        chosen = 'bulger'; return;
    end
    % Accuracy is governed by truncationSigmas, not by the collection
    % size: the Mobius method's agreement with enumeration tracks the
    % truncation budget and is closest at K_a = r_a. The route is
    % therefore chosen on cost alone from here on.
    % Relative-periodic measure note: the Möbius method computes the all-image
    % (transposition-integral) form, Bulger's the single-wrap (minimum-image)
    % form; they diverge above sigma/P = 0.03. The dispatch always takes the
    % faster path (cost model below); when that path is the all-image Möbius
    % method and sigma/P is above the threshold it warns and points to
    % method='bulger' for the canonical single-wrap measure.
    pwSize = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny, kVecY);
    pwCost = pwSize * pwPerEntryMs(anyPer, r_max);
    centresOk = sigmaOverPMax <= 0.03;   % _ORBIT_SIGMA_OVER_P_THRESHOLD
    orbitCost = predictOrbitCostMs(rVec, kVec, A, Nx, Ny, relVec, ...
                                   nuVec, centresOk, kVecY);
    if pwCost <= orbitCost
        chosen = 'bulger';
    else
        chosen = 'mobius';
    end
end


function ms = pwPerEntryMs(anyPer, r_max)
    % Per-entry cost of Bulger's joint tuple-pair kernel in n_J . n_K
    % units, per tensor order (the r! side asymmetry is absorbed into
    % the constant, so it rises mildly with r). Provisional values from
    % the Python calibration at representative scale; MATLAB values to
    % be set from bench_ma_dispatch.
    %
    % That calibration gave the two densities the same value count and
    % the same event count, where the three matrices the inner product
    % needs are each the size of the cross matrix, and it was fitted
    % against the cross matrix alone -- so the fitted figures each
    % absorb a factor of three. They appear below as those figures
    % divided by three, which leaves every equal-count workload
    % predicting exactly what it predicted when the fit was made, while
    % predictPairwiseKernelSize now returns the total across the three
    % matrices and so responds correctly when the counts differ.
    if anyPer
        table = [NaN, 1.1e-4, 1.6e-4] / 3;   % r = 2, 3
    else
        table = [NaN, 1.0e-4, 1.4e-4] / 3;
    end
    ms = table(min(max(r_max, 2), numel(table)));
end


function sz = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny, kVecY)
    % Tuple-pair kernel entries summed over the three matrices the inner
    % product needs. Each density contributes a permutation-side and a
    % combination-side tuple count,
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
    % understates the work by 317.
    if nargin < 6 || isempty(kVecY)
        kVecY = kVec;
    end
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
    sz = permX * combY + permX * combX + permY * combY;
end


function ms = predictOrbitCostMs(rVec, kVec, A, Nx, Ny, relVec, ...
                                  nuVec, centresOk, kVecY)
    % Per-attribute sum, mirror of Python _predict_orbit_cost_ms:
    % relative attributes are priced at the cheaper of the
    % tuple-centres closed form and the batched grid contraction
    % (three matrices each), with the centres term blocked above the
    % sigma/P threshold. Constants provisional pending
    % bench_ma_dispatch calibration; measured entries cover r = 2, 3,
    % doubling per order above (over-pricing the Möbius side —
    % routing bias toward Bulger's method, the cheap-to-mispick
    % side).
    ABS        = [NaN, 3.0, 11.2, 45.0, 150.0, 500.0, 1500.0, 4500.0];
    GRID_OP    = [NaN, 3.0e-5, 5.0e-5];    % r = 2, 3
    CENTRES_OP = [NaN, 4.0e-5, 9.0e-5];    % r = 2, 3
    REL_BASE = 5.0;
    if nargin < 9 || isempty(kVecY)
        kVecY = kVec;
    end
    ms = 0;
    if any(relVec)
        ms = REL_BASE;
    end
    pairs = Nx * Ny;
    for a = 1:A
        ra = rVec(a); Ka = kVec(a); KaY = kVecY(a);
        if relVec(a) && ra >= 2
            gridOp = GRID_OP(min(max(ra, 2), numel(GRID_OP)));
            if ra > 3    % beyond tabulated orders: double per order
                gridOp = gridOp * 2^(ra - 3);
            end
            perPair = 3 * nuVec(a) * Ka * Ka * gridOp;
            if centresOk && Ka >= ra && KaY >= ra
                centresOp = CENTRES_OP(min(max(ra, 2), numel(CENTRES_OP)));
                if ra > 3
                    centresOp = centresOp * 2^(ra - 3);
                end
                mX = factorial(ra) * combCount(Ka, ra);
                mY = factorial(ra) * combCount(KaY, ra);
                centresOps = mX * mY + mX * mX + mY * mY;
                perPair = min(perPair, centresOps * centresOp);
            end
            ms = ms + pairs * perPair;
        elseif ra >= 2
            ms = ms + ABS(ra);
        end
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

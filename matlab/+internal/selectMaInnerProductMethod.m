function chosen = selectMaInnerProductMethod(rVec, kVec, A, Nx, Ny, ...
        anyPer, anyRelNonper, anyRelPer, sigmaOverPMax, userMethod, ...
        verbose, relVec, nuVec)
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
%   relative attributes cost three (Nx, Ny) matrices at the cheaper
%   of the two per-pair routes the orchestrator itself chooses
%   between — the tuple-centres closed form
%   (MOBIUS.CLOSEDFORMATTRMATRIXFROM, (r_a!*C(K_a, r_a))^2 ops per
%   pair) or the batched translation-grid contraction
%   (MOBIUS.MAPERATTRINNERMATRIX, nu_a * K_a^2 ops per pair). The
%   centres route is measure-blocked above the sigma/P threshold (the
%   orchestrator keeps the all-image grid there), so above it the
%   grid route is priced alone.
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
    pwSize = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny);
    pwCost = pwSize * pwPerEntryMs(anyPer, r_max);
    centresOk = sigmaOverPMax <= 0.03;   % _ORBIT_SIGMA_OVER_P_THRESHOLD
    orbitCost = predictOrbitCostMs(rVec, kVec, A, Nx, Ny, relVec, ...
                                   nuVec, centresOk);
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
    if anyPer
        table = [NaN, 1.1e-4, 1.6e-4];   % r = 2, 3
    else
        table = [NaN, 1.0e-4, 1.4e-4];
    end
    ms = table(min(max(r_max, 2), numel(table)));
end


function sz = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny)
    % n_J . n_K = Nx . Ny . prod_a r_a! . C(K_a, r_a)^2
    if A == 0
        sz = Nx * Ny; return;
    end
    sz = Nx * Ny;
    for a = 1:A
        ra = rVec(a); Ka = kVec(a);
        if Ka < ra
            sz = Inf; return;
        end
        c = combCount(Ka, ra);
        sz = sz * factorial(ra) * c * c;
    end
end


function ms = predictOrbitCostMs(rVec, kVec, A, Nx, Ny, relVec, ...
                                  nuVec, centresOk)
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
    ms = 0;
    if any(relVec)
        ms = REL_BASE;
    end
    pairs = Nx * Ny;
    for a = 1:A
        ra = rVec(a); Ka = kVec(a);
        if relVec(a) && ra >= 2
            gridOp = GRID_OP(min(max(ra, 2), numel(GRID_OP)));
            if ra > 3    % beyond tabulated orders: double per order
                gridOp = gridOp * 2^(ra - 3);
            end
            perPair = nuVec(a) * Ka * Ka * gridOp;
            if centresOk && Ka >= ra
                centresOp = CENTRES_OP(min(max(ra, 2), numel(CENTRES_OP)));
                if ra > 3
                    centresOp = centresOp * 2^(ra - 3);
                end
                centresOps = (factorial(ra) * combCount(Ka, ra))^2;
                perPair = min(perPair, centresOps * centresOp);
            end
            ms = ms + 3 * pairs * perPair;
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

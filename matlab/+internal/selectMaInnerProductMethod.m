function [chosen, pwCostOut, orbitCostOut] = selectMaInnerProductMethod( ...
        rVec, kVec, A, Nx, Ny, ...
        anyPer, anyRelNonper, anyRelPer, sigmaOverPMax, userMethod, ...
        verbose, relVec, nuVec, kVecY)
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
    % Priced by the same fitted law. The per-entry form this replaces
    % assumed a fixed cost per kernel entry; measurement contradicts
    % that, the per-entry cost falling as the arrays grow, which is what
    % the fitted exponent below 1 carries.
    pwCost = relRouteCostMs('bulger', r_max, pwSize);
    centresOk = sigmaOverPMax <= 0.03;   % _ORBIT_SIGMA_OVER_P_THRESHOLD
    orbitCost = predictOrbitCostMs(rVec, kVec, A, Nx, Ny, relVec, ...
                                   nuVec, centresOk, kVecY);
    pwCostOut = pwCost;
    orbitCostOut = orbitCost;
    if pwCost <= orbitCost
        chosen = 'bulger';
    else
        chosen = 'mobius';
    end
end


function ms = relRouteCostMs(route, r_a, term)
%RELROUTECOSTMS  Predicted wall time (ms) for one route, from its law.
%
%   Cost model for the method comparison: one power law per route and
%   tuple order,
%
%       t_ms = exp(a_r) * term ^ b_r
%
%   on the quantity each route works over -- Bulger's method and the
%   tuple-centres route on the tuple-pair entries they materialise, the
%   translation grid on the node count times the larger value count.
%   Every term carries the event-pair count, since both methods price
%   per pair. The Mobius side takes the smaller of its two routes, as
%   the orchestrator does. Each law is fitted against the quantity the
%   caller passes, not an idealisation of it, so the intercepts absorb
%   the constant factors between them; a refit must use the same terms.
%   Orders above 4 reuse the r = 4 row.
%
%   Fitted on 684 cells: r in {2, 3, 4}, value counts 5 to 40, event
%   counts 1 to 64, three kernel widths, both periodicities, three
%   weight profiles, equal and unequal value counts, each route timed in
%   isolation with relAttrRoute pinning it. Cross-validated on the
%   routing decision, eight-fold: 0.93 against 0.62 for the count-based
%   model it replaces.
%
%   Constants are per-language: the two implementations amortise
%   differently. Refit with tools/calibrateRelIpCost.m, and run it with
%   'check', true first -- three earlier fits shipped badly because an
%   axis was missing from the sweep, and the check asserts that each
%   axis varies what it claims to.
    switch route
        case 'bulger'
            A = [-7.2149, -8.0168, -7.7822];
            B = [ 0.6970,  0.7493,  0.7500];
        case 'centres'
            A = [-7.7429, -8.6263, -8.8269];
            B = [ 0.6800,  0.7781,  0.8029];
        case 'grid'
            A = [-4.1388, -2.1985, -0.5409];
            B = [ 0.3916,  0.5021,  0.5768];
        otherwise
            error('mpt:badRoute', 'Unknown route ''%s''.', route);
    end
    idx = min(max(r_a, 2), 4) - 1;
    ms = exp(A(idx)) * max(term, 1)^B(idx);
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
    % No flat relative base: each route's law carries its own intercept,
    % so adding one would double-count the setup it already prices.
    REL_BASE = 0.0;
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
            perPair = relRouteCostMs('grid', ra, ...
                pairs * nuVec(a) * max(Ka, KaY));
            if centresOk && Ka >= ra && KaY >= ra
                centresOp = CENTRES_OP(min(max(ra, 2), numel(CENTRES_OP)));
                if ra > 3
                    centresOp = centresOp * 2^(ra - 3);
                end
                mX = factorial(ra) * combCount(Ka, ra);
                mY = factorial(ra) * combCount(KaY, ra);
                perPair = min(perPair, relRouteCostMs('centres', ra, ...
                    pairs * (mX * mY + mX * mX + mY * mY)));
            end
            ms = ms + perPair;
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

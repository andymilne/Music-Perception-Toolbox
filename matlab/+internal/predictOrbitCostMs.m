function ms = predictOrbitCostMs(rVec, kVec, A, Nx, Ny, relVec, ...
                                  nuVec, centresOk, kVecY, ...
                                  skipXX, skipYY)
%INTERNAL.PREDICTORBITCOSTMS  Flat orbit (Moebius) route price, in ms.
%
%   MS = INTERNAL.PREDICTORBITCOSTMS(RVEC, KVEC, A, NX, NY, RELVEC,
%   NUVEC, CENTRESOK, KVECY, SKIPXX, SKIPYY) prices the flat
%   per-attribute orbit route over A attributes.
%
%   Promoted out of INTERNAL.SELECTMAINNERPRODUCTMETHOD's local function
%   of the same name: INTERNAL.NESTEDCOST prices a nested density's
%   flat companion attributes with exactly this model, as the Python
%   _nested_cost._flat_companion_cost_ms prices them with
%   _predict_orbit_cost_ms. The selector delegates here, so the flat
%   selector and the nested cost model cannot drift apart.
%
%   See also INTERNAL.SELECTMAINNERPRODUCTMETHOD, INTERNAL.NESTEDCOST.

    % Per-attribute sum, mirror of Python _predict_orbit_cost_ms:
    % relative attributes are priced at the cheaper of the
    % tuple-centres closed form and the batched grid contraction
    % (three matrices each) by the fitted laws of
    % INTERNAL.RELROUTECOSTMS, with the centres term blocked above the
    % sigma/P threshold; absolute attributes at the per-order constants
    % ABS (measured r = 2..6, extrapolated above).
    %
    % SKIPXX / SKIPYY exclude a self matrix that is memoised or not
    % consumed (mirroring predictPairwiseKernelSize). The centres term
    % drops the skipped self work exactly; the grid term and the
    % per-order absolute constants were fitted on the full
    % three-matrix computation, so they are scaled by the fraction of
    % matrices still to be computed --- an approximation that
    % under-discounts (setup is not per-matrix), biasing near-crossover
    % routing toward Bulger's method, the cheap-to-mispick side.
    % Defaults false reproduce the full-triple pricing exactly.
    ABS        = [NaN, 3.0, 11.2, 45.0, 150.0, 500.0, 1500.0, 4500.0];
    % No flat relative base is added: each route's law carries its own
    % multiplicative intercept, so adding one would double-count the
    % setup it already prices; the per-attribute floor below is applied
    % with max instead.
    % Setup floor for the Moebius route on one relative attribute, in
    % ms, as [fixed, perMatrix] by tuple order (rows r = 2, 3, 4; higher
    % orders reuse the r = 4 row); the floor for a call computing
    % nMatrices of the three inner matrices is fixed + perMatrix *
    % nMatrices, applied with max, not added. Twin of the Python
    % dispatch._ORBIT_REL_FLOOR_MS. Both laws in relRouteCostMs are
    % multiplicative in their term, so neither carries the route's fixed
    % per-attribute setup (orbit-table fetch, quadrature nodes,
    % dispatch), and as the term shrinks they extrapolate below a wall
    % time the route cannot go under; it bites at r = 2, where the term
    % is smallest. Values are per-language: measure with
    % tests/bench_orbit_rel_floor.m (median wall time of the cosine call
    % alone on a cold pair, three matrices, and on a pair with both self
    % products memoised, one matrix, minimised over K and solved for the
    % two coefficients). Measured on the maintainer's machine, the same
    % one the other MATLAB cost constants were fitted on: at r = 2,
    % 0.40 ms cold and 0.22 ms warm; at r = 3, 1.34 and 0.63 ms; at
    % r = 4, 24.4 and 8.5 ms (there the route is not flat in K, so the
    % smallest K is the floor). Two to three times the Python row on the
    % same machine, which is the spread a fixed cost has between the two
    % implementations; it only decides routing where the fitted laws
    % predict below it (r = 2 at small K, and a warm near-tie there).
    ORBIT_REL_FLOOR_MS = [0.137, 0.086; 0.280, 0.353; 0.497, 7.956];
    if nargin < 9 || isempty(kVecY)
        kVecY = kVec;
    end
    if nargin < 10 || isempty(skipXX); skipXX = false; end
    if nargin < 11 || isempty(skipYY); skipYY = false; end
    nMatrices = 1 + double(~skipXX) + double(~skipYY);
    ms = 0;
    pairs = Nx * Ny;
    for a = 1:A
        ra = rVec(a); Ka = kVec(a); KaY = kVecY(a);
        if relVec(a) && ra >= 2
            perPair = internal.relRouteCostMs('grid', ra, ...
                pairs * nuVec(a) * max(Ka, KaY) * (nMatrices / 3));
            if centresOk && Ka >= ra && KaY >= ra
                mX = factorial(ra) * combCount(Ka, ra);
                mY = factorial(ra) * combCount(KaY, ra);
                centresSize = pairs * mX * mY;
                if ~skipXX
                    centresSize = centresSize + pairs * mX * mX;
                end
                if ~skipYY
                    centresSize = centresSize + pairs * mY * mY;
                end
                perPair = min(perPair, internal.relRouteCostMs('centres', ra, ...
                    centresSize));
            end
            % Setup floor (see ORBIT_REL_FLOOR_MS above).
            fl = ORBIT_REL_FLOOR_MS(min(max(ra, 2), 4) - 1, :);
            perPair = max(perPair, fl(1) + fl(2) * nMatrices);
            ms = ms + perPair;
        elseif ra >= 2
            ms = ms + ABS(ra) * (nMatrices / 3);
        end
    end
end


function c = combCount(nn, kk)
%COMBCOUNT  Binomial coefficient C(nn, kk), integer-valued.
    if kk < 0 || kk > nn
        c = 0; return;
    end
    c = 1;
    for ii = 0:kk - 1
        c = c * (nn - ii) / (ii + 1);
    end
    c = round(c);
end

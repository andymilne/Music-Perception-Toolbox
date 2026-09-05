%% bench_orbit_rel_floor.m
%  Measure the Moebius route's per-attribute setup floor,
%  ORBIT_REL_FLOOR_MS in internal.predictOrbitCostMs (twin of the Python
%  tools/calibrate_orbit_rel_floor.py, which fills
%  dispatch._ORBIT_REL_FLOOR_MS).
%
%  The two relative-route cost laws in internal.relRouteCostMs are
%  multiplicative in their term, so they carry no fixed setup cost and
%  extrapolate below the route's wall time at small value counts. The
%  floor guards against that: for each tuple order it holds
%  [fixed, perMatrix] in ms, applied with max to the per-attribute price
%  of a call computing nMatrices of the three inner matrices.
%
%  For each order r = 2, 3, 4 this times the cosine call alone (the
%  densities are built outside the timer) on a *cold* pair, where all
%  three matrices are computed, and on a *warm* pair whose self inner
%  products are memoised (the densities returned by the first call are
%  reused), where only the cross matrix is; the median over repeats is
%  minimised over the smallest feasible K, where the route is flat in K,
%  and the two coefficients solve from
%
%      cold = fixed + 3 * perMatrix,     warm = fixed + perMatrix.
%
%  Relative-periodic at sigma/P = 0.025 is used because it measured the
%  same as or lower than relative-non-periodic at every order, so it is
%  the conservative row.
%
%  HOW TO RUN
%  ----------
%      >> bench_orbit_rel_floor
%
%  Paste the printed ORBIT_REL_FLOOR_MS row into
%  +internal/predictOrbitCostMs.m. Expect a factor of two or three of
%  spread between machines: this is a guard, not a fitted term, and it
%  only decides routing where the fitted laws predict below it (r = 2 at
%  small K). Takes well under a minute.

fprintf('\n=== bench_orbit_rel_floor ===\n');

borf_prevShowHints = mptDefaults('showHints', false);
borf_hintsCleanup  = onCleanup(@() mptDefaults(borf_prevShowHints)); %#ok<NASGU>

borf_P        = 12.0;
borf_sop      = 0.025;
borf_coldReps = 15;
borf_warmReps = 25;

borf_dens = @(seed, K, r) borfDens(seed, K, r, borf_P, borf_sop);
borf_row  = zeros(3, 2);

for borf_r = 2:4
    borf_cold = zeros(1, 5);
    borf_warm = zeros(1, 5);
    for borf_j = 1:5
        borf_K = borf_r + borf_j - 1;
        borf_t = zeros(1, borf_coldReps);
        for borf_i = 1:borf_coldReps
            borf_x = borf_dens(1, borf_K, borf_r);
            borf_y = borf_dens(2, borf_K, borf_r);
            borf_t0 = tic;
            cosSimExpTens(borf_x, borf_y, 'method', 'mobius', 'verbose', false);
            borf_t(borf_i) = toc(borf_t0);
        end
        borf_cold(borf_j) = 1e3 * median(borf_t);

        borf_x = borf_dens(1, borf_K, borf_r);
        borf_y = borf_dens(2, borf_K, borf_r);
        % One warming call: its returned densities carry the memoised
        % self inner products.
        [~, borf_x, borf_y] = cosSimExpTens(borf_x, borf_y, ...
            'method', 'mobius', 'verbose', false);
        borf_t = zeros(1, borf_warmReps);
        for borf_i = 1:borf_warmReps
            borf_t0 = tic;
            cosSimExpTens(borf_x, borf_y, 'method', 'mobius', 'verbose', false);
            borf_t(borf_i) = toc(borf_t0);
        end
        borf_warm(borf_j) = 1e3 * median(borf_t);
    end
    borf_t3 = min(borf_cold);
    borf_t1 = min(borf_warm);
    borf_pm = max((borf_t3 - borf_t1) / 2, 0);
    borf_f  = max(borf_t1 - borf_pm, 0);
    borf_row(borf_r - 1, :) = [round(borf_f, 3), round(borf_pm, 3)];
    fprintf(['r=%d: cold (3 matrices) min %.3f ms, warm (1 matrix) min ' ...
             '%.3f ms -> fixed %.3f, perMatrix %.3f; cold by K: %s\n'], ...
            borf_r, borf_t3, borf_t1, borf_f, borf_pm, ...
            mat2str(round(borf_cold, 2)));
end

fprintf('\nORBIT_REL_FLOOR_MS = [%.3f, %.3f; %.3f, %.3f; %.3f, %.3f];\n', ...
        borf_row(1, 1), borf_row(1, 2), borf_row(2, 1), borf_row(2, 2), ...
        borf_row(3, 1), borf_row(3, 2));

mptDefaults(borf_prevShowHints);
clear borf_hintsCleanup


function d = borfDens(seed, K, r, P, sop)
    rng(seed, 'twister');
    p = sort(P * rand(K, 1));
    w = 0.2 + 0.8 * rand(K, 1);
    d = buildExpTens(p, w, sop * P, r, true, true, P, 'verbose', false);
end

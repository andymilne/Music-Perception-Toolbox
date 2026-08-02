%% bench_ip_dispatch.m
%  Times 'bulger', 'mobius', and 'auto' on single EDO-approximation
%  pairs (JI harmonic reference vs n-EDO, rel-per, sigma = 6) across
%  tensor orders r = 2..5. Reports, per (r, n) cell:
%
%    * c_pw, c_orb, ratio — per-op costs of the two paths and their
%      unit-cost ratio. With the slabbed translation grid, c_orb should
%      be roughly flat across n; the per-r medians inform the fitted
%      constants in relRouteCostMs
%      (+internal/selectMaInnerProductMethod.m), which
%      tools/calibrateRelIpCost measures.
%
%    * auto~ — which forced method the 'auto' timing sits closer to
%      (an inference; with 'showHints' on, the dispatch messages name
%      the choice directly).
%
%    * probe_ovh — t_auto minus the faster forced method. Where 'auto'
%      pre-screens this is ~0; where it probes, it measures the
%      one-time probe cost (probe timings are cached per session, so
%      within a batched sweep this cost is paid once, not per pair).
%
%  The reference uses eight prime harmonics (K_x = 8) rather than the
%  demo's five, so that the sweep reaches r = 6 with a collection
%  comfortably larger than the tuple size. Per-op costs are insensitive
%  to the reference size.
%
%  A per-path time cap skips larger n for a path once a single run
%  exceeds TIME_CAP seconds (reported as NaN), so the bench stays
%  bounded on machines where one path is very slow at high r. The
%  Möbius timing is additionally wrapped in try/catch so a missing
%  orbit table for some r reports rather than aborts.
%
%  A final SYMMETRIC section times K x K pairs (an EDO against a
%  transposed copy). The asymmetric sections' measurements are
%  dominated by the large side's self-norm, whose kernel shapes match
%  a large-by-large cross term, so the constants should transfer:
%  c_orb should read the same flat value here. c_pw is expected to
%  degrade at the largest symmetric sizes (all three inner products
%  are large simultaneously, so the pairwise working set leaves its
%  resident regime at ~3x smaller K than in the asymmetric sections);
%  the probe measures the actual pair in that band, so 'auto' should
%  nevertheless track the faster method — that, not the c_pw value,
%  is the pass criterion there.
%
%  Run from anywhere with the toolbox on the path.

refPitches = log2([1, 3, 5, 7, 11, 13, 17, 19]) * 1200;
sigma  = 6;
isRel  = 1;
isPer  = 1;
period = 1200;

rList     = [2, 3, 4, 5];
nListPerR = {[40, 60, 80, 100], [15, 20, 30, 40], ...
             [10, 12, 15, 18], [8, 9, 10, 12]};
bellPerR  = [2, 5, 15, 52];
nReps     = 3;
TIME_CAP  = 60;   % seconds; per-path, per-r skip threshold

K_x = numel(refPitches);
N_u = internal.autoNtauDefault(period, sigma);
ff  = @(K, k) prod(K:-1:(K - k + 1)) * (K >= k);

fprintf('bench_ip_dispatch: sigma = %g, N_u = %d, K_x = %d, cap = %ds\n', ...
    sigma, N_u, K_x, TIME_CAP);

for ri = 1:numel(rList)
    r     = rList(ri);
    nList = nListPerR{ri};
    B_r   = bellPerR(ri);

    fprintf('\n--- r = %d ---\n', r);
    fprintf('%6s %10s %10s %10s %10s %10s %8s %8s %11s\n', ...
        'n-EDO', 't_bul(s)', 't_mob(s)', 't_auto(s)', ...
        'c_pw(ns)', 'c_orb(ns)', 'ratio', 'auto~', 'probe_ovh');

    ratios  = NaN(1, numel(nList));
    skipBul = false;
    skipMob = false;
    for i = 1:numel(nList)
        n = nList(i);
        edoPitches = (0:n-1) * (1200 / n);

        % --- forced bulger ---
        if skipBul
            tBul = NaN;
        else
            tStart = tic;   % warmup doubles as the cap check
            cosSimExpTens(refPitches, [], edoPitches, [], ...
                sigma, r, isRel, isPer, period, 'method', 'bulger');
            tWarm = toc(tStart);
            if tWarm > TIME_CAP
                tBul = tWarm;
                skipBul = true;
            else
                tB = zeros(1, nReps);
                for k = 1:nReps
                    tStart = tic;
                    cosSimExpTens(refPitches, [], edoPitches, [], ...
                        sigma, r, isRel, isPer, period, 'method', 'bulger');
                    tB(k) = toc(tStart);
                end
                tBul = median(tB);
            end
        end

        % --- forced mobius ---
        if skipMob
            tMob = NaN;
        else
            try
                tStart = tic;
                cosSimExpTens(refPitches, [], edoPitches, [], ...
                    sigma, r, isRel, isPer, period, 'method', 'mobius');
                tWarm = toc(tStart);
                if tWarm > TIME_CAP
                    tMob = tWarm;
                    skipMob = true;
                else
                    tM = zeros(1, nReps);
                    for k = 1:nReps
                        tStart = tic;
                        cosSimExpTens(refPitches, [], edoPitches, [], ...
                            sigma, r, isRel, isPer, period, ...
                            'method', 'mobius');
                        tM(k) = toc(tStart);
                    end
                    tMob = median(tM);
                end
            catch err
                fprintf('  mobius unavailable at r = %d: %s\n', ...
                    r, err.message);
                tMob = NaN;
                skipMob = true;
            end
        end

        % --- auto (single rep after warmup) ---
        cosSimExpTens(refPitches, [], edoPitches, [], ...
            sigma, r, isRel, isPer, period);
        tStart = tic;
        cosSimExpTens(refPitches, [], edoPitches, [], ...
            sigma, r, isRel, isPer, period);
        tAuto = toc(tStart);

        % --- per-op costs ---
        P_x = ff(K_x, r);
        P_y = ff(n, r);
        pwOps  = P_x * P_y + P_x^2 + P_y^2;
        orbOps = B_r * N_u * (K_x * n + K_x^2 + n^2);
        c_pw  = tBul / pwOps  * 1e9;
        c_orb = tMob / orbOps * 1e9;
        ratios(i) = c_orb / c_pw;

        if isnan(tBul) || isnan(tMob)
            autoPick = '?';
        elseif abs(tAuto - tBul) <= abs(tAuto - tMob)
            autoPick = 'bulger';
        else
            autoPick = 'mobius';
        end
        probeOvh = tAuto - min(tBul, tMob);

        fprintf('%6d %10.3f %10.3f %10.3f %10.2f %10.2f %8.2f %8s %11.3f\n', ...
            n, tBul, tMob, tAuto, c_pw, c_orb, ratios(i), ...
            autoPick, probeOvh);
    end

    okRatios = ratios(~isnan(ratios));
    if ~isempty(okRatios)
        fprintf('Median unit-cost ratio at r = %d: %.1f\n', ...
            r, median(okRatios));
    else
        fprintf('No complete ratio measurements at r = %d.\n', r);
    end
end

%% === Symmetric K x K section ===
symList  = {[40, 70, 100], [20, 30, 40]};
symR     = [2, 3];
symBell  = [2, 5];

for ri = 1:numel(symR)
    r   = symR(ri);
    B_r = symBell(ri);

    fprintf('\n--- symmetric K x K, r = %d ---\n', r);
    fprintf('%9s %10s %10s %10s %10s %10s %8s %8s %11s\n', ...
        'K x K', 't_bul(s)', 't_mob(s)', 't_auto(s)', ...
        'c_pw(ns)', 'c_orb(ns)', 'ratio', 'auto~', 'probe_ovh');

    skipBul = false;
    skipMob = false;
    for K = symList{ri}
        pA = (0:K-1) * (1200 / K);
        pB = mod(pA + 37.3, 1200);

        if skipBul
            tBul = NaN;
        else
            tStart = tic;
            cosSimExpTens(pA, [], pB, [], ...
                sigma, r, isRel, isPer, period, 'method', 'bulger');
            tWarm = toc(tStart);
            if tWarm > TIME_CAP
                tBul = tWarm;
                skipBul = true;
            else
                tB = zeros(1, nReps);
                for k = 1:nReps
                    tStart = tic;
                    cosSimExpTens(pA, [], pB, [], ...
                        sigma, r, isRel, isPer, period, ...
                        'method', 'bulger');
                    tB(k) = toc(tStart);
                end
                tBul = median(tB);
            end
        end

        if skipMob
            tMob = NaN;
        else
            try
                tStart = tic;
                cosSimExpTens(pA, [], pB, [], ...
                    sigma, r, isRel, isPer, period, 'method', 'mobius');
                tWarm = toc(tStart);
                if tWarm > TIME_CAP
                    tMob = tWarm;
                    skipMob = true;
                else
                    tM = zeros(1, nReps);
                    for k = 1:nReps
                        tStart = tic;
                        cosSimExpTens(pA, [], pB, [], ...
                            sigma, r, isRel, isPer, period, ...
                            'method', 'mobius');
                        tM(k) = toc(tStart);
                    end
                    tMob = median(tM);
                end
            catch err
                fprintf('  mobius unavailable at r = %d: %s\n', ...
                    r, err.message);
                tMob = NaN;
                skipMob = true;
            end
        end

        cosSimExpTens(pA, [], pB, [], sigma, r, isRel, isPer, period);
        tStart = tic;
        cosSimExpTens(pA, [], pB, [], sigma, r, isRel, isPer, period);
        tAuto = toc(tStart);

        P_K = ff(K, r);
        pwOps  = 3 * P_K^2;
        orbOps = B_r * N_u * 3 * K^2;
        c_pw  = tBul / pwOps  * 1e9;
        c_orb = tMob / orbOps * 1e9;

        if isnan(tBul) || isnan(tMob)
            autoPick = '?';
        elseif abs(tAuto - tBul) <= abs(tAuto - tMob)
            autoPick = 'bulger';
        else
            autoPick = 'mobius';
        end
        probeOvh = tAuto - min(tBul, tMob);

        fprintf('%4dx%-4d %10.3f %10.3f %10.3f %10.2f %10.2f %8.2f %8s %11.3f\n', ...
            K, K, tBul, tMob, tAuto, c_pw, c_orb, c_orb / c_pw, ...
            autoPick, probeOvh);
    end
end

fprintf(['\nPer-r medians inform relRouteCostMs in\n' ...
         '+internal/selectMaInnerProductMethod.m, which\n' ...
         'tools/calibrateRelIpCost fits. Where t_mob < t_bul, the\n' ...
         'Möbius method wins outright\n' ...
         'at that size; pairwise cost grows as n^(2r), so the\n' ...
         'crossover moves to smaller n as r rises.\n']);

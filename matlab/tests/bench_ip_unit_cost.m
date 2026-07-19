%% bench_ip_unit_cost.m
%  Diagnostic for cosSimExpTens auto-dispatch on this machine, using the
%  EDO-approximation workload (JI reference vs n-EDO, r = 2, rel-per).
%
%  Answers two questions:
%
%  1. Is the mass-aware cancellation diagnostic active? With the current
%     +mobius/orbitInnerRelSingleMultiset.m, an explicit 'method','mobius' call on
%     this workload returns the orbit-path value, which differs from the
%     'bulger' value by ~1e-9 (translation-grid integration accuracy).
%     If the two values are EXACTLY equal, the orbit result was
%     discarded by the cancellation guard and recomputed via the
%     pairwise path — meaning the +mobius files on the path predate the
%     mass-aware diagnostic, and 'auto' pays for both paths on every
%     Möbius-routed pair.
%
%  2. What is the per-op cost of a translation-grid orbit kernel op
%     relative to a pairwise kernel op on this machine? This is the
%     ORBIT_GRID_OP_UNIT_COST constant in localOrbitIPGridFactors
%     (cosSimExpTens.m). The shipped value is calibrated on the Python
%     implementation; the ratio is implementation-dependent, so if
%     'auto' routes to the Möbius method where the pairwise path is
%     faster (or vice versa), this measurement gives the corrected
%     value.
%
%  Run from anywhere with the toolbox on the path. Takes ~1 minute.

refPitches = [0, log2(3), log2(5), log2(7), log2(11)] * 1200;
sigma  = 6;
r      = 2;
isRel  = 1;
isPer  = 1;
period = 1200;

nList  = [40, 60, 80, 100];
nReps  = 3;   % timed repetitions per cell; median reported

K_x = numel(refPitches);
N_u = internal.autoNtauDefault(period, sigma);
B_r = 2;   % Bell number, r = 2

ff = @(K, k) prod(K:-1:(K - k + 1)) * (K >= k);

fprintf('bench_ip_unit_cost: sigma = %g, N_u = %d, K_x = %d\n\n', ...
    sigma, N_u, K_x);
fprintf('%6s %12s %12s %14s %12s %12s %8s\n', ...
    'n-EDO', 't_bulger(s)', 't_mobius(s)', '|s_mob-s_bul|', ...
    'c_pw(ns)', 'c_orb(ns)', 'ratio');

ratios   = zeros(1, numel(nList));
diffs    = zeros(1, numel(nList));
for i = 1:numel(nList)
    n = nList(i);
    edoPitches = (0:n-1) * (1200 / n);

    % Warm-up (both paths) so caches and orbit tables are hot.
    sBul = cosSimExpTens(refPitches, [], edoPitches, [], ...
        sigma, r, isRel, isPer, period, 'method', 'bulger');
    sMob = cosSimExpTens(refPitches, [], edoPitches, [], ...
        sigma, r, isRel, isPer, period, 'method', 'mobius');

    tB = zeros(1, nReps);
    tM = zeros(1, nReps);
    for k = 1:nReps
        tStart = tic;
        sBul = cosSimExpTens(refPitches, [], edoPitches, [], ...
            sigma, r, isRel, isPer, period, 'method', 'bulger');
        tB(k) = toc(tStart);
        tStart = tic;
        sMob = cosSimExpTens(refPitches, [], edoPitches, [], ...
            sigma, r, isRel, isPer, period, 'method', 'mobius');
        tM(k) = toc(tStart);
    end
    tBul = median(tB);
    tMob = median(tM);

    % Op counts (shared unit: one kernel evaluation).
    P_x = ff(K_x, r);
    P_y = ff(n, r);
    pwOps  = P_x * P_y + P_x^2 + P_y^2;
    orbOps = B_r * N_u * (K_x * n + K_x^2 + n^2);

    c_pw  = tBul / pwOps  * 1e9;
    c_orb = tMob / orbOps * 1e9;
    ratios(i) = c_orb / c_pw;
    diffs(i)  = abs(sMob - sBul);

    fprintf('%6d %12.3f %12.3f %14.3e %12.2f %12.2f %8.2f\n', ...
        n, tBul, tMob, diffs(i), c_pw, c_orb, ratios(i));
end

fprintf('\n--- Verdicts ---\n');
if max(diffs) == 0
    fprintf(['[1] mobius output is EXACTLY equal to bulger output: the\n' ...
             '    cancellation guard discarded the orbit result and fell\n' ...
             '    back to the pairwise path. The +mobius files on the\n' ...
             '    MATLAB path predate the mass-aware diagnostic; update\n' ...
             '    +mobius/orbitInnerRelSingleMultiset.m and\n' ...
             '    +mobius/innerProductOrbitGrid.m. The unit-cost\n' ...
             '    measurement below is contaminated by the fallback\n' ...
             '    (mobius timings include a pairwise recompute) — rerun\n' ...
             '    after updating.\n']);
else
    fprintf(['[1] mobius and bulger outputs differ by ~%.1e: the\n' ...
             '    mass-aware cancellation diagnostic is active and the\n' ...
             '    orbit result is being used.\n'], max(diffs));
    fprintf(['[2] Measured per-op unit-cost ratio (median across n):\n' ...
             '    ORBIT_GRID_OP_UNIT_COST = %.1f\n' ...
             '    (shipped value 1.6, calibrated on the Python\n' ...
             '    implementation). If the measured value differs\n' ...
             '    substantially, set it in localOrbitIPGridFactors in\n' ...
             '    cosSimExpTens.m.\n'], median(ratios));
end

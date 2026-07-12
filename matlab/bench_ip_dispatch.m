%% bench_ip_dispatch.m
%  Times 'bulger', 'mobius', and 'auto' on single EDO-approximation
%  pairs (JI reference vs n-EDO, r = 2, rel-per, sigma = 6). Reports:
%
%    * c_pw, c_orb, ratio — per-op costs of the two paths and their
%      unit-cost ratio. With the slabbed translation grid, c_orb should
%      be roughly flat across n; the median ratio is the value for
%      ORBIT_GRID_OP_UNIT_COST in localOrbitIPGridFactors
%      (cosSimExpTens.m). If mobius never beats bulger for n <= 102,
%      the constant should be large enough that the bulger pre-screen
%      covers that whole range.
%
%    * probe overhead — t_auto minus the faster of the two forced
%      methods. Nonzero only where 'auto' probes (neither pre-screen
%      fires); measures the full per-pair probe cost including subset
%      density builds and warm-up passes.
%
%  Run from anywhere with the toolbox on the path. Takes ~1 minute.

refPitches = [0, log2(3), log2(5), log2(7), log2(11)] * 1200;
sigma  = 6;
r      = 2;
isRel  = 1;
isPer  = 1;
period = 1200;

nList  = [40, 60, 80, 95, 100];
nReps  = 3;

K_x = numel(refPitches);
N_u = internal.autoNtauDefault(period, sigma);
B_r = 2;
ff  = @(K, k) prod(K:-1:(K - k + 1)) * (K >= k);

fprintf('bench_ip_dispatch: sigma = %g, N_u = %d, K_x = %d\n\n', ...
    sigma, N_u, K_x);
fprintf('%6s %10s %10s %10s %10s %10s %7s %11s\n', ...
    'n-EDO', 't_bul(s)', 't_mob(s)', 't_auto(s)', ...
    'c_pw(ns)', 'c_orb(ns)', 'ratio', 'probe_ovh');

ratios = zeros(1, numel(nList));
for i = 1:numel(nList)
    n = nList(i);
    edoPitches = (0:n-1) * (1200 / n);

    tB = zeros(1, nReps); tM = zeros(1, nReps); tA = zeros(1, nReps);
    % Warm-up all three.
    cosSimExpTens(refPitches, [], edoPitches, [], ...
        sigma, r, isRel, isPer, period, 'method', 'bulger');
    cosSimExpTens(refPitches, [], edoPitches, [], ...
        sigma, r, isRel, isPer, period, 'method', 'mobius');
    cosSimExpTens(refPitches, [], edoPitches, [], ...
        sigma, r, isRel, isPer, period);
    for k = 1:nReps
        tStart = tic;
        cosSimExpTens(refPitches, [], edoPitches, [], ...
            sigma, r, isRel, isPer, period, 'method', 'bulger');
        tB(k) = toc(tStart);
        tStart = tic;
        cosSimExpTens(refPitches, [], edoPitches, [], ...
            sigma, r, isRel, isPer, period, 'method', 'mobius');
        tM(k) = toc(tStart);
        tStart = tic;
        cosSimExpTens(refPitches, [], edoPitches, [], ...
            sigma, r, isRel, isPer, period);
        tA(k) = toc(tStart);
    end
    tBul = median(tB); tMob = median(tM); tAuto = median(tA);

    P_x = ff(K_x, r);
    P_y = ff(n, r);
    pwOps  = P_x * P_y + P_x^2 + P_y^2;
    orbOps = B_r * N_u * (K_x * n + K_x^2 + n^2);

    c_pw  = tBul / pwOps  * 1e9;
    c_orb = tMob / orbOps * 1e9;
    ratios(i) = c_orb / c_pw;
    probeOvh  = tAuto - min(tBul, tMob);

    fprintf('%6d %10.3f %10.3f %10.3f %10.2f %10.2f %7.2f %11.3f\n', ...
        n, tBul, tMob, tAuto, c_pw, c_orb, ratios(i), probeOvh);
end

fprintf(['\nMedian unit-cost ratio: %.1f (current ORBIT_GRID_OP_UNIT_COST' ...
         ' = 7.5).\n'], median(ratios));
fprintf(['If t_mob > t_bul at every n here, mobius does not win in this\n' ...
         'range on this machine and the constant should be raised until\n' ...
         'the bulger pre-screen covers it; if probe_ovh dominates the\n' ...
         'auto/bulger gap instead, the probe cost is the target.\n']);

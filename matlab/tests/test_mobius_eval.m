%% test_mobius_eval.m — set partitions and single multiset point evaluators
%
%  Tests for the v2.2 point-evaluator machinery in matlab/+mobius/:
%    getSetPartitionsWithMobius, evalOrbitAbs, evalOrbitRel.
%
%  Mirrors the relevant sections of python/tests/test_mobius.py.
%
%  Standalone-runnable; appends to `results` when invoked from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


%% ---- Set-partition counts match Bell numbers ----

% B_r counts: B_1=1, B_2=2, B_3=5, B_4=15, B_5=52, B_6=203.
expectedCounts = [1 2 5 15 52 203];
ok = true;
for r = 1:6
    parts = mobius.getSetPartitionsWithMobius(r);
    if numel(parts) ~= expectedCounts(r)
        ok = false; break
    end
end
results{end+1, 1} = 'mobius.getSetPartitionsWithMobius: count = B_r for r=1..6';
results{end, 2}   = ok;


%% ---- Set-partition: each partition covers {1..r} exactly once ----

ok = true;
for r = 2:5
    parts = mobius.getSetPartitionsWithMobius(r);
    for k = 1:numel(parts)
        all_idx = [];
        for b = 1:numel(parts(k).blocks)
            all_idx = [all_idx, parts(k).blocks{b}]; %#ok<AGROW>
        end
        if ~isequal(sort(all_idx), 1:r)
            ok = false; break
        end
    end
    if ~ok; break; end
end
results{end+1, 1} = 'mobius.getSetPartitionsWithMobius: blocks cover {1..r} exactly';
results{end, 2}   = ok;


%% ---- Set-partition: mu = prod_l (-1)^(m_l - 1) * (m_l - 1)! ----

ok = true;
for r = 2:5
    parts = mobius.getSetPartitionsWithMobius(r);
    for k = 1:numel(parts)
        expectedMu = 1;
        for b = 1:numel(parts(k).blocks)
            m = numel(parts(k).blocks{b});
            if m > 1
                expectedMu = expectedMu * (-1)^(m - 1) * factorial(m - 1);
            end
        end
        if parts(k).mu ~= expectedMu
            ok = false; break
        end
    end
    if ~ok; break; end
end
results{end+1, 1} = 'mobius.getSetPartitionsWithMobius: mu matches block-product formula';
results{end, 2}   = ok;


%% ---- evalOrbitAbs vs direct enumeration ----

ok = true; allFinite = true;
seedBase = 1234;
for r = 2:3
    for n = [5, 7]
        rng(seedBase + r * 100 + n, 'twister');
        p = sort(2000 * rand(n, 1));
        w = 0.5 + rand(n, 1);
        sigma = 100.0;

        % Query points near actual source positions so the tensor
        % value is meaningfully non-zero. (Random queries far from
        % sources land in the tail of every Gaussian and the tensor
        % is at machine epsilon there, where rel-err is meaningless.)
        n_q = 6;
        x = zeros(r, n_q);
        for q = 1:n_q
            % Pick r random sources, jitter each by sigma/2.
            picks = randperm(n, r);
            x(:, q) = p(picks) + sigma * 0.5 * randn(r, 1);
        end

        valDirect = v22_directEvalAbs(p, w, sigma, r, x);
        valOrbit = mobius.evalOrbitAbs(p, w, sigma, r, x);

        relErr = max(abs(valOrbit - valDirect) ./ max(abs(valDirect), 1e-300));
        if ~all(isfinite(valOrbit)); allFinite = false; end
        if relErr >= 1e-10
            ok = false;
        end
    end
end
results{end+1, 1} = 'mobius.evalOrbitAbs: matches direct enumeration (r=2..3, n=5,7)';
results{end, 2}   = ok && allFinite;


%% ---- evalOrbitAbs r=1: reduces to direct sum ----

% T_abs(x) = sum_i w_i * exp(-(x - p_i)^2 / (2 sigma^2)) for r=1.
rng(99, 'twister');
n = 6;
p = sort(1000 * rand(n, 1));
w = 0.5 + rand(n, 1);
sigma = 30.0;
x_q = (200:200:1200);  % (1, n_q) row vector — evalOrbitAbs needs (r=1, n_q)

valOrbit = mobius.evalOrbitAbs(p, w, sigma, 1, x_q);
% Direct sum: diffs of shape (n_q, n) from broadcasting.
diffs = x_q' - p';  % (n_q, 1) - (1, n) = (n_q, n)
expected = exp(-(diffs .^ 2) / (2 * sigma^2)) * w;  % (n_q, 1)
results{end+1, 1} = 'mobius.evalOrbitAbs: r=1 reduces to single-Gaussian-sum';
results{end, 2}   = max(abs(valOrbit - expected)) < 1e-12;


%% ---- evalOrbitAbs periodic mode wraps differences ----

% At periodic period P, two queries that differ by integer multiples
% of P should give the same result.
P = 1200;
rng(77, 'twister');
n = 5;
p = sort(P * rand(n, 1));
w = 0.5 + rand(n, 1);
sigma = 50.0;
r = 2;
n_q = 3;
x_base = P * rand(r, n_q);
x_shifted = x_base + P;  % shift one full period

val1 = mobius.evalOrbitAbs(p, w, sigma, r, x_base, ...
    'is_per', true, 'period', P);
val2 = mobius.evalOrbitAbs(p, w, sigma, r, x_shifted, ...
    'is_per', true, 'period', P);
results{end+1, 1} = 'mobius.evalOrbitAbs: periodic mode is period-translation invariant';
results{end, 2}   = max(abs(val1 - val2)) < 1e-12;


%% ---- evalOrbitAbs returns finite cancellation ratio ----

rng(5, 'twister');
n = 8;
p = sort(2000 * rand(n, 1));
w = 0.5 + rand(n, 1);
sigma = 30.0;
x = 2000 * rand(3, 4);
[vals, ratios] = mobius.evalOrbitAbs(p, w, sigma, 3, x, ...
    'returnCancellationRatio', true);
results{end+1, 1} = 'mobius.evalOrbitAbs: returns finite cancellation ratios';
results{end, 2}   = all(isfinite(vals)) && all(isfinite(ratios)) && ...
                   all(ratios >= 0) && all(ratios <= 1 + 1e-12);


%% ---- evalOrbitRel: degenerate r=1 returns sum(w) ----

rng(1, 'twister');
n = 4;
p = sort(1000 * rand(n, 1));
w = 0.5 + rand(n, 1);
sigma = 25.0;
x_rel = zeros(0, 5);  % r-1 = 0 for r=1
vals = mobius.evalOrbitRel(p, w, sigma, 1, x_rel);
results{end+1, 1} = 'mobius.evalOrbitRel: r=1 degenerate case returns sum(w)';
results{end, 2}   = numel(vals) == 5 && all(abs(vals - sum(w)) < 1e-14);


%% ---- evalOrbitRel: integral of T_abs over u recovers T_rel ----

% T_rel(Δ) = (1/Z_t) * ∫ T_abs(u, u+Δ_1, ..., u+Δ_{r-1}) du.
% Compare evalOrbitRel against a manual quadrature using evalOrbitAbs.
rng(2, 'twister');
n = 6;
p = sort(2000 * rand(n, 1));
w = 0.5 + rand(n, 1);
sigma = 30.0;
r = 3;
x_rel = randn(r - 1, 4) * 200;

valsAuto = mobius.evalOrbitRel(p, w, sigma, r, x_rel);

% Manual quadrature mirroring the function's own grid.
u_min = min(p) - max(0, max(x_rel(:))) - 8 * sigma;
u_max = max(p) - min(0, min(x_rel(:))) + 8 * sigma;
N_u = max(64, ceil((u_max - u_min) / sigma * 10));
u_grid = linspace(u_min, u_max, N_u);
n_q = size(x_rel, 2);
F = zeros(N_u, n_q);
for j = 1:N_u
    u = u_grid(j);
    x_full = zeros(r, n_q);
    x_full(1, :) = u;
    x_full(2:end, :) = u + x_rel;
    F(j, :) = mobius.evalOrbitAbs(p, w, sigma, r, x_full)';
end
manualIntegral = trapz(u_grid, F, 1);
Z_t = sigma * sqrt(2 * pi / r);
valsManual = manualIntegral(:) / Z_t;

results{end+1, 1} = 'mobius.evalOrbitRel: matches manual u-grid quadrature';
results{end, 2}   = max(abs(valsAuto - valsManual)) < 1e-10 * max(abs(valsManual));


%% ---- evalOrbitRel periodic: integral over [0, P) is period-invariant ----

P = 1200;
rng(3, 'twister');
n = 5;
p = sort(P * rand(n, 1));
w = 0.5 + rand(n, 1);
sigma = 60.0;
r = 2;
x_rel = randn(r - 1, 3) * 100;
x_rel_shifted = x_rel + P;

vals1 = mobius.evalOrbitRel(p, w, sigma, r, x_rel, 'is_per', true, 'period', P);
vals2 = mobius.evalOrbitRel(p, w, sigma, r, x_rel_shifted, 'is_per', true, 'period', P);
results{end+1, 1} = 'mobius.evalOrbitRel: periodic mode is period-translation invariant';
results{end, 2}   = max(abs(vals1 - vals2)) < 1e-9 * max(abs(vals1));


%% ---- evalOrbitRel periodic: factored culling matches direct ----
% In the valid regime (truncation window and query span both within half
% the circle) the factored-periodic strategy --- circular tabulation plus
% circular Lagrange read-back --- must match the direct strategy to the
% read-back accuracy tied to truncationSigmas. Exercises the periodic
% culling path. sigma = 30 keeps the window (~630) below half the period.
P = 1200;
rng(9, 'twister');
n = 7;
p = sort(P * rand(n, 1));
w = 0.5 + rand(n, 1);
sigma = 30.0;
r = 3;
x_rel = randn(r - 1, 4) * 40;   % tight spans, well within half the period
valsFac = mobius.evalOrbitRel(p, w, sigma, r, x_rel, ...
    'is_per', true, 'period', P, 'factored', 'on');
valsDir = mobius.evalOrbitRel(p, w, sigma, r, x_rel, ...
    'is_per', true, 'period', P, 'factored', 'off');
results{end+1, 1} = 'mobius.evalOrbitRel: periodic factored matches direct (valid regime)';
results{end, 2}   = max(abs(valsFac - valsDir)) < 1e-7 * max(abs(valsDir));


%% ---- evalOrbitRel: spectral strategy agrees with forced strategies ----
% Under 'auto' the spectral (Fourier) strategy engages for r = 2..4
% above its thresholds (n_q >= 16/32/64, K >= 2/8/16); its values must
% agree with the forced direct and factored strategies to within an
% order of the truncation floor, in both modes. The suite baseline
% (Inf) resolves to the 1e-12 floor.
rng(31, 'twister');
sp_K = 20;
sp_p = sort(1200 * rand(sp_K, 1));
sp_w = 0.5 + rand(sp_K, 1);
sp_ok = true;
for sp_r = 2:4
    sp_x = (rand(sp_r - 1, 80) - 0.5) * 400;
    for sp_per = [false, true]
        if sp_per, sp_P = 1200; else, sp_P = 0; end
        va = mobius.evalOrbitRel(sp_p, sp_w, 20.0, sp_r, sp_x, ...
            'is_per', sp_per, 'period', sp_P);                 % auto -> spectral
        vd = mobius.evalOrbitRel(sp_p, sp_w, 20.0, sp_r, sp_x, ...
            'is_per', sp_per, 'period', sp_P, 'factored', 'off');
        vf = mobius.evalOrbitRel(sp_p, sp_w, 20.0, sp_r, sp_x, ...
            'is_per', sp_per, 'period', sp_P, 'factored', 'on');
        pk = max(abs(vd));
        sp_ok = sp_ok && max(abs(va - vd)) <= 1e-11 * pk ...
                      && max(abs(va - vf)) <= 1e-11 * pk;
    end
end
results{end + 1, 1} = 'evalOrbitRel: spectral strategy matches direct and factored (r=2..4, both modes)';
results{end, 2}   = sp_ok;

% Periodic translation invariance under auto (wrapped deltas).
sp_x2 = (rand(1, 80) - 0.5) * 400;
va1 = mobius.evalOrbitRel(sp_p, sp_w, 20.0, 2, sp_x2, ...
    'is_per', true, 'period', 1200);
va2 = mobius.evalOrbitRel(sp_p, sp_w, 20.0, 2, sp_x2 + 3 * 1200, ...
    'is_per', true, 'period', 1200);
results{end + 1, 1} = 'evalOrbitRel: spectral strategy period-translation invariant';
results{end, 2}   = max(abs(va1 - va2)) <= 1e-12 * max(abs(va1));

% K = 1: no distinct pairs; the value is zero at floating point (the
% gate stands down at K = 1; the fall-through strategies cancel to ~0).
sp_v1 = mobius.evalOrbitRel(600.0, 1.0, 15.0, 2, [0.0, 30.0], ...
    'is_per', false, 'period', 0);
results{end + 1, 1} = 'evalOrbitRel: K=1 evaluates to ~0 (no distinct pairs)';
results{end, 2}   = max(abs(sp_v1)) < 1e-12;


%% ---- resolveSamplesPerSigma: explicit passthrough and derived values ----
% The derived count follows spp = max(2, ceil(k*sqrt(r)/(2*pi)) + 1) with
% k the resolved truncation width; values below mirror the Python
% resolver exactly (verified cross-language).
sppOK = internal.resolveSamplesPerSigma(10, 3, 6) == 10 ...
     && internal.resolveSamplesPerSigma([], 2, 6) == 3 ...
     && internal.resolveSamplesPerSigma([], 3, 6) == 3 ...
     && internal.resolveSamplesPerSigma([], 5, 6) == 4 ...
     && internal.resolveSamplesPerSigma([], 2, Inf) == 3 ...
     && internal.resolveSamplesPerSigma([], 3, Inf) == 4 ...
     && internal.resolveSamplesPerSigma([], 8, Inf) == 5 ...
     && internal.resolveSamplesPerSigma([], 2, 3) == 2;
results{end+1, 1} = 'internal.resolveSamplesPerSigma: passthrough and derived table';
results{end, 2}   = sppOK;


%% ---- standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    fprintf('\n=== mobius set-partition + eval tests ===\n\n');
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== Results: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        error('test_mobius_eval:failed', '%d test(s) failed.', nFail);
    end
end

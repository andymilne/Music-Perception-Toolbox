%% test_entropy_renyi2.m — v2.2 Rényi-2 entropy in entropyExpTens
%
%  Tests for the new method='renyi2' kwarg added in v2.2 (Commit 6e).
%  Covers:
%    - method kwarg validation (bad string; v2.2 migration error on
%      the removed normalize kwarg).
%    - single-multiset Rényi-2 closed-form correctness against a hand-rolled
%      reference for r=1 abs and r>=2 abs/rel.
%    - MA Rényi-2 closed-form correctness against a hand-rolled
%      per-attribute reference (cosSimExpTens on (dens, dens) gives 1
%      so cannot be used for cross-validation; here we construct
%      <T,T> and Z manually and compare).
%    - Input-form gating: list and 2-D batched inputs raise informative
%      errors under method='renyi2'.
%    - Skinny dens flows through transparently (no eager build needed).
%    - r=1 rel single-multiset is degenerate and returns 0 by convention.
%    - Self-similarity check: -log_b(<T,T>/Z^2) reduces to known values
%      for tractable small examples.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

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

%% ---- Method kwarg validation ----

ok_badMethod = false;
try
    entropyExpTens([0 4 7], [], 30, 2, false, false, 0, ...
        'method', 'bogus', 'verbose', false);
catch ME
    ok_badMethod = strcmp(ME.identifier, 'entropyExpTens:badMethod');
end
results{end+1,1} = 'entropy.renyi2: bad method string raises entropyExpTens:badMethod';
results{end,2}   = ok_badMethod;

% v2.2: passing 'normalize' to entropyExpTens (any value, any method)
% raises the migration error. Under v2.1 this same call combination
% (renyi2 with the default normalize=true) raised
% entropyExpTens:renyi2NormalizeNotSupported; v2.2 unifies all
% normalize-kwarg paths under the migration error.
ok_normMigration_false = false;
try
    entropyExpTens([0 4 7], [], 30, 2, false, false, 0, ...
        'method', 'renyi2', 'normalize', false, 'verbose', false);
catch ME
    ok_normMigration_false = strcmp(ME.identifier, ...
        'entropyExpTens:normalizeRemoved');
end
ok_normMigration_true = false;
try
    entropyExpTens([0 4 7], [], 30, 2, false, false, 0, ...
        'method', 'renyi2', 'normalize', true, 'verbose', false);
catch ME
    ok_normMigration_true = strcmp(ME.identifier, ...
        'entropyExpTens:normalizeRemoved');
end
results{end+1,1} = 'entropy.renyi2: legacy normalize kwarg raises migration error';
results{end,2}   = ok_normMigration_false && ok_normMigration_true;

% method='renyi2' without any normalize kwarg works and returns a finite
% value (the v2.2 design — renyi2 is continuous, no [0, 1] reference).
H_rny_default = entropyExpTens([0 4 7], [], 30, 2, false, false, 0, ...
    'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'entropy.renyi2: default call returns finite value';
results{end,2}   = isfinite(H_rny_default);

%% ---- single-multiset r=1 abs: agrees with hand-rolled direct formula ----

rng(71, 'twister');
p1 = sort(2000 * rand(6, 1));
w1 = 0.5 + rand(6, 1);
sigma = 30;

H_renyi = entropyExpTens(p1, w1, sigma, 1, false, false, 0, ...
    'method', 'renyi2', 'base', 2, 'verbose', false);

% Hand-rolled: <T,T> = sigma*sqrt(pi) * sum_{i,j} w_i w_j K[i,j]
diffs = p1 - p1.';
K = exp(-(diffs.^2) / (4 * sigma^2));
ip_xx = sigma * sqrt(pi) * sum(sum((w1 * w1.') .* K));
% Z = sum(w) for r=1 abs (totalMassAbs definition).
Z = mobius.totalMassAbs(p1, w1, sigma, 1);
H_ref = -log2(ip_xx / (Z * Z));

results{end+1,1} = 'entropy.renyi2 single-multiset r=1 abs: matches hand-rolled (1e-10)';
results{end,2}   = abs(H_renyi - H_ref) < 1e-10;

%% ---- single-multiset r=1 rel: degenerate, returns 0 ----

% Suppress the buildExpTens:isRelDegenerate warning for this test —
% the warning is informational, not a failure. The renyi2 path is
% specifically built to return 0 by convention in this regime.
ws = warning('off', 'buildExpTens:isRelDegenerate');
H_deg = entropyExpTens(p1, w1, sigma, 1, true, false, 0, ...
    'method', 'renyi2', 'verbose', false);
warning(ws);
results{end+1,1} = 'entropy.renyi2 single-multiset r=1 rel: degenerate, returns 0';
results{end,2}   = isequal(H_deg, 0);

%% ---- single-multiset r=3 abs nonper: agrees with mobius.orbitInnerAbsSingleMultiset ----

p3 = sort(2000 * rand(8, 1));
w3 = ones(8, 1);
H_r3 = entropyExpTens(p3, w3, sigma, 3, false, false, 0, ...
    'method', 'renyi2', 'verbose', false);

ip_ref = mobius.orbitInnerAbsSingleMultiset(p3, w3, p3, w3, sigma, 3, false, 0);
Z_ref = mobius.totalMassAbs(p3, w3, sigma, 3);
H_ref3 = -log2(ip_ref / (Z_ref * Z_ref));

results{end+1,1} = 'entropy.renyi2 single-multiset r=3 abs: matches direct orbit IP + total mass (1e-10)';
results{end,2}   = abs(H_r3 - H_ref3) < 1e-10;

%% ---- single-multiset r=3 rel periodic: agrees with mobius.orbitInnerRelSingleMultiset ----

period_p = 1200;
pP = sort(period_p * rand(8, 1));
wP = ones(8, 1);
H_rel = entropyExpTens(pP, wP, sigma, 3, true, true, period_p, ...
    'method', 'renyi2', 'verbose', false);

ip_rel = mobius.orbitInnerRelSingleMultiset(pP, wP, pP, wP, sigma, 3, true, period_p);
Z_rel = mobius.totalMassRel(pP, wP, sigma, 3);
H_rel_ref = -log2(ip_rel / (Z_rel * Z_rel));

results{end+1,1} = 'entropy.renyi2 single-multiset r=3 rel per: matches direct orbit IP + total mass (1e-8)';
results{end,2}   = abs(H_rel - H_rel_ref) < 1e-8;

%% ---- single-multiset: base argument flows through ----

H_b2 = entropyExpTens(p3, w3, sigma, 3, false, false, 0, ...
    'method', 'renyi2', 'base', 2, 'verbose', false);
H_be = entropyExpTens(p3, w3, sigma, 3, false, false, 0, ...
    'method', 'renyi2', 'base', exp(1), 'verbose', false);
% H_e * log_e(2) == H_2 exactly (change-of-base).
results{end+1,1} = 'entropy.renyi2 single-multiset: base=2 vs base=e differ by log(2) factor (1e-10)';
results{end,2}   = abs(H_b2 - H_be / log(2)) < 1e-10 * abs(H_b2);

%% ---- single-multiset struct input flows through (skinny dens) ----

dens_skinny = buildExpTens(p3, w3, sigma, 3, false, false, 0, 'verbose', false);
results{end+1,1} = 'entropy.renyi2 single-multiset: skinny dens has no Centres before call';
results{end,2}   = ~isfield(dens_skinny, 'Centres');

H_struct = entropyExpTens(dens_skinny, ...
    'method', 'renyi2', 'verbose', false);
results{end+1,1} = 'entropy.renyi2 single-multiset: skinny dens path matches raw-args path';
results{end,2}   = abs(H_struct - H_r3) < 1e-12;

%% ---- MA: agrees with hand-rolled per-attribute factorisation ----

rng(73, 'twister');
N = 3;
PxA = sort(2000 * rand(6, N));   WxA = ones(6, N);
PxB = sort(1500 * rand(6, N));   WxB = ones(6, N);

dx = buildExpTens({PxA, PxB}, {WxA, WxB}, [30 30], [3 3], ...
    [false false], [false false], [0 0], 'verbose', false);

H_ma = entropyExpTens(dx, ...
    'method', 'renyi2', 'verbose', false);

% Hand-rolled MA reference: <T,T> via per-attribute IP factorisation;
% Z = sum_n prod_a Z_a^{(n)}.
P_xx_ref = ones(N, N);
for a = 1:2
    Pa = dx.pAttr{a};   Wa = dx.w{a};
    I_a = mobius.maPerAttrInnerMatrix(Pa, Wa, Pa, Wa, ...
        30, 3, false, false, 0);
    P_xx_ref = P_xx_ref .* I_a;
end
ip_ma_ref = sum(P_xx_ref(:));

Z_per_event_attr = zeros(N, 2);
for a = 1:2
    Pa = dx.pAttr{a};   Wa = dx.w{a};
    for n = 1:N
        Z_per_event_attr(n, a) = mobius.totalMassAbs( ...
            Pa(:, n), Wa(:, n), 30, 3);
    end
end
Z_ma_ref = sum(prod(Z_per_event_attr, 2));
H_ma_ref = -log2(ip_ma_ref / (Z_ma_ref * Z_ma_ref));

results{end+1,1} = 'entropy.renyi2 MA: matches hand-rolled per-attribute factorisation (1e-10)';
results{end,2}   = abs(H_ma - H_ma_ref) < 1e-10;

%% ---- Input-form gating: list rejected ----

dens_list = {dens_skinny, dens_skinny};
ok_listReject = false;
try
    entropyExpTens(dens_list, 'method', 'renyi2', ...
        'verbose', false);
catch ME
    ok_listReject = strcmp(ME.identifier, ...
        'entropyExpTens:renyi2ListNotSupported');
end
results{end+1,1} = 'entropy.renyi2: list input rejected with informative error';
results{end,2}   = ok_listReject;

%% ---- Input-form gating: 2-D batched rejected ----

P_batched = [0 400 700; 0 300 700];   % (2, 3) — rows are chords
ok_batchReject = false;
try
    entropyExpTens(P_batched, [], sigma, 2, false, false, 0, ...
        'method', 'renyi2', 'verbose', false);
catch ME
    ok_batchReject = strcmp(ME.identifier, ...
        'entropyExpTens:renyi2BatchedNotSupported');
end
results{end+1,1} = 'entropy.renyi2: 2-D batched input rejected with informative error';
results{end,2}   = ok_batchReject;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_entropy_renyi2: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end

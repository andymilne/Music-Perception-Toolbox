%% test_kernel_truncation.m — v2.2 internal.gaussianKernelSum and mptDefaults
%
%  Mirrors python/tests/test_kernel_truncation.py.
%
%  Covers:
%    - Exact path (truncationSigmas=Inf) is bit-identical to a direct
%      broadcast-subtract-exp-sum reference.
%    - Truncated path agrees with exact to better than exp(-k^2/2)
%      relative error.
%    - Rel-mode quadratic form correctness (with and without truncation).
%    - Periodic 1-D abs truncates on the circle when the window fits
%      inside it (2*truncationSigmas*sigma < period).
%    - Single precision degrades to ~1e-7 relative.
%    - Defaults machinery: factory values, set/get/reset, restore via
%      prev struct, per-call overrides global.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone_kt = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults_kt
    cleanupDefaults_kt = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone_kt = false;
end

% Reset defaults before testing.
mptDefaults('reset');
% reset now yields the factory truncationSigmas = 6; pin Inf so the
% exact-path comparisons below are untruncated.
mptDefaults('truncationSigmas', Inf);
% Inf now resolves to the 1e-12 accuracy-floor width (~7.43 sigma), not
% literally exhaustive summation. The bit-parity assertions below
% ("exact matches reference", "Inf is exact") need genuinely exact
% behaviour, so widen the floor to 1e-300 (~37 sigma, effectively
% exhaustive) for the duration of this test. This is the golden-value
% regeneration override; it is restored explicitly at the end of the
% file (and via onCleanup if the script exits early).
kt_prevEps = internal.accuracyFloor('setEps', 1e-300);
kt_epsCleanup = onCleanup(@() internal.accuracyFloor('setEps', kt_prevEps)); %#ok<NASGU>

% Test data
rng(0, 'twister');
kt_dim = 2; kt_nJ = 60; kt_nQ = 8;
kt_C = 100 * rand(kt_dim, kt_nJ);
kt_wJ = 0.5 + rand(kt_nJ, 1);
kt_X = 100 * rand(kt_dim, kt_nQ);
kt_sigma = 5.0;

ref_abs = local_ref_kernel_sum(kt_C, kt_wJ, kt_X, kt_sigma, false, 0, false, 0);
ref_rel = local_ref_kernel_sum(kt_C, kt_wJ, kt_X, kt_sigma, true, 3, false, 0);

% Exact path matches reference
v = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma);
results{end+1, 1} = 'kernel: exact matches reference (abs)';
results{end, 2} = max(abs(v - ref_abs)) < 1e-12 * max(abs(ref_abs));

v = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
    'isRel', true, 'r', 3);
results{end+1, 1} = 'kernel: exact matches reference (rel, r=3)';
results{end, 2} = max(abs(v - ref_rel)) < 1e-12 * max(abs(ref_rel));

% Truncated path across k values
for k = [4, 5, 6, 8]
    % Per-centre kernel bound exp(-k^2/2); cumulative bound scales
    % with sum(|wJ|) ~ 60. Use 200x for safety.
    bound = 200 * exp(-k^2 / 2);

    v_trunc = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
        'truncationSigmas', k);
    rel = max(abs(v_trunc - ref_abs) ./ max(abs(ref_abs), 1e-30));
    results{end+1, 1} = sprintf('kernel: truncated abs within bound k=%d', k); %#ok<*AGROW>
    results{end, 2} = rel < bound;

    v_trunc = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
        'isRel', true, 'r', 3, 'truncationSigmas', k);
    rel = max(abs(v_trunc - ref_rel) ./ max(abs(ref_rel), 1e-30));
    results{end+1, 1} = sprintf('kernel: truncated rel within bound k=%d', k);
    results{end, 2} = rel < bound;
end

% truncationSigmas=Inf is bit-identical to exact
v_inf = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
    'truncationSigmas', Inf);
results{end+1, 1} = 'kernel: truncationSigmas=Inf is exact';
results{end, 2} = max(abs(v_inf - ref_abs)) < 1e-12 * max(abs(ref_abs));

% Periodic 1-D abs truncates on the circle when the window fits inside it
% (2*truncationSigmas*sigma < period). At truncationSigmas=6 the truncated
% sum sits at the ~exp(-18) floor below the exact periodic sum; at Inf
% (resolved to the accuracy floor) it sits at the ~exp(-27.6) floor, i.e.
% effectively exact.
rng(7, 'twister');
kt_Cp = 1200 * rand(1, 40);
kt_wp = 0.5 + rand(40, 1);
kt_Xp = 1200 * rand(1, 5);
kt_sigp = 30.0;
ref = local_ref_kernel_sum(kt_Cp, kt_wp, kt_Xp, kt_sigp, false, 0, true, 1200);
peak = max(abs(ref));
v6 = internal.gaussianKernelSum(kt_Cp, kt_wp, kt_Xp, kt_sigp, ...
    'truncationSigmas', 6, 'isPer', true, 'period', 1200);
err6 = max(abs(v6 - ref));
results{end+1, 1} = 'kernel: periodic 1-D truncates on the circle at 6 sigma';
results{end, 2} = err6 > 1e-11 * peak && err6 < 1e-6 * peak;
vInf = internal.gaussianKernelSum(kt_Cp, kt_wp, kt_Xp, kt_sigp, ...
    'truncationSigmas', Inf, 'isPer', true, 'period', 1200);
errInf = max(abs(vInf - ref));
results{end+1, 1} = 'kernel: periodic 1-D at Inf is effectively exact';
results{end, 2} = errInf < 1e-11 * peak;

% Single precision
v_single = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
    'kernelPrecision', 'single');
rel = max(abs(v_single - ref_abs) ./ max(abs(ref_abs), 1e-30));
results{end+1, 1} = 'kernel: single precision within bound';
results{end, 2} = rel < 1e-5;

results{end+1, 1} = 'kernel: single precision returns double output';
results{end, 2} = isa(v_single, 'double');

% Edge cases
v_empty = internal.gaussianKernelSum([0, 1], [1; 1], zeros(1, 0), 1.0, ...
    'truncationSigmas', 6);
results{end+1, 1} = 'kernel: empty queries';
results{end, 2} = isequal(size(v_empty), [1, 0]);

v1 = internal.gaussianKernelSum(5, 2, [5, 5.5, 10], 1.0, ...
    'truncationSigmas', 6);
ref1 = local_ref_kernel_sum(5, 2, [5, 5.5, 10], 1.0, false, 0, false, 0);
results{end+1, 1} = 'kernel: single centre';
results{end, 2} = max(abs(v1 - ref1)) < 1e-12 * max(max(abs(ref1)), 1e-30);

v_far = internal.gaussianKernelSum([0, 1, 2], [1; 1; 1], 100, 1.0, ...
    'truncationSigmas', 6);
results{end+1, 1} = 'kernel: query outside centres bbox';
results{end, 2} = v_far < 1e-30;

% Bad inputs raise
ok_err = false;
try
    internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
        'truncationSigmas', -1);
catch
    ok_err = true;
end
results{end+1, 1} = 'kernel: bad truncationSigmas raises';
results{end, 2} = ok_err;

ok_err = false;
try
    internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
        'kernelPrecision', 'quad');
catch
    ok_err = true;
end
results{end+1, 1} = 'kernel: bad precision raises';
results{end, 2} = ok_err;

ok_err = false;
try
    internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, 'isRel', true);
catch
    ok_err = true;
end
results{end+1, 1} = 'kernel: rel without r raises';
results{end, 2} = ok_err;

% Defaults machinery
mptDefaults('reset');
d = mptDefaults();
results{end+1, 1} = 'defaults: factory values';
results{end, 2} = d.truncationSigmas == 6 && strcmp(d.kernelPrecision, 'double');

mptDefaults('truncationSigmas', 6);
results{end+1, 1} = 'defaults: set and get';
results{end, 2} = mptDefaults('truncationSigmas') == 6 ...
    && strcmp(mptDefaults('kernelPrecision'), 'double');

mptDefaults('reset');
mptDefaults('truncationSigmas', 4);
prev = mptDefaults('truncationSigmas', 8, 'kernelPrecision', 'single');
results{end+1, 1} = 'defaults: set returns previous';
results{end, 2} = prev.truncationSigmas == 4 ...
    && strcmp(prev.kernelPrecision, 'double');

mptDefaults(prev);
results{end+1, 1} = 'defaults: restore from prev struct';
results{end, 2} = mptDefaults('truncationSigmas') == 4 ...
    && strcmp(mptDefaults('kernelPrecision'), 'double');

mptDefaults('reset');
d = mptDefaults();
results{end+1, 1} = 'defaults: reset to factory';
results{end, 2} = d.truncationSigmas == 6 && strcmp(d.kernelPrecision, 'double');

ok_err = false;
try
    mptDefaults('nonsense');
catch
    ok_err = true;
end
results{end+1, 1} = 'defaults: unknown name raises (get)';
results{end, 2} = ok_err;

ok_err = false;
try
    mptDefaults('nonsense', 42);
catch
    ok_err = true;
end
results{end+1, 1} = 'defaults: unknown name raises (set)';
results{end, 2} = ok_err;

ok_err = false;
try
    mptDefaults('truncationSigmas', 0);
catch
    ok_err = true;
end
results{end+1, 1} = 'defaults: zero truncationSigmas raises';
results{end, 2} = ok_err;

ok_err = false;
try
    mptDefaults('kernelPrecision', 'quad');
catch
    ok_err = true;
end
results{end+1, 1} = 'defaults: bad precision raises';
results{end, 2} = ok_err;

% Helper consults defaults; per-call overrides
mptDefaults('truncationSigmas', Inf);
v1 = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma);
results{end+1, 1} = 'kernel: uses global truncationSigmas=Inf (exact)';
results{end, 2} = max(abs(v1 - ref_abs)) < 1e-12 * max(abs(ref_abs));

mptDefaults('truncationSigmas', 6);
v2 = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma);
rel = max(abs(v2 - ref_abs) ./ max(abs(ref_abs), 1e-30));
results{end+1, 1} = 'kernel: uses default truncationSigmas=6';
results{end, 2} = rel < 200 * exp(-18);

mptDefaults('truncationSigmas', 4);
v3 = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma, ...
    'truncationSigmas', Inf);
results{end+1, 1} = 'kernel: per-call overrides default';
results{end, 2} = max(abs(v3 - ref_abs)) < 1e-12 * max(abs(ref_abs));

% ---------------------------------------------------------------------
% Regression: the 1-D truncated path used to fail with a 100 GB
% allocation at maxWin == 1 because MATLAB's "vector source + vector
% index" rule returned a result matching the source orientation rather
% than the index shape. With cSorted as a row and idxClipped a
% (nQ, 1) column, cSorted(idxClipped) came out (1, nQ), and the
% subsequent xAxis(:) - pSlices outer-broadcast to (nQ, nQ).
%
% These two tests pin down the fix:
%   (a) sigma chosen so windows contain at most 1 source — maxWin=1
%       (would OOM pre-fix)
%   (b) the demo's exact failing scenario (sigma_eff = 10/sqrt(2), 
%       mean-of-pairs queries from a 481-point grid)
% ---------------------------------------------------------------------
mptDefaults('reset');

% (a) maxWin == 1 case.
nQ_big = 120000;
C_big = [0, 200, 400, 500, 700, 900, 1100];   % min source gap = 100
wJ_big = ones(7, 1);
sigma_w1 = 10;   % threshold = 60 < 100, so windows contain <=1 centre
X_big = linspace(0, 1200, nQ_big);
ref_w1 = local_ref_kernel_sum(C_big, wJ_big, X_big, sigma_w1, ...
    false, 0, false, 0);
v_w1 = internal.gaussianKernelSum(C_big, wJ_big, X_big, sigma_w1, ...
    'truncationSigmas', 6);
rel_w1 = max(abs(v_w1 - ref_w1)) ./ max(max(abs(ref_w1)), eps);
results{end+1, 1} = 'kernel: truncated path correct at maxWin == 1';
results{end, 2} = rel_w1 < 200 * exp(-18);

% (b) demo_expTensorPlots config 3, the exact failing inputs.
sigma_eff = 10 / sqrt(2);
res = 481;
x_1d = linspace(0, 1200, res);
[Ga, Gb] = meshgrid(x_1d, x_1d);
upperMask = triu(true(res));
Xu = [Ga(upperMask)'; Gb(upperMask)'];
mean_x = sum(Xu, 1) / 2;
ref_demo = local_ref_kernel_sum(C_big, wJ_big, mean_x, sigma_eff, ...
    false, 0, false, 0);
v_demo = internal.gaussianKernelSum(C_big, wJ_big, mean_x, sigma_eff, ...
    'truncationSigmas', 6);
rel_demo = max(abs(v_demo - ref_demo)) ./ max(max(abs(ref_demo)), eps);
results{end+1, 1} = 'kernel: demo_expTensorPlots config 3 inputs';
results{end, 2} = rel_demo < 200 * exp(-18);

% Final reset so the test doesn't leak state.
mptDefaults('reset');

if standalone_kt
    nTests_kt = size(results, 1);
    nPass_kt = sum([results{:, 2}]);
    fprintf('  test_kernel_truncation: %d/%d passed\n', nPass_kt, nTests_kt);
    if nPass_kt < nTests_kt
        for ii = 1:nTests_kt
            if ~results{ii, 2}
                fprintf('    FAIL: %s\n', results{ii, 1});
            end
        end
    end
end

% Local function — at the bottom of a script (R2016b+).
function v = local_ref_kernel_sum(C, wJ, X, sigma, isRel, r, isPer, period)
    C = C(:, :);
    X = X(:, :);
    wJ = wJ(:);
    dim = size(C, 1);
    nJ  = size(C, 2);
    nQ  = size(X, 2);
    D = reshape(C, dim, nJ, 1) - reshape(X, dim, 1, nQ);
    if isPer
        D = D - period * floor(D / period + 0.5);
    end
    if isRel
        Q = sum(D .* D, 1) - sum(D, 1).^2 / r;
    else
        Q = sum(D .* D, 1);
    end
    E = reshape(exp(-Q(:) / (2 * sigma^2)), nJ, nQ);
    v = wJ' * E;
end

% Restore the accuracy-floor override (paired with the setEps at the
% top). Explicit clear fires the onCleanup restore deterministically,
% so the widened floor cannot leak into subsequent test files.
if exist('kt_epsCleanup', 'var')
    clear kt_epsCleanup
end

% Restore caller's pre-test defaults eagerly when run
% standalone (fires the helper's onCleanup destructor on
% script exit; guarded so we don't clear a like-named
% variable when this file was run from test_mpt.m, where
% the standalone branch was skipped).
if exist('cleanupDefaults_kt', 'var')
    clear cleanupDefaults_kt
end

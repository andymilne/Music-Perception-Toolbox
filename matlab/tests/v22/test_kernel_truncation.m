%% test_kernel_truncation.m — v2.2 internal.gaussianKernelSum and mptDefaults
%
%  Mirrors python/tests/v22/test_kernel_truncation.py.
%
%  Covers:
%    - Exact path (truncationSigmas=Inf) is bit-identical to a direct
%      broadcast-subtract-exp-sum reference.
%    - Truncated path agrees with exact to better than exp(-k^2/2)
%      relative error.
%    - Rel-mode quadratic form correctness (with and without truncation).
%    - Periodic mode falls through to exact regardless of truncationSigmas.
%    - Single precision degrades to ~1e-7 relative.
%    - Defaults machinery: factory values, set/get/reset, restore via
%      prev struct, per-call overrides global.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone_kt = true;
else
    standalone_kt = false;
end

% Reset defaults before testing.
mptDefaults('reset');

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

% Periodic mode falls through to exact
rng(7, 'twister');
kt_Cp = 1200 * rand(1, 40);
kt_wp = 0.5 + rand(40, 1);
kt_Xp = 1200 * rand(1, 5);
kt_sigp = 30.0;
v = internal.gaussianKernelSum(kt_Cp, kt_wp, kt_Xp, kt_sigp, ...
    'truncationSigmas', 6, 'isPer', true, 'period', 1200);
ref = local_ref_kernel_sum(kt_Cp, kt_wp, kt_Xp, kt_sigp, false, 0, true, 1200);
results{end+1, 1} = 'kernel: periodic falls through to exact';
results{end, 2} = max(abs(v - ref)) < 1e-12 * max(abs(ref));

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
results{end, 2} = isinf(d.truncationSigmas) && strcmp(d.kernelPrecision, 'double');

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
results{end, 2} = isinf(d.truncationSigmas) && strcmp(d.kernelPrecision, 'double');

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
mptDefaults('reset');
v1 = internal.gaussianKernelSum(kt_C, kt_wJ, kt_X, kt_sigma);
results{end+1, 1} = 'kernel: uses default truncationSigmas=Inf';
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

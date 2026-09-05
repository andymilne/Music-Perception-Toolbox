%% test_eval_routing.m — v3 evalExpTens centres-path routing parity
%
%  Mirrors python/tests/test_eval_routing.py. Verifies that the
%  v3 refactor of localEvalSingleMultisetCentres (routing through
%  internal.gaussianKernelSum) is:
%
%    - Within the accuracy-floor truncation bound of the v2.0
%      untruncated reference at default settings (Inf resolves to the
%      ~7.43 sigma floor; kernelPrecision='double').
%    - Within bounded relative error at finite truncationSigmas.
%    - Within ~1e-5 relative at kernelPrecision='single'.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to.

if ~exist('results', 'var')
    results = {};
    standalone_er = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults_er
    cleanupDefaults_er = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone_er = false;
end

mptDefaults('reset');
% reset now yields the factory truncationSigmas = 6; pin Inf so the
% "default matches reference" comparisons below exercise the accuracy-
% floor width (Inf resolves to ~7.43 sigma), checked within the
% truncation bound against the untruncated reference.
mptDefaults('truncationSigmas', Inf);

% --- Battery of representative cases ---
% Each row: {label, K, r, isRel, isPer, period, sigma, nQ}
er_cases = { ...
    'abs r=2 K=6',     6, 2, false, false, 0,    8,  30; ...
    'rel r=2 K=6',     6, 2, true,  false, 0,    8,  30; ...
    'abs r=3 K=5',     5, 3, false, false, 0,   12,  20; ...
    'rel r=3 K=5',     5, 3, true,  false, 0,   12,  20; ...
    'rel r=3 K=8 s=30',8, 3, true,  false, 0,   30,  50; ...
    'rel-per r=3 K=6', 6, 3, true,  true,  1200,25,  25; ...
    'abs-per r=2 K=6', 6, 2, false, true,  1200,25,  25; ...
};

for ic = 1:size(er_cases, 1)
    label  = er_cases{ic, 1};
    K      = er_cases{ic, 2};
    r      = er_cases{ic, 3};
    isRel  = er_cases{ic, 4};
    isPer  = er_cases{ic, 5};
    period = er_cases{ic, 6};
    sigma  = er_cases{ic, 7};
    nQ     = er_cases{ic, 8};

    % Deterministic seed per case (label hash)
    rng(sum(double(label)), 'twister');
    p = sort(1000 * rand(K, 1));
    w = 0.5 + rand(K, 1);
    dens = buildExpTens(p, w, sigma, r, isRel, isPer, period, ...
        'verbose', false);
    % Populate heavy fields (Centres, wJ) so the test body can read
    % them. Production code calls internal.ensureExpTensExpensive inside the
    % evalExpTens centres-branch before invoking localEvalSingleMultisetCentres.
    dens = internal.ensureExpTensExpensive(dens);
    if isRel
        dim = r - 1;
    else
        dim = r;
    end
    if isPer
        X = period * rand(dim, nQ);
    else
        X = 1000 * rand(dim, nQ);
    end

    % --- Reference: frozen v2.0 body ---
    ref = local_ref_eval(dens, X);

    % --- Default / explicit Inf: the reference sums untruncated, but Inf
    %     now resolves to the accuracy-floor width (~7.43 sigma), so eval
    %     truncates there. Check against the reference within the same
    %     truncation bound the finite-k cases use below, at the resolved
    %     floor width. Reduction-order differences (e.g. the periodic wrap
    %     retargeted from `mod` to `D - period .* floor(D/period + 0.5)`,
    %     mathematically equivalent but a different FP-op sequence) sit far
    %     below this bound.
    kFloor      = internal.accuracyFloor('resolve', Inf);
    weight_mass = sum(abs(dens.wJ));
    floorBound  = max(1e-12, 10 * weight_mass * exp(-kFloor^2 / 2));

    v_default = evalExpTens(dens, X, 'method', 'centres', 'verbose', false);
    results{end+1, 1} = sprintf('eval_routing: default matches reference (%s)', label); %#ok<*AGROW>
    results{end, 2} = max(abs(v_default(:) - ref(:))) < floorBound;

    % --- Explicit Inf/double: same floor-width truncation bound ---
    v_inf = evalExpTens(dens, X, 'method', 'centres', ...
        'truncationSigmas', Inf, 'kernelPrecision', 'double', ...
        'verbose', false);
    results{end+1, 1} = sprintf('eval_routing: Inf/double matches reference (%s)', label);
    results{end, 2} = max(abs(v_inf(:) - ref(:))) < floorBound;

    % --- truncationSigmas at k=4..6 within cumulative bound ---
    % Non-periodic only: under the truncation contract, periodic mode
    % also truncates (inf resolves to the accuracy-floor width), so the
    % former "periodic ignores truncation" check is obsolete and removed.
    if ~isPer
        for k = [4, 5, 6]
            v_trunc = evalExpTens(dens, X, 'method', 'centres', ...
                'truncationSigmas', k, 'verbose', false);
            weight_mass = sum(abs(dens.wJ));
            bound = max(1e-12, 10 * weight_mass * exp(-k^2 / 2));
            err = max(abs(v_trunc - ref));
            results{end+1, 1} = sprintf(...
                'eval_routing: truncation k=%d within bound (%s)', k, label);
            results{end, 2} = err < bound;
        end
    end

    % --- Single precision within bound ---
    v_single = evalExpTens(dens, X, 'method', 'centres', ...
        'kernelPrecision', 'single', 'verbose', false);
    weight_mass = sum(abs(dens.wJ));
    bound_single = max(1e-12, 1e-5 * weight_mass);
    err_single = max(abs(v_single - ref));
    results{end+1, 1} = sprintf(...
        'eval_routing: single precision within bound (%s)', label);
    results{end, 2} = err_single < bound_single;
end

% --- Global default picked up; per-call overrides ---
rng(42, 'twister');
p = sort(1000 * rand(6, 1));
w = 0.5 + rand(6, 1);
sigma = 12;
dens = buildExpTens(p, w, sigma, 3, true, false, 0, 'verbose', false);
dens = internal.ensureExpTensExpensive(dens);
X = 1000 * rand(2, 20);
ref = local_ref_eval(dens, X);

% Default Inf: matches reference (non-periodic, so still bit-identical
% in practice, but use the same tolerance contract for consistency)
v1 = evalExpTens(dens, X, 'method', 'centres', 'verbose', false);
results{end+1, 1} = 'eval_routing: default Inf matches reference (global)';
results{end, 2} = local_within_rtol(v1, ref, 1e-12);

% Set global default = 6
mptDefaults('truncationSigmas', 6);
v2 = evalExpTens(dens, X, 'method', 'centres', 'verbose', false);
weight_mass = sum(abs(dens.wJ));
bound = max(1e-12, 10 * weight_mass * exp(-18));
results{end+1, 1} = 'eval_routing: global default 6 within bound';
results{end, 2} = max(abs(v2 - ref)) < bound;

% Per-call Inf overrides global=4
mptDefaults('truncationSigmas', 4);
v3 = evalExpTens(dens, X, 'method', 'centres', ...
    'truncationSigmas', Inf, 'verbose', false);
results{end+1, 1} = 'eval_routing: per-call overrides global';
results{end, 2} = local_within_rtol(v3, ref, 1e-12);

mptDefaults('reset');

if standalone_er
    nTests_er = size(results, 1);
    nPass_er = sum([results{:, 2}]);
    fprintf('  test_eval_routing: %d/%d passed\n', nPass_er, nTests_er);
    if nPass_er < nTests_er
        for ii = 1:nTests_er
            if ~results{ii, 2}
                fprintf('    FAIL: %s\n', results{ii, 1});
            end
        end
    end
end


function tf = local_within_rtol(actual, ref, rtol)
%LOCAL_WITHIN_RTOL  True iff max relative error <= rtol.
%   Mirrors numpy.testing.assert_allclose(actual, ref, rtol=rtol, atol=0).
%   Returns true for empty/all-zero references (vacuously).
    err = max(abs(actual(:) - ref(:)));
    denom = max(abs(ref(:)));
    if denom == 0
        tf = (err == 0);
    else
        tf = (err / denom) <= rtol;
    end
end


function v = local_ref_eval(dens, X)
%LOCAL_REF_EVAL  Frozen v2.0 evalFull body for parity reference.
%   Assumes `dens` already has its heavy fields materialised (caller
%   has invoked internal.ensureExpTensExpensive). The frozen body reads
%   the flat single-multiset layout, so present the density through the
%   view (single-multiset densities are the A = N = 1 corner of a
%   MaetDensity).
    dens = internal.singleMultisetView(dens);
    Centres = dens.Centres;
    wJ      = dens.wJ;
    sigma   = dens.sigma;
    r       = dens.r;
    dim     = dens.dim;
    isRel   = dens.isRel;
    isPer   = dens.isPer;
    J       = dens.period;
    nJ      = dens.nJ;
    nQ      = size(X, 2);

    D = reshape(Centres, dim, nJ, 1) - reshape(X, dim, 1, nQ);
    if isPer
        D = mod(D + J/2, J) - J/2;
    end
    if isRel
        Qvec = sum(D.^2, 1) - sum(D, 1).^2 / r;
    else
        Qvec = sum(D.^2, 1);
    end
    E = reshape(exp(-Qvec(:) / (2 * sigma^2)), nJ, nQ);
    v = wJ(:)' * E;
end

% Restore caller's pre-test defaults eagerly when run
% standalone (fires the helper's onCleanup destructor on
% script exit; guarded so we don't clear a like-named
% variable when this file was run from test_mpt.m, where
% the standalone branch was skipped).
if exist('cleanupDefaults_er', 'var')
    clear cleanupDefaults_er
end

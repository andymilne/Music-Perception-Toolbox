%% test_eval_ma_orbit.m — factored MA Möbius evaluator (mobius.evalMaOrbit)
%
%  Verifies the factored evaluator f(x) = sum_n prod_a [per-attribute SA
%  density](x_a) against two references:
%    1. the existing joint-centres MA path (evalExpTens 'method','centres'),
%       for internal consistency across abs/rel/mixed/periodic and N>1;
%    2. Python centres-path values (tests/ma_eval_parity.json), for
%       cross-language parity.
%  Both to ~1e-7 (relative-mode quadrature noise); absolute-only configs
%  agree far tighter.
%
%  Twin coverage of python/tests/test_ma_orbit.py.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_eo
    cleanupDefaults_eo = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% Exact evaluation for the parity comparisons (Inf -> effectively exhaustive).
eo_prevEps = internal.accuracyFloor('setEps', 1e-300);

%% ---- Internal consistency: factored vs joint-centres, across modes ----

cfgs = {
  % label, sigma, r, isRel, isPer, period, N
  {'A2 abs r2',    [6 5],     [2 2],   [false false],       [false false],      [0 0],       1}
  {'A2 rel r2',    [6 5],     [2 2],   [true true],         [false false],      [0 0],       1}
  {'A2 mixed',     [6 5],     [2 3],   [true false],        [false false],      [0 0],       1}
  {'A2 rel N2',    [6 5],     [2 2],   [true true],         [false false],      [0 0],       2}
  {'A3 mixed',     [6 5 30],  [2 3 2], [true false true],   [false false true], [0 0 1200],  1}
};

rng(42, 'twister');
for ci = 1:numel(cfgs)
    c = cfgs{ci};
    [label, sig, rv, rel, per, P, N] = c{:};
    A = numel(sig); K = 6;
    pas = cell(A, 1);
    for a = 1:A
        pas{a} = 100 * rand(K, N);
    end
    wpas = repmat({[]}, A, 1);
    dens = buildExpTens(pas, wpas, sig, rv, rel, per, P, 'verbose', false);
    xq = 100 * rand(dens.dim, 8);

    vCentres = evalExpTens(dens, xq, 'method', 'centres', 'verbose', false);
    vFactored = mobius.evalMaOrbit(dens, xq);

    denom = max(max(abs(vCentres)), 1e-12);
    relErr = max(abs(vCentres(:) - vFactored(:))) / denom;
    results{end+1, 1} = sprintf('evalMaOrbit %s: factored matches centres (1e-6)', label); %#ok<*AGROW>
    results{end, 2} = relErr < 1e-6;
end

%% ---- Cross-language parity: factored vs Python centres values ----

jsonPath = fullfile(fileparts(mfilename('fullpath')), 'ma_eval_parity.json');
if exist(jsonPath, 'file')
    raw = jsondecode(fileread(jsonPath));
    for ci = 1:numel(raw)
        cc = raw(ci);
        A = numel(cc.sig);
        % jsondecode gives pAttr as a cell (ragged) or numeric array;
        % normalise to a cell of (K, N) matrices.
        pas = cell(A, 1);
        for a = 1:A
            if iscell(cc.pAttr)
                pa = cc.pAttr{a};
            else
                pa = squeeze(cc.pAttr(a, :, :));
            end
            pas{a} = reshape(pa, cc.K, cc.N);
        end
        wpas = repmat({[]}, A, 1);
        dens = buildExpTens(pas, wpas, cc.sig(:)', cc.r(:)', ...
            logical(cc.rel(:)'), logical(cc.per(:)'), cc.P(:)', ...
            'verbose', false);
        xq = reshape(cc.x, dens.dim, []);
        vFactored = mobius.evalMaOrbit(dens, xq);
        vRef = cc.v(:);
        denom = max(max(abs(vRef)), 1e-12);
        relErr = max(abs(vFactored(:) - vRef)) / denom;
        results{end+1, 1} = sprintf('evalMaOrbit %s: matches Python centres (1e-6)', cc.label);
        results{end, 2} = relErr < 1e-6;
    end
else
    results{end+1, 1} = 'evalMaOrbit: Python parity JSON present';
    results{end, 2} = false;
end

%% ---- Cancellation ratio returned in [0, 1] ----

rng(7, 'twister');
pas = {100 * rand(6, 1); 100 * rand(6, 1)};
dens = buildExpTens(pas, {[]; []}, [6 5], [2 2], [true true], ...
    [false false], [0 0], 'verbose', false);
xq = 100 * rand(dens.dim, 5);
[~, rat] = mobius.evalMaOrbit(dens, xq, 'returnCancellationRatio', true);
results{end+1, 1} = 'evalMaOrbit: cancellation ratio in [0, 1]';
results{end, 2} = all(rat >= 0) && all(rat <= 1 + 1e-9);

% Restore the accuracy floor.
internal.accuracyFloor('setEps', eo_prevEps);

%% ---- Standalone reporting ----
if standalone
    nFail = 0;
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true), nFail = nFail + 1; end
    end
    fprintf('test_eval_ma_orbit: %d passed, %d failed (of %d)\n', ...
        size(results, 1) - nFail, nFail, size(results, 1));
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    clear eo_epsCleanup
    if exist('cleanupDefaults_eo', 'var')
        clear cleanupDefaults_eo
    end
end

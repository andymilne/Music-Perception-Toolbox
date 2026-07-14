%% test_ma_eval_dispatch.m — MA eval dispatch through public evalExpTens
%
%  Verifies the wired dispatch (internal.selectMaEval routing evalExpTens's
%  MA path between the factored mobius.evalMaOrbit and the joint-centres
%  accumulator):
%    - method 'centres', 'mobius', 'auto' agree (raw and pdf-normalised);
%    - cross-language parity against Python auto values;
%    - user overrides honoured.
%
%  Twin coverage of python tests/test_ma_orbit.py dispatch cases.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_md
    cleanupDefaults_md = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

md_prevEps = internal.accuracyFloor('setEps', 1e-300);

%% ---- Method agreement across modes (raw and pdf) ----
cfgs = {
  {'A2 abs r2',  [6 5],    [2 2],   [false false],     [false false],      [0 0],      1}
  {'A2 rel r2',  [6 5],    [2 2],   [true true],       [false false],      [0 0],      1}
  {'A2 mixed',   [6 5],    [2 3],   [true false],      [false false],      [0 0],      1}
  {'A3 mixed',   [6 5 30], [2 3 2], [true false true], [false false true], [0 0 1200], 1}
};
rng(77, 'twister');
for ci = 1:numel(cfgs)
    c = cfgs{ci};
    [label, sig, rv, rel, per, P, N] = c{:};
    A = numel(sig); K = 6;
    pas = cell(A, 1);
    for a = 1:A, pas{a} = 100 * rand(K, N); end
    wpas = repmat({[]}, A, 1);
    dens = buildExpTens(pas, wpas, sig, rv, rel, per, P, 'verbose', false);
    xq = 100 * rand(dens.dim, 8);

    for nz = {'none', 'pdf'}
        nzs = nz{1};
        vC = evalExpTens(dens, xq, nzs, 'method', 'centres', 'verbose', false);
        vM = evalExpTens(dens, xq, nzs, 'method', 'mobius',  'verbose', false);
        vA = evalExpTens(dens, xq, nzs, 'method', 'auto',    'verbose', false);
        denom = max(max(abs(vC)), 1e-12);
        okCM = max(abs(vC(:) - vM(:))) / denom < 1e-6;
        okA  = (max(abs(vA(:) - vM(:))) / denom < 1e-9) || ...
               (max(abs(vA(:) - vC(:))) / denom < 1e-9);
        results{end+1, 1} = sprintf('MA dispatch %s (%s): centres==mobius==auto', label, nzs); %#ok<*AGROW>
        results{end, 2} = okCM && okA;
    end
end

%% ---- Cross-language parity against Python auto values ----
jsonPath = fullfile(fileparts(mfilename('fullpath')), 'ma_dispatch_parity.json');
if exist(jsonPath, 'file')
    raw = jsondecode(fileread(jsonPath));
    for ci = 1:numel(raw)
        cc = raw(ci);
        A = numel(cc.sig);
        pas = cell(A, 1);
        for a = 1:A
            if iscell(cc.pAttr), pa = cc.pAttr{a}; else, pa = squeeze(cc.pAttr(a, :, :)); end
            pas{a} = reshape(pa, cc.K, cc.N);
        end
        wpas = repmat({[]}, A, 1);
        dens = buildExpTens(pas, wpas, cc.sig(:)', cc.r(:)', ...
            logical(cc.rel(:)'), logical(cc.per(:)'), cc.P(:)', 'verbose', false);
        xq = reshape(cc.x, dens.dim, []);
        vA = evalExpTens(dens, xq, cc.norm, 'method', 'auto', 'verbose', false);
        vRef = cc.v(:);
        denom = max(max(abs(vRef)), 1e-12);
        results{end+1, 1} = sprintf('MA dispatch %s: matches Python auto (1e-6)', cc.label);
        results{end, 2} = max(abs(vA(:) - vRef)) / denom < 1e-6;
    end
end

internal.accuracyFloor('setEps', md_prevEps);

%% ---- Standalone reporting ----
if standalone
    nFail = 0;
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true), nFail = nFail + 1; end
    end
    fprintf('test_ma_eval_dispatch: %d passed, %d failed (of %d)\n', ...
        size(results, 1) - nFail, nFail, size(results, 1));
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    if exist('cleanupDefaults_md', 'var'), clear cleanupDefaults_md; end
end

%% test_unequal_k_rel.m — relative-mode cosine between unequal-size collections
%
%  Regression for the batched relative orbit inner product
%  (mobius.relInnerBatched via mobius.orbitInnerRelSingleMultiset): it previously
%  assumed both collections had the same cardinality K, reshaping the
%  kernel to (pairs, K, K). Comparing a 3-note triad against a larger
%  scale (the demo_edoApprox workload: 4:5:6 triad vs EDO sets) crashed
%  in the reshape. The underlying orbit primitive
%  (mobius.innerProductOrbitGrid) accepts an (N_u, K_x, K_y) kernel with
%  separate weight vectors, so unequal sizes are well defined; the fix
%  threads K_x and K_y through relInnerBatched.
%
%  Verifies the orbit path (a) no longer errors and (b) matches the
%  single-wrap Bulger reference (Python, method='bulger') to quadrature
%  tolerance, for periodic and non-periodic relative mode.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_uk
    cleanupDefaults_uk = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

uk_prevEps = internal.accuracyFloor('setEps', 1e-300);

%% ---- Orbit path handles unequal K and matches Bulger (MATLAB) ----
triad = [0; 386; 702];
rng(0, 'twister');
for Kb = [5, 7, 12]
    scale = sort(1200 * rand(Kb, 1));
    for per = [true, false]
        P = 1200.0 * per;
        dx = buildExpTens(triad, ones(3, 1), 30.0, 2, true, per, P, 'verbose', false);
        dy = buildExpTens(scale, ones(Kb, 1), 30.0, 2, true, per, P, 'verbose', false);
        % Must not error, and orbit must match bulger.
        ok = false;
        try
            sM = cosSimExpTens(dx, dy, 'method', 'mobius', 'verbose', false);
            sB = cosSimExpTens(dx, dy, 'method', 'bulger', 'verbose', false);
            ok = isfinite(sM) && abs(sM - sB) < 1e-8;
        catch err
            fprintf('  unequal-K K3vsK%d per=%d errored: %s\n', Kb, per, err.message);
            ok = false;
        end
        results{end+1, 1} = sprintf('unequal-K rel K3 vs K%d per=%d: orbit==bulger', Kb, per); %#ok<*AGROW>
        results{end, 2} = ok;
    end
end

%% ---- Cross-language: orbit matches Python bulger references ----
jsonPath = fullfile(fileparts(mfilename('fullpath')), 'unequal_k_rel_parity.json');
if exist(jsonPath, 'file')
    raw = jsondecode(fileread(jsonPath));
    for ci = 1:numel(raw)
        cc = raw(ci);
        dx = buildExpTens(cc.triad(:), ones(numel(cc.triad), 1), 30.0, 2, ...
            true, logical(cc.per), cc.period, 'verbose', false);
        dy = buildExpTens(cc.scale(:), ones(cc.Kb, 1), 30.0, 2, ...
            true, logical(cc.per), cc.period, 'verbose', false);
        sM = cosSimExpTens(dx, dy, 'method', 'mobius', 'verbose', false);
        results{end+1, 1} = sprintf('unequal-K rel K3 vs K%d per=%d: matches Python (1e-6)', ...
            cc.Kb, cc.per);
        results{end, 2} = abs(sM - cc.s) < 1e-6;
    end
end

%% ---- Symmetry: cos(A,B) == cos(B,A) with sizes swapped ----
dxT = buildExpTens(triad, ones(3, 1), 30.0, 2, true, true, 1200, 'verbose', false);
scale7 = sort(1200 * rand(7, 1));
dyS = buildExpTens(scale7, ones(7, 1), 30.0, 2, true, true, 1200, 'verbose', false);
sAB = cosSimExpTens(dxT, dyS, 'method', 'mobius', 'verbose', false);
sBA = cosSimExpTens(dyS, dxT, 'method', 'mobius', 'verbose', false);
results{end+1, 1} = 'unequal-K rel: cos(A,B) == cos(B,A) (size order symmetric)';
results{end, 2} = abs(sAB - sBA) < 1e-9;

internal.accuracyFloor('setEps', uk_prevEps);

%% ---- Standalone reporting ----
if standalone
    nFail = 0;
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true), nFail = nFail + 1; end
    end
    fprintf('test_unequal_k_rel: %d passed, %d failed (of %d)\n', ...
        size(results, 1) - nFail, nFail, size(results, 1));
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    if exist('cleanupDefaults_uk', 'var'), clear cleanupDefaults_uk; end
end

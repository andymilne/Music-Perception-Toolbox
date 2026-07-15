%% test_sa_unified.m — single-collection densities served through the unified path
%
%  After the vector build returns a MaetDensity (A = N = 1), the
%  single-collection consumers (evalExpTens, cosSimExpTens,
%  entropyExpTens) recognise that corner by SHAPE (internal.isSaShaped)
%  and serve it through the specialized single-collection pipeline,
%  reading fields via internal.saView. There is one density type; the
%  single-collection case is a shape-gated specialization, mirroring
%  Python's is_sa_shaped / sa_view architecture.
%
%  Verifies eval, entropy, and cosine on vector-built densities match
%  Python references, and that a single-collection MaetDensity is
%  numerically indistinguishable from how it was served before the flip.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_su
    cleanupDefaults_su = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

su_prevEps = internal.accuracyFloor('setEps', 1e-300);

%% ---- Shape predicate recognises the vector build as single-collection ----
dV = buildExpTens([0;3;7;11;14;18], ones(6,1), 6.0, 2, false, false, 0, ...
    'verbose', false);
results{end+1,1} = 'isSaShaped: vector build is single-collection'; %#ok<*AGROW>
results{end,2} = internal.isSaShaped(dV) && strcmp(dV.tag, 'MaetDensity');

%% ---- Cross-language parity: eval / entropy / cosine ----
jsonPath = fullfile(fileparts(mfilename('fullpath')), 'sa_unified_parity.json');
if exist(jsonPath, 'file')
    raw = jsondecode(fileread(jsonPath));
    for ci = 1:numel(raw)
        cc = raw(ci);
        p = cc.p(:); w = ones(numel(p), 1);
        d = buildExpTens(p, w, cc.sig, cc.r, logical(cc.rel), ...
            logical(cc.per), cc.P, 'verbose', false);
        xq = reshape(cc.x, cc.r - cc.rel, []);

        % eval
        ev = evalExpTens(d, xq, 'verbose', false);
        evRef = cc.ev(:);
        okEv = max(abs(ev(:) - evRef)) / max(max(abs(evRef)), 1e-12) < 1e-6;
        results{end+1,1} = sprintf('unified eval %s matches Python', cc.label);
        results{end,2} = okEv;

        % entropy
        H = entropyExpTens(d, 'nPointsPerDim', 48, 'xMin', -30, 'xMax', 1230, ...
            'verbose', false);
        okH = abs(H - cc.H) < 1e-5;
        results{end+1,1} = sprintf('unified entropy %s matches Python', cc.label);
        results{end,2} = okH;

        % cosine vs shifted copy
        d2 = buildExpTens(p + 50, w, cc.sig, cc.r, logical(cc.rel), ...
            logical(cc.per), cc.P, 'verbose', false);
        cs = cosSimExpTens(d, d2, 'verbose', false);
        okCs = abs(cs - cc.cs) < 1e-6;
        results{end+1,1} = sprintf('unified cosine %s matches Python', cc.label);
        results{end,2} = okCs;
    end
end

%% ---- Self-cosine is 1 (single-collection through unified path) ----
dS = buildExpTens([0;4;7;12], ones(4,1), 30.0, 2, true, true, 1200, 'verbose', false);
results{end+1,1} = 'unified cosine: self-similarity == 1';
results{end,2} = abs(cosSimExpTens(dS, dS, 'verbose', false) - 1) < 1e-9;

internal.accuracyFloor('setEps', su_prevEps);

%% ---- Standalone reporting ----
if standalone
    nFail = 0;
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true), nFail = nFail + 1; end
    end
    fprintf('test_sa_unified: %d passed, %d failed (of %d)\n', ...
        size(results, 1) - nFail, nFail, size(results, 1));
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    if exist('cleanupDefaults_su', 'var'), clear cleanupDefaults_su; end
end

%% test_windowed_premaet.m — pre-MAET windowedSimilarity / windowedEntropy
%
%  Validates the pre-MAET windowedSimilarity and windowedEntropy against
%  the equivalent inline pipeline (weightEvents / translateAttributes /
%  buildExpTens / entropyExpTens / cosSimExpTens) they replace. Mirror of
%  Python's tests/test_windowed_premaet.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

tol   = 1e-9;
SD    = 1.0;
WW    = 2.0 * sqrt(3.0) * SD;          % variance-matched rectangular support
SIGP  = 0.12;
SIGT  = 0.05;

% Deterministic carrier (no RNG dependence across languages): a rising
% pitch line and a near-uniform onset grid.
N      = 24;
pitch  = (48 + (0:N-1)) ;                       % 1 x N rising pitches
onset  = cumsum(0.45 + 0.1 * mod((0:N-1), 3));  % 1 x N onsets
pAttr  = {pitch, onset};
centres = linspace(onset(1), onset(end), 9);

% query (a small triad template on pitch x time)
query  = {[60, 64, 67], [0, 0.5, 1.0]};
muQ    = mean(query{2});
qExt   = max(query{2}) - min(query{2});

% ----- 1. entropy, drop the time axis (differential & renyi2) -----
methodsShapes = {'differential', 0.0; 'renyi2', 0.0; 'renyi2', 1.0};
for k = 1:size(methodsShapes, 1)
    method = methodsShapes{k, 1};
    shape  = methodsShapes{k, 2};
    ref = zeros(1, numel(centres));
    for i = 1:numel(centres)
        if shape == 1.0
            [pw, ww] = weightEvents(pAttr, [], 2, 1, centres(i), shape, ...
                'width', WW, 'dropInputAttr', true);
        else
            [pw, ww] = weightEvents(pAttr, [], 2, 1, centres(i), shape, ...
                'sd', SD, 'dropInputAttr', true);
        end
        dens = buildExpTens(pw, ww, SIGP, 1, false, false, 0.0, 'verbose', false);
        ref(i) = entropyExpTens(dens, 'method', method, 'verbose', false);
    end
    got = windowedEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], centres, ...
        'contextWindow', {shape, WW}, 'method', method, ...
        'windowAttr', 2, 'dropWindowAttr', true, 'verbose', false);
    ok = max(abs(got(:) - ref(:))) < tol;
    results(end+1, :) = {sprintf('windowedEntropy drop window axis %s shape=%g', method, shape), ok}; %#ok<SAGROW>
end

% ----- 2. entropy, retain the time axis (joint pitch-time renyi2) --------
ref = zeros(1, numel(centres));
for i = 1:numel(centres)
    [pw, ww] = weightEvents(pAttr, [], 2, 1, centres(i), 1.0, ...
        'width', WW, 'dropInputAttr', false);
    dens = buildExpTens(pw, ww, [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], 'verbose', false);
    ref(i) = entropyExpTens(dens, 'method', 'renyi2', 'verbose', false);
end
got = windowedEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
    [false, false], [0.0, 0.0], centres, ...
    'contextWindow', {1.0, WW}, 'method', 'renyi2', 'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
results(end+1, :) = {'windowedEntropy joint (retain axis)', ...
    max(abs(got(:) - ref(:))) < tol}; %#ok<SAGROW>

% ----- 3. similarity, locked template sweep (oneSidedDenom & cosine) -----
for nm = {'oneSidedDenom', 'cosine'}
    normalize = nm{1};
    ref = zeros(1, numel(centres));
    for i = 1:numel(centres)
        [pc, wc] = weightEvents(pAttr, [], 2, 1, centres(i), 1.0, ...
            'width', qExt, 'dropInputAttr', false);
        offs = {[], centres(i) - muQ};
        [pq, wq] = translateAttributes(query, [], offs);
        ref(i) = cosSimExpTens(pc, wc, pq, wq, [SIGP, SIGT], [1, 1], ...
            [false, false], [false, false], [0.0, 0.0], ...
            'normalize', normalize, 'verbose', false);
    end
    got = windowedSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
        [false, false], [false, false], [0.0, 0.0], centres, ...
        'normalize', normalize, 'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
    ok = isequal(size(got), [1, numel(centres)]) && max(abs(got(:) - ref(:))) < tol;
    results(end+1, :) = {sprintf('windowedSimilarity locked %s', normalize), ok}; %#ok<SAGROW>
end

% ----- 4. similarity, decoupled 2-D correlogram (lag sweep) --------------
anchors = centres(2:8);
tau     = linspace(-1.0, 1.0, 9);
half    = 1.5;
ref = nan(numel(anchors), numel(tau));
for ia = 1:numel(anchors)
    a = anchors(ia);
    keep = abs(onset - a) <= half;
    pc = {pitch(keep), onset(keep)};
    for it = 1:numel(tau)
        offs = {[], (a - tau(it)) - muQ};
        [pq, wq] = translateAttributes(query, [], offs);
        ref(ia, it) = cosSimExpTens(pc, [], pq, wq, [SIGP, SIGT], [1, 1], ...
            [false, false], [false, false], [0.0, 0.0], ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
end
qc2d = anchors(:) - tau(:).';        % nA x nTau absolute query centres
got = windowedSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
    [false, false], [false, false], [0.0, 0.0], anchors, ...
    'queryCentres', qc2d, 'contextWindow', {1.0, 2 * half}, ...
    'normalize', 'oneSidedDenom', 'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
ok = isequal(size(got), size(ref)) && max(abs(got(:) - ref(:))) < tol;
results(end+1, :) = {'windowedSimilarity decoupled correlogram (2-D)', ok}; %#ok<SAGROW>

% ----- 5. generative sweep matches explicit centres ----------------------
explicit = onset(1):1.0:onset(end);
gExp = windowedSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
    [false, false], [false, false], [0.0, 0.0], explicit, ...
    'normalize', 'oneSidedDenom', 'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
gGen = windowedSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
    [false, false], [false, false], [0.0, 0.0], [], ...
    'start', onset(1), 'stop', onset(end), 'step', 1.0, ...
    'normalize', 'oneSidedDenom', 'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
ok = isequal(size(gGen), size(gExp)) && max(abs(gGen(:) - gExp(:))) < tol;
results(end+1, :) = {'windowedSimilarity generative == explicit', ok}; %#ok<SAGROW>

% ----- 6. error paths -----------------------------------------------------
gotR2 = windowedEntropy(pAttr, [], [SIGP, SIGT], [1, 2], [false, false], ...
    [false, false], [0.0, 0.0], centres, 'contextWindow', {1.0, WW}, ...
    'windowAttr', 2, 'dropWindowAttr', true, 'verbose', false);
okR2 = isequal(size(gotR2), [1, numel(centres)]) && all(isfinite(gotR2(:)));
results(end+1, :) = {'windowedEntropy drop r>=2 now allowed', okR2}; %#ok<SAGROW>

threwMarg = false;
try
    windowedEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], centres, 'contextWindow', {1.0, WW}, ...
        'windowAttr', 2, 'dropWindowAttr', false, 'marginalise', 1, ...
        'verbose', false);
catch
    threwMarg = true;
end
results(end+1, :) = {'windowedEntropy marginalise not implemented', threwMarg}; %#ok<SAGROW>

threwWidth = false;
try
    windowedEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], centres, 'contextWindow', {1.0, []}, ...
        'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
catch
    threwWidth = true;
end
results(end+1, :) = {'windowedEntropy requires width', threwWidth}; %#ok<SAGROW>

threwBoth = false;
try
    windowedSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
        [false, false], [false, false], [0.0, 0.0], centres, ...
        'step', 1.0, 'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
catch
    threwBoth = true;
end
results(end+1, :) = {'windowedSimilarity centres+step errors', threwBoth}; %#ok<SAGROW>

% ----- 7. single-axis == one-entry multi-axis sweep ---------------------
sgl = windowedSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
    [false, false], [false, false], [0.0, 0.0], centres, ...
    'normalize', 'oneSidedDenom', 'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false);
mlt = windowedSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
    [false, false], [false, false], [0.0, 0.0], ...
    'sweep', {2, centres}, 'drop', {2, false}, ...
    'normalize', 'oneSidedDenom', 'verbose', false);
ok = isequal(size(sgl), size(mlt)) && max(abs(sgl(:) - mlt(:))) < tol;
results(end+1, :) = {'windowedSimilarity single == one-entry sweep', ok}; %#ok<SAGROW>

% ----- 8. multi-axis two-axis map (pitch swept+compared, time dropped) ---
patP = [60, 63, 60, 65]; patT = [0, 0.5, 1.5, 2.0];
PP = [patP, patP + 5]; TT = [patT, patT + 10];
[pb8, wb8, sb8] = bindEvents({PP, TT}, [], [4, 4], 'step', 1, 'relOuter', [false, false]);
[qb8, qw8, ~]   = bindEvents({patP, patT}, [], [4, 4], 'step', 1, 'relOuter', [false, false]);
R8 = windowedSimilarity(pb8, wb8, qb8, qw8, [SIGP, SIGT], [1, 1], [false, false], ...
    [false, false], [0.0, 0.0], 'sweep', {2, [1.0, 11.0]; 1, [62.0, 67.0]}, ...
    'drop', {2, true; 1, false}, 'specs', sb8, 'verbose', false);
ok = isequal(size(R8), [2, 2]) && R8(1,1) > 0.99 && R8(2,2) > 0.99 ...
     && R8(1,2) < 0.5 && R8(2,1) < 0.5;
results(end+1, :) = {'windowedSimilarity two-axis map', ok}; %#ok<SAGROW>

% ----- 9. locate is wired (centroid vs start peak at different centres) --
[qb9, qw9, qs9] = bindEvents({[60, 64], [0.0, 0.6]}, [], [1, 2], 'step', 1, 'relOuter', [false, false]);
cc9 = linspace(-0.4, 0.7, 12);
common9 = {'windowAttr', 2, 'dropWindowAttr', false, 'contextWindow', {1.0, 1.5}, ...
           'normalize', 'cosine', 'specs', qs9, 'verbose', false};
aC = windowedSimilarity(qb9, qw9, qb9, qw9, [SIGP, SIGT], [1, 2], [false, false], ...
    [false, false], [0.0, 0.0], cc9, 'locate', 'centroid', common9{:});
aS = windowedSimilarity(qb9, qw9, qb9, qw9, [SIGP, SIGT], [1, 2], [false, false], ...
    [false, false], [0.0, 0.0], cc9, 'locate', 'start', common9{:});
[~, iC] = max(aC); [~, iS] = max(aS);
ok = max(aC) > 0.9 && max(aS) > 0.9 && abs(cc9(iC) - cc9(iS)) > 0.2 && ~isequal(aC, aS);
results(end+1, :) = {'windowedSimilarity locate wired', ok}; %#ok<SAGROW>


if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_windowed_premaet: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}
            fprintf('  FAIL: %s\n', results{i, 1});
        end
    end
end

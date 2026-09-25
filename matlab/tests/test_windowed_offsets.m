%% test_windowed_offsets.m — offsets in windowedSimilarity
%
%  Translation of the query indexed by its offset (article, Sec. 3,
%  attribute translation). The travelling-window form is pinned to window
%  centres written out by hand as the offset plus the query's position,
%  and the one-pass sweep route to the per-offset comparison. Mirror of
%  Python's tests/test_windowed_offsets.py.
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

claveO = [0 3 6 10 12];
tO   = [claveO, claveO + 16];
pO   = transformAttributes([60 62 64 67 64 67 69 71 74 71], [], {'midi', 'cents'});
ctxO = {pO, tO};
qryO = {pO(1:3), tO(1:3)};                     % C D E, centroid onset 3
geomO = {[20 0.5], [1 1], [false false], [true false], [1200 0]};

% ---- offsets equal centres at offset + the query's position -------------
muO = -4:0.5:24;
newO = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, [], 'offsets', muO, ...
    'windowAttr', 2, 'contextWindow', {'gaussian', 16}, 'verbose', false);
oldO = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, muO + 3, ...
    'windowAttr', 2, 'dropWindowAttr', false, ...
    'contextWindow', {'gaussian', 16}, 'verbose', false);
[~, iMaxO] = max(newO);
results(end+1, :) = {'windowedSimilarity offsets = centres at offset + position', ...
    isequal(newO, oldO) && muO(iMaxO) == 0}; %#ok<SAGROW>

% ---- correlogram route = per-offset comparison ---------------------------
% With the centres at each offset plus the query's position, the diagonal
% of the (routed) correlogram is the window travelling with the query,
% which is compared offset by offset (never routed).
cenO = [0 8 16 24];
muR = -4:2:24;
newR = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, muR + 3, 'offsets', muR, ...
    'windowAttr', 2, 'contextWindow', {'gaussian', 16}, 'verbose', false);
oldR = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, [], 'offsets', muR, ...
    'windowAttr', 2, 'contextWindow', {'gaussian', 16}, 'verbose', false);
newC = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, cenO, 'offsets', muR, ...
    'windowAttr', 2, 'contextWindow', {'gaussian', 16}, 'verbose', false);
results(end+1, :) = {'windowedSimilarity correlogram route = per-offset comparison', ...
    isequal(size(newR), [numel(muR), numel(muR)]) ...
    && isequal(size(newC), [numel(cenO), numel(muR)]) ...
    && max(abs(diag(newR).' - oldR(:).')) <= 1e-10}; %#ok<SAGROW>

% ---- a translated attribute with no window is pure translation ----------
muP = -2:1:22;
gotP = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, [], ...
    'offsets', {2, muP}, 'verbose', false);
dxP = buildMaet(ctxO, [], geomO{:}, 'verbose', false);
dyP = buildMaet(qryO, [], geomO{:}, 'verbose', false);
wantP = sweepSimMaet(dxP, dyP, [zeros(size(muP)); muP], ...
    'normalize', 'oneSidedDenom', 'verbose', false);
results(end+1, :) = {'windowedSimilarity offsets without a window = sweepSimMaet', ...
    max(abs(gotP(:) - wantP(:))) <= 1e-10}; %#ok<SAGROW>

% ---- pitch translated, time windowed and dropped: one pass = per offset --
muM  = -600:100:600;
cenM = 0:2:28;
qPos = mean(qryO{1});
newM = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, [], ...
    'offsets', {1, muM}, 'sweep', {2, cenM}, 'drop', {2, true}, ...
    'contextWindow', {2, struct('shape', 'gaussian', 'width', 12)}, ...
    'verbose', false);
oldM = windowedSimilarity(ctxO, [], qryO, [], geomO{:}, [], ...
    'sweep', {1, muM + qPos; 2, cenM}, 'drop', {1, false; 2, true}, ...
    'contextWindow', {1, struct('shape', 'rect', 'width', 1e7); ...
                      2, struct('shape', 'gaussian', 'width', 12)}, ...
    'verbose', false);
results(end+1, :) = {'windowedSimilarity multi-window-attribute offsets = per-offset comparison', ...
    isequal(size(newM), [numel(muM), numel(cenM)]) ...
    && max(abs(newM(:) - oldM(:))) <= 1e-10}; %#ok<SAGROW>

% ---- refusals -----------------------------------------------------------
okR = true;
try
    windowedSimilarity(ctxO, [], qryO, [], geomO{:}, [], 'offsets', 0:3, ...
        'windowAttr', 2, 'dropWindowAttr', true);
    okR = false;
catch err
    okR = okR && strcmp(err.identifier, 'windowedSimilarity:offsetsDropped');
end
try
    windowedSimilarity(ctxO, [], qryO, [], [20 0.5], [1 1], [false true], ...
        [true false], [1200 0], [], 'offsets', 0:3, 'windowAttr', 2);
    okR = false;
catch err
    okR = okR && strcmp(err.identifier, 'windowedSimilarity:offsetsRelative');
end
results(end+1, :) = {'windowedSimilarity offsets refusals', okR}; %#ok<SAGROW>

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_windowed_offsets: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}
            fprintf('  FAIL: %s\n', results{i, 1});
        end
    end
end

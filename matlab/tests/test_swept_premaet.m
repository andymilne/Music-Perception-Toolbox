%% test_swept_premaet.m — pre-MAET sweptSimilarity / sweptEntropy
%
%  Validates the pre-MAET sweptSimilarity and sweptEntropy against
%  the equivalent inline pipeline (weightEvents / translateAttributes /
%  buildMaet / entropyMaet / simMaet) they replace. Mirror of
%  Python's tests/test_swept_premaet.py.
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

% Deterministic triple (no RNG dependence across languages): a rising
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

% ----- 1. entropy, drop the time attribute (differential & renyi2) -----
methodsShapes = {'differential', 0.0; 'renyi2', 0.0; 'renyi2', 1.0};
for k = 1:size(methodsShapes, 1)
    method = methodsShapes{k, 1};
    shape  = methodsShapes{k, 2};
    ref = zeros(1, numel(centres));
    for i = 1:numel(centres)
        if shape == 1.0
            [pw, ww] = unpackPreMaet(weightEvents(pAttr, [], 2, 1, centres(i), shape, ...
                'width', WW, 'dropInputAttr', true));
        else
            [pw, ww] = unpackPreMaet(weightEvents(pAttr, [], 2, 1, centres(i), shape, ...
                'sd', SD, 'dropInputAttr', true));
        end
        dens = buildMaet(pw, ww, SIGP, 1, false, false, 0.0, 'verbose', false);
        ref(i) = entropyMaet(dens, 'method', method, 'verbose', false);
    end
    got = sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
        'window', {2, {shape, 'width', WW}}, 'drop', 2, 'method', method, ...
        'verbose', false);
    ok = max(abs(got(:) - ref(:))) < tol;
    results(end+1, :) = {sprintf('sweptEntropy drop window attribute %s shape=%g', method, shape), ok}; %#ok<SAGROW>
end

% ----- 1b. the discrete methods take their grid through sweptEntropy ------
gridArgs = {'nPointsPerDim', 400, 'xMin', 40, 'xMax', 80};
for method = {'shannon', 'normalized'}
    ref = zeros(1, numel(centres));
    for i = 1:numel(centres)
        [pw, ww] = unpackPreMaet(weightEvents(pAttr, [], 2, 1, centres(i), 0.0, ...
            'sd', SD, 'dropInputAttr', true));
        dens = buildMaet(pw, ww, SIGP, 1, false, false, 0.0, 'verbose', false);
        ref(i) = entropyMaet(dens, 'method', method{1}, gridArgs{:}, ...
            'verbose', false);
    end
    got = sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
        'window', {2, {'gaussian', 'width', WW}}, 'drop', 2, 'method', method{1}, ...
        gridArgs{:}, 'verbose', false);
    ok = max(abs(got(:) - ref(:))) < tol;
    try
        sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
            [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
            'window', {2, {'gaussian', 'width', WW}}, 'drop', 2, 'method', method{1}, ...
            'verbose', false);
        ok = false;                     % the grid is required
    catch
    end
    results(end+1, :) = {sprintf('sweptEntropy %s takes the grid', method{1}), ok}; %#ok<SAGROW>
end

% ----- 2. entropy, retain the time attribute (joint pitch-time renyi2) --------
ref = zeros(1, numel(centres));
for i = 1:numel(centres)
    [pw, ww] = unpackPreMaet(weightEvents(pAttr, [], 2, 1, centres(i), 1.0, ...
        'width', WW, 'dropInputAttr', false));
    dens = buildMaet(pw, ww, [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], 'verbose', false);
    ref(i) = entropyMaet(dens, 'method', 'renyi2', 'verbose', false);
end
got = sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
    [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
    'window', {2, {'rect', 'width', WW}}, 'method', 'renyi2', 'verbose', false);
results(end+1, :) = {'sweptEntropy joint (retain attribute)', ...
    max(abs(got(:) - ref(:))) < tol}; %#ok<SAGROW>

% ----- 3. similarity, locked template sweep (oneSidedDenom & cosine) -----
for nm = {'oneSidedDenom', 'cosine'}
    normalize = nm{1};
    ref = zeros(1, numel(centres));
    for i = 1:numel(centres)
        [pc, wc] = unpackPreMaet(weightEvents(pAttr, [], 2, 1, centres(i), 1.0, ...
            'width', qExt, 'dropInputAttr', false));
        offs = {[], centres(i) - muQ};
        [pq, wq] = unpackPreMaet(translateAttributes(query, [], offs));
        ref(i) = simMaet(pc, wc, pq, wq, [SIGP, SIGT], [1, 1], ...
            [false, false], [false, false], [0.0, 0.0], ...
            'normalize', normalize, 'verbose', false);
    end
    got = sweptSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
        [false, false], [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
        'align', {2, 'both'}, 'window', {2, {'rect', 'width', qExt}}, ...
        'normalize', normalize, 'verbose', false);
    ok = isequal(size(got), [1, numel(centres)]) && max(abs(got(:) - ref(:))) < tol;
    results(end+1, :) = {sprintf('sweptSimilarity locked %s', normalize), ok}; %#ok<SAGROW>
end

% ----- 4. similarity, correlogram (lag measured from each window value) ---
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
        [pq, wq] = unpackPreMaet(translateAttributes(query, [], offs));
        ref(ia, it) = simMaet(pc, [], pq, wq, [SIGP, SIGT], [1, 1], ...
            [false, false], [false, false], [0.0, 0.0], ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
end
qc2d = anchors(:) - tau(:).';        % nA x nTau query positions
got = sweptSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
    [false, false], [false, false], [0.0, 0.0], ...
    'sweep', {2, {anchors, qc2d}}, 'align', {2, 'independent'}, ...
    'window', {2, {'rect', 'width', 2 * half}}, 'normalize', 'oneSidedDenom', ...
    'verbose', false);
ok = isequal(size(got), size(ref)) && max(abs(got(:) - ref(:))) < tol;
results(end+1, :) = {'sweptSimilarity independent correlogram (2-D)', ok}; %#ok<SAGROW>

% ----- 5. generated sweep values match listed ones -----------------------
% A window-only role: its generated sweep values default to the context's
% extent, stepped at half the window's sd.
geom5 = {[SIGP, SIGT], [1, 1], [false, false], [false, false], [0.0, 0.0]};
common5 = {'align', {2, 'window'}, 'drop', 2, 'window', {2, {'rect', 'width', 1.0}}, ...
           'normalize', 'oneSidedDenom', 'verbose', false};
explicit = onset(1):1.0:onset(end);
gExp = sweptSimilarity(pAttr, [], query, [], geom5{:}, ...
    'sweep', {2, explicit}, common5{:});
gGen = sweptSimilarity(pAttr, [], query, [], geom5{:}, ...
    'start', {2, onset(1)}, 'stop', {2, onset(end)}, 'step', {2, 1.0}, ...
    common5{:});
gDef = sweptSimilarity(pAttr, [], query, [], geom5{:}, ...
    'step', {2, 1.0}, common5{:});            % start / stop: the context's extent
% A Gaussian window takes half its sd as the default step; a pure
% rectangle takes its pieces instead (test_swept_sweep_values).
gauss5 = {'align', {2, 'window'}, 'drop', 2, 'window', {2, {'gaussian', 'width', 1.0}}, ...
          'normalize', 'oneSidedDenom', 'verbose', false};
stHalf = 1.0 / (2 * sqrt(3)) / 2;          % half the sd, width-1 equivalent
nHalf = floor((onset(end) - onset(1)) / stHalf + 1e-9) + 1;
gHalf = sweptSimilarity(pAttr, [], query, [], geom5{:}, ...
    'sweep', {2, onset(1) + stHalf * (0:nHalf - 1)}, gauss5{:});
gDflt = sweptSimilarity(pAttr, [], query, [], geom5{:}, ...
    'start', {2, onset(1)}, gauss5{:});       % step: half the window's sd
ok = isequal(size(gGen), size(gExp)) && max(abs(gGen(:) - gExp(:))) < tol ...
     && isequal(gDef, gGen) && isequal(size(gDflt), size(gHalf)) ...
     && max(abs(gDflt(:) - gHalf(:))) < tol;
results(end+1, :) = {'sweptSimilarity generated == listed sweep values', ok}; %#ok<SAGROW>

% ----- 6. error paths -----------------------------------------------------
gotR2 = sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 2], [false, false], ...
    [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
    'window', {2, {'rect', 'width', WW}}, 'drop', 2, 'verbose', false);
okR2 = isequal(size(gotR2), [1, numel(centres)]) && all(isfinite(gotR2(:)));
results(end+1, :) = {'sweptEntropy drop r>=2 now allowed', okR2}; %#ok<SAGROW>

threwWidth = false;
try
    sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
        'window', {2, {'rect'}}, 'verbose', false);
catch errW
    threwWidth = strcmp(errW.identifier, 'sweptEntropy:badWindow');
end
results(end+1, :) = {'sweptEntropy requires width', threwWidth}; %#ok<SAGROW>

threwBoth = false;
try
    sweptSimilarity(pAttr, [], query, [], [SIGP, SIGT], [1, 1], ...
        [false, false], [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
        'step', {2, 1.0}, 'align', {2, 'both'}, 'window', {2, {'rect', 'width', 1.0}}, ...
        'verbose', false);
catch errB
    threwBoth = strcmp(errB.identifier, 'sweptSimilarity:sweepAndGenerator');
end
results(end+1, :) = {'sweptSimilarity sweep + step errors', threwBoth}; %#ok<SAGROW>

% ----- 8. two swept attributes (pitch translated with its window, time dropped) ---
patP = [60, 63, 60, 65]; patT = [0, 0.5, 1.5, 2.0];
PP = [patP, patP + 5]; TT = [patT, patT + 10];
[pb8, wb8, sb8] = unpackPreMaet(bindEvents({PP, TT}, [], [4, 4], 'step', 1, 'relOuter', [false, false]));
[qb8, qw8, ~]   = unpackPreMaet(bindEvents({patP, patT}, [], [4, 4], 'step', 1, 'relOuter', [false, false]));
R8 = sweptSimilarity(pb8, wb8, qb8, qw8, [SIGP, SIGT], [1, 1], [false, false], ...
    [false, false], [0.0, 0.0], 'sweep', {2, [1.0, 11.0]; 1, [62.0, 67.0]}, ...
    'align', {2, 'window'; 1, 'both'}, 'drop', 2, ...
    'window', {1, {'rect', 'width', 5.0}; 2, {'rect', 'width', 2.0}}, 'specs', sb8, 'verbose', false);
% one dimension per swept attribute, in attribute order: (pitch, time)
ok = isequal(size(R8), [2, 2]) && R8(1,1) > 0.99 && R8(2,2) > 0.99 ...
     && R8(1,2) < 0.5 && R8(2,1) < 0.5;
results(end+1, :) = {'sweptSimilarity two-attribute map', ok}; %#ok<SAGROW>

% ----- 8b. locate as a per-attribute map -------------------------------------
% A map {a, rule; ...} names each window attribute's rule, and an attribute the map
% does not name takes 'centroid'. So a map naming only the time attribute must
% reproduce the scalar default on the pitch attribute, and differ from the
% scalar rule applied to both.
common8b = {'sweep', {2, [1.0, 11.0]; 1, [62.0, 67.0]}, ...
            'align', {2, 'window'; 1, 'both'}, 'drop', 2, ...
            'window', {1, {'rect', 'width', 5.0}; 2, {'rect', 'width', 2.0}}, 'specs', sb8, ...
            'verbose', false};
R8b = @(loc) sweptSimilarity(pb8, wb8, qb8, qw8, [SIGP, SIGT], [1, 1], ...
    [false, false], [false, false], [0.0, 0.0], 'locate', loc, common8b{:});
mapDefault = R8b({2, 'centroid'});
mapStart   = R8b({2, 'start'});
mapBoth    = R8b({1, 'centroid'; 2, 'start'});
bothStart  = R8b('start');
ok = max(abs(mapDefault(:) - R8(:))) < tol && ...
     max(abs(mapStart(:) - mapBoth(:))) < tol && ...
     max(abs(mapStart(:) - bothStart(:))) > 0.1;
results(end+1, :) = {'sweptSimilarity locate map', ok}; %#ok<SAGROW>

% ----- 9. locate is wired (centroid vs start peak at different centres) --
[qb9, qw9, qs9] = unpackPreMaet(bindEvents({[60, 64], [0.0, 0.6]}, [], [1, 2], 'step', 1, 'relOuter', [false, false]));
cc9 = linspace(-0.4, 0.7, 12);
common9 = {'sweep', {2, cc9}, 'align', {2, 'both'}, 'window', {2, {'rect', 'width', 1.5}}, ...
           'normalize', 'cosine', 'specs', qs9, 'verbose', false};
aC = sweptSimilarity(qb9, qw9, qb9, qw9, [SIGP, SIGT], [1, 2], [false, false], ...
    [false, false], [0.0, 0.0], 'locate', 'centroid', common9{:});
aS = sweptSimilarity(qb9, qw9, qb9, qw9, [SIGP, SIGT], [1, 2], [false, false], ...
    [false, false], [0.0, 0.0], 'locate', 'start', common9{:});
[~, iC] = max(aC); [~, iS] = max(aS);
ok = max(aC) > 0.9 && max(aS) > 0.9 && abs(cc9(iC) - cc9(iS)) > 0.2 && ~isequal(aC, aS);
results(end+1, :) = {'sweptSimilarity locate wired', ok}; %#ok<SAGROW>


% ---- A query windowed with weightEvents before the call ----------------
%  The documented route for a window on the query. The query is
%  asymmetric --- two close events plus one far one --- so a narrow window
%  around the close pair reshapes it rather than rescaling it, and the
%  reshaped query matches the context better.
pcQ = {[0, 100, 200, 300, 400, 500], [0, 1, 2, 3, 4, 5]};
wcQ = {ones(1, 6), ones(1, 6)};
pqQ = {[0, 100, 500], [0, 1, 4]};
wqQ = {ones(1, 3), ones(1, 3)};
commonQ = {'sweep', {2, 0:5}, 'align', {2, 'both'}, ...
           'window', {2, {'gauss', 'width', 2.0}}, 'normalize', 'cosine', 'verbose', false};
runQ = @(pq, wq, varargin) sweptSimilarity(pcQ, wcQ, pq, wq, [30.0, 0.25], ...
    [1, 1], [false, false], [false, false], [0.0, 0.0], commonQ{:}, varargin{:});
[pqW, wqW] = unpackPreMaet(weightEvents(pqQ, wqQ, 2, 1, 0.5, 0, 'sd', 0.8, ...
    'dropInputAttr', false));
plainQ  = runQ(pqQ, wqQ);
narrowQ = runQ(pqW, wqW);
ok = ~isequal(plainQ, narrowQ) && max(narrowQ) > max(plainQ) && ...
     min(narrowQ) >= 0 && max(narrowQ) <= 1 + 1e-12;
results(end+1, :) = {'sweptSimilarity query windowed with weightEvents', ok}; %#ok<SAGROW>


% ---- Windows are evaluated by weightEvents' implementation ---------------
%  Every profile of weightEvents aligned at a reference value is a window,
%  with the same factors: exponentials (symmetric and extending to one
%  side only) and a function handle, compared with the inline weightEvents
%  pipeline.
profs = {struct('shape', 'exponential', 'sd', 1.0), {'exponential', 'sd', 1.0}; ...
         struct('shape', 'exponentialBefore', 'decayRate', 0.5), ...
             {'exponentialBefore', 'decayRate', 0.5}; ...
         struct('shape', 'exponentialAfter', 'sd', 1.5), ...
             {'exponentialAfter', 'sd', 1.5}; ...
         @(d) exp(-abs(d)), {@(d) exp(-abs(d))}};
for k = 1:size(profs, 1)
    weArgs = profs{k, 2};
    ref = zeros(1, numel(centres));
    for i = 1:numel(centres)
        [pw, ww] = unpackPreMaet(weightEvents(pAttr, [], 2, 1, centres(i), ...
            weArgs{1}, weArgs{2:end}, 'dropInputAttr', true));
        dens = buildMaet(pw, ww, SIGP, 1, false, false, 0.0, 'verbose', false);
        ref(i) = entropyMaet(dens, 'method', 'renyi2', 'verbose', false);
    end
    got = sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
        [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
        'window', {2, profs{k, 1}}, 'drop', 2, 'method', 'renyi2', ...
        'verbose', false);
    ok = max(abs(got(:) - ref(:))) < tol;
    results(end+1, :) = {sprintf('window profile %d is weightEvents', k), ok}; %#ok<SAGROW>
end

% Serial-position profiles are anchored at the first and last events, not
% at the sweep value, so they are refused as windows; an exponential has no
% width; closed edges belong to rectangles; the scale must be named, and
% only the window's own names are taken.
badWins = {struct('shape', 'uShape', 'sd', 1.0), 'sweptEntropy:badWindow'; ...
           {'exponential', 'width', 2.0}, 'sweptEntropy:window:widthWithNamedProfile'; ...
           {'gaussian', 'width', 2.0, 'edges', 'closed'}, 'sweptEntropy:window:edgesNotRect'; ...
           {'rect', 2.0}, 'sweptEntropy:positionalWindow'; ...
           {1.0, 2.0, 'closed'}, 'sweptEntropy:positionalWindow'; ...
           {'gaussian', 'sigma', 1.0}, 'sweptEntropy:badWindow'; ...
           {'gaussian', 'sd'}, 'sweptEntropy:badWindow'; ...
           struct('shape', 'gaussian', 'sigma', 1.0), 'sweptEntropy:badWindow'};
for k = 1:size(badWins, 1)
    try
        sweptEntropy(pAttr, [], [SIGP, SIGT], [1, 1], [false, false], ...
            [false, false], [0.0, 0.0], 'sweep', {2, centres}, ...
            'window', {2, badWins{k, 1}}, 'drop', 2, 'verbose', false);
        ok = false;
    catch e
        ok = strcmp(e.identifier, badWins{k, 2});
    end
    results(end+1, :) = {sprintf('window refusal %d', k), ok}; %#ok<SAGROW>
end

% On a periodic attribute the window's displacement wraps, as weightEvents
% wraps it: a window aligned at 0 on a cycle of 4 reaches an event at 3.9.
pPer = {[60 64 67], [0.1 3.9 2.0]};
[pw, ww] = unpackPreMaet(weightEvents(pPer, [], 2, 1, 0.0, 1.0, ...
    'width', 1.0, 'isPer', true, 'period', 4.0, 'dropInputAttr', true));
dens = buildMaet(pw, ww, 0.5, 1, false, false, 0.0, 'verbose', false);
ref = entropyMaet(dens, 'method', 'renyi2', 'verbose', false);
got = sweptEntropy(pPer, [], [0.5, 0.1], [1, 1], [false, false], ...
    [false, true], [0.0, 4.0], 'sweep', {2, 0}, 'window', {2, {'rect', 'width', 1.0}}, ...
    'drop', 2, 'method', 'renyi2', 'verbose', false);
ok = isequal(ww{1}(:).', [1 1 0]) && abs(got - ref) < tol;
results(end+1, :) = {'window wraps on a periodic attribute', ok}; %#ok<SAGROW>

% weightEvents reduces a multi-valued input by locate, and closes a
% rectangle's upper edge with edges 'closed'.
tK = [0 1 2; 0.5 1.5 2.5];
[~, wS] = unpackPreMaet(weightEvents({[60 62 64], tK}, [], 2, 1, 1.0, 1.0, ...
    'width', 2.0, 'locate', 'start', 'dropInputAttr', true));
[~, wO] = unpackPreMaet(weightEvents({[60 62 64], [0 1 2]}, [], 2, 1, 1.0, ...
    1.0, 'width', 2.0, 'dropInputAttr', true));
[~, wC] = unpackPreMaet(weightEvents({[60 62 64], [0 1 2]}, [], 2, 1, 1.0, ...
    1.0, 'width', 2.0, 'edges', 'closed', 'dropInputAttr', true));
ok = isequal(wS{1}(:).', [1 1 0]) && isequal(wO{1}(:).', [1 1 0]) && ...
     isequal(wC{1}(:).', [1 1 1]);
results(end+1, :) = {'weightEvents locate and edges', ok}; %#ok<SAGROW>

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_swept_premaet: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}
            fprintf('  FAIL: %s\n', results{i, 1});
        end
    end
end

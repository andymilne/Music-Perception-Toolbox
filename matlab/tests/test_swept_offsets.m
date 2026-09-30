%% test_swept_offsets.m — what the sweep values of sweptSimilarity apply to
%
%  The window, the query, both together, or each independently
%  ('align').
%  Each role is pinned to the composition it stands for, written out with
%  weightEvents / translateAttributes / simMaet; the one-pass route through
%  sweepSimMaet is pinned to the composition; and every refusal names the
%  role the caller probably meant. Mirror of Python's
%  tests/test_swept_offsets.py.
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
qRefO = mean(qryO{2});
sdO = 16 / (2 * sqrt(3));

% ---- 'both': the window and the query at each sweep value -----------------
sB = -2:2:26;
gotB = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sB}, ...
    'align', {2, 'both'}, 'window', {2, {'gaussian', 16}}, 'verbose', false);
[gotB0, muB0] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sB}, ...
    'align', {2, 'both'}, 'queryRef', {2, 0}, ...
    'window', {2, {'gaussian', 16}}, 'verbose', false);
[~, muB] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sB}, ...
    'align', {2, 'both'}, 'window', {2, {'gaussian', 16}}, 'verbose', false);
wantB = zeros(1, numel(sB)); wantB0 = zeros(1, numel(sB));
for i = 1:numel(sB)
    [pc, wc] = unpackPreMaet(weightEvents(ctxO, [], 2, 1, sB(i), 0.0, ...
        'sd', sdO, 'dropInputAttr', false));
    [pq, wq] = unpackPreMaet(translateAttributes(qryO, [], {[], sB(i) - qRefO}));
    wantB(i) = simMaet(pc, wc, pq, wq, geomO{:}, 'normalize', 'oneSidedDenom', ...
        'verbose', false);
    % One rule: the window is aligned at sB(i) and the query's reference
    % lands there, whatever it is; queryRef 0 puts the query's start, not
    % its middle, under the window's centre.
    [pq, wq] = unpackPreMaet(translateAttributes(qryO, [], {[], sB(i)}));
    wantB0(i) = simMaet(pc, wc, pq, wq, geomO{:}, 'normalize', 'oneSidedDenom', ...
        'verbose', false);
end
results(end+1, :) = {'sweptSimilarity both = window and query at each value', ...
    max(abs(gotB - wantB)) <= 1e-12 && max(abs(gotB0 - wantB0)) <= 1e-12}; %#ok<SAGROW>
results(end+1, :) = {'sweptSimilarity returns the translation mu = s - queryRef', ...
    isequal(muB0{2}, sB) && max(abs(muB{2} - (sB - qRefO))) <= 1e-12 ...
    && isempty(muB{1})}; %#ok<SAGROW>

% ---- 'independent': the correlogram, shared and per-row query values ------
wvI = [0 8 16 24];
qvI = -2:2:26;
gotI = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, {wvI, qvI}}, ...
    'align', {2, 'independent'}, 'window', {2, {'gaussian', 16}}, 'verbose', false);
lagI = wvI(:) - [2 0 -2];
gotL = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, {wvI, lagI}}, ...
    'align', {2, 'independent'}, 'window', {2, {'gaussian', 16}}, 'verbose', false);
wantI = zeros(numel(wvI), numel(qvI)); wantL = zeros(numel(wvI), 3);
for a = 1:numel(wvI)
    [pc, wc] = unpackPreMaet(weightEvents(ctxO, [], 2, 1, wvI(a), 0.0, ...
        'sd', sdO, 'dropInputAttr', false));
    for b = 1:numel(qvI)
        [pq, wq] = unpackPreMaet(translateAttributes(qryO, [], {[], qvI(b) - qRefO}));
        wantI(a, b) = simMaet(pc, wc, pq, wq, geomO{:}, ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
    for b = 1:3
        [pq, wq] = unpackPreMaet(translateAttributes(qryO, [], {[], lagI(a, b) - qRefO}));
        wantL(a, b) = simMaet(pc, wc, pq, wq, geomO{:}, ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
end
results(end+1, :) = {'sweptSimilarity independent = window and query placed separately', ...
    isequal(size(gotI), [numel(wvI), numel(qvI)]) ...
    && max(abs(gotI(:) - wantI(:))) <= 1e-10 ...
    && max(abs(gotL(:) - wantL(:))) <= 1e-10}; %#ok<SAGROW>

% ---- 'query': pure translation, no window --------------------------------
muP = -2:1:22;
gotP = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, muP}, ...
    'align', {2, 'query'}, 'queryRef', {2, 0}, 'verbose', false);
dxP = buildMaet(ctxO, [], geomO{:}, 'verbose', false);
dyP = buildMaet(qryO, [], geomO{:}, 'verbose', false);
wantP = sweepSimMaet(dxP, dyP, [zeros(size(muP)); muP], ...
    'normalize', 'oneSidedDenom', 'verbose', false);
% The canonical call: a bare sweep is translation over the whole context,
% the sweep values being the offsets (align 'query' and queryRef 0 are the
% defaults).
[plainP, muPlain] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, ...
    'sweep', {2, muP}, 'verbose', false);
results(end+1, :) = {'sweptSimilarity query = sweepSimMaet', ...
    max(abs(gotP(:) - wantP(:))) <= 1e-10 && isequal(plainP, gotP) ...
    && isequal(muPlain{2}, muP)}; %#ok<SAGROW>

% ---- pitch translated alone, time windowed and dropped: one pass = composition
muM  = -600:100:600;
tvM = 0:4:28;
sdM = 12 / (2 * sqrt(3));
newM = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, ...
    'sweep', {1, muM; 2, tvM}, 'align', {1, 'query'; 2, 'window'}, ...
    'drop', 2, 'window', {2, {'gaussian', 12}}, 'queryRef', {1, 0}, ...
    'verbose', false);
wantM = zeros(numel(muM), numel(tvM));
for b = 1:numel(tvM)
    [pc, wc] = unpackPreMaet(weightEvents(ctxO, [], 2, 1, tvM(b), 0.0, ...
        'sd', sdM, 'dropInputAttr', true));
    for a = 1:numel(muM)
        pq = {qryO{1} + muM(a)};
        wantM(a, b) = simMaet(pc, wc, pq, [], 20, 1, false, true, 1200, ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
end
results(end+1, :) = {'sweptSimilarity query + window (dropped) = composition', ...
    isequal(size(newM), [numel(muM), numel(tvM)]) ...
    && max(abs(newM(:) - wantM(:))) <= 1e-10}; %#ok<SAGROW>

% ---- 'independent' window values generated by start / stop / step --------
qvG = 0:4:24;
gen = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, {[], qvG}}, ...
    'step', {2, 4}, 'align', {2, 'independent'}, 'window', {2, {'rect', 8}}, ...
    'verbose', false);
lst = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, {0:4:28, qvG}}, ...
    'align', {2, 'independent'}, 'window', {2, {'rect', 8}}, 'verbose', false);
results(end+1, :) = {'sweptSimilarity independent window values generated', ...
    isequal(gen, lst)}; %#ok<SAGROW>

% ---- 'window' on a compared attribute: the query compared in place --------
%  The query is compared as written with the windowed context, and windows
%  that tile the context give contributions that sum to the whole-context
%  similarity (the inner product is linear in the weights).
sC = 0:2:28;
gotC = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sC}, ...
    'align', {2, 'window'}, 'window', {2, {'gaussian', 8}}, 'verbose', false);
wantC = zeros(1, numel(sC));
for i = 1:numel(sC)
    [pc, wc] = unpackPreMaet(weightEvents(ctxO, [], 2, 1, sC(i), 0.0, ...
        'sd', 8 / (2 * sqrt(3)), 'dropInputAttr', false));
    wantC(i) = simMaet(pc, wc, qryO, [], geomO{:}, 'normalize', 'oneSidedDenom', ...
        'verbose', false);
end
tilesC = (-2:4:26) + 2;                        % half-open [c-2, c+2)
partsC = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, tilesC}, ...
    'align', {2, 'window'}, 'window', {2, {'rect', 4}}, 'verbose', false);
wholeC = simMaet(ctxO, [], qryO, [], geomO{:}, 'normalize', 'oneSidedDenom', ...
    'verbose', false);
results(end+1, :) = {'sweptSimilarity window on a compared attribute compares in place', ...
    max(abs(gotC - wantC)) <= 1e-12 ...
    && abs(sum(partsC) - wholeC) <= 1e-12 * abs(wholeC)}; %#ok<SAGROW>

% ---- a bare attribute takes default sweep values -------------------------
%  For a translation, every offset at which query and context overlap,
%  stepped at h, half the peaks' sd (sigma * sqrt(2 / r) / 2), on the
%  values' lattice;
%  on a periodic attribute, one period;
%  start / stop / step override one default at a time. 'both' translates
%  the query too and takes the same defaults; 'window' keeps the context's
%  extent at half the window's sd.
[SD, muD2] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', 2, ...
    'verbose', false);
loD = min(tO) - max(qryO{2}); hiD = max(tO) - min(qryO{2});
SL = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, muD2{2}}, ...
    'verbose', false);
[~, muP1] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', 1, ...
    'verbose', false);
S12 = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', [1 2], ...
    'verbose', false);
[~, muS] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', 2, ...
    'step', {2, 1}, 'verbose', false);
% a bare number applies to the one swept attribute
[~, muBare] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', 2, ...
    'start', 0, 'step', 1, 'verbose', false);
[~, muB2] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', 2, ...
    'align', {2, 'both'}, 'window', {2, {'rect', 8}}, 'verbose', false);
hO = @(sg) sg * sqrt(2) / 2;                % h at r = 1
latO = @(g, h) g / ceil(g / h - 1e-9);       % the lattice step at most h
stT = latO(1, hO(geomO{1}(2)));
stP = latO(100, hO(geomO{1}(1)));
nP = round(1200 / stP);
okD = abs(muD2{2}(1) - loD) < 1e-9 && abs(muD2{2}(end) - hiD) < 1e-9 ...
    && max(abs(diff(muD2{2}) - stT)) < 1e-9 && isequal(SD, SL) ...
    && muP1{1}(1) == 0 && muP1{1}(end) < 1200 && numel(muP1{1}) == nP ...
    && isequal(size(S12), [nP, numel(muD2{2})]) ...
    && max(abs(muS{2} - (loD:1:hiD))) < 1e-9 ...
    && max(abs(muBare{2} - (0:1:hiD))) < 1e-9 ...
    && max(abs(muB2{2} - muD2{2})) < 1e-9 ...
    && numel(sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', 2, ...
        'align', {2, 'window'}, 'drop', 2, 'window', {2, {0.0, 8}}, ...
        'verbose', false)) == floor((max(tO) - min(tO)) / (8 / (2 * sqrt(3)) / 2) + 1e-9) + 1;
results(end+1, :) = {'sweptSimilarity bare attribute takes default sweep values', ...
    okD}; %#ok<SAGROW>

% ---- the default step lands on every exact match -----------------------
%  A whole fraction of the lattice the values lie on, at most h, half the
%  peaks' sd: onsets on a grid of 1 with h = 0.15 step at 1/7 (not 0.15,
%  which misses the whole-number offsets), and a context transposed off
%  the query's lattice is still found exactly. Values on no lattice keep h.
geomL = {[20 0.15 * sqrt(2)], [1 1], [false false], [true false], [1200 0]};
[SLt, muLt] = sweptSimilarity(ctxO, [], qryO, [], geomL{:}, 'sweep', 2, ...
    'verbose', false);
[~, k0] = min(abs(muLt{2}));
[SSh, muSh] = sweptSimilarity({pO + 13.7, tO}, [], qryO, [], geomO{:}, ...
    'sweep', [1 2], 'verbose', false);
[~, iSh] = max(SSh(:));
[iP, iT] = ind2sub(size(SSh), iSh);
[~, muLo] = sweptSimilarity({pO, tO + 0.05 * sin(1:numel(tO))}, [], ...
    qryO, [], geomO{:}, 'sweep', 2, 'verbose', false);
okL = max(abs(diff(muLt{2}) - 1 / 7)) < 1e-9 ...
    && abs(muLt{2}(k0)) < 1e-9 && SLt(k0) > 1 - 1e-9 ...
    && min(abs(muLt{2} - 16)) < 1e-9 ...
    && abs(muSh{1}(iP) - 13.7) < 1e-9 && abs(muSh{2}(iT)) < 1e-9 ...
    && SSh(iSh) > 1 - 1e-9 ...
    && max(abs(diff(muLo{2}) - hO(geomO{1}(2)))) < 1e-9;
results(end+1, :) = {'sweptSimilarity default step lands on every exact match', ...
    okL}; %#ok<SAGROW>

% ---- refusals -----------------------------------------------------------
winR = {2, {'rect', 8}};
cases = {
    {geomO, 'sweep', {2, 0:3}, 'align', {2, 'both'}, 'window', winR, 'drop', 2}, 'sweptSimilarity:dropTranslated';
    {{[20 0.5], [1 1], [false true], [true false], [1200 0]}, 'sweep', {2, 0:3}, 'align', {2, 'both'}, 'window', winR}, 'sweptSimilarity:translateRelative';
    {geomO, 'sweep', {2, 0:3}, 'align', {2, 'both'}, 'window', winR, 'drop', 1}, 'sweptSimilarity:dropNotWindow';
    {geomO, 'sweep', {2, 0:3}, 'align', {2, 'query'}, 'window', winR}, 'sweptSimilarity:windowForQuery';
    {geomO, 'sweep', {2, 0:3}, 'align', {2, 'window'}, 'drop', 2}, 'sweptSimilarity:noWindow';
    {geomO, 'sweep', {2, 0:3}, 'window', winR}, 'sweptSimilarity:windowForQuery';
    {geomO, 'sweep', {2, {0:3, 0:3}}, 'align', {2, 'independent'}}, 'sweptSimilarity:noWindow';
    {geomO, 'sweep', {2, 0:3}, 'align', {1, 'query'; 2, 'query'}}, 'sweptSimilarity:alignWithoutSweep';
    {geomO, 'sweep', {2, 0:3}, 'align', {2, 'follow'}, 'window', winR}, 'sweptSimilarity:badAlign';
    {geomO, 'sweep', {2, 0:3}, 'align', {2, 'independent'}, 'window', winR}, 'sweptSimilarity:independentPair';
    {geomO, 'sweep', {2, {0:3, zeros(2, 3)}}, 'align', {2, 'independent'}, 'window', winR}, 'sweptSimilarity:independentRows';
    {{[20 NaN], [1 1], [false false], [true false], [1200 0]}, 'sweep', 2}, 'sweptSimilarity:stepRequired';
    {geomO, 'sweep', 2, 'align', {2, 'independent'}, 'window', winR}, 'sweptSimilarity:independentPair';
    {geomO, 'sweep', {2, 0:3}, 'align', {2, 'both'}, 'window', winR, 'queryRef', {1, 0}}, 'sweptSimilarity:queryRefUnused';
    {geomO}, 'sweptSimilarity:noSweep';
    {geomO, 'sweep', {2, 0:3}, 'step', {2, 1}, 'align', {2, 'both'}, 'window', winR}, 'sweptSimilarity:sweepAndGenerator';
    {geomO, 'sweep', [1 2], 'step', 1}, 'sweptSimilarity:bareGenerator'};
okR = true;
for k = 1:size(cases, 1)
    c = cases{k, 1};
    try
        sweptSimilarity(ctxO, [], qryO, [], c{1}{:}, c{2:end}, 'verbose', false);
        okR = false;
        fprintf('  refusal case %d did not raise\n', k);
    catch err
        if ~strcmp(err.identifier, cases{k, 2})
            okR = false;
            fprintf('  refusal case %d raised %s\n', k, err.identifier);
        end
    end
end
results(end+1, :) = {'sweptSimilarity refusals', okR}; %#ok<SAGROW>

% ---- a window that travels with the query may be left out ----------------
sD = 0:26;
gotD = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sD}, ...
    'align', {2, 'both'}, 'verbose', false);
sameD = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sD}, ...
    'align', {2, 'both'}, 'window', {2, {'rect', 6, 'closed'}}, ...
    'verbose', false);
gaussD = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sD}, ...
    'align', {2, 'both'}, 'window', {2, {'gaussian', []}}, 'verbose', false);
wideD = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, 'sweep', {2, sD}, ...
    'align', {2, 'both'}, 'window', {2, {'gaussian', 6}}, 'verbose', false);
results(end+1, :) = {'sweptSimilarity default window holds the query', ...
    isequal(gotD, sameD) && abs(gotD(4) - 1) <= 1e-12 ...
    && isequal(gaussD, wideD)}; %#ok<SAGROW>

% ---- a travelling window that leaves out query events warns ---------------
runW = @(spec) sweptSimilarity(ctxO, [], qryO, [], geomO{:}, ...
    'sweep', {2, sD}, 'align', {2, 'both'}, 'window', {2, spec}, ...
    'verbose', false);
okW = true;
lastwarn('');
halfW = runW({'rect', 6});
[msg, id] = lastwarn();
okW = okW && strcmp(id, 'sweptSimilarity:windowCutsQuery') ...
    && contains(msg, 'leaves out 1 of the query''s 3');
lastwarn('');
closedW = runW({'rect', 6, 'closed'});
runW({'rect', 6.5}); runW({'gaussian', 2});
[~, id] = lastwarn();
okW = okW && isempty(id) && abs(closedW(4) - 1) <= 1e-12 && halfW(4) < closedW(4);
results(end+1, :) = {'sweptSimilarity warns when a window cuts the query', ...
    okW}; %#ok<SAGROW>

% ---- the offsets mu read through differencing ------------------------------
%  With query and context written from time 0, mu is the time at which the
%  query starts in the context, and keeps that reading through differencing:
%  the motif and its differenced form, whose middles (and so windows) differ,
%  peak at the same offsets.
kwR = {'step', {2, 0.5}, 'align', {2, 'both'}, ...
       'window', {2, {'gaussian', 16}}, 'verbose', false};
[plainR, muP] = sweptSimilarity(ctxO, [], qryO, [], geomO{:}, kwR{:});
ctxD = unpackPreMaet(differenceEvents(ctxO, [], [1 0]));
qryD = unpackPreMaet(differenceEvents(qryO, [], [1 0]));
[diffR, muD] = sweptSimilarity(ctxD, [], qryD, [], [20 * sqrt(2), 0.5], [1 1], ...
    [false false], [false false], [0 0], kwR{:});
[~, iP] = max(plainR);
i0 = find(muD{2} == 0, 1); i16 = find(muD{2} == 16, 1);
results(end+1, :) = {'sweptSimilarity offsets read through differencing', ...
    muP{2}(iP) == 0 && abs(diffR(i0) - max(diffR)) <= 1e-4 * max(diffR) ...
    && abs(diffR(i16) - max(diffR)) <= 1e-4 * max(diffR) ...
    && diffR(i0) > diffR(i0 - 1) && diffR(i0) > diffR(i0 + 1)}; %#ok<SAGROW>

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_swept_offsets: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}
            fprintf('  FAIL: %s\n', results{i, 1});
        end
    end
end

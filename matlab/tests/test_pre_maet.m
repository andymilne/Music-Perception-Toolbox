%% test_pre_maet.m — packPreMaet and unpackPreMaet
%
%  packPreMaet holds the three parts of a pre-MAET -- pAttr, wAttr, and specs
%  -- in one struct. What is pinned here is that it is genuinely the same
%  pre-MAET however it is passed: every operator takes the whole pre-MAET
%  or the loose triple and returns the whole, the parts survive the round
%  trip, and the two call forms give identical results.
%
%  Mirror of Python tests/test_pre_maet.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

pC   = {[60 62 64 65], [0 1 2 3]};
metC = [1 0.5 0.75 0.5];
wC   = {metC, metC};
spC  = flatSpecs(pC, 'r', 1, 'names', {'pitch', 'time'});
KWC  = {'sigma', [0.5 0.25], 'per', [false false], ...
        'period', [0 0], 'verbose', false};

% ---- Construction ----

pmC = packPreMaet(pC, wC, spC);
results{end+1,1} = 'packPreMaet: has the three fields'; %#ok<SAGROW>
results{end,2}   = isequal(sort(fieldnames(pmC)), ...
                           sort({'pAttr'; 'wAttr'; 'specs'}));

[pBack, wBack, sBack] = unpackPreMaet(pmC);
results{end+1,1} = 'packPreMaet: unpack round trip'; %#ok<SAGROW>
results{end,2}   = isequaln(pBack, pC) && isequaln(wBack, wC) ...
                   && isequaln(sBack, spC);

pmBare = packPreMaet(pC);
results{end+1,1} = 'packPreMaet: unset parts are []'; %#ok<SAGROW>
results{end,2}   = isempty(pmBare.wAttr) && isempty(pmBare.specs);

results{end+1,1} = 'packPreMaet: pre-MAET in, pre-MAET out'; %#ok<SAGROW>
results{end,2}   = isequaln(packPreMaet(pmC), pmC);

spRel = spC;
spRel{1}.rel = true;
pmRel = packPreMaet(pmC, [], spRel);
results{end+1,1} = 'packPreMaet: one part replaced, the rest in place'; %#ok<SAGROW>
results{end,2}   = pmRel.specs{1}.rel && isequaln(pmRel.wAttr, wC) ...
                   && ~pmC.specs{1}.rel;

results{end+1,1} = 'packPreMaet: scalar weights pass through'; %#ok<SAGROW>
results{end,2}   = isequal(packPreMaet(pC, 2).wAttr, 2);

results{end+1,1} = 'packPreMaet: wAttr length mismatch errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() packPreMaet(pC, {metC}));

results{end+1,1} = 'packPreMaet: specs length mismatch errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() packPreMaet(pC, [], spC(1)));

results{end+1,1} = 'packPreMaet: a single spec must be wrapped'; %#ok<SAGROW>
results{end,2}   = throwsError(@() packPreMaet(pC, [], spC{1}));

results{end+1,1} = 'packPreMaet: a bare matrix is not a pAttr'; %#ok<SAGROW>
results{end,2}   = throwsError(@() packPreMaet([60 62 64]));

results{end+1,1} = 'packPreMaet: unpack rejects a non-pre-MAET'; %#ok<SAGROW>
results{end,2}   = throwsError(@() unpackPreMaet(pC));

% A density struct also has a pAttr field; it must not be read as a
% pre-MAET, since showPreMaet accepts both.
densC = buildMaet(pC, wC, 'specs', spC, KWC{:});
results{end+1,1} = 'packPreMaet: a density is not a pre-MAET'; %#ok<SAGROW>
results{end,2}   = ~internal.isPreMaet(densC);

% ---- Operators: whole-pre-MAET form equals loose-triple form ----

results{end+1,1} = 'packPreMaet: differenceEvents'; %#ok<SAGROW>
results{end,2}   = preMaetSame(differenceEvents(pmC, [1 0]), ...
                             differenceEvents(pC, wC, [1 0], 'specs', spC));

results{end+1,1} = 'packPreMaet: bindEvents'; %#ok<SAGROW>
results{end,2}   = preMaetSame(bindEvents(pmC, [2 2]), ...
                             bindEvents(pC, wC, [2 2], 'specs', spC));

results{end+1,1} = 'packPreMaet: translateAttributes'; %#ok<SAGROW>
results{end,2}   = preMaetSame(translateAttributes(pmC, {5, 0}), ...
                             translateAttributes(pC, wC, {5, 0}, ...
                                                 'specs', spC));

results{end+1,1} = 'packPreMaet: transformAttributes'; %#ok<SAGROW>
results{end,2}   = preMaetSame( ...
    transformAttributes(pmC, {{'affine', 'scale', 2}, []}), ...
    transformAttributes(pC, wC, {{'affine', 'scale', 2}, []}, ...
                        'specs', spC));

results{end+1,1} = 'packPreMaet: weightEvents'; %#ok<SAGROW>
results{end,2}   = preMaetSame( ...
    weightEvents(pmC, 2, 2, 1.5, 0, 'sd', 1, 'dropInputAttr', false), ...
    weightEvents(pC, wC, 2, 2, 1.5, 0, 'sd', 1, ...
                 'dropInputAttr', false, 'specs', spC));

% The specs travel with the pre-MAET, so a composition needs no threading.
chained  = differenceEvents(bindEvents(pmC, [2 2]), [1 1]);
[pB2, wB2, sB2] = unpackPreMaet(bindEvents(pC, wC, [2 2], 'specs', spC));
threaded = differenceEvents(pB2, wB2, [1 1], 'specs', sB2);
results{end+1,1} = 'packPreMaet: operators compose without threading specs'; %#ok<SAGROW>
results{end,2}   = preMaetSame(chained, threaded);

results{end+1,1} = 'packPreMaet: weights may not be passed twice'; %#ok<SAGROW>
results{end,2}   = throwsError(@() differenceEvents(pmC, wC, [1 0]));

results{end+1,1} = 'packPreMaet: readPreMaet returns a pre-MAET'; %#ok<SAGROW>
csvText = writePreMaet([], pmC);
pmRead  = readPreMaet(csvText);
results{end,2}   = internal.isPreMaet(pmRead) ...
                   && max(abs(pmRead.pAttr{1} - pC{1})) < 1e-12;

% ---- The boundary: build, eval, cosine ----

densCar = buildMaet(pmC, KWC{:});
Xq = [60; 0];
results{end+1,1} = 'packPreMaet: buildMaet takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(evalMaet(densCar, Xq, 'verbose', false) ...
                       - evalMaet(densC, Xq, 'verbose', false)) < 1e-12;

results{end+1,1} = 'packPreMaet: buildMaet rejects weights twice'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildMaet(pmC, wC, KWC{:}));

spK = spC;
for a = 1:2
    spK{a}.sigma = KWC{2}(a);
    spK{a}.per = false;
    spK{a}.period = 0;
end
pmK = packPreMaet(pC, wC, spK);
results{end+1,1} = 'packPreMaet: evalMaet takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(evalMaet(pmK, Xq, 'verbose', false) ...
                       - evalMaet(densC, Xq, 'verbose', false)) < 1e-12;

results{end+1,1} = 'packPreMaet: entropyMaet takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(entropyMaet(pmK, 'method', 'renyi2', ...
                                      'verbose', false) ...
                       - entropyMaet(densC, 'method', 'renyi2', ...
                                        'verbose', false)) < 1e-10;

results{end+1,1} = 'packPreMaet: simMaet takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(simMaet(pmK, pmK, 'verbose', false) - 1) < 1e-10;

% ---- sweptSimilarity and sweptEntropy take whole pre-MAETs ----

pWC = {[60 64 67 60 64 67], [0 1 2 5 6 7]};
pWQ = {[60 64 67], [0 1 2]};
spW = {struct('name','pitch','r',1,'rel',false,'exch',true, ...
              'sigma',0.5,'per',true,'period',12), ...
       struct('name','time','r',1,'rel',false,'exch',true, ...
              'sigma',0.25,'per',false,'period',0)};
pmWC = packPreMaet(pWC, [], spW);
pmWQ = packPreMaet(pWQ, [], spW);
ctrW = 0:0.5:7;
kwW  = {'sweep', {2, ctrW}, 'align', {2, 'both'}, ...
        'window', {2, {'rect', 'width', 2.5}}, 'verbose', false};

wsPm  = sweptSimilarity(pmWC, pmWQ, kwW{:});
wsPos = sweptSimilarity(pWC, [], pWQ, [], [0.5 0.25], [1 1], ...
    [false false], [true false], [12 0], kwW{:});
results{end+1,1} = 'packPreMaet: sweptSimilarity matches the positional form'; %#ok<SAGROW>
results{end,2}   = isequal(size(wsPm), size(wsPos)) ...
                   && max(abs(wsPm - wsPos)) == 0;

kwE = {'sweep', {2, ctrW}, 'window', {2, {'gaussian', 'width', 2}}, 'method', 'renyi2', ...
       'verbose', false};
wePm  = sweptEntropy(pmWC, kwE{:});
wePos = sweptEntropy(pWC, [], [0.5 0.25], [1 1], [false false], ...
    [true false], [12 0], kwE{:});
results{end+1,1} = 'packPreMaet: sweptEntropy matches the positional form'; %#ok<SAGROW>
results{end,2}   = max(abs(wePm - wePos)) == 0;

wsSel = sweptSimilarity(pmWC, pmWQ, 'sigma', {2, []}, kwW{:});
wsRef = sweptSimilarity(pWC, [], pWQ, [], [2 0.25], [1 1], ...
    [false false], [true false], [12 0], kwW{:});
results{end+1,1} = 'packPreMaet: selective override sweeps one attribute'; %#ok<SAGROW>
results{end,2}   = max(abs(wsSel - wsRef)) == 0 ...
                   && max(abs(wsSel - wsPm)) > 1e-6;

results{end+1,1} = 'packPreMaet: windowed form requires specs'; %#ok<SAGROW>
results{end,2}   = throwsError(@() sweptSimilarity(packPreMaet(pWC), ...
    packPreMaet(pWQ), kwW{:}));

spW2 = spW; spW2{1}.rel = true;
results{end+1,1} = 'packPreMaet: windowed form requires agreeing geometry'; %#ok<SAGROW>
results{end,2}   = throwsError(@() sweptSimilarity(pmWC, ...
    packPreMaet(pWQ, [], spW2), kwW{:}));

% ---- a nested query may hold fewer values than the context ----------
% The two sides of one comparison share the nesting they are read under,
% not their inner cardinality: a context whose events hold four values
% compares against a query whose events hold two.
nqCtx = [60 62 64 65; 67 69 71 72; 60 62 64 65].';
nqQry = [60 64; 67 71].';
nqAxC = 0:(size(nqCtx, 2) - 1);
nqAxQ = 0:(size(nqQry, 2) - 1);
nqBind = @(v, ax) bindEvents({v, ax}, [], [2 1], 'relOuter', true, ...
    'specs', flatSpecs({v, ax}, 'r', 1, 'rel', false, 'exch', true, ...
                       'sigma', 0.5, 'per', false, 'period', 0));
nqC = nqBind(nqCtx, nqAxC);
nqQ = nqBind(nqQry, nqAxQ);
nqPAttr = unpackPreMaet(nqC);
nqCentres = nqPAttr{2};
nqOut = sweptSimilarity(nqC, nqQ, 'sweep', {2, nqCentres}, ...
    'align', {2, 'window'}, 'window', {2, {'rect', 'width', 1.0}}, 'drop', 2, ...
    'normalize', 'oneSidedDenom', 'verbose', false);
nqOut = nqOut(:).';
nqDirect = zeros(1, numel(nqCentres));
nqQD = buildMaet(selectPreMaet(nqQ, 'attributes', 1), 'verbose', false);
for nqK = 1:numel(nqCentres)
    nqOne = selectPreMaet(nqC, 'attributes', 1, 'events', nqK);
    nqDirect(nqK) = simMaet(buildMaet(nqOne, 'verbose', false), nqQD, ...
                            'normalize', 'oneSidedDenom', 'verbose', false);
end
results{end+1,1} = 'packPreMaet: a nested query may be narrower than the context'; %#ok<SAGROW>
results{end,2}   = numel(nqOut) == numel(nqCentres) ...
                   && all(isfinite(nqOut)) ...
                   && max(abs(nqOut - nqDirect)) < 1e-12;

% Flat against nested is not one comparison.
nqFlat = packPreMaet({nqQry}, [], flatSpecs({nqQry}, 'r', 1, 'rel', false, ...
                                            'exch', true, 'sigma', 0.5));
nqNest = bindEvents({nqQry}, [], 2, 'relOuter', true, ...
    'specs', flatSpecs({nqQry}, 'r', 1, 'rel', false, 'exch', true, ...
                       'sigma', 0.5));
results{end+1,1} = 'packPreMaet: both sides must share the nesting'; %#ok<SAGROW>
results{end,2}   = throwsError(@() sweptSimilarity(nqNest, nqFlat, ...
    'sweep', {1, 0}, 'align', {1, 'both'}, 'window', {1, {'rect', 'width', 1.0}}, ...
    'verbose', false));

% ---- a cell of pre-MAETs stands wherever a cell of densities does ----

mkPm = @(v) packPreMaet({v, [0 1 2]}, [], ...
    flatSpecs({v, [0 1 2]}, 'sigma', [0.5 0.25], ...
              'per', [true false], 'period', [12 0]));
pmL1 = mkPm([60 64 67]);
pmL2 = mkPm([62 65 69]);
dL1  = buildMaet(pmL1, 'verbose', false);
dL2  = buildMaet(pmL2, 'verbose', false);

gotLL = simMaet({pmL1, pmL1}, {pmL2, pmL2}, 'verbose', false);
refLL = simMaet({dL1, dL1}, {dL2, dL2}, 'verbose', false);
results{end+1,1} = 'packPreMaet: cell vs cell matches a cell of densities'; %#ok<SAGROW>
results{end,2}   = abs(gotLL{1} - refLL{1}) == 0 && abs(gotLL{2} - refLL{2}) == 0;

gotSL = simMaet(pmL1, {pmL2, pmL1}, 'verbose', false);
results{end+1,1} = 'packPreMaet: scalar vs cell broadcasts'; %#ok<SAGROW>
results{end,2}   = abs(gotSL{2} - 1) < 1e-12;

Xq2 = [60; 0];
gotEv = evalMaet({pmL1, pmL2}, Xq2, 'verbose', false);
refEv = evalMaet({dL1, dL2}, Xq2, 'verbose', false);
results{end+1,1} = 'packPreMaet: evalMaet takes a cell'; %#ok<SAGROW>
results{end,2}   = abs(gotEv{1} - refEv{1}) == 0 && abs(gotEv{2} - refEv{2}) == 0;

% Translated copies, one translateAttributes call each, stand as a cell
% of pre-MAETs, built on the shared geometry.
copiesT = arrayfun(@(mu) translateAttributes(pmL1, {mu, []}), [0 3 7], ...
                   'UniformOutput', false);
gotSw = simMaet(pmL1, copiesT, 'verbose', false);
refSw = cellfun(@(c) buildMaet(c, 'verbose', false), copiesT, ...
                'UniformOutput', false);
refSwS = simMaet(dL1, refSw, 'verbose', false);
okSw = abs(gotSw{1} - 1) < 1e-12;
for m = 1:3
    okSw = okSw && abs(gotSw{m} - refSwS{m}) == 0;
end
results{end+1,1} = 'packPreMaet: translated copies stand as a cell'; %#ok<SAGROW>
results{end,2}   = okSw;

% ---- r / rel / exch overrides ----

pChord = {[60 62 64; 64 65 67], [0 1 2]};
spCh   = flatSpecs(pChord, 'r', 1);
pmCh   = packPreMaet(pChord, [], spCh);
KWCh   = {'sigma', [0.5 0.25], 'per', [false false], ...
          'period', [0 0], 'verbose', false};

dR = buildMaet(pmCh, 'r', [2 1], KWCh{:});
results{end+1,1} = 'packPreMaet: r overrides the specs'; %#ok<SAGROW>
results{end,2}   = isequal(double(dR.r(:)'), [2 1]);

dRS = buildMaet(pmCh, 'r', [2 1], 'rel', [true false], ...
                   'exch', [false true], KWCh{:});
results{end+1,1} = 'packPreMaet: rel and exch override the specs'; %#ok<SAGROW>
results{end,2}   = isequal(logical(dRS.rel(:)'), [true false]) ...
                   && isequal(logical(dRS.exch(:)'), [false true]);

% A sweep over any of the six parameters stays one call per value, and
% leaves the pre-MAET it sweeps unchanged.
sigmas = [0.25 0.5 1];
got = zeros(1, numel(sigmas));
for k = 1:numel(sigmas)
    dK = buildMaet(pmC, 'sigma', [sigmas(k) 0.25], ...
                      'per', [false false], 'period', [0 0], ...
                      'verbose', false);
    got(k) = dK.sigma(1);
end
results{end+1,1} = 'packPreMaet: a sweep is one call per value'; %#ok<SAGROW>
results{end,2}   = max(abs(got - sigmas)) < 1e-12 ...
                   && ~isfield(pmC.specs{1}, 'sigma');

pmBound = bindEvents(pmC, [2 2]);
results{end+1,1} = 'packPreMaet: nested geometry is not overridable'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildMaet(pmBound, 'r', 2, KWC{:}));

results{end+1,1} = 'packPreMaet: wrong-length override errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildMaet(pmC, 'r', [1 1 1], KWC{:}));

results{end+1,1} = 'packPreMaet: unknown name-value errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildMaet(pmC, 'sigmaa', 0.5, ...
                                                'verbose', false));


if standalone
    nPass = 0; nFail = 0;
    for i = 1:size(results, 1)
        if results{i,2}
            nPass = nPass + 1;
            fprintf('  PASS  %s\n', results{i,1});
        else
            nFail = nFail + 1;
            fprintf('  FAIL  %s\n', results{i,1});
        end
    end
    fprintf('\n=== test_pre_maet: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_pre_maet:failed', '%d test(s) failed.', nFail);
    end
end

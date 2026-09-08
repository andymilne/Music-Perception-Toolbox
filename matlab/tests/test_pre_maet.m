%% test_pre_maet.m — preMaet and unpackPreMaet
%
%  preMaet holds the three parts of a pre-MAET -- pAttr, wAttr, and specs
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
spC  = flatSpecs(pC, 'r', 1, 'name', {'pitch', 'time'});
KWC  = {'sigma', [0.5 0.25], 'isPer', [false false], ...
        'period', [0 0], 'verbose', false};

% ---- Construction ----

pmC = preMaet(pC, wC, spC);
results{end+1,1} = 'preMaet: has the three fields'; %#ok<SAGROW>
results{end,2}   = isequal(sort(fieldnames(pmC)), ...
                           sort({'pAttr'; 'wAttr'; 'specs'}));

[pBack, wBack, sBack] = unpackPreMaet(pmC);
results{end+1,1} = 'preMaet: unpack round trip'; %#ok<SAGROW>
results{end,2}   = isequaln(pBack, pC) && isequaln(wBack, wC) ...
                   && isequaln(sBack, spC);

pmBare = preMaet(pC);
results{end+1,1} = 'preMaet: unset parts are []'; %#ok<SAGROW>
results{end,2}   = isempty(pmBare.wAttr) && isempty(pmBare.specs);

results{end+1,1} = 'preMaet: pre-MAET in, pre-MAET out'; %#ok<SAGROW>
results{end,2}   = isequaln(preMaet(pmC), pmC);

spRel = spC;
spRel{1}.rel = true;
pmRel = preMaet(pmC, [], spRel);
results{end+1,1} = 'preMaet: one part replaced, the rest in place'; %#ok<SAGROW>
results{end,2}   = pmRel.specs{1}.rel && isequaln(pmRel.wAttr, wC) ...
                   && ~pmC.specs{1}.rel;

results{end+1,1} = 'preMaet: scalar weights pass through'; %#ok<SAGROW>
results{end,2}   = isequal(preMaet(pC, 2).wAttr, 2);

results{end+1,1} = 'preMaet: wAttr length mismatch errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() preMaet(pC, {metC}));

results{end+1,1} = 'preMaet: specs length mismatch errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() preMaet(pC, [], spC(1)));

results{end+1,1} = 'preMaet: a single spec must be wrapped'; %#ok<SAGROW>
results{end,2}   = throwsError(@() preMaet(pC, [], spC{1}));

results{end+1,1} = 'preMaet: a bare matrix is not a pAttr'; %#ok<SAGROW>
results{end,2}   = throwsError(@() preMaet([60 62 64]));

results{end+1,1} = 'preMaet: unpack rejects a non-pre-MAET'; %#ok<SAGROW>
results{end,2}   = throwsError(@() unpackPreMaet(pC));

% A density struct also has a pAttr field; it must not be read as a
% pre-MAET, since showPreMaet accepts both.
densC = buildExpTens(pC, wC, 'specs', spC, KWC{:});
results{end+1,1} = 'preMaet: a density is not a pre-MAET'; %#ok<SAGROW>
results{end,2}   = ~internal.isPreMaet(densC);

% ---- Operators: whole-pre-MAET form equals loose-triple form ----

results{end+1,1} = 'preMaet: differenceEvents'; %#ok<SAGROW>
results{end,2}   = preMaetSame(differenceEvents(pmC, [1 0]), ...
                             differenceEvents(pC, wC, [1 0], 'specs', spC));

results{end+1,1} = 'preMaet: bindEvents'; %#ok<SAGROW>
results{end,2}   = preMaetSame(bindEvents(pmC, [2 2]), ...
                             bindEvents(pC, wC, [2 2], 'specs', spC));

results{end+1,1} = 'preMaet: translateAttributes'; %#ok<SAGROW>
results{end,2}   = preMaetSame(translateAttributes(pmC, {5, 0}), ...
                             translateAttributes(pC, wC, {5, 0}, ...
                                                 'specs', spC));

results{end+1,1} = 'preMaet: transformAttributes'; %#ok<SAGROW>
results{end,2}   = preMaetSame( ...
    transformAttributes(pmC, {{'affine', 'scale', 2}, []}), ...
    transformAttributes(pC, wC, {{'affine', 'scale', 2}, []}, ...
                        'specs', spC));

results{end+1,1} = 'preMaet: weightEvents'; %#ok<SAGROW>
results{end,2}   = preMaetSame( ...
    weightEvents(pmC, 2, 2, 1.5, 0, 'sd', 1, 'dropInputAttr', false), ...
    weightEvents(pC, wC, 2, 2, 1.5, 0, 'sd', 1, ...
                 'dropInputAttr', false, 'specs', spC));

% The specs travel with the pre-MAET, so a composition needs no threading.
chained  = differenceEvents(bindEvents(pmC, [2 2]), [1 1]);
[pB2, wB2, sB2] = unpackPreMaet(bindEvents(pC, wC, [2 2], 'specs', spC));
threaded = differenceEvents(pB2, wB2, [1 1], 'specs', sB2);
results{end+1,1} = 'preMaet: operators compose without threading specs'; %#ok<SAGROW>
results{end,2}   = preMaetSame(chained, threaded);

results{end+1,1} = 'preMaet: weights may not be passed twice'; %#ok<SAGROW>
results{end,2}   = throwsError(@() differenceEvents(pmC, wC, [1 0]));

results{end+1,1} = 'preMaet: readPreMaet returns a pre-MAET'; %#ok<SAGROW>
csvText = writePreMaet([], pmC);
pmRead  = readPreMaet(csvText);
results{end,2}   = internal.isPreMaet(pmRead) ...
                   && max(abs(pmRead.pAttr{1} - pC{1})) < 1e-12;

% ---- The boundary: build, eval, cosine ----

densCar = buildExpTens(pmC, KWC{:});
Xq = [60; 0];
results{end+1,1} = 'preMaet: buildExpTens takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(evalExpTens(densCar, Xq, 'verbose', false) ...
                       - evalExpTens(densC, Xq, 'verbose', false)) < 1e-12;

results{end+1,1} = 'preMaet: buildExpTens rejects weights twice'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildExpTens(pmC, wC, KWC{:}));

spK = spC;
for a = 1:2
    spK{a}.sigma = KWC{2}(a);
    spK{a}.isPer = false;
    spK{a}.period = 0;
end
pmK = preMaet(pC, wC, spK);
results{end+1,1} = 'preMaet: evalExpTens takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(evalExpTens(pmK, Xq, 'verbose', false) ...
                       - evalExpTens(densC, Xq, 'verbose', false)) < 1e-12;

results{end+1,1} = 'preMaet: entropyExpTens takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(entropyExpTens(pmK, 'method', 'renyi2', ...
                                      'verbose', false) ...
                       - entropyExpTens(densC, 'method', 'renyi2', ...
                                        'verbose', false)) < 1e-10;

results{end+1,1} = 'preMaet: cosSimExpTens takes a pre-MAET'; %#ok<SAGROW>
results{end,2}   = abs(cosSimExpTens(pmK, pmK, 'verbose', false) - 1) < 1e-10;

% ---- windowedSimilarity and windowedEntropy take whole pre-MAETs ----

pWC = {[60 64 67 60 64 67], [0 1 2 5 6 7]};
pWQ = {[60 64 67], [0 1 2]};
spW = {struct('name','pitch','r',1,'rel',false,'sym',true, ...
              'sigma',0.5,'isPer',true,'period',12), ...
       struct('name','time','r',1,'rel',false,'sym',true, ...
              'sigma',0.25,'isPer',false,'period',0)};
pmWC = preMaet(pWC, [], spW);
pmWQ = preMaet(pWQ, [], spW);
ctrW = 0:0.5:7;
kwW  = {'windowAttr', 2, 'dropWindowAttr', false, 'verbose', false};

wsPm  = windowedSimilarity(pmWC, pmWQ, ctrW, kwW{:});
wsPos = windowedSimilarity(pWC, [], pWQ, [], [0.5 0.25], [1 1], ...
    [false false], [true false], [12 0], ctrW, kwW{:});
results{end+1,1} = 'preMaet: windowedSimilarity matches the positional form'; %#ok<SAGROW>
results{end,2}   = isequal(size(wsPm), size(wsPos)) ...
                   && max(abs(wsPm - wsPos)) == 0;

kwE = [kwW, {'contextWindow', {0, 2}, 'method', 'renyi2'}];
wePm  = windowedEntropy(pmWC, ctrW, kwE{:});
wePos = windowedEntropy(pWC, [], [0.5 0.25], [1 1], [false false], ...
    [true false], [12 0], ctrW, kwE{:});
results{end+1,1} = 'preMaet: windowedEntropy matches the positional form'; %#ok<SAGROW>
results{end,2}   = max(abs(wePm - wePos)) == 0;

wsSel = windowedSimilarity(pmWC, pmWQ, ctrW, 'sigma', {2, []}, kwW{:});
wsRef = windowedSimilarity(pWC, [], pWQ, [], [2 0.25], [1 1], ...
    [false false], [true false], [12 0], ctrW, kwW{:});
results{end+1,1} = 'preMaet: selective override sweeps one attribute'; %#ok<SAGROW>
results{end,2}   = max(abs(wsSel - wsRef)) == 0 ...
                   && max(abs(wsSel - wsPm)) > 1e-6;

results{end+1,1} = 'preMaet: windowed form requires specs'; %#ok<SAGROW>
results{end,2}   = throwsError(@() windowedSimilarity(preMaet(pWC), ...
    preMaet(pWQ), ctrW, kwW{:}));

spW2 = spW; spW2{1}.rel = true;
results{end+1,1} = 'preMaet: windowed form requires agreeing geometry'; %#ok<SAGROW>
results{end,2}   = throwsError(@() windowedSimilarity(pmWC, ...
    preMaet(pWQ, [], spW2), ctrW, kwW{:}));

% ---- a cell of pre-MAETs stands wherever a cell of densities does ----

mkPm = @(v) preMaet({v, [0 1 2]}, [], ...
    flatSpecs({v, [0 1 2]}, 'sigma', [0.5 0.25], ...
              'isPer', [true false], 'period', [12 0]));
pmL1 = mkPm([60 64 67]);
pmL2 = mkPm([62 65 69]);
dL1  = buildExpTens(pmL1, 'verbose', false);
dL2  = buildExpTens(pmL2, 'verbose', false);

gotLL = cosSimExpTens({pmL1, pmL1}, {pmL2, pmL2}, 'verbose', false);
refLL = cosSimExpTens({dL1, dL1}, {dL2, dL2}, 'verbose', false);
results{end+1,1} = 'preMaet: cell vs cell matches a cell of densities'; %#ok<SAGROW>
results{end,2}   = abs(gotLL{1} - refLL{1}) == 0 && abs(gotLL{2} - refLL{2}) == 0;

gotSL = cosSimExpTens(pmL1, {pmL2, pmL1}, 'verbose', false);
results{end+1,1} = 'preMaet: scalar vs cell broadcasts'; %#ok<SAGROW>
results{end,2}   = abs(gotSL{2} - 1) < 1e-12;

Xq2 = [60; 0];
gotEv = evalExpTens({pmL1, pmL2}, Xq2, 'verbose', false);
refEv = evalExpTens({dL1, dL2}, Xq2, 'verbose', false);
results{end+1,1} = 'preMaet: evalExpTens takes a cell'; %#ok<SAGROW>
results{end,2}   = abs(gotEv{1} - refEv{1}) == 0 && abs(gotEv{2} - refEv{2}) == 0;

% translateAttributes' sweep form carries one pre-MAET whose pAttr is a
% 1 x M sweep; it stands as a cell of densities on the shared geometry.
pmSweep = translateAttributes(pmL1, {[0 3 7], []});
gotSw = cosSimExpTens(pmL1, pmSweep, 'verbose', false);
refSw = cell(1, 3);
for m = 1:3
    refSw{m} = buildExpTens( ...
        preMaet(pmSweep.pAttr{m}, pmSweep.wAttr, pmSweep.specs), ...
        'verbose', false);
end
refSwS = cosSimExpTens(dL1, refSw, 'verbose', false);
okSw = abs(gotSw{1} - 1) < 1e-12;
for m = 1:3
    okSw = okSw && abs(gotSw{m} - refSwS{m}) == 0;
end
results{end+1,1} = 'preMaet: a sweep pre-MAET is a cell of densities'; %#ok<SAGROW>
results{end,2}   = okSw;

% ---- r / rel / sym overrides ----

pChord = {[60 62 64; 64 65 67], [0 1 2]};
spCh   = flatSpecs(pChord, 'r', 1);
pmCh   = preMaet(pChord, [], spCh);
KWCh   = {'sigma', [0.5 0.25], 'isPer', [false false], ...
          'period', [0 0], 'verbose', false};

dR = buildExpTens(pmCh, 'r', [2 1], KWCh{:});
results{end+1,1} = 'preMaet: r overrides the specs'; %#ok<SAGROW>
results{end,2}   = isequal(double(dR.r(:)'), [2 1]);

dRS = buildExpTens(pmCh, 'r', [2 1], 'rel', [true false], ...
                   'sym', [false true], KWCh{:});
results{end+1,1} = 'preMaet: rel and sym override the specs'; %#ok<SAGROW>
results{end,2}   = isequal(logical(dRS.isRel(:)'), [true false]) ...
                   && isequal(logical(dRS.isSym(:)'), [false true]);

% A sweep over any of the six parameters stays one call per value, and
% leaves the pre-MAET it sweeps unchanged.
sigmas = [0.25 0.5 1];
got = zeros(1, numel(sigmas));
for k = 1:numel(sigmas)
    dK = buildExpTens(pmC, 'sigma', [sigmas(k) 0.25], ...
                      'isPer', [false false], 'period', [0 0], ...
                      'verbose', false);
    got(k) = dK.sigma(1);
end
results{end+1,1} = 'preMaet: a sweep is one call per value'; %#ok<SAGROW>
results{end,2}   = max(abs(got - sigmas)) < 1e-12 ...
                   && ~isfield(pmC.specs{1}, 'sigma');

pmBound = bindEvents(pmC, [2 2]);
results{end+1,1} = 'preMaet: nested geometry is not overridable'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildExpTens(pmBound, 'r', 2, KWC{:}));

results{end+1,1} = 'preMaet: wrong-length override errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildExpTens(pmC, 'r', [1 1 1], KWC{:}));

results{end+1,1} = 'preMaet: unknown name-value errors'; %#ok<SAGROW>
results{end,2}   = throwsError(@() buildExpTens(pmC, 'sigmaa', 0.5, ...
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

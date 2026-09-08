%% test_windowed_nested.m — nested-triple windowedSimilarity / windowedEntropy
%
%  Validates the nested-triple branch of windowedSimilarity and
%  windowedEntropy (the 'specs' argument) against the explicit composition
%  it stands in for (weightEvents / translateAttributes -> buildExpTens with
%  'specs' -> cosSimExpTens / entropyExpTens). To prove the geometry is read
%  from specs and not from the positional r/isRel, the calls pass
%  deliberately wrong flat r/isRel and still match. Mirror of Python's
%  tests/test_windowed_nested.py.
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

tol  = 1e-9;
SIGP = 0.15;
SIGT = 0.125;
AXIS = 2;        % window on time (attribute 2)
TGT  = 1;        % target the bound pitch (attribute 1)
SIG  = [SIGP, SIGT];
ISP  = [false, false];
PER  = [0.0, 0.0];

% Deterministic passage with a clean (+3, -3, +5) statement at 1-based
% notes 9..12, bound into ordered relative four-note super-events plus a
% flat onset-time axis.
N     = 16;
pitch = [0 2 4 5 7 5 4 2 0 3 0 5 7 9 7 5] + 60;
onset = cumsum(repmat(0.5, 1, N));
width = 0.4;     % narrow context window: ~one super-event per centre
qi    = 9;       % the clean statement's super-event (1-based)

% ----- triples: plain and spectrally augmented --------------------------
[pbP, wbP, sbP] = unpackPreMaet(bindEvents({pitch, onset}, [], [4 1], ...
    'step', 1, 'relOuter', true));

Kp = 8;
[ppS, wpS] = addSpectra(pitch, [], 'harmonic', Kp, 'powerlaw', 1.0, 'units', 12);
% addSpectra flattens p_matrix (M x K) column-major (partial-major), so the
% K-by-N value matrix is reshape(.., N, Kp).' (column j = note j's partials).
PIT = reshape(ppS, N, Kp).';
WP  = reshape(wpS, N, Kp).';
[pbS, wbS, sbS] = unpackPreMaet(bindEvents({PIT, onset}, {WP, []}, [4 1], ...
    'step', 1, 'relOuter', true));

centres = pbP{2}(1:size(pbP{1}, 2));   % one centre per super-event (its onset)

% ===== 1. plain: locked nested sweep == hand-built; peaks at statement ====
qryP = {pbP{1}(:, qi), pbP{2}(:, qi)};
wqryP = wbP;                            % slice the query's weights at qi
if iscell(wbP)
    for a = 1:numel(wbP)
        if ~isempty(wbP{a}), wqryP{a} = wbP{a}(:, qi); end
    end
end
muQ = mean(qryP{AXIS}(:));
ref = zeros(1, numel(centres));
for i = 1:numel(centres)
    [pc, wc, sc] = unpackPreMaet(weightEvents(pbP, wbP, AXIS, TGT, centres(i), 1.0, ...
        'width', width, 'dropInputAttr', false, 'specs', sbP));
    dc = buildExpTens(pc, wc, 'sigma', SIG, 'isPer', ISP, 'period', PER, ...
        'specs', sc, 'verbose', false);
    offs = {[], centres(i) - muQ};
    [pq, wq, sq] = unpackPreMaet(translateAttributes(qryP, wqryP, offs, 'specs', sbP));
    dq = buildExpTens(pq, wq, 'sigma', SIG, 'isPer', ISP, 'period', PER, ...
        'specs', sq, 'verbose', false);
    ref(i) = cosSimExpTens(dc, dq, 'normalize', 'oneSidedDenom', 'verbose', false);
end
% deliberately WRONG flat r/isRel: nested mode must ignore them
gotP = windowedSimilarity(pbP, wbP, qryP, wqryP, SIG, [1 1], ...
    [false false], ISP, PER, centres, ...
    'contextWindow', {1.0, width}, 'normalize', 'oneSidedDenom', ...
    'dropWindowAttr', false, 'windowAttr', AXIS, 'targetAttr', TGT, 'specs', sbP, 'verbose', false);
okShape = isequal(size(gotP), [1, numel(centres)]);
results(end+1, :) = {'windowedSimilarity nested locked (plain) == handbuilt', ...
    okShape && max(abs(gotP(:) - ref(:))) < tol}; %#ok<SAGROW>
[~, am] = max(gotP);
results(end+1, :) = {'windowedSimilarity nested locked (plain) peaks at statement', ...
    am == qi && gotP(qi) > 0.99}; %#ok<SAGROW>

% ===== 2. transposition invariance (rel = 1 in specs) ====================
qryT = {qryP{1} + 6, qryP{2}};         % transpose the super-event up a tritone
shifted = windowedSimilarity(pbP, wbP, qryT, wqryP, SIG, [1 1], ...
    [false false], ISP, PER, centres, ...
    'contextWindow', {1.0, width}, 'normalize', 'oneSidedDenom', ...
    'dropWindowAttr', false, 'windowAttr', AXIS, 'targetAttr', TGT, 'specs', sbP, 'verbose', false);
results(end+1, :) = {'windowedSimilarity nested transposition-invariant (rel=1)', ...
    max(abs(gotP(:) - shifted(:))) < tol}; %#ok<SAGROW>

% ===== 3. spectral: locked nested sweep == hand-built ====================
qryS = {pbS{1}(:, qi), pbS{2}(:, qi)};
wqryS = wbS;
if iscell(wbS)
    for a = 1:numel(wbS)
        if ~isempty(wbS{a}), wqryS{a} = wbS{a}(:, qi); end
    end
end
muQS = mean(qryS{AXIS}(:));
refS = zeros(1, numel(centres));
for i = 1:numel(centres)
    [pc, wc, sc] = unpackPreMaet(weightEvents(pbS, wbS, AXIS, TGT, centres(i), 1.0, ...
        'width', width, 'dropInputAttr', false, 'specs', sbS));
    dc = buildExpTens(pc, wc, 'sigma', SIG, 'isPer', ISP, 'period', PER, ...
        'specs', sc, 'verbose', false);
    offs = {[], centres(i) - muQS};
    [pq, wq, sq] = unpackPreMaet(translateAttributes(qryS, wqryS, offs, 'specs', sbS));
    dq = buildExpTens(pq, wq, 'sigma', SIG, 'isPer', ISP, 'period', PER, ...
        'specs', sq, 'verbose', false);
    refS(i) = cosSimExpTens(dc, dq, 'normalize', 'oneSidedDenom', 'verbose', false);
end
gotS = windowedSimilarity(pbS, wbS, qryS, wqryS, SIG, [1 1], ...
    [false false], ISP, PER, centres, ...
    'contextWindow', {1.0, width}, 'normalize', 'oneSidedDenom', ...
    'dropWindowAttr', false, 'windowAttr', AXIS, 'targetAttr', TGT, 'specs', sbS, 'verbose', false);
results(end+1, :) = {'windowedSimilarity nested locked (spectral) == handbuilt', ...
    isequal(size(gotS), [1, numel(centres)]) && max(abs(gotS(:) - refS(:))) < tol}; %#ok<SAGROW>

% ===== 4. nested entropy == hand-built ===================================
eWidth   = 2.0;
eCentres = linspace(min(pbP{2}), max(pbP{2}), 7);
refH = zeros(1, numel(eCentres));
for i = 1:numel(eCentres)
    [pw, ww, sw] = unpackPreMaet(weightEvents(pbP, wbP, AXIS, TGT, eCentres(i), 1.0, ...
        'width', eWidth, 'dropInputAttr', false, 'specs', sbP));
    densH = buildExpTens(pw, ww, 'sigma', SIG, 'isPer', ISP, 'period', PER, ...
        'specs', sw, 'verbose', false);
    refH(i) = entropyExpTens(densH, 'method', 'renyi2', 'verbose', false);
end
gotH = windowedEntropy(pbP, wbP, SIG, [1 1], [false false], ISP, PER, ...
    eCentres, 'contextWindow', {1.0, eWidth}, 'method', 'renyi2', ...
    'dropWindowAttr', false, 'windowAttr', AXIS, 'targetAttr', TGT, 'specs', sbP, 'verbose', false);
results(end+1, :) = {'windowedEntropy nested == handbuilt', ...
    max(abs(gotH(:) - refH(:))) < tol}; %#ok<SAGROW>

% ===== 5. specs = [] reproduces the flat result exactly ==================
pFlat = {[48 50 52 55 57 59 60 62 64 65 67 69 71 72 74 76], onset};
qFlat = {[60 64 67], [0 0.5 1.0]};
fc = linspace(onset(1), onset(end), 9);
aRes = windowedSimilarity(pFlat, [], qFlat, [], [0.12 0.05], [1 1], ...
    [false false], ISP, PER, fc, ...
    'dropWindowAttr', false, 'windowAttr', 2, 'normalize', 'oneSidedDenom', 'verbose', false);
bRes = windowedSimilarity(pFlat, [], qFlat, [], [0.12 0.05], [1 1], ...
    [false false], ISP, PER, fc, ...
    'dropWindowAttr', false, 'windowAttr', 2, 'normalize', 'oneSidedDenom', 'specs', [], 'verbose', false);
results(end+1, :) = {'windowedSimilarity specs=[] == flat', isequal(aRes, bRes)}; %#ok<SAGROW>

% ===== 6. empty window scores exactly 0 (plain and spectral) ============
% Two identical (+3, -3, +5) statements separated by a wide rest, so a
% centre in the rest catches no super-event. The nested (spectral) path
% must agree with the plain path: exactly 0, not NaN or an error. Mirror of
% Python's test_similarity_empty_window_scores_zero.
eIv  = [0 3 0 5];
ePit = [60 + eIv, 60 + eIv];
eOn  = [0 1 2 3, 40 41 42 43];                 % wide rest around t = 20
eCtr = [0 20 40];                              % 20 falls in the rest
eNe  = numel(ePit);
% plain
[epbP, ewbP, esbP] = unpackPreMaet(bindEvents({ePit, eOn}, [], [4 1], 'step', 1, 'relOuter', true));
eqP  = {epbP{1}(:, 1), epbP{2}(:, 1)};
gotEP = windowedSimilarity(epbP, ewbP, eqP, [], SIG, [1 1], [true false], ISP, PER, ...
    eCtr, 'contextWindow', {1.0, 0.6}, 'windowAttr', AXIS, 'dropWindowAttr', true, ...
    'normalize', 'oneSidedDenom', 'specs', esbP, 'verbose', false);
% spectral
[ePITv, eWPv] = addSpectra(ePit, [], 'harmonic', Kp, 'powerlaw', 1.0, 'units', 12);
ePITm = reshape(ePITv, eNe, Kp).';
eWPm  = reshape(eWPv, eNe, Kp).';
[epbS, ewbS, esbS] = unpackPreMaet(bindEvents({ePITm, eOn}, {eWPm, []}, [4 1], 'step', 1, 'relOuter', true));
eqS  = {epbS{1}(:, 1), epbS{2}(:, 1)};
ewqS = {ewbS{1}(:, 1), []};
gotES = windowedSimilarity(epbS, ewbS, eqS, ewqS, SIG, [1 1], [true false], ISP, PER, ...
    eCtr, 'contextWindow', {1.0, 0.6}, 'windowAttr', AXIS, 'dropWindowAttr', true, ...
    'normalize', 'oneSidedDenom', 'specs', esbS, 'verbose', false);
okEmpty = all(isfinite(gotEP(:))) && all(isfinite(gotES(:))) ...
    && gotEP(2) == 0 && gotES(2) == 0 ...
    && gotEP(1) > 0.99 && gotEP(3) > 0.99 ...
    && gotES(1) > 0.99 && gotES(3) > 0.99;
results(end+1, :) = {'windowedSimilarity empty window -> 0 (plain & spectral)', okEmpty}; %#ok<SAGROW>

if standalone
    nPass = sum([results{:, 2}]);
    fprintf('test_windowed_nested: %d/%d passed\n', nPass, size(results, 1));
    for i = 1:size(results, 1)
        if ~results{i, 2}
            fprintf('  FAIL: %s\n', results{i, 1});
        end
    end
end

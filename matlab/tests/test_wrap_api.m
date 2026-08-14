%% test_wrap_api.m
%  Tests for the ``wrap=`` build API. Mirrors the Python
%  tests/test_wrap_api.py.
%
%  Covers:
%   - Default is 'full-image' on every attribute of every density.
%   - 'single-image' opt-in is honoured (per-attribute or scalar).
%   - Wrap is validated: unknown strings error at build time, and
%     per-attribute arrays must match the attribute count.
%   - Full-image cosine is bounded by 1 at large sigma/P, where the
%     single-image kernel is no longer positive definite.
%   - Single-image opt-in reproduces the pre-v3 numerics; agreement
%     with full-image at low sigma/P and divergence at high sigma/P.
%   - Wrap is quiet for non-periodic attributes; wrap affects rel-per
%     dispatch at high sigma/P and is silent below the threshold.
%
%  No local functions: this file is executed as a script from
%  test_mpt.m, so probes are written inline.
%
%  Standalone-runnable; appends to ``results`` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

wrapPERIOD = 1200;

% Silence the abs-per single-image opt-in warning throughout: several
% cases here explicitly opt into single-image at sigma/P above the
% threshold, and the warning is expected there.
wrapPrevW = warning('off', 'buildExpTens:absPerSingleImage');
wrapCleanup = onCleanup(@() warning(wrapPrevW));


%% ---- Default behaviour ----

pW = [0; 100; 300; 700];
wW = ones(4, 1);
dW = buildExpTens({pW}, {wW}, 60, 2, false, true, wrapPERIOD, 'verbose', false);
results(end+1, :) = { ...
    'wrap api: default is full-image (single-attribute)', ...
    isfield(dW, 'wrap') && iscell(dW.wrap) && numel(dW.wrap) == 1 ...
        && strcmp(dW.wrap{1}, 'full-image')}; %#ok<*SAGROW>

pW1 = [0; 100; 300];
pW2 = [50; 200; 400];
wW3 = ones(3, 1);
dW = buildExpTens({pW1, pW2}, {wW3, wW3}, [60 60], [2 2], ...
    [false false], [true true], [wrapPERIOD wrapPERIOD], 'verbose', false);
results(end+1, :) = { ...
    'wrap api: default is full-image (multi-attribute)', ...
    numel(dW.wrap) == 2 ...
        && strcmp(dW.wrap{1}, 'full-image') ...
        && strcmp(dW.wrap{2}, 'full-image')};

dW = buildExpTens({pW1, pW2}, {wW3, wW3}, [60 60], [2 2], ...
    [false false], [true true], [wrapPERIOD wrapPERIOD], ...
    'wrap', 'single-image', 'verbose', false);
results(end+1, :) = { ...
    'wrap api: scalar wrap broadcast', ...
    numel(dW.wrap) == 2 ...
        && strcmp(dW.wrap{1}, 'single-image') ...
        && strcmp(dW.wrap{2}, 'single-image')};

dW = buildExpTens({pW1, pW2}, {wW3, wW3}, [60 60], [2 2], ...
    [false false], [true true], [wrapPERIOD wrapPERIOD], ...
    'wrap', {'full-image', 'single-image'}, 'verbose', false);
results(end+1, :) = { ...
    'wrap api: per-attribute wrap honoured', ...
    numel(dW.wrap) == 2 ...
        && strcmp(dW.wrap{1}, 'full-image') ...
        && strcmp(dW.wrap{2}, 'single-image')};


%% ---- Validation ----

wrapBadRaised = false;
try
    buildExpTens({pW1}, {wW3}, 60, 2, false, true, wrapPERIOD, ...
        'wrap', 'torus', 'verbose', false); %#ok<NASGU>
catch e
    wrapBadRaised = strcmp(e.identifier, 'buildExpTens:invalidWrap');
end
results(end+1, :) = { ...
    'wrap api: unknown wrap string raises invalidWrap', wrapBadRaised};

wrapBadRaised = false;
try
    buildExpTens({pW1, pW2}, {wW3, wW3}, [60 60], [2 2], ...
        [false false], [true true], [wrapPERIOD wrapPERIOD], ...
        'wrap', {'full-image'}, 'verbose', false); %#ok<NASGU>
catch e
    wrapBadRaised = strcmp(e.identifier, 'buildExpTens:invalidWrap');
end
results(end+1, :) = { ...
    'wrap api: wrong-length wrap array raises invalidWrap', wrapBadRaised};


%% ---- Full-image is PD: cosine <= 1 at large sigma/P ----

wrapPa = [0; 100; 300; 700];
wrapPb = [50; 250; 500; 900];
wrapW4 = ones(4, 1);
wrapPDok = true;
for sopIter = [0.20, 0.30, 0.50]
    sig = sopIter * wrapPERIOD;
    d1 = buildExpTens({wrapPa}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
        'verbose', false);
    d2 = buildExpTens({wrapPb}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
        'verbose', false);
    c12 = cosSimExpTens(d1, d2, 'verbose', false);
    c11 = cosSimExpTens(d1, d1, 'verbose', false);
    c22 = cosSimExpTens(d2, d2, 'verbose', false);
    wrapPDok = wrapPDok && (c12 <= 1 + 1e-10) ...
        && (abs(c11 - 1) < 1e-10) && (abs(c22 - 1) < 1e-10);
end
results(end+1, :) = { ...
    'wrap api: full-image cosine bounded by 1 at large sigma/P', ...
    wrapPDok};


%% ---- Single-image opt-in reproduces pre-v3 numerics ----

sig = 0.03 * wrapPERIOD;   % well below threshold
d1F = buildExpTens({wrapPa}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'verbose', false);
d1S = buildExpTens({wrapPa}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
d2F = buildExpTens({wrapPb}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'verbose', false);
d2S = buildExpTens({wrapPb}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
cF = cosSimExpTens(d1F, d2F, 'verbose', false);
cS = cosSimExpTens(d1S, d2S, 'verbose', false);
results(end+1, :) = { ...
    'wrap api: single-image matches full-image below threshold', ...
    abs(cF - cS) < 1e-12};

sig = 0.20 * wrapPERIOD;   % well above threshold
d1F = buildExpTens({wrapPa}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'verbose', false);
d1S = buildExpTens({wrapPa}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
d2F = buildExpTens({wrapPb}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'verbose', false);
d2S = buildExpTens({wrapPb}, {wrapW4}, sig, 2, false, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
cF = cosSimExpTens(d1F, d2F, 'verbose', false);
cS = cosSimExpTens(d1S, d2S, 'verbose', false);
results(end+1, :) = { ...
    'wrap api: single-image differs from full-image above threshold', ...
    abs(cF - cS) > 1e-6};


%% ---- Wrap axis quiet for non-periodic ----

pNP = [0; 100; 300; 700];
wNP = ones(4, 1);
d1 = buildExpTens({pNP}, {wNP}, 50, 2, false, false, 0, 'verbose', false);
d1S = buildExpTens({pNP}, {wNP}, 50, 2, false, false, 0, ...
    'wrap', 'single-image', 'verbose', false);
c1 = cosSimExpTens(d1, d1, 'verbose', false);
c1S = cosSimExpTens(d1S, d1S, 'verbose', false);
results(end+1, :) = { ...
    'wrap api: wrap irrelevant for non-periodic', ...
    (abs(c1 - c1S) < 1e-12) && (abs(c1 - 1) < 1e-12)};


%% ---- Wrap affects rel-per at high sigma/P and is quiet below ----

sig = 0.20 * wrapPERIOD;
d1a = buildExpTens({wrapPa}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'verbose', false);
d1b = buildExpTens({wrapPa}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
d2a = buildExpTens({wrapPb}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'verbose', false);
d2b = buildExpTens({wrapPb}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
cF = cosSimExpTens(d1a, d2a, 'verbose', false);
cS = cosSimExpTens(d1b, d2b, 'verbose', false);
results(end+1, :) = { ...
    'wrap api: wrap affects rel-per above threshold', ...
    abs(cF - cS) > 1e-6};

sig = 0.02 * wrapPERIOD;
d1a = buildExpTens({wrapPa}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'verbose', false);
d1b = buildExpTens({wrapPa}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
d2a = buildExpTens({wrapPb}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'verbose', false);
d2b = buildExpTens({wrapPb}, {wrapW4}, sig, 2, true, true, wrapPERIOD, ...
    'wrap', 'single-image', 'verbose', false);
cF = cosSimExpTens(d1a, d2a, 'verbose', false);
cS = cosSimExpTens(d1b, d2b, 'verbose', false);
results(end+1, :) = { ...
    'wrap api: rel-per wrap silent below threshold', ...
    abs(cF - cS) < 1e-8};


% --- The wrap choice survives density transformations ------------------
%
%  Both transformations rebuild the density, and buildExpTens defaults
%  wrap to 'full-image', so a density built with 'single-image' silently
%  reverted --- changing the measure rather than the speed. It surfaces
%  only above the sigma/period threshold, where the two forms diverge,
%  which is why it went unnoticed: below it the two agree to the floor
%  and nothing looks wrong.

pW = {[1 5 9 2; 3 7 11 4]};
wW = {[1 1 0 1; 1 1 0 1]};
dW = buildExpTens(pW, wW, 1.2, 2, true, true, 12, true, ...
                  'wrap', 'single-image', 'verbose', false);

dWmat = internal.ensureExpTensExpensive(dW);
results(end+1, :) = { ...
    'wrap api: survives materialising the expensive fields', ...
    strcmp(char(dWmat.wrap{1}), 'single-image')};

dWpruned = internal.prunedExpTens(dW);
results(end+1, :) = { ...
    'wrap api: survives event pruning', ...
    strcmp(char(dWpruned.wrap{1}), 'single-image')};

dWboth = internal.ensureExpTensExpensive(internal.prunedExpTens(dW));
results(end+1, :) = { ...
    'wrap api: survives pruning then materialising', ...
    strcmp(char(dWboth.wrap{1}), 'single-image')};


clear wrapCleanup;   % restore warning state via onCleanup

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_wrap_api: %d passed, %d failed\n', nPass, nFail);
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
end

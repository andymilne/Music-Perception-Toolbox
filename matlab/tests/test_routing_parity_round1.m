%% test_routing_parity_round1.m — regressions from the routing-parity audit, round 1
%
%  Mirror of the Python tests/test_routing_parity_round1.py. Each block
%  pins one item of the audit's verified findings so that the two
%  implementations keep taking the same route to the same number for
%  the same input:
%
%    A-1 / B-1  the per-call truncationSigmas reaches the whole nested
%               path (measure rule, kernels, memo key);
%    B-9        the flat selector's wrap rule scans relative-periodic
%               attributes only;
%    B-10       a wrap disagreement between the two densities errors
%               (mpt:wrapMismatch) instead of silently reading densX;
%    B-11       the tau-grid node count follows the per-call width;
%    B-14       nested routes skip <X,X> under 'oneSidedDenom';
%    A-7 / B-8  the Möbius point evaluator's non-finite guard is shared
%               by every route (structural check only: no route can be
%               made to return a non-finite value on demand here);
%    A-9 / B-13 kernelPrecision is honoured on the Möbius and centres
%               evaluation routes, and forwarded on the entropy list form;
%    A-10 / A-11 / B-12  the list forms forward method and
%               truncationSigmas;
%    B-5        the factored MA evaluation route passes the wrap;
%    A-6 / B-3  the flat Möbius centres branch receives the per-call
%               width (value unchanged by construction; smoke check);
%    A-17       the sweep orbit route resolves an explicit Inf;
%    B-15       windowedTensorSimilarity honours truncationSigmas and
%               kernelPrecision.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_rp1
    cleanupDefaults_rp1 = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

rp1_P = 12.0;
rng(7, 'twister');
rp1_VX = sort(rp1_P * rand(9, 1));
rp1_VY = sort(rp1_P * rand(9, 1));
rp1_absWarn = warning('off', 'buildExpTens:absPerSingleImage');
rp1_warnCleanup = onCleanup(@() warning(rp1_absWarn)); %#ok<NASGU>

% --- A-1 / B-1: the nested measure rule reads the per-call width ---
% At sigma/P = 0.045 the threshold is 0.03 at ts = 6 and 0.05 at ts = 4,
% so a forced centres route is refused at the tighter width and
% honoured at the looser one.
dx = rp1Nested(rp1_VX, 0.045 * rp1_P, true, true, 'full-image', rp1_P);
dy = rp1Nested(rp1_VY, 0.045 * rp1_P, true, true, 'full-image', rp1_P);
results{end+1, 1} = 'parity round1: nested centres refused at ts=6 (sigma/P=0.045)'; %#ok<*SAGROW>
results{end, 2}   = throwsErrorWithId(@() cosSimExpTens(dx, dy, ...
    'method', 'centres', 'truncationSigmas', 6, 'verbose', false), ...
    'cosSimExpTens:centresUnavailable');
rp1_ok = false;
try
    v = cosSimExpTens(dx, dy, 'method', 'centres', 'truncationSigmas', 4, ...
                      'verbose', false);
    rp1_ok = isfinite(v);
catch
end
results{end+1, 1} = 'parity round1: nested centres honoured at ts=4 (sigma/P=0.045)';
results{end, 2}   = rp1_ok;

% --- A-1 / B-1: the contraction truncates at the per-call width and
%     keys its memo on it ---
for rp1_isPer = [false true]
    if rp1_isPer; rp1_sig = 1.0; else; rp1_sig = 0.4; end
    ax = rp1Nested(rp1_VX, rp1_sig, false, rp1_isPer, 'full-image', rp1_P);
    ay = rp1Nested(rp1_VY, rp1_sig, false, rp1_isPer, 'full-image', rp1_P);
    ref = cosSimExpTens(ax, ay, 'method', 'contract', 'verbose', false);
    [coarse, axOut] = cosSimExpTens(ax, ay, 'method', 'contract', ...
        'truncationSigmas', 2, 'verbose', false);
    keys = axOut.selfIP.keys;
    keyOk = ~isempty(keys) && all(cellfun(@(k) ~isempty(regexp(k, ...
        '^contract\|2\|', 'once')), keys));
    results{end+1, 1} = sprintf( ...
        'parity round1: nested contraction truncates at ts=2 (isPer=%d)', rp1_isPer);
    results{end, 2}   = abs(coarse - ref) > 1e-6;
    results{end+1, 1} = sprintf( ...
        'parity round1: nested memo keyed on ts=2 (isPer=%d)', rp1_isPer);
    results{end, 2}   = keyOk;
end

% --- A-1 / B-1: the multi-attribute nested path resolves the same width ---
mx = rp1NestedPlusR1(rp1_VX);
my = rp1NestedPlusR1(rp1_VY);
ref = cosSimExpTens(mx, my, 'method', 'contract', 'verbose', false);
[coarse, mxOut] = cosSimExpTens(mx, my, 'method', 'contract', ...
    'truncationSigmas', 2, 'verbose', false);
keys = mxOut.selfIP.keys;
results{end+1, 1} = 'parity round1: nested-MA contraction truncates at ts=2';
results{end, 2}   = abs(coarse - ref) > 1e-6;
results{end+1, 1} = 'parity round1: nested-MA memo keyed on ts=2';
results{end, 2}   = ~isempty(keys) && all(cellfun(@(k) ~isempty(regexp(k, ...
    '^contract_ma\|2\|', 'once')), keys));

% --- B-14: nested route skips <X,X> under 'oneSidedDenom' ---
ax = rp1Nested(rp1_VX, 0.4, false, false, 'full-image', rp1_P);
ay = rp1Nested(rp1_VY, 0.4, false, false, 'full-image', rp1_P);
[v, axOut, ayOut] = cosSimExpTens(ax, ay, 'method', 'contract', ...
    'normalize', 'oneSidedDenom', 'verbose', false);
b = cosSimExpTens(ax, ay, 'method', 'bulger', ...
    'normalize', 'oneSidedDenom', 'verbose', false);
results{end+1, 1} = 'parity round1: nested oneSidedDenom neither forms nor memoises <X,X>';
results{end, 2}   = isfinite(v) && isempty(axOut.selfIP.keys) ...
    && ~isempty(ayOut.selfIP.keys);
results{end+1, 1} = 'parity round1: nested oneSidedDenom agrees with bulger';
results{end, 2}   = abs(v - b) <= 1e-6 * max(abs(b), 1e-12);

% --- B-10: a wrap mismatch errors on the flat and nested paths ---
fx = rp1Flat(1, 0.2 * rp1_P, 2, false, true, 'full-image', rp1_P);
fy = rp1Flat(2, 0.2 * rp1_P, 2, false, true, 'single-image', rp1_P);
results{end+1, 1} = 'parity round1: flat wrap mismatch raises mpt:wrapMismatch';
results{end, 2}   = throwsErrorWithId(@() cosSimExpTens(fx, fy, ...
    'verbose', false), 'mpt:wrapMismatch');
nx = rp1Nested(rp1_VX, 0.2 * rp1_P, true, true, 'full-image', rp1_P);
ny = rp1Nested(rp1_VY, 0.2 * rp1_P, true, true, 'single-image', rp1_P);
results{end+1, 1} = 'parity round1: nested wrap mismatch raises mpt:wrapMismatch';
results{end, 2}   = throwsErrorWithId(@() cosSimExpTens(nx, ny, ...
    'method', 'contract', 'verbose', false), 'mpt:wrapMismatch');

% --- B-9: a rel-nonper attribute at the default wrap does not mix with
%     a rel-per single-image one ---
rp1_ok = false;
try
    chosen = internal.selectMaInnerProductMethod( ...
        [2 2], [6 6], 2, 2, 2, true, true, true, 0.2, 'auto', false, ...
        [true true], [200 200], [6 6], {'full-image', 'single-image'}, 6, ...
        [], [], [], [], [false true]);
    rp1_ok = strcmp(chosen, 'bulger');
catch
end
results{end+1, 1} = 'parity round1: rel-nonper default wrap does not mix with rel-per single-image';
results{end, 2}   = rp1_ok;

% --- B-11: the tau-grid node count follows the width ---
results{end+1, 1} = 'parity round1: autoNtauDefault follows the per-call width';
results{end, 2}   = internal.autoNtauDefault(rp1_P, 0.5, 3) ...
    < internal.autoNtauDefault(rp1_P, 0.5, 8) ...
    && internal.autoNtauDefault(rp1_P, 0.5) == ...
       internal.autoNtauDefault(rp1_P, 0.5, mptDefaults('truncationSigmas'));

% --- A-7 / B-8: the shared non-finite guard leaves a healthy Möbius
%     evaluation untouched, with the guard on and off ---
rng(11, 'twister');
pA = {sort(rp1_P * rand(6, 1), 1), sort(rp1_P * rand(6, 1), 1)};
dMA = buildExpTens(pA, [], [0.8 0.8], [2 2], [false false], ...
                   [false false], [0 0], 'verbose', false);
XqA = rp1_P * rand(4, 7);
refC = evalExpTens(dMA, XqA, 'method', 'centres', 'verbose', false);
phgPrev = mptDefaults('postHocGuards');
mobOn = evalExpTens(dMA, XqA, 'method', 'mobius', 'verbose', false);
mptDefaults('postHocGuards', false);
mobOff = evalExpTens(dMA, XqA, 'method', 'mobius', 'verbose', false);
mptDefaults('postHocGuards', phgPrev);
results{end+1, 1} = 'parity round1: MA mobius eval agrees with centres with the guard on and off';
results{end, 2}   = all(isfinite(mobOn)) && all(isfinite(mobOff)) ...
    && max(abs(mobOn - refC)) <= 1e-8 * max(max(abs(refC)), 1e-12) ...
    && isequal(mobOn, mobOff);

% --- A-9 / B-13: single precision honoured on both eval routes ---
dS = rp1Flat(5, 0.6, 2, false, false, 'full-image', 0, 8, 3);
rng(3, 'twister');
XqS = rp1_P * rand(2, 64);
for rp1_m = {'centres', 'mobius'}
    dbl = evalExpTens(dS, XqS, 'method', rp1_m{1}, 'verbose', false);
    sgl = evalExpTens(dS, XqS, 'method', rp1_m{1}, ...
                      'kernelPrecision', 'single', 'verbose', false);
    results{end+1, 1} = sprintf( ...
        'parity round1: eval kernelPrecision=single honoured (%s)', rp1_m{1});
    results{end, 2}   = max(abs(sgl - dbl)) <= 1e-4 * max(abs(dbl)) ...
        && max(abs(sgl - dbl)) > 0;
end

% --- A-9: entropy list form forwards the width (relative density, grid) ---
dR = rp1Flat(4, 0.5, 2, true, false, 'full-image', 0, 5, 2);
hScalar = entropyExpTens(dR, 'method', 'shannon', 'nPointsPerDim', 64, ...
    'xMin', -6, 'xMax', 6, 'truncationSigmas', 1.5, 'verbose', false);
hList = entropyExpTens({dR}, 'method', 'shannon', 'nPointsPerDim', 64, ...
    'xMin', -6, 'xMax', 6, 'truncationSigmas', 1.5, 'verbose', false);
hDefault = entropyExpTens(dR, 'method', 'shannon', 'nPointsPerDim', 64, ...
    'xMin', -6, 'xMax', 6, 'verbose', false);
if iscell(hList); hList = hList{1}; end
results{end+1, 1} = 'parity round1: entropy list form forwards truncationSigmas';
results{end, 2}   = abs(hList - hScalar) <= 1e-12 * max(abs(hScalar), 1) ...
    && abs(hScalar - hDefault) > 1e-6;

% --- A-10 / B-12: cosine list forms forward method and truncationSigmas ---
ax = rp1Nested(rp1_VX, 0.4, false, false, 'full-image', rp1_P);
ay = rp1Nested(rp1_VY, 0.4, false, false, 'full-image', rp1_P);
sScalar = cosSimExpTens(ax, ay, 'method', 'contract', ...
    'truncationSigmas', 2, 'verbose', false);
sDefault = cosSimExpTens(ax, ay, 'verbose', false);
sPair = cosSimExpTens({ax}, {ay}, 'method', 'contract', ...
    'truncationSigmas', 2, 'verbose', false);
sBroad = cosSimExpTens(ax, {ay}, 'method', 'contract', ...
    'truncationSigmas', 2, 'verbose', false);
results{end+1, 1} = 'parity round1: (cell,cell) list form forwards method and truncationSigmas';
results{end, 2}   = abs(sPair{1} - sScalar) <= 1e-12 ...
    && abs(sScalar - sDefault) > 1e-6;
results{end+1, 1} = 'parity round1: scalar-vs-cell list form forwards method and truncationSigmas';
results{end, 2}   = abs(sBroad{1} - sScalar) <= 1e-12;

% --- A-10: the r = 1 broadcast fast path is gated on method ---
rng(21, 'twister');
r1x = buildExpTens({sort(rp1_P * rand(5, 2), 1)}, [], 0.5, 1, false, ...
                   false, 0, 'verbose', false);
r1y = buildExpTens({sort(rp1_P * rand(5, 2), 1)}, [], 0.5, 1, false, ...
                   false, 0, 'verbose', false);
sAuto = cosSimExpTens(r1x, {r1y}, 'verbose', false);
sMob  = cosSimExpTens(r1x, {r1y}, 'method', 'mobius', 'verbose', false);
sRef  = cosSimExpTens(r1x, r1y, 'method', 'mobius', 'verbose', false);
results{end+1, 1} = 'parity round1: r = 1 broadcast honours method=mobius';
results{end, 2}   = abs(sMob{1} - sRef) <= 1e-12 ...
    && abs(sAuto{1} - sRef) <= 1e-9 * max(abs(sRef), 1);

% --- A-11: evalExpTens list form forwards method ---
vList = evalExpTens({dS}, XqS, 'method', 'mobius', 'verbose', false);
vRef  = evalExpTens(dS, XqS, 'method', 'mobius', 'verbose', false);
results{end+1, 1} = 'parity round1: evalExpTens list form forwards method';
results{end, 2}   = isequal(vList{1}, vRef);

% --- B-5: the factored MA evaluation route honours the wrap ---
outs = struct();
for rp1_w = {'full-image', 'single-image'}
    dW = rp1TwoAbsPer(rp1_w{1}, 21, rp1_P);
    rng(9, 'twister');
    XqW = rp1_P * rand(4, 12);
    outs.(strrep(rp1_w{1}, '-', '_')) = evalExpTens(dW, XqW, ...
        'method', 'centres', 'verbose', false);
end
results{end+1, 1} = 'parity round1: factored eval route: the two wraps differ at sigma/P=0.2';
results{end, 2}   = max(abs(outs.full_image - outs.single_image)) > 1e-6;

% --- A-6 / B-3: flat Möbius centres branch receives the per-call
%     width (relative attribute: value ts-independent by construction) ---
relX = rp1Flat(31, 0.5, 2, true, false, 'full-image', 0, 6, 2);
relY = rp1Flat(32, 0.5, 2, true, false, 'full-image', 0, 6, 2);
relAttrPrev = mptDefaults('relAttrRoute');
mptDefaults('relAttrRoute', 'centres');
sC6 = cosSimExpTens(relX, relY, 'method', 'mobius', ...
    'truncationSigmas', 6, 'verbose', false);
sC4 = cosSimExpTens(relX, relY, 'method', 'mobius', ...
    'truncationSigmas', 4, 'verbose', false);
mptDefaults('relAttrRoute', relAttrPrev);
results{end+1, 1} = 'parity round1: flat Möbius centres branch accepts the per-call width';
results{end, 2}   = isfinite(sC6) && isfinite(sC4);

% --- A-17: the sweep orbit route resolves an explicit Inf ---
rng(31, 'twister');
sx = buildExpTens({randn(4, 5) * 3}, [], 0.9, 2, false, false, NaN, true, ...
                  'verbose', false);
sy = buildExpTens({randn(4, 3) * 3}, [], 0.9, 2, false, false, NaN, true, ...
                  'verbose', false);
offS = [-2 0 1.5];
swInf = sweepCosSimExpTens(sx, sy, offS, 'method', 'orbit', ...
    'truncationSigmas', Inf, 'verbose', false);
swFloor = sweepCosSimExpTens(sx, sy, offS, 'method', 'orbit', ...
    'truncationSigmas', internal.accuracyFloor('sigmas'), 'verbose', false);
results{end+1, 1} = 'parity round1: sweep orbit route resolves an explicit Inf';
results{end, 2}   = isequal(swInf, swFloor);

% --- B-15: windowedTensorSimilarity honours truncationSigmas and
%     kernelPrecision ---
rng(41, 'twister');
ctx = buildExpTens({sort(40 * rand(6, 4), 1)}, [], 1.5, 2, false, false, ...
                   0, 'verbose', false);
qry = buildExpTens({sort(10 + 10 * rand(4, 2), 1)}, [], 1.5, 2, false, ...
                   false, 0, 'verbose', false);
wSpec = struct('size', 4.0, 'mix', 0.0);
wOff = repmat(linspace(-10, 10, 9), 2, 1);
wRef = windowedTensorSimilarity(ctx, qry, wSpec, wOff, 'verbose', false);
wCoarse = windowedTensorSimilarity(ctx, qry, wSpec, wOff, ...
    'truncationSigmas', 1.0, 'verbose', false);
wSingle = windowedTensorSimilarity(ctx, qry, wSpec, wOff, ...
    'kernelPrecision', 'single', 'verbose', false);
results{end+1, 1} = 'parity round1: windowedTensorSimilarity truncates at the per-call width';
results{end, 2}   = isequal(size(wCoarse), size(wRef)) ...
    && max(abs(wCoarse - wRef)) > 1e-6;
results{end+1, 1} = 'parity round1: windowedTensorSimilarity honours kernelPrecision=single';
results{end, 2}   = max(abs(wSingle - wRef)) <= 1e-4 * max(abs(wRef)) ...
    && max(abs(wSingle - wRef)) > 0;

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_routing_parity_round1: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_rp1 rp1_warnCleanup
    if nFail > 0
        error('test_routing_parity_round1:failed', '%d test(s) failed.', nFail);
    end
end


function d = rp1Nested(values, sigma, isRel, isPer, wrap, P)
    % One nested attribute: two levels of r = 2, symmetric, chord tags of
    % three values (twin of the Python _nested helper).
    v = double(values(:));
    tags = repelem(0:(numel(v) / 3 - 1), 3);
    if isRel
        relSpec = [0 1];
    else
        relSpec = [0 0];
    end
    spec = struct('tags', tags, 'r', [2 2], 'sym', [true true], ...
                  'rel', relSpec);
    if isPer
        period = P;
    else
        period = 0;
    end
    d = buildExpTens({v}, {[]}, 'specs', {spec}, 'sigma', sigma, ...
                     'isPer', isPer, 'period', period, 'wrap', {wrap}, ...
                     'verbose', false);
end


function d = rp1NestedPlusR1(values)
    % An absolute nested attribute tensored with an r = 1 absolute one.
    v = double(values(:));
    tags = repelem(0:2, 3);
    sp0 = struct('tags', tags, 'r', [2 2], 'sym', [true true], 'rel', [0 0]);
    sp1 = struct('r', 1, 'sym', true, 'rel', false);
    d = buildExpTens({v, 3.0}, {[], []}, 'specs', {sp0, sp1}, ...
                     'sigma', [0.4 1.0], 'isPer', [false false], ...
                     'period', [0 0], 'verbose', false);
end


function d = rp1Flat(seed, sigma, r, isRel, isPer, wrap, P, K, N)
    if nargin < 8; K = 5; end
    if nargin < 9; N = 2; end
    rng(seed, 'twister');
    p = sort(12.0 * rand(K, N), 1);
    if isPer
        period = P;
    else
        period = 0;
    end
    d = buildExpTens({p}, {[]}, sigma, r, isRel, isPer, period, ...
                     'wrap', {wrap}, 'verbose', false);
end


function d = rp1TwoAbsPer(wrap, seed, P)
    rng(seed, 'twister');
    p = {sort(P * rand(4, 2), 1), sort(P * rand(4, 2), 1)};
    d = buildExpTens(p, {[], []}, [0.2 * P, 0.2 * P], [2 2], ...
                     [false false], [true true], [P P], ...
                     'wrap', {wrap, wrap}, 'verbose', false);
end

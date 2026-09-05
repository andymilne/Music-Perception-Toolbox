%% test_routing_parity_round2.m — regressions from the routing-parity audit, round 2
%
%  Mirror of the Python tests/test_routing_parity_round2.py. Each block
%  pins one item of the audit's verified findings so that the two
%  implementations keep taking the same route to the same number for
%  the same input:
%
%    A-5 / B-7  Rényi-2 entropy of a relative attribute at r = 1 is 0
%               by convention (unit overlap, unit mass), owned by the
%               general per-attribute loop so the single-multiset
%               corner inherits it;
%    A-8        the batched-raw form whitens a kernel covariance at its
%               entry and continues (the only reachable outcome is the
%               ordered-at-r > 1 refusal, as in Python);
%    A-12       the flat selector's working-set guard;
%    A-13       'factored' is not an accepted method (the set is shared);
%    A-14       the raw-MA scalar-vs-list form honours a forced method
%               and its r = 1 fast path returns the per-pair numbers;
%    A-15       the single-attribute helper route agrees with the
%               log-kernel core within the floor and honours
%               kernelPrecision;
%    A-16       the sweep self-IP memo, the nested centres bundle cache,
%               and the density-list dedup (keyed on the wrap too);
%    B-6        Shannon / normalized cell masses on an absolute-periodic
%               axis follow the declared wrap and the resolved width.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_rp2
    cleanupDefaults_rp2 = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

rp2_P = 12.0;
rp2_absWarn = warning('off', 'buildExpTens:absPerSingleImage');
rp2_relWarn = warning('off', 'buildExpTens:isRelDegenerate');
rp2_warnCleanup = onCleanup(@() cellfun(@warning, ...
    {rp2_absWarn, rp2_relWarn})); %#ok<NASGU>

% --- A-5 / B-7: relative r = 1 is 0 at the corner and for N > 1 ---
dCorner = rp2Chord([0 4 7], 1.0, 1, true, false, 'full-image', rp2_P);
results{end+1, 1} = 'parity round2: renyi2 relative r=1 is 0 at the single-multiset corner'; %#ok<*SAGROW>
results{end, 2}   = isequal(entropyExpTens(dCorner, 'method', 'renyi2', ...
    'verbose', false), 0);
dMany = rp2Flat(3, 1.0, 1, true, false, 'full-image', 0, 4, 3);
results{end+1, 1} = 'parity round2: renyi2 relative r=1 is 0 for many events';
results{end, 2}   = isequal(entropyExpTens(dMany, 'method', 'renyi2', ...
    'verbose', false), 0);

% --- A-5 / B-7: a relative r = 1 attribute contributes no entropy ---
rng(5, 'twister');
pAbs = sort(rp2_P * rand(5, 2), 1);
pRel = sort(rp2_P * rand(4, 2), 1);
dAbs = buildExpTens({pAbs}, {[]}, 0.7, 2, false, false, 0, 'verbose', false);
dBoth = buildExpTens({pAbs, pRel}, {[], []}, [0.7 1.0], [2 1], ...
                     [false true], [false false], [0 0], 'verbose', false);
hAbs = entropyExpTens(dAbs, 'method', 'renyi2', 'verbose', false);
hBoth = entropyExpTens(dBoth, 'method', 'renyi2', 'verbose', false);
results{end+1, 1} = 'parity round2: renyi2 relative r=1 attribute contributes no entropy';
results{end, 2}   = isfinite(hAbs) ...
    && abs(hBoth - hAbs) <= 1e-12 * max(abs(hAbs), 1);

% --- A-8: batched-raw kernel covariance is whitened, then the ordered
%     refusal (not batchedUnsupported) is what stops it ---
S2 = [1.0 0.3; 0.3 1.0];
P1b = [0 4; 0 7];
P2b = [0 3; 0 5];
results{end+1, 1} = 'parity round2: batched-raw kernel cov whitens then refuses ordered r=2';
results{end, 2}   = throwsErrorWithId(@() cosSimExpTens(P1b, [], P2b, [], ...
    S2, 2, false, false, 0, false, 'verbose', false), ...
    'cosSimExpTens:batchedOrderedUnsupported');
results{end+1, 1} = 'parity round2: batched-raw malformed kernel cov caught by whitening';
results{end, 2}   = throwsErrorWithId(@() cosSimExpTens(P1b, [], P2b, [], ...
    [1.0 0.5; 0.0 1.0], 2, false, false, 0, false, 'verbose', false), ...
    'mpt:aniso:notSymmetric');

% --- A-12: the working-set guard routes to Möbius ---
% n_J = N * r! * C(K, r): K = 60, r = 3 gives 205 320 tuples per event;
% 100 events at 2 * 3 * 8 bytes each is ~985 MB > 256 MB.
[chosenBig, pwBig, orbBig] = internal.selectMaInnerProductMethod( ...
    3, 60, 1, 100, 100, false, false, false, 0, 'auto', false, ...
    false, [], [], {}, 6, [], [], [], [], false);
[~, pwSmall, orbSmall] = internal.selectMaInnerProductMethod( ...
    3, 60, 1, 5, 5, false, false, false, 0, 'auto', false, ...
    false, [], [], {}, 6, [], [], [], [], false);
chosenWrap = internal.selectMaInnerProductMethod( ...
    3, 60, 1, 100, 100, true, false, true, 0.2, 'auto', false, ...
    true, [], [], {'single-image'}, 6, [], [], [], [], true);
results{end+1, 1} = 'parity round2: selector working-set guard routes to mobius unpriced';
results{end, 2}   = strcmp(chosenBig, 'mobius') && isnan(pwBig) && isnan(orbBig);
results{end+1, 1} = 'parity round2: selector prices the same shape below the budget';
results{end, 2}   = isfinite(pwSmall) && isfinite(orbSmall);
results{end+1, 1} = 'parity round2: selector guard yields to the rel-per wrap rule';
results{end, 2}   = strcmp(chosenWrap, 'bulger');

% --- A-13: 'factored' is not an accepted method ---
fx = rp2Flat(1, 0.6, 2, false, false, 'full-image', 0);
fy = rp2Flat(2, 0.6, 2, false, false, 'full-image', 0);
results{end+1, 1} = 'parity round2: method=factored is rejected (shared method set)';
results{end, 2}   = throwsErrorWithId(@() cosSimExpTens(fx, fy, ...
    'method', 'factored', 'verbose', false), 'cosSimExpTens:badMethod');

% --- A-14: raw-MA scalar-vs-list: the r = 1 fast path returns the
%     per-pair numbers and a forced method is honoured ---
rng(11, 'twister');
refP = {sort(rp2_P * rand(5, 2), 1)};
lstP = cell(1, 4);
for m = 1:4
    lstP{m} = {sort(rp2_P * rand(5, 2), 1)};
end
sFast = cosSimExpTens(refP, [], lstP, [], 0.5, 1, false, false, 0, ...
                      'verbose', false);
sRev = cosSimExpTens(lstP, [], refP, [], 0.5, 1, false, false, 0, ...
                     'verbose', false);
sMob = cosSimExpTens(refP, [], lstP, [], 0.5, 1, false, false, 0, ...
                     'method', 'mobius', 'verbose', false);
sPair = zeros(1, 4);
for m = 1:4
    sPair(m) = cosSimExpTens(refP, [], lstP{m}, [], 0.5, 1, false, ...
                             false, 0, 'method', 'mobius', 'verbose', false);
end
results{end+1, 1} = 'parity round2: raw-MA r=1 list fast path matches the per-pair numbers';
results{end, 2}   = numel(sFast) == 4 ...
    && max(abs([sFast{:}] - sPair)) <= 1e-9 ...
    && max(abs([sRev{:}] - sPair)) <= 1e-9;
results{end+1, 1} = 'parity round2: raw-MA list form honours method=mobius';
results{end, 2}   = max(abs([sMob{:}] - sPair)) <= 1e-12;

% --- A-15: the single-attribute helper route agrees with the
%     log-kernel core within the floor. A second attribute holding one
%     shared value per event multiplies every kernel entry by exactly
%     1 and leaves the tuple counts unchanged, so the two-attribute
%     density is the same inner product through the log-kernel form.
rp2_shapes = {
    {2, false, false, 'full-image', 0.7}
    {3, false, false, 'full-image', 0.9}
    {2, false, true,  'full-image', 0.15 * rp2_P}
    {2, false, true,  'single-image', 0.15 * rp2_P}
    {2, false, true,  'full-image', 0.03 * rp2_P}
    {2, true,  false, 'full-image', 0.6}
    {1, false, true,  'full-image', 0.2 * rp2_P}};
for rp2_i = 1:numel(rp2_shapes)
    sh = rp2_shapes{rp2_i};
    [rA, relA, perA, wrapA, sigA] = sh{:};
    if perA; perVal = rp2_P; else; perVal = 0; end
    rng(20 + rp2_i, 'twister');
    px = sort(rp2_P * rand(6, 2), 1);
    py = sort(rp2_P * rand(5, 2), 1);
    dx1 = buildExpTens({px}, {[]}, sigA, rA, relA, perA, perVal, ...
                       'wrap', {wrapA}, 'verbose', false);
    dy1 = buildExpTens({py}, {[]}, sigA, rA, relA, perA, perVal, ...
                       'wrap', {wrapA}, 'verbose', false);
    dx2 = buildExpTens({px, 3.0 * ones(1, 2)}, {[], []}, [sigA 1.0], ...
                       [rA 1], [relA false], [perA false], [perVal 0], ...
                       'wrap', {wrapA, 'full-image'}, 'verbose', false);
    dy2 = buildExpTens({py, 3.0 * ones(1, 2)}, {[], []}, [sigA 1.0], ...
                       [rA 1], [relA false], [perA false], [perVal 0], ...
                       'wrap', {wrapA, 'full-image'}, 'verbose', false);
    for rp2_ts = {4, Inf}
        s1 = cosSimExpTens(dx1, dy1, 'method', 'bulger', ...
            'truncationSigmas', rp2_ts{1}, 'verbose', false);
        s2 = cosSimExpTens(dx2, dy2, 'method', 'bulger', ...
            'truncationSigmas', rp2_ts{1}, 'verbose', false);
        results{end+1, 1} = sprintf( ...
            ['parity round2: helper route agrees with log-kernel core ' ...
             '(r=%d rel=%d per=%d %s ts=%g)'], rA, relA, perA, wrapA, rp2_ts{1});
        results{end, 2} = abs(s1 - s2) <= 10 * max( ...
            internal.truncationFloor(rp2_ts{1}), 1e-12) * max(abs(s2), 1e-12);
    end
end

% --- A-15: kernelPrecision honoured on the single-attribute cosine ---
kx = rp2Flat(31, 0.6, 2, false, false, 'full-image', 0, 8, 2);
ky = rp2Flat(32, 0.6, 2, false, false, 'full-image', 0, 8, 2);
sDbl = cosSimExpTens(kx, ky, 'method', 'bulger', 'verbose', false);
sSgl = cosSimExpTens(kx, ky, 'method', 'bulger', ...
                     'kernelPrecision', 'single', 'verbose', false);
results{end+1, 1} = 'parity round2: single-attribute cosine honours kernelPrecision=single';
results{end, 2}   = abs(sSgl - sDbl) > 0 && abs(sSgl - sDbl) <= 1e-4 * abs(sDbl);
[~, kxS, ~] = cosSimExpTens(kx, ky, 'method', 'bulger', ...
                            'kernelPrecision', 'single', 'verbose', false);
[~, kxD, ~] = cosSimExpTens(kx, ky, 'method', 'bulger', 'verbose', false);
results{end+1, 1} = 'parity round2: single-precision self IP is keyed apart from double';
results{end, 2}   = ~isequal(kxS.selfIP.keys, kxD.selfIP.keys) ...
    && any(cellfun(@(k) contains(k, '|single'), kxS.selfIP.keys));

% --- A-16: sweep self-IP memo under the 'sweep' key ---
rng(41, 'twister');
sx = buildExpTens({randn(4, 5) * 3}, [], 0.9, 2, false, false, NaN, true, ...
                  'verbose', false);
sy = buildExpTens({randn(4, 3) * 3}, [], 0.9, 2, false, false, NaN, true, ...
                  'verbose', false);
offS = [-2 0 1.5];
[sw1, sxOut, syOut] = sweepCosSimExpTens(sx, sy, offS, 'method', 'mixture', ...
                                         'verbose', false);
swKeys = syOut.selfIP.keys;
sw2 = sweepCosSimExpTens(sxOut, syOut, offS, 'method', 'mixture', ...
                         'verbose', false);
results{end+1, 1} = 'parity round2: sweep memoises the self IP under the sweep key';
results{end, 2}   = numel(swKeys) == 1 ...
    && strncmp(swKeys{1}, 'sweep|', 6) ...
    && numel(sxOut.selfIP.keys) == 1 ...
    && ~internal.selfIpMemoised(syOut.selfIP) ...
    && isequal(sw1, sw2);

% --- A-16: nested centres bundle memoised on the density struct ---
rng(7, 'twister');
nx = rp2Nested(sort(rp2_P * rand(9, 1)), 0.5, rp2_P);
ny = rp2Nested(sort(rp2_P * rand(9, 1)), 0.5, rp2_P);
[sN1, nxOut, nyOut] = cosSimExpTens(nx, ny, 'method', 'centres', ...
                                    'verbose', false);
sN2 = cosSimExpTens(nxOut, nyOut, 'method', 'centres', 'verbose', false);
results{end+1, 1} = 'parity round2: nested centres bundle cached in selfIP.nestedCentres';
results{end, 2}   = isfield(nxOut.selfIP, 'nestedCentres') ...
    && ~isempty(nxOut.selfIP.nestedCentres{1}) ...
    && isstruct(nxOut.selfIP.nestedCentres{1}) ...
    && isfield(nyOut.selfIP, 'nestedCentres') ...
    && abs(sN1 - sN2) <= 1e-12;

% --- A-16: (cell,cell) dedup keeps pairs apart on the wrap ---
sigW = 0.3 * rp2_P;
xf = rp2Chord([0 4 7], sigW, 2, false, true, 'full-image', rp2_P);
yf = rp2Chord([0 3 7], sigW, 2, false, true, 'full-image', rp2_P);
xs = rp2Chord([0 4 7], sigW, 2, false, true, 'single-image', rp2_P);
ys = rp2Chord([0 3 7], sigW, 2, false, true, 'single-image', rp2_P);
sFull = cosSimExpTens(xf, yf, 'verbose', false);
sSingle = cosSimExpTens(xs, ys, 'verbose', false);
outL = cosSimExpTens({xf, xs, xf}, {yf, ys, yf}, 'verbose', false);
results{end+1, 1} = 'parity round2: density-list dedup keys on the wrap';
results{end, 2}   = abs(sFull - sSingle) > 1e-6 ...
    && abs(outL{1} - sFull) <= 1e-12 ...
    && abs(outL{2} - sSingle) <= 1e-12 ...
    && isequal(outL{1}, outL{3});

% --- B-6: full-image cell masses are those of the wrapped Gaussian ---
% Reference: the wrapped-Gaussian pmf on the same 64 cells, from a
% 101-image erf sum; it is normalised on the circle, so its masses sum
% to the total weight, and the Shannon entropy entropyExpTens reports
% must be the entropy of exactly this pmf.
sigB = 0.3 * rp2_P;
chordB = [0 4 7];
nCells = 64;
edges = linspace(0, rp2_P, nCells + 1);
lo = edges(1:end-1) - rp2_P / (2 * nCells);
hi = lo + rp2_P / nCells;
massRef = zeros(1, nCells);
for c = chordB
    for n = -50:50
        massRef = massRef + 0.5 * (erf((hi - c + n * rp2_P) / (sigB * sqrt(2))) ...
                               - erf((lo - c + n * rp2_P) / (sigB * sqrt(2))));
    end
end
massSingle = zeros(1, nCells);
for c = chordB
    a = lo - c; a = a - rp2_P * round(a / rp2_P);
    b = hi - c; b = b - rp2_P * round(b / rp2_P);
    massSingle = massSingle + 0.5 * (erf(b / (sigB * sqrt(2))) ...
        - erf(a / (sigB * sqrt(2)))) ...
        + (a > b) .* erf((0.5 * rp2_P) / (sigB * sqrt(2)));
end
qRef = massRef / sum(massRef);
hRef = -sum(qRef .* log2(qRef));
dFull = rp2Chord(chordB, sigB, 1, false, true, 'full-image', rp2_P);
dSingle = rp2Chord(chordB, sigB, 1, false, true, 'single-image', rp2_P);
hFull = entropyExpTens(dFull, 'method', 'shannon', 'nPointsPerDim', ...
    nCells, 'truncationSigmas', Inf, 'verbose', false);
hSingle = entropyExpTens(dSingle, 'method', 'shannon', 'nPointsPerDim', ...
    nCells, 'truncationSigmas', Inf, 'verbose', false);
results{end+1, 1} = 'parity round2: full-image cell masses sum to the total mass (1e-10)';
results{end, 2}   = abs(sum(massRef) - numel(chordB)) <= 1e-10 ...
    && abs(sum(massSingle) - numel(chordB)) > 1e-3;
results{end+1, 1} = 'parity round2: shannon full-image entropy is that of the wrapped pmf';
results{end, 2}   = abs(hFull - hRef) <= 1e-10;
results{end+1, 1} = 'parity round2: shannon full-image and single-image differ at sigma/P=0.3';
results{end, 2}   = abs(hFull - hSingle) > 1e-6;

% --- B-6: the two readings coincide below the overlap regime, and the
%     per-call width governs the image count ---
sigS = 0.02 * rp2_P;
hFullS = entropyExpTens(rp2Chord(chordB, sigS, 1, false, true, ...
    'full-image', rp2_P), 'method', 'shannon', 'nPointsPerDim', nCells, ...
    'verbose', false);
hSingleS = entropyExpTens(rp2Chord(chordB, sigS, 1, false, true, ...
    'single-image', rp2_P), 'method', 'shannon', 'nPointsPerDim', nCells, ...
    'verbose', false);
results{end+1, 1} = 'parity round2: shannon wraps coincide at sigma/P=0.02';
results{end, 2}   = abs(hFullS - hSingleS) <= 1e-9;
hNarrow = entropyExpTens(dFull, 'method', 'shannon', 'nPointsPerDim', ...
    nCells, 'truncationSigmas', 1.0, 'verbose', false);
results{end+1, 1} = 'parity round2: shannon full-image image count follows truncationSigmas';
results{end, 2}   = abs(hNarrow - hFull) > 1e-6;

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
    fprintf('\n=== test_routing_parity_round2: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_rp2 rp2_warnCleanup
    if nFail > 0
        error('test_routing_parity_round2:failed', '%d test(s) failed.', nFail);
    end
end


function d = rp2Flat(seed, sigma, r, isRel, isPer, wrap, P, K, N)
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


function d = rp2Chord(values, sigma, r, isRel, isPer, wrap, P)
    if isPer
        period = P;
    else
        period = 0;
    end
    d = buildExpTens({values(:)}, {[]}, sigma, r, isRel, isPer, period, ...
                     'wrap', {wrap}, 'verbose', false);
end


function d = rp2Nested(values, sigma, P) %#ok<INUSD>
    % One relative nested attribute: two levels of r = 2, symmetric,
    % chord tags of three values, non-periodic (twin of the Python
    % _nested helper), on which the centres route is admissible.
    v = double(values(:));
    tags = repelem(0:(numel(v) / 3 - 1), 3);
    spec = struct('tags', tags, 'r', [2 2], 'sym', [true true], ...
                  'rel', [0 1]);
    d = buildExpTens({v}, {[]}, 'specs', {spec}, 'sigma', sigma, ...
                     'isPer', false, 'period', 0, 'verbose', false);
end

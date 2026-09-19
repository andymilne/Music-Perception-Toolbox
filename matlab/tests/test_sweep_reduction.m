%% test_sweep_reduction.m — translation sweeps as a mixture in the offset
%
%  sweepSimMaet replaces one inner product per offset with a single
%  pass over the tuple pairs followed by M evaluations of a Gaussian
%  mixture in the offset. Every test here compares it against the
%  per-offset path it replaces: translate the query explicitly, call
%  simMaet, and require agreement at the parity floor.
%
%  Reference route. The reduction reproduces the pairwise (Bulger) inner
%  product term for term. The orbit (Mobius) route computes the same
%  quantity through signed orbit weights whose cancellation is more
%  exposed to the truncation floor, so comparisons that must hold at the
%  parity floor pin 'method', 'bulger' on the reference.
%
%  Truncation. Both paths apply the same floor to the accumulated
%  log-kernel, and the reduction's up-front prune drops only components
%  that cannot rise above that floor at any offset, so agreement holds
%  with truncation active as well as disabled.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

tol = 1e-12;

% Offsets include an off-peak column as well as an exact-match column.
% The zero column matters: an exact match agrees under a wrong
% 2 sigma^2 denominator as readily as under the correct 4 sigma^2 one,
% because the numerator vanishes there. Only the off-peak columns
% separate them.
baseOff = [-3.1, -1.4, -0.35, 0.0, 0.8, 2.2, 5.0];


% --- Core parity across shapes -----------------------------------------

shapes = {[1 1], [3 3], [4 4], [3 2], [4 2], [5 3]};
for si = 1:numel(shapes)
    K = shapes{si}(1);
    r = shapes{si}(2);
    for isExch = [false true]
        rng(100 + 10 * K + r);
        A = 2;
        pX = {randn(K, 6) * 3, randn(K, 6) * 3};
        pY = {randn(K, 3) * 3, randn(K, 3) * 3};
        off = [baseOff; 0.5 * baseOff];
        sig = [0.9 0.9]; rv = [r r];
        z = [false false]; pd = [NaN NaN]; exch = [isExch isExch];

        dX = buildMaet(pX, [], sig, rv, z, z, pd, exch, 'verbose', false);
        dY = buildMaet(pY, [], sig, rv, z, z, pd, exch, 'verbose', false);
        got = sweepSimMaet(dX, dY, off, ...
            'truncationSigmas', Inf, 'verbose', false);

        ref = zeros(1, size(off, 2));
        for m = 1:size(off, 2)
            pYm = {pY{1} + off(1, m), pY{2} + off(2, m)};
            dYm = buildMaet(pYm, [], sig, rv, z, z, pd, exch, ...
                'verbose', false);
            ref(m) = simMaet(dX, dYm, 'method', 'bulger', ...
                'truncationSigmas', Inf, 'verbose', false);
        end
        dev = max(abs(got - ref) ./ max(abs(ref), 1e-12));
        results{end+1,1} = sprintf( ...
            'sweep: K=%d r=%d exch=%d matches per-offset', K, r, isExch); %#ok<*SAGROW>
        results{end,2} = dev <= tol;
    end
end


% --- Agreement under the default truncation floor -----------------------

rng(7);
pX = {randn(3, 6) * 3, randn(3, 6) * 3};
pY = {randn(3, 3) * 3, randn(3, 3) * 3};
off = [baseOff; 0.5 * baseOff];
sig = [0.9 0.9]; rv = [3 3]; z = [false false]; pd = [NaN NaN];
exch = [true true];
dX = buildMaet(pX, [], sig, rv, z, z, pd, exch, 'verbose', false);
dY = buildMaet(pY, [], sig, rv, z, z, pd, exch, 'verbose', false);
got = sweepSimMaet(dX, dY, off, 'verbose', false);
ref = zeros(1, size(off, 2));
for m = 1:size(off, 2)
    dYm = buildMaet({pY{1} + off(1, m), pY{2} + off(2, m)}, [], ...
        sig, rv, z, z, pd, exch, 'verbose', false);
    ref(m) = simMaet(dX, dYm, 'method', 'bulger', 'verbose', false);
end
results{end+1,1} = 'sweep: agrees with truncation active';
results{end,2} = max(abs(got - ref) ./ max(abs(ref), 1e-12)) <= tol;


% --- The 4 sigma^2 denominator is pinned by the off-peak offsets --------

zeroCol = find(off(1, :) == 0, 1);
offPeak = setdiff(1:size(off, 2), zeroCol);
results{end+1,1} = 'sweep: off-peak offsets agree (4 sigma^2 denominator)';
results{end,2} = max(abs(got(offPeak) - ref(offPeak)) ./ ...
                     max(abs(ref(offPeak)), 1e-12)) <= tol;


% --- oneSidedDenom ------------------------------------------------------

gotOne = sweepSimMaet(dX, dY, off, 'normalize', 'oneSidedDenom', ...
    'truncationSigmas', Inf, 'verbose', false);
refOne = zeros(1, size(off, 2));
for m = 1:size(off, 2)
    dYm = buildMaet({pY{1} + off(1, m), pY{2} + off(2, m)}, [], ...
        sig, rv, z, z, pd, exch, 'verbose', false);
    refOne(m) = simMaet(dX, dYm, 'normalize', 'oneSidedDenom', ...
        'method', 'bulger', 'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: oneSidedDenom matches per-offset';
results{end,2} = max(abs(gotOne - refOne) ./ max(abs(refOne), 1e-12)) <= tol;


% --- An unordered attribute uses the unnormalised permutation sum -------
%
%  A context tuple that is the query with its tuple positions permuted
%  matches only when the attribute is unordered, and then at the
%  unnormalised permutation sum rather than the permutation mean.

pYp = {[0; 4; 7]};
pXp = {pYp{1}([3 1 2], :)};
for isExch = [false true]
    dXp = buildMaet(pXp, [], 0.9, 3, false, false, NaN, isExch, ...
        'verbose', false);
    dYp = buildMaet(pYp, [], 0.9, 3, false, false, NaN, isExch, ...
        'verbose', false);
    g = sweepSimMaet(dXp, dYp, 0, 'truncationSigmas', Inf, ...
        'verbose', false);
    rr = simMaet(dXp, dYp, 'method', 'bulger', ...
        'truncationSigmas', Inf, 'verbose', false);
    results{end+1,1} = sprintf('sweep: permuted tuple, exch=%d', isExch);
    results{end,2} = abs(g - rr) <= tol;
    if isExch
        results{end+1,1} = 'sweep: unordered permuted tuple scores 1';
        results{end,2} = abs(g - 1) <= 1e-12;
    end
end


% --- An unswept relative attribute contributes the shape term alone -----

rng(31);
pXr = {randn(3, 6) * 3, randn(3, 6) * 3};
pYr = {randn(3, 3) * 3, randn(3, 3) * 3};
offR = [baseOff; zeros(1, numel(baseOff))];
relv = [false true];
dXr = buildMaet(pXr, [], sig, rv, relv, z, pd, exch, 'verbose', false);
dYr = buildMaet(pYr, [], sig, rv, relv, z, pd, exch, 'verbose', false);
gotR = sweepSimMaet(dXr, dYr, offR, 'truncationSigmas', Inf, ...
    'verbose', false);
refR = zeros(1, size(offR, 2));
for m = 1:size(offR, 2)
    dYm = buildMaet({pYr{1} + offR(1, m), pYr{2}}, [], sig, rv, ...
        relv, z, pd, exch, 'verbose', false);
    refR(m) = simMaet(dXr, dYm, 'method', 'bulger', ...
        'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: unswept relative attribute matches';
results{end,2} = max(abs(gotR - refR) ./ max(abs(refR), 1e-12)) <= tol;


% --- An unswept periodic attribute goes through the wrapped kernel ------

rng(41);
pXp2 = {randn(3, 6) * 3, mod(randn(3, 6) * 3, 12)};
pYp2 = {randn(3, 3) * 3, mod(randn(3, 3) * 3, 12)};
offP = [baseOff; zeros(1, numel(baseOff))];
perv = [false true]; pdv = [NaN 12];
dXp2 = buildMaet(pXp2, [], sig, rv, z, perv, pdv, exch, 'verbose', false);
dYp2 = buildMaet(pYp2, [], sig, rv, z, perv, pdv, exch, 'verbose', false);
gotP = sweepSimMaet(dXp2, dYp2, offP, 'truncationSigmas', Inf, ...
    'verbose', false);
refP = zeros(1, size(offP, 2));
for m = 1:size(offP, 2)
    dYm = buildMaet({pYp2{1} + offP(1, m), pYp2{2}}, [], ...
        sig, rv, z, perv, pdv, exch, 'verbose', false);
    refP(m) = simMaet(dXp2, dYm, 'method', 'bulger', ...
        'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: unswept periodic attribute matches';
results{end,2} = max(abs(gotP - refP) ./ max(abs(refP), 1e-12)) <= tol;


% --- Nested attributes --------------------------------------------------
%
%  A nested attribute at an inner or intermediate co-transposition unit
%  has a block-diagonal quadratic form: each block removes its own
%  all-ones. That is the relative case read per block, so it contributes
%  one shape term per block and no placement term, and like a relative
%  attribute it cannot be swept.

nestSpec = struct('tags', [0 0 1 1], 'r', [2 2], 'exch', [true false], ...
                  'rel', 'innermost');
absSpec  = struct('tags', [0 0 1 1], 'r', [2 2], 'exch', [true false]);

rng(750);
pXnest = sort(randn(4, 6) * 4, 1);
pYnest = sort(randn(4, 3) * 4, 1);
pXabs2 = randn(2, 6) * 3;
pYabs2 = randn(2, 3) * 3;
svN = [1.0 0.9]; rvN = [1 2]; zN = [false false]; pdN = [0 0];
exchN = [true true];
offN = [zeros(1, numel(baseOff)); baseOff];
dXnest = buildMaet({pXnest, pXabs2}, [], svN, rvN, zN, zN, pdN, exchN, ...
                      'nested', {nestSpec, []}, 'verbose', false);
dYnest = buildMaet({pYnest, pYabs2}, [], svN, rvN, zN, zN, pdN, exchN, ...
                      'nested', {nestSpec, []}, 'verbose', false);
gotN = sweepSimMaet(dXnest, dYnest, offN, ...
                          'truncationSigmas', Inf, 'verbose', false);
refN = zeros(1, size(offN, 2));
for m = 1:size(offN, 2)
    dYm = buildMaet({pYnest, pYabs2 + offN(2, m)}, [], svN, rvN, ...
                       zN, zN, pdN, exchN, 'nested', {nestSpec, []}, ...
                       'verbose', false);
    refN(m) = simMaet(dXnest, dYm, 'method', 'bulger', ...
                            'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: unswept nested inner unit matches per-offset';
results{end,2} = max(abs(gotN - refN)) <= tol;

% An absolute nested attribute is an absolute attribute, so it sweeps.
rng(760);
pXa = sort(randn(4, 6) * 4, 1);
pYa = sort(randn(4, 3) * 4, 1);
dXa = buildMaet({pXa}, [], 1.0, 1, false, false, 0, true, ...
                   'nested', {absSpec}, 'verbose', false);
dYa = buildMaet({pYa}, [], 1.0, 1, false, false, 0, true, ...
                   'nested', {absSpec}, 'verbose', false);
gotA = sweepSimMaet(dXa, dYa, baseOff, ...
                          'truncationSigmas', Inf, 'verbose', false);
refA = zeros(1, numel(baseOff));
for m = 1:numel(baseOff)
    dYm = buildMaet({pYa + baseOff(m)}, [], 1.0, 1, false, false, 0, ...
                       true, 'nested', {absSpec}, 'verbose', false);
    refA(m) = simMaet(dXa, dYm, 'method', 'bulger', ...
                            'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: absolute nested attribute sweeps';
results{end,2} = max(abs(gotA - refA)) <= tol;

ok = false;
try
    sweepSimMaet(dXnest, dYnest, ...
        [baseOff; zeros(1, numel(baseOff))], 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:sweptNested');
end
results{end+1,1} = 'sweep: swept nested inner unit is refused';
results{end,2} = ok;


% --- Relative-periodic, gated on sigma/P --------------------------------
%
%  Below the limit the wrapped-difference and transposition-average
%  measures agree inside the accuracy floor, so the attribute is
%  accepted; above it the default full-image reading is refused, and an
%  explicit single-image wrap names the measure this form computes.

rng(770);
pXrp = {randn(3, 8) * 20, mod(randn(3, 8) * 5, 12)};
pYrp = {randn(3, 3) * 20, mod(randn(3, 3) * 5, 12)};
rvRP = [3 3]; relRP = [false true]; perRP = [false true]; pdRP = [NaN 12];
exchRP = [true true];
offRP = [baseOff; zeros(1, numel(baseOff))];

% The pitch attribute's sigma is set wide enough that the profile is
% not identically zero: at sigma = 0.9 against a spread of 20 the two
% densities barely overlap and every offset returns 0, which no
% comparison can discriminate. At sigma = 5 the peak similarity is
% 4.8e-04 below the limit and 1.2e-01 above it.
svLow = [5 0.02 * 12];          % sigma/P = 0.02, inside the limit
dXlow = buildMaet(pXrp, [], svLow, rvRP, relRP, perRP, pdRP, exchRP, ...
                     'verbose', false);
dYlow = buildMaet(pYrp, [], svLow, rvRP, relRP, perRP, pdRP, exchRP, ...
                     'verbose', false);
gotRP = sweepSimMaet(dXlow, dYlow, offRP, ...
                           'truncationSigmas', Inf, 'verbose', false);
refRP = zeros(1, size(offRP, 2));
for m = 1:size(offRP, 2)
    dYm = buildMaet({pYrp{1} + offRP(1, m), pYrp{2}}, [], svLow, rvRP, ...
                       relRP, perRP, pdRP, exchRP, 'verbose', false);
    refRP(m) = simMaet(dXlow, dYm, 'method', 'bulger', ...
                             'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: rel-per below the sigma/P limit is accepted';
results{end,2} = max(abs(gotRP - refRP)) <= tol;
if ~results{end,2}
    fprintf('  [diagnostic] rel-per below limit: max abs deviation %.3e\n', ...
            max(abs(gotRP - refRP)));
end

svHigh = [5 0.10 * 12];         % sigma/P = 0.10, above the limit
dXhigh = buildMaet(pXrp, [], svHigh, rvRP, relRP, perRP, pdRP, exchRP, ...
                      'verbose', false);
dYhigh = buildMaet(pYrp, [], svHigh, rvRP, relRP, perRP, pdRP, exchRP, ...
                      'verbose', false);
ok = false;
try
    sweepSimMaet(dXhigh, dYhigh, offRP, 'method', 'mixture', ...
                       'truncationSigmas', Inf, 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:relativePeriodic');
end
results{end+1,1} = 'sweep: mixture refuses rel-per above the sigma/P limit';
results{end,2} = ok;

dXsi = buildMaet(pXrp, [], svHigh, rvRP, relRP, perRP, pdRP, exchRP, ...
                    'wrap', {'full-image', 'single-image'}, 'verbose', false);
dYsi = buildMaet(pYrp, [], svHigh, rvRP, relRP, perRP, pdRP, exchRP, ...
                    'wrap', {'full-image', 'single-image'}, 'verbose', false);
ok = true;
try
    gotSI = sweepSimMaet(dXsi, dYsi, offRP, ...
                               'truncationSigmas', Inf, 'verbose', false);
    refSI = zeros(1, size(offRP, 2));
    for m = 1:size(offRP, 2)
        dYm = buildMaet({pYrp{1} + offRP(1, m), pYrp{2}}, [], svHigh, ...
                           rvRP, relRP, perRP, pdRP, exchRP, ...
                           'wrap', {'full-image', 'single-image'}, ...
                           'verbose', false);
        refSI(m) = simMaet(dXsi, dYm, 'method', 'bulger', ...
                                 'truncationSigmas', Inf, 'verbose', false);
    end
    ok = max(abs(gotSI - refSI)) <= tol;
    if ~ok
        fprintf(['  [diagnostic] single-image wrap: max abs deviation ' ...
                 '%.3e\n'], max(abs(gotSI - refSI)));
    end
catch ME
    ok = false;
    fprintf('  [diagnostic] single-image wrap threw %s: %s\n', ...
            ME.identifier, ME.message);
end
results{end+1,1} = 'sweep: single-image wrap names the measure and is honoured';
results{end,2} = ok;


% --- Refusals -----------------------------------------------------------

dRel = buildMaet({randn(3, 4)}, [], 0.9, 3, true, false, NaN, true, ...
    'verbose', false);
ok = false;
try
    sweepSimMaet(dRel, dRel, [0 1.5], 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:sweptRelative');
end
results{end+1,1} = 'sweep: swept relative attribute is refused';
results{end,2} = ok;

dPer = buildMaet({randn(3, 4)}, [], 0.9, 3, false, true, 12, true, ...
    'verbose', false);
ok = false;
try
    sweepSimMaet(dPer, dPer, [0 1.5], 'method', 'mixture', ...
                       'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:periodicAttribute');
end
results{end+1,1} = 'sweep: mixture refuses a swept periodic attribute';
results{end,2} = ok;

% An unswept periodic attribute is accepted, so the refusal above is
% about the sweep, not about periodicity as such.
ok = true;
try
    sweepSimMaet(dPer, dPer, [0 0], 'verbose', false);
catch
    ok = false;
end
results{end+1,1} = 'sweep: unswept periodic attribute is accepted';
results{end,2} = ok;

dRP = buildMaet({randn(2, 4)}, [], 0.9, 2, true, true, 12, true, ...
    'verbose', false);
ok = false;
try
    sweepSimMaet(dRP, dRP, [0 0], 'method', 'mixture', ...
                       'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:relativePeriodic');
end
results{end+1,1} = 'sweep: mixture refuses relative-periodic even unswept';
results{end,2} = ok;

ok = false;
try
    sweepSimMaet(dX, dY, zeros(3, 4), 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:offsetsShape');
end
results{end+1,1} = 'sweep: offsets shape is validated';
results{end,2} = ok;


% --- translateAttributes carries its offsets ----------------------------

[pmSw, sw] = translateAttributes({zeros(2, 3), zeros(2, 3)}, [], ...
    {baseOff, 0.5 * baseOff});
pOut = pmSw.pAttr;
% offsets is A x M, so compare against the A x M matrix rather than a
% flattened vector: unrolling column-major interleaves the attributes,
% which has the same element count as the concatenation and so compares
% silently false rather than erroring.
expectedOff = [baseOff; 0.5 * baseOff];
results{end+1,1} = 'sweep: translateAttributes returns sweep offsets';
results{end,2} = iscell(pOut) && numel(pOut) == numel(baseOff) ...
    && isstruct(sw) && isfield(sw, 'offsets') ...
    && isequal(size(sw.offsets), size(expectedOff)) ...
    && max(abs(sw.offsets(:) - expectedOff(:))) < 1e-15 ...
    && isfield(sw, 'base') && iscell(sw.base) && numel(sw.base) == 2;

[~, swSingle] = translateAttributes({zeros(2, 3)}, [], {5});
results{end+1,1} = 'sweep: single translation carries no sweep struct';
results{end,2} = isempty(swSingle);

% The carried offsets drive the reduction directly.
rng(70);
pXt = {randn(3, 6) * 3, randn(3, 6) * 3};
pYt = {randn(3, 3) * 3, randn(3, 3) * 3};
[~, swT] = translateAttributes(pYt, [], {baseOff, 0.5 * baseOff});
dXt = buildMaet(pXt, [], sig, rv, z, z, pd, exch, 'verbose', false);
dYt = buildMaet(swT.base, [], sig, rv, z, z, pd, exch, 'verbose', false);
gotT = sweepSimMaet(dXt, dYt, swT.offsets, ...
    'truncationSigmas', Inf, 'verbose', false);
refT = zeros(1, numel(baseOff));
for m = 1:numel(baseOff)
    dYm = buildMaet({pYt{1} + swT.offsets(1, m), ...
                        pYt{2} + swT.offsets(2, m)}, [], ...
        sig, rv, z, z, pd, exch, 'verbose', false);
    refT(m) = simMaet(dXt, dYm, 'method', 'bulger', ...
        'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: carried offsets drive the reduction';
results{end,2} = max(abs(gotT - refT) ./ max(abs(refT), 1e-12)) <= tol;


% --- Orbit route --------------------------------------------------------
%
%  The orbit route evaluates the Mobius decomposition at the shifted
%  values instead of forming the placement/shape split, so its cost
%  scales with the orbit count rather than with the tuple-pair count.
%  It is a different decomposition of the same quantity, so it is
%  compared against the orbit reference at the parity floor.
%
%  Deviations are judged in absolute terms: a cosine lives in [-1, 1],
%  and at offsets where the two densities barely overlap its value falls
%  to 1e-16, where a ratio to that value reports noise rather than error.

orbShapes = {[3 2], [4 3], [4 4]};
orbOff = [-2.6, -0.9, 0.0, 1.3, 3.4];
for si = 1:numel(orbShapes)
    K = orbShapes{si}(1);
    r = orbShapes{si}(2);
    rng(800 + si);
    pXo = {randn(K, 5) * 3};
    pYo = {randn(K, 3) * 3};
    dXo = buildMaet(pXo, [], 0.9, r, false, false, NaN, true, ...
                       'verbose', false);
    dYo = buildMaet(pYo, [], 0.9, r, false, false, NaN, true, ...
                       'verbose', false);
    gotO = sweepSimMaet(dXo, dYo, orbOff, 'method', 'orbit', ...
                              'truncationSigmas', Inf, 'verbose', false);
    refO = zeros(1, numel(orbOff));
    for m = 1:numel(orbOff)
        dYm = buildMaet({pYo{1} + orbOff(m)}, [], 0.9, r, false, ...
                           false, NaN, true, 'verbose', false);
        refO(m) = simMaet(dXo, dYm, 'method', 'mobius', ...
                                'truncationSigmas', Inf, 'verbose', false);
    end
    results{end+1,1} = sprintf( ...
        'sweep: orbit route K=%d r=%d matches per-offset', K, r); %#ok<*SAGROW>
    results{end,2} = max(abs(gotO - refO)) <= 1e-12;
    if ~results{end,2}
        fprintf('  [diagnostic] orbit K=%d r=%d: max abs dev %.3e\n', ...
                K, r, max(abs(gotO - refO)));
    end
end

% The two routes are two decompositions of one quantity.
rng(810);
pXm = {randn(3, 5) * 3, randn(3, 5) * 3};
pYm = {randn(3, 3) * 3, randn(3, 3) * 3};
svM = [0.9 0.9]; rvM = [2 2]; zM = [false false]; pdM = [NaN NaN];
exchM = [true true];
offM = [orbOff; 0.4 * orbOff];
dXm = buildMaet(pXm, [], svM, rvM, zM, zM, pdM, exchM, 'verbose', false);
dYm2 = buildMaet(pYm, [], svM, rvM, zM, zM, pdM, exchM, 'verbose', false);
gotMix = sweepSimMaet(dXm, dYm2, offM, 'method', 'mixture', ...
                            'truncationSigmas', Inf, 'verbose', false);
gotOrb = sweepSimMaet(dXm, dYm2, offM, 'method', 'orbit', ...
                            'truncationSigmas', Inf, 'verbose', false);
results{end+1,1} = 'sweep: orbit and mixture routes agree';
results{end,2} = max(abs(gotMix - gotOrb)) <= 1e-11;
if ~results{end,2}
    fprintf('  [diagnostic] orbit vs mixture: max abs dev %.3e\n', ...
            max(abs(gotMix - gotOrb)));
end

% A swept periodic attribute: the mixture refuses it, the orbit route
% carries it, and 'auto' therefore reaches it.
rng(820);
pXp3 = {mod(randn(3, 5) * 4, 12)};
pYp3 = {mod(randn(3, 2) * 4, 12)};
dXp3 = buildMaet(pXp3, [], 0.6, 3, false, true, 12, true, 'verbose', false);
dYp3 = buildMaet(pYp3, [], 0.6, 3, false, true, 12, true, 'verbose', false);
gotP3 = sweepSimMaet(dXp3, dYp3, orbOff, 'method', 'orbit', ...
                           'truncationSigmas', Inf, 'verbose', false);
refP3 = zeros(1, numel(orbOff));
for m = 1:numel(orbOff)
    dYm = buildMaet({pYp3{1} + orbOff(m)}, [], 0.6, 3, false, true, ...
                       12, true, 'verbose', false);
    refP3(m) = simMaet(dXp3, dYm, 'method', 'mobius', ...
                             'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: orbit route carries a swept periodic attribute';
results{end,2} = max(abs(gotP3 - refP3)) <= 1e-12;
if ~results{end,2}
    fprintf('  [diagnostic] orbit swept periodic: max abs dev %.3e\n', ...
            max(abs(gotP3 - refP3)));
end

ok = false;
try
    sweepSimMaet(dXp3, dYp3, orbOff, 'method', 'mixture', ...
                       'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:periodicAttribute');
end
results{end+1,1} = 'sweep: mixture still refuses a swept periodic attribute';
results{end,2} = ok;

gotAuto = sweepSimMaet(dXp3, dYp3, orbOff, ...
                             'truncationSigmas', Inf, 'verbose', false);
results{end+1,1} = 'sweep: auto reaches the swept periodic case via orbit';
results{end,2} = max(abs(gotAuto - refP3)) <= 1e-12;

ok = false;
try
    sweepSimMaet(dXo, dYo, orbOff, 'method', 'nonsense', ...
                       'verbose', false);
catch
    ok = true;
end
results{end+1,1} = 'sweep: an unknown method is rejected';
results{end,2} = ok;


% --- The orbit route declines ordered attributes ------------------------
%
%  The orbit decomposition sums over unordered value subsets with
%  multiplicity, and mobius.maPerAttrInnerMatrix takes no symmetry flag:
%  it computes the symmetrised inner product and nothing else. On an
%  ordered attribute that is a different quantity rather than an
%  approximation of the right one, so the route must decline instead of
%  silently symmetrising, and 'auto' must fall back to the mixture.

rng(830);
pXord = {randn(5, 6) * 3, randn(5, 6) * 3};
pYord = {randn(5, 3) * 3, randn(5, 3) * 3};
svO = [0.9 0.9]; rvO = [3 3]; zO = [false false]; pdO = [NaN NaN];
offOrd = [baseOff; 0.5 * baseOff];
dXord = buildMaet(pXord, [], svO, rvO, zO, zO, pdO, [false false], ...
                     'verbose', false);
dYord = buildMaet(pYord, [], svO, rvO, zO, zO, pdO, [false false], ...
                     'verbose', false);
ok = false;
try
    sweepSimMaet(dXord, dYord, offOrd, 'method', 'orbit', ...
                       'truncationSigmas', Inf, 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepSimMaet:orbitUnsupported');
end
results{end+1,1} = 'sweep: orbit route declines an ordered attribute';
results{end,2} = ok;

gotOrd = sweepSimMaet(dXord, dYord, offOrd, ...
                            'truncationSigmas', Inf, 'verbose', false);
refOrd = zeros(1, size(offOrd, 2));
for m = 1:size(offOrd, 2)
    dYm = buildMaet({pYord{1} + offOrd(1, m), pYord{2} + offOrd(2, m)}, ...
                       [], svO, rvO, zO, zO, pdO, [false false], ...
                       'verbose', false);
    refOrd(m) = simMaet(dXord, dYm, 'method', 'bulger', ...
                              'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: auto falls back to the mixture when ordered';
results{end,2} = max(abs(gotOrd - refOrd)) <= tol;


% --- Default truncation reaches every route -----------------------------
%
%  Every other test here names truncationSigmas explicitly, so the
%  default ([]) never reached the routes' internals. It has to: the
%  Mobius entry points require a scalar and would reject an empty value,
%  which is a failure no accuracy test can find because it is about the
%  argument's shape rather than its value.

rng(840);
pXd = {randn(4, 6) * 3};
pYd = {randn(4, 3) * 3};
dXd = buildMaet(pXd, [], 0.9, 3, false, false, NaN, true, ...
                   'verbose', false);
dYd = buildMaet(pYd, [], 0.9, 3, false, false, NaN, true, ...
                   'verbose', false);
offD = [-2.6, -0.9, 0.0, 1.3, 3.4];
for methodName = {'mixture', 'orbit', 'auto'}
    mn = methodName{1};
    ok = true;
    try
        vD = sweepSimMaet(dXd, dYd, offD, 'method', mn, ...
                                'verbose', false);
        ok = all(isfinite(vD)) && numel(vD) == numel(offD);
    catch ME
        ok = false;
        fprintf('  [diagnostic] default truncation, method %s: %s: %s\n', ...
                mn, ME.identifier, ME.message);
    end
    results{end+1,1} = sprintf( ...
        'sweep: default truncationSigmas works with method %s', mn); %#ok<*SAGROW>
    results{end,2} = ok;
end

% The routes must also agree with each other at the default.
vMixD = sweepSimMaet(dXd, dYd, offD, 'method', 'mixture', ...
                           'verbose', false);
vOrbD = sweepSimMaet(dXd, dYd, offD, 'method', 'orbit', ...
                           'verbose', false);
results{end+1,1} = 'sweep: routes agree at the default truncation';
results{end,2} = max(abs(vMixD - vOrbD)) <= 1e-8;


% --- Standalone summary ---
if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_sweep_reduction: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_sweep_reduction:failed', '%d test(s) failed.', nFail);
    end
end

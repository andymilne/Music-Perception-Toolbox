%% test_sweep_reduction.m — translation sweeps as a mixture in the offset
%
%  sweepCosSimExpTens replaces one inner product per offset with a single
%  pass over the tuple pairs followed by M evaluations of a Gaussian
%  mixture in the offset. Every test here compares it against the
%  per-offset path it replaces: translate the query explicitly, call
%  cosSimExpTens, and require agreement at the parity floor.
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
    for isSym = [false true]
        rng(100 + 10 * K + r);
        A = 2;
        pX = {randn(K, 6) * 3, randn(K, 6) * 3};
        pY = {randn(K, 3) * 3, randn(K, 3) * 3};
        off = [baseOff; 0.5 * baseOff];
        sig = [0.9 0.9]; rv = [r r];
        z = [false false]; pd = [NaN NaN]; sym = [isSym isSym];

        dX = buildExpTens(pX, [], sig, rv, z, z, pd, sym, 'verbose', false);
        dY = buildExpTens(pY, [], sig, rv, z, z, pd, sym, 'verbose', false);
        got = sweepCosSimExpTens(dX, dY, off, ...
            'truncationSigmas', Inf, 'verbose', false);

        ref = zeros(1, size(off, 2));
        for m = 1:size(off, 2)
            pYm = {pY{1} + off(1, m), pY{2} + off(2, m)};
            dYm = buildExpTens(pYm, [], sig, rv, z, z, pd, sym, ...
                'verbose', false);
            ref(m) = cosSimExpTens(dX, dYm, 'method', 'bulger', ...
                'truncationSigmas', Inf, 'verbose', false);
        end
        dev = max(abs(got - ref) ./ max(abs(ref), 1e-12));
        results{end+1,1} = sprintf( ...
            'sweep: K=%d r=%d sym=%d matches per-offset', K, r, isSym); %#ok<*SAGROW>
        results{end,2} = dev <= tol;
    end
end


% --- Agreement under the default truncation floor -----------------------

rng(7);
pX = {randn(3, 6) * 3, randn(3, 6) * 3};
pY = {randn(3, 3) * 3, randn(3, 3) * 3};
off = [baseOff; 0.5 * baseOff];
sig = [0.9 0.9]; rv = [3 3]; z = [false false]; pd = [NaN NaN];
sym = [true true];
dX = buildExpTens(pX, [], sig, rv, z, z, pd, sym, 'verbose', false);
dY = buildExpTens(pY, [], sig, rv, z, z, pd, sym, 'verbose', false);
got = sweepCosSimExpTens(dX, dY, off, 'verbose', false);
ref = zeros(1, size(off, 2));
for m = 1:size(off, 2)
    dYm = buildExpTens({pY{1} + off(1, m), pY{2} + off(2, m)}, [], ...
        sig, rv, z, z, pd, sym, 'verbose', false);
    ref(m) = cosSimExpTens(dX, dYm, 'method', 'bulger', 'verbose', false);
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

gotOne = sweepCosSimExpTens(dX, dY, off, 'normalize', 'oneSidedDenom', ...
    'truncationSigmas', Inf, 'verbose', false);
refOne = zeros(1, size(off, 2));
for m = 1:size(off, 2)
    dYm = buildExpTens({pY{1} + off(1, m), pY{2} + off(2, m)}, [], ...
        sig, rv, z, z, pd, sym, 'verbose', false);
    refOne(m) = cosSimExpTens(dX, dYm, 'normalize', 'oneSidedDenom', ...
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
for isSym = [false true]
    dXp = buildExpTens(pXp, [], 0.9, 3, false, false, NaN, isSym, ...
        'verbose', false);
    dYp = buildExpTens(pYp, [], 0.9, 3, false, false, NaN, isSym, ...
        'verbose', false);
    g = sweepCosSimExpTens(dXp, dYp, 0, 'truncationSigmas', Inf, ...
        'verbose', false);
    rr = cosSimExpTens(dXp, dYp, 'method', 'bulger', ...
        'truncationSigmas', Inf, 'verbose', false);
    results{end+1,1} = sprintf('sweep: permuted tuple, sym=%d', isSym);
    results{end,2} = abs(g - rr) <= tol;
    if isSym
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
dXr = buildExpTens(pXr, [], sig, rv, relv, z, pd, sym, 'verbose', false);
dYr = buildExpTens(pYr, [], sig, rv, relv, z, pd, sym, 'verbose', false);
gotR = sweepCosSimExpTens(dXr, dYr, offR, 'truncationSigmas', Inf, ...
    'verbose', false);
refR = zeros(1, size(offR, 2));
for m = 1:size(offR, 2)
    dYm = buildExpTens({pYr{1} + offR(1, m), pYr{2}}, [], sig, rv, ...
        relv, z, pd, sym, 'verbose', false);
    refR(m) = cosSimExpTens(dXr, dYm, 'method', 'bulger', ...
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
dXp2 = buildExpTens(pXp2, [], sig, rv, z, perv, pdv, sym, 'verbose', false);
dYp2 = buildExpTens(pYp2, [], sig, rv, z, perv, pdv, sym, 'verbose', false);
gotP = sweepCosSimExpTens(dXp2, dYp2, offP, 'truncationSigmas', Inf, ...
    'verbose', false);
refP = zeros(1, size(offP, 2));
for m = 1:size(offP, 2)
    dYm = buildExpTens({pYp2{1} + offP(1, m), pYp2{2}}, [], ...
        sig, rv, z, perv, pdv, sym, 'verbose', false);
    refP(m) = cosSimExpTens(dXp2, dYm, 'method', 'bulger', ...
        'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: unswept periodic attribute matches';
results{end,2} = max(abs(gotP - refP) ./ max(abs(refP), 1e-12)) <= tol;


% --- Refusals -----------------------------------------------------------

dRel = buildExpTens({randn(3, 4)}, [], 0.9, 3, true, false, NaN, true, ...
    'verbose', false);
ok = false;
try
    sweepCosSimExpTens(dRel, dRel, [0 1.5], 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepCosSimExpTens:sweptRelative');
end
results{end+1,1} = 'sweep: swept relative attribute is refused';
results{end,2} = ok;

dPer = buildExpTens({randn(3, 4)}, [], 0.9, 3, false, true, 12, true, ...
    'verbose', false);
ok = false;
try
    sweepCosSimExpTens(dPer, dPer, [0 1.5], 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepCosSimExpTens:periodicAttribute');
end
results{end+1,1} = 'sweep: swept periodic attribute is refused';
results{end,2} = ok;

% An unswept periodic attribute is accepted, so the refusal above is
% about the sweep, not about periodicity as such.
ok = true;
try
    sweepCosSimExpTens(dPer, dPer, [0 0], 'verbose', false);
catch
    ok = false;
end
results{end+1,1} = 'sweep: unswept periodic attribute is accepted';
results{end,2} = ok;

dRP = buildExpTens({randn(2, 4)}, [], 0.9, 2, true, true, 12, true, ...
    'verbose', false);
ok = false;
try
    sweepCosSimExpTens(dRP, dRP, [0 0], 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepCosSimExpTens:relativePeriodic');
end
results{end+1,1} = 'sweep: relative-periodic is refused even unswept';
results{end,2} = ok;

ok = false;
try
    sweepCosSimExpTens(dX, dY, zeros(3, 4), 'verbose', false);
catch ME
    ok = strcmp(ME.identifier, 'sweepCosSimExpTens:offsetsShape');
end
results{end+1,1} = 'sweep: offsets shape is validated';
results{end,2} = ok;


% --- translateAttributes carries its offsets ----------------------------

[pOut, ~, ~, sw] = translateAttributes({zeros(2, 3), zeros(2, 3)}, [], ...
    {baseOff, 0.5 * baseOff});
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

[~, ~, ~, swSingle] = translateAttributes({zeros(2, 3)}, [], {5});
results{end+1,1} = 'sweep: single translation carries no sweep struct';
results{end,2} = isempty(swSingle);

% The carried offsets drive the reduction directly.
rng(70);
pXt = {randn(3, 6) * 3, randn(3, 6) * 3};
pYt = {randn(3, 3) * 3, randn(3, 3) * 3};
[~, ~, ~, swT] = translateAttributes(pYt, [], {baseOff, 0.5 * baseOff});
dXt = buildExpTens(pXt, [], sig, rv, z, z, pd, sym, 'verbose', false);
dYt = buildExpTens(swT.base, [], sig, rv, z, z, pd, sym, 'verbose', false);
gotT = sweepCosSimExpTens(dXt, dYt, swT.offsets, ...
    'truncationSigmas', Inf, 'verbose', false);
refT = zeros(1, numel(baseOff));
for m = 1:numel(baseOff)
    dYm = buildExpTens({pYt{1} + swT.offsets(1, m), ...
                        pYt{2} + swT.offsets(2, m)}, [], ...
        sig, rv, z, z, pd, sym, 'verbose', false);
    refT(m) = cosSimExpTens(dXt, dYm, 'method', 'bulger', ...
        'truncationSigmas', Inf, 'verbose', false);
end
results{end+1,1} = 'sweep: carried offsets drive the reduction';
results{end,2} = max(abs(gotT - refT) ./ max(abs(refT), 1e-12)) <= tol;


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

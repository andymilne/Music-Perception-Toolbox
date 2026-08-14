%% test_gram_quadratic_form.m — Gram forms in the shared inner-product path
%
%  In every non-periodic mode the quadratic form is a squared Euclidean
%  distance between (possibly quotiented) coordinates, so maLogKernel
%  computes it as a Gram matrix --- one matrix product --- rather than
%  building an (r_a, nJ, nK) difference array. Absolute uses the raw
%  coordinates, relative removes the whole tuple's all-ones, and a
%  nested attribute at an inner or intermediate co-transposition unit
%  removes each block's own all-ones and sums over blocks. Periodic
%  attributes keep the difference path, where the wrap makes the form
%  non-Euclidean.
%
%  The tests below are route-independent: they assert properties the
%  computed quantity has, rather than comparing one implementation
%  against another. That matters because the Gram identity's weakness is
%  numerical rather than algebraic --- it forms |u|^2 + |v|^2 - 2 u.v,
%  whose cancellation costs significant digits when the coordinates sit
%  far from the origin. Both operands are therefore shifted by one of
%  the attribute's own values first, which is exact (every quantity here
%  depends on the operands only through their differences) and which is
%  what these tests are placed to protect: measured in the twin Python
%  implementation, dropping the shift moved a cosine by 2.0e-07 at
%  coordinate magnitude 1e6, against 4.6e-13 with it.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


% --- A common shift cannot change a cosine ------------------------------
%
%  Shifting both operands by the same amount leaves every difference
%  unchanged, so the similarity must be unchanged too. The tolerance
%  allows for the inputs themselves losing resolution at magnitude ---
%  float spacing at 1e6 is 1.2e-10 against a data spread of 5 --- while
%  staying far tighter than an unshifted Gram form would reach.

modes = { ...
    'absolute',          false, false, NaN; ...
    'relative',          true,  false, NaN; ...
    'absolute periodic', false, true,  12; ...
};

for mi = 1:size(modes, 1)
    lbl   = modes{mi, 1};
    isRel = modes{mi, 2};
    isPer = modes{mi, 3};
    per   = modes{mi, 4};

    rng(700 + mi);
    K = 4; r = 3;
    baseX = {randn(K, 12) * 5, randn(K, 12) * 5};
    baseY = {randn(K, 4)  * 5, randn(K, 4)  * 5};
    sv = [5 5]; rv = [r r]; rl = [isRel isRel];
    pv = [isPer isPer]; pd = [per per]; sym = [true true];

    dX0 = buildExpTens(baseX, [], sv, rv, rl, pv, pd, sym, 'verbose', false);
    dY0 = buildExpTens(baseY, [], sv, rv, rl, pv, pd, sym, 'verbose', false);
    s0 = cosSimExpTens(dX0, dY0, 'method', 'bulger', 'verbose', false);

    for mag = [1e3, 1e6]
        shiftedX = cellfun(@(M) M + mag, baseX, 'UniformOutput', false);
        shiftedY = cellfun(@(M) M + mag, baseY, 'UniformOutput', false);
        dXm = buildExpTens(shiftedX, [], sv, rv, rl, pv, pd, sym, ...
                           'verbose', false);
        dYm = buildExpTens(shiftedY, [], sv, rv, rl, pv, pd, sym, ...
                           'verbose', false);
        sm = cosSimExpTens(dXm, dYm, 'method', 'bulger', 'verbose', false);
        results{end+1,1} = sprintf( ...
            'gram: %s is shift invariant at magnitude %g', lbl, mag); %#ok<*SAGROW>
        results{end,2} = abs(sm - s0) <= 1e-11;
    end
end


% --- Nested inner unit: block-diagonal form -----------------------------

rng(710);
spec = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true false], ...
              'rel', 'innermost');
baseXn = sort(randn(4, 8) * 4, 1);
baseYn = sort(randn(4, 3) * 4, 1);
dXn0 = buildExpTens({baseXn}, [], 1.0, 1, false, false, 0, true, ...
                    'nested', {spec}, 'verbose', false);
dYn0 = buildExpTens({baseYn}, [], 1.0, 1, false, false, 0, true, ...
                    'nested', {spec}, 'verbose', false);
sn0 = cosSimExpTens(dXn0, dYn0, 'method', 'bulger', 'verbose', false);
dXn1 = buildExpTens({baseXn + 1e6}, [], 1.0, 1, false, false, 0, true, ...
                    'nested', {spec}, 'verbose', false);
dYn1 = buildExpTens({baseYn + 1e6}, [], 1.0, 1, false, false, 0, true, ...
                    'nested', {spec}, 'verbose', false);
sn1 = cosSimExpTens(dXn1, dYn1, 'method', 'bulger', 'verbose', false);
results{end+1,1} = 'gram: nested inner unit is shift invariant';
results{end,2} = abs(sn1 - sn0) <= 1e-11;


% --- A self comparison is exactly 1 under either route ------------------
%
%  The Gram form clamps at zero, so a self match must land on Q = 0
%  rather than on a small negative value that the clamp would hide.

rng(720);
pS = {randn(4, 8) * 5 + 1e6};
for isRel = [false true]
    dS = buildExpTens(pS, [], 5, 3, isRel, false, NaN, true, ...
                      'verbose', false);
    sSelf = cosSimExpTens(dS, dS, 'method', 'bulger', 'verbose', false);
    results{end+1,1} = sprintf( ...
        'gram: self similarity is 1 (isRel=%d, far from origin)', isRel);
    results{end,2} = abs(sSelf - 1) <= 1e-12;
end


% --- The guard declines the Gram form where it would cost accuracy ------
%
%  The Gram rounding scales as eps * spread^2 / (4 sigma^2), so it grows
%  as sigma shrinks against the attribute's spread. At the accuracy
%  floor the difference form runs instead. The value must remain a
%  similarity either way.

rng(730);
pT = {randn(4, 10) * 5};
qT = {randn(4, 4) * 5};
dT1 = buildExpTens(pT, [], 0.01, 3, false, false, NaN, true, 'verbose', false);
dT2 = buildExpTens(qT, [], 0.01, 3, false, false, NaN, true, 'verbose', false);
sTight = cosSimExpTens(dT1, dT2, 'method', 'bulger', ...
                       'truncationSigmas', Inf, 'verbose', false);
results{end+1,1} = 'gram: small sigma at the accuracy floor stays bounded';
results{end,2} = isfinite(sTight) && sTight >= -1 - 1e-12 ...
                 && sTight <= 1 + 1e-12;


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
    fprintf('\n=== test_gram_quadratic_form: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_gram_quadratic_form:failed', '%d test(s) failed.', nFail);
    end
end

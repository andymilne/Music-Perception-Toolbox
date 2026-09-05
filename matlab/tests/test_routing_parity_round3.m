%% test_routing_parity_round3.m — regressions from the routing-parity audit, round 3
%
%  Mirror of the Python tests/test_routing_parity_round3.py. The round
%  removed the dead code the audit found and made explainDispatch
%  report the route the call takes; each block pins one of those
%  outcomes so the two languages keep agreeing:
%
%    explainDispatch on a flat cosine builds the selector's inputs by
%      the same function the call uses (INTERNAL.FLATSELECTORINPUTS: the
%      wrap vector, the per-attribute grid node counts, the memo flags),
%      applies the empty-operand rule before the selector and the
%      ordered-attribute rule after it --- so on a rel-per pair above
%      the sigma/P threshold the report follows the declared wrap as the
%      call does, and on an ordered pair it names Bulger's method under
%      any method;
%    D-6   the eval cost model prices the spectral branch of the Möbius
%      relative evaluator wherever that branch engages (no mode-grid
%      decline, since the evaluator has none);
%    B-16  the Möbius per-attribute matrix has one abs r >= 2 route,
%      and agrees with the enumerated reference at K = r;
%    B-16  the tuple-centres closed form refuses an inner [rel] unit;
%    D     the deleted names stay deleted.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fullfile(fileparts(mfilename('fullpath')), 'reference'));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_rp3
    cleanupDefaults_rp3 = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

rp3_P = 12.0;
rp3_absWarn = warning('off', 'buildExpTens:absPerSingleImage');
rp3_relWarn = warning('off', 'buildExpTens:isRelDegenerate');
rp3_warnCleanup = onCleanup(@() cellfun(@warning, ...
    {rp3_absWarn, rp3_relWarn})); %#ok<NASGU>

% --- explainDispatch follows the wrap rule above the threshold ---
% r = 3, K = 4 vs 5, rel-per at sigma/P = 0.5: above the threshold the
% declared wrap decides without pricing. The report used to omit the
% wrap vector and so named the cost model's pick. The route the call
% took is read off the memo key it wrote on the second operand.
rp3_wraps = {'full-image', 'single-image'};
rp3_expect = {'mobius', 'bulger'};
for rp3_i = 1:2
    rp3_wrap = rp3_wraps{rp3_i};
    x = rp3Flat(1, 0.5 * rp3_P, 3, true, true, rp3_wrap, rp3_P, 4);
    y = rp3Flat(2, 0.5 * rp3_P, 3, true, true, rp3_wrap, rp3_P, 5);
    rep = explainDispatch(x, y);
    [~, ~, yOut] = cosSimExpTens(x, y, 'verbose', false);
    results{end+1, 1} = sprintf( ...
        'parity round3: explain follows the %s wrap rule above the threshold', ...
        rp3_wrap); %#ok<*SAGROW>
    results{end, 2} = strcmp(rep.chosen, rp3_expect{rp3_i}) ...
        && strcmp(rep.decidedBy, 'structural rule') ...
        && rp3RouteOf(yOut, rp3_expect{rp3_i});
    results{end+1, 1} = sprintf( ...
        'parity round3: explain measure line follows the %s wrap', rp3_wrap);
    results{end, 2} = contains(rep.measure, 'transposition average') ...
        == strcmp(rp3_expect{rp3_i}, 'mobius');
end

% --- explainDispatch applies the ordered-attribute rule ---
% An ordered ([sym]=0) attribute at r > 1 has no orbit: the call takes
% Bulger's method whatever method asked for, and so does the report.
rp3_methods = {'auto', 'mobius', 'centres'};
for rp3_i = 1:3
    rp3_m = rp3_methods{rp3_i};
    rng(3, 'twister');
    x = buildExpTens({sort(rp3_P * rand(5, 2), 1)}, {[]}, 1.0, 2, ...
                     false, false, 0, false, 'verbose', false);
    rng(4, 'twister');
    y = buildExpTens({sort(rp3_P * rand(6, 2), 1)}, {[]}, 1.0, 2, ...
                     false, false, 0, false, 'verbose', false);
    rep = explainDispatch(x, y, 'method', rp3_m);
    [~, ~, yOut] = cosSimExpTens(x, y, 'method', rp3_m, 'verbose', false);
    results{end+1, 1} = sprintf( ...
        'parity round3: explain applies the ordered rule under method=%s', rp3_m);
    results{end, 2} = strcmp(rep.chosen, 'bulger') ...
        && contains(rep.decidedBy, 'ordered') ...
        && rp3RouteOf(yOut, 'bulger');
end

% --- explainDispatch uses the memo flags the call uses ---
% After a call has memoised both self inner products, the report prices
% the cross matrix alone, as the call does: its Bulger price drops.
x = rp3Flat(5, 1.0, 3, true, false, 'full-image', rp3_P, 6, 3);
y = rp3Flat(6, 1.0, 3, true, false, 'full-image', rp3_P, 6, 3);
repCold = explainDispatch(x, y);
[~, xOut, yOut] = cosSimExpTens(x, y, 'method', 'bulger', 'verbose', false);
repWarm = explainDispatch(xOut, yOut);
results{end+1, 1} = 'parity round3: explain prices the memoised self products as free';
results{end, 2} = repWarm.routeMs(1) < repCold.routeMs(1);

% --- explainDispatch applies the empty-operand rule ---
rng(7, 'twister');
pZ = sort(rp3_P * rand(5, 2), 1);
z = buildExpTens({pZ}, {zeros(5, 2)}, 1.0, 2, false, false, 0, ...
                 'verbose', false);
y = rp3Flat(8, 1.0, 2, false, false, 'full-image', rp3_P);
rep = explainDispatch(z, y);
results{end+1, 1} = 'parity round3: explain applies the empty-operand rule';
results{end, 2} = isempty(rep.chosen) ...
    && contains(rep.decidedBy, 'empty operand') ...
    && cosSimExpTens(z, y, 'verbose', false) == 0;

% --- the selector inputs are shared with the call ---
x = rp3Flat(9, 0.5 * rp3_P, 3, true, true, 'single-image', rp3_P, 4);
y = rp3Flat(10, 0.5 * rp3_P, 3, true, true, 'single-image', rp3_P, 5);
[selIn, orderedAny, nestedAny] = internal.flatSelectorInputs( ...
    internal.prunedExpTens(x), internal.prunedExpTens(y), 'cosine', []);
[~, xOut, yOut] = cosSimExpTens(x, y, 'verbose', false);
selIn2 = internal.flatSelectorInputs( ...
    internal.prunedExpTens(xOut), internal.prunedExpTens(yOut), 'cosine', []);
results{end+1, 1} = 'parity round3: flatSelectorInputs carries the wrap, per, node and memo inputs';
results{end, 2} = isequal(selIn.wrapVec, {'single-image'}) ...
    && isequal(selIn.perVec, true) && selIn.nuVec(1) > 1 ...
    && ~orderedAny && ~nestedAny ...
    && ~selIn.skipXX && ~selIn.skipYY ...
    && selIn2.skipXX && selIn2.skipYY;

% --- D-6: the eval cost model has no mode-grid decline ---
% A periodic r = 4 relative attribute at small sigma/P passes the
% evaluator's spectral gate; the model prices that branch, whose r = 4
% periodic K term is zero, so the price is the same at K = 16 and 48
% (the node path it used to fall back to carries a K-proportional
% tabulation term).
rp3_sig = 0.004 * rp3_P;
dSmall = rp3Flat(11, rp3_sig, 4, true, true, 'full-image', rp3_P, 16, 1);
dLarge = rp3Flat(12, rp3_sig, 4, true, true, 'full-image', rp3_P, 48, 1);
[~, mSmall] = internal.maEvalCostsMs(dSmall, 256);
[~, mLarge] = internal.maEvalCostsMs(dLarge, 256);
results{end+1, 1} = 'parity round3: eval cost model prices the spectral branch at r=4 periodic';
results{end, 2} = isfinite(mSmall) && isfinite(mLarge) ...
    && abs(mLarge - mSmall) <= 1e-12 * max(abs(mSmall), 1);

% --- B-16: one abs r >= 2 route agrees with enumeration at K = r ---
rng(13, 'twister');
Px = sort(40 * rand(3, 4), 1); Wx = ones(3, 4);
Py = sort(40 * rand(3, 5), 1); Wy = ones(3, 5);
rp3_prevEps = internal.accuracyFloor('setEps', 1e-300);
I = mobius.maPerAttrInnerMatrix(Px, Wx, Py, Wy, 3.0, 3, false, false, 0, ...
                                'truncationSigmas', Inf);
internal.accuracyFloor('setEps', rp3_prevEps);
Iref = zeros(4, 5);
for ii = 1:4
    for jj = 1:5
        Iref(ii, jj) = reference.innerProductDirectAbsSingleMultiset( ...
            Px(:, ii), Wx(:, ii), Py(:, jj), Wy(:, jj), 3.0, 3, false, 0);
    end
end
results{end+1, 1} = 'parity round3: per-attribute batched route matches enumeration at K = r';
results{end, 2} = all(abs(I(:) - Iref(:)) <= 1e-12 * max(abs(Iref(:))));

% --- B-16: the closed form refuses an inner [rel] unit ---
x = rp3Flat(14, 1.0, 2, true, false, 'full-image', rp3_P, 4, 1);
cx = mobius.closedFormAttrCentres(internal.prunedExpTens(x), 1);
Mok = mobius.closedFormAttrMatrixFrom(cx, cx, 'full-image');
cxBad = cx; cxBad.innerBlockSize = 2;
results{end+1, 1} = 'parity round3: closed form serves a flat bundle';
results{end, 2} = isequal(size(Mok), [1 1]) && isfinite(Mok);
results{end+1, 1} = 'parity round3: closed form refuses an inner-unit bundle';
results{end, 2} = throwsErrorWithId( ...
    @() mobius.closedFormAttrMatrixFrom(cxBad, cx, 'full-image'), ...
    'mobius:closedFormAttrMatrixFrom:innerUnit');

% --- D: the orphans are gone ---
rp3_root = fileparts(fileparts(mfilename('fullpath')));
rp3_gone = {fullfile(rp3_root, '+internal', 'warnRelPerAllImage.m'), ...
            fullfile(rp3_root, '+mobius', 'contract.m'), ...
            fullfile(rp3_root, '+mobius', 'innerProductDirectAbsSingleMultiset.m')};
results{end+1, 1} = 'parity round3: deleted and moved routines are gone from the shipped tree';
results{end, 2} = ~any(cellfun(@(f) exist(f, 'file') == 2, rp3_gone)) ...
    && exist(fullfile(rp3_root, 'tests', 'reference', '+reference', ...
                      'contract.m'), 'file') == 2;
rp3_src = fileread(fullfile(rp3_root, '+mobius', 'maPerAttrInnerMatrix.m'));
results{end+1, 1} = 'parity round3: maPerAttrInnerMatrix carries no safe/unsafe partition';
results{end, 2} = ~contains(rp3_src, 'localFillDirectEnumGroups') ...
    && ~contains(rp3_src, 'localPackNanTop') ...
    && ~contains(rp3_src, 'safe_x_mask');
rp3_src = fileread(fullfile(rp3_root, '+internal', 'predictOrbitCostMs.m'));
results{end+1, 1} = 'parity round3: predictOrbitCostMs carries no unused constants';
results{end, 2} = ~contains(rp3_src, 'GRID_OP') ...
    && ~contains(rp3_src, 'CENTRES_OP') && ~contains(rp3_src, 'REL_BASE');

% --- E: the retired cancellationThreshold keyword is rejected ---
% The keyword is gone from the parser, so it is left in varargin and
% trips the usual argument-count usage error rather than being accepted
% and ignored.
x = rp3Flat(21, 1.0, 2, false, false, 'full-image', rp3_P, 4, 1);
y = rp3Flat(22, 1.0, 2, false, false, 'full-image', rp3_P, 4, 1);
results{end+1, 1} = 'parity round3: cancellationThreshold keyword is rejected';
results{end, 2} = throwsErrorWithId( ...
    @() cosSimExpTens(x, y, 'cancellationThreshold', 1e-12, 'verbose', false), ...
    'cosSimExpTens:wrongArgCount');

clear x y z rep repCold repWarm xOut yOut selIn selIn2 orderedAny nestedAny ...
      dSmall dLarge mSmall mLarge Px Wx Py Wy I Iref ii jj cx cxBad Mok pZ

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
    fprintf('\n=== test_routing_parity_round3: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_rp3 rp3_warnCleanup
    if nFail > 0
        error('test_routing_parity_round3:failed', '%d test(s) failed.', nFail);
    end
end


function d = rp3Flat(seed, sigma, r, isRel, isPer, wrap, P, K, N)
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


function tf = rp3RouteOf(densOut, route)
%RP3ROUTEOF  Whether the memo the call wrote on this operand carries the
%   given route's key (the key prefix names the route that ran).
    tf = isfield(densOut, 'selfIP') && isstruct(densOut.selfIP) ...
        && any(strncmp(densOut.selfIP.keys, [route '|'], numel(route) + 1));
end

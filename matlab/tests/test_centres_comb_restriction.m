%% test_centres_comb_restriction.m
%  Bulger's X-side restriction in the Möbius centres route is exact.
%
%  MOBIUS.CLOSEDFORMATTRMATRIXFROM restricts the X side of the
%  (nJx, nJy) centre-overlap array to one representative per
%  tuple-symmetry orbit and scales the sum by the orbit size |G| (r_a!
%  for a flat symmetric attribute, an iterated wreath-product order for
%  a nested one). That is an exact identity, not an approximation, so
%  the restricted and unrestricted matrices -- and the cosines built
%  from them -- must agree to floating point. The identity and the cases
%  where the restriction is declined (a trivial orbit, an ordered flat
%  attribute, an unexpected tiling) are documented
%  in LOCALCOMBRESTRICTION inside MOBIUS.CLOSEDFORMATTRCENTRES.
%
%  Tests:
%    - Structural: the comb side times r_a! is the perm side (this is the
%      condition the identity needs, and what the code checks).
%    - Unit: restricted == unrestricted, entrywise, for rel-per,
%      rel-nonper, abs-per and abs-nonper at r = 2 and r = 3, and on
%      ragged (NaN-padded) values.
%    - The two ways of getting the unrestricted matrix -- clearing the
%      bundle's comb field, and the INTERNAL.COMBRESTRICTIONENABLED
%      switch -- agree exactly.
%    - Declines: r_a = 1 and an ordered (isSym = false) attribute carry
%      no comb bundle.
%    - Cosine level: the switch changes cost, not value.
%    - Nested: the same three claims for a nested attribute, whose perm
%      side is the free orbit of the iterated wreath product
%      INTERNAL.NESTEDORBITMULT gives in place of S_{r_a}. A nested
%      attribute reaches the centres bundle whenever
%      INTERNAL.NESTEDCONTRACT's per-attribute dispatch takes its
%      'centres' route.
%
%  Twin of the Python tests/test_centres_comb_restriction.py.
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

rng(31);

ccr_a = 2;      % attribute 1 is the scalar onset; attribute 2 carries r

% --- Structural: perm side == r_a! * comb side ------------------------
for ccr_r = [2, 3]
    ccr_d = localCcrMake(6, 4, ccr_r, true, true, true, 41);
    ccr_b = mobius.closedFormAttrCentres(ccr_d, ccr_a);
    ccr_ok = ~isempty(ccr_b.comb) ...
        && ccr_b.comb.mult == factorial(ccr_r) ...
        && size(ccr_b.Centres, 2) == ...
           factorial(ccr_r) * size(ccr_b.comb.Centres, 2) ...
        && size(ccr_b.comb.Centres, 1) == size(ccr_b.Centres, 1);
    results{end+1,1} = sprintf( ...
        'centres comb restriction: perm side is r! * comb side (r = %d)', ...
        ccr_r);
    results{end,2}   = ccr_ok;
end

% --- Unit: restricted == unrestricted, entrywise ----------------------
% isRel, isPer, r, label
ccr_cases = { ...
    true,  true,  2, 'rel-per r=2'; ...
    true,  true,  3, 'rel-per r=3'; ...
    true,  false, 2, 'rel-nonper r=2'; ...
    true,  false, 3, 'rel-nonper r=3'; ...
    false, true,  2, 'abs-per r=2'; ...
    false, false, 2, 'abs-nonper r=2'; ...
    false, true,  3, 'abs-per r=3'};
for ccr_ii = 1:size(ccr_cases, 1)
    ccr_isRel = ccr_cases{ccr_ii, 1};
    ccr_isPer = ccr_cases{ccr_ii, 2};
    ccr_r     = ccr_cases{ccr_ii, 3};
    ccr_dx = localCcrMake(5, 4, ccr_r, ccr_isRel, ccr_isPer, true, 11);
    ccr_dy = localCcrMake(5, 5, ccr_r, ccr_isRel, ccr_isPer, true, 22);
    ccr_cx = mobius.closedFormAttrCentres(ccr_dx, ccr_a);
    ccr_cy = mobius.closedFormAttrCentres(ccr_dy, ccr_a);
    ccr_Mr = mobius.closedFormAttrMatrixFrom(ccr_cx, ccr_cy);
    ccr_Mu = mobius.closedFormAttrMatrixFrom( ...
        localCcrDropComb(ccr_cx), localCcrDropComb(ccr_cy));
    ccr_rel = localCcrMaxRel(ccr_Mr, ccr_Mu);
    results{end+1,1} = sprintf( ...
        ['centres comb restriction: %s restricted matches unrestricted ' ...
         '(%.2e)'], ccr_cases{ccr_ii, 4}, ccr_rel);
    results{end,2}   = ccr_rel < 1e-13 && all(ccr_Mu(:) > 0) ...
        && ~isempty(ccr_cx.comb);
end

% --- Ragged (NaN-padded) values ---------------------------------------
ccr_dxR = localCcrMakeRagged(6, 4, 2, true, true, 15);
ccr_dyR = localCcrMakeRagged(6, 4, 2, true, true, 26);
ccr_cxR = mobius.closedFormAttrCentres(ccr_dxR, ccr_a);
ccr_cyR = mobius.closedFormAttrCentres(ccr_dyR, ccr_a);
ccr_relR = localCcrMaxRel( ...
    mobius.closedFormAttrMatrixFrom(ccr_cxR, ccr_cyR), ...
    mobius.closedFormAttrMatrixFrom( ...
        localCcrDropComb(ccr_cxR), localCcrDropComb(ccr_cyR)));
results{end+1,1} = sprintf( ...
    ['centres comb restriction: ragged rel-per restricted matches ' ...
     'unrestricted (%.2e)'], ccr_relR);
results{end,2}   = ccr_relR < 1e-13;

% --- The switch and the dropped field agree exactly -------------------
ccr_dx = localCcrMake(5, 4, 2, true, true, true, 11);
ccr_dy = localCcrMake(5, 5, 2, true, true, true, 22);
ccr_Mdrop = mobius.closedFormAttrMatrixFrom( ...
    localCcrDropComb(mobius.closedFormAttrCentres(ccr_dx, ccr_a)), ...
    localCcrDropComb(mobius.closedFormAttrCentres(ccr_dy, ccr_a)));
internal.combRestrictionEnabled(false);
ccr_cxOff = mobius.closedFormAttrCentres(ccr_dx, ccr_a);
ccr_cyOff = mobius.closedFormAttrCentres(ccr_dy, ccr_a);
ccr_Mswitch = mobius.closedFormAttrMatrixFrom(ccr_cxOff, ccr_cyOff);
internal.combRestrictionEnabled(true);
results{end+1,1} = 'centres comb restriction: switch off == comb field dropped';
results{end,2}   = isempty(ccr_cxOff.comb) && isequal(ccr_Mdrop, ccr_Mswitch);

% --- Declines: r_a = 1 and ordered attributes -------------------------
ccr_d1 = localCcrMake(5, 4, 1, false, true, true, 33);
ccr_b1 = mobius.closedFormAttrCentres(ccr_d1, ccr_a);
results{end+1,1} = 'centres comb restriction: r_a = 1 declines';
results{end,2}   = isempty(ccr_b1.comb);

ccr_dOrd = localCcrMake(5, 4, 2, true, true, false, 34);
ccr_bOrd = mobius.closedFormAttrCentres(ccr_dOrd, ccr_a);
results{end+1,1} = 'centres comb restriction: ordered attribute declines';
results{end,2}   = isempty(ccr_bOrd.comb) ...
    && size(ccr_bOrd.Centres, 2) == nchoosek(4, 2) * 5;

% --- Cosine level: the switch changes cost, not value -----------------
% Small K rel-per: the MA Möbius orchestrator takes the centres route
% here (maRelAttrPrefersCentres, pinned in test_ma_rel_centres.m).
ccr_cosOn = cosSimExpTens( ...
    localCcrMake(8, 4, 2, true, true, true, 51), ...
    localCcrMake(8, 4, 2, true, true, true, 52), ...
    'method', 'mobius', 'verbose', false);
internal.combRestrictionEnabled(false);
ccr_cosOff = cosSimExpTens( ...
    localCcrMake(8, 4, 2, true, true, true, 51), ...
    localCcrMake(8, 4, 2, true, true, true, 52), ...
    'method', 'mobius', 'verbose', false);
internal.combRestrictionEnabled(true);
results{end+1,1} = sprintf( ...
    ['centres comb restriction: cosine unchanged by the restriction ' ...
     '(%.2e)'], abs(ccr_cosOn - ccr_cosOff));
results{end,2}   = abs(ccr_cosOn - ccr_cosOff) < 1e-13 * abs(ccr_cosOff);

% --- Nested attributes: the wreath-product generalisation -------------
% Two units of two values, both levels symmetric: |G| = 2!^2 * 2!^1 = 8
% (each unit permuted independently, then the two units permuted). The
% comb side is the single combination, so the perm side is 8 columns per
% event and the restriction removes a factor of 8.
ccr_nSpec = struct('tags', [0 0 1 1], 'r', [2 2], 'sym', [true true], ...
                   'rel', [0 1]);
results{end+1,1} = 'centres comb restriction: nested |G| is the wreath order';
results{end,2}   = internal.nestedOrbitMult([2 2], [true true]) == 8 ...
    && internal.nestedOrbitMult([2 2], [true false]) == 4 ...
    && internal.nestedOrbitMult([2 2], [false true]) == 2 ...
    && internal.nestedOrbitMult([2 2], [false false]) == 1 ...
    && internal.nestedOrbitMult([1 3], [true true]) == 6;

ccr_nD = localCcrMakeNested(4, ccr_nSpec, 6.0, true, 1200.0, 61);
ccr_nB = mobius.closedFormAttrCentres(ccr_nD, 1);
results{end+1,1} = 'centres comb restriction: nested perm side is |G| * comb side';
results{end,2}   = ~isempty(ccr_nB.comb) && ccr_nB.comb.mult == 8 ...
    && size(ccr_nB.Centres, 2) == 8 * size(ccr_nB.comb.Centres, 2) ...
    && size(ccr_nB.comb.Centres, 1) == size(ccr_nB.Centres, 1) ...
    && ccr_nB.innerBlockSize == 0;

% isRel, isPer, period, sigma, label
ccr_nCases = { ...
    true,  true,  1200.0, 6.0, 'nested rel-per'; ...
    true,  false, 0.0,    6.0, 'nested rel-nonper'; ...
    false, true,  1200.0, 6.0, 'nested abs-per'; ...
    false, false, 0.0,    6.0, 'nested abs-nonper'};
for ccr_ii = 1:size(ccr_nCases, 1)
    ccr_sp = ccr_nSpec;
    if ccr_nCases{ccr_ii, 1}
        ccr_sp.rel = [0 1];
    else
        ccr_sp.rel = [0 0];    % a nested spec needs a per-level vector
    end
    ccr_ndx = localCcrMakeNested(4, ccr_sp, ccr_nCases{ccr_ii, 4}, ...
        ccr_nCases{ccr_ii, 2}, ccr_nCases{ccr_ii, 3}, 71);
    ccr_ndy = localCcrMakeNested(5, ccr_sp, ccr_nCases{ccr_ii, 4}, ...
        ccr_nCases{ccr_ii, 2}, ccr_nCases{ccr_ii, 3}, 72);
    ccr_ncx = mobius.closedFormAttrCentres(ccr_ndx, 1);
    ccr_ncy = mobius.closedFormAttrCentres(ccr_ndy, 1);
    ccr_nMr = mobius.closedFormAttrMatrixFrom(ccr_ncx, ccr_ncy);
    ccr_nMu = mobius.closedFormAttrMatrixFrom( ...
        localCcrDropComb(ccr_ncx), localCcrDropComb(ccr_ncy));
    ccr_nRel = localCcrMaxRel(ccr_nMr, ccr_nMu);
    results{end+1,1} = sprintf( ...
        ['centres comb restriction: %s restricted matches unrestricted ' ...
         '(%.2e)'], ccr_nCases{ccr_ii, 5}, ccr_nRel);
    results{end,2}   = ccr_nRel < 1e-13 && all(ccr_nMu(:) > 0) ...
        && ~isempty(ccr_ncx.comb);
end

% An all-ordered nested attribute has no orbit to collapse: |G| = 1, so
% the restriction declines and the perm side equals the comb side.
ccr_nOrdSpec = struct('tags', [0 0 1 1], 'r', [2 2], ...
                      'sym', [false false], 'rel', [0 1]);
ccr_nOrdB = mobius.closedFormAttrCentres( ...
    localCcrMakeNested(4, ccr_nOrdSpec, 6.0, true, 1200.0, 62), 1);
results{end+1,1} = 'centres comb restriction: all-ordered nested spec declines';
results{end,2}   = isempty(ccr_nOrdB.comb);

% --- Abs-per full-image: the L = 0 single-image short-circuit ---------
% When the truncation budget admits no image beyond the nearest one the
% wrapped Gaussian *is* the nearest-image Gaussian, so the route takes
% the joint Q-form path. The gate must not change the numbers, and it
% must not fire once a second image carries weight. Twin of the Python
% test_abs_per_short_circuit_agrees_with_the_image_sum and
% test_abs_per_above_the_gate_still_sums_images.
ccr_P = 1200.0;
% sigma/P, and a wider accuracy width that lifts the image count above
% zero at that same sigma/P (0 where no finite width does: below
% sigma/P ~ 0.03 the second image underflows before it is ever admitted,
% so the short-circuit is the only representation there).
ccr_absCases = [0.005, 0; 0.02, 20; 0.04, 10];
for ccr_r = [2, 3, 4]
    for ccr_ii = 1:size(ccr_absCases, 1)
        ccr_sop = ccr_absCases(ccr_ii, 1);
        ccr_tsW = ccr_absCases(ccr_ii, 2);
        ccr_sig = ccr_sop * ccr_P;
        ccr_dx = localCcrMakeAbsPer(5, 4, ccr_r, ccr_sig, ccr_P, 71);
        ccr_dy = localCcrMakeAbsPer(5, 5, ccr_r, ccr_sig, ccr_P, 72);
        ccr_cx = mobius.closedFormAttrCentres(ccr_dx, ccr_a);
        ccr_cy = mobius.closedFormAttrCentres(ccr_dy, ccr_a);
        ccr_L = internal.wrappedKernelImageCount(ccr_sig, ccr_P, 6, 4);
        ccr_Mg = mobius.closedFormAttrMatrixFrom( ...
            ccr_cx, ccr_cy, 'full-image', 6);
        % The gate's claim: at L = 0 the full-image declaration and the
        % single-image opt-in are the same number, not merely close.
        ccr_Ms = mobius.closedFormAttrMatrixFrom( ...
            ccr_cx, ccr_cy, 'single-image', 6);
        ccr_gap = max(abs(ccr_Mg(:) - ccr_Ms(:))) / max(abs(ccr_Mg(:)));
        ccr_ok = ccr_L == 0 && ccr_gap < 1e-14;
        if ccr_tsW > 0
            % Same kernel with the image sum actually running: the image
            % the wider width admits is by construction far below the
            % 6-sigma floor, so the short-circuit must match it too.
            ccr_Mi = mobius.closedFormAttrMatrixFrom( ...
                ccr_cx, ccr_cy, 'full-image', ccr_tsW);
            ccr_ok = ccr_ok ...
                && internal.wrappedKernelImageCount( ...
                       ccr_sig, ccr_P, ccr_tsW, 4) >= 1 ...
                && max(abs(ccr_Mg(:) - ccr_Mi(:))) ...
                   / max(abs(ccr_Mi(:))) < 1e-9;
        end
        results{end+1,1} = sprintf( ...
            ['centres abs-per: L = 0 short-circuit matches the image ' ...
             'sum (r = %d, sigma/P = %g, gap %.2e)'], ...
            ccr_r, ccr_sop, ccr_gap);
        results{end,2} = ccr_ok;
    end
end

for ccr_r = [2, 3, 4]
    for ccr_sop = [0.1, 0.2]
        ccr_sig = ccr_sop * ccr_P;
        ccr_dx = localCcrMakeAbsPer(5, 4, ccr_r, ccr_sig, ccr_P, 71);
        ccr_dy = localCcrMakeAbsPer(5, 5, ccr_r, ccr_sig, ccr_P, 72);
        ccr_cx = mobius.closedFormAttrCentres(ccr_dx, ccr_a);
        ccr_cy = mobius.closedFormAttrCentres(ccr_dy, ccr_a);
        ccr_L = internal.wrappedKernelImageCount(ccr_sig, ccr_P, 6, 4);
        ccr_Mf = mobius.closedFormAttrMatrixFrom( ...
            ccr_cx, ccr_cy, 'full-image', 6);
        ccr_Ms = mobius.closedFormAttrMatrixFrom( ...
            ccr_cx, ccr_cy, 'single-image', 6);
        ccr_gap = max(abs(ccr_Mf(:) - ccr_Ms(:))) / max(abs(ccr_Mf(:)));
        results{end+1,1} = sprintf( ...
            ['centres abs-per: above the gate the images are still ' ...
             'summed (r = %d, sigma/P = %g, gap %.2e)'], ...
            ccr_r, ccr_sop, ccr_gap);
        results{end,2} = ccr_L >= 1 && ccr_gap > 1e-9;
    end
end

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_centres_comb_restriction: %d passed, %d failed\n', ...
        nPass, nFail);
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end


function dens = localCcrMake(N, K, r, isRelP, isPerP, isSymP, seed)
    % Scalar onset (attribute 1) plus a K-value attribute (attribute 2).
    rng(seed);
    % Values kept well inside a few tens of sigma so every overlap is
    % nonzero: an entrywise ratio test needs a denominator.
    pitches = rand(K, N) * 60;
    onsets  = (0:N-1) * 250 + randn(1, N) * 10;
    dens = buildExpTens({onsets; pitches}, {[]; []}, [15, 6], [1, r], ...
        [false, isRelP], [false, isPerP], [4000, 1200], ...
        [true, isSymP], 'verbose', false);
end


function dens = localCcrMakeRagged(N, K, r, isRelP, isPerP, seed)
    rng(seed);
    pitches = rand(K, N) * 60;
    pitches(K, 1:3:N) = NaN;   % every third event loses its last value
    onsets  = (0:N-1) * 250 + randn(1, N) * 10;
    dens = buildExpTens({onsets; pitches}, {[]; []}, [15, 6], [1, r], ...
        [false, isRelP], [false, isPerP], [4000, 1200], ...
        [true, true], 'verbose', false);
end


function dens = localCcrMakeAbsPer(N, K, r, sigma, period, seed)
    % Scalar onset (attribute 1) plus a symmetric absolute-periodic
    % K-value attribute (attribute 2) at a chosen sigma and period, so
    % sigma/P can be placed either side of the image-count gate. The
    % values are a shared base spread over the whole period (so the
    % nearest-image reduction really bites) plus a sub-sigma jitter, so
    % that overlaps stay above underflow at the smallest sigma/P: an
    % entrywise ratio test needs a denominator.
    rng(1234);
    base = rand(8, 8) * period;      % fixed draw, so both sides share it
    rng(seed);
    pitches = mod(base(1:K, 1:N) + randn(K, N) * (0.4 * sigma), period);
    onsets  = (0:N-1) * 250 + randn(1, N) * 10;
    dens = buildExpTens({onsets; pitches}, {[]; []}, [15, sigma], ...
        [1, r], [false, false], [false, true], [4000, period], ...
        [true, true], 'verbose', false);
end


function dens = localCcrMakeNested(N, spec, sigma, isPerP, periodP, seed)
    % A single nested attribute (four values in two tagged units), the
    % shape INTERNAL.NESTEDCONTRACT's centres route builds a bundle for.
    % Values kept well inside a few tens of sigma so every overlap is
    % nonzero: an entrywise ratio test needs a denominator.
    rng(seed);
    vals = rand(4, N) * 60;
    dens = buildExpTens({vals}, {[]}, 'specs', {spec}, 'sigma', sigma, ...
        'isPer', isPerP, 'period', periodP, 'verbose', false);
end


function b = localCcrDropComb(b)
    % The unrestricted form: no comb-side bundle, so the matrix builder
    % uses the full perm-vs-perm array (the Python twin passes cx[:10]).
    b.comb = [];
end


function v = localCcrMaxRel(A, B)
    v = max(abs(A(:) - B(:)) ./ max(abs(B(:)), realmin));
end

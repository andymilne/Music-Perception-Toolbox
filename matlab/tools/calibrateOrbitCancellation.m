function results = calibrateOrbitCancellation()
%CALIBRATEORBITCANCELLATION  Calibrate the Mobius accuracy estimate here.
%
%   Twin of python/tools/calibrate_orbit_cancellation.py, and needed
%   separately from it. The quantity measured is floating-point rounding
%   error in a summation, which depends on the order the terms are
%   accumulated in; the two implementations do not accumulate in the same
%   order, so the conservatism measured on one side does not transfer to
%   the other. Until this is run, the figures quoted in
%   INTERNAL.NESTEDCONTRACT rest on Python measurements.
%
%   WHAT IS MEASURED
%
%   The Mobius reduction sums signed terms that largely cancel, so it
%   carries fewer digits than the terms it was built from. Enumeration
%   sums only non-negative terms and loses nothing, so it is the
%   reference. The guard decides whether the Mobius route is accurate
%   enough by comparing an estimate of the rounding error against
%   internal.truncationFloor, the accuracy the caller asked for.
%
%   Everything here is an absolute error on the value scale --- the scale
%   truncationSigmas is stated on --- and never a ratio. A ratio has a
%   denominator that can legitimately approach zero: a node whose true
%   value is zero scores an enormous relative error while being exactly
%   right in absolute terms.
%
%   Three quantities per cell:
%     true      max |Mobius - enumeration|, the error that occurs.
%     estimate  what the guard uses, eps * sum|term| / r!.
%     ratio     estimate / true, the conservatism.
%
%   TWO QUESTIONS
%
%   Safety is whether the guard ever ADMITS a route whose true error
%   exceeds the floor. That is the decision it makes and the thing that
%   would be a defect. It is not the same as the estimate falling below
%   the true error: an estimate can understate an error that is itself
%   far inside the budget, which changes no decision. Both are reported,
%   but only the first is a defect.
%
%   Cost is how far above the true error the estimate sits. Too
%   conservative and the guard refuses a route that was fine, which
%   costs speed but not correctness.
%
%   The weight profile dominates: uniform weights are safe across the
%   shipped range while a steeply peaked profile breaches far earlier, so
%   a sweep that omits the peaked profiles reports a clean result that
%   ordinary music does not obey.
%
%   Usage:
%       results = calibrateOrbitCancellation();
%
%   See also INTERNAL.NESTEDCONTRACT, INTERNAL.TRUNCATIONFLOOR.

    SIGMA   = 0.30;
    N_DRAW  = 12;
    SPREADS = [0.0 0.3 1.0 3.0 12.0 48.0];
    MARGINS = [1 2 3];
    R_MAX   = 8;
    % Enumerating both sides costs C(K, r) * r! kernel products per draw,
    % so cap it rather than let a large cell run for minutes. Cells past
    % it have no reference and are reported as such rather than scored.
    GOLDEN_MAX_TUPLES = 2e6;
    FLOOR_TS = [6, 8, Inf];

    profiles = {'uniform', 'linear decay', 'harmonic', 'one dominant'};
    rng(20260728, 'twister');

    fprintf(['\nMobius rounding error against enumeration, absolute, on ' ...
             'the value scale\n']);
    fprintf('sigma = %g, %d draws per cell, worst case over margins ', ...
            SIGMA, N_DRAW);
    fprintf('K-r = %s and spreads %s\n', mat2str(MARGINS), mat2str(SPREADS));
    fprintf('estimate is the guard''s eps * sum|term| / r!\n\n');

    trueTab = nan(R_MAX, numel(profiles));
    estTab  = nan(R_MAX, numel(profiles));
    refTab  = false(R_MAX, numel(profiles));
    under = {};  admitted = {};

    for pi = 1:numel(profiles)
        fprintf('  %s\n', profiles{pi});
        fprintf('    %3s %12s %12s %14s\n', 'r', 'true', 'estimate', ...
                'estimate/true');
        for r = 2:R_MAX
            wTrue = 0; wEst = 0; referenced = false;
            for m = MARGINS
                K = r + m;
                w = localWeights(profiles{pi}, K);
                for sp = SPREADS
                    [tv, ev] = localCell(r, K, w, sp, SIGMA, N_DRAW, ...
                                         GOLDEN_MAX_TUPLES);
                    wEst = max(wEst, ev);
                    if ~isnan(tv)
                        referenced = true;
                        wTrue = max(wTrue, tv);
                        if ev < tv
                            under{end+1} = {profiles{pi}, r, K, sp, tv, ev}; %#ok<AGROW>
                        end
                        for ts = FLOOR_TS
                            fl = internal.truncationFloor(ts);
                            if ev <= fl && fl < tv
                                admitted{end+1} = {profiles{pi}, r, K, ...
                                    sp, ts, tv, ev, fl}; %#ok<AGROW>
                            end
                        end
                    end
                end
            end
            trueTab(r, pi) = wTrue; estTab(r, pi) = wEst;
            refTab(r, pi) = referenced;
            if referenced
                shown = sprintf('%.2e', wTrue);
                if wTrue > 0
                    ratio = sprintf('%.0fx', wEst / wTrue);
                else
                    ratio = '--';
                end
            else
                shown = 'unreferenced';  ratio = '--';
            end
            fprintf('    %3d %12s %12.2e %14s\n', r, shown, wEst, ratio);
        end
        fprintf('\n');
    end

    fprintf(['Defect test --- the guard admitted a route whose true ' ...
             'error breaches the floor:\n']);
    if isempty(admitted)
        fprintf('    none at any of the floors swept\n');
    else
        for ii = 1:numel(admitted)
            a = admitted{ii};
            fprintf(['    %s, r=%d, K=%d, spread=%g, ts=%g: true %.2e > ' ...
                     'floor %.2e >= estimate %.2e\n'], a{1}, a{2}, a{3}, ...
                    a{4}, a{5}, a{6}, a{8}, a{7});
        end
    end
    fprintf('\nCells where the estimate understated the true error: %d\n', ...
            numel(under));
    if ~isempty(under)
        worstIdx = 1; worstRatio = 0;
        for ii = 1:numel(under)
            rr = under{ii}{5} / under{ii}{6};
            if rr > worstRatio, worstRatio = rr; worstIdx = ii; end
        end
        u = under{worstIdx};
        fprintf(['    worst by %.1fx at %s, r=%d, K=%d, spread=%g ' ...
                 '(true %.2e, estimate %.2e)\n'], worstRatio, u{1}, ...
                u{2}, u{3}, u{4}, u{5}, u{6});
        fprintf(['    None of these changes a decision unless it appears ' ...
                 'above as well.\n']);
    end
    fprintf('\n');

    fprintf('Largest r whose true error stays inside each accuracy floor:\n');
    for ts = FLOOR_TS
        fl = internal.truncationFloor(ts);
        fprintf('  truncationSigmas = %g  (floor %.2e)\n', ts, fl);
        for pi = 1:numel(profiles)
            top = []; unref = false;
            for r = 2:R_MAX
                if ~refTab(r, pi), unref = true; break; end
                if trueTab(r, pi) >= fl, break; end
                top = r;
            end
            if isempty(top)
                verdict = 'none';
            elseif unref
                verdict = sprintf('r <= %d (above unreferenced)', top);
            else
                verdict = sprintf('r <= %d', top);
            end
            fprintf('      %14s: %s\n', profiles{pi}, verdict);
        end
    end
    fprintf('\n');

    results = struct('trueErr', trueTab, 'estimate', estTab, ...
                     'referenced', refTab, 'profiles', {profiles}, ...
                     'admitted', {admitted}, 'understated', {under});
end


function w = localWeights(name, K)
%LOCALWEIGHTS  Weight vectors spanning flat to steeply peaked.
    idx = (0:K-1).';
    switch name
        case 'uniform',      w = ones(K, 1);
        case 'linear decay', w = 1 - 0.9 * idx / max(K - 1, 1);
        case 'harmonic',     w = 1 ./ (idx + 1);
        case 'one dominant', w = [1; 1e-3 * ones(K - 1, 1)];
        otherwise
            error('mpt:badProfile', 'Unknown profile ''%s''.', name);
    end
end


function M = localBlocks(K, w, spread, sigma, nDraw)
%LOCALBLOCKS  Weighted Gaussian kernel blocks, (nDraw, K, K).
    if spread == 0
        V = repmat(12 * rand(nDraw, 1), 1, K);
    else
        V = sort(spread * rand(nDraw, K), 2);
    end
    D = reshape(V, nDraw, K, 1) - reshape(V, nDraw, 1, K);
    M = exp(-(D .^ 2) / (4 * sigma ^ 2)) ...
        .* reshape(w(:) * w(:).', 1, K, K);
end


function [trueErr, estimate] = localCell(r, K, w, spread, sigma, nDraw, cap)
%LOCALCELL  Absolute true error and the guard's estimate for one cell.
    M = localBlocks(K, w, spread, sigma, nDraw);
    [vals, ~, ~, termMassSum] = mobius.innerProductOrbitGrid(M, ...
        ones(K, 1), ones(K, 1), r, 'prefactor', 1.0, ...
        'returnCancellationRatio', true);
    fr = factorial(r);
    orb = vals(:) / fr;
    if isempty(termMassSum)
        estimate = 0;
    else
        estimate = eps * max(termMassSum(:)) / fr;
    end
    nX = factorial(K) / factorial(K - r);        % permutation side
    nY = nchoosek(K, r);                         % combination side
    if nX * nY > cap
        trueErr = NaN;
        return;
    end
    gold = localEnumerate(M, K, r, nDraw);
    trueErr = max(abs(orb - gold));
end


function g = localEnumerate(M, K, r, nDraw)
%LOCALENUMERATE  Reference value by direct enumeration.
%
%   The two sides are not the same tuple set. The X side runs over
%   permutations of each r-combination and the Y side over the
%   combinations alone: that asymmetry is the permutation/combination
%   reduction whose r! cancels in the cosine, and enumerating ordered
%   tuples on both sides instead returns exactly r! times the value the
%   Mobius route computes. Every term is non-negative, so the sum
%   carries no cancellation and serves as the golden value.
    [xtup, ytup] = localTupleSides(K, r);
    nX = size(xtup, 1);  nY = size(ytup, 1);
    g = zeros(nDraw, 1);
    for b = 1:nDraw
        Mb = reshape(M(b, :, :), K, K);
        P = ones(nX, nY);
        for t = 1:r
            P = P .* Mb(xtup(:, t), ytup(:, t).');
        end
        g(b) = sum(P(:));
    end
end


function [xtup, ytup] = localTupleSides(K, r)
%LOCALTUPLESIDES  X side: permutations of each combination; Y side:
%   the combinations. Twin of the Python _tuple_indices with sym = true.
    ytup = nchoosek(1:K, r);
    permsR = perms(1:r);
    xtup = zeros(size(ytup, 1) * size(permsR, 1), r);
    row = 0;
    for c = 1:size(ytup, 1)
        for p = 1:size(permsR, 1)
            row = row + 1;
            xtup(row, :) = ytup(c, permsR(p, :));
        end
    end
end

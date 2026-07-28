function triple = nestedContract(densX, densY, normalize, truncationSigmas, force)
%NESTEDCONTRACT Fast tree-contraction of a single nested attribute's IP.
%   triple = internal.nestedContract(densX, densY, normalize, truncationSigmas)
%
%   Returns a struct with fields xy, xx, yy (the bare inner-product triple)
%   when the case is covered -- exactly one nested attribute, outer or no
%   [rel] (inner_r == 0), cosine or one-sided normalisation, NaN-padded
%   (variable-K) values included -- AND the
%   contraction is estimated cheaper than the enumeration; otherwise [] and
%   the caller routes to the exact Bulger enumeration.
%
%   Mirror of the Python mpt/_tensor/_nested_contraction.py module and the
%   _try_nested_contract dispatcher. Absolute and relative-non-periodic are
%   exact; relative-periodic uses the transposition-average surrogate and
%   warns when sigma/period exceeds the truncationSigmas-implied tolerance.
%   Quadrature node counts, recipe structure, tuple counts and the cost
%   estimate are integer-identical to the Python path so the two languages
%   take the same route; reduction order differs, so values agree to
%   floating-point (not bit-for-bit), as for the surrogate generally.

    if nargin < 5 || isempty(force); force = false; end
    triple = [];
    if ~any(strcmp(normalize, {'cosine', 'oneSidedDenom'}))
        declineContractIfForced(force, ...
            sprintf('unsupported normalisation ''%s''', normalize));
        return;
    end
    if densX.nAttrs ~= densY.nAttrs
        declineContractIfForced(force, ...
            'the two densities have different attribute counts');
        return;
    end
    if densX.nAttrs ~= 1
        % Nested attribute(s) tensored with further attributes: the cosine
        % factorises per event-pair across attributes (JMM Eq 3.4), so each
        % nested factor goes through the contraction and each plain factor
        % through the per-attribute MA matrix, instead of enumerating the
        % joint tuple.
        triple = nestedContractMA(densX, densY, normalize, ...
                                  truncationSigmas, force);
        return;
    end
    if ~isfield(densX, 'nested') || ~iscell(densX.nested) ...
            || ~isfield(densY, 'nested') || ~iscell(densY.nested)
        declineContractIfForced(force, ...
            'both densities must carry the same nested attribute');
        return;
    end
    specX = densX.nested{1};
    specY = densY.nested{1};
    if isempty(specX) || isempty(specY)
        declineContractIfForced(force, ...
            'both densities must carry the same nested attribute');
        return;
    end
    if localInnerR(specX) ~= 0 || localInnerR(specY) ~= 0
        declineContractIfForced(force, ...
            'an inner/intermediate [rel] unit is not yet covered');
        return;   % inner [rel] unit not covered by the contraction yet
    end

    PX = double(densX.pAttr{1});
    PY = double(densY.pAttr{1});
    WX = densX.w{1}; if isempty(WX); WX = ones(size(PX)); end
    WY = densY.w{1}; if isempty(WY); WY = ones(size(PY)); end
    WX = double(WX);
    WY = double(WY);
    % Variable-K (NaN-padded) values: a padded value is exactly equivalent
    % to a zero-weight value at any finite position (every tuple touching it
    % carries zero weight), so the contraction covers it by filling each
    % padded value with an in-range value at weight zero -- the same
    % NaN -> zero-weight idiom as mobius.maPerAttrInnerMatrix.
    mX = isnan(PX);
    mY = isnan(PY);
    if any(mX(:)) || any(mY(:))
        fillVal = min(min(PX(:), [], 'omitnan'), min(PY(:), [], 'omitnan'));
        PX(mX) = fillVal;  WX(mX | isnan(WX)) = 0;
        PY(mY) = fillVal;  WY(mY | isnan(WY)) = 0;
    end

    rLevels   = double(specX.r(:)).';
    symLevels = logical(specX.sym(:)).';
    % tags: rows index values, columns the inner tag levels (L-1 of them). A
    % MATLAB literal tag vector is a row, whereas buildRecipe takes the value
    % count from dimension 1 (matching the Python 1-D convention), so orient
    % single-level (L=2) tags as a column and undo any transposed matrix.
    tagsX = orientTags(double(specX.tags), size(PX, 1), numel(rLevels));
    tagsY = orientTags(double(specY.tags), size(PY, 1), numel(rLevels));
    % The two densities must agree on the per-level read-arities and [sym]
    % flags (same nested attribute); only the leaf cardinalities (tags shape)
    % may differ -- a 4-pitch prototype against an 8-pitch window, say.
    if ~isequal(rLevels, double(specY.r(:)).') ...
            || ~isequal(symLevels, logical(specY.sym(:)).')
        declineContractIfForced(force, ...
            'the two nested attributes differ in [r]/[sym]');
        return;
    end
    sameStruct = isequal(size(tagsX), size(tagsY)) && isequal(tagsX, tagsY);
    isRel  = logical(densX.isRel(1));
    isPer  = logical(densX.isPer(1));
    period = double(densX.period(1));
    sigma  = double(densX.sigma(1));
    % Resolve the truncation width once, at entry, through the shared
    % accuracy-floor resolver: [] -> the mptDefaults default, Inf -> the
    % finite accuracy-floor width (~7.43 sigma, the 1e-12 floor), a finite
    % value passes through unchanged. Per the toolbox contract Inf means
    % "accuracy-floor accuracy", NOT unbounded exact summation, so every
    % downstream isfinite(ts) gate here receives a finite width. (Genuinely
    % exhaustive summation is reachable only by widening the floor eps via
    % internal.accuracyFloor('setEps', ...), as golden regeneration does.)
    ts = internal.accuracyFloor('resolve', truncationSigmas);

    % One recipe per side: the X recipe indexes the X axis of the rectangular
    % leaf kernel, the Y recipe the Y axis. They coincide when the densities
    % share a nesting structure (the common case, incl. all XX/YY products).
    recipeX = buildRecipe(rLevels, symLevels, tagsX, isRel, isPer);
    if sameStruct
        recipeY = recipeX;
    else
        recipeY = buildRecipe(rLevels, symLevels, tagsY, isRel, isPer);
    end

    nX = size(PX, 2);
    nY = size(PY, 2);
    vmin = min(min(PX(:)), min(PY(:)));
    vmax = max(max(PX(:)), max(PY(:)));

    % Speed dispatch (integer/float counts; identical to Python).
    [mPerm, mComb] = tupleCounts(rLevels, symLevels, tagsX);
    work = recipeWork(recipeX);
    if ~sameStruct
        [mpY, mcY] = tupleCounts(rLevels, symLevels, tagsY);
        mPerm = max(mPerm, mpY);
        mComb = max(mComb, mcY);
        work = max(work, recipeWork(recipeY));
    end
    Q = quadNodes(isRel, isPer, sigma, period, vmin, vmax, ts);
    pairTerms = nX * nY + nX * nX + nY * nY;
    costEnum     = pairTerms * mPerm * mComb;
    costContract = pairTerms * Q * work;
    % The enumeration handles only matching nested cardinalities, so the cost
    % race (and its enumeration fallback) applies only when both sides share a
    % structure. When the cardinalities differ the contraction is the sole
    % correct route and is always taken.
    if sameStruct && costContract >= costEnum && ~force
        return;   % enumeration is the faster route (auto only)
    end

    if isRel && isPer
        if isfinite(ts)
            tol = max(exp(-0.5 * ts^2), 1e-12);
        else
            tol = 1e-12;
        end
        if tol < 1
            sopMax = 0.85 / (4 * sqrt(log(1 / tol)));
        else
            sopMax = Inf;
        end
        if sigma / period > sopMax
            warning('mpt:nestedSurrogateResolution', ...
                ['Nested relative-periodic similarity at sigma/period = ' ...
                 '%.3f exceeds the surrogate accuracy threshold %.3f ' ...
                 'implied by truncationSigmas (tolerance %.1e); the ' ...
                 'transposition-average value may depart from the exact ' ...
                 'inner product. Pass method=''bulger'' for the exact ' ...
                 'enumeration.'], sigma / period, sopMax, tol);
        end
    end

    quad = makeQuadrature(isRel, isPer, sigma, period, vmin, vmax, ts);

    ipxy = tripSum(recipeX, recipeY, PX, WX, PY, WY, sigma, period, ts, quad, false);
    ipxx = tripSum(recipeX, recipeX, PX, WX, PX, WX, sigma, period, ts, quad, true);
    ipyy = tripSum(recipeY, recipeY, PY, WY, PY, WY, sigma, period, ts, quad, true);
    triple = struct('xy', ipxy, 'xx', ipxx, 'yy', ipyy);
end


% ----------------------------------------------------------------------
function s = tripSum(recipeA, recipeB, PA, WA, PB, WB, sigma, period, ts, quad, sym)
    % Sum of the per-event-pair inner products over the pair grid.
    %
    % sym=true (self inner products): <e_i,e_j> = <e_j,e_i>, so evaluate
    % only the upper triangle and double the off-diagonal terms. recipeA /
    % recipeB index the PA / PB axes of the rectangular kernel.
    %
    % The contraction machinery is already batched over its leading axis
    % (contractNode reads Q = size(K, 1) and every helper below it is
    % generic in Q), which the absolute mode uses with Q = 1 and the
    % relative-periodic mode with Q = the transposition count. Folding the
    % event pairs into that same axis therefore needs no change to the
    % contraction itself: the pair grid is assembled into one kernel and
    % reduced in a single call per chunk, as on the Python side.
    %
    % The relative-non-periodic mode keeps the per-pair route, because its
    % factored shortcut (ipRelNonperFactored) is chosen per pair and has no
    % batched form.
    if nargin < 11; sym = false; end
    if batchableMode(recipeA, recipeB, PA, WA, PB, WB, quad)
        s = tripSumBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                           sigma, period, ts, quad, sym);
    else
        s = tripSumLooped(recipeA, recipeB, PA, WA, PB, WB, ...
                          sigma, period, ts, quad, sym);
    end
end


% ----------------------------------------------------------------------
function s = tripSumBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                            sigma, period, ts, quad, sym)
    % Sum over the event-pair grid, evaluated in batches.
    [mi, ni] = pairIndices(size(PA, 2), size(PB, 2), sym);
    mult = ones(numel(mi), 1);
    if sym
        % Upper triangle only, so off-diagonal pairs stand for two terms.
        mult(:) = 2.0;
        mult(mi == ni) = 1.0;
    end
    v = pairValuesBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                          sigma, period, ts, quad, mi, ni);
    s = sum(v .* mult);
end


% ----------------------------------------------------------------------
function [mi, ni] = pairIndices(nA, nB, sym)
    % Event-pair index lists. Under sym only the upper triangle is listed;
    % the caller supplies the multiplicity or scatters the transpose.
    if sym
        spans = max(nB - (1:nA) + 1, 0);
        nPairs = sum(spans);
        mi = zeros(nPairs, 1);
        ni = zeros(nPairs, 1);
        p = 0;
        for i = 1:nA
            span = spans(i);
            if span == 0; continue; end
            idx = p + (1:span);
            mi(idx) = i;
            ni(idx) = i:nB;
            p = p + span;
        end
    else
        mi = repelem((1:nA).', nB, 1);
        ni = repmat((1:nB).', nA, 1);
    end
end


% ----------------------------------------------------------------------
function v = pairValuesBatched(recipeA, recipeB, PA, WA, PB, WB, ...
                               sigma, period, ts, quad, mi, ni)
    % Per-event-pair inner products for the listed pairs, with the pairs
    % folded into the contraction's leading batch axis.
    %
    % contractNode and everything below it read the batch extent from
    % size(K, 1), so a batch carrying pairs (and, in relative-periodic mode,
    % pairs times transpositions) needs no change to the contraction. This
    % is the MATLAB form of the Python nested_attr_matrix, which folds the
    % same grid into the leading axis of its contraction.
    nPairs = numel(mi);
    nX = size(PA, 1);
    nY = size(PB, 1);
    v = zeros(nPairs, 1);
    if nPairs == 0
        return;
    end

    isRelPer = strcmp(quad.mode, 'relper');
    isRelNon = strcmp(quad.mode, 'relnonper');
    if isRelPer || isRelNon
        taus = quad.taus(:).';
        T = numel(taus);
    else
        taus = [];
        T = 1;
    end

    % Chunk so the assembled kernel stays within budget: it holds
    % T * nX * nY doubles per pair.
    memBudget = 16e6;
    chunk = max(1, min(nPairs, floor(memBudget / max(T * nX * nY, 1))));

    for c0 = 1:chunk:nPairs
        c1 = min(c0 + chunk - 1, nPairs);
        nb = c1 - c0 + 1;
        cm = mi(c0:c1);
        cn = ni(c0:c1);

        vx = PA(:, cm).';               % (nb, nX)
        vy = PB(:, cn).';               % (nb, nY)
        wx = WA(:, cm).';
        wy = WB(:, cn).';

        if isRelNon
            % Per-pair window into the line grid: taus outside it give
            % kernel entries below the truncation floor, which are zeroed
            % and then summed as exact zeros, so the window changes nothing
            % but the amount of arithmetic.
            [tw, valid] = tauWindow(vx, vy, taus, sigma, ts);
            if isempty(tw)
                tw = repmat(taus, nb, 1);
                valid = [];
            end
            W = size(tw, 2);
            d = reshape(vx, [nb, nX, 1, 1]) ...
                - (reshape(vy, [nb, 1, nY, 1]) + reshape(tw, [nb, 1, 1, W]));
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wx, [nb, nX, 1, 1]) .* reshape(wy, [nb, 1, nY, 1]));
            K = permute(K, [1, 4, 2, 3]);
            K = reshape(K, [nb * W, nX, nY]);
            K = truncK(K, ts);
            vc = contractNode(recipeA, recipeB, K);
            vc = reshape(vc, [nb, W]);
            if ~isempty(valid); vc = vc .* valid; end
            vc = sum(vc, 2);                     % common dtau cancels
        elseif isRelPer
            d = reshape(vx, [nb, nX, 1, 1]) ...
                - (reshape(vy, [nb, 1, nY, 1]) + reshape(taus, [1, 1, 1, T]));
            d = d - period * round(d / period);
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wx, [nb, nX, 1, 1]) .* reshape(wy, [nb, 1, nY, 1]));
            % (nb, nX, nY, T) -> (nb, T, nX, nY) -> (nb*T, nX, nY). The
            % merge is column-major, so the pair index runs fastest; the
            % reshape below inverts it the same way.
            K = permute(K, [1, 4, 2, 3]);
            K = reshape(K, [nb * T, nX, nY]);
            K = truncK(K, ts);
            vc = contractNode(recipeA, recipeB, K);
            vc = sum(reshape(vc, [nb, T]), 2);   % common dtau cancels
        else
            d = reshape(vx, [nb, nX, 1]) - reshape(vy, [nb, 1, nY]);
            if quad.isPer
                d = d - period * round(d / period);
            end
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wx, [nb, nX, 1]) .* reshape(wy, [nb, 1, nY]));
            K = truncK(K, ts);
            vc = contractNode(recipeA, recipeB, K);
            vc = vc(:);
        end

        v(c0:c1) = vc;
    end
end


% ----------------------------------------------------------------------
function tf = batchableMode(recipeX, recipeY, PA, WA, PB, WB, quad)
    % Whether the pair grid can be evaluated in batches.
    %
    % Absolute and relative-periodic modes always can. The relative-non-
    % periodic mode can whenever its closed-form shortcut cannot apply,
    % since that shortcut is chosen per pair and has no batched form: the
    % two structural gates are read from the recipes, and the third asks
    % whether every cell carries a shared leaf template. Where the shortcut
    % could fire the per-pair route is kept, so nothing is given up.
    tf = true;
    if any(strcmp(quad.mode, {'abs', 'relper'}))
        return;
    end
    if ~strcmp(quad.mode, 'relnonper')
        tf = false;
        return;
    end
    if recipeX.sym || recipeY.sym
        return;                        % shortcut needs ordered cells
    end
    if recipeX.r ~= numel(recipeX.children) ...
            || recipeY.r ~= numel(recipeY.children)
        return;                        % shortcut needs the whole cell
    end
    for i = 1:size(PA, 2)
        if isempty(sharedLeafTemplate(recipeX, PA(:, i), WA(:, i)))
            return;
        end
    end
    for j = 1:size(PB, 2)
        if isempty(sharedLeafTemplate(recipeY, PB(:, j), WB(:, j)))
            return;
        end
    end
    tf = false;                        % every cell can use the shortcut
end


% ----------------------------------------------------------------------
function [tw, valid] = tauWindow(vxT, vyT, taus, sigma, ts)
    % Per-pair slice of a uniform line tau-grid, or [] when not worthwhile.
    %
    % On the line the grid spans the whole value range, because any two
    % events may be that far apart, but one event pair aligns only over the
    % taus near its own offset. A kernel entry survives truncation when
    % |v_x - v_y - tau| <= 2*sigma*sqrt(-log(floor)), so the window runs
    % from min(v_x) - max(v_y) to max(v_x) - min(v_y), widened by that
    % margin at each end. Every node outside is zeroed by truncK and then
    % summed as an exact zero.
    %
    % All windows share one width so the pairs stay in a single batch: only
    % the start index varies, and valid masks the tail where a window
    % overruns its own end at the grid edges.
    tw = [];
    valid = [];
    T = numel(taus);
    if T < 3
        return;
    end
    step = taus(2) - taus(1);
    if ~isfinite(step) || step <= 0
        return;                        % not a uniform ascending grid
    end
    if isempty(ts) || ~isfinite(ts)
        return;                        % without truncation no node is removable
    end
    floorv = exp(-0.5 * ts^2);
    if ~(floorv > 0 && floorv < 1)
        return;
    end
    margin = 2 * sigma * sqrt(-log(floorv));

    t0 = taus(1);
    lo = min(vxT, [], 2) - max(vyT, [], 2) - margin;      % (nb, 1)
    hi = max(vxT, [], 2) - min(vyT, [], 2) + margin;
    startI = min(max(floor((lo - t0) / step), 0), T - 1);
    stopI  = min(max(ceil((hi - t0) / step), 0), T - 1);
    W = max(stopI - startI) + 1;
    if W >= T
        return;                        % the window is the grid; nothing saved
    end

    idx = startI + (0:W - 1);                            % (nb, W), 0-based
    valid = double(idx <= stopI);
    idx = min(idx, T - 1) + 1;                           % to 1-based
    tw = taus(idx);                                      % (nb, W)
end


% ----------------------------------------------------------------------
function s = tripSumLooped(recipeA, recipeB, PA, WA, PB, WB, ...
                           sigma, period, ts, quad, sym)
    % Per-event-pair route, retained for the relative-non-periodic mode.
    s = 0.0;
    nA = size(PA, 2);
    nB = size(PB, 2);
    for i = 1:nA
        ai = PA(:, i);
        wi = WA(:, i);
        if sym; j0 = i; else; j0 = 1; end
        for j = j0:nB
            v = nestedIp(recipeA, recipeB, ai, PB(:, j), wi, WB(:, j), ...
                         sigma, period, ts, quad);
            if sym && j ~= i
                s = s + 2.0 * v;
            else
                s = s + v;
            end
        end
    end
end


% ----------------------------------------------------------------------
%  Multi-attribute: nested factor(s) via the contraction, plain factors via
%  the per-attribute MA matrix; combine per event-pair (JMM Eq 3.4).
% ----------------------------------------------------------------------
function triple = nestedContractMA(densX, densY, normalize, truncationSigmas, force)
    triple = [];
    if ~any(strcmp(normalize, {'cosine', 'oneSidedDenom'}))
        declineContractIfForced(force, ...
            sprintf('unsupported normalisation ''%s''', normalize));
        return;
    end
    A = densX.nAttrs;
    % Resolve the truncation width through the shared accuracy-floor
    % resolver (see the entry note above): Inf -> the finite accuracy-floor
    % width, not unbounded exact summation.
    ts = internal.accuracyFloor('resolve', truncationSigmas);
    N_x = densX.N;
    N_y = densY.N;
    P_xy = ones(N_x, N_y);
    P_xx = ones(N_x, N_x);
    P_yy = ones(N_y, N_y);

    for a = 1:A
        isNestedX = isfield(densX, 'nested') && iscell(densX.nested) ...
            && numel(densX.nested) >= a && ~isempty(densX.nested{a});
        isNestedY = isfield(densY, 'nested') && iscell(densY.nested) ...
            && numel(densY.nested) >= a && ~isempty(densY.nested{a});
        sigma  = densX.sigma(a);
        isRel  = logical(densX.isRel(a));
        isPer  = logical(densX.isPer(a));
        period = densX.period(a);
        r_a    = densX.r(a);

        if ~isNestedX && ~isNestedY
            % Plain attribute: reuse the per-attribute MA matrix machinery.
            Px = densX.pAttr{a}; Wx = densX.w{a};
            Py = densY.pAttr{a}; Wy = densY.w{a};
            I_xy = mobius.maPerAttrInnerMatrix(Px, Wx, Py, Wy, ...
                sigma, r_a, isRel, isPer, period, 'truncationSigmas', ts);
            I_xx = mobius.maPerAttrInnerMatrix(Px, Wx, Px, Wx, ...
                sigma, r_a, isRel, isPer, period, 'truncationSigmas', ts);
            I_yy = mobius.maPerAttrInnerMatrix(Py, Wy, Py, Wy, ...
                sigma, r_a, isRel, isPer, period, 'truncationSigmas', ts);
            P_xy = P_xy .* I_xy;
            P_xx = P_xx .* I_xx;
            P_yy = P_yy .* I_yy;
            continue;
        end

        if isNestedX ~= isNestedY
            declineContractIfForced(force, ...
                'an attribute is nested on only one side');
            return;
        end
        specX = densX.nested{a};
        specY = densY.nested{a};
        if localInnerR(specX) ~= 0 || localInnerR(specY) ~= 0
            declineContractIfForced(force, ...
                'an inner/intermediate [rel] unit is not yet covered');
            return;
        end
        PXa = double(densX.pAttr{a});
        PYa = double(densY.pAttr{a});
        rLevels   = double(specX.r(:)).';
        symLevels = logical(specX.sym(:)).';
        if ~isequal(rLevels, double(specY.r(:)).') ...
                || ~isequal(symLevels, logical(specY.sym(:)).')
            declineContractIfForced(force, ...
                'the two nested attributes differ in [r]/[sym]');
            return;
        end
        tagsX = orientTags(double(specX.tags), size(PXa, 1), numel(rLevels));
        tagsY = orientTags(double(specY.tags), size(PYa, 1), numel(rLevels));
        sameStruct = isequal(size(tagsX), size(tagsY)) && isequal(tagsX, tagsY);
        WXa = densX.w{a}; if isempty(WXa); WXa = ones(size(PXa)); end
        WYa = densY.w{a}; if isempty(WYa); WYa = ones(size(PYa)); end
        WXa = double(WXa);
        WYa = double(WYa);
        % Variable-K (NaN-padded) values: fill with an in-range value at
        % weight zero (exactly equivalent; see the single-attribute path).
        mXa = isnan(PXa);
        mYa = isnan(PYa);
        if any(mXa(:)) || any(mYa(:))
            fillVal = min(min(PXa(:), [], 'omitnan'), ...
                min(PYa(:), [], 'omitnan'));
            PXa(mXa) = fillVal;  WXa(mXa | isnan(WXa)) = 0;
            PYa(mYa) = fillVal;  WYa(mYa | isnan(WYa)) = 0;
        end
        recipeX = buildRecipe(rLevels, symLevels, tagsX, isRel, isPer);
        if sameStruct
            recipeY = recipeX;
        else
            recipeY = buildRecipe(rLevels, symLevels, tagsY, isRel, isPer);
        end
        vmin = min(min(PXa(:)), min(PYa(:)));
        vmax = max(max(PXa(:)), max(PYa(:)));
        if isRel && isPer
            if isfinite(ts); tol = max(exp(-0.5 * ts^2), 1e-12); else; tol = 1e-12; end
            if tol < 1; sopMax = 0.85 / (4 * sqrt(log(1 / tol))); else; sopMax = Inf; end
            if sigma / period > sopMax
                warning('mpt:nestedSurrogateResolution', ...
                    ['Nested relative-periodic similarity at sigma/period = ' ...
                     '%.3f exceeds the surrogate accuracy threshold %.3f ' ...
                     'implied by truncationSigmas (tolerance %.1e); the ' ...
                     'transposition-average value may depart from the exact ' ...
                     'inner product. Pass method=''bulger'' for the exact ' ...
                     'enumeration.'], sigma / period, sopMax, tol);
            end
        end
        quad = makeQuadrature(isRel, isPer, sigma, period, vmin, vmax, ts);
        P_xy = P_xy .* nestedAttrInnerMatrix(recipeX, recipeY, ...
            PXa, PYa, WXa, WYa, sigma, period, ts, quad, false);
        P_xx = P_xx .* nestedAttrInnerMatrix(recipeX, recipeX, ...
            PXa, PXa, WXa, WXa, sigma, period, ts, quad, true);
        P_yy = P_yy .* nestedAttrInnerMatrix(recipeY, recipeY, ...
            PYa, PYa, WYa, WYa, sigma, period, ts, quad, true);
    end

    % The joint-tuple enumeration (bulger) mis-shapes a nested attribute's
    % per-event tuples in the MA tensor build, so a nested multi-attribute
    % density must not fall back to it. The contraction is competitive at any
    % size here (the nested factor dominates and the plain factors are the
    % fast per-attribute matrices), so always return the contracted triple.
    triple = struct('xy', sum(P_xy(:)), 'xx', sum(P_xx(:)), 'yy', sum(P_yy(:)));
end


function M = nestedAttrInnerMatrix(recipeA, recipeB, Pa, Pb, Wa, Wb, ...
                                   sigma, period, ts, quad, symmetric)
    % (N_a, N_b) per-event-pair inner matrix for one nested attribute via the
    % tree contraction. symmetric exploits <e_i,e_j> = <e_j,e_i> for the self
    % matrices. The per-attribute prefactor is constant and cancels in the
    % cosine when the attribute matrices are multiplied and summed.
    %
    % The absolute and relative-periodic modes evaluate the whole pair grid
    % in batches; the relative-non-periodic mode keeps the per-pair route,
    % whose factored shortcut is chosen per pair and has no batched form.
    na = size(Pa, 2);
    nb = size(Pb, 2);
    M = zeros(na, nb);

    if batchableMode(recipeA, recipeB, Pa, Wa, Pb, Wb, quad)
        [mi, ni] = pairIndices(na, nb, symmetric);
        v = pairValuesBatched(recipeA, recipeB, Pa, Wa, Pb, Wb, ...
                              sigma, period, ts, quad, mi, ni);
        M(sub2ind([na, nb], mi, ni)) = v;
        if symmetric
            M(sub2ind([na, nb], ni, mi)) = v;
        end
        return;
    end

    for i = 1:na
        ai = Pa(:, i); wi = Wa(:, i);
        if symmetric; j0 = i; else; j0 = 1; end
        for j = j0:nb
            v = nestedIp(recipeA, recipeB, ai, Pb(:, j), wi, Wb(:, j), ...
                         sigma, period, ts, quad);
            M(i, j) = v;
            if symmetric && j ~= i; M(j, i) = v; end
        end
    end
end


function declineContractIfForced(force, reason)
    if force
        error('cosSimExpTens:contractUnavailable', ...
            ['method=''contract'' is not available here: %s. Use ' ...
             'method=''auto'' or method=''bulger''.'], reason);
    end
end

function r = localInnerR(spec)
    r = 0;
    if isstruct(spec) && isfield(spec, 'proj') && isfield(spec, 'relUnit') ...
            && (strcmp(spec.proj, 'inner') || strcmp(spec.proj, 'intermediate'))
        u = spec.relUnit;
        r = prod(spec.r(1:u));
    end
end


% ----------------------------------------------------------------------
%  Recipe: tag tree + permutation/combination index arrays (built once)
% ----------------------------------------------------------------------
function recipe = buildRecipe(rLevels, symLevels, tags, isRel, isPer)
    L = numel(rLevels);
    Ktot = size(tags, 1);
    recipe = buildNode(L - 1, (1:Ktot).', rLevels, symLevels, tags, ...
                       isRel, isPer);
end


function node = buildNode(level, valIdx, rLevels, symLevels, tags, isRel, isPer)
    valIdx = valIdx(:);
    if level == 0
        r0 = rLevels(1);
        sy0 = symLevels(1);
        useOrb = orbitEligible(numel(valIdx), r0, sy0, isRel, isPer);
        if useOrb
            xt = zeros(0, r0); yt = zeros(0, r0);   % lazy: orbit needs no tuples
        else
            [xt, yt] = tupleIndices(numel(valIdx), r0, sy0);
        end
        node = struct('level', 0, 'valIdx', valIdx, 'children', {{}}, ...
                      'xtup', xt, 'ytup', yt, 'r', r0, 'sym', sy0, ...
                      'useOrbit', useOrb);
        return;
    end
    col = level;                       % 1-based tag column (Python col=level-1)
    keys = tags(valIdx, col);
    uk = unique(keys);                 % ascending
    children = cell(1, numel(uk));
    for c = 1:numel(uk)
        sub = valIdx(keys == uk(c));
        children{c} = buildNode(level - 1, sub, rLevels, symLevels, tags, ...
                                isRel, isPer);
    end
    rl = rLevels(level + 1);           % Python r_levels[level]
    syl = symLevels(level + 1);
    useOrb = orbitEligible(numel(children), rl, syl, isRel, isPer);
    if useOrb
        xt = zeros(0, rl); yt = zeros(0, rl);
    else
        [xt, yt] = tupleIndices(numel(children), rl, syl);
    end
    node = struct('level', level, 'valIdx', valIdx, 'children', {children}, ...
                  'xtup', xt, 'ytup', yt, 'r', rl, 'sym', syl, ...
                  'useOrbit', useOrb);
end


function tf = orbitEligible(g, r, sym, isRel, isPer)
    % Per-level orbit-vs-enumeration choice, reusing the shared flat policy
    % (internal.orbitBeatsPairwisePerAttr K-vs-r crossover +
    % internal.orbitSafeForPrecision g>=r+2 guard), applied with K = g. For
    % r in 7..8 (no K-threshold entry; enumeration's C(g,r)*r! infeasible)
    % orbit is the only viable route when precision-safe.
    ORBIT_R_MAX_SHIPPED = 8;     % match Python _ORBIT_R_MAX_SHIPPED
    tf = false;
    if ~sym || r < 2 || r > ORBIT_R_MAX_SHIPPED
        return;
    end
    if ~internal.orbitSafeForPrecision(r, g)   % precision guard (g >= r+2)
        return;
    end
    if r <= 6
        tf = internal.orbitBeatsPairwisePerAttr(r, g, isRel, isPer);
    else
        tf = true;                % r in 7..8: enumeration infeasible
    end
end


function [xt, yt] = tupleIndices(n, r, sym)
    % X side: permutations of r-combinations if sym, else combinations.
    % Y side: combinations. 1-based indices into 1:n.
    if r > n
        xt = zeros(0, r);
        yt = zeros(0, r);
        return;
    end
    if r == 1
        C = (1:n).';
    else
        C = nchoosek(1:n, r);          % (nC x r), each row a combination
    end
    yt = C;
    if sym && r > 1
        P = perms(1:r);                % (r! x r)
        nC = size(C, 1);
        nP = size(P, 1);
        xt = zeros(nC * nP, r);
        idx = 1;
        for p = 1:nP
            xt(idx:idx + nC - 1, :) = C(:, P(p, :));
            idx = idx + nC;
        end
    else
        xt = C;
    end
end


% ----------------------------------------------------------------------
%  Contraction (vectorised over the quadrature batch, dim 1)
% ----------------------------------------------------------------------
function v = contractNode(xn, yn, K)
    % Bottom-up, batched over the quadrature (dim 1) AND over sibling pairs.
    % K is (Q, nX, nY): the X axis is indexed by xn values, the Y axis by yn
    % values. For XX/YY (and equal-cardinality XY) xn and yn coincide and this
    % is the original single-tree walk; differing leaf spans are handled per
    % level by combinePair's per-size tuple sourcing.
    if xn.level == 0
        block = K(:, xn.valIdx, yn.valIdx);
        v = combinePair(block, xn.r, xn.sym, xn.useOrbit && yn.useOrbit);
    else
        Mc = subtreeOverlaps(xn.children, yn.children, K);
        v = combinePair(Mc, xn.r, xn.sym, xn.useOrbit && yn.useOrbit);
    end
end


function s = nodeSpan(node)
    if node.level == 0
        s = numel(node.valIdx);
    else
        s = numel(node.children);
    end
end


function tf = siblingsUniform(nodes)
    rep = nodes{1};
    span = nodeSpan(rep);
    tf = true;
    for k = 1:numel(nodes)
        nd = nodes{k};
        if nodeSpan(nd) ~= span || nd.r ~= rep.r || nd.sym ~= rep.sym ...
                || nd.useOrbit ~= rep.useOrbit
            tf = false; return;
        end
    end
end


function M = leafOverlaps(xnodes, ynodes, K)
    % (Q, gx, gy) pairwise overlaps among leaf siblings, X-side vs Y-side.
    gx = numel(xnodes);
    gy = numel(ynodes);
    Q = size(K, 1);
    nX = size(K, 2);
    nY = size(K, 3);
    r = xnodes{1}.r;
    sym = xnodes{1}.sym;
    if r == 1
        % r0 = 1: M(q,a,b) = sum_{i in Sxa, j in Syb} K(q,i,j) (weights folded).
        Gx = zeros(gx, nX);
        for a = 1:gx
            Gx(a, xnodes{a}.valIdx) = 1.0;
        end
        Gy = zeros(gy, nY);
        for b = 1:gy
            Gy(b, ynodes{b}.valIdx) = 1.0;
        end
        KG = reshape(reshape(K, [Q * nX, nY]) * Gy.', [Q, nX, gy]);  % (q,i,b)
        KGp = reshape(permute(KG, [2, 1, 3]), [nX, Q * gy]);          % (i, q*b)
        MG = Gx * KGp;                                                % (a, q*b)
        M = permute(reshape(MG, [gx, Q, gy]), [2, 1, 3]);            % (Q,gx,gy)
        return;
    end
    ux = siblingsUniform(xnodes);
    uy = siblingsUniform(ynodes);
    if ux && uy
        mx = numel(xnodes{1}.valIdx);
        my = numel(ynodes{1}.valIdx);
        blocks = zeros(gx, gy, Q, mx, my);
        for a = 1:gx
            sa = K(:, xnodes{a}.valIdx, :);
            for b = 1:gy
                blocks(a, b, :, :, :) = reshape(sa(:, :, ynodes{b}.valIdx), ...
                                                [1, 1, Q, mx, my]);
            end
        end
        useOrbit = xnodes{1}.useOrbit && ynodes{1}.useOrbit;
        vals = combinePair(reshape(blocks, [gx * gy * Q, mx, my]), ...
                           r, sym, useOrbit);
        M = permute(reshape(vals, [gx, gy, Q]), [3, 1, 2]);
        return;
    end
    M = zeros(Q, gx, gy);
    for a = 1:gx
        for b = 1:gy
            uo = xnodes{a}.useOrbit && ynodes{b}.useOrbit;
            M(:, a, b) = combinePair(K(:, xnodes{a}.valIdx, ynodes{b}.valIdx), ...
                                     r, sym, uo);
        end
    end
end


function M = subtreeOverlaps(xnodes, ynodes, K)
    % (Q, gx, gy) pairwise overlaps among sibling subtrees, X-side vs Y-side.
    if xnodes{1}.level == 0
        M = leafOverlaps(xnodes, ynodes, K);
        return;
    end
    gx = numel(xnodes);
    gy = numel(ynodes);
    Q = size(K, 1);
    r = xnodes{1}.r;
    sym = xnodes{1}.sym;
    xsizes = zeros(1, gx);
    xflat = {};
    for k = 1:gx
        xsizes(k) = numel(xnodes{k}.children);
        xflat = [xflat, xnodes{k}.children];   %#ok<AGROW>
    end
    ysizes = zeros(1, gy);
    yflat = {};
    for k = 1:gy
        ysizes(k) = numel(ynodes{k}.children);
        yflat = [yflat, ynodes{k}.children];   %#ok<AGROW>
    end
    xoffs = [0, cumsum(xsizes)];
    yoffs = [0, cumsum(ysizes)];
    Mc = subtreeOverlaps(xflat, yflat, K);          % (Q, Gcx, Gcy)
    ux = siblingsUniform(xnodes);
    uy = siblingsUniform(ynodes);
    if ux && uy
        gcx = xsizes(1);
        gcy = ysizes(1);
        blocks = zeros(gx, gy, Q, gcx, gcy);
        for a = 1:gx
            ra = xoffs(a) + 1 : xoffs(a) + gcx;
            for b = 1:gy
                cb = yoffs(b) + 1 : yoffs(b) + gcy;
                blocks(a, b, :, :, :) = reshape(Mc(:, ra, cb), ...
                                                [1, 1, Q, gcx, gcy]);
            end
        end
        useOrbit = xnodes{1}.useOrbit && ynodes{1}.useOrbit;
        vals = combinePair(reshape(blocks, [gx * gy * Q, gcx, gcy]), ...
                           r, sym, useOrbit);
        M = permute(reshape(vals, [gx, gy, Q]), [3, 1, 2]);
        return;
    end
    M = zeros(Q, gx, gy);
    for a = 1:gx
        ra = xoffs(a) + 1 : xoffs(a + 1);
        for b = 1:gy
            cb = yoffs(b) + 1 : yoffs(b + 1);
            uo = xnodes{a}.useOrbit && ynodes{b}.useOrbit;
            M(:, a, b) = combinePair(Mc(:, ra, cb), r, sym, uo);
        end
    end
end


function v = combinePair(M, r, sym, useOrbit)
    % Combine a (Q, gx, gy) block at one level: X-side perm tuples over gx,
    % Y-side comb tuples over gy (the r!-cancelled perm x comb form, same
    % scale as combine). gx and gy are read from the block, so unequal X/Y
    % spans -- ragged siblings *or* two densities whose nested cardinalities
    % differ -- are handled directly. For a square block with gx == gy this
    % reproduces the old combineNode exactly, so the X == Y path is unchanged.
    if useOrbit
        empt = zeros(0, r);
        v = combineOrbit(M, r, empt, empt);
        return;
    end
    gx = size(M, 2);
    gy = size(M, 3);
    [xt, ~] = tupleIndices(gx, r, sym);
    [~, yt] = tupleIndices(gy, r, sym);
    v = combine(M, xt, yt);
end


function v = combineOrbit(M, r, xtup, ytup)
    % (B,) = Sum_{cX,cY} perm(M[cX,cY]) via the partition-lattice orbit
    % reduction (= innerProductOrbitGrid / r!), vectorised over the leading
    % batch, with a cancellation guard reverting to enumeration where
    % feasible. Supports rectangular M (gx ~= gy).
    gx = size(M, 2);
    gy = size(M, 3);
    [vals, ratios] = mobius.innerProductOrbitGrid(M, ones(gx, 1), ...
        ones(gy, 1), r, 'prefactor', 1.0, 'returnCancellationRatio', true);
    vals = vals(:) / factorial(r);
    bad = ratios(:) < 1e-10;            % _ORBIT_CANCEL_FLOOR
    if any(bad)
        if size(xtup, 1) > 0
            idx = find(bad);
            vals(idx) = combine(M(idx, :, :), xtup, ytup);
        else
            warning('mpt:nestedOrbitCancellation', ...
                ['Nested orbit reduction lost precision to alternating-sum ' ...
                 'cancellation at a symmetric level where enumeration is ' ...
                 'infeasible; the value may be inaccurate.']);
        end
    end
    v = vals;
end


function v = combine(M, xtup, ytup)
    Tx = size(xtup, 1);
    Ty = size(ytup, 1);
    if Tx == 0 || Ty == 0
        v = zeros(size(M, 1), 1);
        return;
    end
    r = size(xtup, 2);
    P = M(:, xtup(:, 1), ytup(:, 1));              % B x Tx x Ty
    for t = 2:r
        P = P .* M(:, xtup(:, t), ytup(:, t));
    end
    v = sum(sum(P, 3), 2);                          % B x 1
    v = v(:);
end


% ----------------------------------------------------------------------
%  Leaf-kernel batches + per-event-pair bare inner product
% ----------------------------------------------------------------------
function ipv = nestedIp(recipeX, recipeY, vX, vY, wX, wY, sigma, period, ts, quad)
    nX = numel(vX);
    nY = numel(vY);
    switch quad.mode
        case 'abs'
            % Periodicity comes from the density's [per] flag (carried in the
            % quadrature struct), not from whether period happens to be
            % finite: an absolute non-periodic attribute may carry a finite
            % period.
            d = reshape(vX, [1, nX, 1]) - reshape(vY, [1, 1, nY]);   % 1 x nX x nY
            if quad.isPer
                d = d - period * round(d / period);
            end
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wX, [1, nX, 1]) .* reshape(wY, [1, 1, nY]));
            K = truncK(K, ts);
            v = contractNode(recipeX, recipeY, K);
            ipv = v(1);
        case 'relper'
            taus = quad.taus(:);
            T = numel(taus);
            d = reshape(vX, [1, nX, 1]) ...
                - (reshape(vY, [1, 1, nY]) + reshape(taus, [T, 1, 1]));  % T x nX x nY
            d = d - period * round(d / period);
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wX, [1, nX, 1]) .* reshape(wY, [1, 1, nY]));
            K = truncK(K, ts);
            ipv = sum(contractNode(recipeX, recipeY, K));   % common dtau cancels
        case 'relnonper'
            taus = quad.taus(:);
            ipv = ipRelNonperFactored(recipeX, recipeY, vX, vY, wX, wY, ...
                                      sigma, ts, taus);
            if isempty(ipv)
                T = numel(taus);
                d = reshape(vX, [1, nX, 1]) ...
                    - (reshape(vY, [1, 1, nY]) + reshape(taus, [T, 1, 1]));
                K = exp(-d.^2 / (4 * sigma^2));               % no wrap
                K = K .* (reshape(wX, [1, nX, 1]) .* reshape(wY, [1, 1, nY]));
                K = truncK(K, ts);
                ipv = sum(contractNode(recipeX, recipeY, K));
            end
    end
end


function tpl = sharedLeafTemplate(node, v, w)
%SLOTSHAREDLEAFTEMPLATE Detect a spectral-augmentation leaf (mirror of the
%   Python _shared_leaf_template): a two-level node whose children are all
%   r == 1 leaves sharing one partial template (a common offset and weight
%   profile, translated per child by a single reference value). Returns a struct with
%   fields refVals/off/wt, or [] when the node is not of this form.
    tpl = [];
    if node.level ~= 1 || isempty(node.children)
        return;
    end
    rep = node.children{1};
    if rep.level ~= 0 || rep.r ~= 1 || ~isempty(rep.children)
        return;
    end
    s0 = rep.valIdx(:);
    width = numel(s0);
    if width < 2                       % Kp == 1 is a plain fundamental: leave
        return;                        % it on the generic path (no change)
    end
    v0 = v(s0);
    w0 = w(s0);
    off = v0 - v0(1);
    g = numel(node.children);
    refVals = zeros(g, 1);
    for a = 1:g
        ch = node.children{a};
        if ch.level ~= 0 || ch.r ~= 1 || ~isempty(ch.children)
            return;
        end
        sa = ch.valIdx(:);
        if numel(sa) ~= width
            return;
        end
        va = v(sa);
        if ~isequal(va - va(1), off) || ~isequal(w(sa), w0)
            return;
        end
        refVals(a) = va(1);
    end
    tpl = struct('refVals', refVals, 'off', off(:), 'wt', w0(:));
end


function ipv = ipRelNonperFactored(recipeX, recipeY, vX, vY, wX, wY, ...
                                   sigma, ts, taus)
%IPRELNONPERFACTORED Closed-form inner-partial reduction of the relative-non-
%   periodic inner product for spectrally-augmented ordered cells (mirror of
%   the Python _ip_rel_nonper_factored). The inner partial index sums into the
%   template cross-correlation g, and the cell overlap reduces to the reference-value
%   differences: sum_tau prod_a g(carrierX_a - carrierY_a - tau). Returns [] when
%   the structure is not of this form (then the caller uses the generic path).
    ipv = [];
    if recipeX.sym || recipeY.sym
        return;                        % need ordered cells (outer [sym] = 0)
    end
    if recipeX.r ~= numel(recipeX.children) ...
            || recipeY.r ~= numel(recipeY.children)
        return;                        % need the whole cell as one ordered tuple
    end
    tx = sharedLeafTemplate(recipeX, vX, wX);
    ty = sharedLeafTemplate(recipeY, vY, wY);
    if isempty(tx) || isempty(ty)
        return;
    end
    g = numel(tx.refVals);
    if g ~= numel(ty.refVals)         % diagonal needs equal cell lengths
        return;
    end
    dpq = tx.off - ty.off.';                          % Kx x Ky
    wpq = tx.wt * ty.wt.';                            % Kx x Ky
    Kx = size(dpq, 1);
    Ky = size(dpq, 2);
    T = numel(taus);
    delta = reshape(tx.refVals - ty.refVals, [1, g]) ...
            - reshape(taus, [T, 1]);                  % T x g
    arg = reshape(delta, [T, g, 1, 1]) ...
          + reshape(dpq, [1, 1, Kx, Ky]);             % T x g x Kx x Ky
    K = exp(-arg.^2 / (4 * sigma^2)) .* reshape(wpq, [1, 1, Kx, Ky]);
    if ~isempty(ts) && isfinite(ts)
        floorv = exp(-0.5 * ts^2);     % per-term floor, matching truncK exactly
        K(K < floorv) = 0;
    end
    mDiag = sum(sum(K, 4), 3);                        % T x g
    ipv = sum(prod(mDiag, 2));         % common dtau cancels in the cosine
end


function K = truncK(K, ts)
    if isempty(ts) || ~isfinite(ts)
        return;
    end
    floorv = exp(-0.5 * ts^2);
    K(K < floorv) = 0;
end


% ----------------------------------------------------------------------
%  Quadrature (shared across event-pairs and the IP triple)
% ----------------------------------------------------------------------
function quad = makeQuadrature(isRel, isPer, sigma, period, vmin, vmax, ts)
    if ~isRel
        quad = struct('mode', 'abs', 'isPer', logical(isPer));
        return;
    end
    if isfinite(ts)
        tol = max(exp(-0.5 * ts^2), 1e-12);
    else
        tol = 1e-12;
    end
    if isPer
        ntau = internal.autoNtauDefault(period, sigma);
        t = linspace(0, period, ntau + 1);
        quad = struct('mode', 'relper', 'taus', t(1:end - 1));  % endpoint=false
    else
        spread = vmax - vmin;
        pad = (6 + 0.5 * max(0, -log10(max(tol, 1e-16)))) * sigma;
        hi = spread + pad;
        n = max(64, ceil(2 * hi / (sigma / 4)));
        quad = struct('mode', 'relnonper', 'taus', linspace(-hi, hi, n));
    end
end


function Q = quadNodes(isRel, isPer, sigma, period, vmin, vmax, ts)
    if ~isRel
        Q = 1;
        return;
    end
    if isfinite(ts)
        tol = max(exp(-0.5 * ts^2), 1e-12);
    else
        tol = 1e-12;
    end
    if isPer
        Q = internal.autoNtauDefault(period, sigma);
    else
        spread = vmax - vmin;
        pad = (6 + 0.5 * max(0, -log10(max(tol, 1e-16)))) * sigma;
        Q = max(64, ceil(2 * (spread + pad) / (sigma / 4)));
    end
end


% ----------------------------------------------------------------------
%  Analytic tuple counts (elementary symmetric polynomials) + tree work
% ----------------------------------------------------------------------
function [mPerm, mComb] = tupleCounts(rLevels, symLevels, tags)
    L = numel(rLevels);
    Ktot = size(tags, 1);
    mPerm = countSide((1:Ktot).', L - 1, true, rLevels, symLevels, tags);
    mComb = countSide((1:Ktot).', L - 1, false, rLevels, symLevels, tags);
end


function c = countSide(valIdx, level, useSym, rLevels, symLevels, tags)
    if level == 0
        r0 = rLevels(1);
        c = nchoosekCount(numel(valIdx), r0);
        if useSym && symLevels(1)
            c = c * factorial(r0);
        end
        return;
    end
    col = level;
    keys = tags(valIdx, col);
    uk = unique(keys);
    subs = zeros(1, numel(uk));
    for g = 1:numel(uk)
        sub = valIdx(keys == uk(g));
        subs(g) = countSide(sub, level - 1, useSym, rLevels, symLevels, tags);
    end
    rl = rLevels(level + 1);
    c = elemSym(subs, rl);
    if useSym && symLevels(level + 1)
        c = c * factorial(rl);
    end
end


function v = elemSym(xs, k)
    % Elementary symmetric polynomial e_k of xs (e_0 = 1; 0 if k > numel).
    e = zeros(1, k + 1);
    e(1) = 1;
    for ii = 1:numel(xs)
        for j = k + 1:-1:2
            e(j) = e(j) + e(j - 1) * xs(ii);
        end
    end
    v = e(k + 1);
end


function c = nchoosekCount(nn, kk)
    if kk < 0 || kk > nn
        c = 0;
        return;
    end
    c = 1;
    for ii = 0:kk - 1
        c = c * (nn - ii) / (ii + 1);
    end
    c = round(c);
end


function w = recipeWork(node)
    % Orbit-eligible symmetric levels are costed at the orbit reduction's
    % |Omega_r| * g^2 * r rather than the enumerated r! * C(g,r)^2, so the
    % dispatch reflects the route actually taken at each level (mirrors the
    % Python recipe_work).
    if node.useOrbit
        g = nodeSpan(node);
        w = numel(mobius.getOrbitTable(node.r)) * g * g * max(1, node.r);
    else
        w = size(node.xtup, 1) * size(node.ytup, 1) * max(1, node.r);
    end
    if node.level ~= 0
        g = numel(node.children);
        w = w + g * g;
        for c = 1:g
            w = w + recipeWork(node.children{c});
        end
    end
end


function tg = orientTags(tg, nValues, L)
    % Orient a tag array so values index dimension 1 and the L-1 inner tag
    % levels index dimension 2 (the convention buildRecipe expects, matching
    % the Python 1-D tags). A single-level (L == 2) spec is a vector -- a
    % MATLAB literal makes it a row -- so reshape it to a column; an already
    % oriented matrix is left as is, a transposed one is corrected.
    if isvector(tg)
        tg = tg(:);
    elseif size(tg, 1) ~= nValues && size(tg, 2) == nValues
        tg = tg.';
    end
    if size(tg, 1) ~= nValues && L >= 2   %#ok<BDLGI> defensive: keep values on dim 1
        tg = reshape(tg, nValues, []);
    end
end

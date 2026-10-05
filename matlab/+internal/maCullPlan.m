function [plan, C] = maCullPlan(varargin)
%MACULLPLAN  Culling plan for the joint-centres path of evalMaet.
%
%   [PLAN, C] = INTERNAL.MACULLPLAN(CENTRES, XC, NJ, NQ, DIMPERATTR,
%   SIGMA, ISREL, ISPER, PERIOD, INNERR, WRAPCELL, K, QDTYPE) chooses a
%   culling coordinate and finds each query's run of centres. PLAN is []
%   where the dense evaluation is expected to be no slower, and otherwise
%   a struct with fields
%       order  the permutation of the centres that sorts the culling
%              coordinate (with the centres near either end repeated
%              across the wrap when that coordinate is periodic);
%       lo, hi 1 x NQ: query q meets the centres order(lo(q)+1:hi(q)).
%   C holds the constants below (C.cacheChunkBytes, C.pairCost, ...),
%   returned whatever the plan.
%
%   C = INTERNAL.MACULLPLAN('constants') returns the constants alone (the
%   cost model internal.maEvalCostsMs reads C.pairCost).
%
%   [Q, STARTS] = INTERNAL.MACULLPLAN('blocks', LO, HI, NJ, GROUPPAIRS,
%   BLOCKCOST) groups the queries into the blocks evalMaet evaluates: Q
%   holds the queries whose runs are not empty, in the order of their
%   runs, and STARTS the position in Q at which each block starts (the
%   last ending at numel(Q)). See "Blocks" below.
%
%   PREV = INTERNAL.MACULLPLAN('override', S) replaces, for later calls,
%   the constants named by the fields of the struct S, and returns the
%   overrides it replaced; struct() clears them. For benchmarks.
%
%   N = INTERNAL.MACULLPLAN('countBelow', S, T, INCLUSIVE) is, for each
%   entry of T, the number of entries of the sorted vector S strictly
%   below it (INCLUSIVE false) or at or below it (INCLUSIVE true): the
%   two sides of NumPy's searchsorted.
%
%   The joint-centres path pairs every joint centre with every query, so
%   its cost is nJ * nQ however narrow the kernel. Two measures keep it
%   down.
%
%   * Cache-sized chunks. A dense chunk's working set is held to
%     C.cacheChunkBytes, the kernelChunkBytes budget remaining the
%     ceiling: a block sized from the memory budget alone is
%     memory-bound.
%   * Culling. Where one coordinate of the joint space bounds the
%     exponent from below (a culling coordinate), each query meets only
%     the centres within the truncation width on that coordinate. The
%     centres are sorted on it once per call, and each query's centres
%     are then one contiguous run of the sorted order.
%
%   Which coordinates bound the exponent. The exponent is the sum over
%   attributes of Q_a / (2 sigma_a^2), every term nonnegative, and a
%   pair whose exponent exceeds k^2 / 2 (k = truncationSigmas) is set to
%   zero.
%
%   * Absolute attribute: Q_a = sum_j d_j^2 >= d_j^2 for each coordinate
%     j, so |d_j| > k sigma_a puts the pair beyond the truncation width.
%     An attribute with a kernel covariance arrives whitened, as an
%     absolute attribute with sigma = 1, and is no different.
%   * Periodic absolute attribute, single image: the same, with d_j the
%     wrapped difference, so the window is taken on the circle.
%   * Relative attribute, flat or a nested inner unit: the coordinates
%     held are the differences D_j = d_{j+1} - d_0 from position 0, and
%     Q_a is the sum over all pairs of positions of the squared
%     (wrapped, if periodic) difference, divided by r. The pair (0, j+1)
%     contributes D_j^2, and each of the r - 2 other positions
%     contributes two pairs whose differences sum to D_j, so whose
%     squares sum to at least D_j^2 / 2 (a wrapped difference obeys the
%     triangle inequality); hence Q_a >= D_j^2 / 2, and the window is
%     sqrt(2) k sigma_a.
%   * Periodic absolute attribute, full image: not a culling coordinate.
%     Its wrapped Gaussian is evaluated untruncated, so a centre beyond
%     the window still contributes, if below the accuracy floor.
%
%   A culled pair is one that the truncation would have set to zero, so
%   the result is the dense result summed in a different order. The
%   window is widened by a small margin so that rounding in the exponent
%   cannot keep a pair the window excluded; on a periodic coordinate it
%   must stay shorter than the period, so that no centre falls in it
%   twice, and a coordinate whose widened window would not is passed
%   over.
%
%   Blocks. Both ends of a run move forward with the query's coordinate,
%   so neighbouring queries' runs together cover one contiguous range of
%   the sorted order. evalMaet takes the queries in blocks of neighbours
%   and evaluates each block densely over its range: a slice of the
%   centres, the sum over them a matrix-vector product. A centre in the
%   range but outside a query's own run lies beyond the truncation width
%   and contributes nothing. From where the last block ended, each block
%   is the longest that (i) holds no centre twice (a periodic coordinate
%   repeats the centres near either end of the cycle across the wrap, so
%   a range wider than nJ would), (ii) holds at most GROUPPAIRS pairs,
%   and (iii) evaluates at most BLOCKCOST pairs beyond those in its
%   queries' own runs; a block of one query is always accepted. All
%   three counts grow with the block, so its length is found by doubling
%   and then halving. The last condition weighs the pairs a block wastes
%   against the fixed cost of another block: for evenly spread queries,
%   the block whose wasted pairs equal that cost is the one whose cost
%   per query, fixed and wasted together, is least.
%
%   The coordinate is the one whose centres spread furthest relative to
%   its window. Costs are counted in pairs of the dense broadcast: the
%   dense evaluation costs nJ * nQ; the culled one costs the sort plus
%   C.pairCost per pair in the runs (the blocks' wasted pairs and fixed
%   cost spread over them). An estimate from the spread alone rules out
%   a sort that cannot pay; after the sort the pairs are counted
%   exactly. internal.maCullMode forces the choice, for testing.
%
%   Twin of the Python _ma_cull_plan, _ma_cull_candidates,
%   _ma_cull_blocks, and the constants beside them in
%   mpt/_tensor/eval.py.
%
%   See also EVALMAET, INTERNAL.MACULLMODE.

    persistent overrides
    if isempty(overrides)
        overrides = struct();
    end
    if ischar(varargin{1}) && strcmp(varargin{1}, 'override')
        plan = overrides;
        localConstants(varargin{2});   % rejects an unknown name
        overrides = varargin{2};
        C = localConstants(overrides);
        return;
    end
    C = localConstants(overrides);
    if ischar(varargin{1}) && strcmp(varargin{1}, 'countBelow')
        plan = localCountBelow(varargin{2}, varargin{3}, varargin{4});
        return;
    end
    if ischar(varargin{1}) && strcmp(varargin{1}, 'constants')
        plan = C;
        return;
    end
    if ischar(varargin{1}) && strcmp(varargin{1}, 'blocks')
        [plan, C] = localBlocks(varargin{2:6});
        return;
    end
    [Centres, Xc, nJ, nQ, dimPerAttr, sigmaG, isRelG, isPerG, periodG, ...
        innerR, wrapCell, kTrunc, qDtype] = varargin{:};

    plan = [];
    mode = internal.maCullMode();
    if strcmp(mode, 'never') || nJ == 0 || nQ == 0
        return;
    end
    cands = localCandidates(dimPerAttr, sigmaG, isRelG, isPerG, ...
                            periodG, innerR, wrapCell, kTrunc);
    if isempty(cands)
        return;
    end

    best = 0;
    bestRatio = -Inf;
    for i = 1:size(cands, 1)
        a = cands(i, 1);
        P = cands(i, 4);
        h = cands(i, 3);
        if P > 0
            % Widened, the window must stay shorter than the cycle, or it
            % would reach round onto itself and meet a centre twice.
            hmi = localWidened(h, P, ...
                localKey(Centres{a}(cands(i, 2), :), qDtype), ...
                localKey(Xc{a}(cands(i, 2), :), qDtype), qDtype, C);
            if ~(2 * hmi < P)
                continue;
            end
            ratio = P / (2 * h);
        else
            c = localKey(Centres{a}(cands(i, 2), :), qDtype);
            c = c(isfinite(c));
            if isempty(c)
                continue;
            end
            ratio = (max(c) - min(c)) / (2 * h);
        end
        if ratio > bestRatio
            bestRatio = ratio;
            best = i;
        end
    end
    if best == 0
        return;
    end
    a = cands(best, 1);
    j = cands(best, 2);
    h = cands(best, 3);
    P = cands(best, 4);

    dense = double(nJ) * double(nQ);
    sortCost = C.sortCost * double(nJ) * log2(max(double(nJ), 2));
    if ~strcmp(mode, 'always')
        if bestRatio > 0
            share = min(1, 1 / bestRatio);
        else
            share = 1;
        end
        if sortCost + C.pairCost * share * dense >= dense
            return;
        end
    end

    kc = localKey(Centres{a}(j, :), qDtype);
    kx = localKey(Xc{a}(j, :), qDtype);
    hm = localWidened(h, P, kc, kx, qDtype, C);
    if P > 0
        kc = mod(kc, P);
        kx = mod(kx, P);
    end
    [s, order] = sort(kc);
    if P > 0
        pre = s > P - hm;
        suf = s < hm;
        s = [s(pre) - P, s, s(suf) + P];
        order = [order(pre), order, order(suf)];
    end
    lo = localCountBelow(s, kx - hm, false);
    hi = localCountBelow(s, kx + hm, true);
    hi = max(hi, lo);
    if ~strcmp(mode, 'always')
        if sortCost + C.pairCost * sum(hi - lo) >= dense
            return;
        end
    end
    plan = struct('order', order, 'lo', lo, 'hi', hi);
end


function C = localConstants(overrides)
%LOCALCONSTANTS  Constants of the joint-centres path, with OVERRIDES (a
%   struct, possibly empty of fields) applied.
%   cacheChunkBytes  cache-scale cap, in bytes, on the working set of a
%                    dense chunk and of a block of culled queries;
%   pairCost         cost of a pair in a culled query's run, in units of
%                    one pair of the dense broadcast (the blocks' wasted
%                    pairs and fixed cost spread over the runs);
%   blockCost        the fixed cost of a block of culled queries, in the
%                    same unit: the most pairs a block may evaluate
%                    beyond its queries' own runs;
%   sortCost         cost of sorting the centres, per centre and per
%                    factor of two in their number, in the same unit;
%   marginDouble,    relative widening of the culling window at each
%   marginSingle     working precision.
%   The same values as the Python constants (_MA_CACHE_CHUNK_BYTES,
%   _MA_CULL_BLOCK_COST, _MA_CULL_SORT_COST, _MA_CULL_MARGIN), except
%   pairCost, which is measured per language: the ratio printed by
%   bench_ma_joint_cull where the decision to cull is close (a third of
%   the centres in each run), 2.42 on the maintainer's Mac in October
%   2026 against Python's 1.24. MATLAB spreads the dense broadcast over
%   every core, but not a block's smaller arrays; on one thread the two
%   languages' ratios agree.
    C = struct('cacheChunkBytes', 8 * 2^20, 'pairCost', 2.5, ...
               'blockCost', 4096, 'sortCost', 1.0, 'marginDouble', 1e-9, ...
               'marginSingle', 1e-4);
    names = fieldnames(overrides);
    for i = 1:numel(names)
        if ~isfield(C, names{i})
            error('maCullPlan:unknownConstant', ...
                  'No constant named ''%s''.', names{i});
        end
        C.(names{i}) = overrides.(names{i});
    end
end


function hm = localWidened(h, P, kc, kx, qDtype, C)
%LOCALWIDENED  The half-width H widened by the margin, and by rounding in
%   differences of keys as large as any held.
    scale = P;
    fc = abs(kc(isfinite(kc)));
    if ~isempty(fc)
        scale = max(scale, max(fc));
    end
    fx = abs(kx(isfinite(kx)));
    if ~isempty(fx)
        scale = max(scale, max(fx));
    end
    if strcmp(qDtype, 'single')
        margin = C.marginSingle;
    else
        margin = C.marginDouble;
    end
    hm = h * (1 + margin) + 8 * double(eps(qDtype)) * scale;
end


function [q, starts] = localBlocks(lo, hi, nJ, groupPairs, blockCost)
%LOCALBLOCKS  The blocks of queries evalMaet evaluates (see "Blocks"
%   above): Q the queries whose runs are not empty, in the order of their
%   runs, and STARTS the position in Q at which each block starts.
    lo = double(lo(:).');
    hi = double(hi(:).');
    q = find(hi > lo);
    starts = zeros(1, 0);
    if isempty(q)
        return;
    end
    [~, ord] = sortrows([lo(q).', hi(q).']);
    q = q(ord);
    los = lo(q);
    his = hi(q);
    cum = [0, cumsum(his - los)];
    n = numel(q);
    starts = zeros(1, n);
    nb = 0;
    i = 1;
    while i <= n
        nb = nb + 1;
        starts(nb) = i;
        m = 1;
        step = 1;
        while i + m + step - 1 <= n && localFits(los, his, cum, i, ...
                m + step, nJ, groupPairs, blockCost)
            m = m + step;
            step = 2 * step;
        end
        bad = min(m + step, n - i + 2);
        while bad - m > 1
            mid = floor((m + bad) / 2);
            if localFits(los, his, cum, i, mid, nJ, groupPairs, blockCost)
                m = mid;
            else
                bad = mid;
            end
        end
        i = i + m;
    end
    starts = starts(1:nb);
end


function tf = localFits(los, his, cum, i, len, nJ, groupPairs, blockCost)
%LOCALFITS  Whether the block of LEN queries from position I meets the
%   three bounds of localBlocks.
    j = i + len - 1;
    width = his(j) - los(i);
    pairs = width * len;
    tf = width <= nJ && pairs <= groupPairs ...
         && pairs - (cum(j + 1) - cum(i)) <= blockCost;
end


function cands = localCandidates(dimPerAttr, sigmaG, isRelG, isPerG, ...
                                 periodG, innerR, wrapCell, kTrunc)
%LOCALCANDIDATES  Culling coordinates of the joint space, one row each:
%   [attribute, row within its centres, window half-width, period (0
%   when not periodic)].
    cands = zeros(0, 4);
    for a = 1:numel(dimPerAttr)
        da = dimPerAttr(a);
        if da == 0
            continue;
        end
        isRel = isRelG(a) || innerR(a) > 0;
        isPer = isPerG(a);
        if isPer && ~isRel && strcmp(char(wrapCell{a}), 'full-image')
            continue;
        end
        h = double(kTrunc) * double(sigmaG(a));
        if isRel
            h = h * sqrt(2);
        end
        if isPer
            P = double(periodG(a));
        else
            P = 0;
        end
        if ~(h > 0) || (isPer && ~(2 * h < P))
            continue;
        end
        cands = [cands; [repmat(a, da, 1), (1:da).', ...
                         repmat([h, P], da, 1)]]; %#ok<AGROW>
    end
end


function k = localKey(v, qDtype)
%LOCALKEY  A culling coordinate as the evaluation sees it: cast to the
%   working precision, then held in double for the comparisons.
    k = double(cast(v(:).', qDtype));
end


function n = localCountBelow(s, t, inclusive)
%LOCALCOUNTBELOW  For sorted S, the number of its entries strictly below
%   (INCLUSIVE false) or at or below (INCLUSIVE true) each entry of T.
%   One merged sort: the stable order places an entry of T before equal
%   entries of S when they are not to be counted, and after them when
%   they are.
    s = s(:).';
    t = t(:).';
    ns = numel(s);
    nt = numel(t);
    n = zeros(1, nt);
    if nt == 0
        return;
    end
    if inclusive
        [~, ord] = sort([s, t]);
        isT = ord > ns;
        tIdx = ord(isT) - ns;
    else
        [~, ord] = sort([t, s]);
        isT = ord <= nt;
        tIdx = ord(isT);
    end
    sBefore = cumsum(~isT);
    n(tIdx) = sBefore(isT);
end

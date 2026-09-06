function vals = evalNestedAttrOrbit(p, w, tags, rLevels, symLevels, relUnit, ...
                                    sigma, x, opts)
%MOBIUS.EVALNESTEDATTRORBIT  Per-level Möbius point evaluator for a nested
%   attribute (one event).
%
%   vals = mobius.evalNestedAttrOrbit(p, w, tags, rLevels, symLevels, ...
%                                     relUnit, sigma, x, 'is_per', tf, ...
%                                     'period', P, 'wrap', wrap, ...
%                                     'truncationSigmas', ts, ...
%                                     'samplesPerSigma', spp)
%
%   A nested attribute's per-event density is a recursive construction
%   over the tag tree: at the leaf level an r_1-tuple of distinct values
%   from one finest group, at each higher level an r_l-tuple of distinct
%   level-(l-1) sub-tuples. The tuple-centres route materialises every
%   nested tuple (M_perm per event, a product of per-level factorials and
%   binomials) and sums a Gaussian per centre. This evaluator computes the
%   same density without materialising any tuple, by applying the Möbius
%   set-partition decomposition LEVEL BY LEVEL:
%
%     - at a symmetric level, the sum over ordered r-tuples of DISTINCT
%       children is written by inclusion-exclusion over the set partitions
%       pi of the r tuple slots,
%           sum_distinct prod_t M(c_t, t)
%               = sum_pi mu(pi) prod_{B in pi} sum_c prod_{t in B} M(c, t),
%       where M(c, t) is child c's sub-density at the query coordinates of
%       slot t --- the identity mobius.evalOrbitAbs uses, with children in
%       place of values;
%     - at an ordered level the sum runs over children in listed order, a
%       dynamic programme over the children (no factorial);
%     - at the leaf, M(v, t) = w_v theta(x_t - p_v) is the weighted one-body
%       kernel of value v at slot t.
%
%   Because the query side is a single fixed point rather than a summed
%   tuple set, the identity needs no orbit table: only the set-partition
%   lists of mobius.getPartitionBlockStructure (Bell numbers) are read, so
%   every level up to r = 10 is available, as for the flat evaluator.
%
%   Relative levels. A co-transposition unit at level u integrates each
%   level-u block over its own translation, exactly as mobius.evalOrbitRel
%   does over the whole tuple: the reduced block query (x_1, ..., x_{s-1})
%   is lifted to (u, u + x_1, ..., u + x_{s-1}) on a translation grid (the
%   line for a non-periodic attribute, [0, P) for a periodic one --- the
%   all-image measure), the absolute sub-density is evaluated at every
%   node, and the quadrature is divided by the translation-mode normaliser
%   sigma sqrt(2 pi / s). For the outer unit that is the root; for an
%   inner or intermediate unit the integral sits inside the recursion at
%   the unit's level, one independent integral per block.
%
%   Inputs
%       p, w       (K x 1) the event's live (non-NaN) values and weights.
%       tags       (K x (L-1)) grouping tags, innermost grouping first
%                  (a vector for L = 2).
%       rLevels, symLevels   length-L per-level tuple size and symmetry,
%                  innermost first.
%       relUnit    [], NaN, or 0 (absolute) or the 1-based co-transposition
%                  level (buildExpTens stores NaN for an absolute nested
%                  attribute).
%       sigma      kernel width.
%       x          (dim_a x n_q) query coordinates in the attribute's
%                  reduced layout (as documented for buildExpTens centres).
%
%   Output
%       vals       (n_q x 1) raw (un-normalised) per-event density values,
%                  on the scale of the tuple-centres route's raw values.
%
%   Twin of the Python mpt._tensor._nested_mobius_eval.eval_nested_attr_orbit.

    arguments
        p double
        w double
        tags double
        rLevels double
        symLevels
        relUnit
        sigma (1,1) double
        x double
        opts.is_per (1,1) logical = false
        opts.period (1,1) double = 0
        opts.wrap (1,:) char = 'full-image'
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
        opts.samplesPerSigma = []
    end

    p = double(p(:));
    w = double(w(:));
    K = numel(p);
    rLevels = double(rLevels(:)).';
    symLevels = logical(symLevels(:)).';
    L = numel(rLevels);
    if isvector(tags)
        tags = tags(:);
    elseif size(tags, 1) ~= K && size(tags, 2) == K
        tags = tags.';
    end
    if isvector(x) && size(x, 1) ~= 1 && size(x, 2) == 1
        x = x.';   % a single reduced coordinate row given as a column
    end
    n_q = size(x, 2);
    if n_q == 0
        vals = zeros(0, 1);
        return;
    end

    ts = internal.accuracyFloor('resolve', opts.truncationSigmas);
    ctx = struct();
    ctx.p = p; ctx.w = w; ctx.sigma = sigma;
    ctx.isPer = logical(opts.is_per); ctx.period = double(opts.period);
    ctx.wrap = char(opts.wrap); ctx.ts = ts;
    ctx.rLevels = rLevels;
    ctx.s = cumprod(rLevels);          % s(l) = leaf slots of one level-l sub-tuple
    if isempty(relUnit) || any(isnan(relUnit)) || relUnit <= 0
        ctx.relUnit = 0;               % 0 = absolute (levels are 1-based)
        ctx.spp = [];
    else
        ctx.relUnit = double(relUnit);
        sUnit = ctx.s(ctx.relUnit);
        ctx.spp = internal.resolveSamplesPerSigma(opts.samplesPerSigma, ...
                                                  max(2, sUnit), ...
                                                  opts.truncationSigmas);
    end

    root = localBuildNode(L, (1:K).', rLevels, symLevels, tags);
    if size(x, 1) ~= localWidth(ctx, L)
        error('mpt:evalNestedAttrOrbit:queryDim', ...
              ['nested query has %d rows; expected %d for this ' ...
               'attribute''s reduced layout.'], size(x, 1), localWidth(ctx, L));
    end
    vals = localContract(root, x, ctx);
    vals = vals(:);
end


% =====================================================================
%  Tree (levels are 1-based: level 1 = leaf group, level L = root)
% =====================================================================

function node = localBuildNode(level, valIdx, rLevels, symLevels, tags)
    valIdx = valIdx(:);
    if level == 1
        node = struct('level', 1, 'valIdx', valIdx, 'children', {{}}, ...
                      'r', rLevels(1), 'sym', symLevels(1));
        return;
    end
    keys = tags(valIdx, level - 1);
    uk = unique(keys);
    children = cell(1, numel(uk));
    for c = 1:numel(uk)
        children{c} = localBuildNode(level - 1, valIdx(keys == uk(c)), ...
                                     rLevels, symLevels, tags);
    end
    node = struct('level', level, 'valIdx', valIdx, 'children', {children}, ...
                  'r', rLevels(level), 'sym', symLevels(level));
end

function wdt = localWidth(ctx, level)
    % Query rows of one level-`level` sub-tuple in the reduced layout: the
    % absolute span until the co-transposition unit, and thereafter one
    % coordinate fewer per unit block.
    s = ctx.s(level);
    u = ctx.relUnit;
    if u == 0 || level < u
        wdt = s;
    else
        sU = ctx.s(u);
        wdt = (s / sU) * (sU - 1);
    end
end


% =====================================================================
%  Contraction
% =====================================================================

function v = localContract(node, xq, ctx)
    if ctx.relUnit > 0 && node.level == ctx.relUnit
        v = localIntegrateTranslation(node, xq, ctx);
    else
        v = localContractAbs(node, xq, ctx);
    end
end

function v = localContractAbs(node, xq, ctx)
    n_q = size(xq, 2);
    r = node.r;
    if node.level == 1
        vals = node.valIdx;
        nV = numel(vals);
        v = zeros(n_q, 1);
        budget = max(internal.kernelChunkBytesResolved(), 1);
        perQ = max(1, nV * r * 8 * 4);
        step = max(1, floor(budget / perQ));
        pv = ctx.p(vals);
        wv = ctx.w(vals);
        for c0 = 1:step:n_q
            c1 = min(n_q, c0 + step - 1);
            % M(v, t, q) = w_v theta(x(t, q) - p_v)
            d = reshape(xq(:, c0:c1), [1, r, c1 - c0 + 1]) - reshape(pv, [nV, 1, 1]);
            M = reshape(wv, [nV, 1, 1]) .* localTheta(d, ctx);
            v(c0:c1) = localCombine(M, r, node.sym);
        end
        return;
    end
    childW = localWidth(ctx, node.level - 1);
    nC = numel(node.children);
    M = zeros(nC, r, n_q);
    for i = 1:nC
        for b = 1:r
            rows = (b - 1) * childW + (1:childW);
            M(i, b, :) = reshape(localContract(node.children{i}, xq(rows, :), ctx), ...
                                 [1, 1, n_q]);
        end
    end
    v = localCombine(M, r, node.sym);
end

function th = localTheta(d, ctx)
    % One-body kernel per coordinate: Gaussian, or its wrapped form on a
    % periodic attribute under the declared measure.
    if ctx.isPer
        if strcmp(ctx.wrap, 'single-image')
            d = d - ctx.period * round(d / ctx.period);
        else
            th = internal.wrappedGaussian1d(d, ctx.sigma, ctx.period, ctx.ts, 2);
            return;
        end
    end
    th = exp(-(d .* d) / (2 * ctx.sigma^2));
end

function v = localCombine(M, r, sym)
    % Sum over r-tuples of distinct children of prod_t M(c_t, t, :).
    % M is (nChildren x r x n_q). Symmetric: the Möbius set-partition sum
    % over the r slots. Ordered: children in listed order, by a dynamic
    % programme over the children.
    n = size(M, 1);
    n_q = size(M, 3);
    if r == 1
        v = reshape(sum(M(:, 1, :), 1), [n_q, 1]);
        return;
    end
    if n < r
        v = zeros(n_q, 1);
        return;
    end
    if sym
        [uniqueBlocks, partBlockIdx, mus] = mobius.getPartitionBlockStructure(r);
        blockContrib = cell(1, numel(uniqueBlocks));
        for k = 1:numel(uniqueBlocks)
            B = uniqueBlocks{k};
            pr = M(:, B(1), :);
            for t = B(2:end)
                pr = pr .* M(:, t, :);
            end
            blockContrib{k} = reshape(sum(pr, 1), [n_q, 1]);
        end
        v = mobius.mobiusPartitionCombine(blockContrib, partBlockIdx, mus, false);
        return;
    end
    dp = zeros(n_q, r + 1);
    dp(:, 1) = 1;
    for i = 1:n
        for t = r:-1:1
            dp(:, t + 1) = dp(:, t + 1) + dp(:, t) .* reshape(M(i, t, :), [n_q, 1]);
        end
    end
    v = dp(:, r + 1);
end


% =====================================================================
%  Translation quadrature at the co-transposition unit
% =====================================================================

function v = localIntegrateTranslation(node, xq, ctx)
    s = ctx.s(node.level);
    n_q = size(xq, 2);
    if s < 2
        v = repmat(sum(ctx.w(node.valIdx)), n_q, 1);
        return;
    end
    sigma = ctx.sigma;
    if ctx.isPer
        N_u = max(64, ceil(ctx.period / sigma * ctx.spp));
        uGrid = (0:N_u - 1) * (ctx.period / N_u);
        du = ctx.period / N_u;
    else
        pNode = ctx.p(node.valIdx);
        xMin = min(0, min(xq(:)));
        xMax = max(0, max(xq(:)));
        uMin = min(pNode) - xMax - 8 * sigma;
        uMax = max(pNode) - xMin + 8 * sigma;
        N_u = max(64, ceil(max(uMax - uMin, 1) / sigma * ctx.spp));
        uGrid = linspace(uMin, uMax, N_u);
        du = [];
    end
    % Expanded query set: (s x (n_q * N_u)), query-major.
    xFull = [zeros(1, n_q); xq];
    X = reshape(reshape(xFull, [s, n_q, 1]) + reshape(uGrid, [1, 1, N_u]), ...
                [s, n_q * N_u]);
    F = reshape(localContractAbs(node, X, ctx), [n_q, N_u]);
    if ctx.isPer
        integral = sum(F, 2) * du;
    else
        integral = trapz(uGrid, F, 2);
    end
    v = integral / (sigma * sqrt(2 * pi / s));
end

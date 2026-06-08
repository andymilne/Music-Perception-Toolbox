function triple = nestedContract(densX, densY, normalize, truncationSigmas)
%NESTEDCONTRACT Fast tree-contraction of a single nested attribute's IP.
%   triple = internal.nestedContract(densX, densY, normalize, truncationSigmas)
%
%   Returns a struct with fields xy, xx, yy (the bare inner-product triple)
%   when the case is covered -- exactly one nested attribute, outer or no
%   [rel] (inner_r == 0), no NaN-padding, cosine normalisation -- AND the
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

    triple = [];
    if ~strcmp(normalize, 'cosine'); return; end
    if densX.nAttrs ~= 1 || densY.nAttrs ~= 1; return; end
    if ~isfield(densX, 'nested') || ~iscell(densX.nested) ...
            || ~isfield(densY, 'nested') || ~iscell(densY.nested)
        return;
    end
    specX = densX.nested{1};
    specY = densY.nested{1};
    if isempty(specX) || isempty(specY); return; end
    if localInnerR(specX) ~= 0 || localInnerR(specY) ~= 0
        return;   % inner [rel] unit not covered by the contraction yet
    end

    PX = double(densX.pAttr{1});
    PY = double(densY.pAttr{1});
    if any(isnan(PX(:))) || any(isnan(PY(:)))
        return;   % variable-K per event: exact enumeration only
    end
    WX = double(densX.w{1});
    WY = double(densY.w{1});

    rLevels   = double(specX.r(:)).';
    symLevels = logical(specX.sym(:)).';
    tags      = double(specX.tags);
    if size(tags, 2) == 1 && numel(rLevels) > 2
        tags = reshape(tags, size(tags, 1), []);   % defensive (L=2 only is 1-col)
    end
    isRel  = logical(densX.isRel(1));
    isPer  = logical(densX.isPer(1));
    period = double(densX.period(1));
    sigma  = double(densX.sigma(1));
    if isempty(truncationSigmas)
        ts = mptDefaults('truncationSigmas');
    else
        ts = double(truncationSigmas);
    end

    recipe = buildRecipe(rLevels, symLevels, tags);

    nX = size(PX, 2);
    nY = size(PY, 2);
    vmin = min(min(PX(:)), min(PY(:)));
    vmax = max(max(PX(:)), max(PY(:)));

    % Speed dispatch (integer/float counts; identical to Python).
    [mPerm, mComb] = tupleCounts(rLevels, symLevels, tags);
    Q = quadNodes(isRel, isPer, sigma, period, vmin, vmax, ts);
    pairTerms = nX * nY + nX * nX + nY * nY;
    costEnum     = pairTerms * mPerm * mComb;
    costContract = pairTerms * Q * recipeWork(recipe);
    if costContract >= costEnum
        return;   % enumeration is the faster route
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

    ipxy = tripSum(recipe, PX, WX, PY, WY, sigma, period, ts, quad);
    ipxx = tripSum(recipe, PX, WX, PX, WX, sigma, period, ts, quad);
    ipyy = tripSum(recipe, PY, WY, PY, WY, sigma, period, ts, quad);
    triple = struct('xy', ipxy, 'xx', ipxx, 'yy', ipyy);
end


% ----------------------------------------------------------------------
function s = tripSum(recipe, PA, WA, PB, WB, sigma, period, ts, quad)
    s = 0.0;
    nA = size(PA, 2);
    nB = size(PB, 2);
    for i = 1:nA
        ai = PA(:, i);
        wi = WA(:, i);
        for j = 1:nB
            s = s + nestedIp(recipe, ai, PB(:, j), wi, WB(:, j), ...
                             sigma, period, ts, quad);
        end
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
function recipe = buildRecipe(rLevels, symLevels, tags)
    L = numel(rLevels);
    Ktot = size(tags, 1);
    recipe = buildNode(L - 1, (1:Ktot).', rLevels, symLevels, tags);
end


function node = buildNode(level, slots, rLevels, symLevels, tags)
    slots = slots(:);
    if level == 0
        r0 = rLevels(1);
        [xt, yt] = tupleIndices(numel(slots), r0, symLevels(1));
        node = struct('level', 0, 'slots', slots, 'children', {{}}, ...
                      'xtup', xt, 'ytup', yt);
        return;
    end
    col = level;                       % 1-based tag column (Python col=level-1)
    keys = tags(slots, col);
    uk = unique(keys);                 % ascending
    children = cell(1, numel(uk));
    for c = 1:numel(uk)
        sub = slots(keys == uk(c));
        children{c} = buildNode(level - 1, sub, rLevels, symLevels, tags);
    end
    rl = rLevels(level + 1);           % Python r_levels[level]
    [xt, yt] = tupleIndices(numel(children), rl, symLevels(level + 1));
    node = struct('level', level, 'slots', slots, 'children', {children}, ...
                  'xtup', xt, 'ytup', yt);
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
    if xn.level == 0
        sub = K(:, xn.slots, yn.slots);            % Q x m x m
        v = combine(sub, xn.xtup, yn.ytup);
    else
        g = numel(xn.children);
        Q = size(K, 1);
        M = zeros(Q, g, g);
        for a = 1:g
            xa = xn.children{a};
            for b = 1:g
                M(:, a, b) = contractNode(xa, yn.children{b}, K);
            end
        end
        v = combine(M, xn.xtup, yn.ytup);
    end
end


function v = combine(M, xtup, ytup)
    Tx = size(xtup, 1);
    Ty = size(ytup, 1);
    if Tx == 0 || Ty == 0
        v = zeros(size(M, 1), 1);
        return;
    end
    r = size(xtup, 2);
    P = M(:, xtup(:, 1), ytup(:, 1));              % Q x Tx x Ty
    for t = 2:r
        P = P .* M(:, xtup(:, t), ytup(:, t));
    end
    v = sum(sum(P, 3), 2);                          % Q x 1
    v = v(:);
end


% ----------------------------------------------------------------------
%  Leaf-kernel batches + per-event-pair bare inner product
% ----------------------------------------------------------------------
function ipv = nestedIp(recipe, vX, vY, wX, wY, sigma, period, ts, quad)
    n = numel(vX);
    switch quad.mode
        case 'abs'
            isPerAbs = isfinite(period) && period > 0;
            d = reshape(vX, [1, n, 1]) - reshape(vY, [1, 1, n]);   % 1 x n x n
            if isPerAbs
                d = d - period * round(d / period);
            end
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wX, [1, n, 1]) .* reshape(wY, [1, 1, n]));
            K = truncK(K, ts);
            v = contractNode(recipe, recipe, K);
            ipv = v(1);
        case 'relper'
            taus = quad.taus(:);
            T = numel(taus);
            d = reshape(vX, [1, n, 1]) ...
                - (reshape(vY, [1, 1, n]) + reshape(taus, [T, 1, 1]));  % T x n x n
            d = d - period * round(d / period);
            K = exp(-d.^2 / (4 * sigma^2));
            K = K .* (reshape(wX, [1, n, 1]) .* reshape(wY, [1, 1, n]));
            K = truncK(K, ts);
            ipv = sum(contractNode(recipe, recipe, K));   % common dtau cancels
        case 'relnonper'
            taus = quad.taus(:);
            T = numel(taus);
            d = reshape(vX, [1, n, 1]) ...
                - (reshape(vY, [1, 1, n]) + reshape(taus, [T, 1, 1]));
            K = exp(-d.^2 / (4 * sigma^2));               % no wrap
            K = K .* (reshape(wX, [1, n, 1]) .* reshape(wY, [1, 1, n]));
            K = truncK(K, ts);
            ipv = sum(contractNode(recipe, recipe, K));
    end
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
function n = autoNtau(period, sigma, tol)
    base = 2 * pi * period / sigma;
    margin = 1 + 0.5 * max(0, -log10(max(tol, 1e-16))) / 12;
    n = max(64, ceil(base * margin));
end


function quad = makeQuadrature(isRel, isPer, sigma, period, vmin, vmax, ts)
    if ~isRel
        quad = struct('mode', 'abs');
        return;
    end
    if isfinite(ts)
        tol = max(exp(-0.5 * ts^2), 1e-12);
    else
        tol = 1e-12;
    end
    if isPer
        ntau = autoNtau(period, sigma, tol);
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
        Q = autoNtau(period, sigma, tol);
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


function c = countSide(slots, level, useSym, rLevels, symLevels, tags)
    if level == 0
        r0 = rLevels(1);
        c = nchoosekCount(numel(slots), r0);
        if useSym && symLevels(1)
            c = c * factorial(r0);
        end
        return;
    end
    col = level;
    keys = tags(slots, col);
    uk = unique(keys);
    subs = zeros(1, numel(uk));
    for g = 1:numel(uk)
        sub = slots(keys == uk(g));
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
    w = size(node.xtup, 1) * size(node.ytup, 1) * max(1, size(node.xtup, 2));
    if node.level ~= 0
        g = numel(node.children);
        w = w + g * g;
        for c = 1:g
            w = w + recipeWork(node.children{c});
        end
    end
end

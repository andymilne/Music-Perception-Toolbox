function triple = nestedContract(densX, densY, normalize, truncationSigmas, force)
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

    if nargin < 5 || isempty(force); force = false; end
    triple = [];
    if ~strcmp(normalize, 'cosine')
        declineContractIfForced(force, ...
            'the contraction implements cosine normalisation only');
        return;
    end
    if densX.nAttrs ~= 1 || densY.nAttrs ~= 1
        declineContractIfForced(force, ...
            'the contraction supports a single attribute only');
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
    if any(isnan(PX(:))) || any(isnan(PY(:)))
        declineContractIfForced(force, ...
            'variable-K (NaN-padded) events are not covered');
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

    recipe = buildRecipe(rLevels, symLevels, tags, isRel, isPer);

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
    if costContract >= costEnum && ~force
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

    ipxy = tripSum(recipe, PX, WX, PY, WY, sigma, period, ts, quad, false);
    ipxx = tripSum(recipe, PX, WX, PX, WX, sigma, period, ts, quad, true);
    ipyy = tripSum(recipe, PY, WY, PY, WY, sigma, period, ts, quad, true);
    triple = struct('xy', ipxy, 'xx', ipxx, 'yy', ipyy);
end


% ----------------------------------------------------------------------
function s = tripSum(recipe, PA, WA, PB, WB, sigma, period, ts, quad, sym)
    % sym=true (self inner products): <e_i,e_j> = <e_j,e_i>, so evaluate
    % only the upper triangle and double the off-diagonal terms.
    if nargin < 10; sym = false; end
    s = 0.0;
    nA = size(PA, 2);
    nB = size(PB, 2);
    for i = 1:nA
        ai = PA(:, i);
        wi = WA(:, i);
        if sym; j0 = i; else; j0 = 1; end
        for j = j0:nB
            v = nestedIp(recipe, ai, PB(:, j), wi, WB(:, j), ...
                         sigma, period, ts, quad);
            if sym && j ~= i
                s = s + 2.0 * v;
            else
                s = s + v;
            end
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


function node = buildNode(level, slots, rLevels, symLevels, tags, isRel, isPer)
    slots = slots(:);
    if level == 0
        r0 = rLevels(1);
        sy0 = symLevels(1);
        useOrb = orbitEligible(numel(slots), r0, sy0, isRel, isPer);
        if useOrb
            xt = zeros(0, r0); yt = zeros(0, r0);   % lazy: orbit needs no tuples
        else
            [xt, yt] = tupleIndices(numel(slots), r0, sy0);
        end
        node = struct('level', 0, 'slots', slots, 'children', {{}}, ...
                      'xtup', xt, 'ytup', yt, 'r', r0, 'sym', sy0, ...
                      'useOrbit', useOrb);
        return;
    end
    col = level;                       % 1-based tag column (Python col=level-1)
    keys = tags(slots, col);
    uk = unique(keys);                 % ascending
    children = cell(1, numel(uk));
    for c = 1:numel(uk)
        sub = slots(keys == uk(c));
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
    node = struct('level', level, 'slots', slots, 'children', {children}, ...
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
    % xn is yn for the cosine, so the tree is walked once.
    if xn.level == 0
        v = combineNode(K(:, xn.slots, xn.slots), xn);
    else
        v = combineNode(subtreeOverlaps(xn.children, K), xn);
    end
end


function s = nodeSpan(node)
    if node.level == 0
        s = numel(node.slots);
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


function M = leafOverlaps(nodes, K)
    % (Q, g, g) pairwise overlaps among g leaf siblings.
    g = numel(nodes);
    Q = size(K, 1);
    n = size(K, 2);
    if nodes{1}.r == 1
        % r0 = 1: M(q,a,b) = sum_{i in Sa, j in Sb} K(q,i,j) (weights folded).
        G = zeros(g, n);
        for a = 1:g
            G(a, nodes{a}.slots) = 1.0;
        end
        KG = reshape(reshape(K, [Q * n, n]) * G.', [Q, n, g]);   % (q,i,b)
        KGp = reshape(permute(KG, [2, 1, 3]), [n, Q * g]);        % (i, q*b)
        MG = G * KGp;                                             % (a, q*b)
        M = permute(reshape(MG, [g, Q, g]), [2, 1, 3]);          % (Q,g,g)
        return;
    end
    if siblingsUniform(nodes)
        m = numel(nodes{1}.slots);
        blocks = zeros(g, g, Q, m, m);
        for a = 1:g
            sa = K(:, nodes{a}.slots, :);
            for b = 1:g
                blocks(a, b, :, :, :) = reshape(sa(:, :, nodes{b}.slots), ...
                                                [1, 1, Q, m, m]);
            end
        end
        vals = combineNode(reshape(blocks, [g * g * Q, m, m]), nodes{1});
        M = permute(reshape(vals, [g, g, Q]), [3, 1, 2]);
        return;
    end
    M = zeros(Q, g, g);
    for a = 1:g
        for b = 1:g
            M(:, a, b) = combineNode(K(:, nodes{a}.slots, nodes{b}.slots), ...
                                     nodes{a});
        end
    end
end


function M = subtreeOverlaps(nodes, K)
    % (Q, g, g) pairwise overlaps among g sibling subtrees.
    if nodes{1}.level == 0
        M = leafOverlaps(nodes, K);
        return;
    end
    g = numel(nodes);
    Q = size(K, 1);
    sizes = zeros(1, g);
    flat = {};
    for k = 1:g
        sizes(k) = numel(nodes{k}.children);
        flat = [flat, nodes{k}.children];   %#ok<AGROW>
    end
    offs = [0, cumsum(sizes)];
    Mc = subtreeOverlaps(flat, K);          % (Q, Gc, Gc)
    if siblingsUniform(nodes)
        gc = sizes(1);
        blocks = zeros(g, g, Q, gc, gc);
        for a = 1:g
            ra = offs(a) + 1 : offs(a) + gc;
            for b = 1:g
                cb = offs(b) + 1 : offs(b) + gc;
                blocks(a, b, :, :, :) = reshape(Mc(:, ra, cb), ...
                                                [1, 1, Q, gc, gc]);
            end
        end
        vals = combineNode(reshape(blocks, [g * g * Q, gc, gc]), nodes{1});
        M = permute(reshape(vals, [g, g, Q]), [3, 1, 2]);
        return;
    end
    M = zeros(Q, g, g);
    for a = 1:g
        ra = offs(a) + 1 : offs(a + 1);
        for b = 1:g
            cb = offs(b) + 1 : offs(b + 1);
            M(:, a, b) = combineNode(Mc(:, ra, cb), nodes{a});
        end
    end
end


function v = combineNode(M, node)
    % Symmetric-level combine, orbit-reduced when flagged; same scale as
    % combine (the r!-cancelled perm x comb form). Handles rectangular
    % blocks (gx ~= gy), which arise for ragged sibling subtrees.
    gx = size(M, 2);
    gy = size(M, 3);
    if node.useOrbit
        v = combineOrbit(M, node.r, node.xtup, node.ytup);
        return;
    end
    if gx == gy && gx == nodeSpan(node)
        v = combine(M, node.xtup, node.ytup);          % uniform: stored tuples
        return;
    end
    [xt, ~] = tupleIndices(gx, node.r, node.sym);       % ragged: per-size tuples
    [~, yt] = tupleIndices(gy, node.r, node.sym);
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

%% test_inner_product_scale.m — the bare inner product on the canonical scale
%
%  cosSimExpTens(..., 'normalize', 'none') returns <X, Y> on one scale
%  whatever route ran, and the Rényi-2 entropy is computed from it. Both
%  are pinned here against an explicit enumeration of every tuple pair on
%  every shape the routes cover: flat symmetric, ordered, relative,
%  periodic; nested with every level pattern, absolute and with either
%  co-transposition unit, periodic or not; a mixed density; every forced
%  method. Mirror of Python tests/test_inner_product_scale.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_ips
    cleanupDefaults_ips = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end
ips_prevHints = mptDefaults('showHints');
mptDefaults('showHints', false);
ips_prevTs = mptDefaults('truncationSigmas');
mptDefaults('truncationSigmas', Inf);

ips_P = 12.0;
ips_T2 = repelem(0:1, 3);
ips_T3 = repelem(0:2, 3);
ips_T3L = [0 0; 0 0; 1 0; 1 0; 2 1; 2 1; 3 1; 3 1];

% --- flat shapes: {K, N, r, rel, per, sym} ---
ips_flat = { 6,3,2,false,false,true;  6,3,3,false,false,true;  6,3,2,true,false,true; ...
             6,3,3,true,false,true;   6,3,2,false,true,true;   6,3,2,true,true,true; ...
             6,3,1,false,false,true;  6,3,1,false,true,true;   6,3,2,false,false,false; ...
             6,3,3,true,true,false;   6,3,3,true,false,false };
for ips_i = 1:size(ips_flat, 1)
    [K, N, r, rel, per, sym] = ips_flat{ips_i, :};
    d = ipsFlat(K, N, r, rel, per, sym, ips_P, 0);
    [ips_ok, ips_msg] = ipsCheckMethods(d, {'auto', 'bulger', 'mobius', 'centres'});
    results{end+1, 1} = sprintf('ip scale: flat r=%d rel=%d per=%d sym=%d every route on scale%s', r, rel, per, sym, ips_msg); %#ok<*SAGROW>
    results{end, 2} = ips_ok;
    [ips_ip, ips_Z] = ipsReferenceSelfIp(d);
    h = entropyExpTens(d, 'method', 'renyi2', 'base', exp(1), 'verbose', false);
    results{end+1, 1} = sprintf('ip scale: renyi2 flat r=%d rel=%d per=%d sym=%d matches enumeration', r, rel, per, sym);
    results{end, 2} = abs(h - (-log(ips_ip / ips_Z^2))) <= 10 * ipsTol(d);
end

% --- nested shapes: {tags, r, sym, rel, per, K} ---
ips_nested = { ips_T2,[1 2],[1 1],[0 0],false,6;  ips_T2,[1 2],[1 0],[0 0],false,6; ...
               ips_T2,[1 2],[0 1],[0 0],true,6;   ips_T3,[2 2],[1 1],[0 0],false,9; ...
               ips_T3,[2 2],[1 1],[0 0],true,9;   ips_T3,[2 2],[1 1],[1 0],false,9; ...
               ips_T3,[2 2],[1 1],[0 1],false,9;  ips_T3,[2 2],[1 1],[0 1],true,9; ...
               ips_T3,[3 2],[1 1],[0 1],false,9;  ips_T3,[1 3],[1 1],[0 1],false,9; ...
               ips_T3,[1 3],[1 1],[0 1],true,9;   ips_T3,[2 2],[1 0],[1 0],true,9; ...
               ips_T2,[2 2],[0 0],[0 0],false,6;  ips_T3,[2 3],[1 1],[0 1],true,9; ...
               ips_T3L,[2 2 2],[1 1 1],[0 0 0],false,8; ips_T3L,[2 2 2],[1 0 1],[0 0 1],true,8; ...
               ips_T3L,[2 2 2],[0 1 0],[0 0 1],false,8 };
for ips_i = 1:size(ips_nested, 1)
    [tags, r, sym, rel, per, K] = ips_nested{ips_i, :};
    d = ipsNested(tags, r, sym, rel, per, K, ips_P, 0);
    [ips_ok, ips_msg] = ipsCheckMethods(d, {'auto', 'bulger', 'mobius', 'centres', 'contract'});
    results{end+1, 1} = sprintf('ip scale: nested r=[%s] sym=[%s] rel=[%s] per=%d every route on scale%s', num2str(r), num2str(sym), num2str(rel), per, ips_msg);
    results{end, 2} = ips_ok;
    [ips_ip, ips_Z] = ipsReferenceSelfIp(d);
    h = entropyExpTens(d, 'method', 'renyi2', 'base', exp(1), 'verbose', false);
    results{end+1, 1} = sprintf('ip scale: renyi2 nested r=[%s] sym=[%s] rel=[%s] per=%d matches enumeration', num2str(r), num2str(sym), num2str(rel), per);
    results{end, 2} = abs(h - (-log(ips_ip / ips_Z^2))) <= 10 * ipsTol(d);
end

% --- the relative non-periodic contraction's scale does not depend on the data ---
ips_ok = true;
for ips_c = {{1, 0.7}, {2, 1.5}, {0, 0.3}, {3, 1.5}}
    d = ipsNested(ips_T3, [2 2], [1 1], [0 1], false, 9, ips_P, ips_c{1}{1}, ips_c{1}{2});
    [ips_ip, ~] = ipsReferenceSelfIp(d);
    ncOpts = struct('methodName', 'contract', 'forceRoute', 'contract_relnonper', ...
                    'cacheX', struct('keys', {{}}, 'vals', []), ...
                    'cacheY', struct('keys', {{}}, 'vals', []));
    [tr, routes] = internal.nestedContract(d, d, 'none', [], true, ncOpts);
    v = tr.xy * internal.ipCanonicalScale(d, 'contract', routes);
    ips_ok = ips_ok && abs(v - ips_ip) <= 1e-9 * ips_ip;
end
results{end+1, 1} = 'ip scale: relative non-periodic contraction scale is data-independent';
results{end, 2} = ips_ok;

% --- mixed density and the cross term ---
rng(5, 'twister');
p0 = sort(ips_P * rand(6, 3), 1); p1 = sort(ips_P * rand(4, 3), 1); p2 = sort(ips_P * rand(4, 3), 1);
specs = { struct('tags', ips_T2, 'r', [1 2], 'sym', [true true], 'rel', [0 0]), ...
          struct('r', 2, 'sym', true, 'rel', true), struct('r', 2, 'sym', false, 'rel', false) };
d = buildExpTens({p0, p1, p2}, {[], [], []}, 'specs', specs, 'sigma', [0.7 0.5 0.9], ...
                 'isPer', [false false true], 'period', [ips_P ips_P ips_P], 'verbose', false);
[ips_ok, ips_msg] = ipsCheckMethods(d, {'auto', 'bulger', 'mobius', 'centres', 'contract'});
results{end+1, 1} = ['ip scale: mixed nested + relative flat + ordered flat' ips_msg];
results{end, 2} = ips_ok;
e = buildExpTens({p0(:, end:-1:1) + 0.3, p1 + 0.1, p2 - 0.2}, {[], [], []}, 'specs', specs, ...
                 'sigma', [0.7 0.5 0.9], 'isPer', [false false true], 'period', [ips_P ips_P ips_P], 'verbose', false);
xy = cosSimExpTens(d, e, 'normalize', 'none', 'verbose', false);
xx = cosSimExpTens(d, d, 'normalize', 'none', 'verbose', false);
yy = cosSimExpTens(e, e, 'normalize', 'none', 'verbose', false);
c = cosSimExpTens(d, e, 'verbose', false);
results{end+1, 1} = 'ip scale: the cosine recomposes from three bare inner products';
results{end, 2} = abs(xy / sqrt(xx * yy) - c) <= 1e-9 * abs(c);

% --- the bare value does not form a self inner product ---
d = ipsFlat(6, 3, 2, false, false, true, ips_P, 0);
e = ipsFlat(6, 3, 2, false, false, true, ips_P, 1);
[~, dX, dY] = cosSimExpTens(d, e, 'normalize', 'none', 'verbose', false);
results{end+1, 1} = 'ip scale: normalize none forms no self inner product';
results{end, 2} = ~internal.selfIpMemoised(dX.selfIP) && ~internal.selfIpMemoised(dY.selfIP);
[~, ~, dY] = cosSimExpTens(d, e, 'normalize', 'oneSidedDenom', 'verbose', false);
results{end+1, 1} = 'ip scale: oneSidedDenom still memoises <Y,Y>';
results{end, 2} = internal.selfIpMemoised(dY.selfIP);

% --- ragged events ---
rng(4, 'twister');
p = sort(ips_P * rand(6, 3), 1); p(6, 1) = NaN; p(5:6, 3) = NaN;
d = buildExpTens({p}, {[]}, 'specs', {struct('tags', ips_T2, 'r', [1 2], 'sym', [true true], 'rel', [0 0])}, ...
                 'sigma', 0.7, 'isPer', false, 'period', 0, 'verbose', false);
[ips_ip, ips_Z] = ipsReferenceSelfIp(d);
h = entropyExpTens(d, 'method', 'renyi2', 'base', exp(1), 'verbose', false);
results{end+1, 1} = 'ip scale: renyi2 with ragged events matches enumeration';
results{end, 2} = abs(h - (-log(ips_ip / ips_Z^2))) <= 1e-9 * abs(h);

% --- renyi2 takes the inner-product route ---
d = ipsNested(repelem(0:3, 3), [2 3], [1 1], [0 0], false, 12, ips_P, 0);
entropyExpTens(d, 'method', 'renyi2', 'verbose', false);
results{end+1, 1} = 'ip scale: renyi2 runs the route the inner-product selector picks';
results{end, 2} = isequal(internal.lastNestedRoutes(), {'contract'});

mptDefaults('showHints', ips_prevHints);
mptDefaults('truncationSigmas', ips_prevTs);

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
    fprintf('\n=== test_inner_product_scale: %d passed, %d failed (of %d) ===\n\n', nPass, nFail, nPass + nFail);
    clear cleanupDefaults_ips
    if nFail > 0
        error('test_inner_product_scale:failed', '%d test(s) failed.', nFail);
    end
end


function d = ipsFlat(K, N, r, rel, per, sym, P, seed, sigma)
    if nargin < 9; sigma = 0.7; end
    rng(seed, 'twister');
    p = sort(P * rand(K, N), 1);
    w = 0.5 + rand(K, N);
    d = buildExpTens({p}, {w}, 'specs', {struct('r', r, 'sym', sym, 'rel', rel)}, ...
                     'sigma', sigma, 'isPer', per, 'period', P, 'verbose', false);
end


function d = ipsNested(tags, r, sym, rel, per, K, P, seed, sigma)
    if nargin < 9; sigma = 0.7; end
    rng(seed, 'twister');
    p = sort(P * rand(K, 3), 1);
    w = 0.5 + rand(K, 3);
    spec = struct('tags', tags, 'r', r, 'sym', logical(sym), 'rel', rel);
    d = buildExpTens({p}, {w}, 'specs', {spec}, 'sigma', sigma, 'isPer', per, ...
                     'period', P, 'verbose', false);
end


function tol = ipsTol(d)
    % Relative-periodic attributes: the full-image routes and the
    % single-image enumeration differ by the measure gap at sigma/P.
    if any(logical(d.isRel(:)) & logical(d.isPer(:)))
        tol = 1e-3;
    else
        tol = 1e-9;
    end
end


function [ok, msg] = ipsCheckMethods(d, methods)
    [ref, ~] = ipsReferenceSelfIp(d);
    tol = ipsTol(d);
    ok = true; msg = '';
    for i = 1:numel(methods)
        try
            v = cosSimExpTens(d, d, 'normalize', 'none', 'method', methods{i}, 'verbose', false);
        catch err
            % a forced route the shape does not admit; the message says so
            if isempty(strfind(err.message, 'not available')) && isempty(strfind(err.message, 'cannot be honoured')) %#ok<STREMP>
                ok = false; msg = [msg sprintf(' [%s: %s]', methods{i}, err.message)];
            end
            continue;
        end
        if ~(abs(v - ref) <= tol * abs(ref))
            ok = false; msg = [msg sprintf(' [%s: %.9g vs %.9g]', methods{i}, v, ref)];
        end
    end
end


function [ip, Z] = ipsReferenceSelfIp(dens)
    % (<T,T>, Z) by enumeration, composing attributes as the Rényi-2
    % factorisation does.
    dens = internal.prunedExpTens(dens);
    A = double(dens.nAttrs); N = double(dens.N);
    P_xx = ones(N, N); Zs = ones(N, A);
    for a = 1:A
        isNestedA = isfield(dens, 'nested') && numel(dens.nested) >= a && ~isempty(dens.nested{a});
        if ~isNestedA && logical(dens.isRel(a)) && dens.r(a) == 1
            continue;   % 0-D point mass: unit overlap and mass
        end
        [I, Za] = ipsRefAttr(dens, a);
        P_xx = P_xx .* I;
        Zs(:, a) = Za(:);
    end
    ip = sum(P_xx(:)); Z = sum(prod(Zs, 2));
end


function [I_a, Z_a] = ipsRefAttr(dens, a)
%IPSREFATTR  Reference (event, event) inner matrix and per-event mass of
%   one attribute by explicit tuple enumeration on the canonical scale:
%   kernel overlap (pi sigma^2)^(d/2)/sqrt(det M) exp(-Q/(4 sigma^2)),
%   mass (2 pi sigma^2)^(d/2)/sqrt(det M), over the attribute's full
%   ordered tuple set (every arrangement a symmetric level admits). This
%   is the enumeration entropyExpTens carried for nested and ordered
%   attributes until the inner-product machinery served it.
    sig  = dens.sigma(a);
    isper = dens.isPer(a);
    per  = dens.period(a);
    isNestedA = isfield(dens, 'nested') && numel(dens.nested) >= a ...
        && ~isempty(dens.nested{a}) && isstruct(dens.nested{a}) ...
        && isfield(dens.nested{a}, 'tags');
    if isNestedA
        % Nested attribute: rebuild from its resolved spec.
        spec = dens.nested{a};
        da = buildExpTens({dens.pAttr{a}}, {dens.w{a}}, 'specs', {spec}, ...
                          'sigma', sig, 'isPer', isper, 'period', per, ...
                          'lazy', false, 'verbose', false);
    else
        % Flat ordered attribute: rebuild from its flat parameters with
        % isSym = false, so the materialised tuples are the C(K, r_a)
        % ordered sub-tuples (one kernel each, no orbit).
        spec = [];
        r_a0   = dens.r(a);
        isRel0 = dens.isRel(a);
        sym0 = true;
        if isfield(dens, 'isSym') && ~isempty(dens.isSym); sym0 = logical(dens.isSym(a)); end
        da = buildExpTens({dens.pAttr{a}}, {dens.w{a}}, sig, r_a0, ...
                          isRel0, isper, per, sym0, ...
                          'lazy', false, 'verbose', false);
    end
    C   = da.Centres{1};         % (d_a x nJ) reduced centres
    wj  = da.wJ(:);              % (nJ x 1)
    eoj = da.eventOfJ(:);        % (nJ x 1) 1-based event index
    d_a = size(C, 1);
    nj  = numel(wj);
    N   = dens.N;

    blockSize = 0;
    if ~isempty(spec) && isfield(spec, 'proj') ...
            && (strcmp(spec.proj, 'inner') || strcmp(spec.proj, 'intermediate'))
        u = spec.relUnit;
        blockSize = prod(spec.r(1:u));
    end
    isRel = da.isRel(1);
    r_a   = da.r(1);
    detM = internal.quadraticFormDet(r_a, blockSize, isRel);
    vol  = internal.gaussianMassConst(sig, d_a, detM);          % single-kernel mass
    pref = internal.gaussianMassConst(sig, d_a, detM, true);    % overlap prefactor

    I_a = zeros(N, N);
    Z_a = zeros(N, 1);
    if nj > 0
        % Abs-per full-image path: compute the pairwise overlap matrix
        % O directly from per-position theta products, bypassing the Q ->
        % exp(-Q/(4 sigma^2)) formulation which is single-image. This
        % applies only to flat abs-per (blockSize < 2 and not rel);
        % other configurations use the block-diagonal quadratic form
        % below (either always full-image via pairwise wrap for rel,
        % or nested/block-metric that keeps its own semantics).
        wrapA = 'full-image';
        if isfield(dens, 'wrap') && ~isempty(dens.wrap) ...
                && a <= numel(dens.wrap)
            wrapA = char(dens.wrap{a});
        end
        useAbsPerFullImage = isper && ~isRel && blockSize < 2 ...
            && strcmp(wrapA, 'full-image');
        if useAbsPerFullImage
            D = reshape(C, d_a, nj, 1) - reshape(C, d_a, 1, nj);
            ts = internal.accuracyFloor('resolve', []);
            theta = internal.wrappedGaussian1d(D, sig, per, ts, 4);
            O = pref .* reshape(prod(theta, 1), nj, nj);
        else
            Q = ipsBlockMetricQ(C, blockSize, isRel, r_a, isper, per);  % nJ x nJ
            O = pref .* exp(-Q ./ (4 * sig^2));
        end
        WO = (wj * wj.') .* O;
        G = zeros(N, nj);
        G(sub2ind([N, nj], eoj.', 1:nj)) = 1;
        I_a = G * WO * G.';
        Z_a = vol .* (G * wj);
    end
end


function Q = ipsBlockMetricQ(C, blockSize, isRel, r_a, isPer, per)
%LOCALBLOCKMETRICQ  Pairwise block-diagonal co-transposition quadratic
%form on reduced centres. Mirrors the reduced-convention block metric used
%in evalExpTens (qInnerBlocksReducedLocal) and the whole-tuple _compute_Q,
%but operates on the (nJ x nJ) pairwise difference tensor.
    d_a = size(C, 1);
    nj  = size(C, 2);
    % D(k,i,j) = C(k,i) - C(k,j).
    D = reshape(C, d_a, nj, 1) - reshape(C, d_a, 1, nj);
    Q = zeros(nj, nj);
    if blockSize >= 2
        blk = blockSize - 1;          % reduced rows per block
        nBlocks = d_a / blk;
        for b = 1:nBlocks
            rows = (b - 1) * blk + (1:blk);
            Db = D(rows, :, :);
            if isPer
                position0Wrapped = Db - per .* floor(Db ./ per + 0.5);
                Qb = reshape(sum(position0Wrapped .^ 2, 1), nj, nj);
                for i = 1:blk
                    for j = i + 1:blk
                        delta = reshape(Db(i, :, :) - Db(j, :, :), nj, nj);
                        delta = delta - per .* floor(delta ./ per + 0.5);
                        Qb = Qb + delta .^ 2;
                    end
                end
                Qb = Qb / blockSize;
            else
                Qb = reshape(sum(Db .^ 2, 1), nj, nj) ...
                   - reshape(sum(Db, 1) .^ 2, nj, nj) / blockSize;
            end
            Q = Q + Qb;
        end
    elseif isRel && r_a >= 2
        % Whole-tuple reduced relative quotient (outer unit).
        if isPer
            position0Wrapped = D - per .* floor(D ./ per + 0.5);
            Q = reshape(sum(position0Wrapped .^ 2, 1), nj, nj);
            for i = 1:d_a
                for j = i + 1:d_a
                    delta = reshape(D(i, :, :) - D(j, :, :), nj, nj);
                    delta = delta - per .* floor(delta ./ per + 0.5);
                    Q = Q + delta .^ 2;
                end
            end
            Q = Q / r_a;
        else
            Q = reshape(sum(D .^ 2, 1), nj, nj) ...
              - reshape(sum(D, 1) .^ 2, nj, nj) / r_a;
        end
    else
        % Absolute.
        if isPer
            D = D - per .* floor(D ./ per + 0.5);
        end
        Q = reshape(sum(D .^ 2, 1), nj, nj);
    end
end

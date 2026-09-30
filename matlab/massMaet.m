function m = massMaet(dens, nv)
%MASSMAET  Mass of a multi-attribute expectation tensor (MAET) in a region.
%
%   m = massMaet(dens)
%   m = massMaet(dens, 'region', {a, spec; ...}, 'normalize', 'total')
%
%   OVERVIEW. The density of a MAET is a weighted sum of Gaussian kernels,
%   one per tuple. Each kernel is taken here with unit mass, so the mass of
%   the whole density is the sum of its tuples' weight products, and the
%   mass in a region R counts each tuple by the share of its kernel that
%   falls in the region:
%
%       m(R) = sum_j w_j * P_j(R).
%
%   The mass is the integral over R of the density evalMaet returns under
%   'gaussian'; with 'normalize', 'total' it is divided by the whole
%   density's mass, giving the share of the density that lies in the
%   region. Because a kernel has width, a tuple just outside a box still
%   contributes the part of its kernel that crosses the edge.
%
%   The density is a Cartesian product across attributes within each
%   event, so the sum factors into per-event, per-attribute sums, and the
%   joint tuple set is never built.
%
%   INPUTS
%     dens         A density from buildMaet, or a whole pre-MAET, which is
%                  built here. A cell of them gives one value per entry,
%                  as a row vector.
%
%   NAME-VALUE OPTIONS
%     'region'     {a, spec; ...}, a an attribute index. An attribute not
%                  named is integrated over entirely, so without a region
%                  the result is the whole density's mass. spec is one of:
%                    [lo hi]     a box, the same bounds on every
%                                coordinate of the attribute;
%                    a D x 2 matrix, a row [lo hi] per coordinate;
%                    {'gaussian', centre, sd}
%                                a soft region, weighting the density by
%                                exp(-|x - centre|^2 / (2 sd^2)); centre
%                                is a scalar (every coordinate) or one
%                                value per coordinate.
%                  Bounds may be infinite. The coordinates are those of
%                  the attribute's query in evalMaet: the r values of an
%                  absolute tuple, and the r - 1 values above the first
%                  for a relative one. An exchangeable density holds every
%                  ordering of a tuple, so a region on a relative
%                  attribute that should not care about order must be
%                  symmetric: at r = 2, the region and its
%                  negative.
%     'normalize'  'none' (default), the mass; or 'total', its share of
%                  the whole density's mass (NaN where that is zero).
%     'verbose'    Passed to buildMaet for a pre-MAET (default false).
%
%   NOTES
%   On a periodic attribute a box spans at most one period, and its bounds
%   may wrap ([1100 1300] with period 1200). A Gaussian region on a
%   periodic attribute is taken on the nearest image of its centre.
%   Periodic kernels sum their images (full-image) or are taken on the
%   nearest image and renormalized to unit mass on the circle (single-
%   image, and the relative-periodic pairwise wrap).
%
%   A box needs the attribute's kernel to be independent across its
%   coordinates, which holds for an absolute attribute and for a relative
%   one of two values (and a nested attribute whose relative blocks hold
%   two); a relative attribute of three or more has correlated
%   coordinates, and there a box is refused and a Gaussian region, which
%   has a closed form for any covariance, can be used on a non-periodic
%   attribute. An attribute with a matrix-valued kernel covariance
%   (kernelCov) cannot be restricted. The integrals are exact:
%   'truncationSigmas' does not apply.
%
%   Twin of Python mass_maet.
%
%   See also SWEPTMASS, EVALMAET, BUILDMAET, ENTROPYMAET.

arguments
    dens
    nv.region = []
    nv.normalize (1,:) char = 'none'
    nv.verbose (1,1) logical = false
end

normalize = lower(nv.normalize);
if ~any(strcmp(normalize, {'none', 'total'}))
    error('massMaet:badNormalize', ...
        'normalize must be ''none'' or ''total''; got ''%s''.', nv.normalize);
end
if iscell(dens)
    m = zeros(1, numel(dens));
    for k = 1:numel(dens)
        m(k) = massMaet(dens{k}, 'region', nv.region, ...
            'normalize', normalize, 'verbose', nv.verbose);
    end
    return;
end
if internal.isPreMaet(dens)
    dens = buildMaet(dens, 'verbose', nv.verbose);
end
m = localDensityMass(dens, nv.region, normalize);
end


% -------------------------------------------------------------------------
%  One density
% -------------------------------------------------------------------------

function m = localDensityMass(dens, region, normalize)
A    = dens.nAttrs;
N    = dens.N;
P    = dens.pAttr;
W    = dens.w;
rVec = dens.r(:).';
isExchV = dens.isExch(:).';
[keys, specs] = localParseRegion(region, A);

% Per-attribute tuple-index structure over the ever-valid values, as the
% factored evaluation builds it: the index pattern is the same in every
% event, and a tuple touching a value absent in its event carries weight
% zero.
perm = cell(1, A);
for a = 1:A
    everValid = find(any(~isnan(P{a}), 2)).';
    if localIsNested(dens, a)
        spec = dens.nested{a};
        tg = spec.tags;
        if isvector(tg), tg = tg(:); end
        perm{a} = internal.nestedEnumIndices( ...
            everValid, tg(everValid, :), spec.r(:).', spec.exch(:).');
    elseif rVec(a) == 1
        perm{a} = everValid;
    elseif numel(everValid) < rVec(a)
        perm{a} = zeros(rVec(a), 0);
    else
        Ka = size(P{a}, 1);
        perm{a} = internal.enumFlatAttr( ...
            zeros(Ka, 1), everValid, rVec(a), isExchV(a), ones(Ka, 1));
    end
end

innerR = localInnerR(dens, A);
geom = cell(1, numel(keys));
for i = 1:numel(keys)
    a = keys(i);
    geom{i} = localGeometry(dens, a, innerR(a), size(perm{a}, 1));
    localCheckRegion(specs{i}, geom{i}, a);
end

mass = 0;
total = 0;
for n = 1:N
    prodR = 1;
    prodT = 1;
    for a = 1:A
        pm     = perm{a};
        pCol   = P{a}(:, n);
        wCol   = W{a}(:, n);
        absent = isnan(pCol);
        pFill  = pCol;  pFill(absent) = 0;
        wFill  = wCol;  wFill(absent | isnan(wCol)) = 0;
        Dtup   = size(pm, 1);
        M      = size(pm, 2);
        wTuple = prod(reshape(wFill(pm), Dtup, M), 1);
        sAll   = sum(wTuple);
        prodT  = prodT * sAll;
        i = find(keys == a, 1);
        if isempty(i)
            prodR = prodR * sAll;
        else
            u = reshape(pFill(pm), Dtup, M);
            coords = localCoordinates(u, geom{i});
            share = localRegionShare(coords, specs{i}, geom{i});
            prodR = prodR * (wTuple * share(:));
        end
    end
    mass  = mass + prodR;
    total = total + prodT;
end
if strcmp(normalize, 'total')
    if total ~= 0
        m = mass / total;
    else
        m = NaN;
    end
else
    m = mass;
end
end


function tf = localIsNested(dens, a)
tf = isfield(dens, 'nested') && iscell(dens.nested) ...
     && numel(dens.nested) >= a && ~isempty(dens.nested{a});
end


function innerR = localInnerR(dens, A)
% Per-attribute co-transposition block size, as evalMaet computes it: the
% block size s_u where a nested attribute is resolved to an inner or
% intermediate [rel] unit, 0 otherwise.
innerR = zeros(1, A);
if isfield(dens, 'nested') && iscell(dens.nested)
    for a = 1:A
        s = dens.nested{a};
        if ~isempty(s) && isstruct(s) && isfield(s, 'proj') ...
                && (strcmp(s.proj, 'inner') || strcmp(s.proj, 'intermediate'))
            innerR(a) = prod(s.r(1:s.relUnit));
        end
    end
end
end


function g = localGeometry(dens, a, innerR, nRows)
% The attribute's kernel in its query coordinates: the metric M (the
% kernel is exp(-d' M d / 2 sigma^2)), the periodic mode, and whether it
% can be restricted at all.
isRel  = logical(dens.isRel(a));
isPer  = logical(dens.isPer(a));
g.period = dens.period(a);
g.innerR = innerR;
g.hasCov = isfield(dens, 'kernelCov') && iscell(dens.kernelCov) ...
    && numel(dens.kernelCov) >= a && ~isempty(dens.kernelCov{a});
if innerR > 0
    b = innerR;
    block = eye(b - 1) - ones(b - 1) / b;
    g.M = kron(eye(nRows / b), block);
    g.kind = 'rel';
elseif isRel
    d = nRows - 1;
    g.M = eye(d) - ones(d) / nRows;
    g.kind = 'rel';
else
    g.M = eye(nRows);
    g.kind = 'abs';
end
g.dim = size(g.M, 1);
g.diag = isequal(g.M, diag(diag(g.M)));
if ~isPer
    g.mode = 'none';
elseif strcmp(g.kind, 'abs')
    g.mode = 'full';
    if isfield(dens, 'wrap') && iscell(dens.wrap) && numel(dens.wrap) >= a ...
            && ~isempty(dens.wrap{a}) && strcmp(char(dens.wrap{a}), 'single-image')
        g.mode = 'single';
    end
else
    g.mode = 'single';   % the relative-periodic pairwise wrap
end
if g.hasCov
    g.sigma = NaN;
else
    g.sigma = dens.sigma(a);
end
end


function c = localCoordinates(u, g)
% Tuple values u (rows x tuples) in the attribute's query coordinates, as
% the evaluation paths reduce them.
if g.innerR > 0
    b = g.innerR;
    nb = size(u, 1) / b;
    c = zeros(nb * (b - 1), size(u, 2));
    for k = 1:nb
        blk = u((k - 1) * b + (1:b), :);
        c((k - 1) * (b - 1) + (1:(b - 1)), :) = blk(2:end, :) - blk(1, :);
    end
elseif strcmp(g.kind, 'rel')
    c = u(2:end, :) - u(1, :);
else
    c = u;
end
end


% -------------------------------------------------------------------------
%  Regions
% -------------------------------------------------------------------------

function [keys, specs] = localParseRegion(region, A)
keys = [];
specs = {};
if isempty(region)
    return;
end
if ~iscell(region) || size(region, 2) ~= 2
    error('massMaet:badRegion', ...
        '''region'' must be an N x 2 cell {a, spec; ...}, a an attribute index.');
end
keys = zeros(1, size(region, 1));
specs = cell(1, size(region, 1));
for i = 1:size(region, 1)
    a = region{i, 1};
    if ~(isnumeric(a) && isscalar(a) && a == round(a) && a >= 1 && a <= A)
        error('massMaet:badRegion', ...
            '''region'' names attribute %s, out of range for %d attributes.', ...
            mat2str(a), A);
    end
    if any(keys(1:i-1) == a)
        error('massMaet:badRegion', '''region'' names attribute %d twice.', a);
    end
    keys(i) = a;
    specs{i} = localParseSpec(region{i, 2}, a);
end
end


function s = localParseSpec(spec, a)
if iscell(spec)
    if numel(spec) ~= 3 || ~(ischar(spec{1}) || isstring(spec{1})) ...
            || ~strcmpi(spec{1}, 'gaussian')
        error('massMaet:badRegion', ...
            '''region'' for attribute %d: a named region is {''gaussian'', centre, sd}.', a);
    end
    s.type = 'gaussian';
    s.centre = double(spec{2}(:));
    s.sd = double(spec{3});
    if ~(isscalar(s.sd) && isfinite(s.sd) && s.sd > 0)
        error('massMaet:badRegion', ...
            '''region'' for attribute %d: the Gaussian''s sd must be finite and positive.', a);
    end
    if ~all(isfinite(s.centre))
        error('massMaet:badRegion', ...
            '''region'' for attribute %d: the Gaussian''s centre must be finite.', a);
    end
    return;
end
B = double(spec);
if isvector(B) && numel(B) == 2
    B = B(:).';
end
if ~ismatrix(B) || size(B, 2) ~= 2
    error('massMaet:badRegion', ...
        '''region'' for attribute %d: a box is [lo hi], or one row [lo hi] per coordinate.', a);
end
if any(isnan(B(:))) || any(B(:, 1) > B(:, 2))
    error('massMaet:badRegion', ...
        '''region'' for attribute %d: each box row must be [lo hi] with lo <= hi.', a);
end
s.type = 'box';
s.bounds = B;
end


function localCheckRegion(s, g, a)
if g.hasCov
    error('massMaet:kernelCov', ...
        ['''region'' names attribute %d, which has a matrix-valued kernel ' ...
         'covariance; its mass in a region is not supported. Leave it out ' ...
         'of the region to integrate over it.'], a);
end
d = g.dim;
if d == 0
    error('massMaet:noCoordinates', ...
        ['''region'' names attribute %d, which has no coordinates (a ' ...
         'relative attribute of one value); leave it out.'], a);
end
if strcmp(s.type, 'box')
    B = s.bounds;
    if ~any(size(B, 1) == [1, d])
        error('massMaet:badRegion', ...
            ['''region'' for attribute %d: the box has %d rows, but the ' ...
             'attribute has %d coordinates; give one row, or one per ' ...
             'coordinate.'], a, size(B, 1), d);
    end
    if ~g.diag
        error('massMaet:noClosedForm', ...
            ['''region'' for attribute %d: a box has no closed form on a ' ...
             'relative attribute of three or more values, whose coordinates ' ...
             'are correlated. Use a Gaussian region {''gaussian'', centre, ' ...
             'sd}, or r = 2.'], a);
    end
    if ~strcmp(g.mode, 'none') && any(B(:, 2) - B(:, 1) > g.period * (1 + 1e-12))
        error('massMaet:badRegion', ...
            ['''region'' for attribute %d: on a periodic attribute a box ' ...
             'spans at most one period (%g).'], a, g.period);
    end
else
    if ~any(numel(s.centre) == [1, d])
        error('massMaet:badRegion', ...
            ['''region'' for attribute %d: the Gaussian''s centre has %d ' ...
             'values, but the attribute has %d coordinates; give one, or ' ...
             'one per coordinate.'], a, numel(s.centre), d);
    end
    if ~strcmp(g.mode, 'none') && ~g.diag
        error('massMaet:noClosedForm', ...
            ['''region'' for attribute %d: a Gaussian region on a periodic ' ...
             'relative attribute of three or more values is not supported.'], a);
    end
end
end


function share = localRegionShare(coords, s, g)
% Each tuple's kernel's share of the region (unit-mass kernels).
[d, n] = size(coords);
if n == 0
    share = zeros(1, 0);
    return;
end
if g.diag
    sds = g.sigma ./ sqrt(diag(g.M));
    share = ones(1, n);
    for k = 1:d
        if strcmp(s.type, 'box')
            B = s.bounds;
            row = B(min(k, size(B, 1)), :);
            reg = {'box', row(1), row(2)};
        else
            reg = {'gauss', s.centre(min(k, numel(s.centre))), s.sd};
        end
        share = share .* localCoordShare(coords(k, :), reg, sds(k), ...
                                         g.mode, g.period);
    end
    return;
end
% A Gaussian region on correlated coordinates (non-periodic): the integral
% of N(x; c, C) exp(-|x - x0|^2 / 2 s^2), with C = sigma^2 M^-1.
sd = s.sd;
x0 = s.centre;
if isscalar(x0), x0 = repmat(x0, d, 1); end
C = g.sigma^2 * inv(g.M); %#ok<MINV>
S = C + sd^2 * eye(d);
fac = 1 / sqrt(det(eye(d) + C / sd^2));
diffs = coords - x0;
q = sum(diffs .* (S \ diffs), 1);
share = fac * exp(-0.5 * q);
end


function out = localNcdfDiff(a, b)
% Phi(b) - Phi(a), computed on the side of zero that keeps it exact.
[a, b] = localBroadcast(a, b);
out = zeros(size(a));
up = a > 0;
out(up)  = 0.5 * erfc(a(up) / sqrt(2)) - 0.5 * erfc(b(up) / sqrt(2));
out(~up) = 0.5 * erfc(-b(~up) / sqrt(2)) - 0.5 * erfc(-a(~up) / sqrt(2));
end


function [a, b] = localBroadcast(a, b)
if isscalar(a) && ~isscalar(b), a = repmat(a, size(b)); end
if isscalar(b) && ~isscalar(a), b = repmat(b, size(a)); end
end


function out = localSeg(a, b, mu, sd, x0, s)
% Integral over [a, b] of g(y) phi_sd(y - mu), with g = 1 (a box, s empty)
% or g(y) = exp(-(y - x0)^2 / 2 s^2) (a Gaussian region). Zero where the
% interval is empty.
n = numel(mu);
a = a + zeros(1, n);
b = b + zeros(1, n);
if isempty(s)
    out = localNcdfDiff((a - mu) / sd, (b - mu) / sd);
else
    v = sd^2 + s^2;
    tau = sd * s / sqrt(v);
    nu = (mu * s^2 + x0 * sd^2) / v;
    amp = (s / sqrt(v)) * exp(-(mu - x0).^2 / (2 * v));
    out = amp .* localNcdfDiff((a - nu) / tau, (b - nu) / tau);
end
out(~(b > a)) = 0;
end


function total = localCoordShare(c, reg, sd, mode, period)
% One coordinate's share for kernels centred at c (1 x n).
kind = reg{1};
if strcmp(mode, 'none')
    if strcmp(kind, 'box')
        total = localSeg(reg{2}, reg{3}, c, sd, [], []);
    else
        total = localSeg(-Inf, Inf, c, sd, reg{2}, reg{3});
    end
    return;
end
P = period;
if strcmp(kind, 'box')
    a = reg{2}; b = reg{3}; x0 = []; s = [];
else
    a = reg{2} - P / 2; b = reg{2} + P / 2; x0 = reg{2}; s = reg{3};
end
total = zeros(size(c));
if strcmp(mode, 'full')
    reach = 9 * sd;   % images beyond 9 sd carry < 1e-18 into the region
    mLo = floor((a - reach - max(c)) / P);
    mHi = ceil((b + reach - min(c)) / P);
    for mm = mLo:mHi
        total = total + localSeg(a, b, c + mm * P, sd, x0, s);
    end
    return;
end
% Single image: the kernel is phi(wrap(y - c)), a Gaussian cut to one
% period around each image of c, renormalized to unit mass.
mLo = floor((a - max(c)) / P - 0.5) - 1;
mHi = ceil((b - min(c)) / P + 0.5) + 1;
for mm = mLo:mHi
    cm = c + mm * P;
    total = total + localSeg(max(a, cm - P / 2), min(b, cm + P / 2), ...
                             cm, sd, x0, s);
end
total = total / localNcdfDiff(-P / (2 * sd), P / (2 * sd));
end

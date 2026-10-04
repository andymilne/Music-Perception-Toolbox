function m = massMaet(dens, nv)
%MASSMAET  Total mass of a multi-attribute expectation tensor (MAET).
%
%   m = massMaet(dens)
%
%   OVERVIEW. The density of a MAET is a weighted sum of Gaussian kernels,
%   one per tuple. Each kernel is taken here with unit mass, so the mass of
%   the whole density is the sum of its tuples' weight products,
%
%       M = sum_j w_j,
%
%   the integral of the density evalMaet returns under 'gaussian'. Where
%   every weight is 1 it is the number of tuples: at r = 2, the number of
%   ordered or unordered pairs of values, as the attribute is exchangeable
%   or not.
%
%   The mass is the natural normalizer for a count. The one-sided
%   similarity ('normalize', 'oneSidedDenom' in simMaet) of a context with
%   a query counts, in units of the query, the context's tuples that match
%   it; multiplied by the query's mass over the context's, it is their
%   share of the context's tuples.
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
%     'verbose'    Passed to buildMaet for a pre-MAET (default false).
%
%   Twin of Python mass_maet.
%
%   See also SWEPTMASS, SIMMAET, EVALMAET, BUILDMAET.

arguments
    dens
    nv.verbose (1,1) logical = false
end

if iscell(dens)
    m = zeros(1, numel(dens));
    for k = 1:numel(dens)
        m(k) = massMaet(dens{k}, 'verbose', nv.verbose);
    end
    return;
end
if internal.isPreMaet(dens)
    dens = buildMaet(dens, 'verbose', nv.verbose);
end
m = localDensityMass(dens);
end


% -------------------------------------------------------------------------
%  One density
% -------------------------------------------------------------------------

function m = localDensityMass(dens)
A    = dens.nAttrs;
N    = dens.N;
P    = dens.pAttr;
W    = dens.w;
rVec = dens.r(:).';
isExchV = dens.exch(:).';

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

total = 0;
for n = 1:N
    prodT = 1;
    for a = 1:A
        pm     = perm{a};
        pCol   = P{a}(:, n);
        wCol   = W{a}(:, n);
        wFill  = wCol;  wFill(isnan(pCol) | isnan(wCol)) = 0;
        Dtup   = size(pm, 1);
        M      = size(pm, 2);
        prodT  = prodT * sum(prod(reshape(wFill(pm), Dtup, M), 1));
    end
    total = total + prodT;
end
m = total;
end


function tf = localIsNested(dens, a)
tf = isfield(dens, 'nested') && iscell(dens.nested) ...
     && numel(dens.nested) >= a && ~isempty(dens.nested{a});
end

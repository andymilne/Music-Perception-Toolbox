function [M, sv] = sweptMass(varargin)
%SWEPTMASS  Align a window on a context at each of a list of sweep values
%   and take the total mass of the windowed density at each.
%
%   OVERVIEW. At each sweep value s, a window h(delta) on the context is
%   aligned at s and weights each event n on the target attribute, as at
%   sweptEntropy:
%
%       w'(n) = w(n) * h(p_a(n) - s).
%
%   The windowed density is then built and its total mass taken by
%   massMaet: the sum of its tuples' weight products, how much material
%   the window holds (with unit weights and a rectangular window, its
%   number of tuples). It is the natural normalizer for a windowed count:
%   a one-sided sweptSimilarity against a query, multiplied by the query's
%   mass over the window's, is the share of the window's tuples that match
%   the query.
%
%   INPUT FORMS, as at sweptEntropy:
%
%     M = sweptMass(pm, ...)
%     M = sweptMass(pAttr, w, sigma, r, rel, per, period, ...)
%
%   NAME-VALUE OPTIONS (per-attribute maps are N x 2 cells {a, value; ...})
%     'sweep'        {a, values; ...}: the sweep values of attribute a; a
%                    bare attribute index, or a vector of them, asks for
%                    the defaults below.
%     'start', 'stop', 'step'
%                    {a, value; ...}: generate attribute a's sweep values
%                    in place of listing them; a bare number applies to
%                    the swept attribute where 'sweep' names one. start
%                    and stop default to the lowest and highest of the
%                    context's values on the attribute; step defaults to
%                    half the window's sd, and a pure rectangle without a
%                    given step takes its pieces (as at sweptSimilarity).
%     'window'       {a, {shape, name, value, ...}; ...}, with the names
%                    'width' | 'sd' | 'decayRate', 'edges', and 'ref' (so
%                    {a, {'gaussian', 'sd', 4}}), the same as
%                    {a, struct('shape', .., ...); ...}, or {a, f; ...}: the
%                    window on each swept attribute, aligned at the sweep
%                    value, with its scale always named. shape is 'rect',
%                    'gaussian', a number in [0, 1] blending the two
%                    (0 Gaussian, 1 rectangle), or 'exponential',
%                    'exponentialBefore', or 'exponentialAfter'. 'width' is
%                    the full width of the rectangle, and 'sd' may be given
%                    instead (a Gaussian of width w has sd w / (2 sqrt(3)));
%                    the exponentials have no width, and take 'sd' or
%                    'decayRate'. edges is 'halfOpen' (the default) or
%                    'closed' (rectangles only). ref, for a rectangle of
%                    given width, is the point of it placed at the sweep
%                    value: 'centre' (the default), 'start', or 'end'.
%                    f is a function handle taking the displacement
%                    p_a(n) - s. Required for every swept attribute: its
%                    scale is that of the local region, which nothing in
%                    the data can supply. The full account is in
%                    sweptSimilarity.
%     'drop'         Vector of swept attributes marginalized after the
%                    window has weighted the events, and so before the mass
%                    is taken.
%     'locate'       As at sweptSimilarity (default 'centroid').
%     'targetAttr'   The attribute whose weights the window multiplies
%                    (default: the first attribute not dropped).
%     'specs', 'exch', 'verbose'
%                    As at sweptSimilarity.
%
%   The output has one dimension per swept attribute, in attribute order;
%   a single swept attribute gives a 1 x n row. [M, sv] = ... also
%   returns the sweep values, listed or generated, as a 1 x A cell: sv{a}
%   holds attribute a's, empty where a is not swept. They are the axes of
%   M, so plot(sv{a}, M) draws a single sweep.
%
%   See also MASSMAET, SWEPTENTROPY, WEIGHTEVENTS, BUILDMAET.

% Top-level call guard: dispatch throttle + kernelChunkBytes pin, so the
% per-value inner calls announce once per sweep. See internal.callGuard.
guard = internal.callGuard(); %#ok<NASGU>

varargin = internal.sweptPreMaetArgs(varargin, 'sweptMass', 1);
[M, plan] = localSweptMass(varargin{:});
if nargout > 1, sv = internal.sweepValues(plan, numel(varargin{1})); end
end


function [M, plan] = localSweptMass(pAttr, w, sigma, r, isRel, isPer, period, nv)
arguments
    pAttr (1,:) cell
    w
    sigma
    r
    isRel
    isPer
    period
    nv.sweep = []
    nv.start = []
    nv.stop = []
    nv.step = []
    nv.window = []
    nv.drop = []
    nv.locate = 'centroid'
    nv.targetAttr = []
    nv.specs = []
    nv.exch = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.exch) && ~isempty(nv.specs)
    error('sweptMass:exchVsSpecs', ...
        ['exch applies to the flat per-attribute surface; nested ' ...
         'geometry carries its per-level exch inside specs. Pass one ' ...
         'or the other.']);
end
nv.align = []; nv.queryRef = [];
plan = internal.sweptPlan(pAttr, [], nv.specs, isRel, nv, ...
    'sweptMass', struct('sigma', {sigma}, 'per', {isPer}, ...
    'period', {period}));

A = numel(pAttr);
dropAxes = plan.dropAxes;
keep = setdiff(1:A, dropAxes);
if isempty(nv.targetAttr), target = keep(1); else, target = nv.targetAttr; end
if any(target == dropAxes)
    error('sweptMass:targetDropped', ...
        ['targetAttr %d is a dropped attribute: its weights are removed ' ...
         'before the build, so the window factors would be lost. Choose a ' ...
         'kept attribute.'], target);
end

nested = ~isempty(nv.specs);
[sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
exchC = internal.subExchArgs(nv.exch, keep);
dims = plan.dims;
axes = [dims.a];
sizes = arrayfun(@(d) numel(d.vals), dims);
if numel(dims) == 1, M = zeros(1, sizes(1)); else, M = zeros(sizes); end
locates = cell(1, numel(axes));
for k = 1:numel(axes), locates{k} = internal.axisLocate(nv.locate, axes(k)); end
for li = 1:prod(sizes)
    subs = internal.lin2sub(sizes, li);
    at = zeros(1, numel(axes));
    for k = 1:numel(axes), at(k) = dims(k).vals(subs(k)); end
    [pc, wc, sc] = internal.applyWindows(pAttr, w, nv.specs, axes, at, ...
        plan.win(axes), locates, target);
    [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, A);
    if nested
        dens = buildMaet(pc, wc, 'sigma', sg, 'per', pr, 'period', pd, ...
            'specs', sc, 'verbose', false);
    else
        dens = buildMaet(pc, wc, sg, rr, rl, pr, pd, exchC{:}, 'verbose', false);
    end
    M(li) = massMaet(dens);
end
end

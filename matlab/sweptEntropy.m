function [H, sv] = sweptEntropy(varargin)
%SWEPTENTROPY  Align a window on a context at each of a list of sweep
%   values and take the entropy of the windowed density at each.
%
%   OVERVIEW. At each of a list of values s on an attribute, the sweep
%   values, a window h(delta) on the context is aligned with its reference
%   value, delta = 0, at s. It weights each event n on the target
%   attribute (event weighting, as by weightEvents):
%
%       w'(n) = w(n) * h(p_a(n) - s),
%
%   where p_a(n) is event n's value on the swept attribute a. The windowed
%   density is then built and its entropy taken, tracing how the entropy
%   changes across the context. Windows, 'locate', and generated sweep
%   values are as at sweptSimilarity; there is no query, so the sweep
%   values always align the window.
%
%   INPUT FORMS, in the order to reach for them:
%
%     H = sweptEntropy(pm, ...)
%
%   with a whole pre-MAET, whose specs give the geometry. Any of the six
%   per-attribute parameters ('sigma', 'per', 'period', 'r', 'rel',
%   'exch') may be given alongside to override it, as at buildMaet, either
%   in full or selectively as a 1 x A cell whose empty entries keep the
%   spec's value.
%
%     H = sweptEntropy(pAttr, w, sigma, r, rel, per, period, ...)
%
%   the raw positional form.
%
%   NAME-VALUE OPTIONS (per-attribute maps are N x 2 cells {a, value; ...})
%     'sweep'        {a, values; ...}: the sweep values of attribute a; a
%                    bare attribute index, or a vector of them, asks for the
%                    defaults below.
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
%                    window has weighted the events. An attribute kept
%                    stays in the density whose entropy is taken.
%     'locate'       As at sweptSimilarity (default 'centroid').
%     'targetAttr'   The attribute whose weights the window multiplies
%                    (default: the first attribute not dropped).
%     'method', 'base'
%                    As at entropyMaet (defaults 'differential', 2).
%     'nPointsPerDim', 'xMin', 'xMax', 'gridLimit'
%                    The grid of the discrete methods ('shannon',
%                    'normalized'), passed to entropyMaet at every sweep
%                    value, so every window's entropy is taken on the same
%                    grid. nPointsPerDim is required for those methods, and
%                    xMin / xMax for a non-periodic attribute that is kept.
%                    The continuous methods ignore them.
%     'specs', 'exch', 'verbose'
%                    As at sweptSimilarity.
%
%   The output has one dimension per swept attribute, in attribute order;
%   a single swept attribute gives a 1 x n row. [H, sv] = ... also
%   returns the sweep values, listed or generated, as a 1 x A cell: sv{a}
%   holds attribute a's, empty where a is not swept. They are the axes of
%   H, so plot(sv{a}, H) draws a single sweep.
%
%   See also PACKPREMAET, SWEPTSIMILARITY, WEIGHTEVENTS, BUILDMAET,
%            ENTROPYMAET.

% Top-level call guard: dispatch throttle + kernelChunkBytes pin, so the
% per-value inner calls announce once per sweep. See internal.callGuard.
guard = internal.callGuard(); %#ok<NASGU>

varargin = internal.sweptPreMaetArgs(varargin, 'sweptEntropy', 1);
[H, plan] = localSweptEntropy(varargin{:});
if nargout > 1, sv = internal.sweepValues(plan, numel(varargin{1})); end
end


function [H, plan] = localSweptEntropy(pAttr, w, sigma, r, isRel, isPer, period, nv)
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
    nv.method (1,:) char = 'differential'
    nv.base (1,1) double = 2.0
    nv.nPointsPerDim = []
    nv.xMin = NaN
    nv.xMax = NaN
    nv.gridLimit = 1e8
    nv.targetAttr = []
    nv.specs = []
    nv.exch = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.exch) && ~isempty(nv.specs)
    error('sweptEntropy:exchVsSpecs', ...
        ['exch applies to the flat per-attribute surface; nested ' ...
         'geometry carries its per-level exch inside specs. Pass one ' ...
         'or the other.']);
end
nv.align = []; nv.queryRef = [];
plan = internal.sweptPlan(pAttr, [], nv.specs, isRel, nv, ...
    'sweptEntropy', struct('sigma', {sigma}, 'per', {isPer}, ...
    'period', {period}));

A = numel(pAttr);
dropAxes = plan.dropAxes;
keep = setdiff(1:A, dropAxes);
if isempty(nv.targetAttr), target = keep(1); else, target = nv.targetAttr; end
if any(target == dropAxes)
    error('sweptEntropy:targetDropped', ...
        ['targetAttr %d is a dropped attribute: its weights are removed ' ...
         'before the build, so the window factors would be lost. Choose a ' ...
         'compared attribute.'], target);
end
nested = ~isempty(nv.specs);
[sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
exchC = internal.subExchArgs(nv.exch, keep);
dims = plan.dims;
axes = [dims.a];
sizes = arrayfun(@(d) numel(d.vals), dims);
if numel(dims) == 1, H = zeros(1, sizes(1)); else, H = zeros(sizes); end
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
    H(li) = entropyMaet(dens, 'method', nv.method, 'base', nv.base, ...
        'nPointsPerDim', nv.nPointsPerDim, 'xMin', nv.xMin, ...
        'xMax', nv.xMax, 'gridLimit', nv.gridLimit, ...
        'verbose', false);
end
end

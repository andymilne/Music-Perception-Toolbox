function [M, sv] = sweptMass(varargin)
%SWEPTMASS  Align a window on a context at each of a list of sweep values
%   and take the mass of the windowed density in a region at each.
%
%   OVERVIEW. At each sweep value s, a window h(delta) on the context is
%   aligned at s and weights each event n on the target attribute, as at
%   sweptEntropy:
%
%       w'(n) = w(n) * h(p_a(n) - s).
%
%   The windowed density is then built and its mass in 'region' taken by
%   massMaet: how much of the local material lies in the region, or, with
%   'normalize', 'total', what share of it does. The window weights events
%   before the density is built; the region is read from the density, so a
%   tuple just outside it still contributes the part of its kernel that
%   crosses the edge, and a region can select tuples (the intervals of a
%   relative attribute, say) where a window can only weight events.
%
%   INPUT FORMS, as at sweptEntropy:
%
%     M = sweptMass(pm, ...)
%     M = sweptMass(pAttr, w, sigma, r, isRel, isPer, period, ...)
%
%   NAME-VALUE OPTIONS (per-attribute maps are N x 2 cells {a, value; ...})
%     'sweep', 'start', 'stop', 'step', 'window', 'drop', 'locate',
%     'targetAttr'
%                    As at sweptEntropy. A dropped attribute is
%                    marginalized before the mass is taken, so it cannot
%                    be restricted.
%     'region'       {a, spec; ...}, keyed by the context's attribute
%                    indices, as at massMaet. Without it, the mass of the
%                    whole windowed density: the window's weighted tuple
%                    count.
%     'normalize'    As at massMaet: 'none' (default), the mass; 'total',
%                    its share of the windowed density's mass.
%     'specs', 'isExch', 'verbose'
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
    nv.region = []
    nv.normalize (1,:) char = 'none'
    nv.targetAttr = []
    nv.specs = []
    nv.isExch = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.isExch) && ~isempty(nv.specs)
    error('sweptMass:isExchVsSpecs', ...
        ['isExch applies to the flat per-attribute surface; nested ' ...
         'geometry carries its per-level exch inside specs. Pass one ' ...
         'or the other.']);
end
nv.align = []; nv.queryRef = [];
plan = internal.sweptPlan(pAttr, [], nv.specs, isRel, nv, ...
    'sweptMass', struct('sigma', {sigma}, 'isPer', {isPer}, ...
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

% The region is keyed by the context's attributes; after the drop, by
% their positions among those kept.
region = nv.region;
if ~isempty(region)
    [rKeys, rVals] = internal.parseMap(region);
    region = cell(numel(rKeys), 2);
    for i = 1:numel(rKeys)
        if any(rKeys(i) == dropAxes)
            error('sweptMass:regionDropped', ...
                ['''region'' names attribute %d, which is dropped: it is ' ...
                 'marginalized before the mass is taken. Keep it, or leave ' ...
                 'it out of the region.'], rKeys(i));
        end
        region{i, 1} = find(keep == rKeys(i), 1);
        if isempty(region{i, 1})
            error('sweptMass:badRegion', ...
                '''region'' names attribute %s, out of range for %d attributes.', ...
                mat2str(rKeys(i)), A);
        end
        region{i, 2} = rVals{i};
    end
end

nested = ~isempty(nv.specs);
[sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
exchC = internal.subExchArgs(nv.isExch, keep);
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
        dens = buildMaet(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, ...
            'specs', sc, 'verbose', false);
    else
        dens = buildMaet(pc, wc, sg, rr, rl, pr, pd, exchC{:}, 'verbose', false);
    end
    M(li) = massMaet(dens, 'region', region, 'normalize', nv.normalize);
end
end

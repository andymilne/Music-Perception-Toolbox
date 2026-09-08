function H = windowedEntropy(varargin)
%WINDOWEDENTROPY  Slide a window across a context and read its entropy at
%   each position.
%
%   Input forms, in the order to reach for them: a whole pre-MAET,
%   the canonical entry; then the raw positional form, with pAttr,
%   wAttr and the five geometry vectors written out.
%
%   Pre-MAET form. In place of pAttr, w and the five geometry vectors,
%   pass a whole pre-MAET:
%
%     H = windowedEntropy(pm, centres, ...)
%
%   The geometry is read from its specs, and any of the six per-attribute
%   parameters -- 'sigma', 'isPer', 'period', 'r', 'rel', 'sym' -- may be
%   given alongside to override it, as at buildExpTens. An override may
%   name every attribute or be selective, a 1 x A cell whose empty
%   entries keep what the spec carries: 'sigma', {[], s, []} sweeps the
%   second attribute's width and leaves the rest to the pre-MAET.
%
%   Shares the placement, window, 'locate', and 'drop' machinery of
%   windowedSimilarity, with the same single-axis ('windowAttr' + 'centres'
%   + 'dropWindowAttr') and multi-axis ('sweep' + 'drop') surfaces; there is
%   no query, so at each position the windowed (and, for a dropped axis,
%   axis-reduced) density is built and its entropy taken. Because there is
%   no query to size a default window from, an explicit 'contextWindow'
%   width (or sd) is required for every swept axis. 'isSym' is the
%   per-attribute symmetry vector of the flat surface ([] keeps the
%   unordered default); required, in particular, for ordered attributes
%   carrying a matrix-valued kernel covariance (see intervalKernelCov);
%   mutually exclusive with 'specs'. 'marginalise' is
%   reserved for integrating a retained axis out of the density and is not
%   yet implemented.
%
%
%   See also PREMAET, WINDOWEDSIMILARITY, WEIGHTEVENTS, BUILDEXPTENS,
%            ENTROPYEXPTENS.

varargin = internal.windowedPreMaetArgs(varargin, 'windowedEntropy', 1);
H = localWindowedEntropy(varargin{:});
end


function H = localWindowedEntropy(pAttr, w, sigma, r, isRel, isPer, ...
        period, centres, nv)
arguments
    pAttr (1,:) cell
    w
    sigma
    r
    isRel
    isPer
    period
    centres = []
    nv.start = []
    nv.stop = []
    nv.step = []
    nv.contextWindow = {1.0, []}
    nv.windowAttr = []
    nv.dropWindowAttr = []
    nv.sweep = []
    nv.drop = []
    nv.locate = 'centroid'
    nv.method (1,:) char = 'differential'
    nv.base (1,1) double = 2.0
    nv.marginalise = []
    nv.targetAttr = []
    nv.specs = []
    nv.isSym = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.isSym) && ~isempty(nv.specs)
    error('windowedEntropy:isSymVsSpecs', ...
        ['isSym applies to the flat per-attribute surface; nested ' ...
         'geometry carries its per-level sym inside specs. Pass one ' ...
         'or the other.']);
end

if ~isempty(nv.marginalise)
    error('windowedEntropy:marginaliseNotImplemented', ...
        ['marginalise (integrating a retained axis out of the density) is ' ...
         'not yet implemented.']);
end
if ~isempty(nv.sweep)
    if isempty(nv.drop)
        error('windowedEntropy:sweepNeedsDrop', 'multi-axis sweep requires a parallel drop.');
    end
    H = local_we_multi(pAttr, w, sigma, r, isRel, isPer, period, nv.isSym, ...
        nv.sweep, nv.drop, nv.contextWindow, nv.locate, nv.method, nv.base, ...
        nv.targetAttr, nv.specs);
    return;
end
if isempty(nv.dropWindowAttr)
    error('windowedEntropy:dropRequired', ...
        'dropWindowAttr is required (true drops the window axis, false retains it).');
end
H = local_we_single(pAttr, w, sigma, r, isRel, isPer, period, nv.isSym, centres, ...
    nv.start, nv.stop, nv.step, nv.contextWindow, nv.windowAttr, ...
    nv.dropWindowAttr, nv.locate, nv.method, nv.base, nv.targetAttr, nv.specs);
end


% =========================================================================
%  single-axis core
% =========================================================================
function H = local_we_single(pAttr, w, sigma, r, isRel, isPer, period, isSym, ...
        centres, startV, stopV, stepV, contextWindow, windowAttr, ...
        dropWindowAttr, locate, method, base, targetAttr, specs) %#ok<INUSL>
    A = numel(pAttr);
    if isempty(windowAttr), axisIdx = A; else, axisIdx = windowAttr; end
    if axisIdx < 1 || axisIdx > A
        error('windowedEntropy:badWindowAttr', ...
            'windowAttr %d out of range for %d attributes.', axisIdx, A);
    end
    nested = ~isempty(specs);
    [gamma, sd] = internal.singleWindow(contextWindow, NaN, axisIdx);
    if dropWindowAttr, dropAxes = axisIdx; else, dropAxes = []; end
    keep = setdiff(1:A, dropAxes);
    if isempty(keep)
        error('windowedEntropy:dropAll', 'dropping the only attribute leaves no density.');
    end
    if isempty(targetAttr), target = keep(1); else, target = targetAttr; end
    if any(target == dropAxes)
        error('windowedEntropy:targetDropped', 'targetAttr is the dropped axis.');
    end
    ctxCentres = internal.resolveCentres(pAttr, axisIdx, centres, startV, stopV, stepV, sd * 2 * sqrt(3));
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    symC = internal.subSymArgs(isSym, keep);
    H = zeros(1, numel(ctxCentres));
    for i = 1:numel(ctxCentres)
        [pc, wc, sc] = internal.applyWindows(pAttr, w, specs, axisIdx, ...
            ctxCentres(i), gamma, sd, {locate}, target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, A);
        if nested
            dens = buildExpTens(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sc, 'verbose', false);
        else
            dens = buildExpTens(pc, wc, sg, rr, rl, pr, pd, symC{:}, 'verbose', false);
        end
        H(i) = entropyExpTens(dens, 'method', method, 'base', base, 'verbose', false);
    end
end


% =========================================================================
%  multi-axis core
% =========================================================================
function H = local_we_multi(pAttr, w, sigma, r, isRel, isPer, period, isSym, ...
        sweepMap, dropMap, contextWindow, locate, method, base, targetAttr, specs)
    n = numel(pAttr);
    [axes, grids] = internal.parseMap(sweepMap);
    if isempty(axes)
        error('windowedEntropy:emptySweep', 'sweep must name at least one axis.');
    end
    [dAxes, dVals] = internal.parseMap(dropMap);
    if ~isequal(sort(axes), sort(dAxes))
        error('windowedEntropy:dropKeys', 'drop must have one entry per sweep key.');
    end
    dropAxes = [];
    for k = 1:numel(axes)
        if dVals{dAxes == axes(k)}, dropAxes(end + 1) = axes(k); end %#ok<AGROW>
    end
    keep = setdiff(1:n, dropAxes);
    if isempty(keep)
        error('windowedEntropy:dropAll', 'every attribute is dropped; no density remains.');
    end
    if isempty(targetAttr), target = keep(1); else, target = targetAttr; end
    if any(target == dropAxes)
        error('windowedEntropy:targetDropped', 'targetAttr is a dropped axis.');
    end
    nested = ~isempty(specs);
    [cwAxes, cwVals] = internal.parseMap(contextWindow);
    K = numel(axes);
    gammas = zeros(1, K); sds = zeros(1, K); locates = cell(1, K);
    for k = 1:K
        ci = find(cwAxes == axes(k), 1);
        if isempty(ci)
            error('windowedEntropy:requiresWidth', ...
                ['windowedEntropy has no query to size the window; give an ' ...
                 'explicit contextWindow entry for every swept axis (axis %d missing).'], axes(k));
        end
        [gammas(k), sds(k)] = internal.resolveWindowStruct(cwVals{ci}, NaN, axes(k));
        locates{k} = locate;
    end
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    symC = internal.subSymArgs(isSym, keep);
    sizes = cellfun(@numel, grids);
    if K == 1, H = zeros(1, sizes(1)); else, H = zeros(sizes); end
    nTot = prod(sizes);
    for li = 1:nTot
        subs = internal.lin2sub(sizes, li);
        centresK = zeros(1, K);
        for k = 1:K, centresK(k) = grids{k}(subs(k)); end
        [pc, wc, sc] = internal.applyWindows(pAttr, w, specs, axes, centresK, gammas, sds, locates, target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, n);
        if nested
            dens = buildExpTens(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sc, 'verbose', false);
        else
            dens = buildExpTens(pc, wc, sg, rr, rl, pr, pd, symC{:}, 'verbose', false);
        end
        H(li) = entropyExpTens(dens, 'method', method, 'base', base, 'verbose', false);
    end
end


% =========================================================================
%  seam helpers (shared structure with windowedSimilarity)
% =========================================================================

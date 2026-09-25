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
%   parameters -- 'sigma', 'isPer', 'period', 'r', 'rel', 'exch' -- may be
%   given alongside to override it, as at buildMaet. An override may
%   name every attribute or be selective, a 1 x A cell whose empty
%   entries keep what the spec carries: 'sigma', {[], s, []} sweeps the
%   second attribute's width and leaves the rest to the pre-MAET.
%
%   Terms (article, Sec. 3, event weighting). A WINDOW, a non-negative
%   profile h centred at a value c, multiplies each event's weights on one
%   attribute ('targetAttr') by h(p_S(n) - c). The WINDOW ATTRIBUTE is the
%   attribute S the window is defined over, and each CENTRE is a value of
%   it at which the window is placed.
%
%   Takes the window attribute, centres, window and 'locate' as
%   windowedSimilarity does: one window attribute ('windowAttr' with
%   'centres' and 'dropWindowAttr') or several ('sweep' with 'drop').
%   There is no query, so at each centre (or combination of centres) the
%   windowed density is built, with any window attribute whose drop is
%   true first marginalized by removing it from the pre-MAET, and its
%   entropy taken. With no query to size a default window from,
%   'contextWindow' must give a width (or sd) for every window attribute.
%   'isExch' is the
%   per-attribute exchangeability vector of the raw positional form ([] keeps the
%   unordered default); required, in particular, for ordered attributes
%   carrying a matrix-valued kernel covariance (see kernelCov); mutually
%   exclusive with 'specs'. 'marginalize' (also accepted as 'marginalise')
%   is reserved for integrating a
%   compared attribute out of the density and is not yet implemented.
%
%
%   See also PACKPREMAET, WINDOWEDSIMILARITY, WEIGHTEVENTS, BUILDMAET,
%            ENTROPYMAET.

% Top-level call guard: dispatch throttle + kernelChunkBytes pin, so the
% per-centre inner calls announce once per sweep. See internal.callGuard.
guard = internal.callGuard(); %#ok<NASGU>

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
    nv.marginalize = []
    nv.marginalise = []
    nv.targetAttr = []
    nv.specs = []
    nv.isExch = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.isExch) && ~isempty(nv.specs)
    error('windowedEntropy:isExchVsSpecs', ...
        ['isExch applies to the flat per-attribute surface; nested ' ...
         'geometry carries its per-level exch inside specs. Pass one ' ...
         'or the other.']);
end

if ~isempty(nv.marginalise)
    if ~isempty(nv.marginalize)
        error('windowedEntropy:marginalizeTwice', ...
            'give marginalize or its alternative spelling marginalise, not both.');
    end
    nv.marginalize = nv.marginalise;
end
if ~isempty(nv.marginalize)
    error('windowedEntropy:marginalizeNotImplemented', ...
        ['marginalize (integrating a compared attribute out of the density) is ' ...
         'not yet implemented.']);
end
if ~isempty(nv.sweep)
    if isempty(nv.drop)
        error('windowedEntropy:sweepNeedsDrop', 'sweep requires a parallel drop.');
    end
    H = local_we_multi(pAttr, w, sigma, r, isRel, isPer, period, nv.isExch, ...
        nv.sweep, nv.drop, nv.contextWindow, nv.locate, nv.method, nv.base, ...
        nv.targetAttr, nv.specs);
    return;
end
if isempty(nv.dropWindowAttr)
    error('windowedEntropy:dropRequired', ...
        'dropWindowAttr is required (true marginalizes the window attribute, false retains it).');
end
H = local_we_single(pAttr, w, sigma, r, isRel, isPer, period, nv.isExch, centres, ...
    nv.start, nv.stop, nv.step, nv.contextWindow, nv.windowAttr, ...
    nv.dropWindowAttr, nv.locate, nv.method, nv.base, nv.targetAttr, nv.specs);
end


% =========================================================================
%  one window attribute
% =========================================================================
function H = local_we_single(pAttr, w, sigma, r, isRel, isPer, period, isExch, ...
        centres, startV, stopV, stepV, contextWindow, windowAttr, ...
        dropWindowAttr, locate, method, base, targetAttr, specs) %#ok<INUSL>
    A = numel(pAttr);
    if isempty(windowAttr), axisIdx = A; else, axisIdx = windowAttr; end
    if axisIdx < 1 || axisIdx > A
        error('windowedEntropy:badWindowAttr', ...
            'windowAttr %d out of range for %d attributes.', axisIdx, A);
    end
    locate = internal.axisLocate(locate, axisIdx);
    nested = ~isempty(specs);
    [gamma, sd] = internal.singleWindow(contextWindow, NaN, axisIdx);
    if dropWindowAttr, dropAxes = axisIdx; else, dropAxes = []; end
    keep = setdiff(1:A, dropAxes);
    if isempty(keep)
        error('windowedEntropy:dropAll', 'dropping the only attribute leaves no density.');
    end
    if isempty(targetAttr), target = keep(1); else, target = targetAttr; end
    if any(target == dropAxes)
        error('windowedEntropy:targetDropped', 'targetAttr is the marginalized window attribute.');
    end
    ctxCentres = internal.resolveCentres(pAttr, axisIdx, centres, startV, stopV, stepV, sd * 2 * sqrt(3));
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    exchC = internal.subExchArgs(isExch, keep);
    H = zeros(1, numel(ctxCentres));
    for i = 1:numel(ctxCentres)
        [pc, wc, sc] = internal.applyWindows(pAttr, w, specs, axisIdx, ...
            ctxCentres(i), gamma, sd, {locate}, target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, A);
        if nested
            dens = buildMaet(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sc, 'verbose', false);
        else
            dens = buildMaet(pc, wc, sg, rr, rl, pr, pd, exchC{:}, 'verbose', false);
        end
        H(i) = entropyMaet(dens, 'method', method, 'base', base, 'verbose', false);
    end
end


% =========================================================================
%  several window attributes
% =========================================================================
function H = local_we_multi(pAttr, w, sigma, r, isRel, isPer, period, isExch, ...
        sweepMap, dropMap, contextWindow, locate, method, base, targetAttr, specs)
    n = numel(pAttr);
    [axes, grids] = internal.parseMap(sweepMap);
    if isempty(axes)
        error('windowedEntropy:emptySweep', 'sweep must name at least one attribute.');
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
        error('windowedEntropy:targetDropped', 'targetAttr is a marginalized window attribute.');
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
                 'explicit contextWindow entry for every window attribute (attribute %d missing).'], axes(k));
        end
        [gammas(k), sds(k)] = internal.resolveWindowStruct(cwVals{ci}, NaN, axes(k));
        locates{k} = internal.axisLocate(locate, axes(k));
    end
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    exchC = internal.subExchArgs(isExch, keep);
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
            dens = buildMaet(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sc, 'verbose', false);
        else
            dens = buildMaet(pc, wc, sg, rr, rl, pr, pd, exchC{:}, 'verbose', false);
        end
        H(li) = entropyMaet(dens, 'method', method, 'base', base, 'verbose', false);
    end
end


% =========================================================================
%  seam helpers (shared structure with windowedSimilarity)
% =========================================================================

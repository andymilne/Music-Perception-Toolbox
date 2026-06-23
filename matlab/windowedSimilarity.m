function out = windowedSimilarity(pContext, wContext, pQuery, wQuery, ...
        sigma, r, isRel, isPer, period, centres, nv)
%WINDOWEDSIMILARITY  Slide a query across a context and measure their
%   similarity at each position (a pre-MAET cross-correlation).
%
%   Two equivalent argument surfaces:
%
%   * Single axis (the common case): name the swept axis with 'windowAttr'
%     and its positions with 'centres' (or 'start'/'stop'/'step');
%     'dropWindowAttr' says whether that axis is compared (false) or only
%     places the comparison (true). The window is centred on the query's
%     'locate' value (the multiset centroid by default) and sized to the
%     query's extent unless 'contextWindow' = {shape, width} overrides it.
%     'queryCentres' decouples the query's placement from the window's:
%     [] locks them; an A-by-T matrix fixes the window at each centres(a)
%     while the query slides across queryCentres(a, :), giving the lagged
%     correlogram surface.
%   * Multiple axes: give 'sweep' = {axis, positions; ...} and a parallel
%     'drop' = {axis, tf; ...}; the output gains one dimension per swept
%     axis. 'contextWindow' is then a map {axis, struct('shape',..,'width'/'sd',..); ...}.
%
%   'locate' is 'centroid' (default) | 'start' | 'end' | 'mid' | a handle.
%   'targetAttr' is the attribute whose weights absorb the window factors
%   (default: first compared attribute; may coincide with a swept axis).
%   'specs' carries nested geometry from bindEvents. The window-factor and
%   comparison-kernel truncation both read the global mptDefaults setting.
%
%   See also WINDOWEDENTROPY, WEIGHTEVENTS, TRANSLATEATTRIBUTES, COSSIMEXPTENS.

arguments
    pContext (1,:) cell
    wContext
    pQuery   (1,:) cell
    wQuery
    sigma
    r
    isRel
    isPer
    period
    centres = []
    nv.start = []
    nv.stop = []
    nv.step = []
    nv.queryCentres = []
    nv.contextWindow = {1.0, []}
    nv.queryWindow = []
    nv.windowAttr = []
    nv.dropWindowAttr = []
    nv.sweep = []
    nv.drop = []
    nv.locate = 'centroid'
    nv.targetAttr = []
    nv.normalize (1,:) char = 'oneSidedDenom'
    nv.specs = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.sweep)
    if isempty(nv.drop)
        error('windowedSimilarity:sweepNeedsDrop', ...
            'multi-axis sweep requires a parallel drop.');
    end
    if ~isempty(nv.queryCentres) || ~isempty(nv.queryWindow)
        error('windowedSimilarity:multiAxisQuery', ...
            ['queryCentres/queryWindow are single-axis arguments; the ' ...
             'multi-axis sweep form locks the query to the sweep.']);
    end
    out = local_ws_multi(pContext, wContext, pQuery, wQuery, sigma, r, isRel, ...
        isPer, period, nv.sweep, nv.drop, nv.contextWindow, nv.locate, ...
        nv.normalize, nv.targetAttr, nv.specs);
    return;
end
if isempty(nv.dropWindowAttr)
    error('windowedSimilarity:dropRequired', ...
        'dropWindowAttr is required (true places only, false compares).');
end
out = local_ws_single(pContext, wContext, pQuery, wQuery, sigma, r, isRel, ...
    isPer, period, centres, nv.start, nv.stop, nv.step, nv.queryCentres, ...
    nv.contextWindow, nv.queryWindow, nv.windowAttr, nv.dropWindowAttr, ...
    nv.locate, nv.targetAttr, nv.normalize, nv.specs);
end


% =========================================================================
%  single-axis core
% =========================================================================
function out = local_ws_single(pContext, wContext, pQuery, wQuery, sigma, r, ...
        isRel, isPer, period, centres, startV, stopV, stepV, queryCentres, ...
        contextWindow, queryWindow, windowAttr, dropWindowAttr, locate, ...
        targetAttr, normalize, specs) %#ok<INUSL>
    A = numel(pContext);
    if isempty(windowAttr), axisIdx = A; else, axisIdx = windowAttr; end
    if axisIdx < 1 || axisIdx > A
        error('windowedSimilarity:badWindowAttr', ...
            'windowAttr %d out of range for %d attributes.', axisIdx, A);
    end
    nested = ~isempty(specs);
    [gamma, sd] = internal.singleWindow(contextWindow, internal.queryExtent(pQuery, axisIdx), axisIdx);
    if dropWindowAttr, dropAxes = axisIdx; else, dropAxes = []; end
    keep = setdiff(1:A, dropAxes);
    if isempty(keep)
        error('windowedSimilarity:dropAll', ...
            'dropping the only attribute leaves nothing to compare.');
    end
    if isempty(targetAttr), target = keep(1); else, target = targetAttr; end
    if any(target == dropAxes)
        error('windowedSimilarity:targetDropped', ...
            'targetAttr is the dropped axis; its weights are removed before the build.');
    end
    ctxCentres = internal.resolveCentres(pContext, axisIdx, centres, startV, ...
        stopV, stepV, sd * 2 * sqrt(3));
    Ac = numel(ctxCentres);
    if isempty(queryCentres)
        qRows = ctxCentres(:);
        outShape = [1, Ac];
    else
        qc = double(queryCentres);
        if isvector(qc)
            if numel(qc) ~= Ac
                error('windowedSimilarity:qcLen', ...
                    '1-D queryCentres must have length A = %d.', Ac);
            end
            qRows = qc(:); outShape = [1, Ac];
        else
            if size(qc, 1) ~= Ac
                error('windowedSimilarity:qcRows', ...
                    '2-D queryCentres must have first dimension A = %d.', Ac);
            end
            qRows = qc; outShape = size(qc);
        end
    end
    relAxis = internal.axisIsRel(specs, isRel, axisIdx);
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    T = size(qRows, 2);
    out = zeros(Ac, T);
    for a = 1:Ac
        [pc, wc, sc] = internal.applyWindows(pContext, wContext, specs, ...
            axisIdx, ctxCentres(a), gamma, sd, {locate}, target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, A);
        if nested
            dc = buildExpTens(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, ...
                'specs', sc, 'verbose', false);
        end
        for t = 1:T
            if dropWindowAttr || relAxis
                pqT = pQuery; wqT = wQuery; sqT = specs;
            else
                qLoc = mean(internal.locateRow(pQuery{axisIdx}, locate), 'omitnan');
                offs = cell(1, A); offs{axisIdx} = qRows(a, t) - qLoc;
                [pqT, wqT, sqT] = translateAttributes(pQuery, wQuery, offs, 'specs', specs);
            end
            [pq, wq, sq] = internal.dropAxes(pqT, wqT, sqT, dropAxes, A);
            if nested
                dq = buildExpTens(pq, wq, 'sigma', sg, 'isPer', pr, 'period', pd, ...
                    'specs', sq, 'verbose', false);
                out(a, t) = cosSimExpTens(dc, dq, 'normalize', normalize, 'verbose', false);
            else
                out(a, t) = cosSimExpTens(pc, wc, pq, wq, sg, rr, rl, pr, pd, ...
                    'normalize', normalize, 'verbose', false);
            end
        end
    end
    out = reshape(out, outShape);
end


% =========================================================================
%  multi-axis core
% =========================================================================
function out = local_ws_multi(pContext, wContext, pQuery, wQuery, sigma, r, ...
        isRel, isPer, period, sweepMap, dropMap, contextWindow, locate, ...
        normalize, targetAttr, specs)
    n = numel(pContext);
    [axes, grids] = internal.parseMap(sweepMap);
    if isempty(axes)
        error('windowedSimilarity:emptySweep', 'sweep must name at least one axis.');
    end
    [dAxes, dVals] = internal.parseMap(dropMap);
    if ~isequal(sort(axes), sort(dAxes))
        error('windowedSimilarity:dropKeys', 'drop must have one entry per sweep key.');
    end
    dropAxes = [];
    for k = 1:numel(axes)
        if dVals{dAxes == axes(k)}, dropAxes(end + 1) = axes(k); end %#ok<AGROW>
    end
    keep = setdiff(1:n, dropAxes);
    if isempty(keep)
        error('windowedSimilarity:dropAll', ...
            'every attribute is dropped; nothing is left to compare.');
    end
    if isempty(targetAttr), target = keep(1); else, target = targetAttr; end
    if any(target == dropAxes)
        error('windowedSimilarity:targetDropped', 'targetAttr is a dropped axis.');
    end
    nested = ~isempty(specs);
    [cwAxes, cwVals] = internal.parseMap(contextWindow);
    K = numel(axes);
    gammas = zeros(1, K); sds = zeros(1, K); relF = false(1, K);
    locates = cell(1, K); qLocs = zeros(1, K);
    for k = 1:K
        s = [];
        ci = find(cwAxes == axes(k), 1);
        if ~isempty(ci), s = cwVals{ci}; end
        [gammas(k), sds(k)] = internal.resolveWindowStruct(s, ...
            internal.queryExtent(pQuery, axes(k)), axes(k));
        relF(k) = internal.axisIsRel(specs, isRel, axes(k));
        locates{k} = locate;
        qLocs(k) = mean(internal.locateRow(pQuery{axes(k)}, locate), 'omitnan');
    end
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    sizes = cellfun(@numel, grids);
    if K == 1, out = zeros(1, sizes(1)); else, out = zeros(sizes); end
    nTot = prod(sizes);
    for li = 1:nTot
        subs = internal.lin2sub(sizes, li);
        centresK = zeros(1, K);
        for k = 1:K, centresK(k) = grids{k}(subs(k)); end
        offs = cell(1, n); doTrans = false;
        for k = 1:K
            a = axes(k);
            if any(a == dropAxes) || relF(k), continue; end
            offs{a} = centresK(k) - qLocs(k); doTrans = true;
        end
        if doTrans
            [pqT, wqT, sqT] = translateAttributes(pQuery, wQuery, offs, 'specs', specs);
        else
            pqT = pQuery; wqT = wQuery; sqT = specs;
        end
        [pc, wc, sc] = internal.applyWindows(pContext, wContext, specs, axes, ...
            centresK, gammas, sds, locates, target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, n);
        [pq, wq, sq] = internal.dropAxes(pqT, wqT, sqT, dropAxes, n);
        if nested
            dc = buildExpTens(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sc, 'verbose', false);
            dq = buildExpTens(pq, wq, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sq, 'verbose', false);
            out(li) = cosSimExpTens(dc, dq, 'normalize', normalize, 'verbose', false);
        else
            out(li) = cosSimExpTens(pc, wc, pq, wq, sg, rr, rl, pr, pd, 'normalize', normalize, 'verbose', false);
        end
    end
end


% =========================================================================
%  seam helpers
% =========================================================================

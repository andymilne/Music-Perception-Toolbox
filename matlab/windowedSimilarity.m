function out = windowedSimilarity(varargin)
%WINDOWEDSIMILARITY  Slide a query across a context and measure their
%   similarity at each position (a pre-MAET cross-correlation).
%
%   Input forms, in the order to reach for them: two whole pre-MAETs,
%   the canonical entry; then the raw positional form, with the
%   operands' parts and the five geometry vectors written out.
%
%   Pre-MAET form. In place of the operands and the five geometry
%   vectors, pass whole pre-MAETs:
%
%     out = windowedSimilarity(pmContext, pmQuery, centres, ...)
%
%   The shared geometry is read from their specs, and any of the six
%   per-attribute parameters -- 'sigma', 'isPer', 'period', 'r',
%   'rel', 'exch' -- may be given alongside to override it, as at
%   buildMaet. An override may name every attribute or be
%   selective, a 1 x A cell whose empty entries keep what the spec
%   carries: 'sigma', {[], s, []} sweeps the second attribute's
%   width and leaves the rest to the pre-MAET.
%
%   Terms (article, Sec. 3, event weighting and attribute translation).
%   The context is restricted to a local region by a WINDOW, a non-negative
%   profile h centred at a value c: each event's weights on one attribute
%   ('targetAttr', below) are multiplied by h(p_S(n) - c). The WINDOW
%   ATTRIBUTE is the attribute S the window is defined over, and each
%   CENTRE is a value of it at which the window is placed. An OFFSET
%   translates every element of one of the query's attributes by that
%   amount.
%
%   One window attribute (the common case): name it with 'windowAttr'
%   (default: the last attribute) and give the centres as 'centres', or as
%   'start'/'stop'/'step'. 'dropWindowAttr' says whether the window
%   attribute is compared (false) or, once the window has weighted the
%   events, marginalized by removing it from both pre-MAETs (true).
%   'locate' reduces each event's element multiset on the window attribute
%   to the one value the window reads: 'centroid' (the mean; default),
%   'start' (the first element), 'end' (the last), 'mid' (the midpoint of
%   the first and last), or a function handle. The query's POSITION on an
%   attribute is the mean, over its events, of their located values.
%   Unless 'offsets' are given, at each centre the query is translated
%   along the window attribute so that its position is the centre
%   (ordinary cross-correlation), or left as it is where the window
%   attribute is marginalized or relative. The window defaults to a
%   rectangle as wide as the query's extent on the window attribute (the
%   range of its values there); 'contextWindow' = {shape, width} overrides
%   it.
%
%   Several window attributes: give 'sweep' = {a, centres; ...}, where a is
%   an attribute index, and a parallel 'drop' = {a, tf; ...}. The output
%   has one dimension per attribute named, in the order named, and holds
%   every combination of their centres. 'contextWindow' is then a map
%   {a, struct('shape', .., 'width' or 'sd', ..); ...}, and 'locate' may be
%   one too, {a, rule; ...}, an attribute it does not name taking
%   'centroid'.
%
%   'offsets' translate the query, and the output is indexed by the
%   offsets, each measured from the query's values as given. Aligned before
%   any preprocessing, so that the query's first time value equals the
%   context's, an offset is the time from the start of the context to the
%   start of the query; differencing and binding leave the surviving values
%   unchanged, so the reading carries through them. With one window
%   attribute, a vector of offsets and centres = [] lets the window travel
%   with the query: at each offset the query is translated along the window
%   attribute and the window is centred on its translated position (the
%   offset plus its position). With centres as well, the window stays at
%   each centre while the query is translated by each offset, giving a
%   CORRELOGRAM: offsets is a vector, shared by every centre, or C x T,
%   with C the number of centres and T the number of offsets, and the
%   output is C x T. The window attribute must then be compared and
%   absolute. With several window attributes, 'offsets' is a map
%   {a, offsets; ...} naming the attributes to translate: each is compared,
%   and is windowed, the window travelling with the query, only if
%   'contextWindow' names it; no attribute may be named in both 'offsets'
%   and 'sweep'. Where the window does not move with the query --- the
%   correlogram, or a translated attribute with no window --- the offsets
%   at each window position are computed in one pass by sweepSimMaet (a
%   nested attribute on its contraction route), falling back to one
%   comparison per offset where no such route applies.
%
%   'targetAttr' is the attribute whose per-event weights the window
%   multiplies (default: the first compared attribute; it may be the
%   window attribute). %   'specs' carries nested geometry from bindEvents. 'isExch' is the
%   per-attribute exchangeability vector of the raw positional form ([] keeps the
%   unordered default); required, in particular, for ordered attributes
%   carrying a matrix-valued kernel covariance (see kernelCov).
%   It is mutually exclusive with 'specs', whose nesting carries its
%   own per-level exch. The window-factor and
%   comparison-kernel truncation both read the global mptDefaults setting.
%
%   See also PACKPREMAET, WINDOWEDENTROPY, WEIGHTEVENTS, TRANSLATEATTRIBUTES,
%            SIMMAET.

% Top-level call guard: dispatch throttle + kernelChunkBytes pin, so the
% per-centre inner calls announce once per sweep. See internal.callGuard.
guard = internal.callGuard(); %#ok<NASGU>

varargin = internal.windowedPreMaetArgs(varargin, 'windowedSimilarity', 2);
out = localWindowedSimilarity(varargin{:});
end


function out = localWindowedSimilarity(pContext, wContext, pQuery, wQuery, ...
        sigma, r, isRel, isPer, period, centres, nv)
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
    nv.offsets = []
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
    nv.querySpecs = []
    nv.isExch = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.isExch) && ~isempty(nv.specs)
    error('windowedSimilarity:isExchVsSpecs', ...
        ['isExch applies to the flat per-attribute surface; nested ' ...
         'geometry carries its per-level exch inside specs. Pass one ' ...
         'or the other.']);
end

translateOnly = [];
queryPos = [];
if ~isempty(nv.offsets)
    if iscell(nv.offsets)
        if ~isempty(nv.queryWindow)
            error('windowedSimilarity:multiAxisQuery', ...
                ['queryWindow applies with one window attribute; with ' ...
                 'several, the query is placed at each combination of ' ...
                 'centres.']);
        end
        [nv.sweep, nv.drop, translateOnly] = localOffsetsMulti(pQuery, ...
            nv.specs, isRel, nv.offsets, nv.sweep, nv.drop, nv.locate, ...
            nv.contextWindow);
    else
        if ~isempty(nv.sweep)
            error('windowedSimilarity:offsetsForm', ...
                ['with several window attributes (sweep), give offsets ' ...
                 'as a map {a, offsets; ...}.']);
        end
        [centres, queryPos, nv.dropWindowAttr] = localOffsetsSingle(pQuery, ...
            nv.specs, isRel, nv.offsets, centres, nv.start, nv.stop, nv.step, ...
            nv.windowAttr, nv.dropWindowAttr, nv.locate, numel(pContext));
    end
end

if ~isempty(nv.sweep)
    if isempty(nv.drop)
        error('windowedSimilarity:sweepNeedsDrop', ...
            'sweep requires a parallel drop.');
    end
    if ~isempty(nv.queryWindow)
        error('windowedSimilarity:multiAxisQuery', ...
            ['queryWindow applies with one window attribute; with ' ...
             'several (sweep), the query is placed at each combination ' ...
             'of centres.']);
    end
    out = local_ws_multi(pContext, wContext, pQuery, wQuery, sigma, r, isRel, ...
        isPer, period, nv.isExch, nv.sweep, nv.drop, nv.contextWindow, nv.locate, ...
        nv.normalize, nv.targetAttr, nv.specs, nv.querySpecs, translateOnly);
    return;
end
if isempty(nv.dropWindowAttr)
    error('windowedSimilarity:dropRequired', ...
        'dropWindowAttr is required (true places only, false compares).');
end
out = local_ws_single(pContext, wContext, pQuery, wQuery, sigma, r, isRel, ...
    isPer, period, nv.isExch, centres, nv.start, nv.stop, nv.step, queryPos, ...
    nv.contextWindow, nv.queryWindow, nv.windowAttr, nv.dropWindowAttr, ...
    nv.locate, nv.targetAttr, nv.normalize, nv.specs, nv.querySpecs);
end


% =========================================================================
%  one window attribute
% =========================================================================
function out = local_ws_single(pContext, wContext, pQuery, wQuery, sigma, r, ...
        isRel, isPer, period, isExch, centres, startV, stopV, stepV, queryPos, ...
        contextWindow, queryWindow, windowAttr, dropWindowAttr, locate, ...
        targetAttr, normalize, specs, querySpecs) %#ok<INUSL>
%   queryPos is [] (the query placed with the window at each centre) or the
%   A x T query positions of the correlogram, from localOffsetsSingle.
    if isempty(querySpecs), querySpecs = specs; end
    A = numel(pContext);
    if isempty(windowAttr), axisIdx = A; else, axisIdx = windowAttr; end
    if axisIdx < 1 || axisIdx > A
        error('windowedSimilarity:badWindowAttr', ...
            'windowAttr %d out of range for %d attributes.', axisIdx, A);
    end
    locate = internal.axisLocate(locate, axisIdx);
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
            'targetAttr is the marginalized window attribute; its weights are removed before the build.');
    end
    ctxCentres = internal.resolveCentres(pContext, axisIdx, centres, startV, ...
        stopV, stepV, sd * 2 * sqrt(3));
    Ac = numel(ctxCentres);
    if isempty(queryPos)
        qRows = ctxCentres(:);
        outShape = [1, Ac];
    else
        qRows = double(queryPos);
        outShape = size(qRows);
    end
    relAxis = internal.axisIsRel(specs, isRel, axisIdx);
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    exchC = internal.subExchArgs(isExch, keep);
    T = size(qRows, 2);
    out = zeros(Ac, T);
    % The query window attaches to the query, not to the sweep: it is centred
    % on the query's own position on the window attribute and applied before the
    % per-offset translation, so a template's finite extent is a property of
    % the template and does not change as it slides. Resolved once, outside
    % both loops, because neither the query nor its window varies with the
    % sweep position. Twin of the Python _ws_single.
    if ~isempty(queryWindow)
        [qGamma, qSd] = internal.singleWindow(queryWindow, ...
            internal.queryExtent(pQuery, axisIdx), axisIdx);
        qCentre = mean(internal.locateRow(pQuery{axisIdx}, locate), 'omitnan');
        [pQuery, wQuery, sqW] = internal.applyWindows(pQuery, wQuery, querySpecs, ...
            axisIdx, qCentre, qGamma, qSd, {locate}, target);
        if nested, querySpecs = sqW; end
    end
    for a = 1:Ac
        [pc, wc, sc] = internal.applyWindows(pContext, wContext, specs, ...
            axisIdx, ctxCentres(a), gamma, sd, {locate}, target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, A);
        % Correlogram (offsets with centres): the window stays at this
        % centre while the query is translated, so the windowed context is
        % fixed across the offsets and they can be swept in one pass
        % (localOffsetsSingle has refused a dropped or relative window
        % attribute). Twin of the Python _ws_single.
        if ~isempty(queryPos)
            qLoc = mean(internal.locateRow(pQuery{axisIdx}, locate), 'omitnan');
            [pq0, wq0, sq0] = internal.dropAxes(pQuery, wQuery, querySpecs, dropAxes, A);
            offsRow = zeros(numel(keep), T);
            offsRow(keep == axisIdx, :) = qRows(a, :) - qLoc;
            row = localSweepRow(pc, wc, sc, pq0, wq0, sq0, sg, rr, rl, pr, pd, ...
                exchC, nested, offsRow, normalize);
            if ~isempty(row)
                out(a, :) = row;
                continue;
            end
        end
        if nested
            dc = buildMaet(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, ...
                'specs', sc, 'verbose', false);
        end
        for t = 1:T
            if dropWindowAttr || relAxis
                pqT = pQuery; wqT = wQuery; sqT = querySpecs;
            else
                qLoc = mean(internal.locateRow(pQuery{axisIdx}, locate), 'omitnan');
                offs = cell(1, A); offs{axisIdx} = qRows(a, t) - qLoc;
                [pqT, wqT, sqT] = unpackPreMaet(translateAttributes(pQuery, wQuery, offs, 'specs', querySpecs));
            end
            [pq, wq, sq] = internal.dropAxes(pqT, wqT, sqT, dropAxes, A);
            if nested
                dq = buildMaet(pq, wq, 'sigma', sg, 'isPer', pr, 'period', pd, ...
                    'specs', sq, 'verbose', false);
                out(a, t) = simMaet(dc, dq, 'normalize', normalize, 'verbose', false);
            else
                out(a, t) = simMaet(pc, wc, pq, wq, sg, rr, rl, pr, pd, ...
                    exchC{:}, 'normalize', normalize, 'verbose', false);
            end
        end
    end
    out = reshape(out, outShape);
end


% =========================================================================
%  several window attributes
% =========================================================================
function out = local_ws_multi(pContext, wContext, pQuery, wQuery, sigma, r, ...
        isRel, isPer, period, isExch, sweepMap, dropMap, contextWindow, locate, ...
        normalize, targetAttr, specs, querySpecs, translateOnly)
    if nargin < 19, translateOnly = []; end
    if isempty(querySpecs), querySpecs = specs; end
    n = numel(pContext);
    [axes, grids] = internal.parseMap(sweepMap);
    if isempty(axes)
        error('windowedSimilarity:emptySweep', 'sweep must name at least one attribute.');
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
        error('windowedSimilarity:targetDropped', 'targetAttr is a marginalized window attribute.');
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
        locates{k} = internal.axisLocate(locate, axes(k));
        qLocs(k) = mean(internal.locateRow(pQuery{axes(k)}, locates{k}), 'omitnan');
    end
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    exchC = internal.subExchArgs(isExch, keep);
    sizes = cellfun(@numel, grids);
    if K == 1, out = zeros(1, sizes(1)); else, out = zeros(sizes); end
    nTot = prod(sizes);
    % Translated attributes with no window leave the windowed context
    % unchanged across their offsets, so at each position of the other
    % (windowed-only) attributes the context is fixed and the offsets can
    % be swept in one pass. That holds when every other swept attribute is
    % windowed but not translated (dropped, or relative). Twin of the
    % Python _ws_multi.
    isT = ismember(axes, translateOnly);
    winSel = ~isT;
    tK = find(isT); wK = find(~isT);
    routable = ~isempty(tK) && all(ismember(axes(wK), dropAxes) | relF(wK));
    done = false(1, nTot);
    strides = cumprod([1, sizes(1:end-1)]);
    if routable
        [pq0, wq0, sq0] = internal.dropAxes(pQuery, wQuery, querySpecs, dropAxes, n);
        tSizes = sizes(tK); M = prod(tSizes);
        offsM = zeros(numel(keep), M);
        tSubs = zeros(M, numel(tK));
        for m = 1:M
            tSubs(m, :) = internal.lin2sub(tSizes, m);
            for k = 1:numel(tK)
                kk = tK(k);
                offsM(keep == axes(kk), m) = grids{kk}(tSubs(m, k)) - qLocs(kk);
            end
        end
        wSizes = sizes(wK);
        for wi = 1:prod(wSizes)
            wSubs = internal.lin2sub(wSizes, wi);
            cW = zeros(1, numel(wK));
            for k = 1:numel(wK), cW(k) = grids{wK(k)}(wSubs(k)); end
            [pc, wc, sc] = internal.applyWindows(pContext, wContext, specs, ...
                axes(wK), cW, gammas(wK), sds(wK), locates(wK), target);
            [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, n);
            row = localSweepRow(pc, wc, sc, pq0, wq0, sq0, sg, rr, rl, pr, pd, ...
                exchC, nested, offsM, normalize);
            if isempty(row), continue; end
            for m = 1:M
                subs = zeros(1, K);
                subs(wK) = wSubs;
                subs(tK) = tSubs(m, :);
                li = 1 + sum((subs - 1) .* strides);
                out(li) = row(m);
                done(li) = true;
            end
        end
    end
    for li = 1:nTot
        if done(li), continue; end
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
            [pqT, wqT, sqT] = unpackPreMaet(translateAttributes(pQuery, wQuery, offs, 'specs', querySpecs));
        else
            pqT = pQuery; wqT = wQuery; sqT = querySpecs;
        end
        [pc, wc, sc] = internal.applyWindows(pContext, wContext, specs, ...
            axes(winSel), centresK(winSel), gammas(winSel), sds(winSel), ...
            locates(winSel), target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, n);
        [pq, wq, sq] = internal.dropAxes(pqT, wqT, sqT, dropAxes, n);
        if nested
            dc = buildMaet(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sc, 'verbose', false);
            dq = buildMaet(pq, wq, 'sigma', sg, 'isPer', pr, 'period', pd, 'specs', sq, 'verbose', false);
            out(li) = simMaet(dc, dq, 'normalize', normalize, 'verbose', false);
        else
            out(li) = simMaet(pc, wc, pq, wq, sg, rr, rl, pr, pd, ...
                exchC{:}, 'normalize', normalize, 'verbose', false);
        end
    end
end


% =========================================================================
%  seam helpers
% =========================================================================


% =========================================================================
%  offsets: translation of the query (twins of the Python helpers)
% =========================================================================
function qPos = localQueryPosition(pQuery, a, locate)
%LOCALQUERYPOSITION  The query's position on attribute A: the mean, over its
%   events, of each event's element multiset reduced by locate.
    qPos = mean(internal.locateRow(pQuery{a}, internal.axisLocate(locate, a)), ...
        'omitnan');
end


function [centres, qc, dropW] = localOffsetsSingle(pQuery, specs, isRel, ...
        offsets, centres, startV, stopV, stepV, windowAttr, dropW, locate, A)
%LOCALOFFSETSSINGLE  Offsets for one window attribute as the placement
%   (centres and query positions) the comparison uses.
%   Without centres the window travels with the query (its centre is the
%   offset plus the query's position) and QC is []; with centres it stays
%   at each centre while the query is translated by each offset, the
%   correlogram, and QC is the A x T matrix of the query's positions
%   (offset plus its position).
    if ~isempty(startV) || ~isempty(stopV) || ~isempty(stepV)
        error('windowedSimilarity:offsetsAndStart', ...
            ['start/stop/step lay out window centres; with offsets, give ' ...
             'the window centres as centres (or omit them to let the ' ...
             'window travel with the query).']);
    end
    if isempty(dropW), dropW = false; end
    if dropW
        error('windowedSimilarity:offsetsDropped', ...
            ['offsets translate the query along the window attribute, so ' ...
             'that attribute must be compared (dropWindowAttr = false).']);
    end
    if isempty(windowAttr), ax = A; else, ax = windowAttr; end
    if internal.axisIsRel(specs, isRel, ax)
        error('windowedSimilarity:offsetsRelative', ...
            ['attribute %d is relative: translating it leaves every ' ...
             'within-tuple difference unchanged, so there is nothing to ' ...
             'sweep. Give window positions as centres instead.'], ax);
    end
    off = double(offsets);
    qPos = localQueryPosition(pQuery, ax, locate);
    if isempty(centres)
        if ~isvector(off)
            error('windowedSimilarity:offsetsShape', ...
                'without centres, offsets must be a vector.');
        end
        centres = off(:).' + qPos;
        qc = [];
        return;
    end
    c = double(centres(:)).';
    Ac = numel(c);
    if isvector(off) && ~(Ac > 1 && size(off, 1) == Ac)
        qc = repmat(off(:).' + qPos, Ac, 1);
    elseif size(off, 1) == Ac
        qc = off + qPos;
    else
        error('windowedSimilarity:offsetsShape', ...
            ['with %d centres, offsets must be a vector (shared by every ' ...
             'centre) or have %d rows.'], Ac, Ac);
    end
    centres = c;
end


function [sweepOut, dropOut, translateOnly] = localOffsetsMulti(pQuery, specs, ...
        isRel, offsets, sweepMap, dropMap, locate, contextWindow)
%LOCALOFFSETSMULTI  Merge an offsets map {a, offsets; ...} into
%   sweep/drop. Each named attribute is translated and compared; it carries a
%   window only if contextWindow names it (the window then travels with the
%   query), otherwise it is translated with no window.
    [oAxes, oVals] = internal.parseMap(offsets);
    [sAxes, sVals] = internal.parseMap(sweepMap);
    [dAxes, dVals] = internal.parseMap(dropMap);
    [cwAxes, cwVals] = internal.parseMap(contextWindow);
    translateOnly = [];
    for k = 1:numel(oAxes)
        a = oAxes(k);
        if any(sAxes == a)
            error('windowedSimilarity:offsetsAndSweep', ...
                ['attribute %d is named in both offsets and sweep; an ' ...
                 'attribute is either translated (offsets) or only ' ...
                 'windowed (sweep).'], a);
        end
        di = find(dAxes == a, 1);
        if ~isempty(di) && dVals{di}
            error('windowedSimilarity:offsetsDropped', ...
                ['attribute %d is translated, so it is compared: it ' ...
                 'cannot be dropped.'], a);
        end
        if internal.axisIsRel(specs, isRel, a)
            error('windowedSimilarity:offsetsRelative', ...
                ['attribute %d is relative: translating it leaves every ' ...
                 'within-tuple difference unchanged, so there is nothing ' ...
                 'to sweep.'], a);
        end
        sAxes(end + 1) = a; %#ok<AGROW>
        sVals{end + 1} = double(oVals{k}(:)).' + localQueryPosition(pQuery, a, locate); %#ok<AGROW>
        if isempty(di)
            dAxes(end + 1) = a; %#ok<AGROW>
            dVals{end + 1} = false; %#ok<AGROW>
        end
        ci = find(cwAxes == a, 1);
        if isempty(ci) || isempty(cwVals{ci})
            translateOnly(end + 1) = a; %#ok<AGROW>
        end
    end
    [sAxes, ord] = sort(sAxes);
    sVals = sVals(ord);
    sweepOut = [num2cell(sAxes(:)), sVals(:)];
    dropOut = cell(numel(sAxes), 2);
    for k = 1:numel(sAxes)
        dropOut{k, 1} = sAxes(k);
        dropOut{k, 2} = dVals{find(dAxes == sAxes(k), 1)};
    end
end


function row = localSweepRow(pc, wc, sc, pq, wq, sq, sg, rr, rl, pr, pd, ...
        exchC, nested, offs, normalize)
%LOCALSWEEPROW  The query, translated by every offset, against a fixed
%   (already windowed) context, in one pass through sweepSimMaet. Returns
%   [] where no such route applies, so the caller can fall back to
%   comparing offset by offset. A density with a nested attribute takes
%   sweepSimMaet's contraction route, which contracts the nesting level by
%   level with the offsets as a batch dimension. Twin of the Python
%   _sweep_row.
    row = [];
    try
        if nested
            dc = buildMaet(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, ...
                'specs', sc, 'verbose', false);
            dq = buildMaet(pq, wq, 'sigma', sg, 'isPer', pr, 'period', pd, ...
                'specs', sq, 'verbose', false);
        else
            dc = buildMaet(pc, wc, sg, rr, rl, pr, pd, exchC{:}, 'verbose', false);
            dq = buildMaet(pq, wq, sg, rr, rl, pr, pd, exchC{:}, 'verbose', false);
        end
        v = sweepSimMaet(dc, dq, offs, 'normalize', normalize, 'verbose', false);
        v = double(v(:)).';
        if numel(v) == size(offs, 2) && all(isfinite(v))
            row = v;
        end
    catch
        row = [];
    end
end

function [out, mu, sv] = sweptSimilarity(varargin)
%SWEPTSIMILARITY  Compare a query with a context at each of a list of
%   sweep values (a pre-MAET cross-correlation).
%
%   [S, mu] = sweptSimilarity(pmContext, pmQuery, 'sweep', {a, values})
%   [S, mu] = sweptSimilarity(pmContext, pmQuery, 'sweep', a)
%   [S, mu, sv] = sweptSimilarity(...)
%
%   OVERVIEW. The query is compared with the context at each of a list of
%   values on an attribute a, the SWEEP VALUES, giving a profile S whose
%   peaks show where the query best matches the context. At each sweep
%   value the query is translated there (attribute translation, as by
%   translateAttributes), a window on the context is aligned there (event
%   weighting, as by weightEvents), or both. By default only the query is
%   translated, and it is compared with the whole context: the
%   cross-correlation of the query against the context. A window restricts
%   each comparison to a local region of the context; on its own, it
%   compares the query, as written, with each region in turn.
%
%   The similarities are computed in one of two ways. Where the context is
%   the same across the query's translations (always under 'query', and
%   across the query list of 'independent' at each window position),
%   sweepSimMaet computes them all in one pass. Where the window changes
%   with each sweep value ('both', 'window'), simMaet computes them one
%   sweep value at a time. Both give the same values, to numerical
%   precision: sweepSimMaet returns what simMaet would at each
%   translation, but much faster.
%
%   TWO ATTRIBUTES. The SWEPT ATTRIBUTE a is the one the sweep values lie
%   on: the query is translated along it, and a window is a function of
%   displacement along it. The TARGET ATTRIBUTE ('targetAttr', by default
%   the first attribute not dropped) is the one whose per-event weights a
%   window multiplies. They are usually different: a window over time
%   weights the pitch events.
%
%   THE RULE. At each sweep value s on the swept attribute:
%
%     - the query, where it is translated, has its reference value
%       queryRef at s: it is translated by mu = s - queryRef;
%     - a window, where there is one, has its reference value, delta = 0
%       (the midpoint of its symmetric shape), at s.
%
%   Nothing else places anything. For each swept attribute, 'align' says
%   which of the two are placed:
%
%     'query'       The query only (the default): attribute translation
%                   over the whole context. queryRef defaults to 0, so the
%                   sweep values are the offsets added to the query as
%                   written (transpositions, time shifts).
%     'both'        The query and a window, at the same s: a local
%                   comparison. queryRef defaults to the query's middle
%                   (the mean of its events' values on the swept
%                   attribute), so the window is aligned at the query's
%                   middle.
%     'window'      A window only; the query is left as written.
%     'independent' A window at each value of one list and the query at
%                   each value of another, in every combination (a
%                   correlogram). queryRef defaults to the query's middle.
%
%   A window h(delta) aligned at s weights each context event n on the
%   target attribute:
%
%       w'(n) = w(n) * h(p_a(n) - s),
%
%   where p_a(n) is event n's value on the swept attribute a (where the
%   event holds several values, 'locate' reduces them to one), so events
%   far from s are attenuated.
%
%   The window acts on the pre-MAET, before any density is built: p_a(n)
%   is the value the pre-MAET holds for event n (its onset time, say), and
%   only the weights change. Whether the swept attribute is then built in
%   absolute or relative mode, or dropped, matters only afterwards, when
%   the density is built from the weighted events. A relative time
%   attribute of bound events, for example, is still windowed by onset
%   time (each event's onsets reduced to one by 'locate'), and then
%   compared through its onsets measured from the first within each
%   event.
%
%   The second output, mu, is the translation applied to the query at each
%   sweep value, mu = s - queryRef: its shift from where it was written,
%   whatever queryRef is. Plot against mu to read a windowed sweep as
%   offsets. With query and context written from a common origin (the
%   query at the time it was taken from, say), mu keeps its meaning
%   through preprocessing that keeps the values, such as differencing, so
%   profiles with and without it share one axis.
%
%   The query itself is not windowed here. To weight the query's own
%   events by a window, apply weightEvents to the query before the call.
%
%   CHOOSING A ROLE.
%
%     'query'       The canonical sweep: where in the context, or at which
%                   transposition, the query best matches the context as a
%                   whole. Only the kernel width sigma limits which context
%                   events count; all sweep values are computed in one pass.
%     'both'        A local 'query': the window fixes the region of the
%                   context that counts around the query. Under
%                   'normalize', 'cosine' unmatched material inside the
%                   window lowers the score and material outside it is
%                   ignored; as the window widens, 'both' becomes 'query'.
%     'window'      The query is not translated: the window steps through
%                   the context and the query, as written, is compared with
%                   each region in turn. What this measures depends on the
%                   treatment of the swept attribute (below).
%     'independent' Each window position gives a whole profile: for a
%                   best placement that changes across the context, such
%                   as a lag between two parts that drifts over time.
%
%   TREATMENT OF THE SWEPT ATTRIBUTE. Whether the swept attribute is
%   absolute or relative (its specs), and whether it is dropped ('drop'),
%   decides what it contributes to each comparison, whatever the role:
%
%     absolute      Compared by position, so translating the query along
%                   it changes where the query matches, and all four roles
%                   apply. Under 'window' the query is compared in place:
%                   the profile shows where in the context its match with
%                   the query as written comes from (two parts of a piece
%                   on a shared time axis, say, whose similarity the window
%                   resolves in time; under the default normalization,
%                   windows that tile the context give contributions that
%                   sum to the whole-piece similarity). To find where the
%                   query occurs, translate it ('query' or 'both').
%     relative      Compared only up to a common translation of each
%                   tuple, that is, through its values relative to the
%                   lowest (a chord's intervals above its bass, or a bound
%                   event's onsets measured from its first), so the query's
%                   internal spacing must match but its position does not
%                   matter. Translation leaves these relative values
%                   unchanged, so only 'window' applies (it gives what
%                   'both' would). The events are still windowed by the
%                   values the pre-MAET holds (above).
%     dropped       Marginalized after the window has weighted the events
%                   ('drop'), so not compared at all: the query is compared
%                   with what the region contains, not where in it (a local
%                   key, say). Translation has nothing to act on, so only
%                   'window' applies.
%
%   Event differencing (differenceEvents) is not a further treatment but a
%   change of values: the attribute then holds first differences between
%   successive events (inter-onset intervals, pitch steps), and is
%   absolute or relative like any other. As the swept attribute, its
%   windows therefore select by interval size, and translation adds the
%   same amount to every interval (on logarithmically rescaled inter-onset
%   intervals, a tempo change). Usually the attribute differenced (pitch,
%   say) is not the one swept (time), which differencing passes through
%   unchanged at order 0, keeping each event's onset.
%
%   WHEN A WINDOW ON THE CONTEXT IS NEEDED. Translation already localizes
%   on a compared absolute attribute: the kernel lets the query match only
%   material near where it is translated. A window on the context is
%   indispensable where translation cannot localize: on a dropped or
%   relative swept attribute, alone or alongside translation on another
%   attribute (each bar windowed on time, time dropped, and the query
%   translated in pitch: the bar and the transposition of each statement
%   at once). On a translated attribute ('both', 'independent') it does
%   nearly what weighting the query would (weightEvents, aligned at the
%   matching point of the query, then translated), the two differing only
%   in whether a near miss is weighted where the context's event lies or
%   where the query's does; what it adds there is the 'cosine'
%   denominator, the norm of what the window keeps. To ask which part of
%   the query matches, weight the query.
%
%   READING THE OUTPUT. S has one dimension per sweep list, in attribute
%   order; 'independent' contributes two, the window's first. A single
%   list gives a 1 x n row. A sweep value is where the query's reference
%   lands (under 'window', where the window is aligned). mu is a 1 x A
%   cell: mu{a} holds the translations along attribute a, the same size as
%   its (query) sweep list, and is empty where the query is not translated.
%   sv, the third output, is a 1 x A cell of the sweep values themselves,
%   listed or generated: sv{a} holds attribute a's, empty where a is not
%   swept, and under 'independent' the pair {windowValues, queryValues}.
%   They are the axes of S. Under 'query' they equal mu (queryRef is 0);
%   under 'both' they are where the query's middle and the window lie, and
%   under 'window' there is no mu at all.
%
%   INPUT FORMS, in the order to reach for them:
%
%     S = sweptSimilarity(pmContext, pmQuery, ...)
%
%   with two whole pre-MAETs (the canonical entry). The geometry is taken
%   from their specs. Any of the six per-attribute parameters ('sigma',
%   'isPer', 'period', 'r', 'rel', 'exch') may be given alongside to
%   override it, as at buildMaet, either in full or selectively as a
%   1 x A cell whose empty entries keep the spec's value. The two
%   pre-MAETs describe one comparison, so they must agree on 'r', 'rel',
%   'exch', and the nesting; 'sigma', 'isPer', and 'period' may differ,
%   and the context's are used.
%
%     S = sweptSimilarity(pContext, wContext, pQuery, wQuery, ...
%                            sigma, r, isRel, isPer, period, ...)
%
%   the raw positional form, with each operand's per-attribute values and
%   weights and the shared geometry written out as for simMaet.
%
%   NAME-VALUE OPTIONS (per-attribute maps are N x 2 cells {a, value; ...})
%     'sweep'        {a, values; ...}: the sweep values of attribute a. A
%                    bare attribute index a, or a vector of them, asks for
%                    default sweep values on each (see 'start', 'stop',
%                    'step'). For 'independent',
%                    {a, {windowValues, queryValues}}: the window's list
%                    may be [] when start / stop / step generate it, and
%                    the query's may be a matrix with one row per window
%                    value, when the query's placements depend on where
%                    the window is (a lag measured from each window value,
%                    say).
%     'start', 'stop', 'step'
%                    {a, value; ...}: generate attribute a's sweep values from
%                    start to stop in steps of step, in place of listing them;
%                    each overrides one default. A bare number applies to the
%                    swept attribute where 'sweep' names one ('sweep', 2,
%                    'step', 0.5). The defaults depend on the role. Where the
%                    sweep values translate the query ('query', 'both'), start
%                    and stop cover every placement at which the query overlaps
%                    the context (from its highest value on the context's
%                    lowest to its lowest on the context's highest), or one
%                    period on a periodic attribute. step is then at most h,
%                    half the standard deviation of the profile's peaks:
%                    translating the query moves all D coordinates of the
%                    attribute's tuple alike, so the peaks have standard
%                    deviation sigma * sqrt(2 / D), narrower the larger the
%                    tuple (D = r, or the product of a nested attribute's
%                    per-level r; for a kernel covariance Sigma,
%                    sqrt(2 / (1' * inv(Sigma) * 1))), window or no window. The
%                    step is also chosen so that every exact match lies on the
%                    grid. Where the context's values are whole multiples of a
%                    spacing g apart, and so are the query's (and g divides the
%                    period, on a periodic attribute), every exact match is at
%                    the lowest offset plus a whole multiple of g, so the step
%                    is g / k, with k the smallest whole number that brings it
%                    to h or below: onsets on whole beats with h = 0.15 step at
%                    1/7, not at 0.15, which would miss the whole-beat offsets.
%                    Where g is below h it is the step itself, if at least
%                    h / 4. Otherwise (values on no such lattice, or on one too
%                    fine) the step is h, which leaves every peak within a
%                    quarter of its standard deviation of a grid point, at
%                    about 97% of its height or more. Where they place a window
%                    only ('window', and the
%                    window's list of 'independent'), start and stop are the
%                    lowest and highest of the context's values on the
%                    attribute, and step is half the window's sd, since the
%                    profile changes on the scale of the window (a function
%                    handle has no width, so give step). For largely
%                    separate windows, as when the profile's values are to
%                    be used as data, give step as half the window's width
%                    (sqrt(3) sd), so that neighbouring windows overlap by
%                    half. A pure rectangle ('rect', or shape 1) makes the
%                    profile piecewise constant: the windowed context changes
%                    only where an event enters or leaves it, at each event's
%                    value plus or minus half the width. Without a given
%                    step, its default sweep values are these pieces, each
%                    sampled just inside both its ends, so that every value
%                    is the profile's value at its sweep value and a line
%                    plot draws the steps exactly. The query's list of
%                    'independent' is always given explicitly.
%     'align'        {a, 'query' | 'both' | 'window' | 'independent'; ...}:
%                    what is placed at attribute a's sweep values (THE
%                    RULE). Default 'query' for every swept attribute.
%     'window'       {a, {shape, width}; ...}, {a, {shape, width, edges}; ...},
%                    {a, struct('shape', .., 'width' | 'sd' | 'decayRate', ..,
%                    'edges', ..); ...}, or {a, f; ...}: the window h on
%                    attribute a, any profile of weightEvents, which evaluates
%                    it. shape is 'rect', 'gaussian', or a number in [0, 1]
%                    blending the two (0 Gaussian, 1 rectangle); width is the
%                    full width of the rectangle, and a Gaussian of the same
%                    width has standard deviation width / (2 sqrt(3)), which
%                    'sd' may give instead. 'exponential' decays on both sides
%                    of the window's reference value, and 'exponentialBefore' /
%                    'exponentialAfter' on one side only (zero on the other),
%                    scaled by sd or decayRate; a function handle f takes the
%                    displacement p_a(n) - s and returns the factors. The
%                    serial-position profiles of weightEvents, anchored at the
%                    first and last events rather than at the sweep value, are
%                    refused. On a periodic attribute the displacement wraps.
%                    edges, for rectangles: 'halfOpen' (the default for a given
%                    width: the lower edge included, the upper not, so that
%                    windows a width apart share no event, for tiling a
%                    context) or 'closed' (both edges, for holding a query).
%                    Required for 'window' and 'independent', where the width
%                    is the scale of the local region and nothing in the data
%                    can supply it; not allowed for 'query'. For 'both' it may
%                    be left out, or given with width []: the window is then
%                    the smallest one that, placed by THE RULE, holds the
%                    query, with a closed rectangle unless another shape or
%                    edges is given, so an exact match scores 1. A window given
%                    for 'both' that leaves out some of the query's own events
%                    draws a warning, since the query can then never be matched
%                    in full.
%     'drop'         Vector of attributes marginalized after the window has
%                    weighted the events. Only attributes whose align is
%                    'window'.
%     'queryRef'     {a, value; ...}: the query's reference value on
%                    attribute a, the point of the query placed at each
%                    sweep value; only where the query is translated.
%                    Default 0 for 'query' and the query's middle for 'both'
%                    and 'independent'. Under 'query' it only relabels the
%                    output (a sweep value s under queryRef r is the same
%                    comparison as s - r + r2 under r2); under 'both' it
%                    also decides which point of the query lies at the
%                    window's centre. Useful values:
%                      - 0: sweep values are the offsets mu added to the
%                        query as written.
%                      - The query's middle: sweep values are where the
%                        middle lands, and under 'both' the window is
%                        aligned at the query's middle.
%                      - A particular point of the query, such as its first
%                        onset or its root: sweep values are where that
%                        point lands, the time at which a match starts or
%                        the key of a transposition (under 'both', the
%                        window's centre then sits at that point).
%     'locate'       Which single value p_a(n) stands for an event that holds
%                    several values on the swept attribute (the onsets of a
%                    bound super-event, say), both where the window is
%                    evaluated and in the query's middle: 'centroid' (their
%                    mean; the default), 'start' (the first), 'end' (the
%                    last), 'mid' (the midpoint of the first and last), a
%                    function handle taking the K x N value matrix and
%                    returning N values, or a map {a, rule; ...}, an
%                    attribute it does not name taking 'centroid'. It has
%                    no effect where each event holds one value.
%     'targetAttr'   The attribute whose weights the windows multiply
%                    (default: the first attribute not dropped).
%     'normalize'    As at simMaet, with the windowed context as the first
%                    operand and the query as the second. 'oneSidedDenom'
%                    (default) divides by the query's self inner product,
%                    so a windowed context identical to the query scores 1.
%                    'cosine' gives the shape-only cosine similarity,
%                    bounded in [-1, 1]; 'none' the bare inner product.
%     'specs'        Nested geometry from bindEvents (raw form).
%     'isExch'       Per-attribute exchangeability for the raw form ([]
%                    keeps the unordered default). Needed, in particular,
%                    for ordered attributes carrying a matrix-valued kernel
%                    covariance (see kernelCov). Not allowed together with
%                    'specs', whose nesting gives exchangeability level by
%                    level.
%     'verbose'      Accepted for consistency with the other entry points;
%                    the inner comparisons pass 'verbose', false. The
%                    dispatcher's one-line announcement of the route it
%                    chose follows mptDefaults('showHints') instead.
%
%   The window-factor and comparison-kernel truncation both follow the
%   global mptDefaults setting.
%
%   EXAMPLES. Where does E-G occur in the melody C D E G C E G (one note
%   per time unit)? Translating the query in time, with sweep values that
%   are offsets from the query as written (it starts at time 0, so an
%   offset is the time at which it starts):
%
%     cents = @(m) transformAttributes(m, [], {'midi', 'cents'});
%     ctx = {cents([60 62 64 67 60 64 67]), 0:6};
%     qry = {cents([64 67]), [0 1]};
%     geom = {[10 0.2], [1 1], [false false], [true false], [1200 0]};
%     s = 0:5;
%     S = sweptSimilarity(ctx, [], qry, [], geom{:}, 'sweep', {2, s});
%     s(S > 0.99)                    % 2  5
%
%   The same search, local: query and window aligned together ('both';
%   the window by default the smallest closed rectangle that holds the
%   query), stepped across the melody and read against mu:
%
%     [S, mu] = sweptSimilarity(ctx, [], qry, [], geom{:}, ...
%         'step', {2, 0.5}, 'align', {2, 'both'});
%     mu{2}(S > 0.99)                % 2  5
%
%   See also PACKPREMAET, SWEPTENTROPY, WEIGHTEVENTS, TRANSLATEATTRIBUTES,
%            SIMMAET, SWEEPSIMMAET.

% Top-level call guard: dispatch throttle + kernelChunkBytes pin, so the
% per-value inner calls announce once per sweep. See internal.callGuard.
guard = internal.callGuard(); %#ok<NASGU>

varargin = internal.sweptPreMaetArgs(varargin, 'sweptSimilarity', 2);
[out, plan] = localSweptSimilarity(varargin{:});
if nargout > 1, mu = localOffsets(plan, numel(varargin{1})); end
if nargout > 2, sv = internal.sweepValues(plan, numel(varargin{1})); end
end


function [out, plan] = localSweptSimilarity(pContext, wContext, pQuery, wQuery, ...
        sigma, r, isRel, isPer, period, nv)
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
    nv.sweep = []
    nv.start = []
    nv.stop = []
    nv.step = []
    nv.align = []
    nv.window = []
    nv.drop = []
    nv.queryRef = []
    nv.locate = 'centroid'
    nv.targetAttr = []
    nv.normalize (1,:) char = 'oneSidedDenom'
    nv.specs = []
    nv.querySpecs = []
    nv.isExch = []
    nv.verbose (1,1) logical = false
end

if ~isempty(nv.isExch) && ~isempty(nv.specs)
    error('sweptSimilarity:isExchVsSpecs', ...
        ['isExch applies to the flat per-attribute surface; nested ' ...
         'geometry carries its per-level exch inside specs. Pass one ' ...
         'or the other.']);
end
querySpecs = nv.querySpecs;
if isempty(querySpecs), querySpecs = nv.specs; end
plan = internal.sweptPlan(pContext, pQuery, nv.specs, isRel, nv, ...
    'sweptSimilarity', struct('sigma', {sigma}, 'isPer', {isPer}, ...
    'period', {period}, 'r', {r}));
out = localRun(pContext, wContext, pQuery, wQuery, sigma, r, isRel, isPer, ...
    period, nv.isExch, nv.specs, querySpecs, plan, nv.locate, ...
    nv.targetAttr, nv.normalize);
end


function out = localRun(pContext, wContext, pQuery, wQuery, sigma, r, isRel, ...
        isPer, period, isExch, specs, querySpecs, plan, locate, targetAttr, ...
        normalize)
%LOCALRUN  Twin of the Python _run_similarity.
    A = numel(pContext);
    dropAxes = plan.dropAxes;
    keep = setdiff(1:A, dropAxes);
    if isempty(targetAttr), target = keep(1); else, target = targetAttr; end
    if any(target == dropAxes)
        error('sweptSimilarity:targetDropped', ...
            ['targetAttr %d is a dropped attribute: its weights are removed ' ...
             'before the build, so the window factors would be lost. Choose ' ...
             'a compared attribute.'], target);
    end
    nested = ~isempty(specs);
    [sg, rr, rl, pr, pd] = internal.subGeom(sigma, r, isRel, isPer, period, keep);
    exchC = internal.subExchArgs(isExch, keep);
    dims = plan.dims;
    D = numel(dims);
    sizes = arrayfun(@(d) size(d.vals, 2), dims);
    cPos = find(strcmp({dims.kind}, 'ctx'));
    qPos = find(strcmp({dims.kind}, 'query'));
    if D == 1, out = zeros(1, sizes(1)); else, out = zeros(sizes); end
    strides = cumprod([1, sizes(1:end-1)]);
    cSizes = sizes(cPos); qSizes = sizes(qPos);
    nC = prod(cSizes); nQ = prod(qSizes);
    cAxes = [dims(cPos).a];
    for ci = 1:nC
        cSubs = internal.lin2sub(cSizes, ci);
        full = ones(1, D);
        full(cPos) = cSubs;
        at = zeros(1, numel(cPos));
        for k = 1:numel(cPos), at(k) = dims(cPos(k)).vals(cSubs(k)); end
        % Every window is aligned at its sweep value. For 'both', the query
        % is translated so that its reference lands at the same value.
        shifts = cell(1, A);
        for k = 1:numel(cPos)
            a = cAxes(k);
            if plan.withQuery(a)
                shifts{a} = at(k) - plan.qRef(a);
            end
        end
        [pc, wc, sc] = internal.applyWindows(pContext, wContext, specs, ...
            cAxes, at, plan.win(cAxes), localLocates(locate, cAxes), target);
        [pc, wc, sc] = internal.dropAxes(pc, wc, sc, dropAxes, A);
        [pqB, wqB, sqB] = localTranslate(pQuery, wQuery, querySpecs, shifts);
        if isempty(qPos)
            [pq, wq, sq] = internal.dropAxes(pqB, wqB, sqB, dropAxes, A);
            out(1 + sum((full - 1) .* strides)) = localCompare(pc, wc, sc, ...
                pq, wq, sq, sg, rr, rl, pr, pd, exchC, nested, normalize);
            continue;
        end
        % The windowed context is fixed across the query's own translations, so
        % they are computed together where sweepSimMaet applies.
        offs = zeros(numel(keep), nQ);
        qSubsAll = zeros(nQ, numel(qPos));
        for m = 1:nQ
            qSubsAll(m, :) = internal.lin2sub(qSizes, m);
            for k = 1:numel(qPos)
                d = dims(qPos(k));
                offs(keep == d.a, m) = localQValue(d, full, qSubsAll(m, k)) ...
                    - plan.qRef(d.a);
            end
        end
        [pq0, wq0, sq0] = internal.dropAxes(pqB, wqB, sqB, dropAxes, A);
        row = localSweepRow(pc, wc, sc, pq0, wq0, sq0, sg, rr, rl, pr, pd, ...
            exchC, nested, offs, normalize);
        for m = 1:nQ
            full(qPos) = qSubsAll(m, :);
            li = 1 + sum((full - 1) .* strides);
            if ~isempty(row)
                out(li) = row(m);
                continue;
            end
            qShift = cell(1, A);
            for k = 1:numel(qPos)
                d = dims(qPos(k));
                qShift{d.a} = localQValue(d, full, qSubsAll(m, k)) - plan.qRef(d.a);
            end
            [pqT, wqT, sqT] = localTranslate(pqB, wqB, sqB, qShift);
            [pq, wq, sq] = internal.dropAxes(pqT, wqT, sqT, dropAxes, A);
            out(li) = localCompare(pc, wc, sc, pq, wq, sq, sg, rr, rl, pr, ...
                pd, exchC, nested, normalize);
        end
    end
end


function mu = localOffsets(plan, A)
%LOCALOFFSETS  The translation applied to the query at each of its sweep
%   values, mu = s - queryRef: a 1 x A cell, empty where the query is not
%   translated. Twin of the Python _offsets.
    mu = cell(1, A);
    for d = plan.dims
        if plan.hasQRef(d.a) && (strcmp(d.kind, 'query') || plan.withQuery(d.a))
            mu{d.a} = d.vals - plan.qRef(d.a);
        end
    end
end


function v = localQValue(d, full, j)
%LOCALQVALUE  A query sweep value: shared by every window value, or one row
%   per value of the paired window dimension.
    if size(d.vals, 1) > 1
        v = d.vals(full(d.pair), j);
    else
        v = d.vals(j);
    end
end


function locs = localLocates(locate, axes)
    locs = cell(1, numel(axes));
    for k = 1:numel(axes), locs{k} = internal.axisLocate(locate, axes(k)); end
end


function [p, w, sp] = localTranslate(p, w, sp, shifts)
%LOCALTRANSLATE  The query with attribute a translated by shifts{a}.
    if all(cellfun(@isempty, shifts)), return; end
    [p, w, sp] = unpackPreMaet(translateAttributes(p, w, shifts, 'specs', sp));
end


function s = localCompare(pc, wc, sc, pq, wq, sq, sg, rr, rl, pr, pd, exchC, ...
        nested, normalize)
    if nested
        dc = buildMaet(pc, wc, 'sigma', sg, 'isPer', pr, 'period', pd, ...
            'specs', sc, 'verbose', false);
        dq = buildMaet(pq, wq, 'sigma', sg, 'isPer', pr, 'period', pd, ...
            'specs', sq, 'verbose', false);
        s = simMaet(dc, dq, 'normalize', normalize, 'verbose', false);
    else
        s = simMaet(pc, wc, pq, wq, sg, rr, rl, pr, pd, exchC{:}, ...
            'normalize', normalize, 'verbose', false);
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

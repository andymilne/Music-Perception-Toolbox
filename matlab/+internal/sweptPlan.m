function plan = sweptPlan(pContext, pQuery, specs, isRel, nv, fname, geom)
%SWEPTPLAN  Validate the sweep arguments of sweptSimilarity /
%   sweptEntropy and return the sweep plan. Twin of the Python
%   _build_plan.
%
%   plan = internal.sweptPlan(pContext, pQuery, specs, isRel, nv, fname, geom)
%   reads nv.sweep, nv.start, nv.stop, nv.step, nv.align, nv.window,
%   nv.drop, nv.queryRef, and nv.locate. pQuery is [] for
%   sweptEntropy, whose sweep values always align the window. geom
%   (optional) is a struct with fields sigma, isPer, period, and r, from
%   which a translation sweep takes its default range and step. nv.sweep may be a
%   bare attribute index, or a vector of them, for default sweep values.
%
%   One rule places everything: at each sweep value s, a window has its
%   reference (delta = 0) at s, and a translated query has its reference
%   queryRef at s.
%
%   plan.dims      - 1 x D struct array, one per output dimension in order:
%                    kind ('ctx': a window is aligned at each value, and
%                    for 'both' the query's reference is placed there too;
%                    'query': only the query is translated), a
%                    (attribute), vals
%                    (1 x n, or a matrix with one row per value of the
%                    paired window dimension), pair (index of that window
%                    dimension, or 0).
%   plan.dropAxes  - attributes marginalized after weighting.
%   plan.win, plan.hasWin, plan.withQuery
%                  - 1 x A: the context window on each attribute where a
%                    window is aligned (a struct with fields prof, closed,
%                    isPer, and period, evaluated by internal.windowFactor),
%                    and whether the query is translated with it.
%   plan.qRef, plan.hasQRef
%                  - 1 x A: queryRef where the query is translated.

A = numel(pContext);
hasQuery = ~isempty(pQuery);
if nargin < 7, geom = []; end
% A bare attribute index, or a vector of them, asks for default sweep values
% on each.
if isnumeric(nv.sweep) && ~isempty(nv.sweep)
    nv.sweep = [num2cell(nv.sweep(:)), repmat({[]}, numel(nv.sweep), 1)];
end
[swA, swV] = localMap(nv.sweep, A, 'sweep', fname);
% A bare number for start, stop, or step applies to the one swept
% attribute: 'sweep', a, 'step', 0.5.
nv.start = localBareGenerator(nv.start, 'start', swA, fname);
nv.stop  = localBareGenerator(nv.stop, 'stop', swA, fname);
nv.step  = localBareGenerator(nv.step, 'step', swA, fname);
[stA, stV] = localMap(nv.start, A, 'start', fname);
[spA, spV] = localMap(nv.stop, A, 'stop', fname);
[sA, sV]   = localMap(nv.step, A, 'step', fname);
[wA, wV]   = localMap(nv.window, A, 'window', fname);
dropAxes = localList(nv.drop, A, fname);
swept = unique([swA, stA, spA, sA]);
if hasQuery
    [mA, mV] = localMap(nv.align, A, 'align', fname);
    [rA, rV] = localMap(nv.queryRef, A, 'queryRef', fname);
    % Attribute translation over the whole context is the default role.
    extra = setdiff(swept, mA);
    mA = [mA, extra];
    mV = [mV, repmat({'query'}, 1, numel(extra))];
else
    mA = unique([swept, wA]);
    mV = repmat({'window'}, 1, numel(mA));
    rA = []; rV = {};
end
valid = {'window', 'query', 'both', 'independent'};
for k = 1:numel(mA)
    if ~(ischar(mV{k}) || (isstring(mV{k}) && isscalar(mV{k}))) || ...
            ~any(strcmp(char(mV{k}), valid))
        error([fname ':badAlign'], ...
            ['align for attribute %d must be one of ''window'', ' ...
             '''query'', ''both'', ''independent''.'], mA(k));
    end
    mV{k} = char(mV{k});
end
orphan = setdiff(mA, swept);
if ~isempty(orphan)
    error([fname ':alignWithoutSweep'], ...
        ['attribute %d has an align entry but no sweep values; give them ' ...
         'with ''sweep'', {%d, values} (or start / stop / step).'], ...
        orphan(1), orphan(1));
end
if isempty(mA)
    error([fname ':noSweep'], ...
        ['name at least one attribute to sweep, with sweep (or start / ' ...
         'stop / step).']);
end

plan.win = cell(1, A);
plan.hasWin = false(1, A); plan.withQuery = false(1, A);
plan.qRef = nan(1, A); plan.hasQRef = false(1, A);
dims = struct('kind', {}, 'a', {}, 'vals', {}, 'pair', {});

[mA, ord] = sort(mA);
mV = mV(ord);
% The query's reference, wherever the query is translated: the point of the
% query placed at each sweep value. By default 0 where there is no window
% ('query': sweep values are then the offsets added to the query as
% written), and the query's middle where there is one ('both',
% 'independent': window and query sweep values then both refer to centres,
% so under 'both' the window is centred on the query).
for k = 1:numel(mA)
    if strcmp(mV{k}, 'window'), continue; end
    plan.qRef(mA(k)) = localQueryRef(pQuery, mA(k), nv.locate, rA, rV, ...
        ~strcmp(mV{k}, 'query'));
    plan.hasQRef(mA(k)) = true;
end

for k = 1:numel(mA)
    a = mA(k); m = mV{k};
    gen = any(stA == a) || any(spA == a) || any(sA == a);
    rel = internal.axisIsRel(specs, isRel, a);
    wi = find(wA == a, 1);
    % --- the window ---
    if strcmp(m, 'query')
        if ~isempty(wi)
            error([fname ':windowForQuery'], ...
                ['attribute %d has a window, but its align is ''query'' ' ...
                 '(the default: translation over the whole context), which ' ...
                 'has none. Say where the window goes: ''align'', {%d, ' ...
                 '''both''} (window and query at each sweep value), ' ...
                 '''window'' (the window only), or ''independent''.'], a, a);
        end
        win = [];
    elseif strcmp(m, 'window')
        if isempty(wi) && hasQuery
            error([fname ':noWindow'], ...
                ['attribute %d: align ''window'' aligns a window, so ' ...
                 'give its shape and width: ''window'', {%d, {shape, ' ...
                 'width}}.'], a, a);
        elseif isempty(wi)
            error([fname ':noWindow'], ...
                ['attribute %d is swept, so give its window''s shape and ' ...
                 'width: ''window'', {%d, {shape, width}}.'], a, a);
        end
        win = localWindow(wV{wi}, a, 'window', fname, []);
    elseif strcmp(m, 'independent')
        if isempty(wi)
            error([fname ':noWindow'], ...
                ['attribute %d: align ''independent'' aligns a window apart ' ...
                 'from the query, so give its shape and width: ''window'', ' ...
                 '{%d, {shape, width}}.'], a, a);
        end
        win = localWindow(wV{wi}, a, 'window', fname, []);
    else
        % 'both': window and query reference share each sweep value. The
        % window defaults to the smallest closed rectangle that, so placed,
        % holds the query.
        if isempty(wi), spec = []; else, spec = wV{wi}; end
        win = localWindow(spec, a, 'window', fname, ...
            @() localHoldingWidth(pQuery, a, nv.locate, plan.qRef(a), fname));
        localWarnIfQueryCut(pQuery, a, nv.locate, plan.qRef(a), win, fname);
    end
    if ~isempty(win)
        % On a periodic attribute the window's displacement wraps, as
        % weightEvents wraps it.
        [perW, pdW] = localPeriod(geom, a);
        if perW && pdW > 0
            win.isPer = true; win.period = pdW;
        end
    end
    % --- what may be translated, given the attribute's geometry ---
    if ~strcmp(m, 'window')
        if rel
            error([fname ':translateRelative'], ...
                ['attribute %d is relative: translating the query changes ' ...
                 'none of its within-tuple differences, so align ' ...
                 '''%s'' does nothing there. Use align ''window''.'], a, m);
        end
        if any(dropAxes == a)
            error([fname ':dropTranslated'], ...
                ['attribute %d: align ''%s'' translates the query along it, ' ...
                 'so it is compared and cannot be dropped.'], a, m);
        end
    end
    % --- the sweep values ---
    if isempty(win) || ~isfinite(win.prof.sd)
        defStep = [];
    else
        % Half the window's sd: the profile of a window-only sweep changes
        % on the scale of the window, as each event's weight follows it, so
        % this leaves every feature within a quarter-sd of a grid point (as
        % half the peaks' sd does for a translation sweep).
        defStep = win.prof.sd / 2;
    end
    si = find(swA == a, 1);
    listed = ~isempty(si) && ~isempty(swV{si});
    defRange = []; openStop = false;
    if any(strcmp(m, {'query', 'both'}))
        % Where the sweep values translate the query, every placement at
        % which it overlaps the context, stepped at no more than half the
        % standard deviation of the profile's peaks, sigma * sqrt(2 / D) for
        % a tuple of D coordinates (window or no window), on the values'
        % lattice where they lie on one, so that every exact match is on
        % the grid.
        [per, pd] = localPeriod(geom, a);
        kw = localPeakWidth(geom, specs, a);
        if ~isempty(kw)
            if ~per, pd = 0; end
            defStep = localLatticeStep(pContext, pQuery, a, kw / 2, pd);
        elseif strcmp(m, 'query')
            defStep = [];
        end
        if ~listed
            [defRange, openStop] = localTranslationRange(pContext, pQuery, ...
                a, plan.qRef(a), geom, fname);
            if openStop && ~isempty(defStep) && isempty(localGet(sA, sV, a))
                % One period, on the grid through the lowest offset, so the
                % lattice step lands on every exact match.
                c = double(pContext{a}(:)); c = c(isfinite(c));
                q = double(pQuery{a}(:)); q = q(isfinite(q));
                sh = mod(min(c) - max(q), defStep);
                if defStep - sh < 1e-9 * defStep, sh = 0; end
                defRange = defRange + sh;
            end
        end
    end
    if strcmp(m, 'independent')
        if isempty(si) || ~iscell(swV{si}) || numel(swV{si}) ~= 2
            error([fname ':independentPair'], ...
                ['attribute %d: align ''independent'' takes two ' ...
                 'lists, ''sweep'', {%d, {windowValues, queryValues}} (the ' ...
                 'window''s may be [] when start / stop / step generate it).'], ...
                a, a);
        end
        wVals = swV{si}{1}; qVals = swV{si}{2};
        if isempty(qVals)
            error([fname ':independentPair'], ...
                ['attribute %d: the query''s sweep values must be given ' ...
                 'explicitly in ''sweep'', {%d, {windowValues, queryValues}}.'], ...
                a, a);
        end
        if isempty(wVals)
            wVals = localWindowDefault(pContext, a, stA, stV, spA, spV, ...
                sA, sV, defStep, fname, win, nv.locate);
        elseif gen
            error([fname ':sweepAndGenerator'], ...
                ['attribute %d: give the window''s sweep values either in ' ...
                 'sweep or by start / stop / step, not both.'], a);
        end
        wVals = localValues(wVals, a, fname);
        qVals = double(qVals);
        % A row vector is shared by every window value; a matrix (or a
        % column with one entry per window value) holds one row per window
        % value, the query's placements depending on where the window is.
        perRow = ~isvector(qVals) || (numel(wVals) > 1 && ...
            size(qVals, 1) == numel(wVals) && size(qVals, 2) == 1);
        if perRow
            if size(qVals, 1) ~= numel(wVals)
                error([fname ':independentRows'], ...
                    ['attribute %d: a matrix of query values needs one row ' ...
                     'per window value (%d); got %d x %d.'], a, ...
                    numel(wVals), size(qVals, 1), size(qVals, 2));
            end
            if ~all(isfinite(qVals(:)))
                error([fname ':badValues'], ...
                    'sweep for attribute %d holds a non-finite value.', a);
            end
        else
            qVals = localValues(qVals, a, fname);
        end
        dims(end + 1) = struct('kind', 'ctx', 'a', a, 'vals', wVals, 'pair', 0); %#ok<AGROW>
        dims(end + 1) = struct('kind', 'query', 'a', a, 'vals', qVals, ...
            'pair', numel(dims)); %#ok<AGROW>
    else
        if listed && gen
            error([fname ':sweepAndGenerator'], ...
                ['attribute %d: give its sweep values either in sweep or by ' ...
                 'start / stop / step, not both.'], a);
        end
        if listed
            vals = localValues(swV{si}, a, fname);
        elseif strcmp(m, 'window')
            vals = localWindowDefault(pContext, a, stA, stV, spA, spV, ...
                sA, sV, defStep, fname, win, nv.locate);
        else
            vals = localGenerate(pContext, a, stA, stV, spA, spV, sA, sV, ...
                defStep, fname, defRange, openStop);
        end
        if strcmp(m, 'query'), kind = 'query'; else, kind = 'ctx'; end
        dims(end + 1) = struct('kind', kind, 'a', a, 'vals', vals, 'pair', 0); %#ok<AGROW>
    end
    if ~isempty(win)
        plan.hasWin(a) = true;
        plan.win{a} = win;
        plan.withQuery(a) = strcmp(m, 'both');
    end
end

bad = dropAxes(~ismember(dropAxes, mA(strcmp(mV, 'window'))));
if ~isempty(bad)
    error([fname ':dropNotWindow'], ...
        ['drop names attribute %d, which is not a window attribute swept ' ...
         'alone (align ''window''). Dropping marginalizes a window ' ...
         'attribute after the window has weighted the events; to leave an ' ...
         'attribute out of the comparison altogether, leave it out of the ' ...
         'pre-MAETs.'], bad(1));
end
if numel(dropAxes) >= A
    error([fname ':dropAll'], ...
        'every attribute is dropped; nothing is left to compare or measure.');
end

for k = 1:numel(rA)
    if ~plan.hasQRef(rA(k))
        error([fname ':queryRefUnused'], ...
            ['queryRef for attribute %d: the query is not translated along ' ...
             'this attribute (align ''window''), so it has no reference ' ...
             'there to place.'], rA(k));
    end
end
plan.dims = dims;
plan.dropAxes = sort(dropAxes);
end


function [keys, vals] = localMap(m, A, name, fname)
%LOCALMAP  A per-attribute map {a, value; ...}, keys checked against A.
    if isempty(m), keys = []; vals = {}; return; end
    if ~iscell(m) || size(m, 2) ~= 2
        error([fname ':badMap'], ...
            '%s must be an N-by-2 cell {a, value; ...}, a an attribute index.', ...
            name);
    end
    keys = zeros(1, size(m, 1));
    for i = 1:size(m, 1)
        a = m{i, 1};
        if ~(isnumeric(a) && isscalar(a) && a == round(a) && a >= 1 && a <= A)
            error([fname ':badMap'], ...
                '%s: keys are attribute indices 1..%d.', name, A);
        end
        keys(i) = a;
    end
    if numel(unique(keys)) < numel(keys)
        error([fname ':badMap'], '%s names an attribute twice.', name);
    end
    vals = m(:, 2).';
end


function d = localList(v, A, fname)
    d = double(v(:)).';
    if any(d ~= round(d) | d < 1 | d > A)
        error([fname ':badDrop'], 'drop lists attribute indices 1..%d.', A);
    end
    d = unique(d);
end



function win = localWindow(spec, a, name, fname, defaultWidth)
%LOCALWINDOW  The window on attribute a, a struct with fields prof
%   (internal.resolveProfile), closed, isPer, and period, from {shape,
%   width}, {shape, width, edges}, a function handle of the displacement,
%   or struct('shape', .., 'width' | 'sd' | 'decayRate', .., 'edges', ..).
%   The profiles are those of weightEvents: the rectangle-Gaussian family
%   ('rect', 'gaussian', or a number in [0, 1]), scaled by width or sd; the
%   exponentials aligned at the window's reference value ('exponential',
%   and 'exponentialBefore' and 'exponentialAfter', which extend to one
%   side of it only), scaled by sd or decayRate; and a function handle. The serial-position
%   profiles, anchored at the first and last events rather than at the
%   sweep value, are refused. The width may be left out ([] width, or the
%   whole spec []) only where DEFAULTWIDTH, a function handle, supplies
%   one; a rectangle whose width is defaulted is closed unless edges says
%   otherwise, and one whose width is given is half-open unless edges says
%   otherwise. Twin of the Python _parse_window.
    edges = []; sd = NaN; width = NaN; rate = NaN;
    if isempty(spec) && ~isa(spec, 'function_handle'), spec = {'rect', []}; end
    if isa(spec, 'function_handle')
        shape = spec;
    elseif isstruct(spec)
        if isfield(spec, 'shape'), shape = spec.shape; else, shape = 'rect'; end
        if isfield(spec, 'edges'), edges = spec.edges; end
        given = isfield(spec, {'sd', 'width', 'decayRate'});
        if sum(given) > 1
            error([fname ':badWindow'], ...
                ['%s for attribute %d: give one of width, sd, or ' ...
                 'decayRate.'], name, a);
        end
        if given(1), sd = double(spec.sd); end
        if given(2) && ~isempty(spec.width), width = double(spec.width); end
        if given(3), rate = double(spec.decayRate); end
    elseif iscell(spec) && (numel(spec) == 2 || numel(spec) == 3)
        shape = spec{1};
        if ~isempty(spec{2}), width = double(spec{2}); end
        if numel(spec) == 3, edges = spec{3}; end
    else
        error([fname ':badWindow'], ...
            ['%s for attribute %d: give {shape, width}, {shape, width, ' ...
             'edges}, a function handle, or struct(''shape'', .., ' ...
             '''width'' | ''sd'' | ''decayRate'', .., ''edges'', ..).'], ...
            name, a);
    end
    if isempty(shape), shape = 'rect'; end
    named = (ischar(shape) || isstring(shape)) && ~any(strcmpi(char(shape), ...
        {'rect', 'rectangular', 'box', 'gaussian', 'gauss', 'normal'}));
    defaulted = false;
    if ~named && ~isa(shape, 'function_handle') && isnan(sd) && isnan(width)
        if isempty(defaultWidth)
            error([fname ':badWindow'], ...
                '%s for attribute %d: a width is required, {shape, width}.', ...
                name, a);
        end
        width = defaultWidth();
        defaulted = true;
    end
    if named && ~isnan(width) && ~isstruct(spec)
        error([fname ':badWindow'], ...
            ['%s for attribute %d: profile ''%s'' has no width; give ' ...
             'struct(''shape'', ''%s'', ''sd'', ..) or struct(''shape'', ' ...
             '''%s'', ''decayRate'', ..).'], name, a, char(shape), ...
            char(shape), char(shape));
    end
    prof = internal.resolveProfile(shape, sd, width, rate, [], [], 0.5, ...
        [fname ':window']);
    if strcmp(prof.kind, 'anchored')
        error([fname ':badWindow'], ...
            ['%s for attribute %d: profile ''%s'' is anchored at the first ' ...
             'and last events'' values, not at the sweep value, so it ' ...
             'cannot be aligned; weight the events with weightEvents ' ...
             'before the call instead.'], name, a, char(shape));
    end
    if isempty(edges) && strcmp(prof.kind, 'family') && prof.shape == 1
        if defaulted, edges = 'closed'; else, edges = 'halfOpen'; end
    end
    closed = internal.resolveEdges(edges, prof, [fname ':window']);
    win = struct('prof', prof, 'closed', closed, 'isPer', false, 'period', 0);
end

function q = localQueryRef(pQuery, a, locate, rA, rV, middle)
%LOCALQUERYREF  queryRef on attribute a: as given; by default the query's
%   middle (the mean of its events' located values) where MIDDLE, else 0.
    ri = find(rA == a, 1);
    if ~isempty(ri)
        q = double(rV{ri});
    elseif middle
        q = mean(internal.locateRow(pQuery{a}, internal.axisLocate(locate, a)), ...
            'omitnan');
    else
        q = 0;
    end
end


function loc = localQueryLocated(pQuery, a, locate)
    loc = internal.locateRow(pQuery{a}, internal.axisLocate(locate, a));
    loc = loc(isfinite(loc));
end


function w = localHoldingWidth(pQuery, a, locate, ref, fname)
%LOCALHOLDINGWIDTH  The width of the smallest window that, aligned where
%   the query's reference REF lands, holds every one of the query's located
%   values on attribute a.
    loc = localQueryLocated(pQuery, a, locate);
    if isempty(loc), w = 0; else, w = 2 * max(abs(loc - ref)); end
    if ~(w > 0)
        error([fname ':noDefaultWidth'], ...
            ['attribute %d: the query''s events all lie at one value there, ' ...
             'so there is no width to take a default window from; give one, ' ...
             '''window'', {%d, {shape, width}}.'], a, a);
    end
end


function localWarnIfQueryCut(pQuery, a, locate, ref, win, fname)
%LOCALWARNIFQUERYCUT  Warn where the window, aligned where the query's
%   reference REF lands, leaves out some of the query's own events: the
%   query can then never be matched in full.
    loc = localQueryLocated(pQuery, a, locate);
    if isempty(loc), return; end
    nOut = sum(internal.windowFactor(loc, ref, win) == 0);
    if nOut > 0
        if strcmp(win.prof.kind, 'family') && win.prof.shape == 1 && ~win.closed
            hint = [', move queryRef towards the query''s middle, or close ' ...
                    'the rectangle (edges ''closed'')'];
        else
            hint = ' or move queryRef towards the query''s middle';
        end
        warning([fname ':windowCutsQuery'], ...
            ['attribute %d: the window, aligned where the query''s ' ...
             'reference lands, leaves out %d of the query''s %d events, so ' ...
             'the query can never be matched in full. Widen the window%s.'], ...
            a, nOut, numel(loc), hint);
    end
end


function v = localValues(v, a, fname)
    v = double(v(:)).';
    if isempty(v)
        error([fname ':badValues'], 'sweep for attribute %d is empty.', a);
    end
    if ~all(isfinite(v))
        error([fname ':badValues'], ...
            'sweep for attribute %d holds a non-finite value.', a);
    end
end


function c = localWindowDefault(pContext, a, stA, stV, spA, spV, sA, sV, ...
        defStep, fname, win, locate)
%LOCALWINDOWDEFAULT  Generated sweep values where only a window is placed:
%   a pure rectangle's pieces (localRectPieces) where no step is given,
%   and otherwise the uniform grid of localGenerate. Twin of the Python
%   _window_default.
    if isempty(localGet(sA, sV, a)) && ~isempty(win) ...
            && strcmp(win.prof.kind, 'family') && isequal(win.prof.shape, 1) ...
            && isfinite(win.prof.sd)
        v = double(pContext{a}(:)); v = v(isfinite(v));
        if isempty(v)
            error([fname ':noRange'], ...
                ['attribute %d: the context has no finite values there to ' ...
                 'take a default start / stop from.'], a);
        end
        lo = localGet(stA, stV, a); hi = localGet(spA, spV, a);
        if isempty(lo), lo = min(v); end
        if isempty(hi), hi = max(v); end
        if hi < lo
            error([fname ':badStep'], 'attribute %d: stop is below start.', a);
        end
        c = localRectPieces(pContext, a, locate, win, double(lo), double(hi));
        return;
    end
    c = localGenerate(pContext, a, stA, stV, spA, spV, sA, sV, defStep, fname);
end


function c = localRectPieces(pContext, a, locate, win, lo, hi)
%LOCALRECTPIECES  Default sweep values for a rectangular window placed
%   alone. As the window moves, the windowed context changes only where an
%   event enters or leaves it: at each event's located value plus or minus
%   half the window's width (and their images a period apart, on a
%   periodic attribute). Between these breakpoints the profile is
%   constant, so each piece is sampled just inside both its ends, and a
%   line through the values draws the steps exactly, every value being the
%   profile's value at its sweep value. The range [lo, hi] is sampled at
%   its ends. Twin of the Python _rect_pieces.
    hw = win.prof.sd * sqrt(3);
    loc = internal.locateRow(pContext{a}, internal.axisLocate(locate, a));
    loc = double(loc(:)); loc = loc(isfinite(loc));
    b = [loc - hw; loc + hw];
    if win.isPer && win.period > 0
        P = win.period;
        kLo = floor((lo - max(b)) / P) - 1;
        kHi = ceil((hi - min(b)) / P) + 1;
        b = reshape(b + P * (kLo:kHi), [], 1);
    end
    tol = 1e-9 * max([1, abs(lo), abs(hi), hw]);
    b = unique(b(b > lo + tol & b < hi - tol));
    if ~isempty(b)
        b = b([true; diff(b) > tol]);
    end
    edges = [lo; b; hi];
    gaps = diff(edges);
    if any(gaps > 0)
        epsv = min(1e-6 * hw, 0.25 * min(gaps(gaps > 0)));
    else
        epsv = 0;
    end
    c = zeros(1, 2 * numel(b) + 2);
    c(1) = lo;
    c(2:2:2 * numel(b)) = b - epsv;
    c(3:2:2 * numel(b) + 1) = b + epsv;
    if hi > lo
        c(end) = hi;
    else
        c = c(1:end - 1);
    end
end


function c = localGenerate(pContext, a, stA, stV, spA, spV, sA, sV, defStep, ...
        fname, defRange, openStop)
%LOCALGENERATE  Sweep values from start to stop in steps of step. start and
%   stop default to DEFRANGE where given, and otherwise to the lowest and
%   highest of the context's values on the attribute; step defaults to
%   DEFSTEP. With OPENSTOP (a periodic range) the stop value itself is left
%   out.
    if nargin < 11, defRange = []; end
    if nargin < 12, openStop = false; end
    lo = localGet(stA, stV, a); hi = localGet(spA, spV, a);
    stopGiven = ~isempty(hi);
    st = localGet(sA, sV, a);
    if isempty(st)
        if isempty(defStep)
            error([fname ':stepRequired'], ...
                ['attribute %d: give step; there is no kernel width or ' ...
                 'window sd on this attribute to take a default step ' ...
                 'from.'], a);
        end
        st = defStep;
    end
    if isempty(defRange) && (isempty(lo) || isempty(hi))
        v = double(pContext{a}(:)); v = v(isfinite(v));
        if isempty(v)
            error([fname ':noRange'], ...
                ['attribute %d: the context has no finite values there to ' ...
                 'take a default start / stop from.'], a);
        end
        defRange = [min(v), max(v)];
    end
    if isempty(lo), lo = defRange(1); end
    if isempty(hi), hi = defRange(2); end
    if ~(isfinite(st) && st > 0)
        error([fname ':badStep'], 'attribute %d: step must be finite and > 0.', a);
    end
    if hi < lo
        error([fname ':badStep'], 'attribute %d: stop is below start.', a);
    end
    nn = floor((hi - lo) / st + 1e-9) + 1;
    c = lo + st * (0:nn - 1);
    if openStop && ~stopGiven
        c = c(c < hi - 1e-9 * max(1, abs(hi)));
    end
end


function w = localPeakWidth(geom, specs, a)
%LOCALPEAKWIDTH  The standard deviation of the peaks of a translation
%   profile on attribute a. Translating the query moves all D coordinates
%   of the attribute's tuple alike, so each pair of tuples contributes a
%   Gaussian in the offset of variance 2 / (1' Sigma^-1 1) (Milne 2026,
%   Eq. 10 and Online Supplement Sec. 4): sigma * sqrt(2 / D) for an
%   isotropic kernel of width sigma, narrower the larger the tuple. [] where
%   there is no kernel width. Twin of the Python _peak_width.
    w = [];
    if ~isempty(geom) && isfield(geom, 'sigma') && ~isempty(geom.sigma)
        sg = localPick(geom.sigma, a);
        if ~isempty(sg)
            sg = double(sg);
            if ismatrix(sg) && size(sg, 1) == size(sg, 2) && numel(sg) > 1
                o = ones(size(sg, 1), 1);
                q = o' * (sg \ o);
                if isfinite(q) && q > 0
                    w = sqrt(2 / q);
                    return;
                end
            end
        end
    end
    kw = localKernelWidth(geom, a);
    if isempty(kw), return; end
    w = kw * sqrt(2 / max(localTupleDim(geom, specs, a), 1));
end


function D = localTupleDim(geom, specs, a)
%LOCALTUPLEDIM  The number of coordinates of attribute a's tuple: the
%   product of a nested attribute's per-level tuple sizes, or its tuple
%   size r; 1 where neither is given. Twin of the Python _tuple_dim.
    D = 1;
    if iscell(specs) && numel(specs) >= a && isstruct(specs{a}) ...
            && isfield(specs{a}, 'r') && ~isempty(specs{a}.r)
        D = prod(double(specs{a}.r(:)));
        return;
    end
    if ~isempty(geom) && isfield(geom, 'r') && ~isempty(geom.r)
        ra = localPick(geom.r, a);
        if ~isempty(ra), D = prod(double(ra(:))); end
    end
end


function w = localKernelWidth(geom, a)
%LOCALKERNELWIDTH  One kernel standard deviation for attribute a: the
%   scalar sigma, the square root of the largest diagonal entry of a kernel
%   covariance, or the smallest finite entry of a per-level vector; [] where
%   there is none. Twin of the Python _kernel_width.
    w = [];
    if isempty(geom) || ~isfield(geom, 'sigma') || isempty(geom.sigma), return; end
    sg = localPick(geom.sigma, a);
    if isempty(sg), return; end
    sg = double(sg);
    if ismatrix(sg) && size(sg, 1) == size(sg, 2) && numel(sg) > 1
        d = diag(sg); d = d(isfinite(d) & d > 0);
        if ~isempty(d), w = sqrt(max(d)); end
    else
        v = sg(:); v = v(isfinite(v) & v > 0);
        if ~isempty(v), w = min(v); end
    end
end


function x = localPick(v, a)
%LOCALPICK  Entry a of a per-attribute vector or cell ([] where absent).
    x = [];
    if iscell(v)
        if numel(v) >= a, x = v{a}; end
    elseif numel(v) >= a
        x = v(a);
    end
end


function [per, pd] = localPeriod(geom, a)
%LOCALPERIOD  Whether attribute a is periodic, and its period.
    per = false; pd = 0;
    if ~isempty(geom)
        if isfield(geom, 'isPer'), x = localPick(geom.isPer, a); per = ~isempty(x) && logical(x); end
        if isfield(geom, 'period'), x = localPick(geom.period, a); if ~isempty(x), pd = double(x); end; end
    end
end


function st = localLatticeStep(pContext, pQuery, a, half, period)
%LOCALLATTICESTEP  The default translation step on attribute a: at most
%   HALF (half the peaks' sd), and a whole fraction of the lattice the
%   exact matches lie on, where they lie on one. Every exact match is at an
%   offset that is a difference between a context value and a query value.
%   Where the context's values are whole multiples of a spacing g apart,
%   and so are the query's (and g divides PERIOD, where it is positive),
%   every such offset is the lowest one, min(context) - max(query), plus a
%   whole multiple of g. A step of g / k, with k the smallest whole number
%   that brings it to HALF or below, then puts every one of them on a grid
%   through that lowest offset. Where g is below HALF it is the step
%   itself, provided it is at least HALF / 4; otherwise (values on no
%   lattice, or on one too fine to step at) the step is HALF. Twin of the
%   Python _lattice_step.
    st = half;
    d = [];
    ops = {pContext, pQuery};
    for k = 1:2
        v = double(ops{k}{a}(:)); v = v(isfinite(v));
        if ~isempty(v), d = [d; v - min(v)]; end %#ok<AGROW>
    end
    if period > 0, d = [d; period]; end
    tol = 1e-6 * half;
    d = unique(d);
    d = d(d > tol);
    if isempty(d), return; end
    floorG = half / 4;
    g = d(1);
    for i = 2:numel(d)
        big = max(g, d(i)); small = min(g, d(i));
        while small > tol
            r = rem(big, small);
            if small - r <= tol, r = 0; end
            big = small; small = r;
        end
        g = big;
        if g < floorG, return; end
    end
    % The tolerant Euclid can drift; accept g only if every difference is a
    % whole multiple of it.
    if max(abs(d / g - round(d / g))) * g > 1e-4 * half, return; end
    st = g / ceil(g / half - 1e-9);
end


function [r, open] = localTranslationRange(pContext, pQuery, a, ref, geom, fname)
%LOCALTRANSLATIONRANGE  The sweep values at which the query, placed by its
%   reference REF, overlaps the context on attribute a: one period from 0
%   (plus REF) on a periodic attribute; otherwise from the value that puts
%   the query's highest value on the context's lowest to the value that
%   puts its lowest on the context's highest. Twin of the Python
%   _translation_range.
    [per, pd] = localPeriod(geom, a);
    if per && pd > 0
        r = [ref, ref + pd]; open = true; return;
    end
    c = double(pContext{a}(:)); c = c(isfinite(c));
    q = double(pQuery{a}(:)); q = q(isfinite(q));
    if isempty(c) || isempty(q)
        error([fname ':noRange'], ...
            ['attribute %d: the context or the query has no finite values ' ...
             'there to take a default sweep range from; give sweep values ' ...
             'or start / stop.'], a);
    end
    r = [min(c) - max(q), max(c) - min(q)] + ref; open = false;
end


function v = localGet(keys, vals, a)
    i = find(keys == a, 1);
    if isempty(i), v = []; else, v = double(vals{i}); end
end

function v = localBareGenerator(v, name, swA, fname)
% A bare number for start, stop, or step, as {a, v}: it applies to the
% attribute 'sweep' names, and only where it names exactly one.
if ~(isnumeric(v) && isscalar(v))
    return
end
if numel(swA) ~= 1
    error([fname ':bareGenerator'], ...
        ['a bare ''%s'' applies to the one swept attribute, but ''sweep'' ' ...
         'names %d; give ''%s'', {a, value}.'], name, numel(swA), name);
end
v = {swA, v};
end

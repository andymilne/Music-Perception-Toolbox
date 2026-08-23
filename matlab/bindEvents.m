function [pAttrBound, wBound, specs] = bindEvents(pAttr, w, bindOrders, nvArgs)
%BINDEVENTS Bind sliding windows of consecutive events into nested attributes.
%
%   [pAttrBound, wBound, specs] = bindEvents(pAttr, w, bindOrders, ...)
%   is a cross-event preprocessing helper on the (pAttr, w, specs) triple.
%   For each input attribute a, a sliding window of width L_a (bindOrders)
%   is laid across the event axis and the L_a consecutive events are nested
%   into a single output attribute (toolbox spec §6.1/§6.5): the bound
%   events form an ordered outer level (symOuter = 0 by default, lossless),
%   each event's own atom multiset is the inner level.
%
%   The inner level's geometry (r/rel/sym) is read from the incoming
%   specifications --- the attribute's existing specification supplies the inner
%   level(s). specs = [] synthesises flat specs (flatSpecs defaults: r = 1,
%   rel = 0, sym = 1). The outer level defaults to r = L_a (read the whole
%   bound window), sym = 0, rel = 0. L_a = 1 is the no-op: the incoming
%   (flat) spec passes through unchanged. With the defaults and
%   rel = [relIn, 0], the outer r = L_a reading reproduces the old
%   separate-attribute tensor join (§6.5). The genuinely new lever is
%   relOuter = 1 on an absolute attribute, giving the global-transposition
%   quotient rel = [0, 1].
%
%   Event-axis alignment. At the default step = 1 the common output event
%   count is N' = N - max_a L_a + 1 (non-circular) or N (circular);
%   attributes with L_a < max_a L_a keep their leading N' windows (composes
%   with differenceEvents). A step > 1 hops the windows (see the step
%   name-value), shrinking N'; the difference-composition identity then holds
%   at step = 1 only.
%
%   Inputs
%       pAttr      - 1 x A cell of K_a x N per-attribute value matrices.
%       w          - Weights ([], scalar, or 1 x A cell). Same convention
%                    as buildExpTens; bound value weights are the windowed-
%                    and-stacked input weights.
%       bindOrders - Scalar or 1 x A window widths L_a >= 1 (1 = no-op).
%
%   Name-value pairs
%       'circular'   - false (default) or true (wrap window; N' = N).
%       'step'     - Hop between consecutive bound windows along the event
%                      axis (default 1, the fully overlapping slide).
%                      Super-event i reads events [i*step, i*step + L_a),
%                      so step = L_a gives non-overlapping blocks. A single
%                      scalar applies to all attributes (the hop is a property
%                      of the shared event axis). N' = floor((N - max_a L_a) /
%                      step) + 1 (non-circular); for circular = true, N must
%                      be divisible by step and N' = N / step. The bind/
%                      difference composition identity holds at step = 1.
%       'specs'      - [] (synthesise flat via flatSpecs) or a 1 x A cell of
%                      per-attribute specs supplying the inner geometry. An
%                      incoming spec may be flat or already nested: a flat
%                      spec becomes the inner level of a new two-level
%                      attribute, while an already-nested spec is deepened ---
%                      a new outermost level (rOuter/symOuter/relOuter) is
%                      appended above the existing nesting, and tags, r, sym,
%                      and rel each extend by one entry. Repeated binds nest
%                      to arbitrary depth, but each call must be given the
%                      specs returned by the previous one: passing [] on an
%                      already-bound attributes re-synthesises flat specs,
%                      silently discarding the existing nesting and producing
%                      a shallower result.
%       'rOuter'     - [] (default L_a) or scalar/1xA outer-level r.
%       'symOuter'   - outer-level [sym] (default false; bag reading if true).
%       'relOuter'   - outer-level [rel] (default false).
%       'name'       - [] , char, or 1 x A names; overrides any name carried
%                      on the incoming spec, otherwise inherited.
%       'levelNames' - [] or 1 x 2 {inner outer} level names per nested spec.
%                      Applies only when the incoming spec is flat; supplying
%                      it while deepening an already-nested attribute is
%                      rejected (per-level names carry through from the input).
%
%   Outputs
%       pAttrBound - 1 x A cell. For L_a >= 2 a stacked (L_a*K_a) x N'
%                    value matrix; for L_a = 1 the leading-aligned K_a x N'.
%       wBound     - Transformed weights aligned to the value layout.
%       specs      - 1 x A cell of structs: nested {tags,r,sym,rel,...}
%                    for L_a >= 2, the incoming spec unchanged (flat or
%                    nested) for L_a = 1.
%
%   See also BUILDEXPTENS, DIFFERENCEEVENTS, FLATSPECS, TRANSLATEATTRIBUTES.

arguments
    pAttr
    w
    bindOrders
    nvArgs.circular (1, 1) logical = false
    nvArgs.step (1, 1) double {mustBeInteger, mustBePositive} = 1
    nvArgs.specs = []
    nvArgs.rOuter = []
    nvArgs.symOuter = false
    nvArgs.relOuter = false
    nvArgs.name = []
    nvArgs.levelNames = []
    nvArgs.groupBy = []
    nvArgs.groupAtol (1, 1) double = 0
end

% --- Normalise pAttr to a cell of 2-D double matrices ---
if ~iscell(pAttr)
    error('bindEvents:badPAttrType', ...
          'pAttr must be a cell array of per-attribute matrices.');
end
A = numel(pAttr);
if A < 1
    error('bindEvents:noAttrs', 'pAttr must contain at least one attribute.');
end
K = zeros(1, A);
for a = 1:A
    M = pAttr{a};
    if ~isnumeric(M) || ndims(M) > 2
        error('bindEvents:badAttrShape', ...
              'Attribute %d input must be a numeric 2-D matrix.', a);
    end
    if size(M, 1) == 0
        error('bindEvents:emptyAttribute', ...
              ['Attribute %d has K_a = 0 (empty attribute); empty ' ...
               'attributes are not permitted.'], a);
    end
    pAttr{a} = double(M);
    K(a) = size(pAttr{a}, 1);
end

% --- Verify shared event count N ---
nEvents = size(pAttr{1}, 2);
for a = 2:A
    if size(pAttr{a}, 2) ~= nEvents
        error('bindEvents:eventCountMismatch', ...
              ['All attributes must share the same event count N. ' ...
               'Attribute 1 has N=%d; attribute %d has N=%d.'], ...
              nEvents, a, size(pAttr{a}, 2));
    end
end

% --- Parse bindOrders (scalar or 1 x A) ---
if ~isempty(nvArgs.groupBy)
    [pAttrBound, wBound, specs] = localBindEventsRunLength( ...
        pAttr, w, K, A, nEvents, bindOrders, nvArgs);
    return
end

orders = localCanonicaliseBindOrders(bindOrders, A);

% --- Inner geometry from the attribute specifications ---
if isempty(nvArgs.specs)
    specsIn = flatSpecs(pAttr);
else
    specsIn = nvArgs.specs;
    if ~iscell(specsIn) || numel(specsIn) ~= A
        error('bindEvents:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end
end
for a = 1:A
    if isstruct(specsIn{a}) && isfield(specsIn{a}, 'tags') ...
            && ~isempty(nvArgs.levelNames)
        error('bindEvents:levelNamesNested', ...
              ['attribute %d: levelNames is not supported when deepening an ' ...
               'already-nested attribute; the per-level names carry through ' ...
               'from the incoming spec.'], a);
    end
end

% --- Outer-level overrides ---
if isempty(nvArgs.rOuter)
    rOut = double(orders);
else
    rOut = localBcastGeom(nvArgs.rOuter, A, 'rOuter', false);
end
symOut = localBcastGeom(nvArgs.symOuter, A, 'symOuter', true);
relOut = localBcastGeom(nvArgs.relOuter, A, 'relOuter', true);
namesAttr = localBcastNames(nvArgs.name, A);
levelNames = nvArgs.levelNames;
if ~isempty(levelNames) && numel(levelNames) ~= 2
    error('bindEvents:levelNames', ...
          'levelNames must be a 1 x 2 {inner outer} cell (two-level binding).');
end

% --- Output sizes ---
maxOrder = max(orders);
if nvArgs.circular
    if nvArgs.step > 1 && mod(nEvents, nvArgs.step) ~= 0
        error('bindEvents:stepDivisibility', ...
              ['Circular binding with step = %d requires the event ' ...
               'count N = %d to be divisible by step.'], ...
              nvArgs.step, nEvents);
    end
    nPrime = nEvents / nvArgs.step;
    if maxOrder > nEvents
        error('bindEvents:windowTooLarge', ...
              'Circular window size max L = %d exceeds event count N = %d.', ...
              double(maxOrder), nEvents);
    end
else
    nPrime = floor((nEvents - double(maxOrder)) / nvArgs.step) + 1;
    if nPrime < 1
        error('bindEvents:windowTooLarge', ...
              ['Bind orders too high for the input event count: max L = ' ...
               '%d but N = %d (non-circular).'], double(maxOrder), nEvents);
    end
end

% --- Build output value matrices and specs ---
pAttrBound = cell(1, A);
specs = cell(1, A);
for a = 1:A
    L_a = double(orders(a));
    K_a = K(a);                           % flat value count K_total
    Marr = pAttr{a};
    sIn = specsIn{a};
    isNestedIn = isstruct(sIn) && isfield(sIn, 'tags');
    nameInA = localSpecField(sIn, 'name', []);
    if ~isempty(namesAttr{a})
        nm = namesAttr{a};
    else
        nm = nameInA;
    end
    if L_a == 1
        pAttrBound{a} = Marr(:, localLagIndex(0, nPrime, nEvents, nvArgs.circular, nvArgs.step));
        spec = sIn;                       % passthrough (flat or nested)
        if ~isempty(nm); spec.name = nm; end
        specs{a} = spec;
        continue
    end
    blocks = cell(1, L_a);
    for ell = 0:(L_a - 1)
        idx = localLagIndex(ell, nPrime, nEvents, nvArgs.circular, nvArgs.step);
        blocks{ell + 1} = Marr(:, idx);
    end
    pAttrBound{a} = vertcat(blocks{:});
    newColRow = repelem(0:(L_a - 1), K_a);    % 1 x (K_total*L_a)
    if isNestedIn
        % Deepen: append a new outermost grouping level above the existing
        % nesting. The existing tag columns are tiled once per bound
        % super-event; the new column distinguishes the L_a bound
        % super-events. r/sym/rel extend by the new outer level.
        tagsIn = sIn.tags;
        if isvector(tagsIn)
            tagsIn = tagsIn(:);               % K_total x 1 (L_in = 2)
        end
        tagsNew = [repmat(tagsIn, L_a, 1), newColRow.'];
        spec = struct('tags', tagsNew, ...
                      'r',   [sIn.r(:).',            rOut(a)], ...
                      'sym', [logical(sIn.sym(:).'), logical(symOut(a))], ...
                      'rel', [double(sIn.rel(:).'),  double(relOut(a))]);
        if isfield(sIn, 'names') && ~isempty(sIn.names)
            spec.names = [sIn.names(:).', {[]}];
        end
    else
        % Flat input -> two-level nested attribute (unchanged).
        rInA   = double(localSpecField(sIn, 'r',   1));
        relInA = logical(localSpecField(sIn, 'rel', false));
        symInA = logical(localSpecField(sIn, 'sym', true));
        spec = struct('tags', newColRow, 'r', [rInA rOut(a)], ...
                      'sym', [symInA symOut(a)], 'rel', [relInA relOut(a)]);
        if ~isempty(levelNames); spec.names = levelNames; end
    end
    if ~isempty(nm); spec.name = nm; end
    specs{a} = spec;
end

% --- Transform weights ---
wBound = localBindWeightsNested(w, A, orders, K, nEvents, nPrime, nvArgs.circular, nvArgs.step);

end


% =========================================================================
%  Helpers
% =========================================================================

function idx = localLagIndex(ell, nPrime, nEvents, isCircular, step)
    if isCircular
        idx = mod((0:nPrime - 1) * step + ell, nEvents) + 1;
    else
        idx = (0:nPrime - 1) * step + ell + 1;
    end
end


function v = localSpecField(s, f, d)
    % Read field f from spec struct s, defaulting to d if absent/empty.
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f))
        v = s.(f);
    else
        v = d;
    end
end


function out = localBcastGeom(x, A, name, asLogical)
    if ~isnumeric(x) && ~islogical(x)
        error('bindEvents:bcastType', '%s must be numeric or logical.', name);
    end
    v = x(:).';
    if numel(v) == 1
        out = repmat(v, 1, A);
    elseif numel(v) == A
        out = v;
    else
        error('bindEvents:bcastLength', ...
              '%s must be scalar or length-A (%d); got %d.', name, A, numel(v));
    end
    if asLogical
        out = logical(out);
    else
        out = double(out);
    end
end


function names = localBcastNames(name, A)
    names = cell(1, A);
    if isempty(name)
        return;
    end
    if ischar(name) || (isstring(name) && isscalar(name))
        for a = 1:A; names{a} = char(name); end
        return;
    end
    if iscell(name)
        if numel(name) ~= A
            error('bindEvents:nameLength', 'name cell must be length-A (%d).', A);
        end
        for a = 1:A; names{a} = name{a}; end
        return;
    end
    error('bindEvents:nameType', 'name must be [], a char, or a 1 x A cell.');
end


function orders = localCanonicaliseBindOrders(bindOrders, A)
%LOCALCANONICALISEBINDORDERS  Coerce bindOrders to a 1 x A row (scalar/per-attr).
    if ~isnumeric(bindOrders)
        error('bindEvents:badBindOrdersType', ...
              'bindOrders must be numeric; got class %s.', class(bindOrders));
    end
    v = double(bindOrders(:).');
    n = numel(v);
    if n == 1
        orders = v(1) * ones(1, A);
    elseif n == A
        orders = v;
    else
        error('bindEvents:badBindOrdersLength', ...
              ['bindOrders has %d entries; expected scalar (1) or ' ...
               'per-attribute (A = %d).'], n, A);
    end
    if any(orders < 1) || any(orders ~= round(orders))
        error('bindEvents:badBindOrders', ...
              ['All entries of bindOrders must be positive integers ' ...
               '(>= 1; L = 1 is the no-op).']);
    end
end


function wOut = localBindWeightsNested(w, A, orders, K, nEvents, nPrime, isCircular, step)
%LOCALBINDWEIGHTSNESTED  Transform weights to match the nested value layout.
%
%  For L_a >= 2 the per-event weight slices are windowed and stacked into a
%  (L_a*K_a) x N' column aligned with the value stack (per-event weights
%  expanded across the K_a values of their event); for L_a = 1 the weight is
%  leading-aligned. Non-event-dependent inputs ([], scalar, K_a x 1
%  column) are inherited / tiled across the bound values.

    if isempty(w) && ~iscell(w)
        wOut = [];
        return;
    end
    if isnumeric(w) && isscalar(w)
        wOut = double(w);
        return;
    end
    if ~iscell(w)
        error('bindEvents:badWeightsType', ...
              ['w must be [], a scalar, or a cell array of per-attribute ' ...
               'weight inputs.']);
    end
    if numel(w) ~= A
        error('bindEvents:badWeightsLength', ...
              'Weight cell must have length A = %d; got length %d.', A, numel(w));
    end

    wOut = cell(1, A);
    for a = 1:A
        L_a = double(orders(a));
        K_a = K(a);
        wa = w{a};
        eventDep = localWeightHasEventDep(wa, nEvents, a);
        if ~eventDep
            if isempty(wa) || (isnumeric(wa) && isscalar(wa))
                wOut{a} = wa;
            else
                W = double(wa);   % K_a x 1 per-value column
                if L_a == 1
                    wOut{a} = W;
                else
                    wOut{a} = repmat(W, L_a, 1);
                end
            end
            continue;
        end
        % Event-dependent: materialise to K_a x N, window per lag, stack.
        W = double(wa);
        if size(W, 1) == 1 && K_a > 1
            W = repmat(W, K_a, 1);
        end
        if L_a == 1
            wOut{a} = W(:, localLagIndex(0, nPrime, nEvents, isCircular, step));
        else
            blocks = cell(1, L_a);
            for ell = 0:(L_a - 1)
                idx = localLagIndex(ell, nPrime, nEvents, isCircular, step);
                blocks{ell + 1} = W(:, idx);
            end
            wOut{a} = vertcat(blocks{:});
        end
    end
end


function tf = localWeightHasEventDep(wa, N, attrIdx)
    if isempty(wa)
        tf = false; return;
    end
    if ~isnumeric(wa)
        error('bindEvents:badWeightType', ...
              'Attribute %d weight must be numeric.', attrIdx);
    end
    if isscalar(wa)
        tf = false; return;
    end
    sz = size(wa);
    if numel(sz) == 2 && sz(2) == N
        tf = true; return;
    end
    if numel(sz) == 2 && sz(2) == 1
        tf = false; return;
    end
    error('bindEvents:badWeightShape', ...
          ['Attribute %d weight has shape [%s]; expected [], scalar, ' ...
           '[K_a 1] column, [1 %d] row, or [K_a %d] matrix.'], ...
          attrIdx, num2str(sz), N, N);
end


function [pAttrBound, wBound, specs] = localBindEventsRunLength( ...
        pAttr, w, K, A, nEvents, bindOrders, nvArgs)
%LOCALBINDEVENTSRUNLENGTH  Run-length (bind-by-attribute) binding.
%   Consecutive events sharing a constant value on attribute groupBy are
%   gathered into one super-event; a new group begins where the value
%   changes. Group sizes vary, so the outer level is ragged: each
%   super-event is padded to the maximum group size with NaN values carrying
%   zero weight. The inner level preserves each attribute's parameters; the
%   outer tuple size rOuter defaults to the smallest group size. Mirror of
%   Python _bind_events_run_length.
    groupBy  = nvArgs.groupBy;
    groupAtol = nvArgs.groupAtol;
    if ~isempty(bindOrders)
        error('bindEvents:bindOrdersWithGroupBy', ...
            ['bindOrders and groupBy are mutually exclusive: run-length ' ...
             'binding reads the group sizes from the data, so pass ' ...
             'bindOrders = [] when groupBy is given.']);
    end
    if nvArgs.circular
        error('bindEvents:runLengthCircular', ...
            'Circular run-length binding is not yet supported; pass circular = false.');
    end
    if nvArgs.step ~= 1
        error('bindEvents:runLengthStep', ...
            ['step has no meaning for run-length binding (groups are read ' ...
             'from the data, not hopped); leave step at its default.']);
    end
    if ~(isscalar(groupBy) && groupBy == round(groupBy) ...
            && groupBy >= 1 && groupBy <= A)
        error('bindEvents:badGroupBy', ...
            'groupBy must be an attribute index in 1..%d.', A);
    end
    if K(groupBy) ~= 1
        error('bindEvents:groupByCardinality', ...
            ['groupBy attribute %d must have K = 1 (one value per event); ' ...
             'got K = %d. Constancy across multiple values is ambiguous.'], ...
            groupBy, K(groupBy));
    end

    if isempty(nvArgs.specs)
        specsIn = flatSpecs(pAttr);
    else
        specsIn = nvArgs.specs;
        if ~iscell(specsIn) || numel(specsIn) ~= A
            error('bindEvents:badSpecs', ...
                'specs must be a 1 x A (%d) cell, one per attribute.', A);
        end
    end
    for a = 1:A
        if isstruct(specsIn{a}) && isfield(specsIn{a}, 'tags')
            error('bindEvents:runLengthNested', ...
                ['attribute %d: run-length binding of an already-nested ' ...
                 'attribute is not yet supported (flat inputs only).'], a);
        end
    end

    % --- consecutive runs on the grouping attribute ---
    gVals = pAttr{groupBy}(1, :);
    if nEvents == 0
        error('bindEvents:runLengthEmpty', 'groupBy produced no groups.');
    end
    if groupAtol == 0
        changes = gVals(2:end) ~= gVals(1:end-1);
    else
        changes = abs(gVals(2:end) - gVals(1:end-1)) > groupAtol;
    end
    starts = [1, find(changes) + 1, nEvents + 1];
    nPrime = numel(starts) - 1;
    sizes = zeros(1, nPrime);
    src = ones(0, nPrime);                 %#ok<NASGU> placeholder
    for j = 1:nPrime
        sizes(j) = starts(j + 1) - starts(j);
    end
    Lmax = max(sizes);
    Lmin = min(sizes);

    % source event index per (outer position, group) and a validity mask
    src = ones(Lmax, nPrime);
    valid = false(Lmax, nPrime);
    for j = 1:nPrime
        s = sizes(j);
        idx = starts(j):(starts(j + 1) - 1);
        src(1:s, j) = idx(:);
        valid(1:s, j) = true;
    end

    % --- outer-level overrides ---
    if isempty(nvArgs.rOuter)
        rOut = repmat(Lmin, 1, A);
    else
        rOut = localBcastGeom(nvArgs.rOuter, A, 'rOuter', false);
    end
    symOut = localBcastGeom(nvArgs.symOuter, A, 'symOuter', true);
    relOut = localBcastGeom(nvArgs.relOuter, A, 'relOuter', true);
    namesAttr = localBcastNames(nvArgs.name, A);
    levelNames = nvArgs.levelNames;
    if ~isempty(levelNames) && numel(levelNames) ~= 2
        error('bindEvents:levelNames', ...
            'levelNames must be a 1 x 2 {inner outer} cell (two-level binding).');
    end

    pAttrBound = cell(1, A);
    wBound = cell(1, A);
    specs = cell(1, A);
    for a = 1:A
        Marr = pAttr{a};
        K_a = K(a);
        blocks = cell(1, Lmax);
        wblocks = cell(1, Lmax);
        for ell = 1:Lmax
            vmask = valid(ell, :);                 % 1 x nPrime
            cols = Marr(:, src(ell, :));           % K_a x nPrime
            cols(:, ~vmask) = NaN;
            blocks{ell} = cols;
            if isempty(w)
                wb = double(repmat(vmask, K_a, 1));
            else
                wa = w{a};
                wb = wa(:, src(ell, :));
                wb(:, ~vmask) = 0;
            end
            wblocks{ell} = wb;
        end
        pAttrBound{a} = vertcat(blocks{:});
        wBound{a} = vertcat(wblocks{:});

        newColRow = repelem(0:(Lmax - 1), K_a);
        sIn = specsIn{a};
        rInA   = double(localSpecField(sIn, 'r',   1));
        relInA = logical(localSpecField(sIn, 'rel', false));
        symInA = logical(localSpecField(sIn, 'sym', true));
        spec = struct('tags', newColRow, 'r', [rInA rOut(a)], ...
                      'sym', [symInA symOut(a)], 'rel', [relInA relOut(a)]);
        if ~isempty(levelNames); spec.names = levelNames; end
        if ~isempty(namesAttr{a})
            spec.name = namesAttr{a};
        elseif ~isempty(localSpecField(sIn, 'name', []))
            spec.name = localSpecField(sIn, 'name', []);
        end
        specs{a} = spec;
    end
end

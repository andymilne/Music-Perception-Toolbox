function [pAttrDiff, wDiff, groupsDiff] = differenceEvents(pAttr, w, groups, diffOrders, nvArgs)
%DIFFERENCEEVENTS Replace selected attributes' event sequences with differences.
%
%   [pAttrDiff, wDiff, groupsDiff] = differenceEvents(pAttr, w, groups, diffOrders)
%   [pAttrDiff, wDiff, groupsDiff] = differenceEvents(pAttr, w, groups, diffOrders, 'circular', false)
%   is a cross-event preprocessing helper for multi-attribute tensor
%   input. It takes the (pAttr, w, groups) triple that one would
%   otherwise feed to buildExpTens and returns a transformed
%   (pAttrDiff, wDiff, groupsDiff) triple ready to chain into another
%   pre-MAET operation or into buildExpTens.
%
%   Differencing orders are specified per attribute (via Option C
%   syntax — see below). The k_a-th finite difference is applied
%   along the event axis to each attribute. With circular = false
%   (default), the output event count for an attribute with order k_a
%   is N - k_a, and attributes are brought onto a common output grid
%   of length N' = N - max_a k_a by dropping leading max_a k_a - k_a
%   events from each. With circular = true, the event index wraps at
%   the sequence boundary (n - 1 is taken cyclically: position 0 is
%   identified with position N), so every attribute's output retains
%   length N regardless of its order; no alignment drop is needed.
%
%   The output values are emitted raw; periodic groups are NOT wrapped
%   here, regardless of [per] settings. Wrapping (when desired) is the
%   kernel's job in buildExpTens, consulting the group's [per] flag.
%
%   Differencing requires K_a = 1 for any attribute being differenced
%   (k_a > 0). Multi-slot attributes (K_a > 1) are permitted in the
%   input but only as pass-through (k_a = 0); if the analyst specifies
%   a non-zero order for a K_a > 1 attribute, a warning is issued and
%   that attribute is treated as k_a = 0 (still subject to leading-
%   event drop for alignment in the non-circular case, or pass-through
%   in the circular case). The warning is emitted at most once per
%   call.
%
%   Per-attribute weights propagate as a rolling product over the
%   k_a + 1 constituent input events for each differenced attribute,
%   under the standard weights-as-salience reading. Indexing wraps
%   when circular = true; pass-through attributes (k_a = 0) have their
%   leading events dropped (non-circular) or passed unchanged
%   (circular).
%
%   Inputs
%       pAttr      - 1 x A cell array of K_a x N per-attribute value
%                    matrices. K_a >= 1; K_a = 0 (empty attribute) is
%                    rejected. K_a > 1 is permitted as pass-through.
%       w          - Weights. [], scalar, or 1 x A cell of per-attribute
%                    weight inputs (each [], scalar, 1 x N row, K_a x 1
%                    column, or K_a x N matrix). Same convention as
%                    buildExpTens.
%       groups     - Group assignment. [] (each attribute its own group),
%                    a 1 x A index vector, or a 1 x G cell of attribute-
%                    index lists. Matches buildExpTens.
%       diffOrders - Per-attribute or per-group differencing orders
%                    (non-negative integers). Option C syntax:
%
%                      scalar              broadcast to all attributes
%                      1 x A or A x 1      per-attribute
%                      1 x G or G x 1      per-group, broadcast within
%                                          group (G ~= A; the A == G
%                                          case is read as per-attribute,
%                                          identical output)
%                      1 x G cell          per-group, with each cell:
%                                            [] (skip → 0),
%                                            scalar (broadcast in group),
%                                            length-n_g vector
%                                            (per-attribute in group).
%
%   Outputs
%       pAttrDiff  - 1 x A cell of transformed per-attribute matrices,
%                    each K_a x N'.
%       wDiff      - Transformed weights. Shape mirrors w: [] stays [];
%                    a scalar stays a scalar when all attributes share
%                    the same order, expanding to a 1 x A cell of per-
%                    attribute scalars when orders vary; a 1 x A cell
%                    stays a 1 x A cell, with per-attribute event-
%                    dependent entries becoming K_a x N' matrices and
%                    non-event-dependent entries keeping their input
%                    shape.
%       groupsDiff - Same group structure as input (differencing does
%                    not change group membership). Returned for clean
%                    chaining of pre-MAET operations.
%
%   Name-Value
%       'circular' - false (default) or true. When true, the difference
%                    operator wraps at the event-sequence boundary:
%                    Delta p(n) = p(n) - p(prev(n)) with prev(1) = N,
%                    so each attribute's output has N events regardless
%                    of order. When false, the leading k_a events of
%                    each differenced attribute are dropped and all
%                    attributes are aligned to N' = N - max_a k_a.
%                    Suitable for cyclic event sequences (looped
%                    rhythms, ostinati) in which the boundary
%                    difference is a genuine inter-event interval, not
%                    an artefact of the sequence cutting off.
%
%   See also BUILDEXPTENS, BINDEVENTS, TRANSLATEATTRIBUTES, WEIGHTEVENTS.

arguments
    pAttr
    w
    groups
    diffOrders
    nvArgs.circular (1, 1) logical = false
end

% --- Normalise pAttr to a cell of 2-D double matrices ---
if ~iscell(pAttr)
    error('differenceEvents:badPAttrType', ...
          'pAttr must be a cell array of per-attribute matrices.');
end
A = numel(pAttr);
if A < 1
    error('differenceEvents:noAttrs', ...
          'pAttr must contain at least one attribute.');
end
for a = 1:A
    M = pAttr{a};
    if ~isnumeric(M) || ndims(M) > 2
        error('differenceEvents:badAttrShape', ...
              'Attribute %d input must be a numeric 2-D matrix.', a);
    end
    if size(M, 1) == 0
        error('differenceEvents:emptyAttribute', ...
              ['Attribute %d has K_a = 0 (empty attribute); empty ' ...
               'attributes are not permitted.'], a);
    end
    pAttr{a} = double(M);
end

% --- Verify shared event count N ---
nEvents = size(pAttr{1}, 2);
for a = 2:A
    if size(pAttr{a}, 2) ~= nEvents
        error('differenceEvents:eventCountMismatch', ...
              ['All attributes must share the same event count N. ' ...
               'Attribute 1 has N=%d; attribute %d has N=%d.'], ...
              nEvents, a, size(pAttr{a}, 2));
    end
end

% --- Canonicalise groups: 1xA assignment, attrsOfGroup, G ---
[groupOfAttr, attrsOfGroup, G] = localCanonicaliseGroups(groups, A);

% --- Parse diffOrders via Option C → ordersPerAttr (1xA) ---
ordersPerAttr = localCanonicaliseDiffOrders( ...
    diffOrders, A, G, attrsOfGroup);

% --- Handle K_a > 1 with order > 0: warn once and pass through ---
warnedMultiSlot = false;
for a = 1:A
    K_a = size(pAttr{a}, 1);
    if K_a > 1 && ordersPerAttr(a) > 0
        if ~warnedMultiSlot
            warning('differenceEvents:multiSlotAttributeDifferenced', ...
                    ['Attribute %d has K_a = %d but was assigned ' ...
                     'order %d; event differencing requires K_a = 1 ' ...
                     'for differenced attributes (column-wise ' ...
                     'subtraction imposes a cross-event slot ' ...
                     'alignment that within-event slot ' ...
                     'exchangeability does not license). The ' ...
                     'attribute is treated as order 0 (passed ' ...
                     'through with leading-event drop). For voice-' ...
                     'leading or step-size analyses, encode each ' ...
                     'voice as a K_a = 1 attribute and difference ' ...
                     'those. Subsequent multi-slot attributes in ' ...
                     'this call are silenced.'], ...
                    a, K_a, ordersPerAttr(a));
            warnedMultiSlot = true;
        end
        ordersPerAttr(a) = 0;
    end
end
ordersPerAttr = int32(ordersPerAttr);

% --- Compute maxOrder, output event count ---
maxOrder = max(ordersPerAttr);
if isempty(maxOrder)
    maxOrder = int32(0);
end
if nvArgs.circular
    nPrime = nEvents;
    if double(maxOrder) >= nEvents
        error('differenceEvents:orderTooHigh', ...
              ['Differencing order too high for circular mode: ' ...
               'max order = %d but N = %d (need max order < N).'], ...
               double(maxOrder), nEvents);
    end
else
    nPrime = nEvents - double(maxOrder);
    if nPrime < 1
        error('differenceEvents:orderTooHigh', ...
              ['Differencing orders are too high for the input event ' ...
               'count: max order = %d but N = %d.'], ...
              double(maxOrder), nEvents);
    end
end

% --- Difference each attribute's value matrix ---
pAttrDiff = cell(1, A);
for a = 1:A
    k = double(ordersPerAttr(a));
    Md = pAttr{a};
    if nvArgs.circular
        % Cyclic differencing: each pass uses prev(n) = mod(n-2, N) + 1,
        % which keeps the output length at N. Implemented as
        % Md - Md(:, [end, 1:end-1]).
        for step = 1:k
            Md = Md - Md(:, [end, 1:end-1]);
        end
        % No leading-event drop needed: every attribute already has
        % nPrime = N columns.
    else
        for step = 1:k
            Md = Md(:, 2:end) - Md(:, 1:end-1);
        end
        extraDrop = double(maxOrder) - k;
        if extraDrop > 0
            Md = Md(:, extraDrop + 1:end);
        end
    end
    pAttrDiff{a} = Md;
end

% --- Transform weights ---
wDiff = localDifferenceWeights(w, A, ordersPerAttr, nEvents, nPrime, ...
                               nvArgs.circular);

% --- Groups unchanged ---
groupsDiff = groups;

end


% =========================================================================
%  localCanonicaliseDiffOrders — Option C parsing → 1xA per-attribute
% =========================================================================

function ordersPerAttr = localCanonicaliseDiffOrders( ...
    diffOrders, A, G, attrsOfGroup)
%LOCALCANONICALISEDIFFORDERS  Coerce diffOrders to a 1 x A row vector.

    if iscell(diffOrders)
        ordersPerAttr = localCanonicaliseDiffOrdersCell( ...
            diffOrders, A, G, attrsOfGroup);
        localValidateOrders(ordersPerAttr);
        return;
    end
    if ~isnumeric(diffOrders)
        error('differenceEvents:badDiffOrdersType', ...
              ['diffOrders must be numeric or a 1-by-G cell; ' ...
               'got class %s.'], class(diffOrders));
    end

    v = double(diffOrders);
    n = numel(v);
    if n == 1
        % Scalar: broadcast.
        ordersPerAttr = v(1) * ones(1, A);
    elseif n == A
        % Length-A: per-attribute. (Also handles A == G case.)
        ordersPerAttr = v(:).';
    elseif n == G
        % Length-G: per-group, broadcast within group.
        ordersPerAttr = zeros(1, A);
        v = v(:).';
        for g = 1:G
            attrs = attrsOfGroup{g};
            ordersPerAttr(attrs) = v(g);
        end
    else
        error('differenceEvents:badDiffOrdersLength', ...
              ['diffOrders has %d entries; expected scalar (1), ' ...
               'per-attribute (A = %d), or per-group (G = %d).'], ...
              n, A, G);
    end
    localValidateOrders(ordersPerAttr);
end


function ordersPerAttr = localCanonicaliseDiffOrdersCell( ...
    c, A, G, attrsOfGroup)
%LOCALCANONICALISEDIFFORDERSCELL  Process the per-group cell form.

    sz = size(c);
    if numel(sz) ~= 2 || sz(1) ~= 1 || sz(2) ~= G
        error('differenceEvents:badDiffOrdersShape', ...
              ['diffOrders cell array must be 1-by-G = 1-by-%d; ' ...
               'got shape %d-by-%d.'], G, sz(1), sz(2));
    end
    ordersPerAttr = zeros(1, A);
    for g = 1:G
        val = c{g};
        attrs = attrsOfGroup{g};
        if isempty(val)
            % Skip group → order 0 (default).
            continue;
        end
        if ~isnumeric(val)
            error('differenceEvents:badDiffOrdersShape', ...
                  'diffOrders{%d} must be numeric or empty; got %s.', ...
                  g, class(val));
        end
        vec = double(val(:).');
        n_g = numel(attrs);
        if isscalar(vec)
            ordersPerAttr(attrs) = vec;
        elseif numel(vec) == n_g
            ordersPerAttr(attrs) = vec;
        else
            error('differenceEvents:badDiffOrdersShape', ...
                  ['diffOrders{%d} has %d entries; expected scalar ' ...
                   'or n_g = %d.'], g, numel(vec), n_g);
        end
    end
end


function localValidateOrders(ordersPerAttr)
    if any(ordersPerAttr < 0) || any(ordersPerAttr ~= round(ordersPerAttr))
        error('differenceEvents:badDiffOrders', ...
              'All entries of diffOrders must be non-negative integers.');
    end
end


% =========================================================================
%  localDifferenceWeights — weight transformation
% =========================================================================

function wOut = localDifferenceWeights(w, A, ordersPerAttr, nEvents, nPrime, circular)
%LOCALDIFFERENCEWEIGHTS Transform weights under per-attribute orders.
%
%  Each differenced attribute's weights are propagated via a rolling
%  product of width k_a + 1. In non-circular mode, pass-through
%  attributes (k_a = 0) have their leading events dropped to match the
%  common output grid. In circular mode the rolling product wraps at
%  the event-sequence boundary and pass-through attributes are kept at
%  length N.

    if isempty(w) && ~iscell(w)
        wOut = [];
        return;
    end

    % --- Top-level scalar ---
    if isnumeric(w) && isscalar(w)
        c = double(w);
        ordersDouble = double(ordersPerAttr);
        if all(ordersDouble == ordersDouble(1))
            % Uniform orders — shape preserved as a scalar.
            wOut = c ^ (ordersDouble(1) + 1);
            return;
        end
        % Varying orders — emit a per-attribute cell.
        wOut = cell(1, A);
        for a = 1:A
            wOut{a} = c ^ (ordersDouble(a) + 1);
        end
        return;
    end

    if ~iscell(w)
        error('differenceEvents:badWeightsType', ...
              ['w must be [], a scalar, or a cell array of ' ...
               'per-attribute weight inputs.']);
    end
    if numel(w) ~= A
        error('differenceEvents:badWeightsLength', ...
              'Weight cell must have length A = %d; got length %d.', ...
              A, numel(w));
    end

    maxOrder = max(double(ordersPerAttr));
    wOut = cell(1, A);
    for a = 1:A
        k = double(ordersPerAttr(a));
        wa = w{a};
        if ~localWeightHasEventDep(wa, nEvents, a)
            % Non-event-dependent: rolling product of a constant
            % reduces to raising each entry to power k + 1.
            wOut{a} = localRaiseNoEventDep(wa, k + 1);
            continue;
        end
        % Event-dependent (1 x N row or K_a x N matrix).
        W = double(wa);
        if k > 0
            W = localRollingProduct(W, k + 1, circular);
        end
        if circular
            % No alignment drop in circular mode.
        else
            extraDrop = double(maxOrder) - k;
            if extraDrop > 0
                W = W(:, extraDrop + 1:end);
            end
        end
        assert(size(W, 2) == nPrime);
        wOut{a} = W;
    end
end


function out = localRaiseNoEventDep(wa, p)
%LOCALRAISENOEVENTDEP Raise a non-event-dependent weight input to
%power p, shape-preserving. The non-event-dependent inputs reaching
%this helper are [], scalars, or K_a x 1 column broadcasts.
    if isempty(wa)
        out = wa;
        return;
    end
    if p == 1
        out = wa;  % fast path: order 0 attribute
        return;
    end
    out = double(wa) .^ double(p);
end


function tf = localWeightHasEventDep(wa, N, attrIdx)
    % True iff wa's shape carries the N axis. Accepts [], scalar,
    % K_a x 1 column (broadcast per slot), 1 x N row, or K_a x N
    % matrix.
    if isempty(wa)
        tf = false;
        return;
    end
    if ~isnumeric(wa)
        error('differenceEvents:badWeightType', ...
              'Attribute %d weight must be numeric.', attrIdx);
    end
    if isscalar(wa)
        tf = false;
        return;
    end
    sz = size(wa);
    if numel(sz) == 2 && sz(2) == N
        % 1 x N row or K_a x N matrix.
        tf = true;
        return;
    end
    if numel(sz) == 2 && sz(2) == 1
        % K_a x 1 column broadcast (no event dependence).
        tf = false;
        return;
    end
    error('differenceEvents:badWeightShape', ...
          ['Attribute %d weight has shape [%s]; expected [], scalar, ' ...
           '[K_a 1] column, [1 %d] row, or [K_a %d] matrix.'], ...
          attrIdx, num2str(sz), N, N);
end


function out = localRollingProduct(W, width, circular)
    % Rolling product of length-N row windows of width *width*.
    % circular = false: output length N - width + 1; window i covers
    % columns i .. i + width - 1.
    % circular = true: output length N; window n covers columns
    % n - width + 1 .. n (wrapped mod N). Indexing matches the
    % differencing operator's prev(n) = mod(n-2, N) + 1 convention,
    % so the product attached to Delta^k p(n) is the product of
    % w(n), w(prev(n)), w(prev(prev(n))), ..., over width entries.
    [K, N] = size(W);
    if circular
        if width > N
            error('differenceEvents:rollingProductWidth', ...
                  ['Circular rolling-product width %d exceeds event ' ...
                   'count %d.'], width, N);
        end
        out = zeros(K, N);
        for n = 1:N
            % Window columns: prev^0(n), prev^1(n), ..., prev^{width-1}(n)
            % = mod(n - 1, N) + 1, mod(n - 2, N) + 1, ..., mod(n - width, N) + 1.
            idx = mod(n - (1:width), N) + 1;
            out(:, n) = prod(W(:, idx), 2);
        end
    else
        nOut = N - width + 1;
        if nOut < 1
            error('differenceEvents:rollingProductWidth', ...
                  'Rolling-product width %d exceeds event count %d.', ...
                  width, N);
        end
        out = zeros(K, nOut);
        for i = 1:nOut
            out(:, i) = prod(W(:, i:i + width - 1), 2);
        end
    end
end


function [groupOfAttr, attrsOfGroup, G] = localCanonicaliseGroups(groups, A)
    % Return:
    %   groupOfAttr  — 1 x A vector of 1-indexed group labels.
    %   attrsOfGroup — 1 x G cell, attrsOfGroup{g} is the column
    %                  vector of attribute indices in group g.
    %   G            — number of groups.
    if isempty(groups)
        groupOfAttr = 1:A;
    elseif iscell(groups)
        G = numel(groups);
        groupOfAttr = zeros(1, A);
        for g = 1:G
            idx = groups{g};
            if any(idx < 1) || any(idx > A) || any(groupOfAttr(idx) ~= 0)
                error('differenceEvents:badGroups', ...
                      'Invalid cell-form groups specification.');
            end
            groupOfAttr(idx) = g;
        end
        if any(groupOfAttr == 0)
            error('differenceEvents:badGroups', ...
                  'Every attribute must appear in exactly one group.');
        end
    elseif isnumeric(groups) && numel(groups) == A
        groupOfAttr = double(groups(:).');
        if any(groupOfAttr < 1) || any(groupOfAttr ~= round(groupOfAttr))
            error('differenceEvents:badGroups', ...
                  'Numeric groups must be positive integers.');
        end
    else
        error('differenceEvents:badGroupsShape', ...
              ['groups must be [], a length-A numeric vector, or a ' ...
               'cell of index lists.']);
    end
    G = max(groupOfAttr);
    attrsOfGroup = cell(1, G);
    for g = 1:G
        attrsOfGroup{g} = find(groupOfAttr == g);
    end
end

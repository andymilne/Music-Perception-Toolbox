function [pAttrBound, wBound, groupsBound] = bindEvents(pAttr, w, groups, bindOrders, nvArgs)
%BINDEVENTS Bind sliding windows of consecutive events into super-attributes.
%
%   [pAttrBound, wBound, groupsBound] = bindEvents(pAttr, w, groups, bindOrders, 'circular', false)
%   is a cross-event preprocessing helper for multi-attribute tensor
%   input. It takes the (pAttr, w, groups) triple that one would
%   otherwise feed to buildExpTens and returns a transformed
%   (pAttrBound, wBound, groupsBound) triple ready to chain into
%   another pre-MAET operation or into buildExpTens.
%
%   Bind orders are specified per attribute (via Option C syntax —
%   see below). For an input attribute a with bind order L_a, a
%   sliding window of width L_a is laid across the event axis and
%   each lag in the window is emitted as a separate output super-
%   attribute. The total output attribute count is A' = sum_a L_a;
%   each input attribute contributes L_a super-attributes to the
%   output, all in the same group as the source attribute. Lag
%   identity is non-exchangeable, so the L_a copies are emitted as
%   separate attributes rather than packed into a multi-slot one.
%   The original K_a slot structure of each input attribute is
%   preserved in every super-attribute.
%
%   Event-axis alignment. The natural output event count of an
%   attribute with bind order L_a is N - L_a + 1 (non-circular) or N
%   (circular). With per-attribute orders, the common output event
%   count is N' = N - max_a L_a + 1 (non-circular) or N (circular);
%   attributes with L_a < max_a L_a have their trailing
%   max_a L_a - L_a super-events dropped to align all attributes on
%   the same output grid. This is the natural composition partner of
%   differenceEvents' leading-drop alignment: D then B gives the same
%   output (super-attribute by super-attribute) as B then D, for any
%   choice of per-attribute orders.
%
%   Weights. Each output super-attribute inherits the slot weights of
%   the underlying input event at its lag, propagated under the
%   toolbox's standard broadcast convention. buildExpTens then
%   multiplies across attributes during tuple enumeration, so the
%   end-to-end weight of a bound super-event equals the product of
%   the L_a constituent events' weights — the natural pre-MAET
%   factoring of the rolling product.
%
%   Inputs
%       pAttr      - 1 x A cell array of K_a x N per-attribute value
%                    matrices. K_a >= 1; K_a = 0 (empty attribute)
%                    is rejected.
%       w          - Weights. [], scalar, or 1 x A cell of per-attribute
%                    weight inputs (each [], scalar, 1 x N row, K_a x 1
%                    column, or K_a x N matrix). Same convention as
%                    buildExpTens.
%       groups     - Group assignment. [] (each attribute its own group),
%                    a 1 x A index vector, or a 1 x G cell of attribute-
%                    index lists. Matches buildExpTens.
%       bindOrders - Per-attribute or per-group bind orders (positive
%                    integers, >= 1). L = 1 is the no-op (each input
%                    event becomes a one-event super-event = itself).
%                    Option C syntax:
%
%                      scalar              broadcast to all attributes
%                      1 x A or A x 1      per-attribute
%                      1 x G or G x 1      per-group, broadcast within
%                                          group (G ~= A; the A == G
%                                          case is read as per-attribute,
%                                          identical output)
%                      1 x G cell          per-group, with each cell:
%                                            [] (skip → 1),
%                                            scalar (broadcast in group),
%                                            length-n_g vector
%                                            (per-attribute in group).
%
%   Name-value pairs
%       'circular'  - false (default) or true. When true, the sliding
%                     window wraps around the event axis and N' = N
%                     regardless of L_a.
%
%   Outputs
%       pAttrBound  - 1 x A' cell of super-attribute value matrices,
%                     each K_a x N', where A' = sum_a L_a.
%       wBound      - Transformed weights. Shape mirrors w: [] stays
%                     []; a scalar stays a scalar; a 1 x A cell becomes
%                     a 1 x A' cell with each super-attribute carrying
%                     the lag-indexed slice (event-dependent weights)
%                     or the inherited non-event-dependent input
%                     (scalar / [] / K_a x 1 column).
%       groupsBound - 1 x A' numeric vector of group labels. Each input
%                     attribute's L_a super-attributes are placed in
%                     the same group as the source attribute (the group
%                     count is unchanged; group membership expands).
%
%   See also BUILDEXPTENS, DIFFERENCEEVENTS, TRANSLATEATTRIBUTES, WEIGHTEVENTS.

arguments
    pAttr
    w
    groups
    bindOrders
    nvArgs.circular (1, 1) logical = false
end

% --- Normalise pAttr to a cell of 2-D double matrices ---
if ~iscell(pAttr)
    error('bindEvents:badPAttrType', ...
          'pAttr must be a cell array of per-attribute matrices.');
end
A = numel(pAttr);
if A < 1
    error('bindEvents:noAttrs', ...
          'pAttr must contain at least one attribute.');
end
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

% --- Canonicalise groups: 1xA assignment, attrsOfGroup, G ---
[groupOfAttr, attrsOfGroup, G] = localCanonicaliseGroups(groups, A);

% --- Parse bindOrders via Option C → ordersPerAttr (1xA) ---
ordersPerAttr = localCanonicaliseBindOrders( ...
    bindOrders, A, G, attrsOfGroup);

% --- Compute output sizes ---
maxOrder = max(ordersPerAttr);
if nvArgs.circular
    nPrime = nEvents;
    if maxOrder > nEvents
        error('bindEvents:windowTooLarge', ...
              ['Circular window size max L = %d exceeds event ' ...
               'count N = %d.'], double(maxOrder), nEvents);
    end
else
    nPrime = nEvents - double(maxOrder) + 1;
    if nPrime < 1
        error('bindEvents:windowTooLarge', ...
              ['Bind orders too high for the input event count: ' ...
               'max L = %d but N = %d (non-circular).'], ...
              double(maxOrder), nEvents);
    end
end

% --- Total output attribute count ---
A_prime = sum(double(ordersPerAttr));

% --- Build output value matrices and group labels ---
pAttrBound  = cell(1, A_prime);
groupsBound = zeros(1, A_prime);
outIdx = 0;
for a = 1:A
    L_a = double(ordersPerAttr(a));
    g_a = groupOfAttr(a);
    Marr = pAttr{a};   % K_a x N

    for ell = 0:(L_a - 1)
        outIdx = outIdx + 1;
        if nvArgs.circular
            indices = mod((0:nPrime - 1) + ell, nEvents) + 1;
        else
            indices = (ell + 1):(ell + nPrime);
        end
        pAttrBound{outIdx} = Marr(:, indices);
        groupsBound(outIdx) = g_a;
    end
end

% --- Transform weights ---
wBound = localBindWeights(w, A, ordersPerAttr, nEvents, nPrime, ...
                          nvArgs.circular);

end


% =========================================================================
%  localCanonicaliseBindOrders — Option C parsing → 1xA per-attribute
% =========================================================================

function ordersPerAttr = localCanonicaliseBindOrders( ...
    bindOrders, A, G, attrsOfGroup)
%LOCALCANONICALISEBINDORDERS  Coerce bindOrders to a 1 x A row vector.

    if iscell(bindOrders)
        ordersPerAttr = localCanonicaliseBindOrdersCell( ...
            bindOrders, A, G, attrsOfGroup);
        localValidateOrders(ordersPerAttr);
        return;
    end
    if ~isnumeric(bindOrders)
        error('bindEvents:badBindOrdersType', ...
              ['bindOrders must be numeric or a 1-by-G cell; ' ...
               'got class %s.'], class(bindOrders));
    end

    v = double(bindOrders);
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
        error('bindEvents:badBindOrdersLength', ...
              ['bindOrders has %d entries; expected scalar (1), ' ...
               'per-attribute (A = %d), or per-group (G = %d).'], ...
              n, A, G);
    end
    localValidateOrders(ordersPerAttr);
end


function ordersPerAttr = localCanonicaliseBindOrdersCell( ...
    c, A, G, attrsOfGroup)
%LOCALCANONICALISEBINDORDERSCELL  Process the per-group cell form.

    sz = size(c);
    if numel(sz) ~= 2 || sz(1) ~= 1 || sz(2) ~= G
        error('bindEvents:badBindOrdersShape', ...
              ['bindOrders cell array must be 1-by-G = 1-by-%d; ' ...
               'got shape %d-by-%d.'], G, sz(1), sz(2));
    end
    ordersPerAttr = ones(1, A);   % default: skipped groups get L = 1 (no-op)
    for g = 1:G
        val = c{g};
        attrs = attrsOfGroup{g};
        if isempty(val)
            % Skip group → L = 1 (no-op, no super-attr expansion).
            continue;
        end
        if ~isnumeric(val)
            error('bindEvents:badBindOrdersShape', ...
                  'bindOrders{%d} must be numeric or empty; got %s.', ...
                  g, class(val));
        end
        vec = double(val(:).');
        n_g = numel(attrs);
        if isscalar(vec)
            ordersPerAttr(attrs) = vec;
        elseif numel(vec) == n_g
            ordersPerAttr(attrs) = vec;
        else
            error('bindEvents:badBindOrdersShape', ...
                  ['bindOrders{%d} has %d entries; expected scalar ' ...
                   'or n_g = %d.'], g, numel(vec), n_g);
        end
    end
end


function localValidateOrders(ordersPerAttr)
    if any(ordersPerAttr < 1) || any(ordersPerAttr ~= round(ordersPerAttr))
        error('bindEvents:badBindOrders', ...
              ['All entries of bindOrders must be positive integers ' ...
               '(>= 1; L = 1 is the no-op).']);
    end
end


% =========================================================================
%  localBindWeights — weight transformation
% =========================================================================

function wOut = localBindWeights(w, A, ordersPerAttr, nEvents, nPrime, isCircular)
%LOCALBINDWEIGHTS Transform weights under per-attribute binding.
%
%  Each output super-attribute carries the slot weights of the input
%  event at its lag. Non-event-dependent inputs (None, scalar, K_a x 1
%  column) are inherited as-is by every super-attribute; the kernel
%  product over the L_a super-attributes in buildExpTens recovers the
%  rolling product naturally. Event-dependent inputs (1 x N row,
%  K_a x N matrix) are sliced into the output's lag-indexed columns.

    if isempty(w) && ~iscell(w)
        wOut = [];
        return;
    end

    % --- Top-level scalar ---
    if isnumeric(w) && isscalar(w)
        % Inherited by every super-attribute via broadcast.
        wOut = double(w);
        return;
    end

    if ~iscell(w)
        error('bindEvents:badWeightsType', ...
              ['w must be [], a scalar, or a cell array of ' ...
               'per-attribute weight inputs.']);
    end
    if numel(w) ~= A
        error('bindEvents:badWeightsLength', ...
              'Weight cell must have length A = %d; got length %d.', ...
              A, numel(w));
    end

    A_prime = sum(double(ordersPerAttr));
    wOut = cell(1, A_prime);
    outIdx = 0;
    for a = 1:A
        L_a = double(ordersPerAttr(a));
        wa = w{a};
        K_a = NaN;  % to be inferred if needed
        eventDep = localWeightHasEventDep(wa, nEvents, a);

        for ell = 0:(L_a - 1)
            outIdx = outIdx + 1;
            if ~eventDep
                % Non-event-dependent: each super-attr inherits the same input.
                wOut{outIdx} = wa;
                continue;
            end
            % Event-dependent (1 x N row or K_a x N matrix): slice by lag.
            W = double(wa);
            if isCircular
                indices = mod((0:nPrime - 1) + ell, nEvents) + 1;
            else
                indices = (ell + 1):(ell + nPrime);
            end
            wOut{outIdx} = W(:, indices);
        end
    end
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
        error('bindEvents:badWeightType', ...
              'Attribute %d weight must be numeric.', attrIdx);
    end
    if isscalar(wa)
        tf = false;
        return;
    end
    sz = size(wa);
    if numel(sz) == 2 && sz(2) == N
        tf = true;
        return;
    end
    if numel(sz) == 2 && sz(2) == 1
        tf = false;
        return;
    end
    error('bindEvents:badWeightShape', ...
          ['Attribute %d weight has shape [%s]; expected [], scalar, ' ...
           '[K_a 1] column, [1 %d] row, or [K_a %d] matrix.'], ...
          attrIdx, num2str(sz), N, N);
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
                error('bindEvents:badGroups', ...
                      'Invalid cell-form groups specification.');
            end
            groupOfAttr(idx) = g;
        end
        if any(groupOfAttr == 0)
            error('bindEvents:badGroups', ...
                  'Every attribute must appear in exactly one group.');
        end
    elseif isnumeric(groups) && numel(groups) == A
        groupOfAttr = double(groups(:).');
        if any(groupOfAttr < 1) || any(groupOfAttr ~= round(groupOfAttr))
            error('bindEvents:badGroups', ...
                  'Numeric groups must be positive integers.');
        end
    else
        error('bindEvents:badGroupsShape', ...
              ['groups must be [], a length-A numeric vector, or a ' ...
               'cell of index lists.']);
    end
    G = max(groupOfAttr);
    attrsOfGroup = cell(1, G);
    for g = 1:G
        attrsOfGroup{g} = find(groupOfAttr == g);
    end
end

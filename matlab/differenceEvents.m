function [pAttrDiff, wDiff] = differenceEvents(pAttr, w, groups, diffOrders, periods)
%DIFFERENCEEVENTS Replace selected groups' event sequences with differences.
%
%   [pAttrDiff, wDiff] = differenceEvents(pAttr, w, groups, diffOrders, periods)
%   is a cross-event preprocessing helper for multi-attribute tensor
%   input. It takes the (pAttr, w) pair that one would otherwise feed
%   to buildExpTens and returns a transformed (pAttrDiff, wDiff) pair
%   with the same shape conventions, in which the event columns of
%   each selected group have been replaced by k-fold inter-event
%   differences. The output feeds directly into buildExpTens.
%
%   See the MAET specification §7 for the full semantics. In brief:
%
%     Values. For a group with diffOrders(g) = k, the value matrix of
%     each attribute in that group is replaced by its k-th finite
%     difference along the event axis, reducing the event count by k.
%     Order 0 leaves a group unchanged. If periods(g) > 0, each raw
%     difference is wrapped to [-P/2, P/2) (shortest-arc convention,
%     matching cosSimExpTens).
%
%     Weights. Weight inputs follow the toolbox's standard broadcast
%     convention: [] broadcasts 1, a scalar broadcasts uniformly,
%     event-dependent inputs (1 x N row, K_a x N matrix) supply
%     per-event values, and a K_a x 1 column broadcasts per slot.
%     The weight of a difference event is the product of the
%     weights of its k+1 constituent input events — a rolling
%     product of width k+1 along the event axis — interpretable as
%     the probability that all constituents are perceived under the
%     standard weights-as-salience reading.
%
%     Event-count alignment. The output event count is
%     N' = N - max_g k_g. Groups with k_g < max_g k_g have their
%     leading max_g k_g - k_g events dropped to keep columns aligned.
%     Weights are dropped to match.
%
%   Inputs
%       pAttr      - 1 x A cell array of K_a x N per-attribute value
%                    matrices, with K_a = 1 for every attribute.
%                    Same convention as buildExpTens but restricted
%                    to the single-slot case: within-event slot
%                    exchangeability does not license the cross-
%                    event slot correspondence that column-wise
%                    differencing imposes, so attributes with
%                    K_a ~= 1 raise differenceEvents:multiSlotAttribute.
%                    For voice-leading or step-size analyses, encode
%                    each voice as its own K_a = 1 attribute in a
%                    shared group, difference that, then (optionally)
%                    stack the differenced attributes into a single
%                    multi-slot attribute before buildExpTens.
%       w          - Weights. [], scalar, or 1 x A cell of per-attribute
%                    weight inputs (each [], scalar, 1 x N row, K_a x 1
%                    column, or K_a x N matrix). Same convention as
%                    buildExpTens.
%       groups     - Group assignment. [] (each attribute its own group),
%                    a 1 x A index vector, or a 1 x G cell of attribute-
%                    index lists. Matches buildExpTens.
%       diffOrders - 1 x G vector of per-group differencing orders
%                    (non-negative integers). Order 0 leaves the group
%                    unchanged.
%       periods    - 1 x G vector of periods for shortest-arc wrapping
%                    of differences. An entry of 0 (or negative) means
%                    the group is non-periodic and differences are
%                    left unwrapped.
%
%   Outputs
%       pAttrDiff  - 1 x A cell of transformed per-attribute matrices,
%                    each K_a x N'.
%       wDiff      - Transformed weights under the rule described
%                    above. Shape mirrors w: [] stays []; a scalar
%                    stays a scalar when all groups share the same
%                    order, expanding to a 1 x A cell of per-
%                    attribute scalars when orders vary; a 1 x A
%                    cell stays a 1 x A cell, with per-attribute
%                    event-dependent entries becoming K_a x N'
%                    matrices and non-event-dependent entries
%                    keeping their input shape.
%
%   See also BUILDEXPTENS.

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
    pAttr{a} = double(M);
end

% --- Enforce K_a = 1 per attribute ---
% Event differencing requires every attribute to have exactly one
% slot per event. Column-wise subtraction across adjacent events
% imposes a cross-event slot correspondence (slot i at event n-1
% paired with slot i at event n) that within-event slot
% exchangeability does not license; for multi-slot attributes the
% output would silently depend on an arbitrary slot-listing choice.
% K_a = 0 (empty attribute) is also rejected. The principled route
% for voice-leading or step-size analyses is to encode each voice
% as its own K_a = 1 attribute in a shared group, difference that,
% then (optionally) stack the differenced attributes into a single
% multi-slot attribute before buildExpTens. See USER_GUIDE §3
% (Event differencing).
for a = 1:A
    K_a = size(pAttr{a}, 1);
    if K_a ~= 1
        error('differenceEvents:multiSlotAttribute', ...
              ['Attribute %d has K_a = %d; event differencing ' ...
               'requires every attribute to have K_a = 1. Column-' ...
               'wise differencing imposes a cross-event slot ' ...
               'alignment that within-event slot exchangeability ' ...
               'does not license, so multi-slot attributes are ' ...
               'rejected; empty attributes (K_a = 0) are rejected ' ...
               'likewise. For voice-leading or step-size analyses, ' ...
               'encode each voice as a separate K_a = 1 attribute ' ...
               'in a shared group, call differenceEvents on that, ' ...
               'then (optionally) stack the differenced attributes ' ...
               'into a single multi-slot attribute before ' ...
               'buildExpTens. See USER_GUIDE Section 3 (Event ' ...
               'differencing).'], a, K_a);
    end
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

% --- Canonicalise groups to a 1 x A index vector ---
groupOfAttr = localCanonicaliseGroups(groups, A);
G = max(groupOfAttr);

% --- Validate diffOrders and periods ---
diffOrders = double(diffOrders(:).');
if numel(diffOrders) ~= G
    error('differenceEvents:badDiffOrdersLength', ...
          'diffOrders must have length G = %d; got length %d.', ...
          G, numel(diffOrders));
end
if any(diffOrders < 0) || any(diffOrders ~= round(diffOrders))
    error('differenceEvents:badDiffOrders', ...
          'All entries of diffOrders must be non-negative integers.');
end
diffOrders = int32(diffOrders);

periods = double(periods(:).');
if numel(periods) ~= G
    error('differenceEvents:badPeriodsLength', ...
          'periods must have length G = %d; got length %d.', ...
          G, numel(periods));
end

maxOrder = max(diffOrders);
nPrime = nEvents - double(maxOrder);
if nPrime < 1
    error('differenceEvents:orderTooHigh', ...
          ['Differencing orders are too high for the input event ' ...
           'count: max(diffOrders) = %d but N = %d.'], ...
          maxOrder, nEvents);
end

% --- Difference each attribute's value matrix ---
pAttrDiff = cell(1, A);
for a = 1:A
    g = groupOfAttr(a);
    k = double(diffOrders(g));
    P = periods(g);
    Md = pAttr{a};
    for step = 1:k
        Md = Md(:, 2:end) - Md(:, 1:end-1);
        if P > 0
            Md = mod(Md + P/2, P) - P/2;
        end
    end
    extraDrop = double(maxOrder) - k;
    if extraDrop > 0
        Md = Md(:, extraDrop + 1:end);
    end
    pAttrDiff{a} = Md;
end

% --- Transform weights ---
wDiff = localDifferenceWeights(w, A, groupOfAttr, diffOrders, ...
    nEvents, nPrime);

end


% =========================================================================
%  localDifferenceWeights — weight transformation under differencing
% =========================================================================

function wOut = localDifferenceWeights(w, A, groupOfAttr, diffOrders, ...
    nEvents, nPrime)
%LOCALDIFFERENCEWEIGHTS Transform weights under the difference-events
%convention.
%
%  The weight of each difference event is the product of the weights
%  of the k + 1 input events on which the difference depends — a
%  rolling product of width k + 1 along the event axis, applied
%  semantically under the toolbox's broadcast convention. Under the
%  K_a = 1 restriction on differenceEvents inputs, valid per-
%  attribute weight inputs are [], scalar, or 1 x N.

    if isempty(w) && ~iscell(w)
        wOut = [];
        return;
    end

    % --- Top-level scalar ---
    if isnumeric(w) && isscalar(w)
        c = double(w);
        ordersPerAttr = zeros(1, A);
        for a = 1:A
            ordersPerAttr(a) = double(diffOrders(groupOfAttr(a)));
        end
        if all(ordersPerAttr == ordersPerAttr(1))
            % Uniform orders — shape preserved as a scalar.
            wOut = c ^ (ordersPerAttr(1) + 1);
            return;
        end
        % Varying orders — emit a per-attribute cell.
        wOut = cell(1, A);
        for a = 1:A
            wOut{a} = c ^ (ordersPerAttr(a) + 1);
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

    maxOrder = max(diffOrders);
    wOut = cell(1, A);
    for a = 1:A
        g = groupOfAttr(a);
        k = double(diffOrders(g));
        wa = w{a};
        if ~localWeightHasEventDep(wa, nEvents, a)
            % No event dependence — rolling product of a constant
            % reduces to raising each entry to power k + 1. [] stays
            % []; a scalar stays a scalar.
            wOut{a} = localRaiseNoEventDep(wa, k + 1);
            continue;
        end
        % Event-dependent: under K_a = 1 the validated shape is 1 x N.
        W = double(wa);
        if k > 0
            W = localRollingProduct(W, k + 1);
        end
        extraDrop = double(maxOrder) - k;
        if extraDrop > 0
            W = W(:, extraDrop + 1:end);
        end
        assert(size(W, 2) == nPrime);
        wOut{a} = W;
    end
end


function out = localRaiseNoEventDep(wa, p)
%LOCALRAISENOEVENTDEP Raise a non-event-dependent weight input to
%power p, shape-preserving. Under the K_a = 1 restriction on
%differenceEvents inputs, the non-event-dependent inputs reaching
%this helper are just [] and scalars.
    if isempty(wa)
        out = wa;
        return;
    end
    if p == 1
        out = wa;  % fast path: order 0 groups
        return;
    end
    out = double(wa) .^ double(p);
end


function tf = localWeightHasEventDep(wa, N, attrIdx)
    % True iff wa's shape carries the N axis. Under the K_a = 1
    % restriction on differenceEvents inputs, valid per-attribute
    % weight shapes are [], scalar, or 1 x N.
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
    if isequal(size(wa), [1 N])
        tf = true;
        return;
    end
    error('differenceEvents:badWeightShape', ...
          ['Attribute %d weight has shape [%s]; under the K_a = 1 ' ...
           'restriction, expected [], scalar, or [1 %d].'], ...
          attrIdx, num2str(size(wa)), N);
end


function out = localRollingProduct(W, width)
    % Rolling product of width *width* along columns.
    [K, N] = size(W);
    nOut = N - width + 1;
    if nOut < 1
        error('differenceEvents:rollingProductWidth', ...
              'Rolling-product width %d exceeds event count %d.', width, N);
    end
    out = zeros(K, nOut);
    for i = 1:nOut
        out(:, i) = prod(W(:, i:i + width - 1), 2);
    end
end


function groupOfAttr = localCanonicaliseGroups(groups, A)
    % Return a 1 x A vector of 1-indexed group labels.
    if isempty(groups)
        groupOfAttr = 1:A;
        return;
    end
    if iscell(groups)
        % Cell of attribute-index lists, one per group.
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
        return;
    end
    if isnumeric(groups) && numel(groups) == A
        groupOfAttr = double(groups(:).');
        if any(groupOfAttr < 1) || any(groupOfAttr ~= round(groupOfAttr))
            error('differenceEvents:badGroups', ...
                  'Numeric groups must be positive integers.');
        end
        return;
    end
    error('differenceEvents:badGroupsShape', ...
          'groups must be [], a length-A numeric vector, or a cell of index lists.');
end

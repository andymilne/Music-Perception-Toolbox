function [pAttrDiff, wDiff, specs] = differenceEvents(pAttr, w, diffOrders, nvArgs)
%DIFFERENCEEVENTS Replace event sequences with inter-event differences.
%
%   [pAttrDiff, wDiff, specs] = differenceEvents(pAttr, w, diffOrders, ...)
%   is cross-event preprocessing on the canonical (pAttr, w, specs) triple.
%   The k_a-th finite difference is applied along the event axis to each
%   attribute; the returned (pAttrDiff, wDiff, specs) chains into another
%   pre-MAET operation or into buildExpTens(..., 'specs', specs).
%
%   Differencing pairs values position by position: event i's value at
%   position k differences against event
%   i+1's value at position k. This is well-defined exactly when the positions
%   have stable
%   identity --- an ordered attribute ([sym] = 0) or a singleton (K = 1).
%   A symmetric multiset (K > 1, [sym] = 1) is a bag with no positional
%   correspondence, so differencing it is undefined and errors. The rule
%   extends per level for a nested attribute: every level must be ordered
%   (or of size 1). Ragged ordered data (events of differing length) is
%   represented by NaN-padding to a common K; a difference touching a NaN
%   value is NaN, so absence propagates rather than fabricating an interval.
%
%   Differencing changes values only; the spec (tags, r, sym, rel) passes
%   through unchanged. Output values are raw; periodic wrapping is the
%   kernel's job in buildExpTens.
%
%   Inputs
%       pAttr     - 1 x A cell of K_a x N per-attribute value matrices.
%       w         - Weights ([], scalar, or 1 x A cell); rolling product
%                   over the k_a + 1 constituent events per differenced
%                   attribute.
%       diffOrders- Scalar or 1 x A non-negative differencing orders.
%
%   Name-value pairs
%       'circular' - false (default) or true (wrap; N' = N).
%       'specs'    - [] (synthesise flat via flatSpecs) or a 1 x A cell of
%                    per-attribute specs. The ordered-or-singleton guard
%                    reads [sym] from here; the specs pass through unchanged.
%
%   Outputs
%       pAttrDiff - 1 x A cell of differenced matrices, each K_a x N'.
%       wDiff     - Transformed weights.
%       specs     - The attribute specifications, unchanged from input (or synthesised).
%
%   See also BUILDEXPTENS, BINDEVENTS, FLATSPECS, TRANSLATEATTRIBUTES.

arguments
    pAttr
    w
    diffOrders
    nvArgs.circular (1, 1) logical = false
    nvArgs.specs = []
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
K = zeros(1, A);
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
    K(a) = size(pAttr{a}, 1);
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

% --- Attribute specifications: synthesise flat if none supplied ---
if isempty(nvArgs.specs)
    specs = flatSpecs(pAttr);
else
    specs = nvArgs.specs;
    if ~iscell(specs) || numel(specs) ~= A
        error('differenceEvents:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end
end

% --- Parse diffOrders (scalar or 1 x A) ---
ordersPerAttr = localCanonicaliseDiffOrders(diffOrders, A);

% --- Ordered-or-singleton guard for each differenced attribute ---
for a = 1:A
    if double(ordersPerAttr(a)) > 0
        localCheckDifferenceable(specs{a}, K(a), a);
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
              ['Differencing order too high for circular mode: max order ' ...
               '= %d but N = %d (need max order < N).'], ...
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

% --- Difference each attribute's value matrix (position by position; NaN propagates) ---
pAttrDiff = cell(1, A);
for a = 1:A
    k = double(ordersPerAttr(a));
    Md = pAttr{a};
    if nvArgs.circular
        for step = 1:k
            Md = Md - Md(:, [end, 1:end-1]);
        end
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

% --- specs pass through unchanged ---

end


function localCheckDifferenceable(spec, K_a, a)
%LOCALCHECKDIFFERENCEABLE  An attribute is differenceable only if its positions
%   have stable identity across events --- ordered ([sym] = 0) or singleton
%   at every level. A symmetric multiset of size > 1 is a bag with no positional
%   correspondence, so differencing it is undefined.
    if isstruct(spec) && isfield(spec, 'tags')
        tags = double(spec.tags(:).');
        if isfield(spec, 'sym') && ~isempty(spec.sym)
            sym = logical(spec.sym(:).');
        else
            sym = true(1, 2);
        end
        nGroups = numel(unique(tags));           % outer level size
        innerSz = numel(tags) / max(nGroups, 1); % values per group
        innerOk = (~sym(1)) || innerSz == 1;
        outerOk = (~sym(end)) || nGroups == 1;
        if ~(innerOk && outerOk)
            if ~innerOk, bad = 'inner'; else, bad = 'outer'; end
            error('differenceEvents:notDifferenceable', ...
                  ['attribute %d: differencing requires every level ordered ' ...
                   '(or of size 1); the %s level is symmetric with size > 1. ' ...
                   'Set that level''s [sym] = 0 to difference it.'], a, bad);
        end
    else
        if isstruct(spec) && isfield(spec, 'sym')
            sym = logical(spec.sym);
        else
            sym = true;
        end
        if sym && K_a > 1
            error('differenceEvents:notDifferenceable', ...
                  ['attribute %d: differencing requires an ordered attribute ' ...
                   '([sym] = 0) or K = 1; got a symmetric multiset with K = ' ...
                   '%d. A symmetric multiset is a bag with no positional ' ...
                   'correspondence across events. Set [sym] = 0 (e.g. via ' ...
                   'flatSpecs(..., ''sym'', false)) to difference it.'], a, K_a);
        end
    end
end


function orders = localCanonicaliseDiffOrders(diffOrders, A)
%LOCALCANONICALISEDIFFORDERS  Coerce to a 1 x A row (scalar or per-attr).
    if ~isnumeric(diffOrders)
        error('differenceEvents:badDiffOrdersType', ...
              'diffOrders must be numeric; got class %s.', class(diffOrders));
    end
    v = double(diffOrders(:).');
    n = numel(v);
    if n == 1
        orders = v(1) * ones(1, A);
    elseif n == A
        orders = v;
    else
        error('differenceEvents:badDiffOrdersLength', ...
              ['diffOrders has %d entries; expected scalar (1) or ' ...
               'per-attribute (A = %d).'], n, A);
    end
    if any(orders < 0) || any(orders ~= round(orders))
        error('differenceEvents:badDiffOrders', ...
              'All entries of diffOrders must be non-negative integers.');
    end
end


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
    % K_a x 1 column (broadcast per value), 1 x N row, or K_a x N
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

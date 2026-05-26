function pAttrTranslated = translateAttributes(pAttr, groups, offsets, isRel, isPer, periods)
%TRANSLATEATTRIBUTES Translate selected attributes' values by a chosen offset.
%
%   pAttrTranslated = translateAttributes(pAttr, groups, offsets, isRel, isPer, periods)
%   is a per-attribute preprocessing helper for multi-attribute tensor
%   input. It takes the pAttr list one would otherwise feed to
%   buildExpTens and returns a transformed pAttrTranslated list with the
%   same shape conventions, in which selected attributes' values have
%   been shifted by a chosen offset. The output feeds directly into
%   buildExpTens without any further massaging.
%
%   Sliding-comparison context. translateAttributes is the pre-tensor route
%   to a sliding comparison along one or more attribute axes: for each
%   candidate offset mu on a sweep grid, translate the events and
%   compute a similarity against an un-shifted reference. The post-
%   tensor counterpart is windowedSimilarity. Both answer related "slide
%   along an axis" questions but with different operational properties;
%   see USER_GUIDE.md §3.1.
%
%   Two offset-input forms are accepted: a single numeric block (for
%   uniform broadcast or fully per-attribute layouts) or a 1-by-G cell
%   array keyed by group (for mixed-per-group layouts, e.g. broadcast on
%   one group and per-attribute on another in the same call).
%
%     Numeric form. offsets is a numeric matrix with rows indexing
%     attributes and columns indexing sweep positions. Row count must
%     be exactly 1 (broadcast across all attributes) or A (per-
%     attribute). Within-group broadcast is expressed by setting that
%     group's rows equal.
%
%       1-by-1 scalar:        broadcast no sweep. Single translation.
%       1-by-M row (M >= 1):  broadcast M-sweep. Returns 1-by-M cell.
%       A-by-1 column:        per-attribute, single translation.
%                             Returns 1-by-1 cell (matrix-mode).
%       A-by-M matrix:        per-attribute, M-sweep. Returns 1-by-M
%                             cell.
%       G-by-1 column:        per-group, broadcast within group;
%                             single translation. Returns 1-by-1 cell
%                             (matrix-mode), consistent with A-by-1.
%                             Requires A ~= G to disambiguate from
%                             per-attribute (when A == G the per-
%                             attribute reading applies, with
%                             identical numeric output and identical
%                             matrix-mode wrapping).
%       G-by-M matrix:        per-group, broadcast within group;
%                             M-sweep. Same A ~= G requirement.
%       2-D with rows not in {1, A, G}: error.
%
%     Cell form. offsets is a 1-by-G cell array; each cell offsets{g}
%     holds the per-group value following an analogous orientation
%     convention with n_g (the number of attributes in group g)
%     playing the role of A:
%
%       empty []:             skip group (no translation).
%       1-by-1 scalar:        broadcast within group, no sweep.
%       1-by-M row:           broadcast within group, M-sweep.
%       n_g-by-1 column:      per-attribute, no sweep.
%       n_g-by-M matrix:      per-attribute, M-sweep.
%       2-D with rows neither 1 nor n_g: error.
%
%     Groups whose cell is empty (or omitted by passing a shorter
%     cell — disallowed; the cell must be exactly 1-by-G) are not
%     translated. All entries (across both numeric columns and cell
%     entries) implying M > 1 must agree on M; scalar and one-column
%     entries broadcast across the sweep. When any entry implies a
%     sweep or any 2-D shape is used, the output is a 1-by-M cell of
%     1-by-A cells (matrix-mode); otherwise a single 1-by-A cell.
%
%   NaN entries in any numeric block skip the corresponding (attribute,
%   column) cell. +/-Inf is rejected.
%
%   Semantics by group geometry (apply per offset column):
%
%     Absolute non-periodic (isPer(g) = false): every value of every
%     attribute in group g is replaced by value + mu. periods(g) is
%     ignored regardless of its sign.
%
%     Absolute periodic (isPer(g) = true with isRel(g) = false and
%     periods(g) > 0): every value is replaced by value + mu,
%     unwrapped. The wrapped periodic Gaussian kernel of buildExpTens
%     is invariant under any additive shift by a multiple of P, so no
%     canonical wrap of the translated values is required --- the
%     kernel handles periodicity downstream.
%
%     Relative (isRel(g) = true): a uniform shift of every value
%     cancels in every within-tuple difference, so translation on a
%     relative group is a structural no-op. The group is left
%     unchanged. The warning translateAttributes:noOpRelative is emitted
%     at most once per call, even when multiple sweep columns carry
%     finite entries on the relative-group attributes.
%
%   Weights are unaffected by translation and are not part of this
%   function's signature; the caller passes the same w to buildExpTens
%   after translation that they would have passed without it.
%
%   Inputs
%       pAttr    - 1 x A cell of K_a x N per-attribute value matrices.
%                  Same convention as buildExpTens. A 1-D row vector is
%                  taken as a 1 x N row.
%       groups   - Group assignment, same convention as buildExpTens
%                  and differenceEvents.
%       offsets  - Numeric matrix or 1-by-G cell. See the "Two offset-
%                  input forms" block above.
%       isRel    - 1 x G logical vector. Groups with isRel(g) = true
%                  carrying any finite offset on their attributes emit
%                  a single translateAttributes:noOpRelative warning and
%                  pass through unchanged on every column.
%       isPer    - 1 x G logical vector. Accepted for signature
%                  parallelism with the rest of the MAET pipeline; not
%                  consulted by translateAttributes itself.
%       periods  - 1 x G numeric vector. Accepted for signature
%                  parallelism; not consulted by translateAttributes
%                  itself.
%
%   Output
%       pAttrTranslated - 1 x A cell of K_a x N per-attribute value
%                         matrices in vector mode (numeric scalar input
%                         only); a 1 x M cell of 1 x A cells in matrix
%                         mode (any other input).
%
%   Errors
%       translateAttributes:badPAttr           - pAttr is malformed.
%       translateAttributes:badEventCount      - inconsistent N across attrs.
%       translateAttributes:wrongIsRelLength   - isRel has length != G.
%       translateAttributes:wrongIsPerLength   - isPer has length != G.
%       translateAttributes:wrongPeriodsLength - periods has length != G.
%       translateAttributes:wrongOffsetsShape  - offsets not a valid shape.
%       translateAttributes:nonFiniteOffset    - an offset is +/-Inf.
%
%   Warnings
%       translateAttributes:noOpRelative - emitted once per call when any
%                                       relative-group attribute has any
%                                       finite offset entry.
%
%   See also DIFFERENCEEVENTS, BINDEVENTS, BUILDEXPTENS, WINDOWEDSIMILARITY, COSSIMEXPTENS.

% --- Normalise pAttr to a cell of 2-D double matrices ---
if isnumeric(pAttr)
    pAttr = {pAttr};
end
if ~iscell(pAttr) || isempty(pAttr)
    error('translateAttributes:badPAttr', ...
          'pAttr must be a non-empty cell of value matrices.');
end
A = numel(pAttr);
for a = 1:A
    Marr = pAttr{a};
    if isvector(Marr) && (size(Marr, 1) == 1 || size(Marr, 2) == 1)
        Marr = reshape(double(Marr), 1, []);   % row form
    elseif ndims(Marr) > 2 %#ok<ISMAT>
        error('translateAttributes:badPAttr', ...
              'pAttr{%d} must be 1-D or 2-D; got ndims=%d.', a, ndims(Marr));
    else
        Marr = double(Marr);
    end
    pAttr{a} = Marr;
end

% --- Check shared N ---
N = size(pAttr{1}, 2);
for a = 2:A
    if size(pAttr{a}, 2) ~= N
        error('translateAttributes:badEventCount', ...
              ['All attributes must share the same event count N. ' ...
               'Attribute 1 has N=%d; attribute %d has N=%d.'], ...
              N, a, size(pAttr{a}, 2));
    end
end

% --- Canonicalise groups (matches differenceEvents convention) ---
groupOfAttr = localCanonicaliseGroups(groups, A);
G = max(groupOfAttr);

% --- Build attrsOfGroup: 1 x G cell, each entry the attribute indices in that group ---
attrsOfGroup = cell(1, G);
for g = 1:G
    attrsOfGroup{g} = find(groupOfAttr == g);
end

% --- Validate isRel, isPer, periods ---
if numel(isRel) ~= G
    error('translateAttributes:wrongIsRelLength', ...
          'isRel must have length G = %d (number of groups); got %d.', ...
          G, numel(isRel));
end
if numel(isPer) ~= G
    error('translateAttributes:wrongIsPerLength', ...
          'isPer must have length G = %d; got %d.', G, numel(isPer));
end
if numel(periods) ~= G
    error('translateAttributes:wrongPeriodsLength', ...
          'periods must have length G = %d; got %d.', G, numel(periods));
end
isRel = logical(isRel(:).');

% --- Normalise offsets to an A-by-M per-attribute matrix ---
[matrixMode, offsetsPerAttr] = localNormaliseOffsets(offsets, A, G, attrsOfGroup);
M = size(offsetsPerAttr, 2);

% --- Identify relative groups carrying any finite per-attribute offset, ---
% --- and emit at most one no-op warning per call. Zero out the rows so ---
% --- the hot-loop NaN check handles the skip. ---
finiteMask = ~isnan(offsetsPerAttr);
warnedRelative = false;
for g = 1:G
    if ~isRel(g)
        continue;
    end
    attrs = attrsOfGroup{g};
    anyFinite = false;
    for ii = 1:numel(attrs)
        if any(finiteMask(attrs(ii), :))
            anyFinite = true;
            break;
        end
    end
    if ~anyFinite
        continue;
    end
    if ~warnedRelative
        warning('translateAttributes:noOpRelative', ...
                ['Group %d has isRel=true; translation is a structural ' ...
                 'no-op on relative groups (a uniform shift of all values ' ...
                 'cancels in every within-tuple difference). The group is ' ...
                 'left unchanged on every offset column.'], g);
        warnedRelative = true;
    end
    offsetsPerAttr(attrs, :) = NaN;
end

% --- Apply translation, column by column ---
if ~matrixMode
    pAttrTranslated = localTranslateOne(pAttr, offsetsPerAttr(:, 1));
else
    pAttrTranslated = cell(1, M);
    for m = 1:M
        pAttrTranslated{m} = localTranslateOne(pAttr, offsetsPerAttr(:, m));
    end
end

end


% =========================================================================
%  Local helpers
% =========================================================================

function out = localTranslateOne(pAttr, offsetsCol)
%LOCALTRANSLATEONE  Apply one column of per-attribute offsets to pAttr.
    A = numel(pAttr);
    out = cell(1, A);
    for a = 1:A
        Marr = pAttr{a};
        mu = offsetsCol(a);
        if isnan(mu)
            out{a} = Marr;
            continue;
        end
        out{a} = Marr + mu;
    end
end


function [matrixMode, offsetsPerAttr] = localNormaliseOffsets(offsets, A, G, attrsOfGroup)
%LOCALNORMALISEOFFSETS  Coerce public offsets to (A, M) with NaN-skip.
    if iscell(offsets)
        [matrixMode, offsetsPerAttr] = localNormaliseOffsetsCell( ...
            offsets, A, G, attrsOfGroup);
        return;
    end
    if ~isnumeric(offsets)
        error('translateAttributes:wrongOffsetsShape', ...
              ['offsets must be a numeric matrix or a 1-by-G cell array; ' ...
               'got class %s.'], class(offsets));
    end
    if ndims(offsets) > 2 %#ok<ISMAT>
        error('translateAttributes:wrongOffsetsShape', ...
              ['offsets must be a numeric matrix (scalar, row, column, ' ...
               'or 2-D) or a 1-by-G cell; got an array with ndims=%d.'], ...
              ndims(offsets));
    end
    offsets = double(offsets);
    [nRows, nCols] = size(offsets);

    % Reject inf early (NaN is allowed as skip sentinel).
    if any(isinf(offsets(:)))
        error('translateAttributes:nonFiniteOffset', ...
              'offsets entries must be finite (or NaN to skip a cell).');
    end

    if nRows == 1 && nCols == 1
        % Scalar: broadcast no sweep.
        matrixMode = false;
        if isnan(offsets)
            error('translateAttributes:nonFiniteOffset', ...
                  'Scalar offset must be finite; got NaN.');
        end
        offsetsPerAttr = repmat(offsets, A, 1);
        return;
    end
    if nRows == 1
        % 1-by-M row: broadcast sweep with M positions.
        matrixMode = true;
        offsetsPerAttr = repmat(offsets, A, 1);
        return;
    end
    if nRows == A
        % A-by-M (M >= 1): per-attribute. (Also handles A == G case,
        % where per-attribute and per-group are equivalent.)
        matrixMode = true;
        offsetsPerAttr = offsets;
        return;
    end
    if nRows == G
        % G-by-M (M >= 1): per-group, broadcast within group. Expand
        % to per-attribute by replicating each group's row across the
        % attributes of that group. Always returns matrix mode (1-by-M
        % wrapper) for consistency with the A-by-M and cell-form
        % n_g-by-M conventions: any 2-D input with nRows > 1 is a
        % per-axis spec and gets a sweep wrapper. The A == G case is
        % handled above (per-attribute interpretation; identical
        % numeric output, same matrix-mode wrapping).
        matrixMode = true;
        offsetsPerAttr = zeros(A, nCols);
        for g = 1:G
            attrs = attrsOfGroup{g};
            offsetsPerAttr(attrs, :) = repmat(offsets(g, :), numel(attrs), 1);
        end
        return;
    end
    error('translateAttributes:wrongOffsetsShape', ...
          ['offsets is a 2-D array with shape %d-by-%d; row count must ' ...
           'be 1 (broadcast across all attributes), A = %d (per-' ...
           'attribute), or G = %d (per-group, broadcast within group). ' ...
           'For mixed-per-group layouts, use the 1-by-G cell form.'], ...
          nRows, nCols, A, G);
end


function [matrixMode, offsetsPerAttr] = localNormaliseOffsetsCell(offsets, A, G, attrsOfGroup)
%LOCALNORMALISEOFFSETSCELL  Process the polymorphic 1-by-G cell form.
    sz = size(offsets);
    if numel(sz) ~= 2 || sz(1) ~= 1 || sz(2) ~= G
        error('translateAttributes:wrongOffsetsShape', ...
              ['offsets cell array must be 1-by-G = 1-by-%d (one cell ' ...
               'per group, in group order); got shape %d-by-%d.'], ...
              G, sz(1), sz(2));
    end

    % --- First pass: validate and determine sweep dimension M ---
    M = 1;
    matrixMode = false;
    valArrs = cell(1, G);
    for g = 1:G
        val = offsets{g};
        if isempty(val)
            valArrs{g} = [];
            continue;
        end
        if ~isnumeric(val)
            error('translateAttributes:wrongOffsetsShape', ...
                  'offsets{%d} must be numeric or empty; got class %s.', ...
                  g, class(val));
        end
        if ndims(val) > 2 %#ok<ISMAT>
            error('translateAttributes:wrongOffsetsShape', ...
                  'offsets{%d} must be 2-D (scalar, row, column, or matrix); got ndims=%d.', ...
                  g, ndims(val));
        end
        val = double(val);
        if any(isinf(val(:)))
            error('translateAttributes:nonFiniteOffset', ...
                  'offsets{%d}: entries must be finite (or NaN to skip a cell).', g);
        end
        [nRows, nCols] = size(val);
        if nCols > 1
            matrixMode = true;
            if M == 1
                M = nCols;
            elseif nCols ~= M
                error('translateAttributes:wrongOffsetsShape', ...
                      ['offsets{%d} has shape %d-by-%d; sweep dimension ' ...
                       '%d does not match the %d sweep positions ' ...
                       'established by other entries.'], ...
                      g, nRows, nCols, nCols, M);
            end
        elseif nRows > 1
            % column vector (n_g-by-1) — single-column matrix-mode.
            matrixMode = true;
        end
        valArrs{g} = val;
    end

    % --- Second pass: distribute into (A, M) ---
    offsetsPerAttr = NaN(A, M);
    for g = 1:G
        val = valArrs{g};
        if isempty(val)
            continue;
        end
        attrs = attrsOfGroup{g};
        n_g = numel(attrs);
        [nRows, nCols] = size(val);
        if nRows == 1 && nCols == 1
            % scalar: broadcast within group, no sweep.
            if isnan(val)
                error('translateAttributes:nonFiniteOffset', ...
                      ['offsets{%d}: scalar offset must be finite (use ' ...
                       'empty [] to skip a group); got NaN.'], g);
            end
            offsetsPerAttr(attrs, :) = val;
        elseif nRows == 1
            % 1-by-M row: broadcast within group, sweep.
            for ii = 1:n_g
                offsetsPerAttr(attrs(ii), :) = val;
            end
        elseif nRows == n_g
            % n_g-by-M (M >= 1): per-attribute.
            for ii = 1:n_g
                offsetsPerAttr(attrs(ii), :) = val(ii, :);
            end
        else
            error('translateAttributes:wrongOffsetsShape', ...
                  ['offsets{%d} is %d-by-%d; row count must be 1 ' ...
                   '(broadcast within group) or %d (per-attribute, ' ...
                   'matching the number of attributes in group %d).'], ...
                  g, nRows, nCols, n_g, g);
        end
    end
end


function groupOfAttr = localCanonicaliseGroups(groups, A)
    % Return a 1 x A vector of 1-indexed group labels.
    if isempty(groups)
        groupOfAttr = 1:A;
        return;
    end
    if iscell(groups)
        % Explicit partition: cell of vectors of attribute indices.
        G = numel(groups);
        groupOfAttr = zeros(1, A);
        for g = 1:G
            attrs = groups{g}(:).';
            groupOfAttr(attrs) = g;
        end
        if any(groupOfAttr == 0)
            error('translateAttributes:badGroups', ...
                  ['Partition does not cover all attributes; some ' ...
                   'attribute has no group assigned.']);
        end
        return;
    end
    groups = groups(:).';
    if numel(groups) ~= A
        error('translateAttributes:badGroups', ...
              ['groups vector must have length A = %d (one entry per ' ...
               'attribute); got length %d.'], A, numel(groups));
    end
    % Re-index to 1..G contiguous.
    uniq = unique(groups, 'stable');
    groupOfAttr = zeros(1, A);
    for ii = 1:numel(uniq)
        groupOfAttr(groups == uniq(ii)) = ii;
    end
end

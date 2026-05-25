function pAttrTranslated = translateEvents(pAttr, groups, offsets, isRel, isPer, periods)
%TRANSLATEEVENTS Translate selected groups' event values by a per-group offset.
%
%   pAttrTranslated = translateEvents(pAttr, groups, offsets, isRel, isPer, periods)
%   is a cross-event preprocessing helper for multi-attribute tensor
%   input. It takes the pAttr list one would otherwise feed to
%   buildExpTens and returns a transformed pAttrTranslated list with the
%   same shape conventions, in which every value of every attribute
%   belonging to a selected group has been shifted by the group's
%   offset. The output feeds directly into buildExpTens without any
%   further massaging.
%
%   Sliding-comparison context. translateEvents is the pre-tensor route
%   to a sliding comparison along one or more attribute-group axes: for
%   each candidate offset mu on a sweep grid, translate the events and
%   compute a similarity against an un-shifted reference. The post-
%   tensor counterpart is windowedSimilarity. Both answer related "slide
%   along an axis" questions but with different operational properties;
%   see USER_GUIDE.md §3.1.
%
%   Two offset-input shapes are accepted:
%
%     Vector form. offsets is a 1-by-G numeric row vector (or a 1-by-1
%     scalar when G = 1). One translation is performed and
%     pAttrTranslated has the same 1-by-A cell shape as pAttr. NaN
%     entries mean "do not translate this group".
%
%     Matrix form (sweep). offsets is a G-by-M numeric matrix. M is
%     the number of sweep positions. Translation is applied column by
%     column: for each m in 1..M, the function builds a translated
%     copy of pAttr using offsets(:, m). The return is a 1-by-M cell
%     of 1-by-A cells. M = 1 still yields a 1-by-1 cell wrapper, never
%     collapses to the vector-form return. Pass the result directly
%     into the raw-MA list mode of cosSimExpTens (cosSimExpTens with
%     pAttrCell on either operand) to score the sweep in one call.
%
%     Note on column vectors. A length-G column vector (e.g. [5; 7]
%     for G = 2) is a G-by-1 matrix and so reads as matrix form with
%     M = 1, NOT as vector form. To pass a vector form in MATLAB use
%     the 1-by-G row layout. This convention is the only way to make
%     M = 1 reachable, since MATLAB does not distinguish a length-G
%     column vector from a G-by-1 2-D array.
%
%   Semantics by group geometry (apply per offset column in matrix form):
%
%     Absolute non-periodic (isPer(g) = false): every value of every
%     attribute in group g is replaced by value + mu. periods(g) is
%     ignored regardless of its sign.
%
%     Absolute periodic (isPer(g) = true with isRel(g) = false and
%     periods(g) > 0): values are translated and wrapped to [0, P) via
%     mod(value + mu, P). The wrapped periodic Gaussian kernel of
%     buildExpTens is invariant under any additive shift by a multiple
%     of P, so this canonical wrap yields the same MAET as leaving the
%     values unwrapped — the wrap is for tidiness, not correctness.
%
%     Relative (isRel(g) = true): a uniform shift of every value
%     cancels in every within-tuple difference, so translation on a
%     relative group is a structural no-op. The group is left
%     unchanged. The warning translateEvents:noOpRelative is emitted
%     at most once per call, even when the matrix form has many
%     columns with finite entries on the relative-group row.
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
%       offsets  - Either a 1-by-G numeric row vector (single
%                  translation; 1-by-1 scalar when G = 1) or a
%                  G-by-M numeric matrix (M-column sweep, including
%                  G-by-1 columns for M = 1). Entries that are NaN
%                  mean "do not translate this group" for the
%                  corresponding column — the attributes in that
%                  group pass through unchanged on that column.
%                  Finite entries are the translation amount.
%                  Indexing is 1-based to match MATLAB's group-index
%                  convention. The Python counterpart accepts a
%                  sparse dict (vector form) or a (G, M) ndarray
%                  (matrix form).
%       isRel    - 1 x G logical vector of relative-mode flags, same
%                  convention as buildExpTens. Groups with isRel(g) =
%                  true that have at least one finite offset entry
%                  emit a single translateEvents:noOpRelative warning
%                  and pass through unchanged on every column.
%       isPer    - 1 x G logical vector of periodic-mode flags, same
%                  convention as buildExpTens. Wrapping after
%                  translation is applied only when isPer(g) = true
%                  AND periods(g) > 0.
%       periods  - 1 x G numeric vector of periods, same convention as
%                  buildExpTens. Consulted only when isPer(g) = true.
%                  For non-periodic groups (isPer(g) = false) the entry
%                  is ignored, so it is safe to declare a group's
%                  natural period (e.g. 12 for pitch class) even when
%                  operating in non-periodic mode for a particular
%                  analysis.
%
%   Outputs
%       pAttrTranslated - For vector-form offsets: a 1 x A cell of
%                         K_a x N matrices, same shapes as the input,
%                         with translation applied to the selected
%                         groups. For matrix-form offsets: a 1 x M
%                         cell of such 1 x A cells, one per offset
%                         column. pAttr is not mutated.
%
%   Errors
%       translateEvents:wrongOffsetsShape - offsets has the wrong shape
%                                            (not a length-G vector and
%                                            not a G-by-M matrix).
%       translateEvents:wrongIsRelLength   - isRel has length ~= G.
%       translateEvents:wrongIsPerLength   - isPer has length ~= G.
%       translateEvents:wrongPeriodsLength - periods has length ~= G.
%       translateEvents:badPAttr           - pAttr is not a cell.
%       translateEvents:badEventCount      - attributes have differing N.
%       translateEvents:nonFiniteOffset    - an offset is +/-Inf.
%
%   Warnings
%       translateEvents:noOpRelative - emitted once per call when any
%                                       relative-group row has any
%                                       finite offset entry.
%
%   See also DIFFERENCEEVENTS, BINDEVENTS, BUILDEXPTENS, WINDOWEDSIMILARITY, COSSIMEXPTENS.

% --- Normalise pAttr to a cell of 2-D double matrices ---
if isnumeric(pAttr)
    pAttr = {pAttr};
end
if ~iscell(pAttr) || isempty(pAttr)
    error('translateEvents:badPAttr', ...
          'pAttr must be a non-empty cell of value matrices.');
end
A = numel(pAttr);
for a = 1:A
    M = pAttr{a};
    if isvector(M) && (size(M, 1) == 1 || size(M, 2) == 1)
        M = reshape(double(M), 1, []);   % row form
    elseif ndims(M) > 2 %#ok<ISMAT>
        error('translateEvents:badPAttr', ...
              'pAttr{%d} must be 1-D or 2-D; got ndims=%d.', a, ndims(M));
    else
        M = double(M);
    end
    pAttr{a} = M;
end

% --- Check shared N ---
N = size(pAttr{1}, 2);
for a = 2:A
    if size(pAttr{a}, 2) ~= N
        error('translateEvents:badEventCount', ...
              ['All attributes must share the same event count N. ' ...
               'Attribute 1 has N=%d; attribute %d has N=%d.'], ...
              N, a, size(pAttr{a}, 2));
    end
end

% --- Canonicalise groups (matches differenceEvents convention) ---
groupOfAttr = localCanonicaliseGroups(groups, A);
G = max(groupOfAttr);

% --- Validate isRel, isPer, periods ---
if numel(isRel) ~= G
    error('translateEvents:wrongIsRelLength', ...
          'isRel must have length G = %d (number of groups); got %d.', ...
          G, numel(isRel));
end
if numel(isPer) ~= G
    error('translateEvents:wrongIsPerLength', ...
          'isPer must have length G = %d; got %d.', G, numel(isPer));
end
if numel(periods) ~= G
    error('translateEvents:wrongPeriodsLength', ...
          'periods must have length G = %d; got %d.', G, numel(periods));
end
isRel   = logical(isRel(:).');
isPer   = logical(isPer(:).');
periods = double(periods(:).');

% --- Detect offset-input shape: vector (single) or matrix (sweep) ---
% Disambiguation rules (MATLAB-specific, because a length-G column
% vector and a G-by-1 matrix share the same shape):
%   * 1-by-G row vector (or 1-by-1 scalar when G == 1)  → vector form
%   * G == 1, 1-by-N row vector with N > 1              → matrix form (1×N sweep)
%   * 2-D matrix with G rows, M columns (M >= 1,
%     including the G-by-1 column-vector case)         → matrix form (M-column sweep)
%   * Anything else                                     → error
% Note: a length-G column vector (e.g. [5; 7] for G = 2) is treated
% as a G-by-1 matrix → matrix form with M = 1. Vector form in MATLAB
% requires a 1-by-G ROW vector (or a scalar when G = 1). This is the
% only way to make M = 1 reachable without an extra flag, since
% MATLAB does not distinguish a length-G column vector from a G-by-1
% 2-D array.
offsets = double(offsets);
if ndims(offsets) > 2 %#ok<ISMAT>
    error('translateEvents:wrongOffsetsShape', ...
          ['offsets must be a 1-by-G row vector (single translation) ' ...
           'or a G-by-M matrix (M-column sweep); got an array with ' ...
           'ndims=%d.'], ndims(offsets));
end
[nRows, nCols] = size(offsets);
if nRows == 1 && nCols == G
    % 1-by-G row vector (includes 1-by-1 scalar when G == 1).
    matrixMode = false;
    offsetCols = offsets(:);                     % G x 1 internally
elseif G == 1 && nRows == 1 && nCols > 1
    % G == 1, length-N row vector with N > 1: matrix form 1-by-N sweep.
    matrixMode = true;
    offsetCols = offsets;                        % 1 x N
elseif nRows == G
    % 2-D with G rows (includes the G-by-1 column-vector case, which
    % is matrix form with M = 1).
    matrixMode = true;
    offsetCols = offsets;                        % G x M
else
    error('translateEvents:wrongOffsetsShape', ...
          ['offsets must be a 1-by-G = 1-by-%d row vector (single ' ...
           'translation) or a G-by-M = %d-by-M matrix (M-column ' ...
           'sweep); got shape %d-by-%d.'], G, G, nRows, nCols);
end
if any(isinf(offsetCols(~isnan(offsetCols))))
    error('translateEvents:nonFiniteOffset', ...
          'offsets entries must be finite (or NaN to skip a group).');
end
M = size(offsetCols, 2);

% --- Identify groups with at least one finite offset across columns, ---
% --- and emit at most one relative-group no-op warning per call. ---
anyFiniteByGroup = any(~isnan(offsetCols), 2).';   % 1 x G logical
groupTouchable = false(1, G);                       % can a finite mu apply?
warnedRelative = false;
for g = 1:G
    if ~anyFiniteByGroup(g)
        continue;
    end
    if isRel(g)
        if ~warnedRelative
            warning('translateEvents:noOpRelative', ...
                    ['Group %d has isRel=true; translation is a structural ' ...
                     'no-op on relative groups (a uniform shift of all values ' ...
                     'cancels in every within-tuple difference). The group is ' ...
                     'left unchanged on every offset column.'], g);
            warnedRelative = true;
        end
        continue;
    end
    groupTouchable(g) = true;
end

% --- Apply translation ---
if ~matrixMode
    pAttrTranslated = localTranslateOne(pAttr, groupOfAttr, ...
                                         offsetCols(:, 1), groupTouchable, ...
                                         isPer, periods);
else
    pAttrTranslated = cell(1, M);
    for m = 1:M
        pAttrTranslated{m} = localTranslateOne(pAttr, groupOfAttr, ...
                                                offsetCols(:, m), groupTouchable, ...
                                                isPer, periods);
    end
end

end


% =========================================================================
%  Local helpers
% =========================================================================

function out = localTranslateOne(pAttr, groupOfAttr, offsetsCol, ...
                                  groupTouchable, isPer, periods)
%LOCALTRANSLATEONE  Apply one column of offsets to pAttr.
    A = numel(pAttr);
    out = cell(1, A);
    for a = 1:A
        g = groupOfAttr(a);
        Marr = pAttr{a};
        mu = offsetsCol(g);
        if ~groupTouchable(g) || isnan(mu)
            out{a} = Marr;
            continue;
        end
        Marr = Marr + mu;
        if isPer(g) && periods(g) > 0
            Marr = mod(Marr, periods(g));
        end
        out{a} = Marr;
    end
end


function groupOfAttr = localCanonicaliseGroups(groups, A)
    % Return a 1 x A vector of 1-indexed group labels.
    if isempty(groups)
        groupOfAttr = 1:A;
        return;
    end
    if iscell(groups)
        G = numel(groups);
        groupOfAttr = zeros(1, A);
        for g = 1:G
            idx = groups{g};
            if any(idx < 1) || any(idx > A) || any(groupOfAttr(idx) ~= 0)
                error('translateEvents:badGroups', ...
                      'Invalid cell-form groups specification.');
            end
            groupOfAttr(idx) = g;
        end
        if any(groupOfAttr == 0)
            error('translateEvents:badGroups', ...
                  'Every attribute must appear in exactly one group.');
        end
        return;
    end
    if isnumeric(groups) && numel(groups) == A
        groupOfAttr = double(groups(:).');
        if any(groupOfAttr < 1) || any(groupOfAttr ~= round(groupOfAttr))
            error('translateEvents:badGroups', ...
                  'Numeric groups must be positive integers.');
        end
        return;
    end
    error('translateEvents:badGroupsShape', ...
          'groups must be [], a length-A numeric vector, or a cell of index lists.');
end

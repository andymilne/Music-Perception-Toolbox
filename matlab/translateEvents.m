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
%   Semantics by group geometry:
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
%     unchanged and a warning with identifier
%     translateEvents:noOpRelative is emitted.
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
%       offsets  - 1 x G numeric vector of per-group offsets. Entries
%                  that are NaN mean "do not translate this group" —
%                  the corresponding group's values pass through
%                  unchanged. Finite entries are the translation
%                  amount for that group. Indexing is 1-based to match
%                  MATLAB's group-index convention. The Python
%                  counterpart accepts a sparse dict instead.
%       isRel    - 1 x G logical vector of relative-mode flags, same
%                  convention as buildExpTens. Groups with isRel(g) =
%                  true that have a finite offset emit a
%                  translateEvents:noOpRelative warning and pass
%                  through unchanged.
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
%       pAttrTranslated - 1 x A cell of K_a x N matrices, same shapes
%                         as the input, with translation applied to
%                         the selected groups. pAttr is not mutated.
%
%   Errors
%       translateEvents:wrongOffsetsLength - offsets has length ~= G.
%       translateEvents:wrongIsRelLength   - isRel has length ~= G.
%       translateEvents:wrongIsPerLength   - isPer has length ~= G.
%       translateEvents:wrongPeriodsLength - periods has length ~= G.
%       translateEvents:badPAttr           - pAttr is not a cell.
%       translateEvents:badEventCount      - attributes have differing N.
%       translateEvents:nonFiniteOffset    - an offset is +/-Inf.
%
%   Warnings
%       translateEvents:noOpRelative - emitted when a relative group
%                                       has a finite offset.
%
%   See also DIFFERENCEEVENTS, BINDEVENTS, BUILDEXPTENS, WINDOWEDSIMILARITY.

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

% --- Validate isRel, isPer, periods, offsets ---
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
if numel(offsets) ~= G
    error('translateEvents:wrongOffsetsLength', ...
          ['offsets must have length G = %d (one entry per group, ' ...
           'NaN = no translation); got %d.'], G, numel(offsets));
end
isRel   = logical(isRel(:).');
isPer   = logical(isPer(:).');
periods = double(periods(:).');
offsets = double(offsets(:).');
if any(isinf(offsets(~isnan(offsets))))
    error('translateEvents:nonFiniteOffset', ...
          'offsets entries must be finite (or NaN to skip a group).');
end

% --- Identify groups to translate, emitting warnings for relative ones ---
groupsToTranslate = false(1, G);
for g = 1:G
    if isnan(offsets(g))
        continue;   % skip
    end
    if isRel(g)
        warning('translateEvents:noOpRelative', ...
                ['Group %d has isRel=true; translation is a structural ' ...
                 'no-op on relative groups (a uniform shift of all values ' ...
                 'cancels in every within-tuple difference). The group ' ...
                 'is left unchanged.'], g);
        continue;
    end
    groupsToTranslate(g) = true;
end

% --- Apply translation ---
pAttrTranslated = cell(1, A);
for a = 1:A
    g = groupOfAttr(a);
    M = pAttr{a};
    if ~groupsToTranslate(g)
        pAttrTranslated{a} = M;        % unchanged
        continue;
    end
    mu = offsets(g);
    M = M + mu;
    if isPer(g) && periods(g) > 0
        M = mod(M, periods(g));
    end
    pAttrTranslated{a} = M;
end

end


% =========================================================================
%  Local helpers
% =========================================================================

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

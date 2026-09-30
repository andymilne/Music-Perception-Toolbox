function pm = translateAttributes(varargin)
%TRANSLATEATTRIBUTES Translate attributes' positions by per-row offsets.
%
%   PM = translateAttributes(PM0, offsets, ...) and
%   PM = translateAttributes(pAttr, wAttr, offsets, ...) are per-attribute
%   preprocessing on the pre-MAET. Selected attributes' positions are
%   shifted by a chosen offset and the transformed pre-MAET feeds straight
%   into buildMaet (or a further pre-MAET step). Weights and specs pass
%   through unchanged; only the values move.%
%   The pre-MAET may be passed whole, as packPreMaet builds it, or in
%   its parts as pAttr and wAttr with the specs as a name-value; the two
%   forms are the same call.
%
%   Offsets and the value matrix (read this first). An attribute's values
%   are stored as a K_total x N matrix: one column per event and one row
%   per element, row k holding the k-th element of every event's element
%   multiset, where K_total is the number of elements in one event's
%   element multiset (counting every element of a nested one, such as a
%   super-event's). The spec's tags label the rows (one entry per row). An
%   offset is likewise per row (per-value): one offset per row, held
%   CONSTANT across events --- that constancy is what makes D(T(p)) ==
%   D(p).
%
%   Offsets are a 1 x A cell, one entry per attribute, each entry one of:
%       []                  - do not translate this attribute.
%       scalar              - broadcast to all K_total values.
%       column (K_total x 1)- per-value offsets.
%
%   NaN entries skip the corresponding value (left untranslated); +/-Inf is
%   rejected.
%
%   One call makes one translation. A translation sweep --- a query
%   compared with a context at each of many offsets --- is sweptSimilarity
%   (pre-MAETs) or sweepSimMaet (densities), which compute every offset in
%   one pass rather than building a copy per offset.
%
%   Relative attributes. is_rel is read per-attribute from specs (no
%   separate argument). A uniform shift cancels in every within-tuple
%   difference, so on an attribute whose OUTERMOST level is relative a
%   uniform finite offset is a structural no-op: that attribute is left
%   unchanged and a single warning is emitted per call. A non-uniform
%   (per-value) offset is NOT a no-op even on a relative attribute --- it
%   shifts the within-tuple differences --- so it applies. is_per/period
%   are not consulted here (translation emits unwrapped values; the
%   periodic kernel in buildMaet wraps downstream) and stay separate.
%
%   Inputs
%       pm      - Pre-MAET, in place of pAttr and wAttr.
%       pAttr   - 1 x A cell of K_total x N per-attribute value matrices,
%                 or of attributes given per event (see packPreMaet).
%       wAttr   - Weights ([], scalar, or 1 x A cell); passed through.
%       offsets - 1 x A cell of per-attribute offsets (see above).
%
%   Name-value pairs
%       'specs' - [] (synthesise flat via flatSpecs) or a 1 x A cell of
%                 per-attribute specs supplying is_rel (outermost level).
%
%   Outputs
%       pm    - Pre-MAET with the translated pAttr; wAttr and specs are
%               unchanged from input (or synthesised).
%
%   See also SWEPTSIMILARITY, SWEEPSIMMAET, DIFFERENCEEVENTS,
%   BINDEVENTS, FLATSPECS, BUILDMAET.

[pAttr, wAttr, specsPm, rest] = internal.preMaetArgs(varargin);
[pOut, w, specs] = localTranslateAttributes(pAttr, wAttr, specsPm, rest{:});
pm = packPreMaet(pOut, w, specs);
end


function [pOut, w, specs] = localTranslateAttributes(pAttr, w, specsPm, offsets, nvArgs)
arguments
    pAttr
    w
    specsPm
    offsets
    nvArgs.specs = []
end

if isempty(nvArgs.specs)
    nvArgs.specs = specsPm;
end

% --- Normalise pAttr ---
if ~iscell(pAttr)
    error('translateAttributes:badPAttrType', ...
          'pAttr must be a cell array of attribute value matrices.');
end
A = numel(pAttr);
if A < 1
    error('translateAttributes:noAttrs', ...
          'pAttr must contain at least one attribute.');
end
pArr = cell(1, A);
for a = 1:A
    M = pAttr{a};
    if ~isnumeric(M) || ndims(M) > 2
        error('translateAttributes:badAttrShape', ...
              'Attribute %d must be a numeric 1-D or 2-D matrix.', a);
    end
    pArr{a} = double(M);
end
nEvents = size(pArr{1}, 2);
for a = 2:A
    if size(pArr{a}, 2) ~= nEvents
        error('translateAttributes:eventCountMismatch', ...
              ['All attributes must share the same event count N. ' ...
               'Attribute 1 has N=%d; attribute %d has N=%d.'], ...
              nEvents, a, size(pArr{a}, 2));
    end
end
K = zeros(1, A);
for a = 1:A
    K(a) = size(pArr{a}, 1);          % K_total (rows) per attribute
end

% --- Attribute specifications ---
if isempty(nvArgs.specs)
    specs = flatSpecs(pArr);
else
    specs = nvArgs.specs;
    if ~iscell(specs) || numel(specs) ~= A
        error('translateAttributes:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end
end

% --- Normalise offsets to per-attribute K_a x 1 columns (NaN = skip) ---
blocks = localNormaliseOffsets(offsets, K, A);

% --- Relative no-op: a uniform finite shift on an outermost-relative ---
% --- attribute is a structural no-op; skip it and warn. ---
warned = false;
for a = 1:A
    cm = blocks{a};
    if localOutermostRelative(specs{a}) && ~isempty(cm) ...
            && all(isfinite(cm)) && all(cm == cm(1))
        blocks{a}(:) = NaN;
        warned = true;
    end
end
if warned
    warning('translateAttributes:noOp', ...
            ['A uniform finite offset was applied to an attribute whose ' ...
             'outermost level is relative; a uniform shift cancels in ' ...
             'every within-tuple difference, so it is a structural no-op ' ...
             'and that attribute is left unchanged. (A non-uniform ' ...
             'per-value offset would apply, as it shifts the relative ' ...
             'structure.)']);
end

% --- Apply: value + per-value offset, broadcast across events; NaN values ---
% --- are left untranslated. ---
pOut = cell(1, A);
for a = 1:A
    off = blocks{a};
    fin = isfinite(off);
    if ~any(fin)
        pOut{a} = pArr{a};
    else
        off(~fin) = 0;
        pOut{a} = pArr{a} + off;          % implicit expansion across columns
    end
end

end


% =========================================================================
%  Helpers
% =========================================================================

function blocks = localNormaliseOffsets(offsets, K, A)
%LOCALNORMALISEOFFSETS  Coerce a 1 x A offsets cell to per-attribute
%   K_a x 1 columns (NaN marks skipped values): [] skips the attribute, a
%   scalar is broadcast to every value, a K_a x 1 column is per value.
    if ~iscell(offsets) || numel(offsets) ~= A
        error('translateAttributes:offsetsShape', ...
              ['offsets must be a length-A (%d) cell, one entry per ' ...
               'attribute ([], a scalar, or a K_total x 1 column).'], A);
    end
    blocks = cell(1, A);
    for a = 1:A
        o = offsets{a};
        if isempty(o)
            blocks{a} = NaN(K(a), 1);     % skip this attribute
            continue;
        end
        if ~isnumeric(o)
            error('translateAttributes:offsetType', ...
                  'offsets{%d} must be numeric or [].', a);
        end
        o = double(o);
        if any(isinf(o(:)))
            error('translateAttributes:offsetInf', ...
                  ['offsets{%d} contains +/-Inf; entries must be finite ' ...
                   'or NaN (NaN skips a value).'], a);
        end
        if isscalar(o)
            blocks{a} = repmat(o, K(a), 1);
        elseif iscolumn(o) && numel(o) == K(a)
            blocks{a} = o;
        else
            error('translateAttributes:offsetShape', ...
                  ['offsets{%d} must be [], a scalar, or a K_total x 1 ' ...
                   'column (K_total = %d); got %s. One call makes one ' ...
                   'translation: for a translation sweep use ' ...
                   'sweptSimilarity or sweepSimMaet.'], ...
                  a, K(a), mat2str(size(o)));
        end
    end
end


function tf = localOutermostRelative(spec)
%LOCALOUTERMOSTRELATIVE  Whether the attribute's outermost level is
%   relative: flat rel truthy, nested rel vector with truthy last entry,
%   or the rel = 'outermost' selector.
    tf = false;
    if ~isstruct(spec)
        return;
    end
    if isfield(spec, 'tags')
        if isfield(spec, 'rel') && ~isempty(spec.rel)
            rel = spec.rel;
            if ischar(rel) || isstring(rel)
                tf = strcmp(char(rel), 'outermost');
            else
                relv = rel(:);
                tf = logical(relv(end));
            end
        end
    else
        if isfield(spec, 'rel') && ~isempty(spec.rel)
            tf = logical(spec.rel(1));
        end
    end
end

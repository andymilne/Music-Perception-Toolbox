function [pOut, w, specs] = translateAttributes(pAttr, w, offsets, nvArgs)
%TRANSLATEATTRIBUTES Translate attributes' values by per-slot offsets (carrier).
%
%   [pOut, w, specs] = translateAttributes(pAttr, w, offsets, ...)
%   is per-attribute preprocessing on the (pAttr, w, specs) carrier.
%   Selected attributes' values are shifted by a chosen offset and the
%   transformed triple feeds straight into buildExpTens (or a further
%   pre-MAET step). Weights and specs pass through unchanged; only the
%   values move.
%
%   Slot-axis alignment (read this first). Everything hangs off one axis:
%   the slot axis of an attribute, whose length is K_total (the number of
%   leaf values in one event/super-event). In the value matrix the slot
%   axis is the ROWS (K_total x N: slots down, sequence positions across).
%   The spec's tags label that same axis (one entry per row). An offset is
%   likewise per-slot: one value per row, held CONSTANT across the sequence
%   (column) axis --- that constancy is what makes D(T(p)) == D(p).
%
%   Offsets are a 1 x A cell, one entry per attribute, each entry one of:
%       []                  - do not translate this attribute.
%       scalar              - broadcast to all K_total slots.
%       column (K_total x 1)- per-slot, single translation.
%       row    (1 x M)      - per-sweep global shift: one scalar per sweep
%                             index, broadcast across slots (M copies).
%       matrix (K_total x M)- per-slot by sweep index: slots down, sweep
%                             index across.
%   (Orientation disambiguates: a single per-slot offset is a COLUMN; a
%   sweep of global shifts is a ROW. This differs from the Python list
%   form, which uses 1-D vs 2-D; the semantics and outputs are identical.)
%
%   NaN entries skip the corresponding slot (left untranslated); +/-Inf is
%   rejected. All swept entries must agree on M (scalar, column, and single
%   sweep-column entries broadcast across the call's M).
%
%   Sweep. When any entry implies M > 1 the call is a batched sweep: it
%   returns M translated copies --- a 1 x M cell of 1 x A value-cells ---
%   each a separate pre-MAET input, sharing one w and one specs. With M = 1
%   it returns a single 1 x A value-cell.
%
%   Relative attributes. is_rel is read per-attribute from specs (no
%   separate argument). A uniform shift cancels in every within-tuple
%   difference, so on an attribute whose OUTERMOST level is relative a
%   uniform finite offset is a structural no-op: that column is left
%   unchanged and a single warning is emitted per call. A non-uniform
%   (per-slot) offset is NOT a no-op even on a relative attribute --- it
%   shifts the within-tuple differences --- so it applies. is_per/period
%   are not consulted here (translation emits unwrapped values; the
%   periodic kernel in buildExpTens wraps downstream) and stay separate.
%
%   Inputs
%       pAttr   - 1 x A cell of K_total x N per-attribute value matrices.
%       w       - Weights ([], scalar, or 1 x A cell); passed through.
%       offsets - 1 x A cell of per-attribute offsets (see above).
%
%   Name-value pairs
%       'specs' - [] (synthesise flat via flatSpecs) or a 1 x A cell of
%                 per-attribute specs supplying is_rel (outermost level).
%
%   Outputs
%       pOut  - Single (M = 1): 1 x A cell of K_total x N matrices.
%               Sweep (M > 1): 1 x M cell of such cells.
%       w     - Same as input.
%       specs - The carrier specs, unchanged (or synthesised).
%
%   See also DIFFERENCEEVENTS, BINDEVENTS, FLATSPECS, BUILDEXPTENS.

arguments
    pAttr
    w
    offsets
    nvArgs.specs = []
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

% --- Carrier specs ---
if isempty(nvArgs.specs)
    specs = flatSpecs(pArr);
else
    specs = nvArgs.specs;
    if ~iscell(specs) || numel(specs) ~= A
        error('translateAttributes:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end
end

% --- Normalise offsets to per-attribute K_a x M blocks (NaN = skip) ---
[matrixMode, Msweep, blocks] = localNormaliseOffsets(offsets, K, A);

% --- Relative no-op: a uniform finite shift on an outermost-relative ---
% --- attribute is a structural no-op; skip that column, warn once. ---
warned = false;
for a = 1:A
    if ~localOutermostRelative(specs{a})
        continue;
    end
    for m = 1:Msweep
        cm = blocks{a}(:, m);
        if all(isfinite(cm)) && ~isempty(cm) && all(cm == cm(1))
            blocks{a}(:, m) = NaN;
            warned = true;
        end
    end
end
if warned
    warning('translateAttributes:noOp', ...
            ['A uniform finite offset was applied to an attribute whose ' ...
             'outermost level is relative; a uniform shift cancels in ' ...
             'every within-tuple difference, so it is a structural no-op ' ...
             'and that column is left unchanged. (A non-uniform per-slot ' ...
             'offset would apply, as it shifts the relative structure.)']);
end

% --- Apply: value + per-slot offset, broadcast across events; NaN slots ---
% --- are left untranslated. ---
colsOut = cell(1, Msweep);
for m = 1:Msweep
    colList = cell(1, A);
    for a = 1:A
        M = pArr{a};
        off = blocks{a}(:, m);
        fin = isfinite(off);
        if ~any(fin)
            colList{a} = M;
        else
            add = off;
            add(~fin) = 0;
            colList{a} = M + add;        % implicit expansion across columns
        end
    end
    colsOut{m} = colList;
end

if matrixMode
    pOut = colsOut;                       % 1 x M cell of 1 x A cells
else
    pOut = colsOut{1};                    % 1 x A cell
end

end


% =========================================================================
%  Helpers
% =========================================================================

function [matrixMode, M, blocks] = localNormaliseOffsets(offsets, K, A)
%LOCALNORMALISEOFFSETS  Coerce a 1 x A offsets cell to per-attribute
%   K_a x M blocks (NaN marks skipped slots). Orientation disambiguates:
%   scalar -> all slots; column -> per-slot; row -> per-sweep; matrix ->
%   per-slot x sweep. Returns [matrixMode, M, blocks].
    if ~iscell(offsets) || numel(offsets) ~= A
        error('translateAttributes:offsetsShape', ...
              ['offsets must be a length-A (%d) cell, one entry per ' ...
               'attribute ([], scalar, column per-slot, row sweep, or ' ...
               'K_total x M block).'], A);
    end
    raw = cell(1, A);
    M = 1;
    for a = 1:A
        o = offsets{a};
        if isempty(o)
            raw{a} = [];                  % skip this attribute
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
                   'or NaN (NaN skips a slot).'], a);
        end
        if isscalar(o)
            raw{a} = o;                   % broadcast to all slots
        elseif isrow(o)
            raw{a} = o;                   % per-sweep global shift
            M = max(M, size(o, 2));
        elseif iscolumn(o)
            if numel(o) ~= K(a) && numel(o) ~= 1
                error('translateAttributes:offsetLength', ...
                      ['offsets{%d} is a length-%d column; expected ' ...
                       'K_total = %d (per-slot).'], a, numel(o), K(a));
            end
            raw{a} = o;                   % per-slot, single
        else
            if size(o, 1) ~= K(a) && size(o, 1) ~= 1
                error('translateAttributes:offsetRows', ...
                      ['offsets{%d} has %d rows; expected 1 or K_total ' ...
                       '= %d (slots down).'], a, size(o, 1), K(a));
            end
            raw{a} = o;                   % per-slot x sweep
            M = max(M, size(o, 2));
        end
    end
    matrixMode = M > 1;
    blocks = cell(1, A);
    for a = 1:A
        r = raw{a};
        if isempty(r)
            blocks{a} = NaN(K(a), M);
            continue;
        end
        if isscalar(r)
            r = repmat(r, K(a), M);
        else
            if size(r, 1) == 1 && K(a) > 1
                r = repmat(r, K(a), 1);   % broadcast across slots
            end
            if size(r, 2) == 1 && M > 1
                r = repmat(r, 1, M);      % broadcast across sweep
            elseif size(r, 2) ~= 1 && size(r, 2) ~= M
                error('translateAttributes:sweepM', ...
                      ['offsets{%d} has %d sweep columns; expected 1 or ' ...
                       'M = %d (all swept entries must agree on M).'], ...
                      a, size(r, 2), M);
            end
        end
        blocks{a} = r;
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

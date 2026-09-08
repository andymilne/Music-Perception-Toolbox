function [pm, sweep] = translateAttributes(varargin)
%TRANSLATEATTRIBUTES Translate attributes' positions by per-row offsets.
%
%   PM = translateAttributes(PM0, offsets, ...) and
%   PM = translateAttributes(pAttr, wAttr, offsets, ...) are per-attribute
%   preprocessing on the pre-MAET. Selected attributes' positions are
%   shifted by a chosen offset and the transformed pre-MAET feeds straight
%   into buildExpTens (or a further pre-MAET step). Weights and specs pass
%   through unchanged; only the values move.%
%   The pre-MAET may be passed whole, as preMaet builds it, or in
%   its parts as pAttr and wAttr with the specs as a name-value; the two
%   forms are the same call.
%
%   Value-axis alignment (read this first). Everything hangs off one axis:
%   the value axis of an attribute, whose length is K_total (the number of
%   leaf values in one event/super-event). In the value matrix the value
%   axis is the ROWS (K_total x N: values down, sequence positions across).
%   The spec's tags label that same axis (one entry per row). An offset is
%   likewise per-value: one offset per row, held CONSTANT across the sequence
%   (column) axis --- that constancy is what makes D(T(p)) == D(p).
%
%   Offsets are a 1 x A cell, one entry per attribute, each entry one of:
%       []                  - do not translate this attribute.
%       scalar              - broadcast to all K_total values.
%       column (K_total x 1)- per-value, single translation.
%       row    (1 x M)      - per-sweep global shift: one scalar per sweep
%                             index, broadcast across values (M copies).
%       matrix (K_total x M)- per-value by sweep index: values down, sweep
%                             index across.
%   (Orientation disambiguates: a single per-value offset is a COLUMN; a
%   sweep of global shifts is a ROW. This differs from the Python list
%   form, which uses 1-D vs 2-D; the semantics and outputs are identical.)
%
%   NaN entries skip the corresponding value (left untranslated); +/-Inf is
%   rejected. All swept entries must agree on M (scalar, column, and single
%   sweep-column entries broadcast across the call's M).
%
%   Sweep. When any entry implies M > 1 the call is a batched sweep: it
%   returns M translated copies --- a 1 x M cell of 1 x A position-cells ---
%   each a separate pre-MAET input, sharing one w and one specs. With M = 1
%   it returns a single 1 x A position-cell.
%
%   Relative attributes. is_rel is read per-attribute from specs (no
%   separate argument). A uniform shift cancels in every within-tuple
%   difference, so on an attribute whose OUTERMOST level is relative a
%   uniform finite offset is a structural no-op: that column is left
%   unchanged and a single warning is emitted per call. A non-uniform
%   (per-value) offset is NOT a no-op even on a relative attribute --- it
%   shifts the within-tuple differences --- so it applies. is_per/period
%   are not consulted here (translation emits unwrapped values; the
%   periodic kernel in buildExpTens wraps downstream) and stay separate.
%
%   Inputs
%       pm      - Pre-MAET, in place of pAttr and wAttr.
%       pAttr   - 1 x A cell of K_total x N per-attribute value matrices.
%       wAttr   - Weights ([], scalar, or 1 x A cell); passed through.
%       offsets - 1 x A cell of per-attribute offsets (see above).
%
%   Name-value pairs
%       'specs' - [] (synthesise flat via flatSpecs) or a 1 x A cell of
%                 per-attribute specs supplying is_rel (outermost level).
%
%   Outputs
%       pm    - Pre-MAET. Its pAttr is, for a single translation
%               (M = 1), a 1 x A cell of K_total x N matrices, and for a
%               sweep (M > 1) a 1 x M cell of such cells; wAttr and specs
%               are unchanged from input (or synthesised).
%       sweep - Struct describing the sweep, for callers that go on to
%               sweepCosSimExpTens: .offsets is an A x M matrix of the
%               per-attribute uniform translations, with NaN in any
%               (attribute, sweep index) cell whose offset was not
%               uniform across that attribute's positions, and .base is the
%               1 x A cell of untranslated value matrices. Empty for a
%               single translation (M = 1). The offsets are carried
%               rather than recovered: recovering them from the
%               translated values would mean comparing floating-point
%               differences against a tolerance, and no tolerance both
%               admits every honestly translated sweep and preserves the
%               toolbox parity floor.
%
%   Cross-language note. The Python translateAttributes attaches this
%   information to its returned sweep list, so cosSimExpTens picks it up
%   with no change at the call site. MATLAB cell arrays cannot carry
%   attached data, so here it is a fourth output the caller passes on
%   explicitly.
%
%   See also DIFFERENCEEVENTS, BINDEVENTS, FLATSPECS, BUILDEXPTENS,
%   SWEEPCOSSIMEXPTENS.

[pAttr, wAttr, specsPm, rest] = internal.preMaetArgs(varargin);
[pOut, w, specs, sweep] = localTranslateAttributes(pAttr, wAttr, specsPm, rest{:});
if isempty(sweep)
    pm = preMaet(pOut, w, specs);
else
    % Sweep form: pOut holds one length-A cell per sweep index, so the
    % parts do not share a length and the cross-checks do not
    % apply.
    pm = struct('pAttr', {pOut}, 'wAttr', {w}, 'specs', {specs});
end
end


function [pOut, w, specs, sweep] = localTranslateAttributes(pAttr, w, specsPm, offsets, nvArgs)
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
             'and that column is left unchanged. (A non-uniform per-value ' ...
             'offset would apply, as it shifts the relative structure.)']);
end

% --- Apply: value + per-value offset, broadcast across events; NaN values ---
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
    % Carry the offsets with the sweep. A cell is uniform when every
    % value of that attribute moved by the same finite amount (a NaN
    % entry leaves its value in place, so it breaks uniformity unless
    % the whole column is NaN, which is no translation at all).
    uni = NaN(A, Msweep);
    for a = 1:A
        for m = 1:Msweep
            cm = blocks{a}(:, m);
            fin = isfinite(cm);
            if ~any(fin)
                uni(a, m) = 0;
            elseif all(fin) && all(cm == cm(1))
                uni(a, m) = cm(1);
            end
        end
    end
    sweep = struct('offsets', uni, 'base', {pArr});
else
    pOut = colsOut{1};                    % 1 x A cell
    sweep = struct([]);
end

end


% =========================================================================
%  Helpers
% =========================================================================

function [matrixMode, M, blocks] = localNormaliseOffsets(offsets, K, A)
%LOCALNORMALISEOFFSETS  Coerce a 1 x A offsets cell to per-attribute
%   K_a x M blocks (NaN marks skipped values). Orientation disambiguates:
%   scalar -> all values; column -> per-value; row -> per-sweep; matrix ->
%   per-value x sweep. Returns [matrixMode, M, blocks].
    if ~iscell(offsets) || numel(offsets) ~= A
        error('translateAttributes:offsetsShape', ...
              ['offsets must be a length-A (%d) cell, one entry per ' ...
               'attribute ([], scalar, column per-value, row sweep, or ' ...
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
                   'or NaN (NaN skips a row).'], a);
        end
        if isscalar(o)
            raw{a} = o;                   % broadcast to all values
        elseif isrow(o)
            raw{a} = o;                   % per-sweep global shift
            M = max(M, size(o, 2));
        elseif iscolumn(o)
            if numel(o) ~= K(a) && numel(o) ~= 1
                error('translateAttributes:offsetLength', ...
                      ['offsets{%d} is a length-%d column; expected ' ...
                       'K_total = %d (per-value).'], a, numel(o), K(a));
            end
            raw{a} = o;                   % per-value, single
        else
            if size(o, 1) ~= K(a) && size(o, 1) ~= 1
                error('translateAttributes:offsetRows', ...
                      ['offsets{%d} has %d rows; expected 1 or K_total ' ...
                       '= %d (values down).'], a, size(o, 1), K(a));
            end
            raw{a} = o;                   % per-value x sweep
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
                r = repmat(r, K(a), 1);   % broadcast across values
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

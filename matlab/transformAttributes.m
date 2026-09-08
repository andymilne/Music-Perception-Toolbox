function pm = transformAttributes(varargin)
%TRANSFORMATTRIBUTES Map attribute values through named transforms, scale
%   conversions, or user functions.
%
%   PM = transformAttributes(PM0, transforms, ...) and
%   PM = transformAttributes(pAttr, wAttr, transforms, ...) are
%   per-attribute preprocessing on the pre-MAET: every value of each
%   selected attribute is passed through the transform given for that
%   attribute, and the returned pre-MAET feeds straight into buildExpTens
%   or a further pre-MAET step.%
%   The pre-MAET may be passed whole, as preMaet builds it, or in
%   its parts as pAttr and wAttr with the specs as a name-value; the two
%   forms are the same call.
%   In the bare-array form the input is a numeric array rather than a
%   pre-MAET, and the transformed array is returned in its place. Weights pass through unchanged. The map is
%   elementwise, so it composes with the other preprocessors in either
%   order, and the order carries meaning: 'log' THEN differenceEvents
%   gives log ratios (the natural representation of inter-onset-interval
%   ratios, and of intervals from frequencies in Hz), whereas
%   differenceEvents THEN a compressive transform with 'sign', true gives
%   signed compressed magnitudes.
%
%   Bare-array form. When pAttr is a numeric array rather than a cell it
%   is treated as a single attribute and the transformed array is returned
%   alone (w and 'specs' must be empty; 'sign' must be false). This is the
%   one-line conversion that convertPitch used to provide:
%
%       cents = transformAttributes(fHz, [], {'hz', 'cents'});
%
%   Transforms. transforms is a 1 x A cell (one entry per attribute) or a
%   single entry broadcast to every attribute (a cell whose length differs
%   from A is taken as one entry; with A = 2 wrap a scale pair in its own
%   cell, {{'hz', 'cents'}, []}). Each entry is one of:
%
%     []                   - leave the attribute unchanged.
%     'name'               - a named transform with default parameters;
%     {'name', 'p', v, ...}  with name-value parameters:
%
%         name      parameters                     domain          sign
%         'log'     'base' (e), 'offset' (0)       x + offset > 0  yes
%         'power'   'exponent' (required)          x >= 0          yes
%         'affine'  'scale' (1), 'offset' (0)      any             no
%
%       'log' computes log(x + offset) / log(base); with the default
%       offset a zero is refused, and admitting zeros means writing the
%       constant down (log(x + c) is not unit-free: the unit of x is then
%       part of the model). 'affine' is the only transform compatible
%       with a periodic attribute (isPer true at build); the others
%       change the metric and so cannot be wrapped.
%     {'from', 'to'}       - a pitch-scale conversion among 'hz', 'midi',
%                            'cents' (100 x MIDI), 'octave' (MIDI / 12),
%                            'mel', 'bark', 'erb', 'greenwood'; every pair
%                            routes through Hz.
%     struct               - struct('name', ..., param, ...) or
%                            struct('from', ..., 'to', ...).
%     @f                   - a function handle applied to the attribute's
%                            K_total x N value matrix; the result must have
%                            the same shape and be finite everywhere.
%
%   Domain. Values outside a transform's domain are refused with a message
%   naming the attribute and events and the remedies. In particular a zero
%   under 'log' (a zero inter-onset interval, say) is an error rather than
%   -Inf: bind simultaneous events first, drop them deliberately, or admit
%   them with an explicit 'offset'.
%
%   Name-value pairs
%       'specs' - [] (synthesise flat via flatSpecs) or a 1 x A cell of
%                 attribute specifications.
%       'sign'  - false (default), true, or a 1 x A logical vector. On a
%                 magnitude transform ('log' or 'power') the transform is applied
%                 to |x| and a SIGN ATTRIBUTE with values in {-1/2, 0, +1/2}
%                 is inserted immediately after the source attribute. The
%                 attribute count grows by one for each such attribute, so
%                 downstream per-attribute arguments (sigma, r, rel, sym,
%                 wrap, diffOrders, ...) must include the new column; this
%                 is why the insertion is explicit rather than automatic.
%                 The sign attribute copies the source's spec with rel
%                 cleared and the name suffixed '_sign', and its weights
%                 (when w is a per-attribute cell) copy the source's. A
%                 small kernel width on the sign attribute makes it
%                 effectively categorical.
%
%   Output
%       pm - Pre-MAET: pAttr is a 1 x A' cell of K_total x N
%            matrices (A' = A plus the number of sign attributes), wAttr
%            is as input, extended for sign attributes, and specs is a
%            1 x A' cell, extended likewise. In the bare-array form the
%            transformed numeric array is returned instead.
%
%   See also PREMAET, DIFFERENCEEVENTS, BINDEVENTS, TRANSLATEATTRIBUTES,
%   WEIGHTEVENTS, BUILDEXPTENS.

if ~isempty(varargin) && (isnumeric(varargin{1}) || islogical(varargin{1}))
    % Bare-array form: one array in, the transformed array out. There is
    % no pre-MAET here, so there is none to return.
    rest = varargin(2:end);
    if isempty(rest)
        wBare = [];
    else
        wBare = rest{1};
        rest = rest(2:end);
    end
    pm = localTransformAttributes(varargin{1}, wBare, [], rest{:});
    return;
end
[pAttr, wAttr, specsPm, rest] = internal.preMaetArgs(varargin);
[pOut, w, specs] = localTransformAttributes(pAttr, wAttr, specsPm, rest{:});
pm = preMaet(pOut, w, specs);
end


function [pOut, w, specs] = localTransformAttributes(pAttr, w, specsPm, transforms, nvArgs)
arguments
    pAttr
    w = []
    specsPm = []
    transforms = []
    nvArgs.specs = []
    nvArgs.sign = false
end

if isempty(nvArgs.specs)
    nvArgs.specs = specsPm;
end

% --- Bare-array form ---
if isnumeric(pAttr) || islogical(pAttr)
    if ~isempty(w) || ~isempty(nvArgs.specs)
        error('transformAttributes:bareForm', ...
              ['In the bare-array form (pAttr a numeric array) w and ' ...
               '''specs'' must be empty.']);
    end
    if any(logical(nvArgs.sign))
        error('transformAttributes:bareForm', ...
              ['In the bare-array form ''sign'' must be false; use the ' ...
               'cell form to append a sign attribute.']);
    end
    x = double(pAttr);
    if isvector(x)
        xm = reshape(x, 1, []);
    else
        xm = x;
    end
    out = localTransformAttributes({xm}, [], [], {transforms});
    pOut = reshape(out{1}, size(x));
    return;
end

% --- Normalise pAttr ---
if ~iscell(pAttr)
    error('transformAttributes:badPAttrType', ...
          'pAttr must be a cell array of attribute value matrices.');
end
A = numel(pAttr);
if A < 1
    error('transformAttributes:noAttrs', ...
          'pAttr must contain at least one attribute.');
end
pArr = cell(1, A);
for a = 1:A
    M = pAttr{a};
    if ~isnumeric(M) || ndims(M) > 2
        error('transformAttributes:badAttrShape', ...
              'Attribute %d must be a numeric 1-D or 2-D matrix.', a);
    end
    pArr{a} = double(M);
end
nEvents = size(pArr{1}, 2);
for a = 2:A
    if size(pArr{a}, 2) ~= nEvents
        error('transformAttributes:eventCountMismatch', ...
              ['All attributes must share the same event count N. ' ...
               'Attribute 1 has N=%d; attribute %d has N=%d.'], ...
              nEvents, a, size(pArr{a}, 2));
    end
end

% --- Attribute specifications ---
if isempty(nvArgs.specs)
    specsIn = flatSpecs(pArr);
else
    specsIn = nvArgs.specs;
    if ~iscell(specsIn) || numel(specsIn) ~= A
        error('transformAttributes:specsLength', ...
              'specs must be a length-A (%d) cell, one per attribute.', A);
    end
end

% --- Transforms: per-attribute cell, or one entry broadcast ---
if iscell(transforms) && numel(transforms) == A
    entries = transforms;
else
    entries = repmat({transforms}, 1, A);
end
parsed = cell(1, A);
for a = 1:A
    parsed{a} = localParseTransform(entries{a}, a);
end

% --- Sign flags ---
sgn = nvArgs.sign;
if isscalar(sgn)
    signV = repmat(logical(sgn), 1, A);
else
    if numel(sgn) ~= A
        error('transformAttributes:signLength', ...
              'sign must be a logical scalar or a length-A (%d) vector.', A);
    end
    signV = logical(sgn(:)).';
end
for a = 1:A
    if signV(a) && (isempty(parsed{a}) || ~parsed{a}.magnitude)
        if isempty(parsed{a})
            what = 'no transform';
        else
            what = parsed{a}.label;
        end
        error('transformAttributes:signNotMagnitude', ...
              ['%s: sign=true applies only to a magnitude transform ' ...
               '(''log'' or ''power''); got %s.'], ...
              localAttrLabel(a, specsIn{a}), what);
    end
end

% --- Weights as a per-attribute cell when they are one ---
wList = [];
if iscell(w)
    if numel(w) ~= A
        error('transformAttributes:weightsLength', ...
              'w must be [], a scalar, or a length-A (%d) cell.', A);
    end
    wList = w;
end

pOut = {};
wOut = {};
specs = {};
for a = 1:A
    x = pArr{a};
    tr = parsed{a};
    if isempty(tr)
        pOut{end+1} = x; %#ok<AGROW>
        specs{end+1} = specsIn{a}; %#ok<AGROW>
        if iscell(wList)
            wOut{end+1} = wList{a}; %#ok<AGROW>
        end
        continue;
    end
    if ~all(isfinite(x(:)))
        error('transformAttributes:nonFiniteInput', ...
              '%s: values must be finite; non-finite at %s.', ...
              localAttrLabel(a, specsIn{a}), localOffending(~isfinite(x)));
    end
    if signV(a)
        src = abs(x);
    else
        src = x;
    end
    localCheckDomain(x, src, tr, a, specsIn{a}, signV(a));
    y = double(localApply(tr, src, a, specsIn{a}));
    if ~isequal(size(y), size(x))
        error('transformAttributes:badOutputShape', ...
              '%s: %s returned size [%s]; expected [%s].', ...
              localAttrLabel(a, specsIn{a}), tr.label, ...
              num2str(size(y)), num2str(size(x)));
    end
    bad = ~isfinite(y);
    if any(bad(:))
        vals = src(bad);
        error('transformAttributes:nonFiniteOutput', ...
              ['%s: %s produced non-finite values at %s (inputs %s). ' ...
               'The transform must return a finite value for every ' ...
               'input.'], localAttrLabel(a, specsIn{a}), tr.label, ...
              localOffending(bad), ...
              strjoin(arrayfun(@(v) sprintf('%g', v), ...
                               vals(1:min(6, numel(vals))).', ...
                               'UniformOutput', false), ', '));
    end
    pOut{end+1} = y; %#ok<AGROW>
    specs{end+1} = localTransformedSpec(specsIn{a}, tr, signV(a)); %#ok<AGROW>
    if iscell(wList)
        wOut{end+1} = wList{a}; %#ok<AGROW>
    end
    if signV(a)
        % The two signs are the vertices of the 2-point simplex, at the
        % toolbox's default unit edge length (simplexVertices(2) is
        % [+1/2, -1/2]), with a zero step at the centroid. Coding them
        % +/-1 would put them at edge length 2 and so on a different
        % scale from every other categorical attribute.
        pOut{end+1} = 0.5 * sign(x); %#ok<AGROW>
        % The sign axis is a fresh three-level categorical: its width is
        % a modelling choice, not an image of the source's.
        sgn = localSignSpec(specsIn{a});
        for kf = {'sigma', 'period'}
            if isfield(sgn, kf{1}); sgn.(kf{1}) = NaN; end
        end
        specs{end+1} = sgn; %#ok<AGROW>
        if iscell(wList)
            wOut{end+1} = wList{a}; %#ok<AGROW>
        end
    end
end
if iscell(wList)
    w = wOut;
end
end


% =====================================================================
%  Scale conversions (routed through each family's base scale)
% =====================================================================

function names = localPitchScales()
    names = {'hz', 'midi', 'cents', 'octave', 'mel', 'bark', 'erb', 'greenwood'};
end

function out = localConvertScale(values, fromScale, toScale)
    src = lower(char(fromScale));
    tgt = lower(char(toScale));
    for s = {src, tgt}
        if ~any(strcmp(s{1}, localPitchScales()))
            error('transformAttributes:unknownScale', ...
                  'Unknown scale ''%s''. Known scales: %s.', ...
                  s{1}, strjoin(localPitchScales(), ', '));
        end
    end
    if strcmp(src, tgt)
        out = values;
        return;
    end
    out = localPitchFromHz(localPitchToHz(values, src), tgt);
end

function f = localPitchToHz(v, scale)
    % 'octave' is MIDI / 12 (octaves above MIDI 0 = 8.1758 Hz), sharing
    % the origin of 'midi' and 'cents' rather than needing a reference.
    switch scale
        case 'hz',        f = v;
        case 'midi',      f = 440 * 2 .^ ((v - 69) / 12);
        case 'cents',     f = 440 * 2 .^ ((v - 6900) / 1200);
        case 'octave',    f = 440 * 2 .^ (v - 69 / 12);
        case 'mel',       f = 700 * (10 .^ (v / 2595) - 1);
        case 'bark',      f = 1960 * (v + 0.53) ./ (26.28 - v);
        case 'erb',       f = (10 .^ (v / 21.4) - 1) / 0.00437;
        case 'greenwood', f = 165.4 * (10 .^ (2.1 * v) - 0.88);
    end
end

function out = localPitchFromHz(f, scale)
    switch scale
        case 'hz',        out = f;
        case 'midi',      out = 69 + 12 * log2(f / 440);
        case 'cents',     out = 6900 + 1200 * log2(f / 440);
        case 'octave',    out = 69 / 12 + log2(f / 440);
        case 'mel',       out = 2595 * log10(1 + f / 700);
        case 'bark',      out = 26.81 ./ (1 + 1960 ./ f) - 0.53;
        case 'erb',       out = 21.4 * log10(0.00437 * f + 1);
        case 'greenwood', out = log10(f / 165.4 + 0.88) / 2.1;
    end
end

% =====================================================================
%  Named transforms
% =====================================================================

function tbl = localNamedTable()
    % name -> domain ('any' | 'nonneg' | 'positive'), default parameters
    %         (struct), required parameters, magnitude. 'log' takes
    %         log(x + offset): the domain is x + offset > 0, so with the
    %         default offset = 0 a zero is refused, and admitting zeros
    %         means writing the constant down (log(x + c) is not unit-free).
    tbl = struct();
    tbl.log    = struct('domain', 'positive', 'defaults', struct('base', exp(1), 'offset', 0), 'required', {{}}, 'magnitude', true);
    tbl.power  = struct('domain', 'nonneg',   'defaults', struct(), 'required', {{'exponent'}}, 'magnitude', true);
    tbl.affine = struct('domain', 'any',      'defaults', struct('scale', 1, 'offset', 0), 'required', {{}}, 'magnitude', false);
end

function y = localApplyNamed(name, x, p)
    switch name
        case 'log',    y = log(x + p.offset) / log(p.base);
        case 'power',  y = x .^ p.exponent;
        case 'affine', y = p.scale * x + p.offset;
    end
end

function d = localScaleDomain(src)
    % Source scales with a restricted domain.
    switch src
        case 'hz',           d = 'positive';
        case {'mel', 'erb'}, d = 'nonneg';
        otherwise,           d = 'any';
    end
end


% =====================================================================
%  Parsing a transform entry
% =====================================================================

function tr = localParseTransform(entry, a)
    % Returns [] for the identity, otherwise a struct with fields
    % kind ('named' | 'scale' | 'callable'), name, params, domain,
    % magnitude, label.
    tr = [];
    if isempty(entry)
        return;
    end
    where = sprintf('transforms{%d}', a);
    if isa(entry, 'function_handle')
        tr = struct('kind', 'callable', 'name', func2str(entry), ...
                    'func', entry, 'params', struct(), 'domain', 'any', ...
                    'magnitude', false, 'label', 'user function');
        return;
    end
    if isstruct(entry)
        fn = lower(fieldnames(entry));
        vals = struct2cell(entry);
        s = cell2struct(vals, fn, 1);
        if isfield(s, 'from') || isfield(s, 'to')
            if ~(isfield(s, 'from') && isfield(s, 'to'))
                error('transformAttributes:badTransform', ...
                      '%s: a scale conversion needs both ''from'' and ''to''.', where);
            end
            params = rmfield(s, {'from', 'to'});
            tr = localScaleTransform(s.from, s.to, params, where);
            return;
        end
        if ~isfield(s, 'name')
            error('transformAttributes:badTransform', ...
                  ['%s: a struct transform needs ''name'' (or ''from''/''to'' ' ...
                   'for a scale conversion).'], where);
        end
        params = rmfield(s, 'name');
        tr = localNamedTransform(s.name, params, where);
        return;
    end
    if ischar(entry) || (isstring(entry) && isscalar(entry))
        tr = localNamedTransform(char(entry), struct(), where);
        return;
    end
    if iscell(entry)
        if isempty(entry) || ~(ischar(entry{1}) || isstring(entry{1}))
            error('transformAttributes:badTransform', ...
                  ['%s: a cell transform must start with a name: ' ...
                   '{''name'', ''param'', value, ...} or a scale pair ' ...
                   '{''from'', ''to''}.'], where);
        end
        first = lower(char(entry{1}));
        if any(strcmp(first, localPitchScales()))
            % Scale pair, optionally followed by name-value parameters.
            if numel(entry) < 2 || ~(ischar(entry{2}) || isstring(entry{2}))
                error('transformAttributes:badTransform', ...
                      ['%s: ''%s'' is a scale name; a scale conversion is a ' ...
                       'pair {''%s'', ''cents''}.'], where, first, first);
            end
            params = localNvToStruct(entry(3:end), where);
            tr = localScaleTransform(first, char(entry{2}), params, where);
            return;
        end
        params = localNvToStruct(entry(2:end), where);
        tr = localNamedTransform(first, params, where);
        return;
    end
    error('transformAttributes:badTransform', ...
          ['%s: unrecognised transform. Use a name (''log'', ''power'', ' ...
           '''affine''), a scale pair such as {''hz'', ''cents''}, a ' ...
           'struct, or a function handle; [] leaves the attribute unchanged.'], ...
          where);
end

function s = localNvToStruct(nv, where)
    if mod(numel(nv), 2) ~= 0
        error('transformAttributes:badTransform', ...
              '%s: parameters must be name-value pairs.', where);
    end
    s = struct();
    for i = 1:2:numel(nv)
        if ~(ischar(nv{i}) || isstring(nv{i}))
            error('transformAttributes:badTransform', ...
                  '%s: parameter names must be char.', where);
        end
        s.(lower(char(nv{i}))) = nv{i+1};
    end
end

function tr = localNamedTransform(name, params, where)
    key = lower(char(name));
    tbl = localNamedTable();
    names = fieldnames(tbl);
    if ~isfield(tbl, key)
        hint = '';
        if any(strcmp(key, localPitchScales()))
            hint = sprintf([' ''%s'' is a scale name; a scale conversion is a ' ...
                            'pair such as {''%s'', ''cents''}, and with A = 2 ' ...
                            'a pair must be wrapped in its own cell so that it ' ...
                            'is not read as two per-attribute transforms.'], key, key);
        end
        error('transformAttributes:unknownTransform', ...
              ['%s: unknown transform ''%s''. Known transforms: %s; scale ' ...
               'conversions are given as a pair {''from'', ''to''}.%s'], ...
              where, char(name), strjoin(names.', ', '), hint);
    end
    row = tbl.(key);
    given = fieldnames(params);
    allowed = [fieldnames(row.defaults); row.required(:)];
    unknown = setdiff(given, allowed);
    if ~isempty(unknown)
        error('transformAttributes:badParameter', ...
              '%s: ''%s'' does not take parameter(s) %s; it takes %s.', ...
              where, key, strjoin(unknown.', ', '), strjoin(sort(allowed).', ', '));
    end
    missing = setdiff(row.required, given);
    if ~isempty(missing)
        error('transformAttributes:badParameter', ...
              '%s: ''%s'' requires parameter(s) %s.', where, key, ...
              strjoin(reshape(missing, 1, []), ', '));
    end
    full = row.defaults;
    for i = 1:numel(given)
        full.(given{i}) = double(params.(given{i}));
    end
    if strcmp(key, 'log') && ~(full.base > 0 && full.base ~= 1)
        error('transformAttributes:badParameter', ...
              '%s: base must be positive and not 1.', where);
    end
    tr = struct('kind', 'named', 'name', key, 'func', [], 'params', full, ...
                'domain', row.domain, 'magnitude', row.magnitude, ...
                'label', sprintf('''%s''', key));
end

function tr = localScaleTransform(src, tgt, params, where)
    src = lower(char(src));
    tgt = lower(char(tgt));
    given = fieldnames(params);
    if ~isempty(given)
        error('transformAttributes:badParameter', ...
              '%s: a scale conversion takes no parameters; got %s.', where, ...
              strjoin(given.', ', '));
    end
    % Validate the pair now so that a bad name is reported before any
    % values are touched.
    localConvertScale(1, src, tgt);
    tr = struct('kind', 'scale', 'name', {{src, tgt}}, 'func', [], ...
                'params', struct(), 'domain', localScaleDomain(src), ...
                'magnitude', false, ...
                'label', sprintf('{''%s'', ''%s''}', src, tgt));
end

function y = localApply(tr, x, a, spec)
    switch tr.kind
        case 'scale'
            y = localConvertScale(x, tr.name{1}, tr.name{2});
        case 'named'
            try
                y = localApplyNamed(tr.name, x, tr.params);
            catch ME
                if strcmp(ME.identifier, 'transformAttributes:badParameter')
                    error('transformAttributes:badParameter', '%s: %s: %s', ...
                          localAttrLabel(a, spec), tr.label, ME.message);
                end
                rethrow(ME);
            end
        otherwise
            y = tr.func(x);
    end
end


% =====================================================================
%  Domain checks and messages
% =====================================================================

function s = localAttrLabel(a, spec)
    s = sprintf('attribute %d', a);
    if isstruct(spec) && isfield(spec, 'name') && ~isempty(spec.name)
        s = sprintf('%s (''%s'')', s, char(spec.name));
    end
end

function s = localOffending(mask)
    % Format the first few (row, event) indices where mask holds
    % (1-based, matching MATLAB indexing).
    [r, c] = find(mask);
    n = numel(r);
    lim = min(6, n);
    parts = cell(1, lim);
    for i = 1:lim
        parts{i} = sprintf('(value %d, event %d)', r(i), c(i));
    end
    s = strjoin(parts, ', ');
    if n > lim
        s = sprintf('%s, ... (%d in all)', s, n);
    end
end

function localCheckDomain(x, src, tr, a, spec, signOn)
    % x is the attribute's values and src the array the transform will
    % see (|x| when the sign attribute is requested). For 'log' the
    % domain applies to src + offset.
    label = localAttrLabel(a, spec);
    if strcmp(tr.domain, 'any')
        return;
    end
    neg = x < 0;
    if any(neg(:)) && ~signOn
        remedy = '';
        if tr.magnitude
            remedy = [' Pass ''sign'', true for this attribute to transform ' ...
                      'the magnitudes |x| and append a sign attribute ' ...
                      '(-1/2, 0, +1/2) immediately after it.'];
        end
        error('transformAttributes:domainNegative', ...
              '%s: %s is undefined for negative values; negatives at %s.%s', ...
              label, tr.label, localOffending(neg), remedy);
    end
    if ~strcmp(tr.domain, 'positive')
        return;
    end
    offset = 0;
    if isfield(tr.params, 'offset')
        offset = tr.params.offset;
    end
    bad = (src + offset) <= 0;
    if any(bad(:))
        if strcmp(tr.kind, 'named') && offset ~= 0
            error('transformAttributes:domainZero', ...
                  ['%s: %s with offset %g is undefined where x + offset <= 0; ' ...
                   'values at %s.'], label, tr.label, offset, localOffending(bad));
        end
        error('transformAttributes:domainZero', ...
              ['%s: %s is undefined at zero; zero values at %s. Remedies, ' ...
               'in the usual order of preference: bind simultaneous events ' ...
               'first (bindEvents) if the zeros are the inter-onset ' ...
               'intervals of chords or grace notes; drop those events ' ...
               'deliberately; or admit them with an explicit offset, ' ...
               '{''log'', ''offset'', c} = log(x + c), bearing in mind that ' ...
               'log(x + c) is not unit-free, so the unit of x is then part ' ...
               'of the model.'], label, tr.label, localOffending(bad));
    end
end

function s = localSignSpec(spec)
    % The spec of a sign attribute: the source's structure, rel cleared,
    % name suffixed '_sign'.
    if ~isstruct(spec)
        s = struct('r', 1, 'rel', false, 'sym', true, 'name', 'sign');
        return;
    end
    s = spec;
    if isfield(s, 'rel')
        rel = s.rel;
        if ischar(rel) || isstring(rel)
            s.rel = false;
        elseif numel(rel) > 1
            s.rel = false(size(rel));
        else
            s.rel = false;
        end
    else
        s.rel = false;
    end
    if isfield(s, 'name') && ~isempty(s.name)
        s.name = [char(s.name) '_sign'];
    else
        s.name = 'sign';
    end
end


function out = localTransformedSpec(spec, tr, magnitude)
%LOCALTRANSFORMEDSPEC  One attribute's spec after a transform.
%
%   An affine map (or a conversion within the log-frequency family)
%   carries the width and the period across by the same constant; a
%   covariance, being in squared units, takes its square. Anything
%   non-linear leaves NA -- not because a width would be meaningless in
%   the new coordinate, but because there is no canonical value to carry
%   over, the local scaling varying across the attribute's range; the
%   analyst supplies the width the new units call for. Taking magnitudes
%   for a sign attribute folds the axis, which no width survives either.
    out = spec;
    if magnitude
        gain = [];
    else
        gain = localTransformGain(tr);
    end
    for kf = {'sigma', 'period'}
        key = kf{1};
        if ~isfield(out, key) || isempty(out.(key))
            continue;
        end
        if isempty(gain)
            out.(key) = NaN;
        elseif isscalar(out.(key))
            out.(key) = double(out.(key)) * gain;
        else
            out.(key) = double(out.(key)) * gain^2;
        end
    end
end


function gain = localTransformGain(tr)
%LOCALTRANSFORMGAIN  The constant a transform multiplies a width by, or [].
%
%   [] means no width carries across. A width remains perfectly
%   meaningful in the new coordinate -- after a log it is a width on the
%   log axis, so it expresses a ratio rather than a difference -- but a
%   non-linear map has a local scaling that varies with position, so no
%   single value is the image of the old one. The caller records NA to
%   say there is no canonical choice, not that a width is meaningless. The
%   log-frequency scales (midi, cents, octave) are affine images of one
%   another, differing only in unit; every other scale (Hz, mel, bark,
%   erb, greenwood) is a non-linear map of these.
    logFreq = struct('midi', 1, 'cents', 100, 'octave', 1/12);
    gain = [];
    if isempty(tr)
        gain = 1;
        return;
    end
    switch tr.kind
        case 'scale'
            src = lower(tr.name{1}); tgt = lower(tr.name{2});
            if isfield(logFreq, src) && isfield(logFreq, tgt)
                gain = logFreq.(tgt) / logFreq.(src);
            end
        case 'named'
            if strcmp(tr.name, 'affine')
                gain = abs(double(tr.params.scale));
            elseif strcmp(tr.name, 'power') && ...
                    double(tr.params.exponent) == 1
                gain = 1;
            end
    end
end

function specs = flatSpecs(pAttr, nvArgs)
%FLATSPECS Build a cell of flat (one-level) specs for bare attributes.
%
%   specs = flatSpecs(pAttr, 'r', r, 'rel', rel, 'sym', sym, 'name', name)
%   is a convenience constructor for the canonical attribute
%   specifications: it wraps a cell of per-attribute value matrices in
%   flat spec structs
%   struct('r', ., 'rel', ., 'sym', .[, 'name', .]), broadcasting scalar
%   geometry across attributes. This is the trivial flat-specs synthesis at
%   the entry of a pre-MAET chain (raw attributes carry no level structure
%   yet) and an ergonomic alternative to hand-writing flat structs for
%   buildExpTens(..., 'specs', specs).
%
%   Inputs
%       pAttr - 1 x A cell of per-attribute value matrices (used only for
%               its length A; values are not inspected).
%
%   Name-value pairs
%       'r'    - scalar or 1 x A per-attribute tuple size (default 1).
%       'rel'  - scalar or 1 x A [rel] (default false).
%       'sym'  - scalar or 1 x A [sym] (default true).
%       'name' - [], char, or 1 x A cell of per-attribute names.
%
%   Output
%       specs - 1 x A cell of flat spec structs, ready for
%               buildExpTens('specs', specs) or to thread through the
%               pre-MAET operators.
%
%   See also BUILDEXPTENS, BINDEVENTS, DIFFERENCEEVENTS.

arguments
    pAttr
    nvArgs.r = 1
    nvArgs.rel = false
    nvArgs.sym = true
    nvArgs.name = []
end

if ~iscell(pAttr)
    error('flatSpecs:badPAttr', ...
          'pAttr must be a cell of per-attribute matrices.');
end
A = numel(pAttr);
rV   = localBcast(nvArgs.r,   A, 'r',   false);
relV = localBcast(nvArgs.rel, A, 'rel', true);
symV = localBcast(nvArgs.sym, A, 'sym', true);
names = localNames(nvArgs.name, A);

specs = cell(1, A);
for a = 1:A
    s = struct('r', rV(a), 'rel', relV(a), 'sym', symV(a));
    if ~isempty(names{a})
        s.name = names{a};
    end
    specs{a} = s;
end
end


function out = localBcast(x, A, name, asLogical)
    if ~isnumeric(x) && ~islogical(x)
        error('flatSpecs:bcastType', '%s must be numeric or logical.', name);
    end
    v = x(:).';
    if numel(v) == 1
        out = repmat(v, 1, A);
    elseif numel(v) == A
        out = v;
    else
        error('flatSpecs:bcastLength', ...
              '%s must be scalar or length-A (%d); got %d.', name, A, numel(v));
    end
    if asLogical
        out = logical(out);
    else
        out = double(out);
    end
end


function names = localNames(name, A)
    names = cell(1, A);
    if isempty(name)
        return;
    end
    if ischar(name) || (isstring(name) && isscalar(name))
        for a = 1:A; names{a} = char(name); end
        return;
    end
    if iscell(name)
        if numel(name) ~= A
            error('flatSpecs:nameLength', 'name cell must be length-A (%d).', A);
        end
        for a = 1:A; names{a} = name{a}; end
        return;
    end
    error('flatSpecs:nameType', 'name must be [], a char, or a 1 x A cell.');
end

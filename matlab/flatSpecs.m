function specs = flatSpecs(pAttr, nvArgs)
%FLATSPECS Build a cell of flat (one-level) specs for bare attributes.
%
%   specs = flatSpecs(pAttr, 'r', r, 'rel', rel, 'sym', sym, 'name', name)
%   specs = flatSpecs(pAttr, ..., 'sigma', s, 'isPer', per, 'period', P)
%   is a convenience constructor for the canonical attribute
%   specifications: it wraps a cell of per-attribute value matrices in
%   flat spec structs
%   struct('r', ., 'rel', ., 'sym', .[, 'name', .][, 'sigma', .,
%   'isPer', ., 'period', .]), broadcasting scalar geometry across
%   attributes. This is the trivial flat-specs synthesis at the entry of a
%   pre-MAET chain (raw attributes carry no level structure yet) and an
%   ergonomic alternative to hand-writing flat structs for
%   buildExpTens(..., 'specs', specs).
%
%   The kernel parameters are optional here and compulsory at the tensor
%   (User Guide 3.7.10). Given them, the specs are a complete pre-MAET
%   geometry and nothing further need be supplied at the build:
%
%     pm = preMaet(pAttr, wAttr, flatSpecs(pAttr, 'r', [2 1], ...
%              'sigma', [0.5 0.25], 'isPer', [true false], ...
%              'period', [12 0]));
%     dens = buildExpTens(pm);
%
%   Omitted, they are simply absent from the specs, and buildExpTens then
%   names the attribute that still needs one. NaN is NA, the third state:
%   a width that a step could not carry forward.
%
%   Inputs
%       pAttr - 1 x A cell of per-attribute value matrices (used only for
%               its length A; values are not inspected).
%
%   Name-value pairs
%       'r'    - scalar or 1 x A per-attribute tuple size (default 1).
%       'rel'  - scalar or 1 x A [rel] (default false).
%       'sym'  - scalar or 1 x A [sym] (default true).
%       'name'   - [], char, or 1 x A cell of per-attribute names.
%       'sigma'  - [] or scalar / 1 x A per-attribute kernel width; a
%                  per-attribute entry may be a matrix-valued kernel
%                  covariance.
%       'isPer'  - [] or scalar / 1 x A per-attribute periodicity.
%       'period' - [] or scalar / 1 x A per-attribute period (inert where
%                  not periodic).
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
    nvArgs.sigma = []
    nvArgs.isPer = []
    nvArgs.period = []
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
    kernelNames = {'sigma', 'isPer', 'period'};
    kernelVals = {nvArgs.sigma, nvArgs.isPer, nvArgs.period};
    for f = 1:numel(kernelNames)
        if isempty(kernelVals{f})
            continue;
        end
        v = localBcastKernel(kernelVals{f}, A, kernelNames{f});
        s.(kernelNames{f}) = v{a};
    end
    specs{a} = s;
end
end


function out = localBcastKernel(v, A, what)
%LOCALBCASTKERNEL  Broadcast an optional kernel parameter across attributes.
%   Unlike the structural geometry, an entry may be a matrix (a kernel
%   covariance), so the value is not coerced to a numeric row: a scalar or
%   a bare matrix applies to every attribute, and a 1 x A cell is taken
%   per attribute.
    if iscell(v)
        if numel(v) ~= A
            error('flatSpecs:badKernelLength', ...
                  '''%s'' must be a scalar or have length A = %d; got %d.', ...
                  what, A, numel(v));
        end
        out = v;
        return;
    end
    if ~isvector(v) || isscalar(v)
        out = repmat({v}, 1, A);       % one covariance, or one scalar
        return;
    end
    if numel(v) ~= A
        error('flatSpecs:badKernelLength', ...
              '''%s'' must be a scalar or have length A = %d; got %d.', ...
              what, A, numel(v));
    end
    out = num2cell(v(:).');
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

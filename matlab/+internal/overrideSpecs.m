function specs = overrideSpecs(specs, rKw, relKw, symKw, A)
%OVERRIDESPECS  Apply the r / rel / sym keyword overrides to specs.
%
%   The kernel parameters sigma, isPer, and period are resolved after the
%   specs are read, so a keyword can override them there. The tuple size
%   and the [rel] and [sym] flags are read out of the specs themselves, so
%   an override has to be written into the specs first. A supplied keyword
%   wins for every attribute, exactly as it does for the kernel
%   parameters, which is what lets a sweep over any of the six per-
%   attribute parameters stay a single call.
%
%   An override may be SELECTIVE: a 1 x A cell whose empty entries keep
%   what the spec carries, so a sweep names only the attribute it varies.
%
%   Level-structured geometry is excluded: on a nested attribute r, rel,
%   and sym are per-level vectors whose meaning depends on the nesting, so
%   a scalar override has no unambiguous reading and the spec is the place
%   to change them.
    names = {'r', 'rel', 'sym'};
    vals = {rKw, relKw, symKw};
    for f = 1:numel(names)
        v = vals{f};
        if isempty(v)
            continue;
        end
        v = localBcastOverride(v, A, names{f});
        for a = 1:A
            if isempty(v{a})            % selective: keep the spec's
                continue;
            end
            sp = specs{a};
            if ~isstruct(sp)
                error('buildExpTens:overrideNotStruct', ...
                      ['''%s'' needs each specs entry to be a struct; ' ...
                       'attribute %d is not.'], names{f}, a);
            end
            if isfield(sp, 'tags')
                error('buildExpTens:overrideNested', ...
                      ['''%s'' cannot override a nested attribute ' ...
                       '(attribute %d); on a nested attribute %s is ' ...
                       'per-level, so set it in the spec.'], ...
                      names{f}, a, names{f});
            end
            if strcmp(names{f}, 'r')
                sp.r = double(v{a});
            else
                sp.(names{f}) = logical(v{a});
            end
            specs{a} = sp;
        end
    end
end


function v = localBcastOverride(v, A, what)
    if internal.isSelectiveOverride(v, A)
        return;                      % a cell; empty entries keep the spec
    end
    if iscell(v)
        v = cell2mat(v);
    end
    v = double(v(:)).';
    if isscalar(v)
        v = repmat(v, 1, A);
    elseif numel(v) ~= A
        error('buildExpTens:overrideLength', ...
              '''%s'' must be a scalar or have length A = %d; got %d.', ...
              what, A, numel(v));
    end
    v = num2cell(v);
end

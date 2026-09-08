function out = resolveKernelParam(given, fromSpecs, what, names, A, hasDefault)
%RESOLVEKERNELPARAM  One kernel parameter per attribute, keyword or spec.
%
%   A supplied keyword wins for every attribute; otherwise the specs are
%   read. The keyword may also be SELECTIVE -- a 1 x A cell whose empty
%   entries keep what the spec carries -- which is what lets a sweep name
%   the one attribute it varies and leave the rest to the pre-MAET. An
%   entry that is NA (a preprocessing step could not carry it forward) or
%   absent from both is an error naming the attribute, since the tensor
%   cannot be built without it -- the pre-MAET is complete at the
%   boundary even where the pre-MAET was not. 'period' has a default of
%   0, being inert on a non-periodic attribute.
    if ~isempty(given)
        if ~internal.isSelectiveOverride(given, A)
            out = given;
            return;
        end
        out = fromSpecs;
        for a = 1:A
            if ~isempty(given{a})
                out{a} = given{a};
            end
        end
    else
        out = fromSpecs;
    end
    anyCov = false;
    for a = 1:A
        v = out{a};
        isNA = isempty(v) || ((isnumeric(v) || islogical(v)) ...
                              && isscalar(v) && any(isnan(double(v))));
        if ~isNA
            anyCov = anyCov || (isnumeric(v) && ~isscalar(v));
            continue;
        end
        if hasDefault
            out{a} = 0;
            continue;
        end
        if isempty(names{a})
            who = sprintf('%d', a);
        else
            who = sprintf('''%s''', names{a});
        end
        if isempty(v)
            error('buildExpTens:kernelMissing', ...
                ['No %s for attribute %s: give it in the spec ' ...
                 '(specs{%d}.%s) or pass %s to buildExpTens.'], ...
                what, who, a, what, what);
        end
        error('buildExpTens:kernelNA', ...
            ['%s for attribute %s is NA: a preprocessing step could ' ...
             'not carry it forward, so it must be supplied again -- ' ...
             'set specs{%d}.%s or pass %s to buildExpTens.'], ...
            what, who, a, what, what);
    end
    % A cell of scalars collapses to the numeric vector the downstream
    % path expects; a cell holding a kernel covariance must stay a cell.
    isScalarNum = @(v) (isnumeric(v) || islogical(v)) && isscalar(v);
    if ~anyCov && all(cellfun(isScalarNum, out))
        out = cellfun(@double, out);
    end
end

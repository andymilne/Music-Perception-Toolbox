function args = expandPreMaetPair(args)
%EXPANDPREMAETPAIR  Replace a pre-MAET in buildExpTens's arguments.
%   buildExpTens's leading positional arguments are p and w, so a whole
%   pre-MAET in the p slot expands into those two, and its specs are
%   attached as the 'specs' name-value where the call did not name specs
%   itself.
    if isempty(args) || ~internal.isPreMaet(args{1})
        return;
    end
    pm = args{1};
    hasSpecs = false;
    for k = 2:numel(args)
        if (ischar(args{k}) || (isstring(args{k}) && isscalar(args{k}))) ...
                && strcmpi(char(args{k}), 'specs')
            hasSpecs = true;
            break;
        end
    end
    args = [{pm.pAttr, pm.wAttr}, args(2:end)];
    if ~isempty(pm.specs) && ~hasSpecs
        args = [args, {'specs', pm.specs}];
    end
end

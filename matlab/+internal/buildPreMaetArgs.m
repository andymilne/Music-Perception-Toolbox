function args = buildPreMaetArgs(args)
%BUILDPREMAETARGS  Build any whole pre-MAET among positional arguments.
%   A pre-MAET is a density in waiting: it holds everything buildExpTens
%   needs, so it stands wherever a density does and is built here. That
%   goes for a CELL of them too: a cell of pre-MAETs stands wherever a
%   cell of densities does, so the list and scalar-vs-list forms take
%   them without the caller building each one first.
%
%   The sweep form of translateAttributes returns a pre-MAET whose pAttr
%   is a 1 x M cell of per-attribute cells sharing one geometry. That is
%   one such list, and is expanded the same way.
%
%   The loose triple has no such form, since the three parts are not
%   distinguishable from the surrounding positional geometry.
%
%   A 'verbose' name-value among the arguments governs these builds too,
%   so a quiet call stays quiet.
    verbose = localVerbose(args);
    for k = 1:numel(args)
        a = args{k};
        if internal.isPreMaet(a)
            if localIsSweep(a)
                args{k} = localBuildSweep(a, verbose);
            else
                args{k} = buildExpTens(a, 'verbose', verbose);
            end
        elseif iscell(a) && localAnyPreMaet(a)
            for j = 1:numel(a)
                if internal.isPreMaet(a{j})
                    a{j} = buildExpTens(a{j}, 'verbose', verbose);
                end
            end
            args{k} = a;
        end
    end
end


function tf = localIsSweep(pm)
%LOCALISSWEEP  True when pAttr holds one length-A cell per sweep index.
    p = pm.pAttr;
    tf = iscell(p) && ~isempty(p) && all(cellfun(@iscell, p));
end


function dens = localBuildSweep(pm, verbose)
%LOCALBUILDSWEEP  One density per sweep entry, on the shared geometry.
    M = numel(pm.pAttr);
    dens = cell(1, M);
    for m = 1:M
        dens{m} = buildExpTens(preMaet(pm.pAttr{m}, pm.wAttr, pm.specs), ...
                               'verbose', verbose);
    end
end


function tf = localVerbose(args)
%LOCALVERBOSE  The call's 'verbose' setting, true where unstated.
    tf = true;
    for k = 1:numel(args) - 1
        if (ischar(args{k}) || isstring(args{k})) ...
                && strcmpi(args{k}, 'verbose')
            v = args{k + 1};
            if islogical(v) || isnumeric(v)
                tf = logical(v);
            end
        end
    end
end


function tf = localAnyPreMaet(c)
    tf = false;
    for j = 1:numel(c)
        if internal.isPreMaet(c{j})
            tf = true;
            return;
        end
    end
end

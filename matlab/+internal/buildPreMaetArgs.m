function args = buildPreMaetArgs(args)
%BUILDPREMAETARGS  Build any whole pre-MAET among positional arguments.
%   A pre-MAET is a density in waiting: it holds everything buildMaet
%   needs, so it stands wherever a density does and is built here. That
%   goes for a CELL of them too: a cell of pre-MAETs stands wherever a
%   cell of densities does, so the list and scalar-vs-list forms take
%   them without the caller building each one first.
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
            args{k} = buildMaet(a, 'verbose', verbose);
        elseif iscell(a) && localAnyPreMaet(a)
            for j = 1:numel(a)
                if internal.isPreMaet(a{j})
                    a{j} = buildMaet(a{j}, 'verbose', verbose);
                end
            end
            args{k} = a;
        end
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

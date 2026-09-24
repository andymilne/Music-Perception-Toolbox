function v = axisLocate(locate, axis)
%AXISLOCATE  The locating rule for one swept axis.
%
%   v = internal.axisLocate(locate, axis) reads a per-axis map
%   {axis, value; ...} --- the twin of the Python dict --- and returns
%   that axis's rule, or 'centroid' where the map does not name it.
%   Anything else (a char rule or a function handle) applies to every
%   axis and is returned unchanged.
    if iscell(locate) && ismatrix(locate) && size(locate, 2) == 2 && ...
            ~isempty(locate) && ...
            all(cellfun(@(x) isnumeric(x) && isscalar(x), locate(:, 1)))
        [keys, vals] = internal.parseMap(locate);
        i = find(keys == axis, 1);
        if isempty(i), v = 'centroid'; else, v = vals{i}; end
    else
        v = locate;
    end
end

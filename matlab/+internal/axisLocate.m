function v = axisLocate(locate, axis)
%AXISLOCATE  The locating rule for one window attribute.
%
%   v = internal.axisLocate(locate, a) reads a per-attribute map
%   {a, rule; ...} --- the twin of the Python dict --- and returns
%   attribute a's rule, or 'centroid' where the map does not name it.
%   Anything else (a char rule or a function handle) applies to every
%   window attribute and is returned unchanged.
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

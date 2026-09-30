function [keys, vals] = parseMap(m)
%PARSEMAP  Parse an N-by-2 {a, value; ...} cell (a an attribute index) into
%   key vector + value cell.
    if isempty(m), keys = []; vals = {}; return; end
    if ~iscell(m) || size(m, 2) ~= 2
        error('mptWindowing:badMap', ...
            'a per-attribute map must be an N-by-2 cell {a, value; ...}, a an attribute index.');
    end
    keys = zeros(1, size(m, 1));
    for i = 1:size(m, 1), keys(i) = m{i, 1}; end
    vals = m(:, 2).';
end

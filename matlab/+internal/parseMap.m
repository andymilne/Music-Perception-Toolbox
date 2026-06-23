function [keys, vals] = parseMap(m)
%PARSEMAP  Parse an N-by-2 {axis, value; ...} cell into key vector + value cell.
    if isempty(m), keys = []; vals = {}; return; end
    if ~iscell(m) || size(m, 2) ~= 2
        error('mptWindowing:badMap', ...
            'sweep/drop/contextWindow map must be an N-by-2 cell {axis, value; ...}.');
    end
    keys = zeros(1, size(m, 1));
    for i = 1:size(m, 1), keys(i) = m{i, 1}; end
    vals = m(:, 2).';
end

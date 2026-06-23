function loc = locateRow(M, locate)
%LOCATEROW  Reduce a K-by-N attribute to the 1-by-N value its window centres
%   on: the multiset centroid by default, or 'start'/'end'/'mid'/a handle.
    M = double(M);
    if isa(locate, 'function_handle')
        loc = reshape(double(locate(M)), 1, []); return;
    end
    switch lower(char(locate))
        case 'centroid', loc = mean(M, 1, 'omitnan');
        case 'start',    loc = M(1, :);
        case 'end',      loc = M(end, :);
        case 'mid',      loc = 0.5 * (M(1, :) + M(end, :));
        otherwise
            error('mptWindowing:badLocate', ...
                ['locate must be ''centroid'', ''start'', ''end'', ''mid'', ' ...
                 'or a function handle; got ''%s''.'], char(locate));
    end
end

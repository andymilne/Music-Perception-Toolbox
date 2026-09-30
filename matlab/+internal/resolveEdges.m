function closed = resolveEdges(edges, prof, prefix)
%RESOLVEEDGES  Whether a rectangle includes its upper edge: EDGES is empty
%   or 'halfOpen' (lower edge only, the default) or 'closed' (both), and
%   only a rectangle has edges to close. Twin of the Python _resolve_edges.
    if nargin < 3 || isempty(prefix), prefix = 'weightEvents'; end
    closed = false;
    if isempty(edges), return; end
    if ~(ischar(edges) || isstring(edges)) || ...
            ~any(strcmp(char(edges), {'halfOpen', 'closed'}))
        error([prefix ':badEdges'], ...
              'edges must be ''halfOpen'' or ''closed''.');
    end
    closed = strcmp(char(edges), 'closed');
    if closed && ~(strcmp(prof.kind, 'family') && prof.shape == 1)
        error([prefix ':edgesNotRect'], ...
              ['edges ''closed'' applies to a rectangle (shape 1) alone; ' ...
               'the other profiles have no edges to close.']);
    end
end

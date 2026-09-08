function d = preMaetDepth(node)
%PREMAETDEPTH  Bracket depth of a parsed cell (0 for a flat multiset).
d = 0;
while ~isempty(node) && iscell(node) && iscell(node{1})
    d = d + 1;
    node = node{1};
end
end

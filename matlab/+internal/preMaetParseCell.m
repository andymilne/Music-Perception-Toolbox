function [node, exchLevels] = preMaetParseCell(text)
%PREMAETPARSECELL  Parse one pre-MAET cell into a bracket tree.
%
%   [node, exchLevels] = internal.preMaetParseCell(TEXT) reads a cell in
%   the notation of Milne (2026) and of showPreMaet: braces for an
%   unordered multiset, parentheses for an ordered one, nested brackets
%   for a nested attribute, and 60^(0.6) for a weighted value.
%
%   node is a nested cell of 1 x 2 [value weight] leaves (weight NaN where
%   none was written), outermost grouping first; exchLevels runs innermost
%   first, as a spec's exch field does.
%
%   See also READPREMAET, INTERNAL.PREMAETSTACK.
text = strtrim(text);
if isempty(text)
    node = {}; exchLevels = [];
    return;
end
if text(1) ~= '{' && text(1) ~= '('
    node = {internal.preMaetParseLeaf(text)}; exchLevels = [];
    return;
end
if text(1) == '{'
    exch = 1; closer = '}';
else
    exch = 0; closer = ')';
end
if text(end) ~= closer
    error('readPreMaet:brackets', 'Unbalanced brackets in cell ''%s''.', text);
end
inner = strtrim(text(2:end-1));
if isempty(inner)
    node = {}; exchLevels = exch;
    return;
end
parts = internal.preMaetSplitTop(inner);
nested = false;
for k = 1:numel(parts)
    p = strtrim(parts{k});
    if ~isempty(p) && (p(1) == '{' || p(1) == '(')
        nested = true; break;
    end
end
if nested
    node = cell(1, numel(parts)); below = [];
    for k = 1:numel(parts)
        [node{k}, lv] = internal.preMaetParseCell(parts{k});
        if isempty(below); below = lv; end
    end
    exchLevels = [below, exch];
else
    node = cell(1, numel(parts));
    for k = 1:numel(parts)
        node{k} = internal.preMaetParseLeaf(parts{k});
    end
    exchLevels = exch;
end
end

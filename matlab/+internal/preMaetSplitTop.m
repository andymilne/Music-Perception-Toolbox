function out = preMaetSplitTop(text)
%PREMAETSPLITTOP  Split on commas that are not inside a bracket.
out = {}; depth = 0; cur = '';
for i = 1:numel(text)
    ch = text(i);
    if ch == '{' || ch == '('
        depth = depth + 1;
    elseif ch == '}' || ch == ')'
        depth = depth - 1;
    end
    if ch == ',' && depth == 0
        out{end+1} = strtrim(cur); %#ok<AGROW>
        cur = '';
    else
        cur(end+1) = ch; %#ok<AGROW>
    end
end
out{end+1} = strtrim(cur);
end

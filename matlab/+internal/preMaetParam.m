function v = preMaetParam(text)
%PREMAETPARAM  A parameter cell: [] is absent, NA is NaN, else a value or
%   a parenthesised per-level tuple.
text = strtrim(text);
if isempty(text)
    v = [];
    return;
end
if strcmpi(text, 'NA')
    v = NaN;
    return;
end
if text(1) == '(' && text(end) == ')'
    parts = internal.preMaetSplitTop(text(2:end-1));
    v = zeros(1, numel(parts));
    for k = 1:numel(parts)
        if strcmpi(strtrim(parts{k}), 'NA')
            v(k) = NaN;
        else
            v(k) = str2double(parts{k});
        end
    end
    return;
end
v = str2double(text);
end

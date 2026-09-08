function leaf = preMaetParseLeaf(text)
%PREMAETPARSELEAF  '60' or '60^(0.6)' -> [value weight], weight NaN if absent.
text = strtrim(text);
k = strfind(text, '^');
if isempty(k)
    leaf = [str2double(text), NaN];
    return;
end
val = str2double(text(1:k(1)-1));
wt = strtrim(text(k(1)+1:end));
if numel(wt) > 1 && wt(1) == '(' && wt(end) == ')'
    wt = wt(2:end-1);
end
leaf = [val, str2double(wt)];
end

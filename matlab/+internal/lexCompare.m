function cmp = lexCompare(a, b)
%LEXCOMPARE Lexicographic comparison of two row vectors.
%   cmp = internal.lexCompare(a, b)
%
%   Returns -1 if a < b lexicographically, 0 if a == b, +1 if a > b.
%   Compares element-by-element at the first differing index.

    idx = find(a ~= b, 1);
    if isempty(idx)
        cmp = 0;
    elseif a(idx) < b(idx)
        cmp = -1;
    else
        cmp = 1;
    end
end

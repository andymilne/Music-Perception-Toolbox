function e = queryExtent(pQuery, axis)
%QUERYEXTENT  Span of the query's finite values on an attribute (default width).
    v = double(pQuery{axis}(:)); v = v(isfinite(v));
    if isempty(v), e = 0; else, e = max(v) - min(v); end
end

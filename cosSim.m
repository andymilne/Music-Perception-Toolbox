function s = cosSim(x,y)
%COSSIM Cosine similarity of two vectors
%   Given a vector x and vector y, both with the same number of entries,
%
%   s = cosSim(x,y)
%
%   returns their cosine similarity

persistent ipxx ipyy xLast yLast

if isnan(sum(x)) || isnan(sum(y))
    s = NaN;
else
    
% Make all inputs column vectors
xNew = ~isequal(x,xLast);
yNew = ~isequal(y,yLast);
if xNew
    x = x(:);
    ipxx = x'*x;
end
if yNew
    y = y(:);
    ipyy = y'*y;
end

% Calculate their cosine similarity
s = (x'*y)/sqrt(ipxx*ipyy);
end

xLast = x;
yLast = y;
end
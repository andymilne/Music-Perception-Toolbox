function d = pDist(x,y,p)
%PDIST p-norm distance between the two vectors
%   Given a vector x and a vector y, both with the same number of entries,
%
%   d = pDist(x,y,p)
%
%   returns their p-norm distance. When there are only two arguments, p is
%   taken to be 2, which gives the Euclidean distance between the two vectors;
%   when p = 1, this function gives the taxicab distance; when p = inf, it
%   gives the maximum difference.

if isnan(sum(x)) || isnan(sum(y))
    d = NaN;
else
    
if nargin == 3
    if p<1 && p>0
        warning('p-values between 0 and 1 do not provide a true metric')
    end
    if p<0
        error('p must be nonnegative')
    end
end
    
if nargin < 3
    p = 2;
end
    
% Make all inputs column vectors
x = x(:);
y = y(:);

diff = x - y;

% Calculate their p-norm distance
d = sum(abs(diff.^p))^(1/p);
end

end
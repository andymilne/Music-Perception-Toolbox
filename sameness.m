function [sq, nDiff] = sameness(x_ind)
% Sameness quotient (Carey 2002, 2007) of indicator vector x_ind
% This is one minus the ratio of the number of different specific sizes taken
% by all generic intervals and the maximum possible such sizes for a pattern
% with K events.

if ~isvector(x_ind)
    error('The first argument must be a vector.') 
elseif ~isempty(x_ind(x_ind~=0 & x_ind~=1))
    error('The indicator vector must be binary.')
end

N = numel(x_ind);
K = sum(x_ind);
eventIndex = find(x_ind);
edges = (0:N) - 0.5;

% Calculate all steps sizes
allRotations = eventIndex(mod((0:K-1)' + (0:K-1), K) + 1); % adapted from https://au.mathworks.com/matlabcentral/fileexchange/22858-circulant-matrix/content/circulant.m

sizeSpan = zeros(K-1,K);
for i=1:K-1
    sizeSpan(i,:) = mod(allRotations(i+1,:) - allRotations(1,:),N);
end
% Specific sizes of intervals in the scale organized by generic size (row 1 is 
% seconds, row 2 is thirds, ...) and reference pitch (the nth column has the 
% nth scale pitch as the reference).

size_counts = zeros(K-1,N);
for i=1:K-1
    size_counts(i,:) = histcounts(sizeSpan(i,:)',edges);
end
% Numbers of intervals for each possible duple of generic size and specific
% size. Rows give generic intervals size -- the first row is seconds, the
% second row thirds, etc. Columns give speific size -- first column is zero
% units, second column is one unit, etc.

allProds = kron(size_counts,size_counts);
relevant_prods = allProds(1:K:end,:);
nDiff = sum(sum(relevant_prods,2) - sum(size_counts.^2,2))/2;
maxDiffs = K*(K-1)*(K-1)/2;

sq = 1 - (nDiff/maxDiffs);

end
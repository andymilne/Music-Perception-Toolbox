function h = histEntropy(v,isNorm)
%HISTENTROPY Normalized entropy of a histogram.
%
%   h = histEntropy(v): Entropy (normalized by default) of a histogram v, where the
%   histogram is normalized to convert it into a probability mass function so
%   its emtropy can be calculated. Normalized entropy is entropy divided by
%   maximum possible entropy; it is, therefore, unitless and in the interval
%   [0,1]; it allows for comparisons between differently sized sample spaces. 
%
%   isNorm = 0 returns the unnormalized entropy. By default, isNorm = 1.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University.

%   TO DO -- consider allowing any number of vector arguments such that the
%   entropy is calculated for their sum but divided by the sum of entropies of
%   each vector. This could serve as a normalization different to the one used
%   here, but possibly more relevant to the prupose at hand. 

% Input checks
if ~isvector(v) || min(v)<0
    error('The input must be a non-negative vector.')
end
if nargin < 2
    isNorm = 1;
end

% Normalize to make into probability mass function
v = v(:);
histNorm = v./sum(v);

% Size of sample space
N = numel(v);

% Calculate entropy
logHistNorm = -log(histNorm);
logHistNorm(isinf(logHistNorm)) = 0;
if isNorm == 1
    h = histNorm'*logHistNorm/log(N);
else
    h = histNorm'*logHistNorm;
end

end
function [cq,nInC] = coherence(v,isStrict)
%COHERENCE Coherence quotient and number of coherence failures.
%   cq = COHERENCE(v) returns the coherence quotient of the binary
%   indicator vector v. The coherence quotient (Carey 2002, 2007) is unity
%   minus the ratio of the number of coherence failures and the maximum
%   possible number of coherence failues for a pattern with k events. A
%   coherence failure occurs when a pair of events with a larger generic
%   span than another two notes/onsets does not have a greater specific
%   size (Balzano 1982).
%
%   cq = COHERENCE(v,isStrict): when isStrict == 1 or is omitted, a
%   coherence failure is deemed to occur according to the above criteria;
%   when isStrict == 0 a coherence failure is deemed to occur only when a
%   pair of events with a larger generic span than another two events has a
%   smaller specific size. These correspond to Rothernberg's distinction
%   between strict propriety and propriety (1978).
%   
%   [cq,nInC] = COHERENCE(...) also returns the number of coherence failures.

% Input checks
if ~isvector(v)
    error('The first argument must be a vector.') 
elseif ~isempty(v(v~=0 & v~=1))
    error('The indicator vector must be binary: comprising only ones and zeros.')
end

if nargin < 2
    isStrict = 1;
end

N = numel(v);
K = sum(v);
eventIndex = find(v);

% Calculate all steps sizes
allRotations = zeros(K,K);
for k = 0:K-1
    allRotations(k+1,:) = circshift(eventIndex,[0 -k]);
end

sizeSpan = zeros(K-1,K);
for k = 1:K-1
    sizeSpan(k,:) = mod(allRotations(k+1,:) - allRotations(1,:),N);
end
sizeSpan = sort(sizeSpan,2); % each row is a span of a particular size
% from 1 to K-1 (e.g., seconds, thirds, fourths, ..., K-1ths), each column
% is a specific size of all spans of that size, each entry is a specific
% size.

% The "tensor sum" returns a matrix containing all possible additions.
% Because I have entered one of the matrices in negative form, the indices
% take the following form as expressed by the indices of size_span:
% (1,1)-(1,1) (1,1)-(1,2), ..., (1,K)-(1,1) (1,K)-(1,2) (1,K)-(1,K) 
% (1,1)-(2,1) (1,1)-(2,2), ..., (1,K)-(2,1) (1,K)-(2,2) (1,K)-(2,K)
%      :           :                 :           :           :
% (K,1)-(K,1) (K,1)-(2,2), ..., (K,K)-(K,1) (K,K)-(K,2) (K,K)-(K,K)
allDiffs = tensorSum(sizeSpan,-sizeSpan);

% Now I need to pick out all relevant rows of the above matrix. These are
% rows where a larger generic span is compared with a smaller generic span.
listVec = 1:K-2;
listMat = repmat(listVec,[K-2 1]);
listMat = tril(listMat);
relDiffInd = tril(bsxfun(@plus,(K-1)*(1:K-2)',listMat));
relDiffInd = relDiffInd(:);
relDiffInd = sort(relDiffInd(relDiffInd>0));
relevantDiffs = allDiffs(relDiffInd,:);

if isStrict == 1
    incoherences = find(relevantDiffs<=0);
elseif isStrict == 0
    incoherences = find(relevantDiffs<0);
end
nInC = length(incoherences(:));
maxInC = K*(K-1)*(K-2)*(3*K-5)/24;

cq = 1 - (nInC/maxInC);
end


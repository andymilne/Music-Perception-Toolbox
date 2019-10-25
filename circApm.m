function [R,rPhase,rLag] = circApm(x_pc)
%CIRCAPM Circular autocorrelation phase matrix.
%   Given a weighted indicator vector x_pc, which represents periodic
%   notes/onsets by nonnegative weights and pitch/time by index, R =
%   CIRCAPM(x_pc), returns the circular autocorrelation phase matrix (APM).
%
%   This function adapts the method described by Eck (2006) for calculating
%   a non-circular APM. It also corrects the erroneous summation limits Eck
%   provides in Eq (5). The row number indexes lag; so row 1 has a lag of
%   1, row N has a lag of N (which correponds to a lag 0). The column
%   number minus one indexes the phase value (in pulse units); so column 1
%   has phase 0, column N has phase N - 1.
%
%   [R,rPhase] = circApm(w) also returns the column sum of the APM, which
%   can be used as a model of "metrical weight" (Parncutt 1994).
%
%   [R,rPhase,rLag] = circApm(w) also returns the row sum of the APM, which
%   is identical to the autocorrelation of W.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University

% Input checks
if min(x_pc) < 0
    error('The input must be a non-negative vector.')
end

% Preliminaries
N = length(x_pc);

% Identify the indices of the indicator vector that will be pairwise multiplied 
nIndMat1 = (1:N)' * (0:N-1);
nIndMat2 = (1:N)' * (1:N);

% Increment the indices by phase in the third dimension
PhaseInd = permute(bsxfun(@times,ones(N,N,N),(0:N-1)'),[2 3 1]);
% PhaseInd = permute(ones(N,N,N).*(0:N-1)',[2 3 1]);
nIndTens1 = bsxfun(@plus,PhaseInd,nIndMat1);
% nIndTens1 = PhaseInd + nIndMat1;
nIndTens2 = bsxfun(@plus,PhaseInd,nIndMat2);
% nIndTens2 = PhaseInd + nIndMat2;

% Create an order-3 tensor of ones (valid entries) and zeros (invalid
% entries)
invalidInd1 = zeros(N,N,N);
invalidInd1(nIndTens1 < N) = 1;
invalidInd2 = permute(repmat(tril(ones(N,N)),[1 1 N]),[1 3 2]);
invalidInd = invalidInd1.*invalidInd2;

% NB the bounds used here are correct because the row sum of the
% resulting APM is equivalent to the circular autocorrelation (as
% tested below). However, these bounds are not the same as those in the
% equations used in Eck's papers. The correct bound has to involve phi
% -- instead of the summation occuring from i=0 to i=floor(N/k-1), it
% should be over i=0 to i=ceil((N-phi)/k) - 1, where k is the lag (from
% 1 to N), phi is index of the phase dimension in the APM, N is the
% number of pulses (length of the vector being analysed).

% Now get the values from the indicator vector using the previously
% calculated indices
indicatorTens1 = x_pc(mod(nIndTens1,N)+1);
indicatorTens2 = x_pc(mod(nIndTens2,N)+1);

% Multiply and sum to get APM matrix
R = squeeze(sum(indicatorTens1.*indicatorTens2.*invalidInd,2));
if nargout > 1
    rPhase = sum(R,1);
end
if nargout > 2
    rLag = sum(R,2);
end

% Check the APM row sum equals conventional circular autocorrelation
% AC_test = round(cconv(tap_data_aggLoops.Indicator{rhythm}, ...
%     fliplr(tap_data_aggLoops.Indicator{rhythm}), ...
%     N))';
end


function [R,rPhase,rLag] = circApm(x_ind)
%CIRCAPM Circular autocorrelation phase matrix.
%   Given a weighted indicator vector x_ind, which represents periodic
%   notes/onsets by nonnegative weights and pitch/time by index, R =
%   CIRCAPM(x_pc), returns the circular autocorrelation phase matrix (APM).
%
%   This function adapts the method described by Eck (2006) for calculating a
%   non-circular APM. The row number minus one indexes lag; so row 1 has a lag
%   of 0, row N has a lag of N-1. The column number minus one indexes the phase
%   value (in pulse units); so column 1 has phase 0, column N has phase N-1.
%
%   [R,rPhase] = circApm(w) also returns the column sum of the APM, which
%   can be used as a model of "metrical weight" (Parncutt 1994).
%
%   [R,rPhase,rLag] = circApm(w) also returns the row sum of the APM, which
%   is identical to the autocorrelation of x_ind.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University


% Input checks
if min(x_ind) < 0
    error('The input must be a non-negative vector.')
end

% Preliminaries
N = length(x_ind);

% Identify the indices of the indicator vector that will be in each dot product
nIndArray = (1:N)' * (0:N-1) + reshape((0:N-1), [1 1 N]) ;
R = nan(N, N);
for phi = 0 : N-1
    for k = 0 : N-1
        R(k+1, phi+1) ...
            = x_ind(mod(nIndArray(mod(k-1, N)+1, :, mod(phi, N)+1), N) + 1) ...
            * x_ind(mod(nIndArray(mod(k-1, N)+1, :, mod(k+phi, N)+1), N) + 1)';
    end
end
R = R/N;

if nargout > 1
    rPhase = sum(R,1);
end
if nargout > 2
    rLag = sum(R,2);
end

end

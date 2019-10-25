function h = modeHeight(x_ind)
%MODEHEIGHT Mean angle of pitches/times with respect to first (or all).
%
%   Given a weighted indicator vector x_ind, which represents periodic
%   pitches/beats by nonnegative weights and pitch/time by index,
%
%   h = MODEHEIGHT(x_ind),
%
%   returns, for each index, the difference between the sum of interval sizes 
%   upwards minus the sum of interval sizes downwards. Sizes are normalized to
%   the size of the period
% 
%   TO DO - rewrite for pitch/time class sets

% Input checks
if min(x_ind) < 0 || ~isvector(x_ind)
    error('The input must be a non-negative vector.')
end

% Preliminaries
N = numel(x_ind);
eventIndex = find(x_ind);

% heightArrayUp(n,k) gives the pitch/time of the kth note/event (non-zero
% value) in the indicator vector upwards relative to the nth chroma/pulse in 
% the indicator vector. heightArrayUp(n,k) does the same, but downwards.
heightArrayUp = mod(eventIndex - (0:N-1)' - 1, N);
heightArrayDown = mod(-(eventIndex - (0:N-1)' - 1), N);

h = (sum(heightArrayUp,2) - sum(heightArrayDown,2))/N;

end


function x_a = pitch2Argand(x_p, period)
%ARGAND Argand vector representation of pitch/time vector.
%   Given the pitch/time vector x_p and its period
%
%   x_a = argand(x, period)
%
%   returns the pitch/time vector's Argand representation.
%

if nargin < 2
    period = 1;
end

% Preliminaries: make pitch/times modulo the period and normalize
x_p = x_p(:);
x_p = mod(x_p, period); % mod period
x_p = x_p/period; % normalize
x_p = sort(x_p);

% Calculations
x_a = exp(2*pi*1i*x_p);

end

function b = bal(x_p, period)
%BAL Balance of a pitch or time class set.
%   Given the pitch/time vector x_p and its period
%
%   b = bal(x, period)
%
%   returns the pitch/time class set's balance.
%
%   Balance is equivalent to unity minus the circular variance of the
%   pattern.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University.


K = numel(x_p);
x_a = pitch2Argand(x_p, period);
dftScale = fft(x_a)/K;

b = 1 - abs(dftScale(0+1)); 

end

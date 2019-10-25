function e = eve(x_p, period)
%BAL Evenness of a pitch or time class set.
%   Given the pitch/time vector x_p and its period
%
%   e = eve(x, period)
%
%   returns the pitch/time class set's evenness.
%
%   Evenness is the circular variance of the pitch/time displacements
%   between the kth note/onset and the kth equal division of the period,
%   for all K notes/onsets.

x_a = pitch2Argand(x_p, period);
dftScale = fft(x_a)/K;

e = abs(dftScale(1+1)); 

end

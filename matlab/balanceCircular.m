function b = balanceCircular(p, w, period)
%BALANCECIRCULAR Balance of a pitch-class set (or time-class set).
%
%   b = balanceCircular(p, w, period):
%
%   Computes the balance of a set of points on a circle, defined as:
%
%     b = 1 - |F(0)|
%
%   where F(k) is the k-th DFT coefficient of the set (see dftCircular).
%   |F(0)| is the magnitude of the weighted centre of gravity on the unit
%   circle (the mean of exp(2*pi*1i*p/period)).
%
%   Balance ranges from 0 to 1:
%     b = 1: perfectly balanced — the centre of gravity is at the centre
%            of the circle. Examples: the whole-tone scale, the augmented
%            triad, or any equal-step scale in pitch; isochronous rhythms.
%     b = 0: maximally unbalanced — all weight concentrated at one point.
%
%   Perfect balance is a necessary condition for maximal evenness but is
%   not sufficient: a set can be perfectly balanced without being
%   maximally even (see evennessCircular).
%
%   For further information, see:
%     Milne, A. J., Bulger, D., & Herff, S. A. (2017). Exploring the
%       space of perfectly balanced rhythms and scales. Journal of
%       Mathematics and Music, 11(2-3), 101-133.
%     Milne, A. J. & Herff, S. A. (2020). The perceptual relevance of
%       balance, evenness, and entropy in musical rhythms. Cognition,
%       203, 104233.
%
%   Inputs:
%     p      — Pitch-class (or time-class) values (vector of length K).
%     w      — Weights (vector of length K, or empty for uniform).
%     period — Period of the circular domain.
%
%   Output:
%     b      — Balance (scalar, range [0, 1]).
%
%   Examples:
%     % Perfectly balanced: augmented triad
%     b = balanceCircular([0, 400, 800], [], 1200);   % b = 1.000
%
%     % Nearly balanced: diatonic scale
%     b = balanceCircular([0, 200, 400, 500, 700, 900, 1100], [], 1200);
%
%     % Unbalanced: chromatic cluster
%     b = balanceCircular([0, 100, 200], [], 1200);
%
%   See also evennessCircular, dftCircular.

[~, mag] = dftCircular(p, w, period);
b = 1 - mag(1);  % mag(1) = |F(k=0)|

end

function h = rAdEntropy(x_ind,r,isNorm)
%RADENTROPY Relative r-ad entropy.
%   Given a binary indicator vector v, which represents periodic notes/events
%   by ones and pitch/time by index, h = rAdEntropy(v,r) returns the entropy,
%   in bits, of relative r-ads of pitches/times.
%
%   For example, when considering r = 2, the diatonic indicator vector [1 0 1 0
%   1 1 0 1 0 1 0 1] has two dyads of size 1 (minor seconds), five dyads of
%   size 2 (major seconds), four dyads of size 3 (minor thirds), three dyads of
%   size 4 (major thirds), and so on. This defines a probability mass function
%   over relative dyads (intervals) and, hence, an associated entropy.
%
%   When considering r = 3, the diatonic indicator vector has no triads with
%   intervals above the lowest note of (1,1), two triads with intervals of
%   (2,1), no triads with intervals of (3,1), two triads with intervals (4,1),
%   ..., two triads with intervals (1,2), five dyads with intervals (2,2), four
%   triads with intervals (3,2), and so on. This defines a probability mass
%   function over all possible relative triads and, hence, an associated
%   entropy.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University
%
%   See also STEPENTROPY, HISTENTROPY, SPHISTENTROPY

if nargin < 3
    isNorm = 1;
end

x_p = ind2Pitch(x_ind);
x_w = 1;
sigma = eps;
kerLen = 0;
isRel = 1;
isPer = 1;
limits = numel(x_ind);
if r < 3
    isSparse = 0;
else
    isSparse = 1;
end

X_r = expectationTensor(x_p, x_w, ...
                        sigma, kerLen, r, isRel, isPer, limits, ...
                        isSparse)
if r < 3
    X_r(abs(X_r) < 0.0001) = 0;
    h = histEntropy(X_r(:), isNorm);
else
    h = spHistEntropy(X_r(:), isNorm);
end

end
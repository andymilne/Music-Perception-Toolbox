function h = expTensorEntropy(x_p, x_w, sigma, kerLen, ...
                              r, isRel, isPer, limits)
%RADENTROPY Relative r-ad entropy.
%   Given a pitch time class vector x_pc, with weights x_w, and other 
%   parameters for generating an expectation array, 
%   h = expTensorEntropy(x_p, x_w, sigma, kerLen, ...
%                              r, isRel, isPer, limits, ...
%                              isSparse, doPlot, tol)
%   returns the normalized entropy of the resulting expectation array.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University
%
%   See also STEPENTROPY, HISTENTROPY, SPHISTENTROPY, RADENTROPY

X_r = expectationTensor(x_p, x_w, ...
    sigma, kerLen, r, isRel, isPer, limits, ...
    isSparse);
if r < 3
    h = histEntropy(X_r(:));
else
    h = spHistEntropy(X_r(:));
end

end
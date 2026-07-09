function tf = isKernelCov(sigmaEntry)
%ISKERNELCOV  True if sigmaEntry is a matrix-valued kernel covariance.
%
%   A kernel covariance is a numeric 2-D array with both dimensions
%   greater than 1. Scalars (including 1x1 matrices) and vectors are
%   the ordinary isotropic sigma.

    tf = isnumeric(sigmaEntry) && ismatrix(sigmaEntry) ...
        && size(sigmaEntry, 1) > 1 && size(sigmaEntry, 2) > 1;
end

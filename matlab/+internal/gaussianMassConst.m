function mass = gaussianMassConst(sigma, dim, detM, half)
%INTERNAL.GAUSSIANMASSCONST  Gaussian normalisation constant (c*pi*sigma^2)^(dim/2) / sqrt(detM).
%
%   With HALF = false (default) c = 2: the single-kernel *mass*, the
%   integral of one un-normalised Gaussian exp(-Q_M(d) / (2 sigma^2))
%   over the DIM-dimensional attribute space. With HALF = true c = 1:
%   the *overlap* form (pi sigma^2)^(dim/2) / sqrt(detM), the mass of
%   the product of two such kernels (used by the entropy read-outs).
%   DETM comes from internal.quadraticFormDet.
%
%   Density values are divided by the mass to normalise to unit peak
%   or to a pdf; the eval and harmony paths equivalently multiply by
%   its reciprocal, which is (2 pi sigma^2)^(-dim/2) * sqrt(detM). The
%   consumers therefore call this helper and divide, keeping every
%   normalisation site pointing at the same algebraic form.
%
%   Twin of Python mpt._tensor.dispatch._gaussian_mass_const.

    if nargin < 4 || isempty(half)
        half = false;
    end
    if logical(half)
        c = 1.0;
    else
        c = 2.0;
    end
    mass = (c * pi * double(sigma)^2) ^ (double(dim) / 2.0) / sqrt(double(detM));
end

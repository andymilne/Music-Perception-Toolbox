function sop = absPerSigmaOverPThreshold()
%ABSPERSIGMAOVERPTHRESHOLD  sigma/P above which the absolute-periodic
%   single-image measure departs from the full-image measure.
%
%   The absolute periodic kernel is the periodization of the Gaussian:
%   the sum over every image of the difference. The single-image
%   (nearest-image) kernel that ``wrap = 'single-image'`` opts into
%   approximates it, and the two agree only while sigma/P is small.
%
%   This constant gates a diagnostic, not a computation. Nothing routes
%   on it: the absolute-periodic image count is derived from the
%   caller's truncationSigmas, so the full-image kernel needs no
%   threshold in sigma/P and keeping the nearest image alone is simply
%   the case where one term already meets the floor. This value decides
%   only when a user who has forced wrap = 'single-image' is told that
%   the choice has begun to depart from the default measure.
%
%   Fixed rather than resolved from truncationSigmas, because what
%   settles it is positive-definiteness rather than accuracy. The
%   single-image kernel is a product of one-dimensional nearest-image
%   kernels, one per coordinate, and a product of positive-definite
%   kernels is positive-definite, so its one-dimensional Fourier
%   coefficients decide the question for every tuple order at once. By
%   Bochner's theorem the kernel is positive-definite exactly when every
%   coefficient is non-negative: measured, they sit at floating-point
%   noise (about 2e-16) through sigma/P = 0.044 and lift off from 0.046,
%   reaching -4.9e-5 by 0.10. Past that the induced cosine similarity is
%   not bounded by 1 and the quantity is not a similarity at all, which
%   no accuracy setting has authority to permit.
%
%   The threshold sits below the onset with margin. It is exposed as a
%   function so that the warning and the tests that exercise its
%   boundary read one source; hardcoding it in both let the two drift.
%
%   Twin of Python _ABS_PER_SIGMA_OVER_P_THRESHOLD.
%
%   See also INTERNAL.MAYBEWARNABSPERSINGLEIMAGE,
%   INTERNAL.RELPERSIGMAOVERPTHRESHOLD.

    sop = 0.04;
end

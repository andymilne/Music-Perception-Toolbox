function H = entropyExpTens(p, w, sigma, r, isRel, isPer, period, nvArgs)
% ENTROPYEXPTENS Shannon entropy (in bits) of an expectation tensor.
%
%   H = ENTROPYEXPTENS(p, w, sigma, r, isRel, isPer, period) returns the
%   Shannon entropy of the expectation tensor defined by the weighted
%   multiset (p, w), where p represents pitches or positions. The tensor
%   is built, evaluated at query points,
%   normalized to a probability distribution, and the Shannon entropy
%   returned. By default the entropy is normalized to [0, 1] by dividing
%   by log_base(N), where N is the number of grid points.
%
%   H = ENTROPYEXPTENS(T) returns the Shannon entropy of a pre-built
%   expectation tensor struct T (as returned by buildExpTens). When a
%   struct is passed, the remaining positional arguments are not
%   required.
%
%   H = ENTROPYEXPTENS(..., Name, Value) specifies additional options
%   using one or more name-value arguments.
%
%   The differential entropy of a Gaussian mixture has no closed-form
%   analytic solution, because the logarithm of a sum of Gaussians does
%   not simplify. This function therefore discretizes the expectation
%   tensor over a fine grid and computes the Shannon entropy of the
%   resulting probability mass function. The accuracy of this
%   approximation depends on the ratio of sigma to the grid spacing;
%   accuracy can be verified by comparing results at different
%   resolutions.
%
%   For periodic tensors (isPer = true), the domain is [0, period) and
%   query points are automatically determined. For non-periodic tensors
%   (isPer = false), the user must specify the domain bounds via xMin
%   and xMax. These should be wide enough to capture the full support
%   of the distribution (e.g., at least 3*sigma beyond the outermost
%   values).
%
%   The convention 0 * log(0) = 0 is applied.
%
%   Inputs
%       p       - Pitch or position values (vector, or matrix for
%                 multi-dimensional tensors where each column is a
%                 dimension).
%                 Alternatively, a struct as returned by buildExpTens,
%                 in which case w, sigma, r, isRel, isPer, and period
%                 are not required.
%       w       - Weights (vector, same length as p / number of rows).
%       sigma   - Gaussian bandwidth in cents.
%       r       - Tuple size (positive integer; r >= 2 if isRel == true).
%       isRel   - Logical: true for relative (transposition-invariant).
%       isPer   - Logical: true for periodic domain.
%       period  - Period of the domain (e.g., 12 for octave).
%
%   Name-Value Arguments
%       'spectrum'      - Cell array of arguments to pass to addSpectra
%                         (everything after p and w). If provided,
%                         partials are added to the multiset before
%                         building the tensor. Example:
%                           'spectrum', {'harmonic', 12, 'powerlaw', 1}
%                           'spectrum', {'stretched', 8, 1.02, 'powerlaw', 1}
%                         If omitted, pitches are used as-is.
%       'normalize'     - Logical (default: true). If true, divides the
%                         entropy by log_base(N), where N is the total
%                         number of grid points. This gives a value in
%                         [0, 1] that is independent of the (arbitrary)
%                         grid resolution: 0 = all mass at one point,
%                         1 = uniform distribution.
%       'base'          - Logarithm base (default: 2).
%                         Common choices: 2 (bits), exp(1) (nats),
%                         10 (hartleys). When 'normalize' is true, the
%                         base cancels and has no effect on the result.
%       'nPointsPerDim' - Number of query points per dimension
%                         (default: 1200). Total points evaluated is
%                         nPointsPerDim^D where D is the number of
%                         dimensions.
%       'xMin'          - Lower bound of the domain (required when
%                         isPer = false).
%       'xMax'          - Upper bound of the domain (required when
%                         isPer = false).
%
%   Output
%       H       - Shannon entropy. When 'normalize' is true (default),
%                 H is in [0, 1]. When false, H is in the units
%                 determined by 'base'.
%
%   Example
%       % Normalized entropy from pitch classes directly
%       H = entropyExpTens([0 4 7], ones(1,3), 10, 1, false, true, 12)
%
%       % Unnormalized entropy in bits
%       H = entropyExpTens([0 4 7], ones(1,3), 10, 1, false, true, 12, ...
%                          'normalize', false)
%
%       % Entropy from a pre-built tensor
%       T = buildExpTens([0 4 7], ones(1,3), 10, 1, false, true, 12);
%       H = entropyExpTens(T)
%
%       % Entropy with spectral enrichment
%       H = entropyExpTens([0 4 7], ones(1,3), 10, 1, false, true, 12, ...
%                          'spectrum', {'harmonic', 24, 'powerlaw', 1})
%
%       % Entropy of a non-periodic tensor with explicit bounds
%       H = entropyExpTens([0 4 7], ones(1,3), 10, 1, false, false, 12, ...
%                          'xMin', -1, 'xMax', 8)
%
%       % Entropy of a 1-D interval tensor (lower resolution for speed)
%       H = entropyExpTens([0 4; 4 7], ones(2,1), 10, 2, true, true, 12, ...
%                          'nPointsPerDim', 600)
%
%   See also BUILDEXPTENS, EVALEXPTENS, COSSIMEXPTENS.

    arguments
        p
        w = []
        sigma = []
        r = []
        isRel = []
        isPer = []
        period = []
        nvArgs.spectrum = {}
        nvArgs.normalize (1,1) logical = true
        nvArgs.base (1,1) {mustBePositive} = 2
        nvArgs.nPointsPerDim (1,1) {mustBePositive, mustBeInteger} = 1200
        nvArgs.xMin (1,1) {mustBeNumeric} = NaN
        nvArgs.xMax (1,1) {mustBeNumeric} = NaN
    end

    % Determine whether a pre-built tensor struct was passed.
    if isstruct(p)
        T = p;
        isPer = T.isPer;
        period = T.period;
    else
        % Validate that all positional arguments were provided.
        if isempty(w) || isempty(sigma) || isempty(r) || ...
           isempty(isRel) || isempty(isPer) || isempty(period)
            error('entropyExpTens:missingArgs', ...
                  ['When p is not a pre-built tensor struct, all ' ...
                   'positional arguments (p, w, sigma, r, isRel, ' ...
                   'isPer, period) are required.']);
        end

        % Apply spectral enrichment if requested.
        if ~isempty(nvArgs.spectrum)
            if ~iscell(nvArgs.spectrum)
                error('entropyExpTens:badSpectrum', ...
                      '''spectrum'' value must be a cell array of addSpectra arguments.');
            end
            [p, w] = addSpectra(p, w, nvArgs.spectrum{:});
        end

        % Build expectation tensor.
        T = buildExpTens(p, w, sigma, r, isRel, isPer, period, ...
                         'verbose', false);
    end

    % Validate bounds for non-periodic case.
    if ~isPer
        if isnan(nvArgs.xMin) || isnan(nvArgs.xMax)
            error('entropyExpTens:missingBounds', ...
                  'xMin and xMax must be specified when isPer = false.');
        end
        if nvArgs.xMin >= nvArgs.xMax
            error('entropyExpTens:invalidBounds', ...
                  'xMin must be less than xMax.');
        end
    end

    % Construct per-dimension query points.
    if isPer
        % Periodic: evenly spaced on [0, period), exclude duplicate endpoint.
        x = linspace(0, period, nvArgs.nPointsPerDim + 1);
        x = x(1:end-1);
    else
        % Non-periodic: evenly spaced on [xMin, xMax].
        x = linspace(nvArgs.xMin, nvArgs.xMax, nvArgs.nPointsPerDim);
    end

    % Evaluate tensor. For D dimensions, evalExpTens handles the meshgrid
    % internally; the result is nPointsPerDim^D values.
    t = evalExpTens(T, x);

    % Normalize to a probability distribution.
    q = t(:) / sum(t(:));

    % Total number of grid points (before removing zeros). This is the N
    % used for entropy normalization: H_max = log_base(N) for a uniform
    % distribution over N bins.
    N = numel(q);

    % Apply 0 * log(0) = 0 convention.
    q(q == 0) = [];

    % Shannon entropy.
    H = -sum(q .* (log(q) / log(nvArgs.base)));

    % Normalize to [0, 1] by dividing by the maximum possible entropy
    % (that of a uniform distribution over N bins). This removes the
    % dependence on the arbitrary grid resolution.
    if nvArgs.normalize
        H = H / (log(N) / log(nvArgs.base));
    end

end
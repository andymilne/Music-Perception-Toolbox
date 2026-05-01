function [magMean, magStd, mags] = dftCircularSimulate(p, w, period, sigma, nvArgs)
%DFTCIRCULARSIMULATE Monte Carlo Argand DFT under positional jitter.
%
%   [magMean, magStd] = dftCircularSimulate(p, w, period, sigma)
%   estimates the distribution of Argand-DFT coefficient magnitudes
%   under independent Gaussian positional jitter on each event:
%
%       P_k = (p_k + eta_k) mod period,   eta_k ~ N(0, sigma^2)
%
%   For each draw, the perturbed positions are sorted (a perceptual
%   re-ordering at the level of the listener: events identified only
%   by their sorted position on the cycle, not by which underlying
%   event they came from), the resulting Argand vector is formed
%   z_j = w_j * exp(2*pi*1i*P_j/period), and the DFT is computed.
%
%   The function returns the per-coefficient mean and standard
%   deviation of the magnitudes |F(k)|, k = 0, 1, ..., K-1, taken
%   across draws.
%
%   [magMean, magStd, mags] = dftCircularSimulate(...) additionally
%   returns the full nDraws x K sample matrix, useful for
%   histograms, quantile analysis, or anything that needs more than
%   first and second moments.
%
%   For sigma -> 0 the mean magnitudes converge to the deterministic
%   Argand-DFT magnitudes from dftCircular, and the standard
%   deviations converge to zero. Resort effects are negligible while
%   sigma is small relative to the smallest event-to-event gap;
%   beyond that, resort is captured by sorting each draw.
%
%   Inputs
%       p       - Event positions (vector of length K).
%       w       - Weights (vector of length K, scalar, or empty for
%                 all ones).
%       period  - Period of the circular domain.
%       sigma   - Positional jitter standard deviation (positive
%                 scalar, in the same units as p and period).
%
%   Name-Value Arguments
%       'nDraws' - Number of MC draws (default 10000).
%       'rngSeed' - Optional seed for the random number generator
%                  for reproducibility. Default: not set (uses the
%                  current global state). Pass a non-negative
%                  integer to seed.
%
%   Outputs
%       magMean - 1 x K row vector of mean magnitudes E[|F(k)|].
%       magStd  - 1 x K row vector of magnitude standard deviations.
%       mags    - nDraws x K matrix of magnitude samples (only
%                 allocated when requested as a third output).
%
%   Example
%       % Son clave under sigma = 1/8 of a pulse
%       p = [0, 3, 6, 10, 12]; T = 16;
%       [m, s] = dftCircularSimulate(p, [], T, 1/8);
%       fprintf('|F(1)| = %.3f +/- %.3f\n', m(2), s(2));
%
%   See also DFTCIRCULAR, BALANCECIRCULAR, EVENNESSCIRCULAR,
%   PROJCENTROID.

    arguments
        p (:,1) {mustBeNumeric}
        w
        period (1,1) {mustBeNumeric, mustBePositive}
        sigma (1,1) {mustBeNumeric, mustBeNonnegative}
        nvArgs.nDraws (1,1) {mustBePositive, mustBeInteger} = 10000
        nvArgs.rngSeed = []
    end

    p = p(:);
    K = numel(p);

    if isempty(w)
        w = ones(K, 1);
        uniformWeights = true;
    elseif isscalar(w)
        w = w * ones(K, 1);
        uniformWeights = true;
    else
        w = w(:);
        if numel(w) ~= K
            error('dftCircularSimulate:weightLength', ...
                  'w must have the same number of entries as p (or be empty).');
        end
        uniformWeights = all(w == w(1));
    end

    if ~isempty(nvArgs.rngSeed)
        rng(nvArgs.rngSeed);
    end

    nDraws = nvArgs.nDraws;

    % --- Generate jittered positions, K x nDraws ---

    eta = sigma * randn(K, nDraws);
    P = mod(p + eta, period);

    % --- Sort each column (resort under jitter) ---

    if uniformWeights
        % Uniform weights: sortIdx is irrelevant, just sort P.
        Psorted = sort(P, 1);
        wScaledRoots = exp(2 * pi * 1i * Psorted / period);
        z = w(1) * wScaledRoots;        % w(1) is the common weight
        sumW = w(1) * K;
    else
        % Non-uniform weights: gather weights along sortIdx.
        [Psorted, sortIdx] = sort(P, 1);
        wMatrix = w(sortIdx);                       % K x nDraws
        z = wMatrix .* exp(2 * pi * 1i * Psorted / period);
        sumW = sum(w);
    end

    % --- FFT along columns ---

    F = fft(z, [], 1) / sumW;       % K x nDraws complex
    magsAll = abs(F);                % K x nDraws

    magMean = mean(magsAll, 2).';    % 1 x K
    magStd  = std(magsAll, 0, 2).';  % 1 x K (default normalisation)

    if nargout > 2
        mags = magsAll.';            % nDraws x K
    end
end

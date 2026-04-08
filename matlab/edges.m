function [e, eSigned] = edges(p, w, period, x, nvArgs)
%EDGES Edge detection on a circular pitch-class or time-class set.
%
%   e = edges(p, w, period) computes the "edginess" of each position
%   in 0, 1, ..., period-1 by evaluating the circular convolution of
%   the weighted event set with the first derivative of a von Mises
%   kernel, and taking absolute values. The von Mises kernel is the
%   circular analogue of a Gaussian; its derivative detects abrupt
%   changes in event density around the circle — the circular analogue
%   of edge detection in image processing.
%
%   Positions near a sharp transition between event-dense and
%   event-sparse regions receive high edge weights; positions in
%   uniformly dense or uniformly sparse regions receive low weights.
%
%   e = edges(p, w, period, x) evaluates the edge weights at the
%   query points specified in the vector x (in the same units as p
%   and period) instead of at integer positions 0:period-1.
%
%   [e, eSigned] = edges(...) also returns the signed edge weights
%   (before taking absolute values). Positive values indicate a
%   rising edge (event density increasing in the clockwise
%   direction); negative values indicate a falling edge.
%
%   The edge weight at a query point x_n is computed as:
%
%     eSigned(x_n) = sum_k w_k * vM'(x_n; p_k, kappa)
%
%   where vM'(x; c, kappa) is the first derivative of the von Mises
%   distribution centred at c:
%
%     vM'(x; c, kappa) = -kappa * sin(2*pi*(x - c) / period)
%                        * exp(kappa * cos(2*pi*(x - c) / period))
%                        / (2*pi * I_0(kappa))
%
%   and I_0 is the modified Bessel function of the first kind.
%
%   Inputs:
%     p      — Pitch-class (or time-class) values (vector of length K).
%              Values are interpreted modulo 'period'.
%     w      — Weights (vector of length K, or empty for uniform).
%     period — Period of the circular domain (e.g., 12 for chromatic
%              pitch classes, 1200 for cents, or a rhythmic cycle
%              length).
%     x      — (Optional) Query points at which to evaluate the edge
%              weights (vector). Default: 0:period-1.
%
%   Name-Value Arguments:
%     'kappa' — Concentration parameter of the von Mises kernel
%               (default: 6.7). Larger values give a narrower kernel
%               that detects sharper edges; smaller values give a
%               broader kernel that responds to more gradual
%               transitions. kappa is analogous to 1/sigma^2 for a
%               Gaussian.
%
%   Outputs:
%     e       — Absolute edge weights (row vector, same length as x
%               or as 0:period-1). Non-negative.
%     eSigned — Signed edge weights (row vector). Positive = rising
%               edge (density increasing clockwise), negative =
%               falling edge.
%
%   Examples:
%     % Edge weights of a diatonic scale (12 chromatic positions)
%     e = edges([0, 2, 4, 5, 7, 9, 11], [], 12);
%     bar(0:11, e);
%     xlabel('Pitch class'); ylabel('Edge weight');
%
%     % Fine grid in cents (0.1-cent resolution)
%     x = 0:0.1:1199.9;
%     [e, eSigned] = edges([0, 200, 400, 500, 700, 900, 1100], ...
%                          [], 1200, x);
%     plot(x, eSigned);
%
%     % Sharper edge detection (larger kappa)
%     e = edges([0, 2, 4, 5, 7, 9, 11], [], 12, [], 'kappa', 20);
%
%     % Son clave rhythm (16-step cycle)
%     e = edges([0, 3, 6, 10, 12], [], 16);
%     bar(0:15, e);
%     xlabel('Pulse'); ylabel('Edge weight');
%
%   References:
%     Milne, A. J., Dean, R. T., & Bulger, D. (2023). The effects of
%       rhythmic structure on tapping accuracy. Attention, Perception,
%       & Psychophysics, 85, 2673-2699.
%       (Introduced this predictor — adapting standard edge-detection
%       techniques for images to circular rhythmic patterns.)
%
%   See also meanOffset, projCentroid, circApm.

    arguments
        p (:,1) {mustBeNumeric}
        w (:,1) {mustBeNumeric}
        period (1,1) {mustBePositive}
        x {mustBeNumeric} = []
        nvArgs.kappa (1,1) {mustBePositive} = 6.7
    end

    % === Input defaults ===

    K = numel(p);

    if isempty(w)
        w = ones(K, 1);
    end
    if isscalar(w)
        w = w * ones(K, 1);
    end

    if numel(w) ~= K
        error('w must have the same number of entries as p (or be empty).');
    end

    if isempty(x)
        x = 0:period-1;
    end
    x = x(:).';

    kappa = nvArgs.kappa;

    % === Evaluate the von Mises derivative at query points ===
    % For each query point x_n, sum over all events p_k:
    %
    %   eSigned(x_n) = sum_k w_k * vM'(x_n; p_k, kappa)
    %
    % where vM'(x; c, kappa) = -kappa * sin(theta) * exp(kappa * cos(theta))
    %                          / (2*pi * I_0(kappa))
    % and theta = 2*pi*(x - c) / period.
    %
    % Vectorized: theta is nQ x K, result is 1 x nQ.

    nQ = numel(x);
    theta = 2 * pi * (x(:) - p(:)') / period;  % nQ x K

    normConst = 2 * pi * besseli(0, kappa);
    kernel = -kappa * sin(theta) .* exp(kappa * cos(theta)) / normConst;

    % Weighted sum across events
    eSigned = (kernel * w(:)).';  % 1 x nQ

    % Absolute edge weights
    e = abs(eSigned);

end
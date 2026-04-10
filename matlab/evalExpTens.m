function vals = evalExpTens(varargin)
%EVALEXPTENS Evaluate an r-ad expectation tensor density at query points.
%
%   vals = evalExpTens(dens, X):
%   vals = evalExpTens(dens, X, normalize):
%   vals = evalExpTens(dens, X, ..., 'verbose', false):
%   Evaluates the density using a precomputed struct from buildExpTens.
%
%   vals = evalExpTens(p, w, sigma, r, isRel, isPer, period, X):
%   vals = evalExpTens(p, w, sigma, r, isRel, isPer, period, X, normalize):
%   vals = evalExpTens(p, w, sigma, r, isRel, isPer, period, X, ..., 'verbose', false):
%   Evaluates the density from raw arguments (builds tuples internally).
%
%   The density at a query point x is:
%     f(x) = sum_j prod(w_j) * exp(-(x - c_j)' * M * (x - c_j) / (2*sigma^2))
%   where the sum is over all ordered r-tuples drawn from (p, w), c_j is
%   the centre for the j-th tuple, and M is the appropriate quadratic form.
%
%   Dimensionality and the relative case:
%     When isRel == false (absolute), the query points are r-dimensional
%     pitch or position vectors, the tuple centres are r-tuples of
%     values from p, and M = I.
%
%     When isRel == true (relative / transposition-invariant), the density
%     is constant along the all-ones direction in R^r, so the effective
%     dimensionality is r - 1. The function works entirely in this reduced
%     space:
%       - Each r-tuple is reduced to an (r-1)-dimensional interval vector
%         by taking differences from the first element in the tuple:
%           c_j = (p_j2 - p_j1, p_j3 - p_j1, ..., p_jr - p_j1)
%       - Query points X should be (r-1)-dimensional interval vectors.
%       - The quadratic form in the reduced space is:
%           Q = sum(delta.^2) - sum(delta)^2 / r
%         where delta = x - c_j. Note the denominator is r (not r-1).
%
%     In summary, X should have dim rows, where dim = r - isRel.
%
%   Inputs:
%     dens      — Precomputed density struct from buildExpTens (OR pass the
%                 raw arguments p, w, sigma, r, isRel, isPer, period instead)
%     X         — Query points: dim x nQ matrix, where dim = r - isRel.
%                 Each column is a point at which to evaluate the density.
%                 For isRel == false: r-dimensional pitch or position
%                 vectors.
%                 For isRel == true:  (r-1)-dimensional interval vectors.
%     normalize — Optional string controlling normalization (default: 'none'):
%
%                 'none' (default):
%                   Raw weighted sum of Gaussian kernels. The absolute value
%                   depends on sigma, the number of tuples, and the weight
%                   magnitudes. Only the relative values across query points
%                   are meaningful ("this r-ad is three times more expected
%                   than that one"). Sufficient for visualization and cosine
%                   similarity, where the normalization cancels.
%
%                 'gaussian':
%                   Each Gaussian component is normalized to integrate to 1
%                   over the domain. This is achieved by multiplying the raw
%                   density by the constant:
%                     (2*pi*sigma^2)^(-dim/2) * det(M)^(1/2)
%                   where det(M) = 1 for the absolute case (M = I) and
%                   det(M) = 1/r for the relative case (M = I - 11'/r in
%                   the reduced (r-1)-dimensional space).
%                   The total integral of the density equals sum(wJ), the
%                   sum of all tuple weight products. This mode is useful
%                   for comparing densities computed with different sigma
%                   values: increasing sigma spreads the same mass over a
%                   wider area (peak height decreases), rather than inflating
%                   the total integral.
%
%                 'pdf':
%                   Full probability density normalization. Applies the
%                   Gaussian normalization above, then divides by sum(wJ)
%                   so that the density integrates to 1 over the domain.
%                   This gives a probabilistic interpretation: the value at
%                   a query point is the probability density of observing
%                   that particular r-ad. Useful for computing entropy,
%                   for use as a prior in Bayesian models, or for comparing
%                   densities across multisets of different sizes.
%
%                 Computational cost of normalization: negligible. Both
%                 modes involve only a single scalar multiply across the
%                 output vector — O(nQ) vs the O(nJ * nQ) kernel evaluation.
%
%   Output:
%     vals      — 1 x nQ row vector of density values at each query point
%
%   Optional name-value pair (all calling conventions):
%     'verbose' — Logical (default: true). If false, suppresses console
%                 output (time estimates, progress messages).
%
%   See also buildExpTens, cosSimExpTens.

% === Parse arguments ===
% Strategy: first determine whether a precomputed struct was passed as
% the first argument. Then extract X, normalize, and verbose from the
% remaining arguments.

verbose = true;  % default

% Strip 'verbose' name-value pair from varargin
verboseIdx = [];
for i = 1:numel(varargin)
    if (ischar(varargin{i}) || isstring(varargin{i})) && strcmpi(varargin{i}, 'verbose')
        if i + 1 <= numel(varargin)
            verbose = logical(varargin{i + 1});
        end
        verboseIdx = [i, i + 1]; %#ok<AGROW>
        break;
    end
end
if ~isempty(verboseIdx)
    varargin(verboseIdx) = [];
end

% Strip trailing normalize string
normalize = 'none';  % default
if numel(varargin) >= 1 && (ischar(varargin{end}) || isstring(varargin{end}))
    candidate = lower(varargin{end});
    if ismember(candidate, {'none', 'gaussian', 'pdf'})
        normalize = candidate;
        varargin(end) = [];
    end
end

% Dispatch: struct vs raw arguments
nArgs = numel(varargin);

if nArgs >= 1 && isstruct(varargin{1}) && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'ExpTensDensity')
    % --- Precomputed struct: evalExpTens(dens, X [, normalize]) ---
    dens = varargin{1};
    if nArgs ~= 2
        error(['Usage: evalExpTens(dens, X [, normalize]) or ' ...
            'evalExpTens(p, w, sigma, r, isRel, isPer, period, X [, normalize]).\n' ...
            'normalize must be ''none'', ''gaussian'', or ''pdf''.']);
    end
    X = varargin{2};

elseif nArgs == 8
    % --- Raw arguments: evalExpTens(p, w, sigma, r, isRel, isPer, period, X [, normalize]) ---
    p_arg     = varargin{1};
    w_arg     = varargin{2};
    sigma_arg = varargin{3};
    r_arg     = varargin{4};
    isRel_arg = varargin{5};
    isPer_arg = varargin{6};
    J_arg     = varargin{7};
    X         = varargin{8};
    dens = buildExpTens(p_arg, w_arg, sigma_arg, r_arg, isRel_arg, ...
                        isPer_arg, J_arg, 'verbose', verbose);
else
    error(['Usage: evalExpTens(dens, X [, normalize]) or ' ...
        'evalExpTens(p, w, sigma, r, isRel, isPer, period, X [, normalize]).\n' ...
        'normalize must be ''none'', ''gaussian'', or ''pdf''.']);
end

% === Unpack parameters ===

Centres = dens.Centres;
wJ      = dens.wJ;
nJ      = dens.nJ;
sigma   = dens.sigma;
r       = dens.r;
dim     = dens.dim;
isRel   = dens.isRel;
isPer   = dens.isPer;
J       = dens.period;

% === Validate query points ===

if size(X, 1) ~= dim
    error(['X must have %d rows (each column is a %d-dimensional ' ...
        'query point). For isRel = true, dim = r - 1 = %d.'], ...
        dim, dim, dim);
end

nQ = size(X, 2);

% === Estimated computation time ===
nPairs = double(nJ) * double(nQ);
estimateCompTime(nPairs, dim, 'evalExpTens', verbose);

% === Evaluate the density ===

vals = evalCore(Centres, wJ, nJ, X, nQ);

% === Apply normalization ===
%
% The raw output is:
%   f(x) = sum_j wJ(j) * exp(-Q_j(x) / (2*sigma^2))
%
% Two independent normalizations can be applied:
%
%   1. Gaussian normalization constant:
%      Makes each Gaussian component integrate to 1 over R^dim.
%      The constant is (2*pi*sigma^2)^(-dim/2) * det(M)^(1/2), where M
%      is the quadratic form matrix in the effective (dim-dimensional)
%      space. For the absolute case, M = I and det(M) = 1. For the
%      relative case, M = I_(r-1) - 11'/r in the reduced (r-1)-
%      dimensional interval space; its eigenvalues are (r-2) ones and
%      one eigenvalue of 1/r, giving det(M) = 1/r.
%      After this normalization, the density integrates to sum(wJ).
%
%   2. Mixture weight normalization:
%      Divides by sum(wJ) so the density integrates to 1 — a proper
%      probability density.
%
% Both normalizations are global scalar multiplies, so they do not
% change the shape of the density or the relative ordering of values.
% Their computational cost is negligible.

if ~strcmp(normalize, 'none')
    % --- Gaussian normalization ---
    % Determinant of the quadratic form matrix in the reduced space
    if isRel
        detM = 1 / r;  % det(I_(r-1) - 11'/r) = 1/r
    else
        detM = 1;       % det(I) = 1
    end

    gaussConst = (2 * pi * sigma^2)^(-dim / 2) * sqrt(detM);
    vals = vals * gaussConst;

    if strcmp(normalize, 'pdf')
        % --- Mixture weight normalization ---
        % Divide by the sum of all tuple weight products so that
        % the density integrates to 1 over the domain.
        sumW = sum(wJ);
        if sumW > 0
            vals = vals / sumW;
        else
            warning('Sum of weight products is zero; cannot normalize to pdf.');
        end
    end
end


% =====================================================================
%  NESTED HELPER FUNCTIONS
%  (r, dim, sigma, J, isPer, isRel are in scope from the parent.)
% =====================================================================

    % -----------------------------------------------------------------
    %  evalCore
    %  Evaluate the density with automatic memory-aware chunking.
    % -----------------------------------------------------------------
    function vals = evalCore(Centres, wJ, nJ, X, nQ)
        bytesNeeded = (dim + 1) * double(nJ) * double(nQ) * 8;

        try
            memInfo  = memory;
            memLimit = memInfo.MaxPossibleArrayBytes * 0.5;
        catch
            memLimit = 4e9;
        end

        if bytesNeeded <= memLimit
            vals = evalFull(Centres, wJ, nJ, X, nQ);
        else
            chunkSize = max(1, ...
                floor(memLimit / ((dim + 1) * double(nJ) * 8)));

            vals = zeros(1, nQ);
            for c = 1:chunkSize:nQ
                cEnd = min(c + chunkSize - 1, nQ);
                idx  = c:cEnd;
                vals(idx) = evalFull(Centres, wJ, nJ, X(:, idx), numel(idx));
            end
        end
    end

    % -----------------------------------------------------------------
    %  evalFull
    %  Fully vectorized density evaluation (no chunking).
    % -----------------------------------------------------------------
    function v = evalFull(Centres, wJ, nJ, Xq, nQc)
        % Difference vectors: (dim x nJ x 1) - (dim x 1 x nQc)
        D = reshape(Centres, dim, nJ, 1) - reshape(Xq, dim, 1, nQc);

        if isPer
            D = mod(D + J/2, J) - J/2;
        end

        % Quadratic form
        if isRel
            Qvec = sum(D.^2, 1) - sum(D, 1).^2 / r;
        else
            Qvec = sum(D.^2, 1);
        end

        % Gaussian kernel: flatten then reshape to nJ x nQc
        E = reshape(exp(-Qvec(:) / (2 * sigma^2)), nJ, nQc);

        % Weighted sum: (1 x nJ) * (nJ x nQc) -> (1 x nQc)
        v = wJ(:)' * E;
    end

end
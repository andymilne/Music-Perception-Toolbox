function dens = buildExpTens(p, w, sigma, r, isRel, isPer, period, varargin)
%BUILDEXPTENS Precompute an r-ad expectation tensor density object.
%
%   dens = buildExpTens(p, w, sigma, r, isRel, isPer, period):
%   dens = buildExpTens(..., 'verbose', false):
%   Precomputes the tuple index sets, pitch or position matrices, weight
%   vectors, and (for the relative case) reduced interval centres for the
%   weighted multiset (p, w). The returned struct can be passed to
%   evalExpTens and cosSimExpTens in place of the raw arguments,
%   avoiding redundant recomputation across multiple calls.
%
%   This is especially beneficial when:
%     - Evaluating the same density at many different query-point sets
%     - Computing cosine similarities between a fixed reference and many
%       comparison sets
%     - The multiset is large (the tuple enumeration cost grows rapidly)
%
%   Inputs:
%     p      — Pitch or position values (vector of length n)
%     w      — Weights (vector of length n, or empty/scalar for uniform)
%     sigma  — Standard deviation of the Gaussian kernel
%     r      — Tuple size (positive integer; r >= 2 if isRel == true)
%     isRel  — If true, use transposition-invariant (relative) quadratic
%              form (induced by the Riemannian metric on the quotient
%              space R^r / R*1)
%     isPer  — If true, wrap differences to periodic interval [-J/2, J/2)
%     period — Period J for periodic wrapping
%
%   Optional name-value pair:
%     'verbose' — Logical (default: true). If false, suppresses console
%                 output.
%
%   Output:
%     dens   — Struct with precomputed data, containing:
%              .tag      — 'ExpTensDensity' (for identification)
%              .p, .w    — Original pitch or position and weight
%                          vectors (columns)
%              .sigma, .r, .isRel, .isPer, .period — Parameters
%              .dim      — Effective dimensionality = r - isRel
%              .Centres  — dim x nJ matrix of tuple centres (reduced if
%                          isRel == true; full values otherwise)
%              .wJ       — 1 x nJ weight products for all ordered r-tuples
%              .nJ       — Number of ordered r-tuples = P(n, r)
%              .U_perm   — r x nJ_perm matrix of perm-side values
%                          (for cosSimExpTens)
%              .w_perm   — 1 x nJ_perm perm-side weight products
%              .nJ_perm  — Number of perm-side tuples (= nJ)
%              .V_comb   — r x nK matrix of comb-side values
%              .wv_comb  — 1 x nK comb-side weight products
%              .nK       — Number of comb-side tuples = C(n, r)
%
%   Terminology:
%     The object constructed here is an unnormalized r-ad Gaussian mixture
%     density: a weighted sum of Gaussian kernels centred at all ordered
%     r-tuples drawn from a weighted multiset. The term "expectation
%     tensor" is used because (a) the density represents the expected
%     perceptual distribution of r-ads given a weighted multiset, smoothed
%     by perceptual uncertainty sigma, and (b) "tensor" refers to the fact
%     that the discretized density is a rank-r array (an r-dimensional
%     grid of values), and its domain is the r-fold product of pitch or
%     position space with itself. This is "tensor" in the numerical/data
%     multidimensional array) rather than the strict algebraic sense
%     (a multilinear map with specific transformation properties).
%
%   See also evalExpTens, cosSimExpTens.

% === Parse optional 'verbose' name-value pair ===

verbose = true;  % default
for i = 1:numel(varargin)
    if (ischar(varargin{i}) || isstring(varargin{i})) && strcmpi(varargin{i}, 'verbose')
        if i + 1 <= numel(varargin)
            verbose = logical(varargin{i + 1});
        end
        break;
    end
end

% === Input validation and defaults ===

p = p(:);
w = w(:);

if isempty(w)
    w = ones(numel(p), 1);
end
if isscalar(w)
    if w == 0
        warning('All weights in w are zero.');
    end
    w = w * ones(numel(p), 1);
end

if rem(r, 1) || r < 1
    error('''r'' must be a positive integer.');
elseif r > numel(p)
    error('''r'' must not exceed the number of pitches.');
elseif numel(p) ~= numel(w)
    error('w must have the same number of entries as p.');
end

if isRel && r < 2
    error('For relative densities, ''r'' must be at least 2.');
end

dim = r - isRel;

% === Compute problem sizes ===
n      = numel(p);
nPerms = factorial(r);
nCombs = nchoosek(n, r);   % scalar: number of combinations
nJ     = nPerms * nCombs;   % total ordered r-tuples (perm side)
nK     = nCombs;            % total unordered r-tuples (comb side)

% === Estimated computation time ===
% buildExpTens is dominated by combinatorial indexing (nchoosek, perms,
% index matrix fill), not kernel evaluation. Print problem sizes so the
% user knows what to expect.
if verbose
    fprintf('buildExpTens: building %d ordered %d-tuples from %d pitches.\n', ...
        nJ, r, n);
end

% === Build all ordered r-tuples (perm side) ===
% P(n, r) = r! * C(n, r) ordered r-tuples, used by both evalExpTens
% (for density evaluation) and cosSimExpTens (as the first argument
% to the inner product).

allPerms = perms(1:r)';               % r x r!
nck      = nchoosek(1:numel(p), r)';  % r x C(n, r)

Ju     = zeros(r, nJ);
offset = 0;
for i = 1:nPerms
    Ju(:, offset + 1 : offset + nCombs) = nck(allPerms(:, i), :);
    offset = offset + nCombs;
end

% Full r-dimensional pitch matrix and weight vector (perm side)
U_perm = reshape(p(Ju), r, nJ);
w_perm = reshape(prod(reshape(w(Ju), r, nJ), 1), 1, nJ);

% === Build r-combinations (comb side, for cosSimExpTens) ===
% C(n, r) unordered r-subsets — only combinations are needed on one side
% of the inner product because the kernel is symmetric.

Kv      = nck;  % reuse the combination indices already computed
V_comb  = reshape(p(Kv), r, nK);
wv_comb = reshape(prod(reshape(w(Kv), r, nK), 1), 1, nK);

% === Reduce tuple centres if relative ===
% Each r-tuple (p1, p2, ..., pr) maps to the (r-1)-dimensional interval
% vector (p2 - p1, p3 - p1, ..., pr - p1). Used by evalExpTens.
if isRel
    Centres = U_perm(2:r, :) - U_perm(1, :);  % (r-1) x nJ
else
    Centres = U_perm;                           % r x nJ
end

% === Pack into struct ===

dens.tag     = 'ExpTensDensity';
dens.p       = p;
dens.w       = w;
dens.sigma   = sigma;
dens.r       = r;
dens.isRel   = isRel;
dens.isPer   = isPer;
dens.period  = period;
dens.dim     = dim;

% For evalExpTens
dens.Centres = Centres;
dens.wJ      = w_perm;   % same weight products, just aliased
dens.nJ      = nJ;

% For cosSimExpTens
dens.U_perm  = U_perm;
dens.w_perm  = w_perm;
dens.nJ_perm = nJ;
dens.V_comb  = V_comb;
dens.wv_comb = wv_comb;
dens.nK      = nK;

end
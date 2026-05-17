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
%   valsCell = evalExpTens({d_1, ..., d_n}, X [, normalize]):
%   valsCell = evalExpTens({d_1, ..., d_n}, {X_1, ..., X_n} [, normalize]):
%   List mode. Iterates over a cell array of density structs,
%   returning a 1-by-n cell array of value vectors. The X argument is
%   broadcast to all densities, or a length-n cell of per-density query
%   matrices may be passed for per-density evaluation. Option II shape
%   rule: a length-1 list returns a length-1 cell.
%
%   vals = evalExpTens(P, W, sigma, r, isRel, isPer, period, X [, normalize]):
%   Batched-raw mode. P is an nRows-by-K matrix of pitches (rows
%   = multisets); X is shared across all rows. Returns an nRows-by-nQ
%   matrix of values. Detection is by P having both dimensions > 1.
%   Row vectors and column vectors fall through to the existing scalar
%   SA raw path for backward compatibility.
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
%     'method'  — 'auto' (default), 'centres', 'mobius', or 'direct'
%                 (synonym for 'centres'). Point-evaluation strategy:
%                 'auto' selects via a per-call cost model; 'centres'
%                 forces the centres-array path (fast at low r);
%                 'mobius' forces the Möbius point evaluator (faster
%                 at r >= 3 since it bypasses the (dim, n_j) centres
%                 tensor whose memory and runtime scale as K!/(K-r)!).
%                 No-op on the MA path (MA always uses centres). See
%                 User Guide §4 ("Method selection").
%     'truncationSigmas' — Numeric scalar or []. Override the toolbox-
%                 wide mptDefaults('truncationSigmas') setting for this
%                 call. Centres path only; skips Gaussian contributions
%                 whose centre-to-query distance exceeds k*sigma
%                 (kernel floor exp(-k^2/2)). [] (default) means use
%                 the global default (factory: Inf).
%     'kernelPrecision' — 'double', 'single', or [] for the global
%                 default. Override the toolbox-wide kernelPrecision
%                 setting for this call. Centres path only; 'single'
%                 casts the kernel matrix to float32 for a ~2x speedup
%                 at ~7 sig fig precision.
%
%   See also buildExpTens, cosSimExpTens.

% === Parse arguments ===
% Strategy: first determine whether a precomputed struct was passed as
% the first argument. Then extract X, normalize, and verbose from the
% remaining arguments.

% Top-level call guard: resets the dispatch-message throttle on entry
% from outside the toolbox so that each user call announces afresh,
% while keeping inner sub-calls within the same top-level call
% throttled. See internal.dispatchScope.
guard = internal.dispatchScope(); %#ok<NASGU>  cleared by onCleanup

verbose = true;  % default
method = 'auto';  % 'auto' | 'centres' (alias 'direct') | 'mobius'
truncationSigmas = [];   % []: use mptDefaults at the helper level
kernelPrecision  = [];   % []: use mptDefaults at the helper level

% Strip 'verbose', 'method', 'truncationSigmas', and 'kernelPrecision'
% name-value pairs from varargin. Accept them anywhere in the trailing
% kwargs; preserve positional order of the remaining args.
removeIdx = false(1, numel(varargin));
i = 1;
while i <= numel(varargin)
    if (ischar(varargin{i}) || isstring(varargin{i})) && i + 1 <= numel(varargin)
        key = lower(char(varargin{i}));
        switch key
            case 'verbose'
                verbose = logical(varargin{i + 1});
                removeIdx(i)     = true;
                removeIdx(i + 1) = true;
                i = i + 2;
                continue;
            case 'method'
                method = lower(char(varargin{i + 1}));
                if ~ismember(method, {'auto', 'centres', 'direct', 'mobius'})
                    error('evalExpTens:badMethod', ...
                          ['''method'' must be ''auto'', ''centres'', ' ...
                           '''direct'', or ''mobius''; got ''%s''.'], method);
                end
                removeIdx(i)     = true;
                removeIdx(i + 1) = true;
                i = i + 2;
                continue;
            case 'truncationsigmas'
                truncationSigmas = varargin{i + 1};
                removeIdx(i)     = true;
                removeIdx(i + 1) = true;
                i = i + 2;
                continue;
            case 'kernelprecision'
                kernelPrecision = varargin{i + 1};
                removeIdx(i)     = true;
                removeIdx(i + 1) = true;
                i = i + 2;
                continue;
        end
    end
    i = i + 1;
end
varargin = varargin(~removeIdx);

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

% --- WindowedMaetDensity: evaluate underlying density, multiply by window ---
if nArgs >= 1 && isstruct(varargin{1}) && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'WindowedMaetDensity')
    wmd = varargin{1};
    if nArgs ~= 2
        error(['Usage for a WindowedMaetDensity: evalExpTens(wmd, X).']);
    end
    X = varargin{2};
    underlying = localEvalMA(internal.ensureExpTensExpensive(wmd.dens), X, ...
        normalize, verbose, truncationSigmas, kernelPrecision);
    % Evaluate the window function on the query points and multiply.
    W_vals = localEvaluateWindowOnQuery(wmd, X);
    vals = underlying .* W_vals;
    return;
end

% --- MA path: MaetDensity struct ---
if nArgs >= 1 && isstruct(varargin{1}) && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'MaetDensity')
    dens = internal.ensureExpTensExpensive(varargin{1});
    if nArgs ~= 2
        error(['Usage for a MaetDensity: evalExpTens(dens, X [, normalize]).\n' ...
            'X is either a cell {X_1, ..., X_A} of per-attribute query matrices, ' ...
            'or a single dim x nQ matrix with attribute rows stacked. ' ...
            'normalize must be ''none'', ''gaussian'', or ''pdf''.']);
    end
    X = varargin{2};
    vals = localEvalMA(dens, X, normalize, verbose, ...
        truncationSigmas, kernelPrecision);
    return;
end

% --- LIST path: nArgs == 2, first arg is a cell array of density structs ---
%   evalExpTens({d_1, ..., d_n}, X [, normalize])
%   evalExpTens({d_1, ..., d_n}, {X_1, ..., X_n} [, normalize])
%   Returns a 1-by-n cell of value vectors (Option II shape rule).
%   If X is a cell of length n, treated as per-density query points;
%   otherwise broadcast to all densities.
if nArgs == 2 && iscell(varargin{1}) ...
        && ~isempty(varargin{1}) && isstruct(varargin{1}{1})
    densCell = varargin{1};
    Xarg = varargin{2};
    vals = localEvalDensityList(densCell, Xarg, normalize, verbose);
    return;
end

% --- BATCHED-RAW path: nArgs == 8, first arg is a 2-D pitch matrix ---
%   evalExpTens(P, W, sigma, r, isRel, isPer, period, X [, normalize])
%   where P is nRows-by-K (rows = multisets); X is shared across rows.
%   Returns an nRows-by-nQ matrix of values.
%   Detection: numeric first arg with both dimensions > 1 (genuine
%   matrix). Row vectors and column vectors fall through to the
%   existing scalar SA raw path.
if nArgs == 8 && isnumeric(varargin{1}) ...
        && size(varargin{1}, 1) > 1 && size(varargin{1}, 2) > 1
    vals = localEvalBatchedRaw( ...
        varargin{1}, varargin{2}, varargin{3}, varargin{4}, ...
        varargin{5}, varargin{6}, varargin{7}, varargin{8}, ...
        normalize, verbose);
    return;
end

if nArgs >= 1 && isstruct(varargin{1}) && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'ExpTensDensity')
    % --- Precomputed struct: evalExpTens(dens, X [, normalize]) ---
    %   Kept skinny here: Möbius branch reads only cheap fields; centres
    %   branch ensures heavy fields on demand.
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
    % Build skinny: Möbius branch may not need heavy fields.
    dens = buildExpTens(p_arg, w_arg, sigma_arg, r_arg, isRel_arg, ...
                        isPer_arg, J_arg, 'verbose', verbose);
else
    error(['Usage: evalExpTens(dens, X [, normalize]) or ' ...
        'evalExpTens(p, w, sigma, r, isRel, isPer, period, X [, normalize]).\n' ...
        'normalize must be ''none'', ''gaussian'', or ''pdf''.']);
end

% === Validate query points (cheap fields only) ===

if size(X, 1) ~= dens.dim
    error(['X must have %d rows (each column is a %d-dimensional ' ...
        'query point). For isRel = true, dim = r - 1 = %d.'], ...
        dens.dim, dens.dim, dens.dim);
end

nQ = size(X, 2);

% === SA dispatch — two orthogonal axes ===
%
% Routing axis (forced vs discretionary):
%   - Explicit method override or hard rules (r <= 1, K - r < 2) force
%     the routing inline, with no dispatcher function call.
%   - Otherwise the unified dispatcher runs, with prescreen and (if
%     needed) probe.
%
% Execution axis (default kwargs vs feature kwargs):
%   - When truncationSigmas is empty or Inf AND kernelPrecision is
%     empty or 'double', the inline direct-broadcast path is used.
%     This is FP-identical to the helper at these settings but avoids
%     the helper's arguments-block validation and cell-array kwargs
%     construction (~hundreds of microseconds per call in MATLAB).
%   - Otherwise the helper is invoked.
%
% These two axes are independent. The fast-path is the
% (forced centres, default kwargs) corner where most consumer
% per-row tight loops live — templateHarmonicity, spectralEntropy,
% entropyExpTens scalar, etc.

% ---- Routing axis ----
if strcmp(method, 'centres') || strcmp(method, 'direct')
    chosen = 'centres';
    probed = false;
elseif strcmp(method, 'mobius')
    chosen = 'mobius';
    probed = false;
elseif strcmp(method, 'auto')
    K_src = numel(dens.p);
    if dens.r <= 1 || (K_src - dens.r) < 2
        % Hard rules force centres without a dispatcher call.
        chosen = 'centres';
        probed = false;
        if dens.r <= 1
            hardRuleReason = sprintf('r = %d', dens.r);
        else
            hardRuleReason = sprintf('K - r = %d < 2', K_src - dens.r);
        end
        % Dispatch messages bypass per-call verbose; they're gated by
        % the toolbox-wide showHints flag and throttled to once per
        % top-level user call per unique (funcName, chosen, reason)
        % triple (via +internal/dispatchScope).
        internal.maybeShowDispatchMsg('evalExpTens', chosen, ...
            hardRuleReason, 0, false);
    else
        % Discretionary case — dispatcher decides via prescreen / probe.
        [chosen, probed, estSec, routingReason] = localSelectAndEstimateSA( ...
            dens, X, nQ, method, truncationSigmas, kernelPrecision, verbose);
        internal.maybeShowDispatchMsg('evalExpTens', chosen, ...
            routingReason, estSec, probed);
    end
else
    error('evalExpTens:badMethod', ...
          ['''method'' must be ''auto'', ''centres'', ''direct'', ' ...
           'or ''mobius''; got ''%s''.'], method);
end

% ---- Execution axis: detect default-kwargs mode ----
% Important: empty ([]) means "use the global default", not "no
% feature". So we must consult mptDefaults before deciding the
% fast-path — a globally-set finite truncation or 'single' precision
% must still route through the helper.
if isempty(truncationSigmas)
    truncResolved = mptDefaults('truncationSigmas');
else
    truncResolved = truncationSigmas;
end
if isempty(kernelPrecision)
    precResolved = mptDefaults('kernelPrecision');
else
    precResolved = kernelPrecision;
end
useDefaultKwargs = ~isfinite(truncResolved) && strcmp(precResolved, 'double');

% Fire the kernel-evaluation hint once per session when the centres
% path is about to run with default kwargs. Catches the bypass
% case (which skips internal.gaussianKernelSum and would otherwise
% miss the hint).
if strcmp(chosen, 'centres') && useDefaultKwargs
    internal.maybeShowKernelEvalHint();
end

vals = [];
ranOrbit = false;
if strcmp(chosen, 'mobius')
    vals = localEvalSAOrbit(dens, X, false, truncationSigmas, kernelPrecision);
    % Post-hoc finiteness fallback. Mirrors the cosine-path safety net:
    % if the Möbius alternating partition sum produces non-finite output (extreme
    % sigma -> 0 regime), fall back to centres rather than propagating
    % NaN/Inf into the user's result.
    if ~all(isfinite(vals(:)))
        if verbose
            warning('evalExpTens:mobiusNonFiniteFallback', ...
                    ['evalExpTens Möbius method produced non-finite ' ...
                     'values; falling back to centres path.']);
        end
        chosen = 'centres';
    else
        ranOrbit = true;
    end
end

if ~ranOrbit
    % Centres branch (also entered for explicit 'centres'/'direct'
    % method, and for Möbius-then-fallback).
    dens = internal.ensureExpTensExpensive(dens);
    if useDefaultKwargs
        % Inline direct broadcast. FP-identical to the
        % helper at default settings, but skips the helper's
        % arguments-block validation and cell-array kwargs.
        vals = localEvalSACentresFast(dens, X, nQ);
    else
        % Feature kwargs requested — route through the full helper.
        vals = localEvalSACentres(dens, X, nQ, false, ...
            truncationSigmas, kernelPrecision);
    end
end

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
    sigma = dens.sigma;
    r     = dens.r;
    dim   = dens.dim;
    isRel = dens.isRel;

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
        % the density integrates to 1 over the domain. Needs wJ from
        % heavy fields; ensure if not already populated (Möbius branch
        % skipped the ensure).
        if ~isfield(dens, 'wJ')
            dens = internal.ensureExpTensExpensive(dens);
        end
        sumW = sum(dens.wJ);
        if sumW > 0
            vals = vals / sumW;
        else
            warning('Sum of weight products is zero; cannot normalize to pdf.');
        end
    end
end


end

% =========================================================================
%  SA evaluation dispatch helpers (method='auto'|'centres'|'mobius')
% =========================================================================

function chosen = localSelectSAEvalMethod(r, K, nQ, isRel, isPer, ...
                                            sigmaOverP, userMethod) %#ok<INUSD>
%LOCALSELECTSAEVALMETHOD  Choose the evaluation path for SA evalExpTens.
%
%   nQ and sigmaOverP are accepted for signature parity with future cost
%   models; current logic does not use them.
%
%   Routing rules (in order):
%     1. userMethod 'centres'/'direct'/'mobius' overrides everything.
%     2. r <= 1: the Möbius method reduces to the direct sum; centres is simpler.
%     3. isRel: the Möbius relative-mode evaluator does B_r * r * K * N_u work per query
%        (where N_u ~ 1000 for typical sigma/period), versus
%        K^r work per query for centres. For typical music-cog regimes
%        (K up to ~100, r up to 4) centres wins despite the K^r factor
%        because N_u is large and B_r * r * K * N_u > K^r. Auto stays
%        on centres; users wanting the Möbius relative-mode evaluator (e.g. for very large K
%        where centres memory blows up) opt in explicitly with
%        method='mobius'.
%     4. r == 2 and K <= 8: centres is competitive; avoids partition-
%        table dispatch overhead.
%     5. r > 8: shipped orbit tables stop at r=8 (build cost warned).
%     6. K-vs-r precision guard: orbit's Möbius alternating partition sum can
%        suffer catastrophic cancellation when K is too close to r.

    if strcmp(userMethod, 'centres') || strcmp(userMethod, 'direct')
        chosen = 'centres';
        return;
    end
    if strcmp(userMethod, 'mobius')
        chosen = 'mobius';
        return;
    end
    if ~strcmp(userMethod, 'auto')
        error('evalExpTens:badMethod', ...
              ['''method'' must be ''auto'', ''centres'', ''direct'', ' ...
               'or ''mobius''; got ''%s''.'], userMethod);
    end
    if r <= 1
        chosen = 'centres';
        return;
    end
    if isRel
        chosen = 'centres';
        return;
    end
    if r == 2 && K <= 8
        chosen = 'centres';
        return;
    end
    if r > 8   % _ORBIT_R_MAX_SHIPPED
        chosen = 'centres';
        return;
    end
    if K - r < 2   % _ORBIT_K_MINUS_R_MIN
        chosen = 'centres';
        return;
    end
    chosen = 'mobius';
end


% =========================================================================
%  Unified path-selection + time-estimate probe
%
%  The probe-based dispatcher replaces the heuristic rule for the
%  discretionary cases. Genuinely hard rules (correctness / feasibility)
%  stay as rules; everything else is decided by timing both paths on a
%  small probe and picking the faster. The probe time also produces the
%  user-facing time estimate, so dispatcher and estimator share a single
%  load-bearing measurement that auto-adapts to any future optimisation.
% =========================================================================

function s = localFormatTime(t)
%LOCALFORMATTIME  Short human-readable duration string.
    if t < 1
        s = sprintf('%.0f ms', t * 1000);
    elseif t < 60
        s = sprintf('%.1f s', t);
    elseif t < 3600
        s = sprintf('%.1f min', t / 60);
    else
        s = sprintf('%.1f hr', t / 3600);
    end
end


function nBytes = localEstimateCentresArrayBytes(K, r, isRel)
%LOCALESTIMATECENTRESARRAYBYTES  Centres-array memory estimate.
%   Returns K!/(K-r)! * dim * 8, where dim is r-1 for rel mode and
%   r for abs mode.
    if K < r
        nBytes = 0;
        return;
    end
    nJ = 1;
    for k = (K - r + 1):K
        nJ = nJ * k;
    end
    if isRel
        dim = max(r - 1, 1);
    else
        dim = r;
    end
    nBytes = nJ * dim * 8;
end


function t = localProbeEvalPath(dens, xProbe, pathName, ...
        truncationSigmas, kernelPrecision)
%LOCALPROBEEVALPATH  Time a small slice of the chosen eval path.
%   Returns elapsed seconds. The probe uses the actual code path
%   that will run for the full workload, so future optimisations
%   are automatically reflected.
%
%   Runs the path twice on xProbe: a warmup pass (discarded) to
%   stabilise CPU caches and one-shot table loads, then a timed
%   pass. Without the warmup, whichever path ran most recently on
%   the full workload comes into the probe with hot caches and
%   gets unfairly favoured; the dispatcher would then deterministically
%   flip back to the other path on subsequent calls with identical
%   inputs.
    if strcmp(pathName, 'centres')
        densMat = internal.ensureExpTensExpensive(dens);
        % Warmup pass (discarded).
        localEvalSACentres(densMat, xProbe, size(xProbe, 2), false, ...
            truncationSigmas, kernelPrecision);
        % Timed pass.
        tStart = tic;
        localEvalSACentres(densMat, xProbe, size(xProbe, 2), false, ...
            truncationSigmas, kernelPrecision);
        t = toc(tStart);
    else  % 'mobius'
        % Warmup pass (discarded).
        localEvalSAOrbit(dens, xProbe, false, ...
            truncationSigmas, kernelPrecision);
        % Timed pass.
        tStart = tic;
        localEvalSAOrbit(dens, xProbe, false, ...
            truncationSigmas, kernelPrecision);
        t = toc(tStart);
    end
end


function [chosen, probed, estSec, routingReason] = localSelectAndEstimateSA( ...
        dens, X, nQ, method, truncationSigmas, kernelPrecision, ...
        verbose) %#ok<INUSD>
%LOCALSELECTANDESTIMATESA  Unified dispatcher + time estimate for SA eval.
%
%   Hard rules decide first (correctness / feasibility), then the
%   discretionary case is decided by probing both paths and picking
%   the faster.
%
%   Returns:
%     chosen        — 'centres' or 'mobius'.
%     probed        — true if a probe ran (verbose message includes a
%                     time estimate only then).
%     estSec        — extrapolated full-workload time in seconds; 0 if
%                     no probe ran.
%     routingReason — short string describing why this path was
%                     chosen (e.g. 'r <= 1', 'rel-mode pre-screen',
%                     'estimated 4.5 s'). Used by the caller to emit
%                     a verbose dispatch message.

    % Probing parameters.
    PROBE_MIN_NQ = 200;
    PROBE_N = 50;
    CENTRES_PROBE_MEM_BUDGET = 4 * 1024^3;  % 4 GB
    % Above this r, the Möbius method becomes infeasible: B_r explodes from 115,975
    % at r=10 to 5e13 at r=20, and set-partition enumeration becomes
    % impractical. r > this falls back to centres-only routing.
    ORBIT_R_MAX_FEASIBLE = 10;
    % Bell numbers (set-partition counts) for r = 1..10.
    BELL = [1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975];
    % Pre-screens: skip the probe when one path clearly dominates.
    %  - CENTRES_DOMINANCE (rel mode): route TO centres. Orbit-rel
    %    does u-grid quadrature with N_u sub-evals per query, so its
    %    PROBE is prohibitively expensive for the typical case.
    %  - ORBIT_DOMINANCE (abs mode): route TO the Möbius method. For abs mode,
    %    centres cost per query is K^r and the Möbius cost is B_r*r*K;
    %    ratio K^(r-1)/(B_r*r) is ~1000x at K=72 r=3. Must fire
    %    BEFORE the tiny-workload shortcut so large-K abs workloads
    %    (pattern-finding and other music-cog tasks at typical
    %    24-72-partial harmonic templates) get cheap routing at
    %    any n_q.
    PRESCREEN_CENTRES_DOMINANCE = 3.0;
    PRESCREEN_ORBIT_DOMINANCE   = 3.0;

    r = double(dens.r);
    K = numel(dens.p);
    isRel = dens.isRel;
    estSec = 0.0;
    probed = false;
    routingReason = '';

    % ---- Rule 1: user override ----
    if strcmp(method, 'centres') || strcmp(method, 'direct')
        chosen = 'centres';
        routingReason = 'user override';
        return;
    end
    if strcmp(method, 'mobius')
        chosen = 'mobius';
        routingReason = 'user override';
        return;
    end
    if ~strcmp(method, 'auto')
        error('evalExpTens:badMethod', ...
              ['''method'' must be ''auto'', ''centres'', ''direct'', ' ...
               'or ''mobius''; got ''%s''.'], method);
    end

    % ---- Rule 2: Möbius method degenerate at r <= 1 ----
    if r <= 1
        chosen = 'centres';
        routingReason = sprintf('r = %d', r);
        return;
    end

    % ---- Rule 3: Möbius cancellation guard ----
    if K - r < 2   % _ORBIT_K_MINUS_R_MIN
        chosen = 'centres';
        routingReason = sprintf('K - r = %d < 2', K - r);
        return;
    end

    % ---- Rule 4: centres memory budget ----
    centresBytes = localEstimateCentresArrayBytes(K, r, isRel);
    if centresBytes > CENTRES_PROBE_MEM_BUDGET
        % Centres infeasible. Orbit is the only candidate, but it has
        % its own r-limit (B_r explodes).
        if r > ORBIT_R_MAX_FEASIBLE
            error('evalExpTens:infeasibleR', ...
                  ['r=%d requires more than %d GB for the centres ' ...
                   'array (K=%d), and the Möbius method is infeasible at r > %d ' ...
                   '(B_r explodes). Reduce r or check inputs.'], ...
                  r, floor(CENTRES_PROBE_MEM_BUDGET / 1024^3), K, ...
                  ORBIT_R_MAX_FEASIBLE);
        end
        chosen = 'mobius';
        routingReason = 'centres memory budget exceeded';
        return;
    end

    % ---- Abs-mode pre-screen: route TO the Möbius method when it clearly wins ----
    % For abs mode, centres cost per query is K^r (materialised
    % density has n_j = K^r tuples), and the Möbius absolute-mode per-query cost is
    % B_r * r * K. The ratio is K^(r-1) / (B_r * r); for K=72 r=3
    % it's ~1000x, meaning the tiny-workload shortcut below would
    % otherwise force centres for n_q<200 even when the Möbius method is 1000x
    % faster. Must run BEFORE the tiny-workload shortcut so that
    % large-K abs workloads (pattern-finding and other music-cog
    % tasks at typical 24-72-partial harmonic templates) get the
    % cheap routing decision they deserve at any n_q.
    if ~isRel && r >= 2 && r <= ORBIT_R_MAX_FEASIBLE
        if r <= numel(BELL)
            B_r_abs = BELL(r);
        else
            B_r_abs = 1e9;
        end
        centresCostAbs = double(K)^r;
        orbitCostAbs = double(B_r_abs) * r * double(K);
        if orbitCostAbs * PRESCREEN_ORBIT_DOMINANCE < centresCostAbs
            chosen = 'mobius';
            routingReason = 'abs-mode pre-screen';
            return;
        end
    end

    % ---- Shortcut: tiny workload, skip probing ----
    if nQ < PROBE_MIN_NQ
        chosen = 'centres';
        routingReason = sprintf('nQ = %d < %d', nQ, PROBE_MIN_NQ);
        return;
    end

    % ---- Rel-mode pre-screen: route TO centres when centres clearly wins ----
    % the Möbius relative-mode evaluator's u-grid quadrature makes its probe expensive at
    % typical sigma/period; pre-screen using cost ratio.
    if isRel && r >= 2
        sigma = dens.sigma;
        if dens.isPer
            N_u_est = max(64, ceil(10 * dens.period / sigma));
        else
            p_min = min(dens.p);
            p_max = max(dens.p);
            x_min_abs = min(X(:));
            x_max_abs = max(X(:));
            if isempty(x_min_abs); x_min_abs = 0; end
            if isempty(x_max_abs); x_max_abs = 0; end
            u_min = p_min - max(0, x_max_abs) - 8 * sigma;
            u_max = p_max - min(0, x_min_abs) + 8 * sigma;
            N_u_est = max(64, ceil(max(u_max - u_min, 1) / sigma * 10));
        end
        if r <= numel(BELL)
            B_r = BELL(r);
        else
            B_r = 1e9;
        end
        centresCost = double(K)^(r - 1);
        orbitCost = double(B_r) * r * N_u_est;
        if centresCost * PRESCREEN_CENTRES_DOMINANCE < orbitCost
            chosen = 'centres';
            routingReason = 'rel-mode pre-screen';
            return;
        end
    end

    % ---- Probe both paths ----
    if r >= 2 && r <= ORBIT_R_MAX_FEASIBLE
        mobius.getSetPartitionsWithMobius(r);
    end
    if r > ORBIT_R_MAX_FEASIBLE
        chosen = 'centres';
        routingReason = sprintf('r = %d > %d (Möbius infeasible)', ...
                                r, ORBIT_R_MAX_FEASIBLE);
        return;
    end

    nProbe = min(PROBE_N, nQ);
    sampleIdx = round(linspace(1, nQ, nProbe));
    xProbe = X(:, sampleIdx);

    tCentres = localProbeEvalPath(dens, xProbe, 'centres', ...
        truncationSigmas, kernelPrecision);
    tOrbit = localProbeEvalPath(dens, xProbe, 'mobius', ...
        truncationSigmas, kernelPrecision);

    if tCentres <= tOrbit
        chosen = 'centres';
        tProbe = tCentres;
    else
        chosen = 'mobius';
        tProbe = tOrbit;
    end

    estSec = tProbe * (double(nQ) / double(nProbe));
    probed = true;
    routingReason = 'probe';   % caller formats as 'estimated X s'
end


function vals = localEvalSAOrbit(dens, X, verbose, ...
        truncationSigmas, kernelPrecision) %#ok<INUSD>
%LOCALEVALSAORBIT  Orbit-Mobius point evaluator for SA densities.
%
%   Routes to mobius.evalOrbitAbs (absolute mode) or mobius.evalOrbitRel
%   (relative mode). Returns a 1-by-nQ row vector, matching the centres
%   path's output shape.
%
%   truncationSigmas and kernelPrecision are forwarded to the Möbius
%   evaluators (Stage 4: the non-periodic per-block kernel sum routes
%   through internal.gaussianKernelSum with sigma_eff = sigma/sqrt(m),
%   gaining truncation natively).

    p      = dens.p;
    w      = dens.w;
    sigma  = dens.sigma;
    r      = dens.r;
    isRel  = dens.isRel;
    isPer  = dens.isPer;
    period = dens.period;
    nQ     = size(X, 2);

    % Build kwarg list — pass through only when explicitly supplied
    % at the evalExpTens call level; otherwise the Möbius evaluators
    % consult mptDefaults themselves.
    kw = {'is_per', isPer, 'period', period};
    if ~isempty(truncationSigmas)
        kw = [kw, {'truncationSigmas', truncationSigmas}];
    end
    if ~isempty(kernelPrecision)
        kw = [kw, {'kernelPrecision', kernelPrecision}];
    end

    if isRel
        % evalOrbitRel expects X_rel as (r-1, nQ).
        vals = mobius.evalOrbitRel(p(:), w(:), sigma, r, X, kw{:});
    else
        % evalOrbitAbs expects X as (r, nQ).
        vals = mobius.evalOrbitAbs(p(:), w(:), sigma, r, X, kw{:});
    end

    vals = reshape(vals, 1, nQ);
end


function vals = localEvalSACentres(dens, X, nQ, verbose, ...
        truncationSigmas, kernelPrecision)
%LOCALEVALSACENTRES  Centres-array path for SA evaluation.
%
%   Routes through internal.gaussianKernelSum so that the
%   truncationSigmas and kernelPrecision options apply uniformly across
%   centres-path consumers. Default settings (truncationSigmas = Inf,
%   kernelPrecision = 'double') produce FP-bit-identical output to the
%   pre-truncation centres-path implementation.

    Centres = dens.Centres;
    wJ      = dens.wJ;
    nJ      = dens.nJ;
    sigma   = dens.sigma;
    r       = dens.r;
    dim     = dens.dim;
    isRel   = dens.isRel;
    isPer   = dens.isPer;
    J       = dens.period;

    % Build the keyword list for the helper. Pass-through only when
    % values were supplied at this call's level; otherwise the helper
    % consults mptDefaults.
    kw = {};
    if isRel
        kw = [kw, {'isRel', true, 'r', r}];
    end
    if isPer
        kw = [kw, {'isPer', true, 'period', J}];
    end
    if ~isempty(truncationSigmas)
        kw = [kw, {'truncationSigmas', truncationSigmas}];
    end
    if ~isempty(kernelPrecision)
        kw = [kw, {'kernelPrecision', kernelPrecision}];
    end

    vals = internal.gaussianKernelSum(Centres, wJ, X, sigma, kw{:});
end


% =========================================================================
%  localEvalSACentresFast — inline direct path for tiny workloads
%
%  Skips the internal.gaussianKernelSum helper entirely. Used by the
%  fast-path bypass at the top of evalExpTens when r <= 1, no
%  truncation, no precision override, and verbose=false. This restores
%  the per-call cost profile for per-row scalar consumers like
%  templateHarmonicity_scalar and spectralEntropy_scalar, where the
%  helper's per-call overhead (arguments block + validation + cell-array
%  kwargs building) dominates over the tiny actual compute.
%
%  Handles abs and per modes, both r=1 and r=0. is_rel=true with r<=1
%  is a structural impossibility at this entry point (the dispatcher
%  hard-rules it out before reaching here). is_per uses sawtooth
%  reduction; otherwise direct broadcast.
% =========================================================================

function vals = localEvalSACentresFast(dens, X, nQ)
%LOCALEVALSACENTRESFAST  Inline direct broadcast for the centres path.
%
%   Skips the internal.gaussianKernelSum helper entirely. Used by the
%   centres-path dispatcher when default kwargs apply (no truncation,
%   no precision override). FP-identical to the helper at these
%   settings.
%
%   Handles all (r, isRel, isPer) combinations:
%     - abs: Q(D) = sum(D .^ 2)
%     - rel: Q(D) = sum(D .^ 2) - sum(D)^2 / r
%     - per: D wrapped to (-J/2, J/2] before quadratic-form evaluation.
%
%   Avoids the per-call overhead of (a) the unified dispatcher when
%   hard rules force the route and (b) the helper's arguments-block
%   validation and cell-array kwargs construction. In MATLAB this
%   saves ~hundreds of microseconds per evalExpTens call, which is
%   the dominant cost for per-row scalar consumers like
%   templateHarmonicity / spectralEntropy in tight loops.

    Centres = dens.Centres;
    wJ      = dens.wJ;
    nJ      = dens.nJ;
    sigma   = dens.sigma;
    r       = dens.r;
    dim     = dens.dim;
    isRel   = dens.isRel;
    isPer   = dens.isPer;
    J       = dens.period;

    if nJ == 0 || nQ == 0
        vals = zeros(1, nQ);
        return;
    end

    % FP-identical to localExactKernelSum (in internal.gaussianKernelSum):
    % use D .^ 2 (not D .* D), mod-based periodic wrap (not round-based),
    % and direct division by (2 * sigma^2) (not multiplication by an
    % inverse), to preserve ULP-for-ULP equivalence with v2.0/v2.1.
    %
    % Memory-aware nQ chunking matches the helper: peak per-chunk
    % allocation is (dim+1)*nJ*nQc*8 bytes for the difference tensor
    % plus per-block intermediates. Without chunking, large workloads
    % (e.g. K=72 r=3 nQ=29161 → 155 GB) hit MATLAB's array-size cap.

    % Peak per-chunk transient ~ (2*dim + 2) * nJ * nQc * 8 (broadcast
    % difference, its square, and the summed/exponentiated intermediate
    % are briefly co-resident).
    bytesPerScalar = 8;  % default-mode is always double
    bytesNeeded = (2 * dim + 2) * double(nJ) * double(nQ) * bytesPerScalar;
    memLimit = internal.kernelChunkBytesResolved();

    if bytesNeeded <= memLimit
        vals = evalChunk(Centres, wJ, X, nQ, dim, nJ, sigma, r, isRel, isPer, J);
    else
        chunkSize = max(1, floor(memLimit / ...
            ((2 * dim + 2) * double(nJ) * bytesPerScalar)));
        vals = zeros(1, nQ);
        for c0 = 1:chunkSize:nQ
            c1 = min(c0 + chunkSize - 1, nQ);
            idx = c0:c1;
            vals(idx) = evalChunk(Centres, wJ, X(:, idx), numel(idx), ...
                dim, nJ, sigma, r, isRel, isPer, J);
        end
    end
end


function v = evalChunk(Centres, wJ, X, nQc, dim, nJ, sigma, r, isRel, isPer, J)
%EVALCHUNK  Single-chunk direct broadcast for localEvalSACentresFast.
%
%   Mirrors evalChunk in internal.gaussianKernelSum exactly so the
%   default-mode output is FP-bit-identical to v2.0/v2.1.

    D = reshape(Centres, dim, nJ, 1) - reshape(X, dim, 1, nQc);
    if isPer
        D = mod(D + J / 2, J) - J / 2;
    end
    if isRel
        Qvec = sum(D .^ 2, 1) - sum(D, 1) .^ 2 / r;
    else
        Qvec = sum(D .^ 2, 1);
    end
    E = reshape(exp(-Qvec(:) / (2 * sigma^2)), nJ, nQc);
    v = wJ(:).' * E;
end


% =========================================================================
%  localEvalMA — multi-attribute (MAET) evaluation
% =========================================================================


function vals = localEvalMA(dens, X, normalize, verbose, ...
        truncationSigmas, kernelPrecision)
%LOCALEVALMA  Evaluate a MaetDensity at query points.
%
%   Accepts X as either a 1 x A cell of per-attribute query matrices
%   (each dim_a x nQ), or a single dim x nQ matrix with attribute rows
%   stacked in attribute order. A 1-D input is coerced to 1 x nQ and is
%   valid only when the total dim equals 1.
%
%   The truncationSigmas and kernelPrecision kwargs control numerical
%   mode of the inner Q-accumulator. Single precision casts
%   intermediates to float32; truncationSigmas drops partials whose
%   accumulated q_total exceeds the threshold (post-filter; saves the
%   final exp + matmul but not the per-attribute Q computation).

    if nargin < 5
        truncationSigmas = [];
    end
    if nargin < 6
        kernelPrecision = [];
    end

    A          = dens.nAttrs;
    N_J        = dens.nJ;
    dim        = dens.dim;
    dimPerAttr = dens.dimPerAttr;
    groupOf    = dens.groupOfAttr;
    r_         = dens.r;
    sigmaG     = dens.sigma;
    isRelG     = dens.isRel;
    isPerG     = dens.isPer;
    periodG    = dens.period;
    Centres    = dens.Centres;
    wJ         = dens.wJ;

    % --- Normalise query-point input to cell form {X_1, ..., X_A} ---

    if iscell(X)
        if numel(X) ~= A
            error('evalExpTens:maQueryCellLength', ...
                  ['Query cell must have length %d (nAttrs); got %d.'], A, numel(X));
        end
        Xc = cell(1, A);
        nQ = [];
        for a = 1:A
            Xa = X{a};
            % Allow 1-D vectors when dim_a == 1
            if isvector(Xa) && dimPerAttr(a) == 1
                Xa = Xa(:).';
            end
            if size(Xa, 1) ~= dimPerAttr(a)
                error('evalExpTens:maQueryAttrRows', ...
                      ['Query for attribute %d must have %d rows; got %d.'], ...
                      a, dimPerAttr(a), size(Xa, 1));
            end
            if isempty(nQ)
                nQ = size(Xa, 2);
            elseif size(Xa, 2) ~= nQ
                error('evalExpTens:maQueryNQMismatch', ...
                      ['All per-attribute query matrices must share the same ' ...
                       'number of columns (nQ). Got %d and %d.'], nQ, size(Xa, 2));
            end
            Xc{a} = double(Xa);
        end
    else
        % Single-matrix form
        Xs = X;
        if isvector(Xs) && dim == 1
            Xs = Xs(:).';
        end
        if size(Xs, 1) ~= dim
            error('evalExpTens:maQueryTotalRows', ...
                  ['Single-matrix query must have %d rows (total dim); got %d. ' ...
                   'For cell-form input, wrap the per-attribute query matrices ' ...
                   'in a 1 x %d cell array.'], dim, size(Xs, 1), A);
        end
        nQ = size(Xs, 2);
        Xc = cell(1, A);
        rowStart = 1;
        for a = 1:A
            rowEnd = rowStart + dimPerAttr(a) - 1;
            Xc{a} = double(Xs(rowStart:rowEnd, :));
            rowStart = rowEnd + 1;
        end
    end

    if nQ == 0
        vals = zeros(1, 0);
        return;
    end

    % --- Estimated computation time (use total dim as a conservative proxy) ---
    nPairs = double(N_J) * double(nQ);
    estimateCompTime(nPairs, dim, 'evalExpTens (MAET)', verbose);

    % --- Core evaluation with memory-aware chunking ---
    % Peak memory per chunk is dominated by the largest per-attribute
    % (dim_a x nJ x nQc) difference tensor plus the (nJ x nQc)
    % accumulator. Use (maxDim + 1) * nJ * 8 bytes as the per-column
    % cost to size the chunk.

    % Peak per-chunk memory is dominated by the largest per-attribute
    % (dim_a, N_J, nQc) difference tensor, its square, and the
    % summed/exponentiated intermediate co-resident during chunk eval.
    bytesPerCol = (2 * max(dimPerAttr) + 2) * double(N_J) * 8;
    memLimit = internal.kernelChunkBytesResolved();
    bytesNeeded = bytesPerCol * double(nQ);

    if bytesNeeded <= memLimit
        vals = maetEvalFull(Xc, nQ);
    else
        chunkSize = max(1, floor(memLimit / max(bytesPerCol, 1)));
        vals = zeros(1, nQ);
        for c = 1:chunkSize:nQ
            cEnd = min(c + chunkSize - 1, nQ);
            idx  = c:cEnd;
            Xc_c = cell(1, A);
            for a = 1:A
                Xc_c{a} = Xc{a}(:, idx);
            end
            vals(idx) = maetEvalFull(Xc_c, numel(idx));
        end
    end

    % --- Normalisation ---

    if strcmp(normalize, 'gaussian') || strcmp(normalize, 'pdf')
        gaussConst = 1;
        for a = 1:A
            g = groupOf(a);
            da = dimPerAttr(a);
            if isRelG(g) && r_(a) >= 2
                detM_a = 1 / r_(a);
            else
                detM_a = 1;
            end
            gaussConst = gaussConst * ...
                (2 * pi * sigmaG(g)^2)^(-da / 2) * sqrt(detM_a);
        end
        vals = vals * gaussConst;

        if strcmp(normalize, 'pdf')
            sumW = sum(wJ);
            if sumW > 0
                vals = vals / sumW;
            else
                warning('evalExpTens:zeroSumW', ...
                        'Sum of weight products is zero; cannot normalise to pdf.');
            end
        end
    end

    % =====================================================================
    %  Inner helper: full MAET evaluation (single chunk)
    % =====================================================================

    function v = maetEvalFull(Xchunk, nQc)
        % Default-mode bypass: when no precision override and no
        % truncation are requested (after resolving against the global
        % mptDefaults), run the inline accumulator with no cast
        % machinery. FP-identical at these settings, but avoids
        % per-attribute cast() calls and the truncation branching that
        % would otherwise impose MATLAB function-call overhead per
        % evalExpTens.
        %
        % Empty ([]) means "consult global default", not "no feature".
        if isempty(truncationSigmas)
            truncResolved = mptDefaults('truncationSigmas');
        else
            truncResolved = truncationSigmas;
        end
        if isempty(kernelPrecision)
            precResolved = mptDefaults('kernelPrecision');
        else
            precResolved = kernelPrecision;
        end
        useDefault = strcmp(precResolved, 'double') && ~isfinite(truncResolved);

        if useDefault
            % Direct double accumulation path.
            Q_total = zeros(N_J, nQc);
            for a = 1:A
                g = groupOf(a);
                da = dimPerAttr(a);
                if da == 0
                    continue;
                end
                Ca = Centres{a};
                Xa = Xchunk{a};
                D_a = reshape(Ca, da, N_J, 1) - reshape(Xa, da, 1, nQc);
                if isPerG(g)
                    Pg = periodG(g);
                    D_a = mod(D_a + Pg/2, Pg) - Pg/2;
                end
                if isRelG(g)
                    Q_a = reshape(sum(D_a.^2, 1), N_J, nQc) ...
                        - reshape(sum(D_a, 1).^2, N_J, nQc) / r_(a);
                else
                    Q_a = reshape(sum(D_a.^2, 1), N_J, nQc);
                end
                Q_total = Q_total + Q_a / (2 * sigmaG(g)^2);
            end
            E = exp(-Q_total);
            v = wJ(:).' * E;
            return;
        end

        % Feature-kwargs path — precision casting and / or
        % post-filter truncation.
        if ~isempty(kernelPrecision) && strcmp(kernelPrecision, 'single')
            qDtype = 'single';
        else
            qDtype = 'double';
        end
        Q_total = zeros(N_J, nQc, qDtype);

        for a = 1:A
            g = groupOf(a);
            da = dimPerAttr(a);
            if da == 0
                continue;
            end
            Ca = cast(Centres{a}, qDtype);
            Xa = cast(Xchunk{a}, qDtype);
            D_a = reshape(Ca, da, N_J, 1) - reshape(Xa, da, 1, nQc);
            if isPerG(g)
                Pg = cast(periodG(g), qDtype);
                D_a = mod(D_a + Pg/2, Pg) - Pg/2;
            end
            if isRelG(g)
                Q_a = reshape(sum(D_a.^2, 1), N_J, nQc) ...
                    - reshape(sum(D_a, 1).^2, N_J, nQc) / cast(r_(a), qDtype);
            else
                Q_a = reshape(sum(D_a.^2, 1), N_J, nQc);
            end
            Q_total = Q_total + Q_a / (2 * cast(sigmaG(g), qDtype)^2);
        end

        % Post-filter truncation: exp(-Q_total) is negligible beyond
        % q_total > k^2/2.
        useTrunc = ~isempty(truncationSigmas) && ...
                   isfinite(truncationSigmas) && truncationSigmas > 0;
        if useTrunc
            qThreshold = double(truncationSigmas)^2 / 2;
            E = exp(-Q_total);
            E(Q_total > cast(qThreshold, qDtype)) = 0;
        else
            E = exp(-Q_total);
        end

        wJq = cast(wJ(:).', qDtype);
        v = double(wJq * E);
    end

end


% =========================================================================
%  Window pointwise evaluator (for WindowedMaetDensity dispatch)
% =========================================================================

function W_vals = localEvaluateWindowOnQuery(wmd, X)
%LOCALEVALUATEWINDOWONQUERY  Evaluate the window function W(x) on query
%points, returning a 1 x nQ vector of window values.

    dens       = wmd.dens;
    A          = dens.nAttrs;
    dimPerAttr = dens.dimPerAttr;
    dim        = dens.dim;
    groupOf    = dens.groupOfAttr;
    sigmaG     = dens.sigma;

    % --- Normalise X to per-attribute cell form (mirrors localEvalMA) ---
    if iscell(X)
        Xc = cell(1, A);
        for a = 1:A
            Xa = X{a};
            if isvector(Xa) && dimPerAttr(a) == 1
                Xa = Xa(:).';
            end
            Xc{a} = double(Xa);
        end
    else
        Xs = X;
        if isvector(Xs) && dim == 1
            Xs = Xs(:).';
        end
        Xc = cell(1, A);
        rowStart = 1;
        for a = 1:A
            rowEnd = rowStart + dimPerAttr(a) - 1;
            Xc{a} = double(Xs(rowStart:rowEnd, :));
            rowStart = rowEnd + 1;
        end
    end

    if A == 0 || isempty(Xc{1})
        W_vals = zeros(1, 0);
        return;
    end
    nQ = size(Xc{1}, 2);
    W_vals = ones(1, nQ);

    for a = 1:A
        g = groupOf(a);
        if ~localIsWindowedGroup(wmd.size(g), wmd.mix(g))
            continue;
        end
        [a_, b_] = localWindowWidthParams(wmd.size(g), wmd.mix(g), sigmaG(g));
        da = dimPerAttr(a);
        centre_a = wmd.centre{a};    % (da, 1)
        centre_a = centre_a(:);
        Xa = Xc{a};                   % (da, nQ)
        for i = 1:da
            u = Xa(i, :) - centre_a(i);
            W_vals = W_vals .* localWindowFactor1D(u, a_, b_);
        end
    end
end


function tf = localIsWindowedGroup(size_g, mix_g)
    tf = isfinite(size_g) && size_g > 0;
end


function [a_rect, b_conv] = localWindowWidthParams(size_g, mix_g, sigma_g)
    s = double(size_g) * double(sigma_g);
    a_rect = s * sqrt(3 * double(mix_g));
    b_conv = s * sqrt(1 - double(mix_g));
end


function W = localWindowFactor1D(u, a_rect, b_conv)
%LOCALWINDOWFACTOR1D  Evaluate the 1-D window function at u.
%   Window = rect(half-a) convolved with Gaussian(b).
    if b_conv == 0
        % Pure rectangular.
        W = double(abs(u) <= a_rect);
    elseif a_rect == 0
        % Pure Gaussian.
        W = exp(-u.^2 / (2 * b_conv^2));
    else
        % Rect-conv-Gaussian, normalised to peak 1.
        arg_plus  = (a_rect + u) / (sqrt(2) * b_conv);
        arg_minus = (a_rect - u) / (sqrt(2) * b_conv);
        numer = 0.5 * (erf(arg_plus) + erf(arg_minus));
        peak = erf(a_rect / (b_conv * sqrt(2)));
        W = numer / peak;
    end
end


% =====================================================================
%  Unified dispatch helpers: density-list and batched-raw modes.
% =====================================================================

function valsCell = localEvalDensityList(densCell, Xarg, normalize, verbose)
%LOCALEVALDENSITYLIST Evaluate a list of density structs at query points.
%
%   X is either broadcast (a single matrix or non-cell input) or
%   per-density (a cell array of length numel(densCell)).

    n = numel(densCell);

    % Decide whether Xarg is per-density or broadcast.
    if iscell(Xarg) && numel(Xarg) == n && ...
            (isempty(Xarg) || ~all(cellfun(@(c) isnumeric(c) && isvector(c), Xarg)))
        % Heuristic: if Xarg is a cell of length n and at least one entry
        % is non-vector or non-numeric (e.g., a per-attribute MA cell),
        % treat as per-density.
        perDensity = true;
    elseif iscell(Xarg) && numel(Xarg) == n
        % All entries are numeric vectors. Could be either per-density
        % (each is a single-attribute query) or a single MA-cell-form X
        % broadcast. Disambiguate by density type: if all densities are
        % SA, treat as per-density. Otherwise broadcast.
        allSA = true;
        for i = 1:n
            if ~isstruct(densCell{i}) || ~isfield(densCell{i}, 'tag') ...
                    || ~strcmp(densCell{i}.tag, 'ExpTensDensity')
                allSA = false;
                break;
            end
        end
        perDensity = allSA;
    else
        perDensity = false;
    end

    valsCell = cell(1, n);
    for i = 1:n
        if ~isstruct(densCell{i})
            error('evalExpTens:listNonStruct', ...
                ['evalExpTens (list mode): cell entries must be density ' ...
                 'structs from buildExpTens; entry %d is not a struct.'], i);
        end
        if perDensity
            Xi = Xarg{i};
        else
            Xi = Xarg;
        end
        valsCell{i} = evalExpTens(densCell{i}, Xi, normalize, ...
            'verbose', verbose);
    end
end


function vals = localEvalBatchedRaw(P, W, sigma, r, isRel, isPer, period, X, normalize, verbose)
%LOCALEVALBATCHEDRAW Batched evaluation from a 2-D pitch matrix.
%
%   P is nRows-by-K; X is shared across all rows. Returns an
%   nRows-by-nQ matrix of values (one row per multiset).

    nRows = size(P, 1);
    nQ = size(X, 2);

    % Pre-allocate. (We do not know nQ when isRel = true and X is dim x nQ
    % until we look at X's shape; size(X, 2) is correct in both cases.)
    vals = zeros(nRows, nQ);

    % Resolve per-row weights. If W is a matrix matching P's shape, use
    % per-row weights; if W is empty, uniform; if W is a vector matching
    % the number of columns of P, broadcast (uncommon but supported).
    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        % Allow broadcast if W matches the number of columns
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';  % row vector
        else
            error('evalExpTens:batchedWeightShape', ...
                ['evalExpTens (batched mode): W must be empty, a matrix the ' ...
                 'same size as P, or a vector matching the number of pitch columns.']);
        end
    end

    for k = 1:nRows
        pRow = P(k, :);
        % Drop NaN entries (consistent with batchCosSimExpTens convention).
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if haveRowWeights
            wK = W(k, validMask);
        elseif ~isempty(W)
            wK = W_broadcast(validMask);
        else
            wK = [];
        end
        if numel(pK) < r
            vals(k, :) = NaN;
            continue;
        end
        vals(k, :) = evalExpTens(pK, wK, sigma, r, isRel, isPer, period, ...
            X, normalize, 'verbose', verbose);
    end
end

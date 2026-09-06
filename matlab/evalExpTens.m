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
%   vals = evalExpTens(pAttr, w, sigma, r, isRel, isPer, periods, X):
%   vals = evalExpTens(pAttr, w, sigma, r, isRel, isPer, periods, X, normalize):
%   vals = evalExpTens(pAttr, w, sigma, r, isRel, isPer, periods, X, ..., 'verbose', false):
%   Raw multi-attribute mode. pAttr is a 1-by-A cell of K_a-by-N
%   attribute matrices (the same shape one would pass to buildExpTens);
%   sigma, r, isRel, isPer, periods are per-attribute vectors.
%   the per-attribute group assignment ([], length-A index vector, or
%   1-by-G cell of index lists). Builds a MaetDensity internally and
%   returns vals as a length-nQ row vector.
%
%   valsCell = evalExpTens({d_1, ..., d_n}, X [, normalize]):
%   valsCell = evalExpTens({d_1, ..., d_n}, {X_1, ..., X_n} [, normalize]):
%   List mode. Iterates over a cell array of density structs,
%   returning a 1-by-n cell array of value vectors. The X argument is
%   broadcast to all densities, or a length-n cell of per-density query
%   matrices may be passed for per-density evaluation; 'method',
%   'truncationSigmas' and 'kernelPrecision' are forwarded to every
%   entry. Option II shape rule: a length-1 list returns a length-1
%   cell.
%
%   vals = evalExpTens(P, W, sigma, r, isRel, isPer, period, X [, normalize]):
%   Batched-raw mode. P is an nRows-by-K matrix of pitches (rows
%   = multisets); X is shared across all rows. Returns an nRows-by-nQ
%   matrix of values. Detection is by P having both dimensions > 1.
%   Row vectors and column vectors fall through to the existing scalar
%   single multiset raw path for backward compatibility.
%
%   Multiset-argument shapes (pick one of three):
%     p      — Vector of length K. single multiset raw form (single multiset,
%              single-attribute).
%     P      — nRows-by-K matrix, both dimensions > 1. BATCHED-RAW
%              form (rows are independent single-attribute-style multisets,
%              processed in lockstep; returns an nRows-by-nQ matrix
%              with one row per multiset).
%     pAttr  — 1-by-A cell of K_a-by-N matrices. MA raw form
%              (multi-attribute; per-attribute centre rows).
%   Lowercase p stands for "pitch or position"; uppercase P is the
%   2-D batched lift; pAttr is the multi-attribute generalisation.
%   The same convention is used in entropyExpTens and cosSimExpTens.
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
%     dens      — Precomputed density struct from buildExpTens. Pass
%                 this in lieu of the raw arguments below; the struct
%                 carries its own sigma, r, isRel, isPer, period (or
%                 their per-group vectors for MA).
%     p / P / pAttr
%               — Pitch or position values for the raw forms. Pick the
%                 shape matching the desired calling convention (see
%                 "Multiset-argument shapes" above):
%                   p     vector of length K       (single multiset raw)
%                   P     nRows-by-K matrix        (BATCHED-RAW)
%                   pAttr 1-by-A cell of K_a-by-N  (MA raw)
%     w / W     — Weights paired with the corresponding p / P / pAttr.
%                 w is a vector (single multiset raw and MA raw); W is an nRows-by-K
%                 matrix (BATCHED-RAW). Pass [] for uniform weights.
%     sigma     — Gaussian bandwidth. Scalar for single multiset raw and BATCHED-RAW;
%                 length-G vector (one per group) for MA raw.
%     r         — Tuple size (positive integer; r >= 2 if isRel == true).
%                 Scalar for single multiset / BATCHED-RAW; length-G vector for MA.
%     isRel     — Logical: true for relative (transposition-invariant).
%                 Scalar for single multiset / BATCHED-RAW; length-G vector for MA.
%     isPer     — Logical: true for periodic domain. Scalar for single multiset /
%                 BATCHED-RAW; length-G vector for MA.
%     period    — Period of the domain. Scalar for single multiset / BATCHED-RAW;
%                 length-G vector for MA (one per group; ignored where
%                 isPer == false).
%     X         — Query points: dim x nQ matrix, where dim = r - isRel.
%                 Each column is a point at which to evaluate the density.
%                 For isRel == false: r-dimensional pitch or position
%                 vectors.
%                 For isRel == true:  (r-1)-dimensional interval vectors.
%                 For MA: a 1-by-A cell of per-attribute query matrices
%                 is also accepted.
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
%     'method'  — 'auto' (default), 'centres', or 'mobius'.
%                 Point-evaluation strategy:
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
%                 the global default (factory: 6).
%     'kernelPrecision' — 'double', 'single', or [] for the global
%                 default. Override the toolbox-wide kernelPrecision
%                 setting for this call. Honoured on every route ---
%                 the Möbius evaluator and each centres leaf (the
%                 single-multiset kernel, the factored per-attribute
%                 kernel and the joint accumulator) --- and forwarded
%                 through the list and batched forms; 'single' casts
%                 the hot-loop arrays to float32 for a ~2x speedup at
%                 ~7 sig fig precision. The Python twin honours it on
%                 the same routes.
%
%   See also buildExpTens, cosSimExpTens.

% === Parse arguments ===
% Strategy: first determine whether a precomputed struct was passed as
% the first argument. Then extract X, normalize, and verbose from the
% remaining arguments.

% Top-level call guard: resets the dispatch-message throttle on entry
% from outside the toolbox so that each user call announces afresh
% (while keeping inner sub-calls within the same top-level call
% throttled), AND pins the resolved kernelChunkBytes budget for the
% lifetime of this call so recursive / nested inner calls (e.g. via
% entropyExpTens or templateHarmonicity dispatching back into
% evalExpTens) share one OS query rather than spawning a vm_stat
% subprocess per call. Single onCleanup, halving the per-call guard
% overhead vs calling internal.dispatchScope and
% internal.kernelChunkBytesResolved('pinForCall') separately.
guard = internal.callGuard(); %#ok<NASGU>

verbose = true;  % default
method = 'auto';  % 'auto' | 'centres' | 'mobius'
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
                if ~ismember(method, {'auto', 'centres', 'mobius'})
                    error('evalExpTens:badMethod', ...
                          ['''method'' must be ''auto'', ''centres'', ' ...
                           'or ''mobius''; got ''%s''.'], method);
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
if nArgs == 0
    error('evalExpTens:noArgs', ...
        'evalExpTens requires at least one positional argument.');
end

% Optional [sym] geometry flag for the raw forms. The raw layouts carry
% one shared geometry (..., period) followed by the query X as the final
% positional. isSym joins the geometry, sitting between period and X:
%   p, w, sigma, r, isRel, isPer, period, isSym, X
% Pop it here (position 8) so the existing raw dispatch -- which expects
% X as the 8th positional -- is unchanged; forward it to buildExpTens
% via symArgs. The two-density and list forms (nArgs == 2) read isSym
% from the precomputed structs and never reach this.
symArgs = {};
isSymRaw = [];
if nArgs == 9
    isSymRaw = varargin{8};
    varargin(8) = [];
    nArgs = numel(varargin);
    if ~isempty(isSymRaw)
        symArgs = {isSymRaw};
    end
end

firstArg = varargin{1};

% ==================================================================
% Canonical dispatch order (mirrors entropyExpTens and cosSimExpTens):
%   1. Struct first operand: switch firstArg.tag.
%   2. Cell first operand:
%        - cell-of-struct  -> LIST (cell of density structs)
%        - cell-of-numeric -> MA raw (cell of attribute matrices)
%   3. Numeric first operand:
%        - 2-D with both dims > 1 -> BATCHED-RAW (rows = multisets)
%        - vector or scalar       -> single multiset raw
%   4. Otherwise -> usage error.
% MA-routed branches (general MaetDensity, LIST, MA raw, BATCHED-RAW)
% return early. Single-multiset branches (a
% single-multiset MaetDensity, single-multiset raw) set `dens` (a flat
% view) and `X` and fall through to the shared single-multiset dispatch
% below. Each detector is positive (no reliance on a preceding check
% having failed) and self-sufficient.
% ==================================================================

USAGE_MSG = ['Usage: evalExpTens(dens, X [, normalize]) or ' ...
    'evalExpTens(p, w, sigma, r, isRel, isPer, period, X [, normalize]) or ' ...
    'evalExpTens(pAttr, w, sigma, r, isRel, isPer, periods, X [, normalize]).\n' ...
    'normalize must be ''none'', ''gaussian'', or ''pdf''.'];

% --- 1. Struct first operand: precomputed density ---
if isstruct(firstArg) && isfield(firstArg, 'tag')
    if nArgs ~= 2
        error(USAGE_MSG);
    end
    X = varargin{2};
    switch firstArg.tag
        case 'MaetDensity'
            if internal.isSingleMultiset(firstArg)
                % Single-multiset corner (A = N = 1): keep the density,
                % and present the flat layout to the fast single-multiset
                % kernels below via the view (the centres branch re-views
                % after ensuring the density's per-tuple fields).
                maet = firstArg;
                dens = internal.singleMultisetView(maet);
            else
                [handled, vals] = localMaSkinnyDispatch(firstArg, X, ...
                    normalize, verbose, truncationSigmas, kernelPrecision, ...
                    method);
                if ~handled
                    dens_ma = internal.ensureExpTensExpensive(firstArg);
                    if internal.densityHasKernelCov(dens_ma)
                        X = internal.whitenQuery(dens_ma, X);
                    end
                    vals = localEvalMA(dens_ma, X, normalize, verbose, ...
                        truncationSigmas, kernelPrecision, method);
                    if ~strcmp(normalize, 'none') && ...
                            internal.densityHasKernelCov(dens_ma)
                        vals = vals * exp(-0.5 * internal.densityLogdetSum(dens_ma));
                    end
                end
                return;
            end
        otherwise
            error('evalExpTens:unknownTag', ...
                'Unknown density struct tag: %s.', firstArg.tag);
    end

% --- 2. Cell first operand: LIST or MA raw, by inner type ---
elseif iscell(firstArg) && ~isempty(firstArg)
    if isstruct(firstArg{1})
        % LIST: cell of density structs.
        if nArgs ~= 2
            error(USAGE_MSG);
        end
        vals = localEvalDensityList(firstArg, varargin{2}, normalize, ...
            verbose, method, truncationSigmas, kernelPrecision);
        return;
    end
    if isnumeric(firstArg{1})
        % MA raw: cell of attribute matrices, length-8 positional form.
        if nArgs ~= 8
            error(USAGE_MSG);
        end
        pAttr_arg  = varargin{1};
        w_arg      = varargin{2};
        sigma_arg  = varargin{3};
        r_arg      = varargin{4};
        isRel_arg  = varargin{5};
        isPer_arg  = varargin{6};
        period_arg = varargin{7};
        X          = varargin{8};
        dens = buildExpTens(pAttr_arg, w_arg, sigma_arg, r_arg, ...
                            isRel_arg, isPer_arg, period_arg, symArgs{:}, ...
                            'verbose', verbose);
        % Try the joint-free (skinny) routes first; they read only the
        % per-attribute fields buildExpTens already returns. Only build
        % the expensive joint fields when the joint accumulator is needed.
        [handled, vals] = localMaSkinnyDispatch(dens, X, normalize, ...
            verbose, truncationSigmas, kernelPrecision, method);
        if ~handled
            dens = internal.ensureExpTensExpensive(dens);
            if internal.densityHasKernelCov(dens)
                X = internal.whitenQuery(dens, X);
            end
            vals = localEvalMA(dens, X, normalize, verbose, ...
                               truncationSigmas, kernelPrecision, method);
            if ~strcmp(normalize, 'none') && internal.densityHasKernelCov(dens)
                vals = vals * exp(-0.5 * internal.densityLogdetSum(dens));
            end
        end
        return;
    end
    error('evalExpTens:badCellContents', ...
        ['Cell first argument must contain either density structs (LIST mode) ' ...
         'or numeric attribute matrices (MA raw mode); first cell entry is of ' ...
         'class %s.'], class(firstArg{1}));

% --- 3. Numeric first operand: BATCHED-RAW or single multiset raw, by shape ---
elseif isnumeric(firstArg)
    if size(firstArg, 1) > 1 && size(firstArg, 2) > 1
        % BATCHED-RAW: 2-D matrix with both dims > 1 (rows = multisets).
        if nArgs ~= 8
            error(USAGE_MSG);
        end
        vals = localEvalBatchedRaw( ...
            varargin{1}, varargin{2}, varargin{3}, varargin{4}, ...
            varargin{5}, varargin{6}, varargin{7}, varargin{8}, ...
            isSymRaw, normalize, verbose, truncationSigmas, kernelPrecision, ...
            method);
        return;
    end
    % single multiset raw: numeric vector or scalar.
    if nArgs ~= 8
        error(USAGE_MSG);
    end
    p_arg     = varargin{1};
    w_arg     = varargin{2};
    sigma_arg = varargin{3};
    r_arg     = varargin{4};
    isRel_arg = varargin{5};
    isPer_arg = varargin{6};
    J_arg     = varargin{7};
    X         = varargin{8};
    % Build the A = N = 1 corner; present the flat single-multiset layout
    % to the fast kernels below via the view. Skinny: the orbit branch may
    % not need heavy fields; the centres branch re-views after ensuring.
    maet = buildExpTens(p_arg, w_arg, sigma_arg, r_arg, isRel_arg, ...
        isPer_arg, J_arg, symArgs{:}, 'verbose', verbose);
    dens = internal.singleMultisetView(maet);
    % Fall through to single-multiset dispatch.

% --- 4. Else: usage error ---
else
    error('evalExpTens:badFirstArg', ...
        ['First argument must be a density struct, a cell array (LIST or MA ' ...
         'raw), or a numeric array (single multiset raw or BATCHED-RAW); got class %s.'], ...
        class(firstArg));
end

% === Validate query points (cheap fields only) ===

% MA query convention: a single-multiset density (A = N = 1) accepts the
% per-attribute cell query form {X_1} as well as a plain matrix, and (for
% a 1-D attribute) a bare vector. Normalise to the flat matrix the fast
% kernels expect, mirroring the general MA path (localEvalMA). Only the
% single-multiset path reaches here; every general-MA / MA-raw
% branch returned early above.
if iscell(X)
    if numel(X) ~= 1
        error('evalExpTens:maQueryCellLength', ...
              'Query cell must have length 1 (nAttrs); got %d.', numel(X));
    end
    X = X{1};
end
if isvector(X) && dens.dim == 1
    X = X(:).';
end

% Matrix-valued kernel covariance: the density's values are stored in
% whitened coordinates with sigma = 1, so the query is transformed
% once by the same map; the Gaussian normalization constant acquires
% det(Sigma)^{-1/2}, applied after the normalize block below.
if internal.densityHasKernelCov(dens)
    X = internal.whitenQuery(dens, X);
end

if size(X, 1) ~= dens.dim
    error(['X must have %d rows (each column is a %d-dimensional ' ...
        'query point). For isRel = true, dim = r - 1 = %d.'], ...
        dens.dim, dens.dim, dens.dim);
end

nQ = size(X, 2);

% === single multiset dispatch — two orthogonal axes ===
%
% Routing axis (forced vs discretionary):
%   - Explicit method override or hard rules (r <= 1) force
%     the routing inline, with no dispatcher function call.
%   - Otherwise the unified dispatcher runs, with prescreen and (if
%     needed) probe.
%
% Execution axis (default kwargs vs feature kwargs):
%   - When the resolved truncationSigmas is Inf AND the resolved
%     kernelPrecision is 'double', the inline direct-broadcast path is
%     used. This is FP-identical to the helper at these settings but avoids
%     the helper's arguments-block validation and cell-array kwargs
%     construction (~hundreds of microseconds per call in MATLAB).
%   - Otherwise the helper is invoked.
%
% These two axes are independent. The fast-path is the
% (forced centres, default kwargs) corner where most consumer
% per-row tight loops live — templateHarmonicity, spectralEntropy,
% entropyExpTens scalar, etc.

% ---- Routing axis ----
if strcmp(method, 'centres')
    chosen = 'centres';
elseif strcmp(method, 'mobius')
    % An ordered ([sym] = 0) attribute at r > 1 has no orbit: the Möbius
    % partition sum realises the symmetrised tuple set, so it would
    % evaluate a different density. Silently substituting centres would
    % hide that the requested method does not apply; silently proceeding
    % would return the wrong values. Twin of the Python
    % _reject_ordered_for_mobius guard.
    if internal.hasOrderedAttr(maet)
        error('mpt:evalExpTens:orderedMobius', ...
            ['method=''mobius'' is not available for an ordered ' ...
             '([sym]=0) attribute at r > 1: the Möbius decomposition ' ...
             'sums over set partitions of {1, ..., r}, which ' ...
             'realises the symmetrised tuple set and so evaluates a ' ...
             'different density. Use method=''centres'' (or ' ...
             'method=''auto'', which selects it).']);
    end
    chosen = 'mobius';
elseif strcmp(method, 'auto')
    % Probe-free cost-model path selection. The single-multiset corner is
    % the A = 1 case of the multi-attribute selector, so route it through
    % the same internal.selectMaEval (twin of _select_ma_eval) rather than
    % a parallel copy: one cost model, one place. The dispatch message
    % announces the routing DECISION only; the time estimate is a separate
    % concern, emitted by estimateCompTime in the executing centres path
    % under verbose.
    [chosen, routingReason] = internal.selectMaEval( ...
        maet, nQ, truncationSigmas);
    internal.maybeShowDispatchMsg('evalExpTens', chosen, routingReason);
else
    error('evalExpTens:badMethod', ...
          ['''method'' must be ''auto'', ''centres'', ' ...
           'or ''mobius''; got ''%s''.'], method);
end

% ---- Execution axis: resolve the truncation width up front ----
% Mirrors cosSimExpTens: internal.accuracyFloor('resolve', ...) maps the
% Inf "exact" sentinel to the finite accuracy-floor width (~7.43 sigma at
% the 1e-12 floor; the resolver honours a temporary epsilon override for
% arbitrary precision), passes finite widths through unchanged, and
% consults the global mptDefaults for an empty ([]) knob. Every
% centres/orbit kernel below therefore receives a finite width and
% truncates uniformly --- there is no literally-untruncated eval path.
truncResolved = internal.accuracyFloor('resolve', truncationSigmas);

vals = [];
ranOrbit = false;
if strcmp(chosen, 'mobius')
    vals = localEvalSingleMultisetOrbit(dens, X, false, truncationSigmas, kernelPrecision);
    % Post-hoc finiteness guard, shared with the multi-attribute routes
    % (localMobiusNonFinite): this corner keeps its own Möbius stack, so
    % it calls the one guard rather than owning a copy of it.
    if localMobiusNonFinite(vals)
        chosen = 'centres';
    else
        ranOrbit = true;
    end
end

if ~ranOrbit
    % Centres branch (also entered for explicit 'centres'
    % method, and for Möbius-then-fallback). Heavy fields needed: ensure
    % them on the density, then re-view for the flat kernel.
    dens = internal.singleMultisetView(internal.ensureExpTensExpensive(maet));
    % Time estimate: the centres kernel evaluates nJ * nQ (tuple, query)
    % pairs. Emitted here, in the executing path, rather than in the
    % dispatcher, so the probe-free selector reports the routing decision
    % only --- matching the Python path, whose centres branch calls
    % estimate_comp_time. Gated on verbose so it adds nothing to the
    % forced-centres tight loops (which run with verbose = false).
    if verbose
        if dens.isRel
            dimEst = max(double(dens.r) - 1, 1);
        else
            dimEst = double(dens.r);
        end
        estimateCompTime(double(dens.nJ) * double(nQ), dimEst, ...
            'evalExpTens (MAET)', verbose);
    end
    % Single centres kernel: internal.gaussianKernelSum via the helper,
    % with the resolved finite truncation width. The former inline
    % untruncated fast path is gone --- with Inf resolved to the accuracy
    % floor there is no untruncated regime to shortcut.
    vals = localEvalSingleMultisetCentres(dens, X, nQ, false, ...
        truncResolved, kernelPrecision);
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
    % Single source of the reduced-space determinant and the mass
    % constant: consumers divide by the mass to normalise, matching
    % the eval/entropy/harmony convention across the toolbox.
    detM = internal.quadraticFormDet(r, 0, isRel);
    vals = vals / internal.gaussianMassConst(sigma, dim, detM);

    if strcmp(normalize, 'pdf')
        % --- Mixture weight normalization ---
        % Divide by the sum of all tuple weight products so that
        % the density integrates to 1 over the domain. Needs wJ from
        % heavy fields; ensure if not already populated (Möbius branch
        % skipped the ensure).
        if ~isfield(dens, 'wJ')
            dens = internal.singleMultisetView(internal.ensureExpTensExpensive(maet));
        end
        sumW = sum(dens.wJ);
        if sumW > 0
            vals = vals / sumW;
        else
            warning('Sum of weight products is zero; cannot normalize to pdf.');
        end
    end
end

% Matrix-valued kernel covariance: the whitened machinery supplied the
% unit-sigma constant; the anisotropic constant differs by
% det(Sigma)^{-1/2}.
if ~strcmp(normalize, 'none') && internal.densityHasKernelCov(dens)
    vals = vals * exp(-0.5 * internal.densityLogdetSum(dens));
end


end

% =========================================================================
%  single multiset evaluation dispatch helpers (method='auto'|'centres'|'mobius')
% =========================================================================








function vals = localEvalSingleMultisetOrbit(dens, X, verbose, ...
        truncationSigmas, kernelPrecision) %#ok<INUSD>
%LOCALEVALSINGLEMULTISETORBIT  Orbit-Mobius point evaluator for single multiset densities.
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

    % --- Auto-prune zero-weight events ---
    % Per-event-level prune here (rather than joint-tuple prune as in
    % the centres path) because the orbit path operates on the raw
    % event positions, not the post-build joint tuples. An event with
    % w(i) == 0 contributes zero to every r-tuple that involves it, so
    % dropping is exact. dens.w may be scalar or per-event; only the
    % per-event case admits selective drop.
    wVec = w(:);
    if numel(wVec) > 1
        keep = wVec ~= 0;
        if ~all(keep)
            if ~any(keep)
                vals = zeros(1, nQ);
                return;
            end
            pVec = p(:);
            p = pVec(keep);
            w = wVec(keep);
        end
    elseif numel(wVec) == 1 && wVec(1) == 0
        vals = zeros(1, nQ);
        return;
    end

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

    % Thread the density's abs-per wrap opt-in through to the orbit
    % evaluator. Without this, evalOrbitAbs falls back to its default
    % 'full-image' and silently ignores wrap='single-image' set on the
    % density at build time. Twin of mobius.evalMaOrbit's per-attribute
    % wrap forwarding. Relative-mode ignores wrap (rel is always
    % all-image after v3+).
    if ~isRel && isfield(dens, 'wrap') && ~isempty(dens.wrap)
        wr = dens.wrap;
        if iscell(wr); wr = wr{1}; end
        kw = [kw, {'wrap', char(wr)}];
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


function vals = localEvalSingleMultisetCentres(dens, X, nQ, verbose, ...
        truncationSigmas, kernelPrecision)
%LOCALEVALSINGLEMULTISETCENTRES  Centres-array path for single multiset evaluation.
%
%   Routes through internal.gaussianKernelSum so that the
%   truncationSigmas and kernelPrecision options apply uniformly across
%   centres-path consumers. At truncationSigmas = Inf and
%   kernelPrecision = 'double' the output is FP-bit-identical to the
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

    % --- Auto-prune zero-weight joint perm-side tuples ---
    % See localEvalMA for the rationale: dens.wJ is the per-attribute
    % weight product, so a tuple with wJ == 0 contributes zero at every
    % query point. Strict zero convention matches the IP-path prune.
    if nJ > 0
        keep = wJ ~= 0;
        if ~all(keep)
            nJ = nnz(keep);
            if nJ == 0
                vals = zeros(1, nQ);
                return;
            end
            wJ = wJ(keep);
            Centres = Centres(:, keep);
        end
    end

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

function vals = localEvalMA(dens, X, normalize, verbose, ...
        truncationSigmas, kernelPrecision, method)
%LOCALEVALMA  Evaluate a MaetDensity at query points.
%
%   Accepts X as either a 1 x A cell of per-attribute query matrices
%   (each dim_a x nQ), or a single dim x nQ matrix with attribute rows
%   stacked in attribute order. A 1-D input is coerced to 1 x nQ and is
%   valid only when the total dim equals 1.
%
%   Dispatches between the joint-centres accumulator (below) and the
%   factored Möbius evaluator MOBIUS.EVALMAORBIT via
%   INTERNAL.SELECTMAEVAL, unless METHOD forces a route. The factored
%   path shares the same normalisation as the centres path.
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
    if nargin < 7 || isempty(method)
        method = 'auto';
    end

    A          = dens.nAttrs;
    N_J        = dens.nJ;
    dim        = dens.dim;
    dimPerAttr = dens.dimPerAttr;
    r_         = dens.r;
    sigmaG     = dens.sigma;
    isRelG     = dens.isRel;
    isPerG     = dens.isPer;
    periodG    = dens.period;
    Centres    = dens.Centres;
    wJ         = dens.wJ;

    % Per-attribute co-transposition block size (see localComputeInnerR).
    innerR = localComputeInnerR(dens, A);

    % --- Auto-prune zero-weight joint perm-side tuples ---
    % The MaetDensity build expands per-attribute value combinations into
    % joint perm-side tuples and computes wJ as the product of
    % per-attribute weights. A tuple with wJ == 0 contributes zero at
    % every query point, so dropping it is exact (matches the strict
    % zero convention of the IP-path prune in
    % mobius.maPerAttrInnerMatrix). Rebinding here is local; dens is
    % untouched on disk.
    if N_J > 0
        keep = wJ ~= 0;
        if ~all(keep)
            N_J = nnz(keep);
            wJ = wJ(keep);
            for a = 1:A
                Centres{a} = Centres{a}(:, keep);
            end
        end
    end

    % --- Normalise query-point input to cell form {X_1, ..., X_A} ---

    [Xc, nQ] = localSplitMaQuery(X, dimPerAttr, dim, A);

    if nQ == 0
        vals = zeros(1, 0);
        return;
    end

    % --- Path dispatch: factored Möbius vs joint-centres accumulator ---
    % User overrides are honoured; 'auto' consults the cost model. The
    % factored evaluator (mobius.evalMaOrbit) returns the raw density,
    % which the shared normalisation block below scales identically to
    % the centres path, so the two routes agree up to that normalisation.
    if strcmp(method, 'centres')
        maChosen = 'centres';
        maReason = 'user override';
    elseif strcmp(method, 'mobius')
        localRejectOrderedForMobius(dens);
        maChosen = 'mobius';
        maReason = 'user override';
    else
        [maChosen, maReason] = internal.selectMaEval( ...
            dens, nQ, truncationSigmas);
    end
    internal.maybeShowDispatchMsg('evalExpTens (MAET)', maChosen, ...
        maReason);

    if strcmp(maChosen, 'mobius')
        % Reconstruct the joint (D, nQ) query from the per-attribute
        % blocks and evaluate the factored orbit form.
        Xjoint = zeros(dim, nQ);
        rs = 1;
        for a = 1:A
            re = rs + dimPerAttr(a) - 1;
            Xjoint(rs:re, :) = Xc{a};
            rs = re + 1;
        end
        vArgs = {};
        if ~isempty(truncationSigmas)
            vArgs = [vArgs, {'truncationSigmas', truncationSigmas}];
        end
        if ~isempty(kernelPrecision)
            vArgs = [vArgs, {'kernelPrecision', kernelPrecision}];
        end
        valsRaw = mobius.evalMaOrbit(dens, Xjoint, vArgs{:});
        % Post-hoc finiteness guard: on a trip, fall through to the
        % centres routes below rather than return NaN/Inf.
        if ~localMobiusNonFinite(valsRaw)
            vals = valsRaw(:).';   % row, matching the centres path shape
            vals = localMaNormalise(vals, dens, normalize, ...
                dimPerAttr, innerR, sigmaG, wJ, A);
            return;
        end
        internal.maybeShowDispatchMsg('evalExpTens (MAET)', 'centres', ...
            'post-hoc guard: non-finite Möbius output');
    end

    % --- Joint-centres accumulator ---
    % The factored centres route (localMaEvalFactored) is not retried
    % here: this function is reached only after localMaSkinnyDispatch
    % has declined the shape, which happens exactly when that route is
    % unsupported (any r_a < 2, or a matrix-valued kernel covariance),
    % so a second attempt could only return empty.

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

    % Distinct-value tables for the abs-per full-image branch, one per
    % attribute, filled on first use and reused across query chunks: the
    % centres do not vary from chunk to chunk, so the sort that finds
    % their distinct values is paid once per call. absPerTableDone
    % records the attributes already considered, so an attribute the
    % predicate declines is not reconsidered on every chunk.
    absPerTable     = cell(1, A);
    absPerTableDone = false(1, A);

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

    vals = localMaNormalise(vals, dens, normalize, ...
        dimPerAttr, innerR, sigmaG, wJ, A);

    % =====================================================================
    %  Inner helper: full MAET evaluation (single chunk)
    % =====================================================================

    function v = maetEvalFull(Xchunk, nQc)
        % Resolve the truncation width once up front: internal.accuracyFloor
        % maps the Inf "exact" sentinel to the finite accuracy-floor width
        % (mirroring the single-multiset path and cosSimExpTens; it honours
        % a temporary epsilon override for arbitrary precision), consults
        % the global mptDefaults for an empty ([]) knob, and passes finite
        % widths through. The post-filter truncation below therefore always
        % applies --- there is no untruncated fast path.
        truncResolved = internal.accuracyFloor('resolve', truncationSigmas);

        % Precision: 'single' casts the accumulation; 'double' is a no-op.
        if ~isempty(kernelPrecision) && strcmp(kernelPrecision, 'single')
            qDtype = 'single';
        else
            qDtype = 'double';
        end
        Q_total = zeros(N_J, nQc, qDtype);
        % Abs-per full-image contribution accumulates multiplicatively as
        % a product of per-attribute per-coordinate theta products rather than
        % additively into Q_total. Kept as [] until the first abs-per
        % full-image attribute is encountered.
        absPerFactor = [];

        % Honour dens.wrap per attribute; absent field defaults to full-image.
        if isfield(dens, 'wrap') && ~isempty(dens.wrap)
            wrapCell = dens.wrap;
        else
            wrapCell = repmat({'full-image'}, 1, A);
        end

        for a = 1:A
            da = dimPerAttr(a);
            if da == 0
                continue;
            end
            Ca = cast(Centres{a}, qDtype);
            Xa = cast(Xchunk{a}, qDtype);
            Pg = cast(periodG(a), qDtype);

            % Abs-per full-image factorises over tuple positions, and
            % every coordinate of an r-tuple is an atom of the same
            % multiset, so the distinct arguments number K rather than
            % one per tuple. Evaluating the wrapped Gaussian once per
            % distinct value and reading the tuple layout off that table
            % presents the same floating-point arguments in the same
            % order, so the result is identical rather than equal to a
            % tolerance. It also avoids the da x N_J x nQc difference
            % array entirely.
            if innerR(a) == 0 && isPerG(a) && ~isRelG(a) ...
                    && strcmp(char(wrapCell{a}), 'full-image')
                if ~absPerTableDone(a)
                    absPerTableDone(a) = true;
                    % Decided on the whole call's query count, not this
                    % chunk's: the table serves every chunk.
                    if internal.tupleValuesRepeat(Ca, nQ)
                        [uV, ~, uI] = unique(Ca(:));
                        absPerTable{a} = {uV, reshape(uI, da, N_J)};
                    end
                end
            end
            if innerR(a) == 0 && isPerG(a) && ~isRelG(a) ...
                    && ~isempty(absPerTable{a})
                uVals = absPerTable{a}{1};
                uInv  = absPerTable{a}{2};
                factorA = [];
                for k = 1:da
                    tableK = internal.wrappedGaussian1d( ...
                        reshape(uVals, [], 1) ...
                        - reshape(Xa(k, :), 1, []), ...
                        double(sigmaG(a)), double(periodG(a)), ...
                        truncResolved, 2);
                    thetaK = tableK(uInv(k, :), :);
                    if isempty(factorA)
                        factorA = thetaK;
                    else
                        factorA = factorA .* thetaK;
                    end
                end
                factorA = cast(reshape(factorA, N_J, nQc), qDtype);
                if isempty(absPerFactor)
                    absPerFactor = factorA;
                else
                    absPerFactor = absPerFactor .* factorA;
                end
                continue;
            end

            D_a = reshape(Ca, da, N_J, 1) - reshape(Xa, da, 1, nQc);
            if innerR(a) > 0
                % Inner [rel] unit: block-diagonal metric over event blocks
                % (reduced convention; pairwise wrap inside the helper).
                Q_a = qInnerBlocksReducedLocal(D_a, innerR(a), a, Pg);
                Q_total = Q_total + Q_a / (2 * cast(sigmaG(a), qDtype)^2);
                continue;
            end
            % Abs-per: full-image via the shared wrapped-Gaussian helper
            % (density-kernel convention with exponent_denominator = 2).
            % Single-image opt-in reduces to the nearest image and falls
            % through to Q-accumulation as in the pre-v3 code.
            if isPerG(a) && ~isRelG(a)
                if strcmp(char(wrapCell{a}), 'full-image')
                    theta = internal.wrappedGaussian1d( ...
                        D_a, double(sigmaG(a)), double(periodG(a)), ...
                        truncResolved, 2);
                    factorA = reshape(prod(theta, 1), N_J, nQc);
                    factorA = cast(factorA, qDtype);
                    if isempty(absPerFactor)
                        absPerFactor = factorA;
                    else
                        absPerFactor = absPerFactor .* factorA;
                    end
                    continue;
                end
                % Single-image opt-in: reduce and fall through.
                D_a = D_a - Pg .* floor(D_a / Pg + 0.5);
            end
            if isRelG(a)
                if isPerG(a)
                    % Pairs with the implicit position 0 vectorised;
                    % the within-reduced-block pairs looped.
                    position0Wrapped = D_a - Pg .* floor(D_a / Pg + 0.5);
                    Q_a = reshape(sum(position0Wrapped .^ 2, 1), N_J, nQc);
                    for i = 1:da
                        for j = i+1:da
                            delta = reshape(D_a(i, :, :) - D_a(j, :, :), N_J, nQc);
                            delta = delta - Pg .* floor(delta / Pg + 0.5);
                            Q_a = Q_a + delta.^2;
                        end
                    end
                    Q_a = Q_a / cast(r_(a), qDtype);
                else
                    Q_a = reshape(sum(D_a.^2, 1), N_J, nQc) ...
                        - reshape(sum(D_a, 1).^2, N_J, nQc) / cast(r_(a), qDtype);
                end
            else
                Q_a = reshape(sum(D_a.^2, 1), N_J, nQc);
            end
            Q_total = Q_total + Q_a / (2 * cast(sigmaG(a), qDtype)^2);
        end

        % Post-filter truncation at the resolved width: exp(-Q_total) is
        % negligible beyond Q_total > k^2/2. truncResolved is always finite
        % (Inf resolves to the accuracy floor), so truncation always applies.
        qThreshold = truncResolved^2 / 2;
        E = exp(-Q_total);
        E(Q_total > cast(qThreshold, qDtype)) = 0;
        % Multiply in the abs-per full-image factor, if any accumulated.
        if ~isempty(absPerFactor)
            E = E .* absPerFactor;
        end

        wJq = cast(wJ(:).', qDtype);
        v = double(wJq * E);
    end

    function Q_a = qInnerBlocksReducedLocal(D_a, rIn, a, Pg)
        % Delegates to the file-scope localQInnerBlocksReduced (shared with
        % the factored centres path), passing this attribute's periodicity
        % from the enclosing scope.
        Q_a = localQInnerBlocksReduced(D_a, rIn, isPerG(a), Pg);
    end

end


function innerR = localComputeInnerR(dens, A)
%LOCALCOMPUTEINNERR  Per-attribute co-transposition block size s_u =
%   prod(r(1:u)) where attribute a is a nested attribute resolved to an
%   inner or intermediate [rel] unit u (1-based), 0 otherwise (flat,
%   absolute, and the whole-tuple outer unit ride the ordinary isRel
%   path). Switches on the block-diagonal metric. Shared by localEvalMA,
%   localMaSkinnyDispatch, and localFactoredSumW.
    innerR = zeros(1, A);
    if isfield(dens, 'nested') && iscell(dens.nested)
        for a = 1:A
            s = dens.nested{a};
            if ~isempty(s) && isstruct(s) && isfield(s, 'proj') ...
                    && (strcmp(s.proj, 'inner') || strcmp(s.proj, 'intermediate'))
                u = s.relUnit;                 % 1-based level index
                innerR(a) = prod(s.r(1:u));    % block size s_u
            end
        end
    end
end


function [Xc, nQ] = localSplitMaQuery(X, dimPerAttr, dim, A)
%LOCALSPLITMAQUERY  Normalise the query-point input to cell form
%   {X_1, ..., X_A}, accepting either a 1 x A cell of per-attribute
%   matrices (each dim_a x nQ) or a single dim x nQ matrix with attribute
%   rows stacked in attribute order. A 1-D input is coerced to 1 x nQ and
%   is valid only when the relevant dim equals 1. Shared by localEvalMA
%   and localMaSkinnyDispatch.
    if iscell(X)
        if numel(X) ~= A
            error('evalExpTens:maQueryCellLength', ...
                  ['Query cell must have length %d (nAttrs); got %d.'], A, numel(X));
        end
        Xc = cell(1, A);
        nQ = [];
        for a = 1:A
            Xa = X{a};
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
end


function [handled, vals] = localMaSkinnyDispatch(dens, X, normalize, ...
        verbose, truncationSigmas, kernelPrecision, method)
%LOCALMASKINNYDISPATCH  Evaluate an MA density without materialising the
%   joint tuple set, when a per-attribute route (factored centres or
%   Möbius) is chosen. Both read only the skinny per-attribute fields, so
%   this runs before internal.ensureExpTensExpensive and skips the joint
%   build entirely --- the memory win, and the only route that survives
%   the huge-joint shapes. Returns handled = false to defer to the joint-
%   materialising path (localEvalMA) for the cases it cannot serve: a
%   matrix-valued kernel covariance (needs the whitened joint accumulator)
%   or a factored centres shape outside localMaEvalFactored's support.
    handled = false;
    vals = [];
    if internal.densityHasKernelCov(dens)
        return;   % kernel covariance: whitening + joint accumulator path
    end
    if nargin < 7 || isempty(method)
        method = 'auto';
    end

    A          = dens.nAttrs;
    dim        = dens.dim;
    dimPerAttr = dens.dimPerAttr;
    innerR     = localComputeInnerR(dens, A);
    [Xc, nQ]   = localSplitMaQuery(X, dimPerAttr, dim, A);
    if nQ == 0
        vals = zeros(1, 0);
        handled = true;
        return;
    end

    % Path dispatch (skinny cost model; user override honoured).
    if strcmp(method, 'centres')
        maChosen = 'centres'; maReason = 'user override';
    elseif strcmp(method, 'mobius')
        localRejectOrderedForMobius(dens);
        maChosen = 'mobius'; maReason = 'user override';
    else
        [maChosen, maReason] = internal.selectMaEval( ...
            dens, nQ, truncationSigmas);
    end

    if strcmp(maChosen, 'mobius')
        internal.maybeShowDispatchMsg('evalExpTens (MAET)', maChosen, maReason);
        Xjoint = zeros(dim, nQ);
        rs = 1;
        for a = 1:A
            re = rs + dimPerAttr(a) - 1;
            Xjoint(rs:re, :) = Xc{a};
            rs = re + 1;
        end
        vArgs = {};
        if ~isempty(truncationSigmas)
            vArgs = [vArgs, {'truncationSigmas', truncationSigmas}];
        end
        if ~isempty(kernelPrecision)
            vArgs = [vArgs, {'kernelPrecision', kernelPrecision}];
        end
        valsRaw = mobius.evalMaOrbit(dens, Xjoint, vArgs{:});
        % Post-hoc finiteness guard: a non-finite Möbius value is not a
        % density value, so fall through to the centres routes below
        % (factored here, else the joint accumulator in localEvalMA).
        if ~localMobiusNonFinite(valsRaw)
            vals = localMaNormaliseSkinny(valsRaw(:).', dens, normalize, ...
                dimPerAttr, innerR, A);
            handled = true;
            return;
        end
        maChosen = 'centres';
        maReason = 'post-hoc guard: non-finite Möbius output';
    end

    % Centres route: try the factored evaluator (no joint build). If the
    % shape is unsupported, defer without emitting the dispatch message
    % (the joint path re-dispatches and emits it there).
    factored = localMaEvalFactored(dens, Xc, nQ, innerR, ...
        truncationSigmas, kernelPrecision);
    if isempty(factored)
        return;
    end
    internal.maybeShowDispatchMsg('evalExpTens (MAET)', maChosen, maReason);
    vals = localMaNormaliseSkinny(factored, dens, normalize, ...
        dimPerAttr, innerR, A);
    handled = true;
end


function sumW = localFactoredSumW(dens, innerR)
%LOCALFACTOREDSUMW  Total joint weight-product mass sum(wJ), computed
%   without building the joint: sum(wJ) = sum_events prod_attributes
%   (sum over that attribute's tuples of the per-tuple weight product).
%   Zero-weight tuples contribute nothing, so this equals the (pruned)
%   joint's sum(wJ). Used by localMaNormaliseSkinny for 'pdf'.
    A      = dens.nAttrs;
    N      = dens.N;
    rVec   = dens.r(:).';
    P      = dens.pAttr;
    W      = dens.w;
    isSymV = dens.isSym(:).';

    permCell = cell(1, A);
    for a = 1:A
        if rVec(a) < 2
            permCell{a} = [];   % r = 1: sum of valid-value weights directly
            continue;
        end
        everValid = find(any(~isnan(P{a}), 2)).';
        if innerR(a) > 0
            spec = dens.nested{a};
            tg = spec.tags;
            if isvector(tg), tg = tg(:); end
            permCell{a} = internal.nestedEnumIndices( ...
                everValid, tg(everValid, :), spec.r(:).', spec.sym(:).');
        else
            Ka = size(P{a}, 1);
            permCell{a} = internal.enumFlatAttr( ...
                zeros(Ka, 1), everValid, rVec(a), isSymV(a), ones(Ka, 1));
        end
    end

    sumW = 0;
    for n = 1:N
        prodA = 1;
        for a = 1:A
            pCol    = P{a}(:, n);
            wCol    = W{a}(:, n);
            absent  = isnan(pCol);
            wFill   = wCol;  wFill(absent | isnan(wCol)) = 0;
            if rVec(a) < 2
                % r = 1: each valid index is a 1-tuple; the total is
                % invariant to the equal-value collapse, so sum directly.
                sA = sum(wFill);
            else
                pm     = permCell{a};
                wTuple = prod(reshape(wFill(pm), size(pm, 1), size(pm, 2)), 1);
                sA     = sum(wTuple);
            end
            prodA = prodA * sA;
        end
        sumW = sumW + prodA;
    end
end


function vals = localMaNormaliseSkinny(vals, dens, normalize, ...
        dimPerAttr, innerR, A)
%LOCALMANORMALISESKINNY  MA normalisation for the joint-free (skinny)
%   routes. Identical to localMaNormalise except the 'pdf' total mass is
%   computed factored (localFactoredSumW) rather than from the joint wJ,
%   so no joint tuple set is needed.
    if ~(strcmp(normalize, 'gaussian') || strcmp(normalize, 'pdf'))
        return;
    end
    r_     = dens.r;
    isRelG = dens.isRel;
    sigmaG = dens.sigma;
    gaussConst = 1;
    for a = 1:A
        da = dimPerAttr(a);
        detM_a = internal.quadraticFormDet(r_(a), innerR(a), isRelG(a));
        gaussConst = gaussConst / internal.gaussianMassConst(sigmaG(a), da, detM_a);
    end
    vals = vals * gaussConst;
    if strcmp(normalize, 'pdf')
        sumW = localFactoredSumW(dens, innerR);
        if sumW > 0
            vals = vals / sumW;
        else
            warning('evalExpTens:zeroSumW', ...
                    'Sum of weight products is zero; cannot normalise to pdf.');
        end
    end
end


function vals = localMaEvalFactored(dens, Xc, nQ, innerR, ...
        truncationSigmas, kernelPrecision)
% LOCALMAEVALFACTORED  Factored multi-attribute centres evaluation.
%
%   The joint density is a Cartesian product across attributes within
%   each event, so its value factors:
%
%       eval(q) = sum_events prod_attributes S_a^(event)(q_a)
%
%   where S_a^(event) is attribute a's r-ad Gaussian mixture for that
%   event, evaluated at the split query. Flat factors go through the
%   culled internal.gaussianKernelSum; nested factors through the dense
%   block-diagonal form. The joint tuple set --- whose size is the
%   product of the per-attribute tuple counts --- is never materialised;
%   the cost is the sum of the per-attribute counts instead.
%
%   Absent values (NaN in a given event) are handled as zero-weight values
%   on a shared enumeration over the ever-valid indices, so events with
%   differing valid-index patterns need no special case.
%
%   Returns the raw (un-normalised) values (1 x nQ) --- the caller applies
%   localMaNormalise, identically to the joint path --- or [] when the
%   shape is outside this path's support (any r_a < 2, whose event-
%   dependent equal-value collapse breaks the shared enumeration, or a
%   matrix-valued kernel covariance), signalling a fall-back to the joint
%   maetEvalFull route. Twin of Python _ma_eval_factored.

    vals = [];   % fall-back sentinel
    A    = dens.nAttrs;
    N    = dens.N;
    rVec = dens.r(:).';
    if any(rVec < 2)
        return;   % r = 1 collapse is event-dependent; use the joint path
    end
    if internal.densityHasKernelCov(dens)
        return;   % matrix-sigma covariance: scalar per-attribute form n/a
    end

    P       = dens.pAttr;
    W       = dens.w;
    isRelV  = dens.isRel(:).';
    isPerV  = dens.isPer(:).';
    periodV = dens.period(:).';
    sigmaV  = dens.sigma(:).';
    isSymV  = dens.isSym(:).';

    % Per-attribute tuple-index structure, enumerated once over the
    % ever-valid indices (non-NaN in at least one event). The index pattern
    % is event-invariant; only the per-event positions and weights change.
    permCell = cell(1, A);
    for a = 1:A
        everValid = find(any(~isnan(P{a}), 2)).';
        if numel(everValid) < rVec(a)
            return;   % too few values for a full tuple; joint path errors
        end
        if innerR(a) > 0
            spec = dens.nested{a};
            tg = spec.tags;
            if isvector(tg)
                tg = tg(:);
            end
            tagsValid = tg(everValid, :);
            permCell{a} = internal.nestedEnumIndices( ...
                everValid, tagsValid, spec.r(:).', spec.sym(:).');
        else
            Ka = size(P{a}, 1);
            permCell{a} = internal.enumFlatAttr( ...
                zeros(Ka, 1), everValid, rVec(a), isSymV(a), ones(Ka, 1));
        end
    end

    total = zeros(1, nQ);
    for n = 1:N
        prodN = ones(1, nQ);
        for a = 1:A
            pm      = permCell{a};
            pCol    = P{a}(:, n);
            wCol    = W{a}(:, n);
            absent  = isnan(pCol);
            % Finite placeholder keeps the kernel finite; zero weight nulls
            % any tuple touching an absent value (0 * exp(finite) = 0).
            pFill = pCol;  pFill(absent) = 0;
            wFill = wCol;  wFill(absent | isnan(wCol)) = 0;
            Dtup    = size(pm, 1);
            M       = size(pm, 2);
            u       = reshape(pFill(pm), Dtup, M);          % Dtup x M
            wTuple  = prod(reshape(wFill(pm), Dtup, M), 1);  % 1 x M
            if innerR(a) > 0
                % Nested: reduce each inner block by its own first coordinate,
                % then a dense block-diagonal quadratic form (nested M is
                % small, so the cull is not needed here).
                rIn  = innerR(a);
                rOut = Dtup / rIn;
                dc   = rOut * (rIn - 1);
                c    = zeros(dc, M);
                for b = 1:rOut
                    blk  = (b - 1) * rIn + (1:rIn);
                    ublk = u(blk, :);
                    c((b - 1) * (rIn - 1) + (1:(rIn - 1)), :) = ...
                        ublk(2:end, :) - ublk(1, :);
                end
                D_a = reshape(c, dc, M, 1) - reshape(Xc{a}, dc, 1, nQ);
                Q_a = localQInnerBlocksReduced( ...
                          D_a, rIn, isPerV(a), periodV(a)) ...
                      / (2 * sigmaV(a)^2);
                S_a = wTuple * exp(-Q_a);               % 1 x nQ
            else
                if isRelV(a)
                    c = u(2:end, :) - u(1, :);          % (r-1) x M reduced
                else
                    c = u;                              % r x M absolute
                end
                kw = {};
                if isRelV(a)
                    kw = [kw, {'isRel', true, 'r', rVec(a)}];
                end
                if isPerV(a)
                    kw = [kw, {'isPer', true, 'period', periodV(a)}];
                    % The attribute's declared wrap goes with it, as on
                    % the single-multiset and joint centres routes:
                    % without it a 'single-image' abs-per attribute was
                    % evaluated full-image whenever this route was taken.
                    if isfield(dens, 'wrap') && iscell(dens.wrap) ...
                            && numel(dens.wrap) >= a ...
                            && ~isempty(dens.wrap{a})
                        kw = [kw, {'wrap', char(dens.wrap{a})}];
                    end
                end
                if ~isempty(truncationSigmas)
                    kw = [kw, {'truncationSigmas', truncationSigmas}];
                end
                if ~isempty(kernelPrecision)
                    kw = [kw, {'kernelPrecision', kernelPrecision}];
                end
                S_a = internal.gaussianKernelSum( ...
                          c, wTuple(:), Xc{a}, sigmaV(a), kw{:});
            end
            prodN = prodN .* S_a(:).';
        end
        total = total + prodN;
    end
    vals = total;
end


function Q_a = localQInnerBlocksReduced(D_a, rIn, isPer, Pg)
% LOCALQINNERBLOCKSREDUCED  Block-diagonal quadratic form for the inner
%   [rel] co-transposition unit (reduced / centres convention). D_a is
%   (rOut*(rIn-1)) x nJ x nQc; each event block is the (rIn-1)-row first-coordinate
%   reduction of an rIn-tuple. Q_a is the sum over blocks of the per-block
%   flat relative quotient form (the within-event intervals, tensor-joined
%   across events). Twin of Python _compute_Q_inner_blocks. Shared by the
%   joint accumulator (maetEvalFull) and the factored centres path.
    nJ_  = size(D_a, 2);
    nQc_ = size(D_a, 3);
    Q_a  = zeros(nJ_, nQc_, 'like', D_a);
    blk  = rIn - 1;
    if blk <= 0
        return;   % rIn == 1: trivial (dim 0) inner space
    end
    nBlocks = size(D_a, 1) / blk;
    for b = 1:nBlocks
        rows = (b - 1) * blk + (1:blk);
        Db = D_a(rows, :, :);
        if isPer
            position0Wrapped = Db - Pg .* floor(Db / Pg + 0.5);
            Qb = reshape(sum(position0Wrapped .^ 2, 1), nJ_, nQc_);
            for i = 1:blk
                for j = i+1:blk
                    delta = reshape(Db(i, :, :) - Db(j, :, :), nJ_, nQc_);
                    delta = delta - Pg .* floor(delta / Pg + 0.5);
                    Qb = Qb + delta .^ 2;
                end
            end
            Qb = Qb / rIn;
        else
            Qb = reshape(sum(Db .^ 2, 1), nJ_, nQc_) ...
               - reshape(sum(Db, 1) .^ 2, nJ_, nQc_) / rIn;
        end
        Q_a = Q_a + Qb;
    end
end


% =====================================================================
%  Unified dispatch helpers: density-list and batched-raw modes.
% =====================================================================

function valsCell = localEvalDensityList(densCell, Xarg, normalize, ...
                                         verbose, method, ...
                                         truncationSigmas, kernelPrecision)
%LOCALEVALDENSITYLIST Evaluate a list of density structs at query points.
%
%   X is either broadcast (a single matrix or non-cell input) or
%   per-density (a cell array of length numel(densCell)). METHOD,
%   TRUNCATIONSIGMAS and KERNELPRECISION are forwarded to every entry,
%   as the Python list form forwards them (earlier versions forwarded
%   ``normalize`` and ``verbose`` alone, so a forced route or a per-call
%   width was silently ignored in list mode).

    if nargin < 5 || isempty(method); method = 'auto'; end
    if nargin < 6; truncationSigmas = []; end
    if nargin < 7; kernelPrecision = []; end
    entryKw = {'verbose', verbose, 'method', method};
    if ~isempty(truncationSigmas)
        entryKw = [entryKw, {'truncationSigmas', truncationSigmas}];
    end
    if ~isempty(kernelPrecision)
        entryKw = [entryKw, {'kernelPrecision', kernelPrecision}];
    end
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
        % broadcast. Disambiguate by density shape: if all densities are
        % single-multiset, treat as per-density. Otherwise broadcast.
        allSingleMultiset = true;
        for i = 1:n
            if ~internal.isSingleMultiset(densCell{i})
                allSingleMultiset = false;
                break;
            end
        end
        perDensity = allSingleMultiset;
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
        valsCell{i} = evalExpTens(densCell{i}, Xi, normalize, entryKw{:});
    end
end


function vals = localEvalBatchedRaw(P, W, sigma, r, isRel, isPer, period, X, isSym, normalize, verbose, truncationSigmas, kernelPrecision, method)
%LOCALEVALBATCHEDRAW Batched evaluation from a 2-D pitch matrix.
%
%   P is nRows-by-K; X is shared across all rows. Returns an
%   nRows-by-nQ matrix of values (one row per multiset).
%
%   truncationSigmas, kernelPrecision and method are forwarded to the
%   per-row evaluation so a caller-supplied kernel width and route apply
%   uniformly across every row (an empty value defers to the toolbox
%   default downstream), as the Python batched form forwards them.
    if nargin < 12, truncationSigmas = []; end
    if nargin < 13, kernelPrecision  = []; end
    if nargin < 14 || isempty(method), method = 'auto'; end

    % The per-row dedup keys rows by a multiset canonical form, which
    % collapses rows that share a multiset but differ in order. That is
    % correct only for the symmetric reading: under isSym = false the
    % order is significant, so the dedup would silently merge distinct
    % ordered densities. Reject rather than return a wrong answer
    % (parity with the Python batched path). Order-aware batched dedup is
    % a tracked follow-up; evaluate ordered densities one row at a time.
    if nargin >= 9 && ~isempty(isSym) && ~all(logical(isSym(:))) && r > 1
        error('evalExpTens:batchedOrderedUnsupported', ...
              ['evalExpTens batched (2-D) input does not yet support ' ...
               'isSym = false (ordered) densities at r > 1: the batched ' ...
               'dedup canonicalises each row''s multiset and would merge ' ...
               'order-distinct rows. Evaluate ordered densities one row ' ...
               'at a time (vector input).']);
    end

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

    rowKw = {'verbose', verbose, 'method', method};
    if ~isempty(truncationSigmas)
        rowKw = [rowKw, {'truncationSigmas', truncationSigmas}];
    end
    if ~isempty(kernelPrecision)
        rowKw = [rowKw, {'kernelPrecision', kernelPrecision}];
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
            X, normalize, rowKw{:});
    end
end


function vals = localMaNormalise(vals, dens, normalize, ...
        dimPerAttr, innerR, sigmaG, wJ, A)
%LOCALMANORMALISE  Shared MA normalisation for the centres and factored paths.
%   Applies the per-attribute Gaussian constant (with the co-transposition
%   block-diagonal metric determinant) for 'gaussian'/'pdf', then divides
%   by the total weight-product mass for 'pdf'. Identical for both the
%   joint-centres accumulator and the factored Möbius evaluator, so the
%   two routes agree up to this scaling. Twin of python _ma_eval_normalize.
    if ~(strcmp(normalize, 'gaussian') || strcmp(normalize, 'pdf'))
        return;
    end
    r_    = dens.r;
    isRelG = dens.isRel;
    gaussConst = 1;
    for a = 1:A
        da = dimPerAttr(a);
        detM_a = internal.quadraticFormDet(r_(a), innerR(a), isRelG(a));
        gaussConst = gaussConst / internal.gaussianMassConst(sigmaG(a), da, detM_a);
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


function tripped = localMobiusNonFinite(vals)
%LOCALMOBIUSNONFINITE  Post-hoc guard on a Möbius point-evaluation result.
%   True (with a warning) when any value is non-finite --- an overflowed
%   alternating sum, or a NaN from a degenerate block --- so the caller
%   re-evaluates through the centres path, which never cancels. One
%   guard for every route (single-multiset, skinny and joint), gated
%   like the cosine path's guard on mptDefaults('postHocGuards'). Twin
%   of the guard in the Python _eval_exp_tens_ma.
    tripped = false;
    if ~logical(mptDefaults('postHocGuards'))
        return;
    end
    if all(isfinite(vals(:)))
        return;
    end
    tripped = true;
    warning('evalExpTens:mobiusNonFiniteFallback', ...
            ['evalExpTens: the Möbius evaluator returned non-finite ' ...
             'values; falling back to the centres path for this call.']);
end


function localRejectOrderedForMobius(dens)
%LOCALREJECTORDEREDFORMOBIUS  Refuse method='mobius' on an ordered or nested attribute.
%
%   The Möbius decomposition sums over set partitions of {1, ..., r},
%   which realises the symmetrised tuple set; on an ordered ([sym]=0)
%   attribute at r > 1 that is a different density, not a faster route
%   to the same one, so an explicit request is an error rather than a
%   silent symmetrisation. The single-multiset path carries the same
%   check inline; this is its MA twin, and the twin of the Python
%   _reject_ordered_for_mobius, which runs for every density shape.
    if internal.hasOrderedAttr(dens)
        error('mpt:evalExpTens:orderedMobius', ...
            ['method=''mobius'' is not available for an ordered ' ...
             '([sym]=0) attribute at r > 1: the Möbius decomposition ' ...
             'sums over set partitions of {1, ..., r}, which ' ...
             'realises the symmetrised tuple set and so evaluates a ' ...
             'different density. Use method=''centres'' (or ' ...
             'method=''auto'', which selects it).']);
    end
    % A nested attribute is the same case: the flat factored evaluator
    % would read the nested values as one flat multiset and evaluate a
    % density with a different tuple set. The point evaluator has no
    % nested analogue of the inner-product contraction, so the joint
    % centres are the only route.
    if isfield(dens, 'nested') && iscell(dens.nested) ...
            && any(~cellfun(@isempty, dens.nested))
        error('mpt:evalExpTens:nestedMobius', ...
            ['method=''mobius'' is not available for a nested ' ...
             'attribute: the flat Möbius evaluator reads the nested ' ...
             'values as one flat multiset and so evaluates a different ' ...
             'density. Use method=''centres'' (or method=''auto'', ' ...
             'which selects it).']);
    end
end

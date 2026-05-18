function s = cosSimExpTens(varargin)
%COSSIMEXPTENS Cosine similarity of two r-ad expectation tensor densities.
%
%   s = cosSimExpTens(dens_x, dens_y):
%   s = cosSimExpTens(dens_x, dens_y, 'verbose', false):
%   Cosine similarity using precomputed density structs from buildExpTens.
%   This avoids recomputing tuple indices and weight products on each call,
%   and is the preferred calling convention when comparing a fixed reference
%   against many other sets.
%
%   s = cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period):
%   s = cosSimExpTens(..., 'verbose', false):
%   Cosine similarity from raw arguments (builds density structs
%   internally via buildExpTens).
%
%   sCell = cosSimExpTens({d_x_1, ..., d_x_n}, {d_y_1, ..., d_y_n}):
%   List mode. Iterates over paired entries of two cell arrays of
%   density structs, returning a 1-by-n cell array of similarity values.
%   Each pair is dispatched to the appropriate scalar form based on its
%   tag (SA or MA). Option II shape rule: a length-1 input
%   returns a length-1 cell (no collapse to scalar).
%
%   Scalar-vs-list broadcasting. Either operand may be a single
%   density struct paired with a cell array of density structs; the
%   single struct is broadcast against every entry of the cell, and a
%   1-by-n cell is returned. Useful for "compare one reference density
%   against many" without first wrapping the reference in {ref} on the
%   call site.
%
%   s = cosSimExpTens(P1, W1, P2, W2, sigma, r, isRel, isPer, period):
%   Batched-raw mode. At least one of P1, P2 is an M-by-K
%   matrix (both dimensions > 1); the function returns an M-by-1
%   vector of similarities. Pass [] for W1 or W2 to use uniform
%   weights. Equivalent to batchCosSimExpTens (which is now deprecated).
%
%   Broadcasting. If one operand is a vector of length K
%   (1-by-K, K-by-1, or 1-D) and the other is M-by-K with M > 1, the
%   vector is broadcast across the matrix's M rows, in NumPy / MATLAB
%   implicit-expansion style. The corresponding weight argument
%   (W1 or W2) is broadcast in lockstep when non-empty. This avoids
%   the explicit repmat(refPitches, M, 1) idiom for the common case
%   "compare one reference multiset against many candidates".
%
%   In batched-raw mode the following name-value options are accepted
%   (all forwarded to the underlying paired-rows implementation):
%     'spectrum'  — Cell of arguments to addSpectra; if supplied,
%                   partials are added to each row's pitches before
%                   computing similarity. Example: {'harmonic', 12,
%                   'powerlaw', 1}. Default: not applied.
%     'precision' — Round pitch and weight values to nDec decimal
%                   places before deduplication, to absorb arithmetic
%                   noise. Default: full floating-point precision.
%     'dedup'     — Logical (default true). Currently a no-op for
%                   'dedup', true; 'dedup', false emits a warning since
%                   the batched implementation always deduplicates.
%
%   Computes the cosine similarity between the r-ad expectation tensor
%   densities of two weighted multisets (p represents pitches or
%   positions), with Gaussian perception error of standard deviation
%   sigma. The cosine similarity is computed analytically — no grid
%   evaluation is required.
%
%   Four variants are available, depending on the flags isPer and isRel:
%   the inner product assumes periodic equivalence with period set by
%   'period' if isPer == true, and assumes transpositional equivalence
%   (relative rather than absolute pitches or positions) if isRel == true.
%   See buildExpTens for further information about these parameters.
%
%   Inputs (struct calling convention):
%     dens_x — Precomputed density struct from buildExpTens.
%     dens_y — Precomputed density struct from buildExpTens.
%              Both structs must share the same r, sigma, isRel, isPer,
%              and (if periodic) period.
%
%   Inputs (raw calling convention):
%     p1     — Pitch or position values for the first multiset (vector
%              of length n_1).
%     w1     — Weights for the first multiset (vector of length n_1, or
%              empty/scalar for all ones).
%     p2     — Pitch or position values for the second multiset (vector
%              of length n_2).
%     w2     — Weights for the second multiset (vector of length n_2, or
%              empty/scalar for all ones).
%     sigma  — Standard deviation of the Gaussian kernel.
%     r      — Tuple size (positive integer; r >= 2 if isRel == true).
%     isRel  — If true, use transposition-invariant (relative)
%              quadratic form.
%     isPer  — If true, wrap differences to periodic interval [-J/2, J/2).
%     period — Period J for periodic wrapping.
%
%   Optional name-value pair (all calling conventions):
%     'verbose' — Logical (default: true). If false, suppresses console
%                 output (time estimates, progress messages).
%     'method'  — 'auto' (default), 'bulger', 'mobius', or 'direct'.
%                 Inner-product decomposition. 'auto' selects via a
%                 per-call cost model (with a timing-probe fallback for
%                 indeterminate SA cases) between Bulger's method
%                 (small r and small K) and the Möbius method (large r
%                 or large K). 'bulger' / 'mobius' force the named
%                 method; 'direct' enumerates every ordered r-tuple on
%                 each side (expensive, immune to cancellation,
%                 primarily for benchmarking). See User Guide §4
%                 ("Method selection").
%     'cancellationThreshold' — Positive scalar (default: 1e-12).
%                 Guards the Möbius alternating-sum against
%                 catastrophic cancellation: when the cancellation
%                 ratio |IP| / sqrt(<A,A>*<B,B>) drops below this
%                 fraction, the dispatcher falls back to Bulger's
%                 method. Lower to relax the guard; raise to force
%                 Bulger's method more aggressively.
%     'truncationSigmas' — Numeric scalar or []. Override the toolbox-
%                 wide mptDefaults('truncationSigmas') setting for this
%                 call. Applies on the centres path (Bulger's method
%                 on the SA inner product); skips Gaussian
%                 contributions whose centre-to-query distance exceeds
%                 k*sigma (kernel floor exp(-k^2/2)). [] (default)
%                 means use the global default (factory: Inf). No
%                 effect on Möbius-method calls.
%     'kernelPrecision' — 'double', 'single', or [] for the global
%                 default. Override the toolbox-wide kernelPrecision
%                 setting for this call. Centres path only; 'single'
%                 casts the kernel matrix to float32 for a ~2x speedup
%                 at ~7 sig fig precision. No effect on Möbius-method
%                 calls.
%
%   Output:
%     s      — Cosine similarity (scalar in [0, 1] for non-negative
%              weights). Returns NaN if r exceeds the number of elements
%              in either multiset.
%
%   Originally by David Bulger, Macquarie University, Australia (2016).
%   Adapted for the Music Perception Toolbox v2 by Andrew J. Milne
%   (The MARCS Institute, Western Sydney University): preallocated
%   permutation indices, vectorized inner product computation, simplified
%   quadratic form, precomputed index/pitch/weight data shared across the
%   three inner product calls, automatic chunking for large arrays, and
%   optional precomputed density structs via buildExpTens.
%
%   See also buildExpTens, evalExpTens.
%   See also batchCosSimExpTens (deprecated; folded into the batched-raw
%   mode of cosSimExpTens above).

% === Parse arguments ===

% Extract optional name-value pairs that may follow the positional
% args. 'verbose' applies to all dispatch arms; 'method' and
% 'cancellationThreshold' apply to SA and MA struct/raw-args paths
% (Möbius dispatch); 'spectrum', 'precision', and 'dedup' are
% valid only for the batched-raw path and are forwarded to
% batchCosSimExpTens. Each is captured (with its index range) and
% removed from varargin before the dispatch sees it, so the dispatch
% logic only has to inspect positional arguments.

% Top-level call guard: see internal.dispatchScope.
guard = internal.dispatchScope(); %#ok<NASGU>

% Pin the resolved kernelChunkBytes budget for the lifetime of this
% call so recursive inner calls — including the batched-raw mode's
% per-unique-pair recursion below — share one OS query rather than
% spawning a vm_stat subprocess per call.
chunkPin = internal.kernelChunkBytesResolved('pinForCall'); %#ok<NASGU>

verbose = true;
method = 'auto';                % 'auto' | 'bulger' | 'mobius'
cancellationThreshold = 1e-12;  % cross-cancellation guard
truncationSigmas = [];          % []: use mptDefaults at the helper level
kernelPrecision  = [];          % []: use mptDefaults at the helper level
spectrumOpt = [];     % []  ⇒ no spectrum kwarg passed downstream
precisionOpt = [];    % []  ⇒ no precision kwarg passed downstream
dedupOpt = [];        % []  ⇒ no dedup kwarg passed downstream
spectrumGiven = false;
precisionGiven = false;
dedupGiven = false;

i = 1;
keepMask = true(1, numel(varargin));
while i <= numel(varargin)
    if (ischar(varargin{i}) || isstring(varargin{i})) && i + 1 <= numel(varargin)
        key = lower(char(varargin{i}));
        switch key
            case 'verbose'
                verbose = logical(varargin{i + 1});
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
            case 'method'
                method = lower(char(varargin{i + 1}));
                if ~ismember(method, {'auto', 'bulger', 'mobius'})
                    error('cosSimExpTens:badMethod', ...
                          ['''method'' must be ''auto'', ''bulger'', ' ...
                           'or ''mobius''; got ''%s''.'], method);
                end
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
            case 'cancellationthreshold'
                cancellationThreshold = double(varargin{i + 1});
                if ~isscalar(cancellationThreshold) || cancellationThreshold <= 0
                    error('cosSimExpTens:badCancellationThreshold', ...
                          '''cancellationThreshold'' must be a positive scalar.');
                end
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
            case 'truncationsigmas'
                truncationSigmas = varargin{i + 1};
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
            case 'kernelprecision'
                kernelPrecision = varargin{i + 1};
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
            case 'spectrum'
                spectrumOpt = varargin{i + 1};
                spectrumGiven = true;
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
            case 'precision'
                precisionOpt = varargin{i + 1};
                precisionGiven = true;
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
            case 'dedup'
                dedupOpt = varargin{i + 1};
                dedupGiven = true;
                keepMask(i)     = false;
                keepMask(i + 1) = false;
                i = i + 2;
                continue;
        end
    end
    i = i + 1;
end
varargin = varargin(keepMask);

nArgs = numel(varargin);

% Determine whether we will dispatch to batched-raw (the only mode
% that accepts 'spectrum', 'precision', and 'dedup'). Reject these
% kwargs early in any other dispatch context so the user gets a
% clear error rather than silent ignore.
%
% Batched-raw fires when nArgs == 9 AND at least one of P1, P2 is a
% genuine 2-D matrix (both dimensions > 1).  The other operand may
% be a vector of matching length, in which case it is broadcast
% against the matrix's rows.
willBatch = false;
if nArgs == 9 && isnumeric(varargin{1}) && isnumeric(varargin{3})
    isP1Mat = size(varargin{1}, 1) > 1 && size(varargin{1}, 2) > 1;
    isP2Mat = size(varargin{3}, 1) > 1 && size(varargin{3}, 2) > 1;
    willBatch = isP1Mat || isP2Mat;
end
if ~willBatch
    if spectrumGiven
        error('cosSimExpTens:spectrumNotApplicable', ...
            ['''spectrum'' is only valid in batched-raw mode (at least one ' ...
             'of P1, P2 must be a 2-D matrix with both dimensions > 1). For ' ...
             'scalar input, apply addSpectra to p and w yourself before calling.']);
    end
    if precisionGiven
        error('cosSimExpTens:precisionNotApplicable', ...
            '''precision'' is only valid in batched-raw mode.');
    end
    if dedupGiven
        error('cosSimExpTens:dedupNotApplicable', ...
            '''dedup'' is only valid in batched-raw mode.');
    end
end

% --- Windowed path: reject WindowedMaetDensity operands ---
% As of v2.2, cosSimExpTens no longer accepts WindowedMaetDensity
% operands. The windowed inner product is a magnitude-aware similarity
% (not a strict cosine similarity in [0, 1]) and is therefore not
% within cosSimExpTens's contract. Use windowedSimilarity for both
% single-offset and multi-offset windowed-similarity calls.
if nArgs == 2 && isstruct(varargin{1}) && isstruct(varargin{2}) ...
        && isfield(varargin{1}, 'tag') && isfield(varargin{2}, 'tag') ...
        && (strcmp(varargin{1}.tag, 'WindowedMaetDensity') ...
         || strcmp(varargin{2}.tag, 'WindowedMaetDensity'))
    error('cosSimExpTens:windowedNotSupported', ...
          ['cosSimExpTens does not accept WindowedMaetDensity ' ...
           'operands. Use windowedSimilarity(densQuery, densContext, ' ...
           'windowSpec, offsets) — pass a single-column offsets ' ...
           'vector for the scalar single-offset case, or a dim x M ' ...
           'matrix for the M-offset sweep.']);
end

% --- MA path: two MaetDensity structs ---
if nArgs == 2 && isstruct(varargin{1}) && isstruct(varargin{2}) ...
        && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'MaetDensity') ...
        && isfield(varargin{2}, 'tag') ...
        && strcmp(varargin{2}.tag, 'MaetDensity')
    % Kept skinny here: Möbius branch reads only cheap fields; Bulger
    % branch ensures heavy fields on demand inside localCosSimMA.
    s = localCosSimMA(varargin{1}, varargin{2}, ...
                       method, cancellationThreshold, verbose);
    return;
end

% --- MA path: raw args (10 positional, first is cell) ---
if nArgs == 10 && iscell(varargin{1})
    if ~iscell(varargin{3})
        error(['cosSimExpTens: p1 is a cell (multi-attribute) but p2 is ' ...
            'not. Both must be the same kind: either both cells (MA) ' ...
            'or both numeric vectors (SA).']);
    end
    pAttr1    = varargin{1};
    w1        = varargin{2};
    pAttr2    = varargin{3};
    w2        = varargin{4};
    sigmaVec  = varargin{5};
    rVec      = varargin{6};
    groups    = varargin{7};
    isRelVec  = varargin{8};
    isPerVec  = varargin{9};
    periodVec = varargin{10};
    % Build skinny: Möbius branch may not need heavy fields.
    dens_x = buildExpTens(pAttr1, w1, sigmaVec, rVec, groups, ...
        isRelVec, isPerVec, periodVec, 'verbose', verbose);
    dens_y = buildExpTens(pAttr2, w2, sigmaVec, rVec, groups, ...
        isRelVec, isPerVec, periodVec, 'verbose', verbose);
    s = localCosSimMA(dens_x, dens_y, method, cancellationThreshold, verbose);
    return;
end

% --- LIST path: nArgs == 2, at least one arg is a cell of density structs ---
%   Three accepted shapes:
%     cosSimExpTens({d_a_1, ..., d_a_n}, {d_b_1, ..., d_b_n})
%       Paired entry-by-entry; cell lengths must match. Returns 1-by-n cell.
%     cosSimExpTens(d_a, {d_b_1, ..., d_b_n})
%     cosSimExpTens({d_a_1, ..., d_a_n}, d_b)
%       Scalar density broadcast against the list; returns 1-by-n cell.
%   Option II shape rule: a length-1 cell returns a length-1 cell.
if nArgs == 2 && (iscell(varargin{1}) || iscell(varargin{2}))
    s = localCosSimDensityList(varargin{1}, varargin{2}, verbose);
    return;
end

% --- BATCHED-RAW path (with optional broadcast) ---
%   At least one of P1, P2 is an M-by-K matrix (both dims > 1).  If
%   the other is a vector of length K, it is broadcast against the
%   matrix's M rows; weights (if non-empty) are broadcast in lockstep.
%   Returns an M-by-1 vector of similarities.
%   The 'spectrum', 'precision', and 'dedup' kwargs (if supplied) are
%   forwarded to batchCosSimExpTens for spectral enrichment, dedup
%   precision tolerance, and dedup on/off respectively.
if willBatch
    P1 = varargin{1};
    W1 = varargin{2};
    P2 = varargin{3};
    W2 = varargin{4};

    isP1Mat = size(P1, 1) > 1 && size(P1, 2) > 1;
    isP2Mat = size(P2, 1) > 1 && size(P2, 2) > 1;

    % Force vector operands to row form (1×K) for uniform broadcast.
    if ~isP1Mat
        P1 = P1(:).';
        if ~isempty(W1), W1 = W1(:).'; end
    end
    if ~isP2Mat
        P2 = P2(:).';
        if ~isempty(W2), W2 = W2(:).'; end
    end

    M1 = size(P1, 1);
    M2 = size(P2, 1);
    if M1 == 1 && M2 > 1
        P1 = repmat(P1, M2, 1);
        if ~isempty(W1), W1 = repmat(W1, M2, 1); end
    elseif M2 == 1 && M1 > 1
        P2 = repmat(P2, M1, 1);
        if ~isempty(W2), W2 = repmat(W2, M1, 1); end
    elseif M1 ~= M2
        error('cosSimExpTens:batchedRowMismatch', ...
            ['Batched-raw P1 and P2 must either have matching row counts, ' ...
             'or one of them must be a single-row reference (vector or 1xK ' ...
             'matrix) to broadcast against the other. Got %d and %d rows.'], ...
            M1, M2);
    end

    s = localCosSimBatchedRaw(P1, W1, P2, W2, ...
        varargin{5}, varargin{6}, varargin{7}, varargin{8}, varargin{9}, ...
        verbose, ...
        spectrumGiven, spectrumOpt, ...
        precisionGiven, precisionOpt, ...
        dedupGiven, dedupOpt);
    return;
end

if nArgs == 2 && isstruct(varargin{1}) && isstruct(varargin{2}) ...
        && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'ExpTensDensity') ...
        && isfield(varargin{2}, 'tag') ...
        && strcmp(varargin{2}.tag, 'ExpTensDensity')
    % --- Precomputed structs (kept skinny for now: Möbius method doesn't
    % need per-tuple fields; Bulger branch ensures them on demand) ---
    dens_x = varargin{1};
    dens_y = varargin{2};

    % Validate that both structs share compatible parameters
    if dens_x.r ~= dens_y.r
        error('Both density structs must have the same r.');
    end
    if dens_x.isRel ~= dens_y.isRel
        error('Both density structs must have the same isRel.');
    end
    if dens_x.isPer ~= dens_y.isPer
        error('Both density structs must have the same isPer.');
    end
    if dens_x.isPer && dens_x.period ~= dens_y.period
        error('Both density structs must have the same period.');
    end
    if dens_x.sigma ~= dens_y.sigma
        error('Both density structs must have the same sigma.');
    end

elseif nArgs == 9
    % --- Raw arguments (SA): build skinny; Bulger branch ensures later ---
    if iscell(varargin{1}) || iscell(varargin{3})
        error(['cosSimExpTens: cell-form p1/p2 (multi-attribute) requires ' ...
            '10 positional arguments: pAttr1, w1, pAttr2, w2, sigmaVec, ' ...
            'rVec, groups, isRelVec, isPerVec, periodVec.']);
    end
    p1     = varargin{1};
    w1     = varargin{2};
    p2     = varargin{3};
    w2     = varargin{4};
    sigma_arg  = varargin{5};
    r_arg      = varargin{6};
    isRel_arg  = varargin{7};
    isPer_arg  = varargin{8};
    J_arg      = varargin{9};

    dens_x = buildExpTens(p1, w1, sigma_arg, r_arg, isRel_arg, isPer_arg, J_arg, ...
                          'verbose', verbose);
    dens_y = buildExpTens(p2, w2, sigma_arg, r_arg, isRel_arg, isPer_arg, J_arg, ...
                          'verbose', verbose);

else
    error(['Usage:\n' ...
        '  SA struct:    cosSimExpTens(dens_x, dens_y [, ''verbose'', tf])\n' ...
        '  SA raw args:  cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period [, ''verbose'', tf])\n' ...
        '  MA struct:    cosSimExpTens(densMA_x, densMA_y [, ''verbose'', tf])\n' ...
        '  MA raw args:  cosSimExpTens(pAttr1, w1, pAttr2, w2, sigmaVec, rVec, groups, isRelVec, isPerVec, periodVec [, ''verbose'', tf])\n' ...
        '  List mode:    cosSimExpTens({d_x_1, ...}, {d_y_1, ...}) -> cell array of values\n' ...
        '  Batched raw:  cosSimExpTens(P1, W1, P2, W2, sigma, r, isRel, isPer, period) -> vector of values\n' ...
        '                (P1, P2 are nRows-by-K matrices; rows are paired multisets).']);
end

% --- Common SA cheap-field setup (used by Möbius and Bulger branches) ---
r     = dens_x.r;
sigma = dens_x.sigma;
isRel = dens_x.isRel;
isPer = dens_x.isPer;
J     = dens_x.period;

% === Early return for degenerate case ===
if r > min(numel(dens_x.p), numel(dens_y.p))
    s = NaN;
    return;
end

% === Method dispatch (auto / bulger / mobius) ===
% Probe-based dispatcher: hard rules + analytical pre-screen decide
% most cases without probe overhead; when neither dominates, both
% paths are timed on a small subset and the faster is picked. The
% probe's extrapolated timing also drives the verbose dispatch message.
[chosen, probed, estSec, routingReason] = localSelectAndEstimateSAIP( ...
    dens_x, dens_y, method, truncationSigmas, kernelPrecision, verbose);

% Dispatch messages bypass per-call verbose; they're gated by the
% toolbox-wide showHints flag and throttled to once per top-level user
% call per unique (funcName, chosen, reason) triple (via
% +internal/dispatchScope).
internal.maybeShowDispatchMsg('cosSimExpTens', chosen, ...
    routingReason, estSec, probed);

ip_xy = NaN; ip_xx = NaN; ip_yy = NaN;  %#ok<NASGU>  initialised below
ranOrbit = false;

if strcmp(chosen, 'mobius')
    [ip_xy, ip_xx, ip_yy, worstRatio] = localCosSimSAOrbit(dens_x, dens_y);

    % Three-layer fallback guard.
    %  1. Cross-cancellation: |<X,Y>| small relative to sqrt(<X,X><Y,Y>).
    %     The Möbius estimate may be dominated by cancellation between
    %     partition terms.
    denomGeo = sqrt(max(ip_xx * ip_yy, 0));
    crossCancel = denomGeo > 0 && abs(ip_xy) < cancellationThreshold * denomGeo;
    %  2. Post-hoc sanity: non-finite, negative auto-IP (unambiguous Gram
    %     diagonal sign flip), or |cosine| > 1.
    corrupted = localOrbitIPsCorrupted(ip_xy, ip_xx, ip_yy);
    %  3. Runtime cancellation diagnostic (worst |sum|/max(|term|) across
    %     the three IPs): below 1e-10 means ~6 surviving decimal digits or
    %     fewer — borderline acceptable for cosine but past this point fall
    %     back to Bulger's method. See V22_DEV_LOG for the empirical regime.
    cancelTooSevere = worstRatio < 1e-10;

    if crossCancel || corrupted || cancelTooSevere
        chosen = 'bulger';   % fall through to the Bulger branch below
    else
        ranOrbit = true;
    end
end

if ~ranOrbit
    % Pairwise branch (also entered when 'method', 'bulger' was set,
    % and when an Möbius-then-fallback occurred). Heavy fields needed.
    dens_x = internal.ensureExpTensExpensive(dens_x);
    dens_y = internal.ensureExpTensExpensive(dens_y);

    Ux_perm  = dens_x.U_perm;
    wx_perm  = dens_x.w_perm;
    nJx      = dens_x.nJ_perm;
    Vx_comb  = dens_x.V_comb;
    wvx_comb = dens_x.wv_comb;
    nKx      = dens_x.nK;

    Uy_perm  = dens_y.U_perm;
    wy_perm  = dens_y.w_perm;
    nJy      = dens_y.nJ_perm;
    Vy_comb  = dens_y.V_comb;
    wvy_comb = dens_y.wv_comb;
    nKy      = dens_y.nK;

    % Estimated computation time for all three inner products:
    % Total pairs = nJx*nKy + nJx*nKx + nJy*nKy
    totalPairs = double(nJx)*double(nKy) + double(nJx)*double(nKx) ...
               + double(nJy)*double(nKy);
    estimateCompTime(totalPairs, r, 'cosSimExpTens', verbose);

    ip_xy = ipCore(Ux_perm, wx_perm, nJx, Vy_comb, wvy_comb, nKy);
    ip_xx = ipCore(Ux_perm, wx_perm, nJx, Vx_comb, wvx_comb, nKx);
    ip_yy = ipCore(Uy_perm, wy_perm, nJy, Vy_comb, wvy_comb, nKy);
end

s = ip_xy / sqrt(ip_xx * ip_yy);


% =====================================================================
%  NESTED HELPER FUNCTIONS
%  (r, sigma, J, isPer, isRel are in scope from the parent workspace.)
% =====================================================================

    % -----------------------------------------------------------------
    %  ipCore
    %  Core inner product between one perm-side (U, wU) and one
    %  comb-side (V, wV).
    %
    %  Two-axis routing (mirrors localSelectAndEstimateSA + the
    %  execution-axis check in evalExpTens):
    %
    %    Routing axis — abs and rel-non-periodic forms have a helper
    %      reduction (sigma_eff = sigma*sqrt(2)); rel+periodic does
    %      not yet and stays on the inline ipFull / chunked path.
    %    Execution axis — even when the helper is available, route
    %      through it only when feature kwargs (truncation, single
    %      precision) are actually requested, after resolving []
    %      against mptDefaults. Default mode runs ipFull / chunked
    %      inline, avoiding the helper's arguments-block validation
    %      and cell-array kwargs construction overhead per call.
    %
    %  This preserves the cost profile of the inline / chunked path for default-mode callers
    %  (e.g. cosSimExpTens in per-pair tight loops like windowedSimilarity)
    %  while enabling the helper's features whenever the user opts in.
    % -----------------------------------------------------------------
    function ipval = ipCore(U, wU, nJ, V, wV, nK)
        canUseHelper = ~(isRel && isPer);

        % Execution-axis decision: resolve defaults first.
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
        useDefaultKwargs = ~isfinite(truncResolved) ...
            && strcmp(precResolved, 'double');

        % Fire the kernel-evaluation hint once per session when Bulger's
        % IP path is about to run with default kwargs. Bulger's path
        % forms a kernel-matrix-of-r-tuple-pairs that benefits from the
        % same truncation / single-precision controls as the centres
        % path.
        if useDefaultKwargs
            internal.maybeShowKernelEvalHint();
        end

        if canUseHelper && ~useDefaultKwargs
            ipval = ipViaHelper(U, wU, V, wV);
            return;
        end

        % Default-mode (or rel+per) path: inline / chunked.
        bytesNeeded = (r + 2) * double(nJ) * double(nK) * 8;

        memLimit = internal.kernelChunkBytesResolved();

        if bytesNeeded <= memLimit
            ipval = ipFull(U, wU, nJ, V, wV, nK);
        else
            chunkSize = max(1, ...
                floor(memLimit / ((r + 2) * double(nJ) * 8)));

            acc = zeros(nJ, 1);
            for c = 1:chunkSize:nK
                cEnd = min(c + chunkSize - 1, nK);
                idx  = c:cEnd;
                nKc  = numel(idx);

                Dc = reshape(U, r, nJ, 1) ...
                   - reshape(V(:, idx), r, 1, nKc);

                % See note in ipFull: outer wrap is only needed when
                % computeQ does not re-wrap pairwise component
                % differences.
                if isPer && ~isRel
                    Dc = Dc - J .* floor(Dc / J + 0.5);
                end

                Qc = computeQ(Dc);

                Ec = reshape(exp(-Qc(:) / (4 * sigma^2)), nJ, nKc);
                acc = acc + Ec * wV(idx)';
            end

            ipval = wU(:)' * acc;
        end
    end

    % -----------------------------------------------------------------
    %  ipViaHelper
    %  Route the centres-IP through internal.gaussianKernelSum. The
    %  helper computes
    %     g(q) = sum_j wJ(j) * exp(-Q(c_j - x_q) / (2*sigma_eff^2))
    %  with sigma_eff = sigma * sqrt(2), so the kernel exponent becomes
    %  Q / (4*sigma^2) — exactly the centres-IP kernel.
    %
    %  The final IP is then sum_q wU(q) * g(q), i.e. a dot product
    %  with the perm-side weights. truncationSigmas and kernelPrecision
    %  are applied uniformly by the helper.
    %
    %  Supports abs (isPer any) and rel-non-periodic. The rel+periodic
    %  pairwise-wrap form is not yet supported by the helper and stays
    %  on the existing ipFull/chunked path.
    % -----------------------------------------------------------------
    function ipval = ipViaHelper(U, wU, V, wV)
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
        sigmaEff = sigma * sqrt(2);
        g = internal.gaussianKernelSum(V, wV(:), U, sigmaEff, kw{:});
        ipval = double(g(:).' * wU(:));
    end


    % -----------------------------------------------------------------
    %  ipFull
    %  Fully vectorized inner product (no chunking).
    %
    %  The computation is:
    %    ip = sum_{j,k} prod(w_j) * prod(w_k) * exp(-Q_jk / (4*s^2))
    %
    %  The quadratic form Q depends on the mode:
    %    Absolute: Q(d) = sum(d.^2)
    %    Relative (non-periodic): Q(d) = sum(d.^2) - sum(d)^2 / r
    %    Relative + periodic: Q(d) = sum_{i<j} wrap(d_i - d_j)^2 / r
    %
    %  The relative quadratic form (induced by the Riemannian metric on
    %  the quotient space R^r / R*1) projects out the mean, yielding
    %  transpositional equivalence. In the periodic case, the pairwise
    %  differences between wrapped components are themselves wrapped to
    %  [-J/2, J/2), which restores exact transposition invariance on the
    %  circle.
    %
    %  The two relative formulas are algebraically identical in the
    %  non-periodic case: sum_{i<j} (d_i - d_j)^2 == r*(sum(d^2) -
    %  sum(d)^2/r).
    %
    %  The weighted sum is computed as wU' * (E * wV), avoiding the full
    %  outer-product weight matrix.
    % -----------------------------------------------------------------
    function ipval = ipFull(U, wU, nJ, V, wV, nK)
        D = reshape(U, r, nJ, 1) - reshape(V, r, 1, nK);

        % The outer wrap is needed only when computeQ does not re-wrap
        % the pairwise component differences (i.e., for isPer and not
        % isRel: Q = sum(D.^2), which requires wrapped D components).
        % For isRel+isPer, computeQ wraps each (D(i)-D(j)) inside
        % (the pairwise-wrap form of Eq 6); that inner wrap is
        % invariant under integer-period shifts of the operands, so
        % wrapping D first is redundant. Skipping it saves ~30-45% of
        % ipFull time across K.
        if isPer && ~isRel
            D = D - J .* floor(D / J + 0.5);
        end

        Qvec = computeQ(D);

        E = reshape(exp(-Qvec(:) / (4 * sigma^2)), nJ, nK);
        ipval = wU(:)' * (E * wV(:));
    end

    % -----------------------------------------------------------------
    %  computeQ
    %  Compute the quadratic form Q from the (already-wrapped)
    %  difference array D (r x nJ x nK or r x nJ x nKc).
    %
    %  When isPer && isRel, pairwise differences between components of
    %  D are wrapped to [-J/2, J/2) before squaring. This restores
    %  exact transposition invariance on the circle, which is otherwise
    %  broken by component-wise wrapping. The two formulas are
    %  algebraically identical in the non-periodic case:
    %     sum_{i<j} (d_i - d_j)^2 == r * (sum(d^2) - sum(d)^2/r).
    % -----------------------------------------------------------------
    function Qvec = computeQ(D)
        if isRel
            if isPer
                Qvec = zeros(1, size(D, 2), size(D, 3));
                for i = 1:r
                    for j = i+1:r
                        delta = D(i,:,:) - D(j,:,:);
                        delta = delta - J .* floor(delta / J + 0.5);
                        Qvec = Qvec + delta.^2;
                    end
                end
                Qvec = Qvec / r;
            else
                Qvec = sum(D.^2, 1) - sum(D, 1).^2 / r;
            end
        else
            Qvec = sum(D.^2, 1);
        end
    end

end

% =========================================================================
%  SA Möbius dispatch helpers (method='auto'|'bulger'|'mobius')
% =========================================================================

function chosen = localSelectSAMethod(r, n_max, isRel, isPer, ...
                                       sigmaOverP, userMethod, n_min, ...
                                       verbose)
%LOCALSELECTSAMETHOD  Choose the inner-product method for SA cosSimExpTens.
%
%   Routing rules (in order):
%     1. userMethod ~= 'auto' overrides everything.
%     2. r <= 1: the Möbius method is undefined for r < 2; Bulger's
%        method is trivially fast.
%     3. r == 2 and n_max <= 8: Bulger's method dominates because the
%        Möbius method's overhead (4 orbit classes, contraction dispatch)
%        exceeds the kernel matvec cost.
%     4. r > 8: shipped orbit tables stop at r=8 (build cost warned).
%     5. K-vs-r precision guard: the Möbius method's alternating partition
%        sum can suffer catastrophic cancellation when n_min is too close
%        to r. Margin is 2 (i.e., n_min - r >= 2 required).
%     6. Periodic-relative beyond sigma/period > 0.03: the Möbius method
%        computes the JMM Eq. 3.4 integral form; Bulger's method computes
%        the single-nearest-image-wrap form. They diverge in this regime.
%        For backward compatibility the toolbox treats Bulger's
%        pairwise-wrap form as canonical; warn and fall back unless the
%        user explicitly asked for 'mobius'.

    if ~strcmp(userMethod, 'auto')
        chosen = userMethod;
        return;
    end
    if r <= 1
        chosen = 'bulger';
        return;
    end
    if r == 2 && n_max <= 8
        chosen = 'bulger';
        return;
    end
    if r > 8   % _ORBIT_R_MAX_SHIPPED
        chosen = 'bulger';
        return;
    end
    if n_min - r < 2   % _ORBIT_K_MINUS_R_MIN
        chosen = 'bulger';
        return;
    end
    if isRel && isPer && sigmaOverP > 0.03   % _ORBIT_SIGMA_OVER_P_THRESHOLD
        if verbose
            warning('cosSimExpTens:mobiusSigmaOverPFallback', ...
                    ['sigma/period = %.3f exceeds the Möbius-method ' ...
                     'threshold (0.03) for periodic-relative mode; ' ...
                     'falling back to Bulger''s method (the pairwise-' ...
                     'wrap form). Pass ''method'', ''bulger'' explicitly ' ...
                     'to silence this warning.'], sigmaOverP);
        end
        chosen = 'bulger';
        return;
    end
    chosen = 'mobius';
end


% =========================================================================
%  SA cos-sim probe-based dispatcher
%
%  Parallels evalExpTens's localSelectAndEstimateSA. Hard rules decide
%  first (correctness / feasibility); analytical pre-screen catches
%  clear-winner cases without paying probe overhead; otherwise time
%  both paths on a small subset of each density and pick the faster.
%
%  Extrapolation. Pairwise IP cost scales as
%  falling_factorial(K_x, r) * falling_factorial(K_y, r) (ordered
%  r-tuple enumeration on each side). Orbit IP cost scales as
%  B_r * K_x * K_y (kernel matrix construction + per-partition
%  tensor contraction). The probe uses K_probe = min(K_x, K_y, 12) events from
%  each side and extrapolates by the appropriate factor.
% =========================================================================

function [chosen, probed, estSec, routingReason] = localSelectAndEstimateSAIP( ...
        dens_x, dens_y, method, truncationSigmas, kernelPrecision, verbose)
%LOCALSELECTANDESTIMATESAIP  Probe-based dispatcher for SA cosSimExpTens.
%
%   Returns (chosen, probed, estSec, routingReason). chosen is 'mobius'
%   or 'bulger'; probed is true iff both paths were actually timed; estSec
%   is the empirical extrapolated estimate when probed, 0 otherwise.
%   routingReason is a short string the caller uses to print a verbose
%   dispatch message in the no-probe cases (and is set to 'probe' when
%   probed is true).

    PRESCREEN_IP_DOMINANCE = 3.0;
    BELL_NUMBERS = struct('r2', 2, 'r3', 5, 'r4', 15, 'r5', 52, ...
                          'r6', 203, 'r7', 877, 'r8', 4140);

    r       = double(dens_x.r);
    K_x     = double(numel(dens_x.p));
    K_y     = double(numel(dens_y.p));
    n_min   = min(K_x, K_y);
    isRel   = logical(dens_x.isRel);
    isPer   = logical(dens_x.isPer);
    sigma   = double(dens_x.sigma);
    period  = double(dens_x.period);
    if isPer && period > 0
        sigmaOverP = sigma / period;
    else
        sigmaOverP = 0;
    end
    routingReason = '';

    % ---- Hard rules ----
    if ~strcmp(method, 'auto')
        chosen = method;
        probed = false;
        estSec = 0;
        routingReason = 'user override';
        return;
    end
    if r <= 1
        chosen = 'bulger';
        probed = false;
        estSec = 0;
        routingReason = sprintf('r = %d', r);
        return;
    end
    if r > 8   % _ORBIT_R_MAX_SHIPPED
        chosen = 'bulger';
        probed = false;
        estSec = 0;
        routingReason = sprintf('r = %d > 8 (Möbius infeasible)', r);
        return;
    end
    if (n_min - r) < 2   % _ORBIT_K_MINUS_R_MIN
        chosen = 'bulger';
        probed = false;
        estSec = 0;
        routingReason = sprintf('min(K_x, K_y) - r = %d < 2', n_min - r);
        return;
    end
    if isRel && isPer && sigmaOverP > 0.03   % _ORBIT_SIGMA_OVER_P_THRESHOLD
        if verbose
            warning('cosSimExpTens:mobiusSigmaOverPFallback', ...
                    ['sigma/period = %.3f exceeds the Möbius-method ' ...
                     'threshold (0.03) for periodic-relative mode; ' ...
                     'falling back to Bulger''s method (the pairwise-' ...
                     'wrap form). Pass ''method'', ''bulger'' explicitly ' ...
                     'to silence this warning.'], sigmaOverP);
        end
        chosen = 'bulger';
        probed = false;
        estSec = 0;
        routingReason = 'sigma/period > 0.03 (rel-per Möbius fallback)';
        return;
    end

    % ---- Analytical cost models ----
    pairwiseFull = localFallingFactorial(K_x, r) ...
                 * localFallingFactorial(K_y, r);
    B_r          = BELL_NUMBERS.(sprintf('r%d', r));
    orbitFull    = B_r * K_x * K_y;

    % ---- Analytical pre-screen ----
    if orbitFull * PRESCREEN_IP_DOMINANCE < pairwiseFull
        chosen = 'mobius';
        probed = false;
        estSec = 0;
        routingReason = 'cost pre-screen';
        return;
    end
    if pairwiseFull * PRESCREEN_IP_DOMINANCE < orbitFull
        chosen = 'bulger';
        probed = false;
        estSec = 0;
        routingReason = 'cost pre-screen';
        return;
    end

    % ---- Probe both paths on a subset ----
    K_probe = min([K_x, K_y, 12]);
    % K_probe - r >= 2 is guaranteed by the precision rule above.

    tPairwise = localProbeIPPath(dens_x, dens_y, K_probe, 'bulger', ...
                                  truncationSigmas, kernelPrecision);
    tOrbit    = localProbeIPPath(dens_x, dens_y, K_probe, 'mobius', ...
                                  truncationSigmas, kernelPrecision);

    % ---- Extrapolate to full workload ----
    pairwiseProbe = localFallingFactorial(K_probe, r) ^ 2;
    if pairwiseProbe > 0
        pairwiseFactor = pairwiseFull / pairwiseProbe;
    else
        pairwiseFactor = 1;
    end
    orbitProbe = K_probe ^ 2;
    if orbitProbe > 0
        orbitFactor = (K_x * K_y) / orbitProbe;
    else
        orbitFactor = 1;
    end

    tPairwiseEst = tPairwise * pairwiseFactor;
    tOrbitEst    = tOrbit * orbitFactor;

    if tPairwiseEst <= tOrbitEst
        chosen = 'bulger';
        estSec = tPairwiseEst;
    else
        chosen = 'mobius';
        estSec = tOrbitEst;
    end
    probed = true;
    routingReason = 'probe';   % caller formats as 'estimated X s'
end


function ff = localFallingFactorial(n, k)
%LOCALFALLINGFACTORIAL  n * (n-1) * ... * (n-k+1); 0 if any factor <= 0.
    if n < k
        ff = 0;
        return;
    end
    ff = 1;
    for i = 0:(k - 1)
        ff = ff * (n - i);
    end
end


function s = localCosSimFormatTime(t)
%LOCALCOSSIMFORMATTIME  Short human-readable duration string.
%
%   Duplicated from evalExpTens.m so cosSimExpTens has no cross-file
%   dependency. Candidate for promotion to +internal/formatTime.m in
%   a future cleanup.
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


function t = localProbeIPPath(dens_x, dens_y, K_probe, path, ...
                               truncationSigmas, kernelPrecision)
%LOCALPROBEIPPATH  Time one cosSimExpTens IP path on a subset.
%
%   Runs the work twice: a warmup pass (discarded) to stabilise CPU
%   caches and one-shot table loads, then a timed pass. Without the
%   warmup, the path that ran most recently on the full workload
%   comes into the probe with hot caches and gets unfairly favoured.

    subX = buildExpTens(dens_x.p(1:K_probe), dens_x.w(1:K_probe), ...
        dens_x.sigma, dens_x.r, dens_x.isRel, dens_x.isPer, ...
        dens_x.period, 'verbose', false);
    subY = buildExpTens(dens_y.p(1:K_probe), dens_y.w(1:K_probe), ...
        dens_y.sigma, dens_y.r, dens_y.isRel, dens_y.isPer, ...
        dens_y.period, 'verbose', false);

    if strcmp(path, 'mobius')
        % Warmup pass (discarded).
        [~, ~, ~, ~] = localCosSimSAOrbit(subX, subY);
        % Timed pass.
        tStart = tic;
        [~, ~, ~, ~] = localCosSimSAOrbit(subX, subY);
        t = toc(tStart);
    else
        subX = internal.ensureExpTensExpensive(subX);
        subY = internal.ensureExpTensExpensive(subY);
        % Warmup pass (discarded).
        localProbePairwiseIP(subX, subY, truncationSigmas, kernelPrecision);
        % Timed pass.
        tStart = tic;
        localProbePairwiseIP(subX, subY, truncationSigmas, kernelPrecision);
        t = toc(tStart);
    end
end


function ip_xy = localProbePairwiseIP(dens_x, dens_y, ...
                                       truncationSigmas, kernelPrecision)
%LOCALPROBEPAIRWISEIP  Minimal IP cost stand-in for Bulger's method (probe).
%
%   Computes <T_x, T_y> via the ipFull-equivalent kernel matvec.
%   The full Bulger's method computes three IPs but their per-call costs
%   scale the same way, so timing one gives a faithful relative
%   ordering against the Möbius probe.

    r       = double(dens_x.r);
    isPer   = logical(dens_x.isPer);
    J       = double(dens_x.period);
    sigma   = double(dens_x.sigma);
    U       = dens_x.U_perm;
    wU      = dens_x.w_perm;
    nJ      = dens_x.nJ_perm;
    V       = dens_y.V_comb;
    wV      = dens_y.wv_comb;
    nK      = dens_y.nK;

    D = reshape(U, r, nJ, 1) - reshape(V, r, 1, nK);
    % Outer wrap only needed for abs+per. For rel+per the pairwise
    % wrap below subsumes it (Eq 6); this matches the form
    % ipFull / cos-sim Bulger actually use, so probe timings remain
    % representative of the dispatched method's cost.
    if isPer && ~dens_x.isRel
        D = D - J .* floor(D / J + 0.5);
    end
    if dens_x.isRel
        if isPer
            Q = zeros(nJ, nK);
            for i = 1:r
                for j = i+1:r
                    delta = reshape(D(i, :, :) - D(j, :, :), nJ, nK);
                    delta = delta - J .* floor(delta / J + 0.5);
                    Q = Q + delta.^2;
                end
            end
            Q = Q / r;
        else
            Q = reshape(sum(D .^ 2, 1), nJ, nK) ...
              - reshape(sum(D, 1) .^ 2, nJ, nK) / r;
        end
    else
        Q = reshape(sum(D .^ 2, 1), nJ, nK);
    end
    if ~isempty(truncationSigmas) && isfinite(truncationSigmas)
        Q(Q > (truncationSigmas * 2 * sigma) ^ 2) = Inf;
    end
    E = exp(-Q / (4 * sigma ^ 2));
    if ~isempty(kernelPrecision) && strcmp(kernelPrecision, 'single')
        E = single(E);
    end
    ip_xy = wU(:).' * (E * wV(:));
end


function [ip_xy, ip_xx, ip_yy, worstRatio] = localCosSimSAOrbit(dens_x, dens_y)
%LOCALCOSSIMSAORBIT  Three SA inner products via the Möbius method.
%
%   Returns ip_xy = <T_X, T_Y>, ip_xx = <T_X, T_X>, ip_yy = <T_Y, T_Y>,
%   and worstRatio = the minimum cancellation ratio across the three
%   alternating sums. A worstRatio near 1 indicates negligible
%   cancellation; values << 1 indicate digits of precision lost.

    sigma  = dens_x.sigma;
    r      = dens_x.r;
    isRel  = dens_x.isRel;
    isPer  = dens_x.isPer;
    period = dens_x.period;
    p_x = dens_x.p; w_x = dens_x.w;
    p_y = dens_y.p; w_y = dens_y.w;

    if isRel
        [ip_xy, r_xy] = mobius.orbitInnerRelSA(p_x, w_x, p_y, w_y, sigma, r, isPer, period);
        [ip_xx, r_xx] = mobius.orbitInnerRelSA(p_x, w_x, p_x, w_x, sigma, r, isPer, period);
        [ip_yy, r_yy] = mobius.orbitInnerRelSA(p_y, w_y, p_y, w_y, sigma, r, isPer, period);
    else
        [ip_xy, r_xy] = mobius.orbitInnerAbsSA(p_x, w_x, p_y, w_y, sigma, r, isPer, period);
        [ip_xx, r_xx] = mobius.orbitInnerAbsSA(p_x, w_x, p_x, w_x, sigma, r, isPer, period);
        [ip_yy, r_yy] = mobius.orbitInnerAbsSA(p_y, w_y, p_y, w_y, sigma, r, isPer, period);
    end
    worstRatio = min([r_xy, r_xx, r_yy]);
end


function corrupted = localOrbitIPsCorrupted(ip_xy, ip_xx, ip_yy)
%LOCALORBITIPSCORRUPTED  Cheap post-hoc sanity check on Möbius-method IPs.
%
%   Triggers on:
%     - non-finite IP (NaN or Inf in any of the three),
%     - negative auto-IP (Gram diagonal must be >= 0; sign flip is
%       unambiguous corruption),
%     - cosine magnitude > 1 + 1e-6 (impossible for a genuine cosine).
%
%   Catches the catastrophic-overflow regime (sigma -> 0 with low K).
%   Does NOT catch the quieter sharp-Gaussian regime where IPs are
%   finite-looking but ~1e-4 to 1e-2 wrong; the cancellation-ratio
%   guard handles that.

    corrupted = false;
    if ~all(isfinite([ip_xy, ip_xx, ip_yy]))
        corrupted = true;
        return;
    end
    if ip_xx < 0 || ip_yy < 0
        corrupted = true;
        return;
    end
    denom = sqrt(ip_xx * ip_yy);
    if denom > 0 && abs(ip_xy) > 1.000001 * denom
        corrupted = true;
    end
end


% =========================================================================
%  localCosSimMA — multi-attribute (MAET) cosine similarity
% =========================================================================

function s = localCosSimMA(dens_x, dens_y, method, cancellationThreshold, verbose)
%LOCALCOSSIMMA  Cosine similarity between two MaetDensities.
%
%   The inner product factors as an elementwise product of per-attribute
%   kernels (Section 2.7 of the MAET specification); no numerical
%   integration is required for Bulger's method.
%
%   Dispatches between Bulger's method and a per-attribute Möbius
%   method based on method ('auto' / 'bulger' / 'mobius') and a
%   simple r-based heuristic. Three-layer guard mirrors the SA
%   dispatcher (cross-cancellation, corruption, non-finite fallback).
%
%   Both densities must share the full parameter structure: number of
%   attributes, group assignment, per-attribute r, and per-group sigma,
%   isRel, isPer, period. Weights and event/slot counts may differ.

    % --- Structural compatibility (cheap fields only) ---
    if dens_x.nAttrs ~= dens_y.nAttrs
        error('cosSimExpTens:nAttrsMismatch', ...
            'Both MaetDensities must have the same nAttrs.');
    end
    if ~isequal(dens_x.groupOfAttr, dens_y.groupOfAttr)
        error('cosSimExpTens:groupsMismatch', ...
            'Both MaetDensities must have the same groupOfAttr.');
    end
    if ~isequal(dens_x.r, dens_y.r)
        error('cosSimExpTens:rMismatch', ...
            'Both MaetDensities must have the same r (per attribute).');
    end
    if ~isequal(dens_x.sigma, dens_y.sigma)
        error('cosSimExpTens:sigmaMismatch', ...
            'Both MaetDensities must have the same sigma (per group).');
    end
    if ~isequal(logical(dens_x.isRel), logical(dens_y.isRel))
        error('cosSimExpTens:isRelMismatch', ...
            'Both MaetDensities must have the same isRel (per group).');
    end
    if ~isequal(logical(dens_x.isPer), logical(dens_y.isPer))
        error('cosSimExpTens:isPerMismatch', ...
            'Both MaetDensities must have the same isPer (per group).');
    end
    perMask = logical(dens_x.isPer);
    if any(dens_x.period(perMask) ~= dens_y.period(perMask))
        error('cosSimExpTens:periodMismatch', ...
            'Both MaetDensities must have the same period for periodic groups.');
    end

    % --- Unpack shared parameters (scope for nested helpers) ---
    A        = dens_x.nAttrs;
    groupOf  = dens_x.groupOfAttr;
    rVec     = dens_x.r;
    sigmaG   = dens_x.sigma;
    isRelG   = logical(dens_x.isRel);
    isPerG   = logical(dens_x.isPer);
    periodG  = dens_x.period;

    % --- Method dispatch ---
    chosen = localSelectMAInnerProductMethod( ...
        rVec, isRelG, sigmaG, isPerG, periodG, method, verbose);

    ip_xy = NaN; ip_xx = NaN; ip_yy = NaN;  %#ok<NASGU>  initialised below
    ranOrbit = false;

    if strcmp(chosen, 'mobius')
        [ip_xy, ip_xx, ip_yy] = localCosSimMAOrbit(dens_x, dens_y);

        % Three-layer fallback guard (mirrors SA path).
        denomGeo = sqrt(max(ip_xx * ip_yy, 0));
        crossCancel = denomGeo > 0 ...
                    && abs(ip_xy) < cancellationThreshold * denomGeo;
        corrupted = localOrbitIPsCorrupted(ip_xy, ip_xx, ip_yy);

        if crossCancel || corrupted
            chosen = 'bulger';
        else
            ranOrbit = true;
        end
    end

    if ~ranOrbit
        % Pairwise branch. Heavy fields needed.
        dens_x = internal.ensureExpTensExpensive(dens_x);
        dens_y = internal.ensureExpTensExpensive(dens_y);

        Ux_perm  = dens_x.U_perm;
        wx_perm  = dens_x.wJ;
        nJx      = dens_x.nJ;
        Vx_comb  = dens_x.V_comb;
        wvx_comb = dens_x.wv_comb;
        nKx      = dens_x.nK;

        Uy_perm  = dens_y.U_perm;
        wy_perm  = dens_y.wJ;
        nJy      = dens_y.nJ;
        Vy_comb  = dens_y.V_comb;
        wvy_comb = dens_y.wv_comb;
        nKy      = dens_y.nK;

        % --- Three inner products ---
        totalPairs = double(nJx)*double(nKy) + double(nJx)*double(nKx) ...
                   + double(nJy)*double(nKy);
        maxR = max(rVec);
        estimateCompTime(totalPairs, maxR, 'cosSimExpTens (MAET)', verbose);

        ip_xy = ipCoreMA(Ux_perm, wx_perm, nJx, Vy_comb, wvy_comb, nKy);
        ip_xx = ipCoreMA(Ux_perm, wx_perm, nJx, Vx_comb, wvx_comb, nKx);
        ip_yy = ipCoreMA(Uy_perm, wy_perm, nJy, Vy_comb, wvy_comb, nKy);
    end

    denom = sqrt(ip_xx * ip_yy);
    if denom == 0
        s = NaN;
    else
        s = ip_xy / denom;
    end

    % =====================================================================
    %  Nested helpers (rVec, sigmaG, isRelG, isPerG, periodG, groupOf, A
    %  are in scope from the parent).
    % =====================================================================

    function ipval = ipCoreMA(U_cell, wU, nJ, V_cell, wV, nK)
        % Memory-aware chunking along the comb-side (nK) dimension.
        maxRa = double(max(rVec));
        bytesNeeded = (maxRa + 2) * double(nJ) * double(nK) * 8;

        memLimit = internal.kernelChunkBytesResolved();

        if bytesNeeded <= memLimit
            ipval = ipFullMA(U_cell, wU, nJ, V_cell, wV, nK);
        else
            chunkSize = max(1, floor(memLimit / ((maxRa + 2) * double(nJ) * 8)));
            acc = zeros(nJ, 1);
            for c = 1:chunkSize:nK
                cEnd = min(c + chunkSize - 1, nK);
                idx  = c:cEnd;
                nKc  = numel(idx);

                V_chunk = cell(1, A);
                for a = 1:A
                    V_chunk{a} = V_cell{a}(:, idx);
                end

                logK = maLogKernel(U_cell, V_chunk, nJ, nKc);
                Ec = exp(logK);
                acc = acc + Ec * wV(idx).';
            end
            ipval = wU(:).' * acc;
        end
    end

    function ipval = ipFullMA(U_cell, wU, nJ, V_cell, wV, nK)
        logK = maLogKernel(U_cell, V_cell, nJ, nK);
        E = exp(logK);                    % nJ x nK
        ipval = wU(:).' * (E * wV(:));
    end

    function logK = maLogKernel(U_cell, V_cell, nJ, nK)
        % Accumulate sum_a -Q_a / (4 sigma_g^2) over attributes.
        logK = zeros(nJ, nK);
        for a = 1:A
            g = groupOf(a);
            r_a = rVec(a);
            D = reshape(U_cell{a}, r_a, nJ, 1) ...
              - reshape(V_cell{a}, r_a, 1, nK);

            % The outer wrap is only needed when computeQaMA does not
            % re-wrap the pairwise component differences (i.e., for
            % isPer and not isRel: Qa = sum(D.^2), which requires
            % wrapped D components). For rel+per, computeQaMA wraps
            % each pairwise (D(i)-D(j)) inside (Eq 6 form); that
            % inner wrap is invariant under integer-period shifts, so
            % wrapping D first is redundant.
            if isPerG(g) && ~isRelG(g)
                P_g = periodG(g);
                D = D - P_g .* floor(D / P_g + 0.5);
            end

            Qa = computeQaMA(D, g, r_a);
            logK = logK - reshape(Qa, nJ, nK) / (4 * sigmaG(g)^2);
        end
    end

    function Qa = computeQaMA(D, g, r_a)
        % Per-attribute quadratic form. Matches the SA computeQ logic:
        %   - is_rel && is_per: pairwise-differences formula (wraps
        %     each pairwise delta to [-P/2, P/2), restores exact
        %     transposition invariance on the circle).
        %   - is_rel && ~is_per: sum(d.^2) - sum(d)^2 / r_a.
        %   - ~is_rel:           sum(d.^2).
        if isRelG(g)
            if isPerG(g)
                sz = size(D);
                if numel(sz) < 3, sz = [sz, 1]; end
                Qa = zeros(1, sz(2), sz(3));
                P_g = periodG(g);
                for i = 1:r_a
                    for j = i+1:r_a
                        delta = D(i, :, :) - D(j, :, :);
                        delta = delta - P_g .* floor(delta / P_g + 0.5);
                        Qa = Qa + delta.^2;
                    end
                end
                Qa = Qa / r_a;
            else
                Qa = sum(D.^2, 1) - sum(D, 1).^2 / r_a;
            end
        else
            Qa = sum(D.^2, 1);
        end
    end

end


% =========================================================================
%  MA Möbius dispatch helpers (method='auto'|'bulger'|'mobius')
% =========================================================================

function chosen = localSelectMAInnerProductMethod(rVec, isRelG, sigmaG, ...
                                                    isPerG, periodG, ...
                                                    userMethod, verbose)
%LOCALSELECTMAINNERPRODUCTMETHOD  Choose the IP method for MA cosSimExpTens.
%
%   Simple heuristic (no cost model; benchmark-driven recalibration
%   pending at Commit 7):
%     1. userMethod ~= 'auto' overrides everything.
%     2. r_max <= 1 -> Bulger (Möbius method undefined).
%     3. r_max > 8 (above _ORBIT_R_MAX_SHIPPED) -> Bulger (no shipped
%        orbit table; user-build cost-preview warning otherwise).
%     4. Periodic-relative beyond sigma/period > 0.03 anywhere -> warn,
%        Bulger. Same convention guard as the SA dispatcher.
%     5. Any rel group at all -> Bulger's method. The Möbius relative-mode
%        evaluator for MA is un-vectorised (per-event-pair loop);
%        Bulger dominates in typical regimes. Users wanting the Möbius
%        relative-mode evaluator opt in explicitly.
%     6. r_max < 3 -> Bulger (the Möbius method at r=2 carries
%        |Omega_2|=4 overhead with the same K^2 asymptotic as Bulger).
%     7. Otherwise -> mobius.
%
%   has_nan is NOT a fallback: the MA Möbius-method wrapper handles
%   ragged K_{a,n} natively. Per-event-pair classification: events
%   with K_eff - r >= 2 (the Möbius-method precision margin) flow
%   through the vectorised batched Möbius evaluator; pairs involving
%   any K_eff - r < 2 event flow through direct r-tuple enumeration
%   (no Möbius alternating sum, hence no cancellation). See
%   mobius.maPerAttrInnerMatrix.

    if ~strcmp(userMethod, 'auto')
        chosen = userMethod;
        return;
    end

    r_max = max(rVec);
    if r_max <= 1
        chosen = 'bulger';
        return;
    end
    if r_max > 8   % _ORBIT_R_MAX_SHIPPED
        chosen = 'bulger';
        return;
    end

    % sigma/period guard (rel + per groups only).
    sigmaOverP_max = 0;
    for g = 1:numel(sigmaG)
        if isRelG(g) && isPerG(g) && periodG(g) > 0
            ratio = sigmaG(g) / periodG(g);
            if ratio > sigmaOverP_max
                sigmaOverP_max = ratio;
            end
        end
    end
    if sigmaOverP_max > 0.03   % _ORBIT_SIGMA_OVER_P_THRESHOLD
        if verbose
            warning('cosSimExpTens:mobiusSigmaOverPFallback', ...
                    ['Maximum sigma/period = %.3f across periodic-relative ' ...
                     'groups exceeds the Möbius-method threshold (0.03); ' ...
                     'falling back to Bulger''s method (the pairwise-wrap ' ...
                     'form). Pass ''method'', ''bulger'' explicitly to ' ...
                     'silence this warning.'], sigmaOverP_max);
        end
        chosen = 'bulger';
        return;
    end

    % Any rel group -> Bulger's method (the Möbius relative-mode
    % evaluator is not vectorised).
    if any(isRelG)
        chosen = 'bulger';
        return;
    end

    if r_max < 3
        chosen = 'bulger';
        return;
    end

    chosen = 'mobius';
end


function [ip_xy, ip_xx, ip_yy] = localCosSimMAOrbit(dens_x, dens_y)
%LOCALCOSSIMMAORBIT  Three MA inner products via per-attribute Möbius method.
%
%   Computes, for each attribute a, an (N_x, N_y) per-attribute inner
%   product matrix I_xy^{(a)}[n_x, n_y] = <T_X^{(a)}_{n_x}, T_Y^{(a)}_{n_y}>
%   (similarly for I_xx, I_yy). The full IP factors as
%       <T_X, T_Y> = sum_{n_x, n_y} prod_a I_xy^{(a)}[n_x, n_y]
%   so we element-wise multiply per-attribute matrices across attributes
%   then sum. NaN-padded events are handled via zero-weight padding in
%   mobius.maPerAttrInnerMatrix.

    A = dens_x.nAttrs;
    N_x = dens_x.N;
    N_y = dens_y.N;

    P_xy = ones(N_x, N_y);
    P_xx = ones(N_x, N_x);
    P_yy = ones(N_y, N_y);

    for a = 1:A
        g       = dens_x.groupOfAttr(a);
        r_a     = dens_x.r(a);
        sigma_g = dens_x.sigma(g);
        isRel_g = dens_x.isRel(g);
        isPer_g = dens_x.isPer(g);
        period_g = dens_x.period(g);

        Px = dens_x.pAttr{a};   Wx = dens_x.w{a};
        Py = dens_y.pAttr{a};   Wy = dens_y.w{a};

        I_xy = mobius.maPerAttrInnerMatrix(Px, Wx, Py, Wy, ...
            sigma_g, r_a, isRel_g, isPer_g, period_g);
        I_xx = mobius.maPerAttrInnerMatrix(Px, Wx, Px, Wx, ...
            sigma_g, r_a, isRel_g, isPer_g, period_g);
        I_yy = mobius.maPerAttrInnerMatrix(Py, Wy, Py, Wy, ...
            sigma_g, r_a, isRel_g, isPer_g, period_g);

        P_xy = P_xy .* I_xy;
        P_xx = P_xx .* I_xx;
        P_yy = P_yy .* I_yy;
    end

    ip_xy = sum(P_xy(:));
    ip_xx = sum(P_xx(:));
    ip_yy = sum(P_yy(:));
end




% =====================================================================
%  Unified dispatch helpers: density-list and batched-raw modes.
% =====================================================================

function sCell = localCosSimDensityList(a, b, verbose)
%LOCALCOSSIMDENSITYLIST List-mode density-struct cosine similarities.
%
%   Three accepted shapes:
%     (cell, cell)   — pairwise; lengths must match. Returns 1-by-n.
%     (cell, struct) — broadcast struct against the cell. Returns 1-by-n.
%     (struct, cell) — broadcast struct against the cell. Returns 1-by-n.
%
%   Each pair dispatches recursively to cosSimExpTens, which selects the
%   appropriate scalar form (MA or SA) based on the entries'
%   tags. Mixed-kind pairs are not prevented at this level; compatibility
%   is checked downstream. `WindowedMaetDensity` entries are rejected
%   at the top of cosSimExpTens (use windowedSimilarity instead).

    aIsCell = iscell(a);
    bIsCell = iscell(b);

    if aIsCell && bIsCell
        if numel(a) ~= numel(b)
            error('cosSimExpTens:listLengthMismatch', ...
                ['cosSimExpTens (list mode): the two cell arrays must have ' ...
                 'matching length, or one operand must be a single density ' ...
                 'struct to broadcast. Got %d and %d.'], numel(a), numel(b));
        end
        n = numel(a);
        sCell = cell(1, n);
        for i = 1:n
            if ~isstruct(a{i}) || ~isstruct(b{i})
                error('cosSimExpTens:listNonStruct', ...
                    ['cosSimExpTens (list mode): cell entries must be ' ...
                     'density structs from buildExpTens; entry %d is not ' ...
                     'a struct.'], i);
            end
            sCell{i} = cosSimExpTens(a{i}, b{i}, 'verbose', verbose);
        end
        return;
    end

    % Mixed shape: exactly one is a cell. The other must be a density
    % struct; broadcast it against every entry of the cell.
    if aIsCell
        cellArg = a;
        scalarArg = b;
        scalarLeft = false;
    else
        cellArg = b;
        scalarArg = a;
        scalarLeft = true;
    end

    if ~isstruct(scalarArg)
        error('cosSimExpTens:listBadBroadcast', ...
            ['cosSimExpTens (list mode): when one operand is a cell of ' ...
             'density structs, the other must be a single density struct ' ...
             'to broadcast. Got a non-struct, non-cell of class %s.'], ...
            class(scalarArg));
    end

    n = numel(cellArg);
    sCell = cell(1, n);
    for i = 1:n
        if ~isstruct(cellArg{i})
            error('cosSimExpTens:listNonStruct', ...
                ['cosSimExpTens (list mode): cell entries must be density ' ...
                 'structs from buildExpTens; entry %d is not a struct.'], i);
        end
        if scalarLeft
            sCell{i} = cosSimExpTens(scalarArg, cellArg{i}, 'verbose', verbose);
        else
            sCell{i} = cosSimExpTens(cellArg{i}, scalarArg, 'verbose', verbose);
        end
    end
end


function s = localCosSimBatchedRaw(P1, W1, P2, W2, sigma, r, isRel, isPer, period, verbose, ...
    spectrumGiven, spectrumOpt, precisionGiven, precisionOpt, dedupGiven, dedupOpt)
%LOCALCOSSIMBATCHEDRAW Batched cosine similarity from paired 2-D inputs.
%
%   P1 and P2 are nRows-by-K_? matrices. Returns a length-nRows vector.
%   The implementation lives here (it was previously hosted in the
%   deprecated batchCosSimExpTens.m, which is now a thin shim that
%   forwards to this code path via the public cosSimExpTens API).
%
%   Pipeline:
%     1. Optional precision rounding.
%     2. Canonicalise each row's A-set and B-set under isPer/isRel so
%        that equivalent multisets map to the same key.
%     3. Deduplicate individual A- and B-sets; deduplicate (A, B) pairs.
%     4. Build one density struct per unique individual set.
%     5. Call cosSimExpTens once per unique (A, B) pair, mapping results
%        back to all matching rows.

    if size(P1, 1) ~= size(P2, 1)
        error('cosSimExpTens:batchedRowMismatch', ...
            ['cosSimExpTens (batched mode): P1 and P2 must have the same ' ...
             'number of rows, got %d and %d.'], size(P1, 1), size(P2, 1));
    end

    pMatA    = P1;
    pMatB    = P2;
    weightsA = W1;
    weightsB = W2;
    if spectrumGiven
        specArgs = spectrumOpt;
        if ~iscell(specArgs)
            error('''spectrum'' value must be a cell array of addSpectra arguments.');
        end
    else
        specArgs = {};
    end
    if precisionGiven
        nDec = precisionOpt;
    else
        nDec = [];
    end
    if dedupGiven && ~dedupOpt
        % The batched-raw implementation always deduplicates internally
        % (the loop below operates on unique (A, B) pairs only). Honour
        % the request with a warning; the resulting numerical output is
        % identical either way.
        warning('cosSimExpTens:dedupNoop', ...
            ['''dedup'', false has no effect: the batched-raw ' ...
             'implementation always deduplicates internally for ' ...
             'speed. Numerical results are unchanged.']);
    end

    % === Input validation ===
    nRows = size(pMatA, 1);
    if ~isempty(weightsA) && ~isequal(size(weightsA), size(pMatA))
        error('weightsA must be the same size as pMatA.');
    end
    if ~isempty(weightsB) && ~isequal(size(weightsB), size(pMatB))
        error('weightsB must be the same size as pMatB.');
    end

    useSpectra  = ~isempty(specArgs);
    useWeightsA = ~isempty(weightsA);
    useWeightsB = ~isempty(weightsB);

    % === Apply precision rounding ===
    if ~isempty(nDec)
        pMatA = round(pMatA, nDec);
        pMatB = round(pMatB, nDec);
        if useWeightsA
            weightsA = round(weightsA, nDec);
        end
        if useWeightsB
            weightsB = round(weightsB, nDec);
        end
    end

    % === Phase 1: Canonicalize and build individual-set keys ===
    % Each set is independently canonicalized under isPer/isRel so that
    % equivalent pitch sets (differing only by octave displacement or
    % transposition) map to the same key. Keys for A-sets and B-sets are
    % built separately to enable individual-set density struct caching.
    nA = size(pMatA, 2);
    nB = size(pMatB, 2);

    keyWidthA = nA * (1 + useWeightsA);
    keyWidthB = nB * (1 + useWeightsB);

    keysA = NaN(nRows, keyWidthA);
    keysB = NaN(nRows, keyWidthB);
    valid = false(nRows, 1);
    s     = NaN(nRows, 1);

    for i = 1:nRows
        pA = pMatA(i, :);
        pB = pMatB(i, :);

        maskA = ~isnan(pA);
        maskB = ~isnan(pB);
        pAv   = pA(maskA);
        pBv   = pB(maskB);

        % Check minimum p-value count
        if numel(pAv) < r || numel(pBv) < r
            continue;
        end

        % Get weights (or empty for all ones)
        if useWeightsA
            wAv = weightsA(i, maskA);
        else
            wAv = [];
        end
        if useWeightsB
            wBv = weightsB(i, maskB);
        else
            wBv = [];
        end

        % Canonicalize each set and apply joint co-transposition
        % normalization for the absolute case.
        if isRel
            % Relative: independent canonicalization
            [pAc, wAc] = internal.canonicalizeSet(pAv, wAv, isRel, isPer, period);
            [pBc, wBc] = internal.canonicalizeSet(pBv, wBv, isRel, isPer, period);
        else
            % Absolute: joint co-transposition normalization.
            % cosSimExpTens(A-c, B-c) = cosSimExpTens(A, B) because the
            % raw tuple differences cancel. Find A's canonical form and
            % apply the same shift to B.

            hasWA = ~isempty(wAv);
            hasWB = ~isempty(wBv);

            % Canonicalize A
            [pAs, siA] = sort(pAv);
            if hasWA, wAs = wAv(siA); else, wAs = []; end

            if isPer
                pAs = mod(pAs, period);
                [pAs, siA2] = sort(pAs);
                if hasWA, wAs = wAs(siA2); end
                % Cyclic canonical form — collapses all rotations
                [pAc, wAc, shift] = internal.cyclicCanonical(pAs, wAs, hasWA, period);
            else
                shift = pAs(1);
                pAc = pAs(:)' - shift;
                if hasWA, wAc = wAs(:)'; else, wAc = []; end
            end

            % Apply the same shift to B
            [pBs, siB] = sort(pBv);
            if hasWB, wBs = wBv(siB); else, wBs = []; end

            if isPer
                pBshifted = mod(pBs - shift, period);
                [pBshifted, siB2] = sort(pBshifted);
                pBc = pBshifted(:)';
                if hasWB, wBc = wBs(siB2)'; else, wBc = []; end
            else
                pBc = pBs(:)' - shift;
                if hasWB, wBc = wBs(:)'; else, wBc = []; end
            end
        end

        % Re-round after canonicalization to collapse floating-point
        % noise introduced by mod-reduction and subtraction.
        if ~isempty(nDec)
            pAc = round(pAc, nDec);
            pBc = round(pBc, nDec);
            if ~isempty(wAc), wAc = round(wAc, nDec); end
            if ~isempty(wBc), wBc = round(wBc, nDec); end
        end

        % Build NaN-padded keys
        keyA = NaN(1, keyWidthA);
        keyA(1:numel(pAc)) = pAc;
        if useWeightsA
            keyA(nA + 1 : nA + numel(wAc)) = wAc;
        end

        keyB = NaN(1, keyWidthB);
        keyB(1:numel(pBc)) = pBc;
        if useWeightsB
            keyB(nB + 1 : nB + numel(wBc)) = wBc;
        end

        keysA(i, :) = keyA;
        keysB(i, :) = keyB;
        valid(i)    = true;
    end

    % === Phase 2: Deduplicate individual sets, then pairs ===
    validIdx = find(valid);
    [uniqueKeysA, ~, mapA] = unique(keysA(validIdx, :), 'rows');
    [uniqueKeysB, ~, mapB] = unique(keysB(validIdx, :), 'rows');
    nUniqueA = size(uniqueKeysA, 1);
    nUniqueB = size(uniqueKeysB, 1);

    pairKeys = [mapA, mapB];
    [uniquePairs, ~, pairMap] = unique(pairKeys, 'rows');
    nUniquePairs = size(uniquePairs, 1);

    if verbose
        fprintf(['cosSimExpTens: %d rows, %d valid, ' ...
                 '%d unique A-sets, %d unique B-sets, ' ...
                 '%d unique pairs.\n'], ...
            nRows, numel(validIdx), nUniqueA, nUniqueB, nUniquePairs);
        if isRel
            if isPer
                fprintf(['  Canonicalization: A-sets and B-sets independently ' ...
                         'normalized for transposition and octave equivalence.\n']);
            else
                fprintf(['  Canonicalization: A-sets and B-sets independently ' ...
                         'normalized for transposition.\n']);
            end
        else
            if isPer
                fprintf(['  Canonicalization: joint co-transposition with ' ...
                         'octave equivalence; B-set counts reflect position ' ...
                         'relative to A.\n']);
            else
                fprintf(['  Canonicalization: joint co-transposition; ' ...
                         'B-set counts reflect position relative to A.\n']);
            end
        end
    end

    % === Phase 3: Build density structs for unique individual sets ===
    densA = cell(nUniqueA, 1);
    for ua = 1:nUniqueA
        [pA_u, wA_u] = localExtractFromKey(uniqueKeysA(ua, :), nA, useWeightsA);
        if useSpectra
            [pA_u, wA_u] = addSpectra(pA_u, wA_u, specArgs{:});
        end
        densA{ua} = buildExpTens(pA_u, wA_u, sigma, r, isRel, isPer, period, ...
                                 'verbose', false);
    end

    densB = cell(nUniqueB, 1);
    for ub = 1:nUniqueB
        [pB_u, wB_u] = localExtractFromKey(uniqueKeysB(ub, :), nB, useWeightsB);
        if useSpectra
            [pB_u, wB_u] = addSpectra(pB_u, wB_u, specArgs{:});
        end
        densB{ub} = buildExpTens(pB_u, wB_u, sigma, r, isRel, isPer, period, ...
                                 'verbose', false);
    end

    if verbose
        fprintf('cosSimExpTens: built %d density structs (%d A + %d B).\n', ...
            nUniqueA + nUniqueB, nUniqueA, nUniqueB);
    end

    % === Phase 3.5: Up-front time estimate ===
    % Calibrate empirically (warm-up + timed sample) and extrapolate
    % to the full unique-pair count, matching the pattern used by the
    % other batched helpers (spectralEntropy, entropyExpTens,
    % templateHarmonicity, virtualPitches, tensorHarmonicity).
    % Threshold 10 s via internal.printBatchedEstimate; gated on
    % verbose for consistency with the rest of the toolbox.
    %
    % Extrapolation is over nUniquePairs (post-dedup), not nRows:
    % that's what the main loop iterates over. Demos that dedup
    % heavily (e.g. transposition sweeps) see a small estimate;
    % demos that don't (e.g. demo_genChainSpcs, every generator-step
    % producing a distinct canonical chord) see one closer to nRows.
    % Adaptive progress-print state. Defaults: silent (showProgress
    % false) and stride 1 (unused while silent). Both are overridden
    % inside the calibration block from the empirical per-pair cost
    % and the estimated total time: progress prints fire only when
    % the loop is expected to take >= 5 s, with cadence set so that
    % each print interval is also >= 5 s.
    progStride = 1;
    showProgress = false;
    if verbose && nUniquePairs >= 2
        nCal = min(5, nUniquePairs);
        sampleIdx = unique(round(linspace(1, nUniquePairs, nCal)));

        % Warm-up call to absorb one-time setup (cache populate, the
        % inner call's first-time dispatch announce, etc.).
        dA_w = densA{uniquePairs(sampleIdx(1), 1)};
        dB_w = densB{uniquePairs(sampleIdx(1), 2)};
        cosSimExpTens(dA_w, dB_w, 'verbose', false);

        % Timed calibration over the sample.
        tCalStart = tic;
        for cs = 1:numel(sampleIdx)
            dA_s = densA{uniquePairs(sampleIdx(cs), 1)};
            dB_s = densB{uniquePairs(sampleIdx(cs), 2)};
            cosSimExpTens(dA_s, dB_s, 'verbose', false);
        end
        tCalTotal = toc(tCalStart);
        tPerPair  = tCalTotal / numel(sampleIdx);
        estTotal  = tCalTotal + tPerPair * nUniquePairs;
        internal.printBatchedEstimate('cosSimExpTens', nUniquePairs, estTotal);
        progStride = internal.progressStride(tPerPair);
        showProgress = estTotal >= 5;
    end

    % === Phase 4: Compute similarity for each unique pair ===
    uniqueS = NaN(nUniquePairs, 1);
    for up = 1:nUniquePairs
        dA = densA{uniquePairs(up, 1)};
        dB = densB{uniquePairs(up, 2)};
        uniqueS(up) = cosSimExpTens(dA, dB, 'verbose', false);

        if verbose && showProgress ...
                && (mod(up, progStride) == 0 || up == nUniquePairs)
            fprintf('  %d / %d unique pairs computed.\n', up, nUniquePairs);
        end
    end

    % === Phase 5: Map results back to all valid rows ===
    s(validIdx) = uniqueS(pairMap);

    if verbose
        fprintf('cosSimExpTens: done.\n');
    end
end


function [p, w] = localExtractFromKey(key, nMax, hasWeights)
%LOCALEXTRACTFROMKEY  Extract pitch and weight vectors from a NaN-padded key.
    pPart = key(1:nMax);
    p = pPart(~isnan(pPart));
    p = p(:);
    if hasWeights
        wPart = key(nMax + 1 : end);
        w = wPart(~isnan(wPart));
        w = w(:);
    else
        w = [];
    end
end

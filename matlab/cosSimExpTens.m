function [s, densXOut, densYOut] = cosSimExpTens(varargin)
%COSSIMEXPTENS Cosine similarity of two r-ad expectation tensor densities.
%
%   s = cosSimExpTens(dens_x, dens_y):
%   s = cosSimExpTens(dens_x, dens_y, 'verbose', false):
%   Cosine similarity using precomputed density structs from buildExpTens.
%   This avoids recomputing tuple indices and weight products on each call,
%   and is the preferred calling convention when comparing a fixed reference
%   against many other sets.
%
%   [s, dens_x, dens_y] = cosSimExpTens(dens_x, dens_y, ...):
%   As above, additionally returning the two operand structs with their
%   self inner products memoised (a 'selfIP' field). A density's self
%   inner product <T, T> depends only on the density itself, so a
%   caller looping scalar calls against a fixed reference can thread
%   the returned struct through the loop and pay the reference's
%   O(N^2) self term once:
%       for m = 1:M
%           [s(m), densRef] = cosSimExpTens(densRef, densQry{m}, ...);
%       end
%   The memo is keyed on everything the value depends on beyond the
%   density's contents (inner-product route, resolved
%   truncationSigmas, and the Möbius route's per-attribute
%   closed-form-vs-grid choices), so stale reuse is structurally
%   impossible; an unrecognised key simply recomputes. The same field
%   also carries, in an optional 'nestedCentres' cell, the materialised
%   tuple-centres bundles the centres routes build per attribute
%   (INTERNAL.NESTEDCENTRESMEMOISED), so a threaded struct also skips
%   that rebuild on later calls. The extra
%   outputs are available only in this density-struct scalar form. In
%   the scalar-vs-cell (sweep) and raw sweep forms the memoisation is
%   applied internally across the sweep, so no threading is needed
%   there.
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
%   tag (single multiset or MA), with the caller's 'method',
%   'truncationSigmas', 'kernelPrecision' and 'normalize' forwarded to
%   every pair. Shape rule: a length-1 input returns a length-1
%   cell (no collapse to scalar).
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
%   s = cosSimExpTens(pAttr1, w1, pAttr2, w2, sigma, r, ...
%                     isRel, isPer, periods):
%   Raw multi-attribute mode. pAttr1 and pAttr2 are each a 1-by-A
%   cell of K_a-by-N value matrices (the same shape one would pass
%   to buildExpTens). Builds the two MaetDensity structs internally
%   and returns a scalar.
%
%   sCell = cosSimExpTens(refPAttr, refW, {pAttrA, pAttrB, ...}, qryW, ...
%                          sigma, r, isRel, isPer, periods):
%   Raw multi-attribute scalar-vs-list mode (sweep). Exactly one of
%   the two pAttr arguments is a cell-of-cells (a 1-by-M cell whose
%   entries are themselves 1-by-A pAttr cells, e.g. the matrix-form
%   output of translateAttributes); the other is a single 1-by-A pAttr
%   cell. The scalar operand is built once; the list operand is
%   built once per entry. Weights for the list side are shared
%   across every entry (a single w value, not a cell of weights).
%   Returns a 1-by-M cell of similarity scalars.
%
%   Broadcasting. If one operand is a vector of length K
%   (1-by-K, K-by-1, or 1-D) and the other is M-by-K with M > 1, the
%   vector is broadcast across the matrix's M rows, in NumPy / MATLAB
%   implicit-expansion style. The corresponding weight argument
%   (W1 or W2) is broadcast in lockstep when non-empty. This avoids
%   the explicit repmat(refPitches, M, 1) idiom for the common case
%   "compare one reference multiset against many candidates".
%
%   Multiset-argument shapes (each operand takes the same shape on both
%   sides; subscript 1 / 2 selects which operand):
%     p1, p2          — Vectors of length K_1, K_2 (may differ). single multiset raw
%                       form (single multiset, single-attribute).
%     P1, P2          — nRows-by-K matrices, both dimensions > 1.
%                       BATCHED-RAW form (rows are independent single-attribute-style
%                       multisets, processed in lockstep; returns an
%                       nRows-by-1 vector of per-row similarities).
%     pAttr1, pAttr2  — 1-by-A cells of K_a-by-N matrices. MA raw form
%                       (multi-attribute; per-attribute centre rows).
%   Lowercase p stands for "pitch or position"; uppercase P is the
%   2-D batched lift; pAttr is the multi-attribute generalisation.
%   The same convention is used in entropyExpTens and evalExpTens.
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
%   Inputs (single multiset raw calling convention):
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
%   Inputs (BATCHED-RAW calling convention):
%     P1, P2 — nRows-by-K matrices of pitch or position values (rows
%              are independent single-attribute-style multisets, paired between P1
%              and P2). At least one of P1, P2 must have both
%              dimensions > 1; the other may be a length-K vector that
%              is broadcast against the matrix's rows.
%     W1, W2 — Weights paired with P1, P2 (same shape, or [] for
%              uniform). Broadcast in lockstep with their P operand.
%     sigma, r, isRel, isPer, period — As in the single multiset raw convention
%              (shared across all rows).
%
%   Inputs (MA raw calling convention):
%     pAttr1, pAttr2  — 1-by-A cells of K_a-by-N matrices (per-attribute
%                       value rows; the same shape one would pass to
%                       buildExpTens). For the scalar-vs-list sweep
%                       form, exactly one of these is a 1-by-M cell of
%                       such cells; the other is a single pAttr cell.
%     w1, w2          — Weights paired with pAttr1, pAttr2 (see
%                       buildExpTens for the accepted shapes). Shared
%                       across every list entry in the sweep form.
%     sigmaVec        — 1-by-G per-group Gaussian widths.
%     rVec            — 1-by-A per-attribute tuple sizes.
%     isRelVec        — 1-by-A per-attribute relative flags.
%     isPerVec        — 1-by-A per-attribute periodic flags.
%     periodVec       — 1-by-A per-attribute period values (ignored where
%                       isPerVec(a) == false).
%
%   Optional name-value pair (all calling conventions):
%     'verbose' — Logical (default: true). If false, suppresses console
%                 output (time estimates, progress messages).
%     'method'  — 'auto' (default), 'bulger', 'centres', 'mobius', or
%                 'contract' (force the nested tree-contraction).
%                 Inner-product decomposition. 'auto' selects via a
%                 per-call cost model between Bulger's method (small r
%                 and small K) and the Möbius method (large r
%                 or large K). 'bulger' / 'mobius' force the named
%                 method; 'centres' enumerates every ordered r-tuple on
%                 each side without restriction (the O(K^(2r)) reference
%                 route: far slower than either at any appreciable K,
%                 but free of the alternating sum and so immune to the
%                 cancellation the Möbius route can suffer). See User
%                 Guide §4 ("Method selection").
%
%                 On a NESTED density the names select among that
%                 path's own routes, since the flat orbit entry point
%                 cannot represent a nested attribute's block-diagonal
%                 inner metric. 'bulger' is still the joint-tuple
%                 enumeration. 'contract' forces the hierarchical
%                 contraction plan, raising rather than falling back on
%                 any case it does not cover; it is rejected on a
%                 non-nested density. 'mobius' names the same plan ---
%                 the per-level orbit (Möbius) reduction is exactly what
%                 the contraction applies at every symmetric level, so
%                 for a nested density 'mobius' and 'contract' coincide.
%                 'centres' forces the materialised-centres route for
%                 every nested attribute; on a relative-periodic
%                 attribute whose declared measure is the default
%                 full-image one, that route is admissible only up to
%                 the sigma/P threshold, above which 'centres' errors
%                 (cosSimExpTens:centresUnavailable) naming the
%                 wrap = 'single-image' opt-in.
%     'truncationSigmas' — Numeric scalar or []. Override the toolbox-
%                 wide mptDefaults('truncationSigmas') setting for this
%                 call. Applies on the centres path (Bulger's method
%                 on the single multiset inner product); skips Gaussian
%                 contributions whose centre-to-query distance exceeds
%                 k*sigma (kernel floor exp(-k^2/2)). [] (default)
%                 means use the global default (factory: Inf). No
%                 effect on Möbius-method calls.
%     'kernelPrecision' — 'double', 'single', or [] for the global
%                 default. Forwarded through every input form (list,
%                 broadcast and batched-raw) and honoured on one
%                 inner-product route, as in the Python twin: the
%                 single-attribute helper route of Bulger's method and
%                 the centres route (one flat attribute that is not
%                 relative-periodic), which evaluates its kernel sum
%                 through internal.gaussianKernelSum and keys its
%                 self-IP memo on the precision. The multi-attribute
%                 log-kernel core has no float32 form, so the value is
%                 not read on any other route. Point evaluation
%                 (evalExpTens) honours it too.
%     'normalize' / 'normalise' — 'cosine' (default) or 'oneSidedDenom'.
%                 Selects the denominator applied to the inner product
%                 <X, Y>. 'cosine' gives the strict shape-only cosine
%                 similarity, dividing by the geometric mean
%                 sqrt(<X, X> * <Y, Y>); the result is bounded in
%                 [-1, 1] and invariant to a positive scalar on either
%                 operand. 'oneSidedDenom' divides by the second
%                 operand's self inner product <Y, Y> alone, giving a
%                 magnitude-aware reading that takes the value 1 on a
%                 self-match (X == Y) and is sensitive to scalar
%                 reweightings of X. Either spelling of the keyword is
%                 accepted; matching on the value is case-insensitive.
%
%   Output:
%     s      — Cosine similarity (scalar in [0, 1] for non-negative
%              weights under normalize = 'cosine'; may exceed 1 under
%              normalize = 'oneSidedDenom' when X has more matching
%              mass than Y carries in total). Returns NaN if r exceeds
%              the number of elements in either multiset.
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
% args. 'verbose' applies to all dispatch arms; 'method' applies to
% the struct and raw-args paths (Möbius dispatch) and is forwarded to
% the per-pair inner calls of the batched-raw path; 'spectrum',
% 'precision', and 'dedup' are
% valid only for the batched-raw path and are forwarded to
% batchCosSimExpTens. Each is captured (with its index range) and
% removed from varargin before the dispatch sees it, so the dispatch
% logic only has to inspect positional arguments.

% Top-level call guard: resets the dispatch-message throttle and pins
% the kernelChunkBytes budget in a single onCleanup. Halves the guard
% overhead vs the previous separate dispatchScope() + pinForCall()
% pair. See internal.callGuard.
guard = internal.callGuard(); %#ok<NASGU>

verbose = true;
method = 'auto';                % 'auto' | 'bulger' | 'mobius'
normalize = 'cosine';           % 'cosine' | 'oneSidedDenom'
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
                if ~ismember(method, ...
                        {'auto', 'bulger', 'centres', 'mobius', 'contract'})
                    error('cosSimExpTens:badMethod', ...
                          ['''method'' must be ''auto'', ''bulger'', ' ...
                           '''centres'', ''mobius'', or ''contract''; ' ...
                           'got ''%s''.'], method);
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
            case {'normalize', 'normalise'}
                % Accept both American and British spellings of the
                % keyword; case-insensitive matching on the value.
                val = char(varargin{i + 1});
                if strcmpi(val, 'cosine')
                    normalize = 'cosine';
                elseif strcmpi(val, 'oneSidedDenom')
                    normalize = 'oneSidedDenom';
                else
                    error('cosSimExpTens:badNormalize', ...
                          ['''normalize'' must be ''cosine'' or ' ...
                           '''oneSidedDenom''; got ''%s''.'], val);
                end
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

% Optional shared [sym] geometry flag. The raw forms (single multiset and MA) carry
% one shared geometry (sigma, r, isRel, isPer, period); isSym joins it
% as an optional trailing positional. Pop it here and normalise nArgs
% back to 9 so the raw-form dispatch below is unchanged; forward it to
% every buildExpTens call via symArgs. (The two-density forms at
% nArgs == 2 read isSym from the precomputed structs and never reach
% this.)
isSymRaw = [];
if nArgs == 10
    isSymRaw = varargin{10};
    varargin(10) = [];
    nArgs = numel(varargin);
end
if isempty(isSymRaw)
    symArgs = {};
else
    symArgs = {isSymRaw};
end

% Determine whether we will dispatch to batched-raw (the only mode
% that accepts 'spectrum', 'precision', and 'dedup'). Reject these
% kwargs early in any other dispatch context so the user gets a
% clear error rather than silent ignore.
%
% Batched-raw fires when nArgs == 9 AND at least one of P1, P2 is a
% genuine 2-D matrix (both dimensions > 1).  The other operand may
% be a vector of matching length, in which case it is broadcast
% against the matrix's rows.
% Compute willBatch early; used both for kwarg validation (only the
% batched-raw mode accepts 'spectrum', 'precision', 'dedup') and as
% the BATCHED-RAW dispatch trigger within the nArgs == 9 branch.
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

% --- Windowed contract check (preserved at top, before canonical dispatch) ---
% As of v2.2, cosSimExpTens does not accept WindowedMaetDensity
% operands. The windowed inner product is a magnitude-aware similarity
% (not a strict cosine similarity in [0, 1]) and is therefore not
% within cosSimExpTens's contract. Use windowedTensorSimilarity for both
% single-offset and multi-offset windowed-similarity calls.
if nArgs == 2 && isstruct(varargin{1}) && isstruct(varargin{2}) ...
        && isfield(varargin{1}, 'tag') && isfield(varargin{2}, 'tag') ...
        && (strcmp(varargin{1}.tag, 'WindowedMaetDensity') ...
         || strcmp(varargin{2}.tag, 'WindowedMaetDensity'))
    error('cosSimExpTens:windowedNotSupported', ...
          ['cosSimExpTens does not accept WindowedMaetDensity ' ...
           'operands. Use windowedTensorSimilarity(densQuery, densContext, ' ...
           'windowSpec, offsets) --- pass a single-column offsets ' ...
           'vector for the scalar single-offset case, or a dim x M ' ...
           'matrix for the M-offset sweep.']);
end

% ==================================================================
% Canonical dispatch order (mirrors entropyExpTens and evalExpTens):
%   nArgs == 2:  precomputed-density forms or LIST.
%     - both struct -> switch on tag pair:
%         * (MaetDensity, MaetDensity), both single-multiset -> prune
%           element level, then fall through to the shared MA inner
%           product below.
%         * (MaetDensity, MaetDensity), general -> MA dens (early return).
%         * tag mismatch                   -> error.
%     - either operand iscell                -> LIST (early return).
%     - otherwise                            -> usage error.
%
%   nArgs == 9:  single-attribute raw form (vectors), with optional
%                batched-raw lift.
%     - either operand iscell                -> "use 10 args" error.
%     - both numeric:
%         * any operand a 2-D matrix         -> BATCHED-RAW (early return).
%         * both vectors                     -> single multiset raw (falls through).
%
%   nArgs == 9:  multi-attribute raw form (cells).
%     - both operands iscell                 -> MA raw (handles single-
%                                                vs-list sub-cases).
%     - otherwise                            -> usage error.
%
%   Otherwise -> usage error.
%
% Each detector is positive and self-sufficient: reordering branches
% within the same nArgs group does not change correctness.
% ==================================================================

USAGE_MSG = ['Usage:\n' ...
    '  single multiset struct:    cosSimExpTens(dens_x, dens_y [, ''verbose'', tf])\n' ...
    '  single multiset raw args:  cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period [, ''verbose'', tf])\n' ...
    '  MA struct:    cosSimExpTens(densMA_x, densMA_y [, ''verbose'', tf])\n' ...
    '  MA raw args:  cosSimExpTens(pAttr1, w1, pAttr2, w2, sigmaVec, rVec, isRelVec, isPerVec, periodVec [, ''verbose'', tf])\n' ...
    '  List mode:    cosSimExpTens({d_x_1, ...}, {d_y_1, ...}) -> cell array of values\n' ...
    '  Batched raw:  cosSimExpTens(P1, W1, P2, W2, sigma, r, isRel, isPer, period) -> vector of values\n' ...
    '                (P1, P2 are nRows-by-K matrices; rows are paired multisets).'];

if nArgs == 2
    a = varargin{1};
    b = varargin{2};

    if isstruct(a) && isstruct(b)
        if ~isfield(a, 'tag') || ~isfield(b, 'tag')
            error('cosSimExpTens:untaggedStruct', ...
                'Both density structs must carry a ''tag'' field.');
        end
        if strcmp(a.tag, 'MaetDensity') && strcmp(b.tag, 'MaetDensity')
            % Seed the self-IP memo caches from the operand structs
            % (empty when absent). The updated caches are attached to
            % the optional second/third outputs below.
            cacheX = localSelfIpFromStruct(a);
            cacheY = localSelfIpFromStruct(b);
            if internal.isSingleMultiset(a) && internal.isSingleMultiset(b)
                % Single-multiset corner (A = N = 1): prune element-level
                % at the density level, then fall through to the shared
                % multi-attribute inner product below.
                maet_x = internal.prunedExpTens(a);
                maet_y = internal.prunedExpTens(b);
            else
                [s, cacheX, cacheY] = localCosSimMA(a, b, method, normalize, ...
                                  verbose, truncationSigmas, cacheX, cacheY, ...
                                  kernelPrecision);
                if nargout > 1
                    densXOut = a; densXOut.selfIP = cacheX;
                    densYOut = b; densYOut.selfIP = cacheY;
                end
                return;
            end
        else
            error('cosSimExpTens:tagMismatch', ...
                ['Both density structs must carry the ''MaetDensity'' ' ...
                 'tag. Got %s and %s.'], a.tag, b.tag);
        end
    elseif iscell(a) || iscell(b)
        % LIST: cell-of-struct on either side (scalar struct may be
        % broadcast against the cell). Inner-element validation occurs
        % within localCosSimDensityList. The self-IP memo is applied
        % internally across the list, so the cache-carrying outputs are
        % not offered here.
        if nargout > 1
            error('cosSimExpTens:selfIpOutputsUnavailable', ...
                ['The cache-carrying outputs are available only in the ' ...
                 'density-struct scalar form; list-mode sweeps memoise ' ...
                 'internally and need no threading.']);
        end
        s = localCosSimDensityList(a, b, normalize, verbose, method, ...
                                   truncationSigmas, kernelPrecision);
        return;
    else
        error('cosSimExpTens:badPairTypes', USAGE_MSG);
    end

elseif nArgs == 9
    if nargout > 1
        error('cosSimExpTens:selfIpOutputsUnavailable', ...
            ['The cache-carrying outputs are available only in the ' ...
             'density-struct scalar form (both operands structs from ' ...
             'buildExpTens). The raw sweep form memoises internally ' ...
             'and needs no threading.']);
    end
    a = varargin{1};
    c = varargin{3};
    if iscell(a) || iscell(c)
        % --- MA raw: cell of attribute matrices (9-arg form).
        %     Distinguishes a single MA pAttr (cell of numeric matrices)
        %     from a list-of-MA (cell of cells) by the first inner
        %     element. ---
        if ~iscell(a) || ~iscell(c)
            error('cosSimExpTens:cellPairNeedsCells', ...
                ['The multi-attribute form requires both p1 (1st) and ' ...
                 'p2 (3rd) to be cells (multi-attribute pAttr).']);
        end
        aIsListOfMA = ~isempty(a) && iscell(a{1});
        bIsListOfMA = ~isempty(c) && iscell(c{1});
        pAttr1    = varargin{1};
        w1        = varargin{2};
        pAttr2    = varargin{3};
        w2        = varargin{4};
        sigmaVec  = varargin{5};
        rVec      = varargin{6};
        isRelVec  = varargin{7};
        isPerVec  = varargin{8};
        periodVec = varargin{9};
        if aIsListOfMA && bIsListOfMA
            error('cosSimExpTens:listVsListNotSupported', ...
                  ['Raw multi-attribute list-vs-list is not supported; pass ' ...
                   'explicit density structs via the density list mode ' ...
                   '(build each entry with buildExpTens first).']);
        end
        if ~aIsListOfMA && ~bIsListOfMA
            dens_x_ma = buildExpTens(pAttr1, w1, sigmaVec, rVec, ...
                isRelVec, isPerVec, periodVec, symArgs{:}, 'verbose', verbose);
            dens_y_ma = buildExpTens(pAttr2, w2, sigmaVec, rVec, ...
                isRelVec, isPerVec, periodVec, symArgs{:}, 'verbose', verbose);
            s = localCosSimMA(dens_x_ma, dens_y_ma, method, normalize, ...
                              verbose, truncationSigmas, [], [], ...
                              kernelPrecision);
            return;
        end
        % Scalar-vs-list broadcast. Build the scalar side once, iterate
        % over the list. Weights for the list side are shared across all
        % entries.
        if bIsListOfMA
            scalarPAttr = pAttr1;  scalarW = w1;
            listPAttr   = pAttr2;  listW   = w2;
            scalarFirst = true;
        else
            scalarPAttr = pAttr2;  scalarW = w2;
            listPAttr   = pAttr1;  listW   = w1;
            scalarFirst = false;
        end
        dens_scalar = buildExpTens(scalarPAttr, scalarW, sigmaVec, rVec, ...
            isRelVec, isPerVec, periodVec, symArgs{:}, 'verbose', verbose);
        M = numel(listPAttr);
        s = cell(1, M);
        densList = cell(1, M);
        for m = 1:M
            densList{m} = buildExpTens(listPAttr{m}, listW, sigmaVec, rVec, ...
                isRelVec, isPerVec, periodVec, symArgs{:}, 'verbose', false);
        end
        % Batched all-r = 1 sweep (see localR1BroadcastFast); the
        % per-pair loop below is the fallback for every other shape.
        if any(strcmp(method, {'auto', 'bulger'}))
            [okFast, sFast] = localR1BroadcastFast(dens_scalar, densList, ...
                scalarFirst, normalize, localSelfIpEmpty(), ...
                truncationSigmas);
            if okFast
                s = sFast;
                return;
            end
        end
        % Thread the scalar side's self-IP memo across the sweep so its
        % self inner product is paid once, not once per entry.
        cacheScalar = localSelfIpEmpty();
        for m = 1:M
            dens_m = densList{m};
            if scalarFirst
                [s{m}, cacheScalar] = localCosSimMA(dens_scalar, dens_m, ...
                                     method, normalize, false, ...
                                     truncationSigmas, cacheScalar, ...
                                     localSelfIpEmpty(), kernelPrecision);
            else
                [s{m}, ~, cacheScalar] = localCosSimMA(dens_m, dens_scalar, ...
                                     method, normalize, false, ...
                                     truncationSigmas, localSelfIpEmpty(), ...
                                     cacheScalar, kernelPrecision);
            end
        end
        return;
    end
    if ~isnumeric(a) || ~isnumeric(c)
        error('cosSimExpTens:badPairTypes', USAGE_MSG);
    end
    if willBatch
        % --- BATCHED-RAW (with optional broadcast) ---
        P1 = varargin{1};
        W1 = varargin{2};
        P2 = varargin{3};
        W2 = varargin{4};
        sigmaBatched = varargin{5};
        % Matrix-valued kernel covariance: whiten both operands here, at
        % the batched-raw entry, and fall through to the isotropic
        % machinery with sigma = 1 (the prefactors cancel under either
        % normalization). This is the Python route (cosine.py whitens
        % before its batched dispatch); the per-row density builds below
        % then see ordinary whitened values. The mode constraints
        % (ordered, absolute, non-periodic, r == K) are enforced per
        % operand first, so a bad covariance raises the same mpt:aniso:*
        % error as the scalar-raw form.
        if internal.isKernelCov(sigmaBatched)
            if spectrumGiven
                error('mpt:aniso:spectrumUnsupported', ...
                    ['''spectrum'' is not supported with a matrix-valued ' ...
                     'kernel covariance (spectral augmentation changes ' ...
                     'the multiset size, breaking r == K).']);
            end
            rIn = varargin{6}; isRelIn = varargin{7}; isPerIn = varargin{8};
            if isempty(isSymRaw)
                isSymIn = true;
            else
                isSymIn = isSymRaw;
            end
            for opIdx = 1:2
                if opIdx == 1
                    opArr = P1; opName = 'sigma (P1)';
                else
                    opArr = P2; opName = 'sigma (P2)';
                end
                if isvector(opArr)
                    kSide = numel(opArr);
                else
                    kSide = size(opArr, 2);
                end
                internal.checkAnisoConstraints(rIn, kSide, isRelIn, ...
                    isPerIn, isSymIn, false, opName);
            end
            [~, Rw] = internal.validateKernelCov(sigmaBatched, round(rIn), ...
                                                 'sigma');
            % whitenValues works on (dim x n) columns: rows are tuples
            % here, so whiten the transpose and transpose back; a vector
            % operand is one tuple and keeps its orientation.
            if isvector(P1)
                P1 = internal.whitenValues(Rw, P1);
            else
                P1 = internal.whitenValues(Rw, P1.').';
            end
            if isvector(P2)
                P2 = internal.whitenValues(Rw, P2);
            else
                P2 = internal.whitenValues(Rw, P2.').';
            end
            sigmaBatched = 1.0;
        end

        isP1Mat = size(P1, 1) > 1 && size(P1, 2) > 1;
        isP2Mat = size(P2, 1) > 1 && size(P2, 2) > 1;

        % Force vector operands to row form (1xK) for uniform broadcast.
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
            sigmaBatched, varargin{6}, varargin{7}, varargin{8}, varargin{9}, ...
            isSymRaw, method, normalize, verbose, ...
            spectrumGiven, spectrumOpt, ...
            precisionGiven, precisionOpt, ...
            dedupGiven, dedupOpt, ...
            truncationSigmas, kernelPrecision);
        return;
    end
    % --- Single-multiset raw: numeric vectors. Builds a MaetDensity at
    %     the A = N = 1 corner and falls through to the shared
    %     multi-attribute inner product below.
    p1     = varargin{1};
    w1     = varargin{2};
    p2     = varargin{3};
    w2     = varargin{4};
    sigma_arg  = varargin{5};
    r_arg      = varargin{6};
    isRel_arg  = varargin{7};
    isPer_arg  = varargin{8};
    J_arg      = varargin{9};

    maet_x = buildExpTens(p1, w1, sigma_arg, r_arg, isRel_arg, isPer_arg, ...
        J_arg, symArgs{:}, 'verbose', verbose);
    maet_y = buildExpTens(p2, w2, sigma_arg, r_arg, isRel_arg, isPer_arg, ...
        J_arg, symArgs{:}, 'verbose', verbose);

else
    error('cosSimExpTens:wrongArgCount', USAGE_MSG);
end

% --- Single-multiset corner (A = N = 1): both entry forms above leave
%     maet_x/maet_y set, so one call to the shared multi-attribute
%     inner product serves both. The struct entry form seeded cacheX /
%     cacheY above; the raw form starts them empty. The memoised
%     values are keyed post-prune and localCosSimMA always prunes
%     first, so attaching them to the unpruned input struct is
%     consistent. ---
structScalarInputs = exist('cacheX', 'var');
if ~structScalarInputs
    cacheX = localSelfIpEmpty();
    cacheY = localSelfIpEmpty();
end
if nargout > 1 && ~structScalarInputs
    error('cosSimExpTens:selfIpOutputsUnavailable', ...
        ['The cache-carrying outputs are available only in the ' ...
         'density-struct scalar form (both operands structs from ' ...
         'buildExpTens).']);
end
[s, cacheX, cacheY] = localCosSimMA(maet_x, maet_y, method, normalize, ...
                  verbose, truncationSigmas, cacheX, cacheY, kernelPrecision);
if nargout > 1
    densXOut = varargin{1}; densXOut.selfIP = cacheX;
    densYOut = varargin{2}; densYOut.selfIP = cacheY;
end

end


function [impossible, reason] = localOrbitIPsImpossible(ip_xy, ip_xx, ip_yy)
%LOCALORBITIPSIMPOSSIBLE  Check for inner products that cannot be correct.
%
%   Triggers on:
%     - non-finite IP (NaN or Inf in any of the three),
%     - negative auto-IP (a self inner product cannot be negative),
%     - cosine magnitude > 1 + 1e-6 (impossible for a genuine cosine).
%
%   All three are mathematically impossible rather than merely
%   inaccurate, so they signal a defect and not a loss of accuracy.
%   Accuracy is governed by truncationSigmas, and this check does NOT
%   test it: a finite, plausible, but insufficiently accurate result
%   passes here.
%
%   REASON names the specific impossibility, for the warning the caller
%   raises before rerouting to enumeration.
%
%   IP_XX may be empty when the first operand's self inner product was
%   not computed (normalize = 'oneSidedDenom' does not consume it); the
%   checks that need it are then skipped and the remaining values are
%   still validated.

    impossible = false;
    reason = '';
    if isempty(ip_xx)
        if ~all(isfinite([ip_xy, ip_yy]))
            impossible = true;
            reason = sprintf(['an inner product is not finite ' ...
                              '(<X,Y> = %g, <Y,Y> = %g)'], ip_xy, ip_yy);
            return;
        end
        if ip_yy < 0
            impossible = true;
            reason = sprintf(['<Y,Y> = %.3e is negative, and a self ' ...
                              'inner product cannot be'], ip_yy);
        end
        return;
    end
    if ~all(isfinite([ip_xy, ip_xx, ip_yy]))
        impossible = true;
        reason = sprintf(['an inner product is not finite ' ...
                          '(<X,Y> = %g, <X,X> = %g, <Y,Y> = %g)'], ...
                         ip_xy, ip_xx, ip_yy);
        return;
    end
    if ip_xx < 0 || ip_yy < 0
        impossible = true;
        if ip_xx < 0
            nm = '<X,X>'; vl = ip_xx;
        else
            nm = '<Y,Y>'; vl = ip_yy;
        end
        reason = sprintf(['%s = %.3e is negative, and a self inner ' ...
                          'product cannot be'], nm, vl);
        return;
    end
    denom = sqrt(ip_xx * ip_yy);
    if denom > 0 && abs(ip_xy) > 1.000001 * denom
        impossible = true;
        reason = sprintf(['the cosine similarity is %.6f, outside ' ...
                          '[-1, 1]'], ip_xy / denom);
    end
end


% =========================================================================
%  localCosSimMA — multi-attribute (MAET) cosine similarity
% =========================================================================

function [s, cacheX, cacheY] = localCosSimMA(dens_x, dens_y, method, ...
                            normalize, verbose, truncationSigmas, ...
                            cacheX, cacheY, kernelPrecision)
%LOCALCOSSIMMA  Cosine similarity between two MaetDensities.
%
%   The inner product factors as an elementwise product of per-attribute
%   kernels (Section 2.7 of the MAET specification); no numerical
%   integration is required for Bulger's method.
%
%   Dispatches between Bulger's method and a per-attribute Möbius
%   method based on method ('auto' / 'bulger' / 'mobius') and a
%   simple r-based heuristic. One post-hoc guard, mirroring the single
%   multiset dispatcher: the Mobius route's inner products are tested
%   for impossible values (non-finite, negative Gram diagonal, cosine
%   outside [-1, 1]) and enumeration is used instead when one is found.
%
%   The trailing ``normalize`` argument selects the denominator
%   applied to the cross inner product: ``'cosine'`` (strict shape-only)
%   divides by the geometric mean of the operand self inner products;
%   ``'oneSidedDenom'`` divides by ``ip_yy`` alone. Under
%   ``'oneSidedDenom'`` the first operand's self inner product is not
%   consumed and is not computed (unless already memoised, when using
%   it is free).
%
%   ``cacheX`` / ``cacheY`` (optional) are self-IP memo structs (see
%   localSelfIpEmpty): each operand's self inner product is looked up
%   before it is computed and stored after, keyed by the route, the
%   resolved truncationSigmas, and the Möbius route's per-attribute
%   closed-form-vs-grid choices (the two routes' values differ by
%   constant prefactors that cancel only within one route's triple, so
%   values never cross keys). The updated structs are returned so a
%   caller looping over pairs can thread them and pay each self term
%   once. If the post-hoc guard rejects a Möbius run, that route's
%   entries are purged before the fallback, so a broken run never
%   seeds the memo.
%
%   Both densities must share the full parameter structure: number of
%   attributes, group assignment, per-attribute r, and per-group sigma,
%   isRel, isPer, period. Weights and event/value counts may differ.

    if nargin < 7 || isempty(cacheX); cacheX = localSelfIpEmpty(); end
    if nargin < 8 || isempty(cacheY); cacheY = localSelfIpEmpty(); end
    % kernelPrecision ([] = the helper's mptDefaults value) is consumed
    % by the single-attribute helper route of ipCoreMA alone, as in the
    % Python core, and keyed into that route's self-IP memo.
    if nargin < 9; kernelPrecision = []; end
    needXX = strcmp(normalize, 'cosine');

    % --- Structural compatibility (cheap fields only) ---
    if ~internal.kernelCovsCompatible(dens_x, dens_y)
        error('mpt:aniso:covMismatch', ...
            ['The two densities were built with different kernel ' ...
             'covariances (or one with a matrix-valued sigma and one ' ...
             'without); inner products require a shared kernel per ' ...
             'attribute.']);
    end
    dens_x = internal.prunedExpTens(dens_x);
    dens_y = internal.prunedExpTens(dens_y);

    % An empty operand has no events to overlap, so the inner product -- and
    % hence the similarity -- is zero. A windowed density whose window caught
    % nothing prunes to zero events here; without this guard it reaches the
    % nested contraction's value-range scan, which has no identity over an
    % empty attribute column. (The raw single-multiset form is unaffected: it
    % is reached only without specs, and an empty windowed density always
    % carries specs.)
    if dens_x.N == 0 || dens_y.N == 0
        s = 0.0;
        return;
    end

    if dens_x.nAttrs ~= dens_y.nAttrs
        error('cosSimExpTens:nAttrsMismatch', ...
            'Both MaetDensities must have the same nAttrs.');
    end
    if ~isequal(dens_x.r, dens_y.r)
        error('cosSimExpTens:rMismatch', ...
            'Both MaetDensities must have the same r (per attribute).');
    end
    if ~isequal(dens_x.sigma, dens_y.sigma)
        error('cosSimExpTens:sigmaMismatch', ...
            'Both MaetDensities must have the same sigma (per attribute).');
    end
    if ~isequal(logical(dens_x.isRel), logical(dens_y.isRel))
        error('cosSimExpTens:isRelMismatch', ...
            'Both MaetDensities must have the same isRel (per attribute).');
    end
    if ~isequal(logical(dens_x.isPer), logical(dens_y.isPer))
        error('cosSimExpTens:isPerMismatch', ...
            'Both MaetDensities must have the same isPer (per attribute).');
    end
    perMask = logical(dens_x.isPer);
    if any(dens_x.period(perMask) ~= dens_y.period(perMask))
        error('cosSimExpTens:periodMismatch', ...
            'Both MaetDensities must have the same period for periodic attributes.');
    end

    % --- Unpack shared parameters (scope for nested helpers) ---
    A        = dens_x.nAttrs;
    rVec     = dens_x.r;
    sigmaG   = dens_x.sigma;
    isRelG   = logical(dens_x.isRel);
    isPerG   = logical(dens_x.isPer);
    periodG  = dens_x.period;

    % --- Method dispatch (mirrors Python _select_ma_inner_product_method) ---
    % The selector's inputs (value counts, sigma/P, grid node counts, the
    % declared wrap vector with its mismatch check, the memo flags read
    % from the caches, and the [sym] flags) are built by
    % INTERNAL.FLATSELECTORINPUTS, shared with explainDispatch so the
    % report cannot drift from the route this call takes.
    [selIn, orderedAny, nestedAny] = internal.flatSelectorInputs( ...
        dens_x, dens_y, normalize, truncationSigmas, cacheX, cacheY);
    wrapG = selIn.wrapVec;

    % Per-attribute co-transposition block size s_u = prod(r(1:u)) where
    % attribute a is a nested attribute resolved to an inner or
    % intermediate [rel] unit u (1-based), 0 otherwise. Used by
    % maLogKernel for the block-diagonal metric.
    innerR = zeros(1, A);
    if isfield(dens_x, 'nested') && iscell(dens_x.nested)
        for a = 1:A
            s = dens_x.nested{a};
            if ~isempty(s) && isstruct(s) && isfield(s, 'proj') ...
                    && (strcmp(s.proj, 'inner') || strcmp(s.proj, 'intermediate'))
                u = s.relUnit;                 % 1-based level index
                innerR(a) = prod(s.r(1:u));    % block size s_u
            end
        end
    end

    tsKeyResolved = truncationSigmas;
    if isempty(tsKeyResolved)
        tsKeyResolved = mptDefaults('truncationSigmas');
    end
    tsKeyResolved = internal.accuracyFloor('resolve', tsKeyResolved);
    % The Bulger and centres arms key their memo on the kernel precision
    % as well (Python: ('bulger', ts, kp, None)): the single-attribute
    % helper route of ipCoreMA honours 'single', and a float32 self
    % inner product must not be served to a double-precision call.
    if isempty(kernelPrecision)
        kpKeyExtra = '';
    else
        kpKeyExtra = char(kernelPrecision);
    end
    chosen = internal.selectMaInnerProductMethod( ...
        selIn.rVec, selIn.kVec, selIn.A, selIn.Nx, selIn.Ny, ...
        selIn.anyPer, selIn.anyRelNonper, selIn.anyRelPer, ...
        selIn.sigmaOverPMax, method, verbose, selIn.relVec, selIn.nuVec, ...
        selIn.kVecY, selIn.wrapVec, selIn.truncationSigmas, ...
        selIn.skipXX, selIn.skipYY, selIn.symVec, ...
        selIn.guardForcedBulger, selIn.perVec);

    % Ordered (isSym = false) attributes are not symmetrised, so the
    % orbit (Möbius) per-attribute inner product does not represent
    % them. Force the pairwise/centres path whenever any attribute is
    % ordered at r_a > 1 (r_a = 1 is vacuous). The centres path reads the
    % actual stored per-attribute centres and is correct either way.
    if orderedAny
        chosen = 'bulger';
    end

    % Nested attributes are not handled by the flat orbit/Möbius entry
    % point: that path would have to flatten the levels into one value set,
    % but the inner unit's metric is block-diagonal (positions couple only
    % within an aligned inner unit), which the flat re-enumeration cannot
    % represent. Route instead to the hierarchical contraction, which
    % contracts the tag tree level by level and itself selects the orbit
    % (Möbius) reduction or permutation/combination enumeration per level;
    % it is not enumeration-only. (nestedAny was resolved above, before
    % the selector, which needs it for its guard flag.)
    if strcmp(method, 'contract') && ~nestedAny
        error('cosSimExpTens:contractUnavailable', ...
            ['method=''contract'' applies to a nested attribute only; ' ...
             'use ''auto'' or ''bulger'' for non-nested densities.']);
    end
    % ``method`` semantics on a nested density:
    %   'bulger'   -- the joint-tuple enumeration, as for a flat density.
    %   'contract' -- the nested contraction plan, forced: an uncovered
    %                 case raises rather than falling back.
    %   'mobius'   -- the same plan. The per-level orbit (Möbius)
    %                 reduction *is* what the contraction applies at every
    %                 symmetric level, so on a nested density 'mobius' and
    %                 'contract' name one route; there is no separate flat
    %                 orbit entry point to ask for (the flat one would have
    %                 to re-enumerate the levels into a single value set,
    %                 which the block-diagonal inner metric forbids).
    %   'centres'  -- the plan with the materialised-centres route forced
    %                 for every nested attribute; raises where that route
    %                 cannot carry the attribute's declared measure.
    % 'centres' and 'mobius' formerly fell through to 'bulger', so the
    % method name described something other than what ran.
    contractTriple = [];
    contractRoutes = {};
    if nestedAny
        if any(strcmp(method, {'auto', 'contract', 'mobius', 'centres'}))
            % Built field by field: struct() with a struct-valued field
            % is safe but reads ambiguously beside the cell-valued case.
            % Clear the price record first, so the announce below cannot
            % read a stale decision from an earlier call when this one
            % declines before it is ever priced.
            internal.lastNestedCosts([]);
            ncOpts = struct();
            ncOpts.methodName = method;
            ncOpts.cacheX = cacheX;
            ncOpts.cacheY = cacheY;
            if strcmp(method, 'centres')
                ncOpts.forceRoute = 'centres';
            end
            [contractTriple, contractRoutes, cacheX, cacheY] = ...
                internal.nestedContract(dens_x, dens_y, normalize, ...
                    truncationSigmas, ~strcmp(method, 'auto'), ncOpts);
        end
        % An empty triple here means method = 'auto' and either the case
        % is not covered by the plan (the forced methods raise instead)
        % or the plan lost the price comparison against the joint-tuple
        % enumeration inside INTERNAL.NESTEDCONTRACT --- the nested twin
        % of the flat selector's Bulger-versus-Moebius choice, whose
        % prices INTERNAL.LASTNESTEDCOSTS records. Either way the
        % enumeration takes it.
        chosen = 'bulger';
    end

    ip_xy = NaN; ip_xx = NaN; ip_yy = NaN;  %#ok<NASGU>  initialised below
    % Announce the decision, as the Python multi-attribute path does. The
    % single-multiset stack announced at its own dispatch point; that
    % stack is no longer reached, so without this the message is lost for
    % every single multiset. On an ordered attribute the tuple-pair path
    % has no permutation expansion to exploit (the perm side equals the
    % comb side), so Bulger's combinations-vs-permutations organisation
    % never runs there and the announce says so.
    chosenLabel = chosen;
    chosenReason = 'ma cost model';
    if nestedAny && isempty(contractTriple) && strcmp(method, 'auto')
        % A nested density that reaches the enumeration under 'auto' did
        % not get here through the flat MA selector: either the plan
        % declined the case or it lost the nested price comparison (the
        % forced methods raise rather than fall back, and
        % method = 'bulger' never asks the plan at all). Say which,
        % rather than crediting a selector that never ran. (Python leaves its
        % "ma_select" reason in place here; the string is diagnostic
        % only, and naming the model that actually decided is worth the
        % divergence.)
        ncCosts = internal.lastNestedCosts();
        if isstruct(ncCosts) && isfield(ncCosts, 'chosen') ...
                && strcmp(ncCosts.chosen, 'bulger')
            chosenReason = 'nested cost model';
        else
            chosenReason = 'nested plan declined';
        end
    end
    if orderedAny && strcmp(chosen, 'bulger')
        chosenLabel = 'bulger (direct on ordered attributes)';
    end
    if ~isempty(contractTriple)
        % Announce what actually ran. The nested plan is not the flat
        % Bulger enumeration, and saying 'bulger' here described the route
        % the contraction had displaced. The per-attribute routes are the
        % informative part, so they are the reason. Mirror of the Python
        % _maybe_show_dispatch_msg("cos_sim_exp_tens", "contract",
        % "nested: ...").
        chosenLabel = 'contract';
        chosenReason = ['nested: ' strjoin(contractRoutes, ',')];
    end
    internal.maybeShowDispatchMsg('cosSimExpTens', chosenLabel, chosenReason);

    ranOrbit = false;
    if ~isempty(contractTriple)
        ip_xy = contractTriple.xy;
        ip_xx = contractTriple.xx;
        ip_yy = contractTriple.yy;
        ranOrbit = true;   % skip both the orbit and the pairwise enumeration
    end

    if strcmp(chosen, 'centres')
        % Unrestricted enumeration of the tuple centres: the O(K^(2r))
        % baseline. It differs from Bulger's route in exactly one
        % respect -- the permutation side is used on BOTH sides, where
        % Bulger uses permutation against combination and multiplies by
        % r!. Everything else is shared: the same ipCoreMA, hence the
        % same truncation, kernel precision, wrap convention and
        % accumulation. Routing through the shared core is what makes
        % the two comparable; an independent re-implementation would
        % measure its own constants rather than the algorithms', and
        % would ignore settings the core honours.
        dens_x = internal.ensureExpTensExpensive(dens_x);
        dens_y = internal.ensureExpTensExpensive(dens_y);
        % Memoise the self terms under this route's own key, exactly as
        % the Bulger arm does. Without this the route recomputes <X,X>
        % and <Y,Y> on every call while the other routes reuse theirs,
        % so a repeated comparison would time three products against
        % one, and a forced 'centres' call would leave no memo for the
        % shared pricing flag to see. Twin of the Python
        % _cos_sim_exp_tens_ma_centres.
        centresKey = localSelfIpKey('centres', tsKeyResolved, kpKeyExtra);
        [xxHit, xxVal] = localSelfIpGet(cacheX, centresKey);
        [yyHit, yyVal] = localSelfIpGet(cacheY, centresKey);
        ip_xy = ipCoreMA(dens_x.U_perm, dens_x.wJ, dens_x.nJ, ...
                         dens_y.U_perm, dens_y.wJ, dens_y.nJ);
        if ~needXX
            ip_xx = [];
        elseif xxHit
            ip_xx = xxVal;
        else
            ip_xx = ipCoreMA(dens_x.U_perm, dens_x.wJ, dens_x.nJ, ...
                             dens_x.U_perm, dens_x.wJ, dens_x.nJ);
            cacheX = localSelfIpSet(cacheX, centresKey, ip_xx);
        end
        if yyHit
            ip_yy = yyVal;
        else
            ip_yy = ipCoreMA(dens_y.U_perm, dens_y.wJ, dens_y.nJ, ...
                             dens_y.U_perm, dens_y.wJ, dens_y.nJ);
            cacheY = localSelfIpSet(cacheY, centresKey, ip_yy);
        end
        ranOrbit = true;   % triple already computed; skip the other arms
    elseif strcmp(chosen, 'mobius')
        [ip_xy, ip_xx, ip_yy, cacheX, cacheY] = localCosSimMAOrbit( ...
            dens_x, dens_y, truncationSigmas, needXX, cacheX, cacheY, ...
            strcmp(method, 'mobius'));

        % Post-hoc correctness check only (mirrors single multiset path);
        % accuracy is governed by truncationSigmas, so no route is
        % diverted on the size of the result. postHocGuards off skips it:
        % the check inspects a result already computed and, when it
        % diverts, pays for Bulger's method on top of this one, so with it
        % active the measured cost of the Mobius route is not the cost of
        % choosing it.
        impossible = false; badReason = '';
        if logical(mptDefaults('postHocGuards'))
            [impossible, badReason] = localOrbitIPsImpossible( ...
                ip_xy, ip_xx, ip_yy);
        end
        if impossible
            warning('mpt:cosSimExpTens:impossibleValue', ...
                ['The Mobius route returned a value that cannot be ' ...
                 'correct: %s. This is a defect, not a loss of ' ...
                 'accuracy, so it is not something truncationSigmas ' ...
                 'governs. Enumeration was used instead; please ' ...
                 'report the inputs.'], badReason);
            % A run the guard rejected must not seed the memo: purge
            % this route's entries from both caches before the
            % fallback.
            cacheX = localSelfIpPurgeRoute(cacheX, 'mobius');
            cacheY = localSelfIpPurgeRoute(cacheY, 'mobius');
        end

        if impossible
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

        % --- Three inner products, with memoised self terms ---
        % The estimate covers only the kernel work this call performs:
        % memoised self terms cost nothing here, and a skipped <X,X>
        % (oneSidedDenom) is never evaluated. The memo is read and
        % written under this route's own key: the routes' values are
        % related by a known constant but are not the same number, so
        % none of them crosses (see localSelfIpKey).
        bulgerKey = localSelfIpKey('bulger', tsKeyResolved, kpKeyExtra);
        [xxHit, xxVal] = localSelfIpGet(cacheX, bulgerKey);
        [yyHit, yyVal] = localSelfIpGet(cacheY, bulgerKey);
        computeXX = needXX && ~xxHit;
        totalPairs = double(nJx)*double(nKy);
        if computeXX
            totalPairs = totalPairs + double(nJx)*double(nKx);
        end
        if ~yyHit
            totalPairs = totalPairs + double(nJy)*double(nKy);
        end
        maxR = max(rVec);
        estimateCompTime(totalPairs, maxR, 'cosSimExpTens (MAET)', verbose);

        ip_xy = ipCoreMA(Ux_perm, wx_perm, nJx, Vy_comb, wvy_comb, nKy);
        if xxHit
            ip_xx = xxVal;
        elseif computeXX
            ip_xx = ipCoreMA(Ux_perm, wx_perm, nJx, Vx_comb, wvx_comb, nKx);
            cacheX = localSelfIpSet(cacheX, bulgerKey, ip_xx);
        else
            ip_xx = [];
        end
        if yyHit
            ip_yy = yyVal;
        else
            ip_yy = ipCoreMA(Uy_perm, wy_perm, nJy, Vy_comb, wvy_comb, nKy);
            cacheY = localSelfIpSet(cacheY, bulgerKey, ip_yy);
        end
    end

    % Final cosine / one-sided-denominator normalisation. An empty
    % ip_xx is legal only under 'oneSidedDenom', whose denominator does
    % not consume it; reaching 'cosine' with it empty is an internal
    % routing defect.
    switch normalize
        case 'cosine'
            if isempty(ip_xx)
                error('cosSimExpTens:missingSelfIp', ...
                    ['normalize=''cosine'' requires <X,X>, but it was ' ...
                     'not computed. This is an internal routing defect.']);
            end
            denom = sqrt(max(ip_xx * ip_yy, 0));
        case 'oneSidedDenom'
            denom = ip_yy;
    end
    if denom == 0
        s = NaN;
    else
        s = ip_xy / denom;
    end

    % =====================================================================
    %  Nested helpers (rVec, sigmaG, isRelG, isPerG, periodG, A
    %  are in scope from the parent).
    % =====================================================================

    function ipval = ipCoreMA(U_cell, wU, nJ, V_cell, wV, nK)
        % Resolve truncationSigmas against mptDefaults for both branches.
        if isempty(truncationSigmas)
            truncResolved = mptDefaults('truncationSigmas');
        else
            truncResolved = truncationSigmas;
        end

        % Dedicated route for the all-r = 1 shape (each event
        % contributes a single joint kernel; the inner product is a
        % smoothed cross-correlation --- the simplest shape the
        % framework supports). Computes the identical quantity to the
        % generic path below --- same per-attribute terms in the same
        % accumulation order, same truncation threshold including the
        % summed-terms tightening, same chunking heuristic --- with a
        % single accumulator and a plain exponential in place of the
        % generic path's per-attribute tensors and masked exponential.
        % Relative attributes at r_a = 1 have a vanishing quadratic
        % form (a 1-tuple has no within-tuple differences) and
        % contribute nothing, exactly as computeQaMA evaluates them.
        % Single-attribute helper route (twin of the Python core's
        % _ip_via_helper): one attribute, no inner [rel] unit, and not
        % relative-periodic (whose pairwise-wrap form the helper cannot
        % take) goes through internal.gaussianKernelSum. The helper
        % computes g(q) = sum_j wV(j) exp(-Q(v_j - u_q) / (2 sigma_eff^2))
        % with sigma_eff = sigma sqrt(2), so its exponent is the
        % centres-IP's Q / (4 sigma^2), and the inner product is wU' g.
        % It carries the spatial-index truncation (and the circular
        % 1-D path on an absolute-periodic attribute), honours
        % kernelPrecision, and takes the same cutoff as the log-kernel
        % form: nTerms = nJ * nK widens the width to
        % sqrt(k^2 + 2 log(nTerms)) inside the helper, which is the
        % -k^2/2 - log(nTerms) threshold of truncLogKernelExp. The
        % numbers agree with the log-kernel form within the accuracy
        % floor; this is the leaf both languages now take.
        if A == 1 && innerR(1) == 0 && ~(isRelG(1) && isPerG(1))
            ipval = ipViaHelper(U_cell{1}, wU, V_cell{1}, wV, ...
                                truncResolved);
            return;
        end

        if all(rVec == 1) && all(innerR == 0)
            ipval = ipR1Direct(U_cell, wU, nJ, V_cell, wV, nK, ...
                               truncResolved);
            return;
        end

        % Memory-aware chunking along the comb-side (nK) dimension.
        maxRa = double(max(rVec));
        bytesNeeded = (maxRa + 2) * double(nJ) * double(nK) * 8;

        memLimit = internal.kernelChunkBytesResolved();

        if bytesNeeded <= memLimit
            ipval = ipFullMA(U_cell, wU, nJ, V_cell, wV, nK, truncResolved);
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
                Ec = internal.truncLogKernelExp(logK, truncResolved, ...
                                        double(nJ) * double(nK));
                acc = acc + Ec * wV(idx).';
            end
            ipval = wU(:).' * acc;
        end
    end

    function ipval = ipViaHelper(U, wU, V, wV, truncResolved)
        % Route the single-attribute centres-IP through
        % internal.gaussianKernelSum (see ipCoreMA). The kernel sum is
        % reduced to one inner product, so the truncation floor has to
        % bound the summed discarded mass over all centre-query pairs
        % rather than each pair individually: nTerms = |U| |V|.
        wrapA = 'full-image';
        if ~isempty(wrapG)
            wrapA = char(wrapG{1});
        end
        kw = {'isRel', logical(isRelG(1)), 'r', double(rVec(1)), ...
              'isPer', logical(isPerG(1)), 'period', double(periodG(1)), ...
              'wrap', wrapA, 'truncationSigmas', truncResolved, ...
              'nTerms', double(size(U, 2)) * double(size(V, 2))};
        if ~isempty(kernelPrecision)
            kw = [kw, {'kernelPrecision', char(kernelPrecision)}];
        end
        sigmaEff = double(sigmaG(1)) * sqrt(2);
        g = internal.gaussianKernelSum(V, wV(:), U, sigmaEff, kw{:});
        ipval = g(:).' * wU(:);
    end

    function ipval = ipR1Direct(U_cell, wU, nJ, V_cell, wV, nK, ...
                                truncResolved)
        % Delegates to the file-scope implementation so the batched
        % broadcast path (localR1BroadcastFast) and this per-pair core
        % share one source.
        ipval = localIpR1Direct(U_cell, wU, nJ, V_cell, wV, nK, ...
            A, sigmaG, isRelG, isPerG, periodG, wrapG, truncResolved);
    end

    function ipval = ipFullMA(U_cell, wU, nJ, V_cell, wV, nK, truncResolved)
        logK = maLogKernel(U_cell, V_cell, nJ, nK);
        E = internal.truncLogKernelExp(logK, truncResolved, ...
                                double(nJ) * double(nK));  % nJ x nK
        ipval = wU(:).' * (E * wV(:));
    end

    function logK = maLogKernel(U_cell, V_cell, nJ, nK)
        % Accumulate sum_a -Q_a / (4 sigma^2) over attributes.
        logK = zeros(nJ, nK);
        for a = 1:A
            r_a = rVec(a);

            % Non-periodic modes: the quadratic form is a squared
            % Euclidean distance between (possibly quotiented)
            % coordinates, so it comes out of one matrix product rather
            % than an (r_a, nJ, nK) difference array. Periodic
            % attributes keep the tensor path below, where the wrap
            % makes the form non-Euclidean. The guard declines the Gram
            % form where its rounding would exceed the accuracy floor.
            if ~isPerG(a) && localGramAccurateEnough( ...
                    U_cell{a}, V_cell{a}, sigmaG(a), truncationSigmas)
                if innerR(a) > 0
                    blockSize = innerR(a);
                elseif isRelG(a)
                    blockSize = r_a;
                else
                    blockSize = 0;
                end
                Qa = localGramQuadraticForm( ...
                    U_cell{a}, V_cell{a}, blockSize);
                logK = logK - Qa / (4 * sigmaG(a)^2);
                continue;
            end

            D = reshape(U_cell{a}, r_a, nJ, 1) ...
              - reshape(V_cell{a}, r_a, 1, nK);

            if innerR(a) > 0
                % Inner [rel] unit: block-diagonal sum of per-event
                % quotient forms (full-tuple convention). The block
                % helper applies the pairwise wrap, so no outer wrap.
                Qa = qInnerBlocks(D, innerR(a), a);
                logK = logK - reshape(Qa, nJ, nK) / (4 * sigmaG(a)^2);
                continue;
            end

            % Abs-per full-image via the shared 1-D wrapped Gaussian
            % (overlap convention, exponent_denominator = 4). The
            % r-tuple full-image kernel factors as prod_k theta(d_k),
            % so log kernel = sum_k log theta(d_k). Single-image
            % opt-in falls through to the Q-form path below with a
            % nearest-image reduction, matching the pre-v3 behaviour.
            if isPerG(a) && ~isRelG(a) ...
                    && strcmp(char(wrapG{a}), 'full-image')
                if isempty(truncationSigmas)
                    tsA = mptDefaults('truncationSigmas');
                else
                    tsA = truncationSigmas;
                end
                tsA = internal.accuracyFloor('resolve', tsA);
                % Single-image short-circuit. When the truncation budget
                % admits no image beyond the nearest one (L = 0, which
                % at the 6-sigma default holds for sigma/P <= 0.059 in
                % this convention), theta(d) *is* the nearest-image
                % Gaussian exp(-d^2 / (4 sigma^2)), so sum_k log
                % theta(d_k) is -Qa / (4 sigma^2) on the nearest-image-
                % reduced differences --- exactly what the Q-form path
                % below computes, without the exp-then-log round trip
                % on the (r_a, nJ, nK) array. Same measure, same number
                % to ~3e-16; measured 1.6x (r = 2) to 3.3x (r = 3)
                % cheaper in Python. Twin of the Python _ma_log_kernel
                % gate.
                if internal.wrappedKernelImageCount( ...
                        sigmaG(a), periodG(a), tsA, 4) > 0
                    theta = internal.wrappedGaussian1d( ...
                        D, sigmaG(a), periodG(a), tsA, 4);
                    logK = logK + reshape( ...
                        sum(log(theta), 1), nJ, nK);
                    continue;
                end
            end

            % The outer wrap is only needed when computeQaMA does not
            % re-wrap the pairwise component differences (i.e., for
            % isPer and not isRel: Qa = sum(D.^2), which requires
            % wrapped D components). For rel+per, computeQaMA wraps
            % each pairwise (D(i)-D(j)) inside (Eq 6 form); that
            % inner wrap is invariant under integer-period shifts, so
            % wrapping D first is redundant.
            if isPerG(a) && ~isRelG(a)
                P_g = periodG(a);
                D = D - P_g .* floor(D / P_g + 0.5);
            end

            Qa = computeQaMA(D, a, r_a);
            logK = logK - reshape(Qa, nJ, nK) / (4 * sigmaG(a)^2);
        end
    end

    function Qa = qInnerBlocks(D, rIn, a)
        % Block-diagonal quadratic form for the inner [rel] co-transposition
        % unit (full-tuple / inner-product convention). D is
        % (rOut*rIn) x nJ x nK; each event block is a full rIn-tuple. Qa is
        % the sum over blocks of the per-block flat relative quotient form
        % (within-event intervals, tensor-joined across events).
        sz = size(D);
        if numel(sz) < 3, sz = [sz, 1]; end
        Qa = zeros(1, sz(2), sz(3));
        nBlocks = floor(sz(1) / rIn);
        P_g = periodG(a);
        for b = 1:nBlocks
            rows = (b - 1) * rIn + (1:rIn);
            Db = D(rows, :, :);
            if isPerG(a)
                for i = 1:rIn
                    for j = i+1:rIn
                        delta = Db(i, :, :) - Db(j, :, :);
                        delta = delta - P_g .* floor(delta / P_g + 0.5);
                        Qa = Qa + delta.^2 / rIn;
                    end
                end
            else
                Qa = Qa + (sum(Db.^2, 1) - sum(Db, 1).^2 / rIn);
            end
        end
    end

    function Qa = computeQaMA(D, a, r_a)
        % Per-attribute quadratic form. Matches the single multiset computeQ logic:
        %   - is_rel && is_per: pairwise-differences formula (wraps
        %     each pairwise delta to [-P/2, P/2), restores exact
        %     transposition invariance on the circle).
        %   - is_rel && ~is_per: sum(d.^2) - sum(d)^2 / r_a.
        %   - ~is_rel:           sum(d.^2).
        if isRelG(a)
            if isPerG(a)
                sz = size(D);
                if numel(sz) < 3, sz = [sz, 1]; end
                Qa = zeros(1, sz(2), sz(3));
                P_g = periodG(a);
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



function [ip_xy, ip_xx, ip_yy, cacheX, cacheY] = localCosSimMAOrbit( ...
    dens_x, dens_y, truncationSigmas, needXX, cacheX, cacheY, ...
    userForcedMobius)
if nargin < 7 || isempty(userForcedMobius), userForcedMobius = false; end
%LOCALCOSSIMMAORBIT  Three MA inner products via per-attribute Möbius method.
%
%   Computes, for each attribute a, an (N_x, N_y) per-attribute inner
%   product matrix I_xy^{(a)}[n_x, n_y] = <T_X^{(a)}_{n_x}, T_Y^{(a)}_{n_y}>
%   (similarly for I_xx, I_yy). The full IP factors as
%       <T_X, T_Y> = sum_{n_x, n_y} prod_a I_xy^{(a)}[n_x, n_y]
%   so we element-wise multiply per-attribute matrices across attributes
%   then sum. NaN-padded events are handled via zero-weight padding in
%   mobius.maPerAttrInnerMatrix.
%
%   NEEDXX = false skips <X,X> when it is neither memoised nor
%   consumed by the caller's normalisation; ip_xx is then []. The self
%   inner products are memoised in CACHEX / CACHEY, keyed on this
%   route's per-attribute closed-form-vs-grid choices (the closed form
%   drops a per-attribute constant prefactor the grid keeps, so a self
%   term is reusable only against triples that made the same choices).
%   The caller purges this route's entries if its post-hoc guard
%   trips, so a broken run never seeds the memo.

    if nargin < 4 || isempty(needXX); needXX = true; end
    if nargin < 5 || isempty(cacheX); cacheX = localSelfIpEmpty(); end
    if nargin < 6 || isempty(cacheY); cacheY = localSelfIpEmpty(); end

    A = dens_x.nAttrs;
    N_x = dens_x.N;
    N_y = dens_y.N;

    % Resolve truncationSigmas to a concrete value at call time so the
    % per-call override path is honoured. [] (no override) defers to
    % mptDefaults inside maPerAttrInnerMatrix.
    if isempty(truncationSigmas)
        truncResolved = mptDefaults('truncationSigmas');
    else
        truncResolved = truncationSigmas;
    end

    % Per-attribute route choice, hoisted because it is part of the
    % self-IP memo key (see the closed-form prefactor note below).
    choices = false(1, A);
    for a = 1:A
        flatAttrA = ~isfield(dens_x, 'nested') ...
            || a > numel(dens_x.nested) || isempty(dens_x.nested{a});
        choices(a) = flatAttrA && mobius.maRelAttrPrefersCentres( ...
            dens_x.pAttr{a}, dens_y.pAttr{a}, dens_x.sigma(a), ...
            dens_x.r(a), dens_x.isRel(a), dens_x.isPer(a), ...
            dens_x.period(a), truncationSigmas, userForcedMobius);
    end
    tsKey = internal.accuracyFloor('resolve', truncResolved);
    orbitKey = localSelfIpKey('mobius', tsKey, char('0' + choices));
    [xxHit, xxVal] = localSelfIpGet(cacheX, orbitKey);
    [yyHit, yyVal] = localSelfIpGet(cacheY, orbitKey);
    computeXX = needXX && ~xxHit;

    P_xy = ones(N_x, N_y);
    if computeXX
        P_xx = ones(N_x, N_x);
    else
        P_xx = [];
    end
    if ~yyHit
        P_yy = ones(N_y, N_y);
    else
        P_yy = [];
    end

    for a = 1:A
        r_a     = dens_x.r(a);
        sigma_g = dens_x.sigma(a);
        isRel_g = dens_x.isRel(a);
        isPer_g = dens_x.isPer(a);
        period_g = dens_x.period(a);

        Px = dens_x.pAttr{a};   Wx = dens_x.w{a};
        Py = dens_y.pAttr{a};   Wy = dens_y.w{a};

        % Flat relative attributes at small K: the pairwise closed
        % form over materialised tuple-centres ((r_a!*C(K, r_a))^2
        % kernel ops per event pair) undercuts the translation-grid
        % contraction (N_u*K^2 ops per pair) by orders of magnitude,
        % and below the sigma/P measure threshold the minimum-image
        % and all-image readings coincide (the predicate enforces
        % that condition). The per-attribute constant prefactor
        % dropped by the closed form multiplies all three matrices of
        % this attribute identically, so it cancels in every supported
        % normalisation. Centres bundles are built once per density
        % and shared by the cross and self matrices. Nested
        % attributes keep the grid path (their contraction routes
        % handle the closed form separately).
        if choices(a)
            % Bundles memoised on the threaded memo structs (twin of the
            % Python _nested_centres_cache; see
            % INTERNAL.NESTEDCENTRESMEMOISED).
            [cxB, cacheX] = internal.nestedCentresMemoised(cacheX, dens_x, a);
            [cyB, cacheY] = internal.nestedCentresMemoised(cacheY, dens_y, a);
            % Per-attribute wrap opt-in (default full-image); the two
            % densities' declarations were checked to agree at the
            % selector site. The per-call width goes with it, as the
            % memo key above already assumes (Python twin:
            % _closed_form_attr_matrix_from(cx, cy, truncation_sigmas,
            % wrap_a)); the closed form reads it only on an abs-per
            % attribute, which the flat gate does not admit today.
            wrapA = 'full-image';
            if isfield(dens_x, 'wrap') && ~isempty(dens_x.wrap) ...
                    && a <= numel(dens_x.wrap)
                wrapA = char(dens_x.wrap{a});
            end
            P_xy = P_xy .* mobius.closedFormAttrMatrixFrom( ...
                cxB, cyB, wrapA, truncResolved);
            if ~isempty(P_xx)
                P_xx = P_xx .* mobius.closedFormAttrMatrixFrom( ...
                    cxB, cxB, wrapA, truncResolved);
            end
            if ~isempty(P_yy)
                P_yy = P_yy .* mobius.closedFormAttrMatrixFrom( ...
                    cyB, cyB, wrapA, truncResolved);
            end
        else
            % Per-attribute wrap opt-in (default full-image). Use
            % dens_x's wrap as authoritative if it and dens_y's differ,
            % matching the closedFormAttrMatrixFrom branch above.
            wrapA = 'full-image';
            if isfield(dens_x, 'wrap') && ~isempty(dens_x.wrap) ...
                    && a <= numel(dens_x.wrap)
                wrapA = char(dens_x.wrap{a});
            end
            P_xy = P_xy .* mobius.maPerAttrInnerMatrix(Px, Wx, Py, Wy, ...
                sigma_g, r_a, isRel_g, isPer_g, period_g, ...
                'truncationSigmas', truncResolved, 'wrap', wrapA);
            if ~isempty(P_xx)
                P_xx = P_xx .* mobius.maPerAttrInnerMatrix(Px, Wx, Px, Wx, ...
                    sigma_g, r_a, isRel_g, isPer_g, period_g, ...
                    'truncationSigmas', truncResolved, 'wrap', wrapA);
            end
            if ~isempty(P_yy)
                P_yy = P_yy .* mobius.maPerAttrInnerMatrix(Py, Wy, Py, Wy, ...
                    sigma_g, r_a, isRel_g, isPer_g, period_g, ...
                    'truncationSigmas', truncResolved, 'wrap', wrapA);
            end
        end
    end

    ip_xy = sum(P_xy(:));
    if xxHit
        ip_xx = xxVal;
    elseif computeXX
        ip_xx = sum(P_xx(:));
        cacheX = localSelfIpSet(cacheX, orbitKey, ip_xx);
    else
        ip_xx = [];
    end
    if yyHit
        ip_yy = yyVal;
    else
        ip_yy = sum(P_yy(:));
        cacheY = localSelfIpSet(cacheY, orbitKey, ip_yy);
    end
end




% =====================================================================
%  Unified dispatch helpers: density-list and batched-raw modes.
% =====================================================================

function sCell = localCosSimDensityList(a, b, normalize, verbose, ...
                                        method, truncationSigmas, ...
                                        kernelPrecision)
%LOCALCOSSIMDENSITYLIST List-mode density-struct cosine similarities.
%
%   Three accepted shapes:
%     (cell, cell)   — pairwise; lengths must match. Returns 1-by-n.
%     (cell, struct) — broadcast struct against the cell. Returns 1-by-n.
%     (struct, cell) — broadcast struct against the cell. Returns 1-by-n.
%
%   Pairwise (cell, cell) mode dispatches recursively to cosSimExpTens;
%   no operand repeats there, so no self-IP memo applies. The two
%   broadcast shapes instead call the scalar-pair dispatcher directly
%   with a memo cache for the shared operand, so its self inner product
%   is paid once across the whole sweep (and, under
%   normalize = 'oneSidedDenom' with the shared operand on the left,
%   not at all). Mixed-kind pairs are not prevented at this level;
%   compatibility is checked downstream. `WindowedMaetDensity` entries
%   are rejected at the top of cosSimExpTens (use
%   windowedTensorSimilarity instead).
%
%   ``normalize``, ``method``, ``truncationSigmas`` and
%   ``kernelPrecision`` are forwarded to each per-pair computation so
%   every list entry takes the route and the width the caller asked
%   for, as the Python list form does. (Earlier versions forwarded
%   ``normalize`` and ``verbose`` alone, so a forced method or a
%   per-call width was silently ignored in list mode.)

    if nargin < 5 || isempty(method); method = 'auto'; end
    if nargin < 6; truncationSigmas = []; end
    if nargin < 7; kernelPrecision = []; end
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
        end
        % Canonical-form dedup (twin of the Python density-list
        % dedup): when every pair is two single-multiset densities,
        % pairs that share a canonical chord pair and the same
        % structural parameters are evaluated once and the value
        % reused. The key is the batched-raw form's pair canonical form
        % (INTERNAL.PAIRCANONICALKEY) plus each side's density
        % parameters; pairs involving a multi-attribute density are
        % computed one by one, as in Python.
        allSingle = n > 0;
        for i = 1:n
            % A whitened (kernel-covariance) density stores values in
            % its own coordinates, which the chord canonical form does
            % not describe; such pairs are computed one by one.
            if ~(internal.isSingleMultiset(a{i}) ...
                    && internal.isSingleMultiset(b{i})) ...
                    || internal.densityHasKernelCov(a{i}) ...
                    || internal.densityHasKernelCov(b{i})
                allSingle = false;
                break;
            end
        end
        if allSingle
            pairKeys = cell(1, n);
            for i = 1:n
                pairKeys{i} = localDensityPairKey(a{i}, b{i});
            end
            [~, firstIdx, mapIdx] = unique(pairKeys, 'stable');
            nUnique = numel(firstIdx);
            if verbose
                fprintf(['cosSimExpTens: %d pairs, %d unique after ' ...
                         'canonical-form dedup.\n'], n, nUnique);
            end
            uniqueVals = cell(1, nUnique);
            for u = 1:nUnique
                i = firstIdx(u);
                uniqueVals{u} = cosSimExpTens(a{i}, b{i}, ...
                                     'normalize', normalize, ...
                                     'method', method, ...
                                     'truncationSigmas', truncationSigmas, ...
                                     'kernelPrecision', kernelPrecision, ...
                                     'verbose', false);
            end
            for i = 1:n
                sCell{i} = uniqueVals{mapIdx(i)};
            end
            return;
        end
        for i = 1:n
            sCell{i} = cosSimExpTens(a{i}, b{i}, ...
                                     'normalize', normalize, ...
                                     'method', method, ...
                                     'truncationSigmas', truncationSigmas, ...
                                     'kernelPrecision', kernelPrecision, ...
                                     'verbose', verbose);
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
    cacheScalar = localSelfIpFromStruct(scalarArg);

    % Batched all-r = 1 broadcast: one shared operand against many
    % queries of identical geometry evaluates every cross term in a
    % single kernel pass, amortising the per-pair dispatch that
    % otherwise dominates point-set-shaped sweeps. ok = false whenever
    % any structural condition fails, and the ordinary per-pair loop
    % below then runs --- raising exactly the errors a genuine mismatch
    % deserves. The fast path is Bulger's kernel pass, so it is offered
    % only where 'auto' or 'bulger' asked for it, as the raw-MA form
    % and the Python twin gate it; a forced 'mobius' or 'centres' takes
    % the per-pair loop and its named route.
    if any(strcmp(method, {'auto', 'bulger'}))
        [okFast, sFast] = localR1BroadcastFast(scalarArg, cellArg, ...
            scalarLeft, normalize, cacheScalar, truncationSigmas);
        if okFast
            sCell = sFast;
            return;
        end
    end

    for i = 1:n
        if ~isstruct(cellArg{i})
            error('cosSimExpTens:listNonStruct', ...
                ['cosSimExpTens (list mode): cell entries must be density ' ...
                 'structs from buildExpTens; entry %d is not a struct.'], i);
        end
        if scalarLeft
            [sCell{i}, cacheScalar] = localScalarPairDispatch( ...
                scalarArg, cellArg{i}, normalize, verbose, ...
                cacheScalar, localSelfIpEmpty(), true, ...
                method, truncationSigmas, kernelPrecision);
        else
            [sCell{i}, cacheScalar] = localScalarPairDispatch( ...
                cellArg{i}, scalarArg, normalize, verbose, ...
                localSelfIpEmpty(), cacheScalar, false, ...
                method, truncationSigmas, kernelPrecision);
        end
    end
end


function [s, cacheShared] = localScalarPairDispatch(dx, dy, normalize, ...
    verbose, cacheX, cacheY, sharedIsX, method, truncationSigmas, ...
    kernelPrecision)
%LOCALSCALARPAIRDISPATCH  One density-struct pair with memo threading.
%
%   Replicates the scalar dispatch cosSimExpTens applies to a
%   two-struct call (the single-multiset corner prunes both operands
%   before the shared multi-attribute inner product; every other shape
%   goes straight to it), with the caller's METHOD, TRUNCATIONSIGMAS and
%   KERNELPRECISION (defaults 'auto', [] and []). Returns the shared
%   operand's updated memo cache (its side selected by SHAREDISX) so the
%   list loop can thread it.

    if ~isfield(dx, 'tag') || ~isfield(dy, 'tag')
        error('cosSimExpTens:untaggedStruct', ...
            'Both density structs must carry a ''tag'' field.');
    end
    if strcmp(dx.tag, 'WindowedMaetDensity') ...
            || strcmp(dy.tag, 'WindowedMaetDensity')
        error('cosSimExpTens:windowedNotSupported', ...
              ['cosSimExpTens does not accept WindowedMaetDensity ' ...
               'operands. Use windowedTensorSimilarity(densQuery, densContext, ' ...
               'windowSpec, offsets) --- pass a single-column offsets ' ...
               'vector for the scalar single-offset case, or a dim x M ' ...
               'matrix for the M-offset sweep.']);
    end
    if ~strcmp(dx.tag, 'MaetDensity') || ~strcmp(dy.tag, 'MaetDensity')
        error('cosSimExpTens:tagMismatch', ...
            ['Both density structs must carry the ''MaetDensity'' ' ...
             'tag. Got %s and %s.'], dx.tag, dy.tag);
    end
    if nargin < 8 || isempty(method); method = 'auto'; end
    if nargin < 9; truncationSigmas = []; end
    if nargin < 10; kernelPrecision = []; end
    if internal.isSingleMultiset(dx) && internal.isSingleMultiset(dy)
        dx = internal.prunedExpTens(dx);
        dy = internal.prunedExpTens(dy);
    end
    [s, cacheX, cacheY] = localCosSimMA(dx, dy, method, normalize, ...
        verbose, truncationSigmas, cacheX, cacheY, kernelPrecision);
    if sharedIsX
        cacheShared = cacheX;
    else
        cacheShared = cacheY;
    end
end


function key = localDensityPairKey(dx, dy)
%LOCALDENSITYPAIRKEY  Canonical-form key for a single-multiset density pair.
%
%   Twin of the key the Python density-list dedup builds: the pair's
%   canonical chord forms from INTERNAL.PAIRCANONICALKEY (independent
%   per side in a relative mode, joint co-transposition in an absolute
%   one; no re-rounding, as the values come from built densities) with
%   the density parameters (sigma, r, isRel, isPer, period) and the
%   per-density declarations (wrap, isSym) of both sides baked in, so
%   two pairs share a key only when the pair core would return the same
%   number for both.
    px = dx.pAttr{1}(:, 1).'; wx = dx.w{1}(:, 1).';
    py = dy.pAttr{1}(:, 1).'; wy = dy.w{1}(:, 1).';
    isRel = logical(dx.isRel(1)); isPer = logical(dx.isPer(1));
    period = double(dx.period(1));
    [pxc, wxc, pyc, wyc] = internal.pairCanonicalKey(px, wx, py, wy, ...
                                                     isRel, isPer, period, []);
    key = sprintf('%s|%s|%s|%s|%s|%s', ...
        sprintf('%.17g,', pxc), sprintf('%.17g,', wxc), ...
        sprintf('%.17g,', pyc), sprintf('%.17g,', wyc), ...
        localDensityParamKey(dx), localDensityParamKey(dy));
end


function key = localDensityParamKey(d)
%LOCALDENSITYPARAMKEY  The density-determining parameters of a
%   single-multiset density as one string (see localDensityPairKey).
    wrapA = 'full-image';
    if isfield(d, 'wrap') && ~isempty(d.wrap)
        wrapA = char(d.wrap{1});
    end
    isSymA = true;
    if isfield(d, 'isSym') && ~isempty(d.isSym)
        isSymA = logical(d.isSym(1));
    end
    key = sprintf('%.17g|%d|%d|%d|%.17g|%s|%d', double(d.sigma(1)), ...
        double(d.r(1)), logical(d.isRel(1)), logical(d.isPer(1)), ...
        double(d.period(1)), wrapA, isSymA);
end


function s = localCosSimBatchedRaw(P1, W1, P2, W2, sigma, r, isRel, isPer, period, ...
    isSym, method, normalize, verbose, ...
    spectrumGiven, spectrumOpt, precisionGiven, precisionOpt, dedupGiven, dedupOpt, ...
    truncationSigmas, kernelPrecision)
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

    if nargin < 20, truncationSigmas = []; end
    if nargin < 21, kernelPrecision  = []; end

    if size(P1, 1) ~= size(P2, 1)
        error('cosSimExpTens:batchedRowMismatch', ...
            ['cosSimExpTens (batched mode): P1 and P2 must have the same ' ...
             'number of rows, got %d and %d.'], size(P1, 1), size(P2, 1));
    end

    % Shared [sym] flag forwarded to every per-row density build.
    if nargin < 10 || isempty(isSym)
        symArgsB = {};
    else
        symArgsB = {isSym};
    end

    % The batched path deduplicates rows by a multiset canonical key,
    % which collapses rows that share a multiset but differ in order.
    % That is correct only for the symmetric reading: under isSym = false
    % the order is significant, so the dedup would silently merge
    % distinct ordered densities. Reject rather than return a wrong
    % answer (parity with the Python batched path). Order-aware batched
    % dedup is a tracked follow-up; use scalar or density-list forms for
    % ordered comparisons.
    if ~isempty(symArgsB) && ~all(logical(isSym(:))) && r > 1
        error('cosSimExpTens:batchedOrderedUnsupported', ...
              ['cosSimExpTens batched (2-D) input does not yet support ' ...
               'isSym = false (ordered) densities at r > 1: the batched ' ...
               'dedup canonicalises each row''s multiset and would merge ' ...
               'order-distinct rows. Build densities individually ' ...
               '(scalar or density-list input) for ordered comparisons.']);
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

        % Canonical form of the pair: the symmetry exploited depends on
        % the mode, and the values are re-rounded afterwards.
        [pAc, wAc, pBc, wBc] = internal.pairCanonicalKey( ...
            pAv, wAv, pBv, wBv, isRel, isPer, period, nDec);

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
    % 'stable' preserves first-occurrence order, so Phase 4 computes
    % unique pairs in input-row order (parity with the Python batched
    % path, which deduplicates via insertion-ordered dict keys).
    % Without it, unique's lexicographic row sort can drastically
    % reorder the work: in an EDO sweep the canonical key's second
    % element is 1200/n, so ascending-key order is *descending* n and
    % the heaviest pairs run first, which misleads anyone reading the
    % progress output as if it followed the input rows.
    validIdx = find(valid);
    [uniqueKeysA, ~, mapA] = unique(keysA(validIdx, :), 'rows', 'stable');
    [uniqueKeysB, ~, mapB] = unique(keysB(validIdx, :), 'rows', 'stable');
    nUniqueA = size(uniqueKeysA, 1);
    nUniqueB = size(uniqueKeysB, 1);

    pairKeys = [mapA, mapB];
    [uniquePairs, ~, pairMap] = unique(pairKeys, 'rows', 'stable');
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
                                 symArgsB{:}, 'verbose', false);
    end

    densB = cell(nUniqueB, 1);
    for ub = 1:nUniqueB
        [pB_u, wB_u] = localExtractFromKey(uniqueKeysB(ub, :), nB, useWeightsB);
        if useSpectra
            [pB_u, wB_u] = addSpectra(pB_u, wB_u, specArgs{:});
        end
        densB{ub} = buildExpTens(pB_u, wB_u, sigma, r, isRel, isPer, period, ...
                                 symArgsB{:}, 'verbose', false);
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
        cosSimExpTens(dA_w, dB_w, 'method', method, ...
                      'normalize', normalize, 'verbose', false);

        % Timed calibration over the sample.
        tCalStart = tic;
        for cs = 1:numel(sampleIdx)
            dA_s = densA{uniquePairs(sampleIdx(cs), 1)};
            dB_s = densB{uniquePairs(sampleIdx(cs), 2)};
            cosSimExpTens(dA_s, dB_s, 'method', method, ...
                          'normalize', normalize, 'verbose', false);
        end
        tCalTotal = toc(tCalStart);
        tPerPair  = tCalTotal / numel(sampleIdx);
        estTotal  = tCalTotal + tPerPair * nUniquePairs;
        internal.printBatchedEstimate('cosSimExpTens', nUniquePairs, estTotal);
        progStride = internal.progressStride(tPerPair);
        showProgress = estTotal >= 5;
    end

    % === Phase 4: Compute similarity for each unique pair ===
    pairKw = {'method', method, ...
              'normalize', normalize, 'verbose', false};
    if ~isempty(truncationSigmas)
        pairKw = [pairKw, {'truncationSigmas', truncationSigmas}];
    end
    if ~isempty(kernelPrecision)
        pairKw = [pairKw, {'kernelPrecision', kernelPrecision}];
    end
    uniqueS = NaN(nUniquePairs, 1);
    for up = 1:nUniquePairs
        dA = densA{uniquePairs(up, 1)};
        dB = densB{uniquePairs(up, 2)};
        uniqueS(up) = cosSimExpTens(dA, dB, pairKw{:});

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

% =========================================================================
%  Self-IP memo helpers. A memo cache is a struct with parallel fields
%  'keys' (1-by-n cellstr) and 'vals' (1-by-n double), keyed by
%  localSelfIpKey. It is carried by value: within one cosSimExpTens
%  call the sweep loops thread it across pairs, and the density-struct
%  scalar form optionally returns it attached to the operand structs
%  (a 'selfIP' field) so a caller's own loop can thread it too. The
%  optional 'nestedCentres' field (1-by-A cell) memoises the centres
%  routes' tuple-centres bundles on the same channel; see
%  INTERNAL.NESTEDCENTRESMEMOISED.
% =========================================================================

function cache = localSelfIpEmpty()
%LOCALSELFIPEMPTY  Fresh, empty self-IP memo cache.
    cache = struct('keys', {{}}, 'vals', []);
end


function cache = localSelfIpFromStruct(d)
%LOCALSELFIPFROMSTRUCT  Read a memo cache from a density struct's
%   'selfIP' field (INTERNAL.SELFIPFROMSTRUCT).
    cache = internal.selfIpFromStruct(d);
end


function tf = localGramAccurateEnough(U, V, sigma, truncationSigmas)
%LOCALGRAMACCURATEENOUGH  Whether the Gram form's rounding is admissible.
%
%   The Gram identity forms |u|^2 + |v|^2 - 2 u.v, so its rounding is
%   relative to the size of those squares rather than to the distance
%   they encode. After the shared shift the coordinates are of the order
%   of the attribute's own spread s, giving an error in the exponent of
%   about eps * s^2 / (4 sigma^2). That is negligible when the spread is
%   comparable to sigma and grows as sigma shrinks against it. Compared
%   against the floor truncationSigmas implies, so a caller asking for
%   accuracy-floor accuracy gets the difference form and one asking for
%   the default gets the fast one.
%
%   Twin of Python _tensor.cosine._gram_is_accurate_enough.
    if isempty(U) || isempty(V)
        tf = true;
        return;
    end
    origin = U(1);
    s2 = max(max(abs(U(:) - origin)), max(abs(V(:) - origin)))^2;
    if s2 == 0
        tf = true;
        return;
    end
    predicted = eps * s2 / (4 * sigma^2);
    tf = predicted <= 0.1 * internal.truncationFloor(truncationSigmas);
end


function Q = localGramQuadraticForm(U, V, blockSize)
%LOCALGRAMQUADRATICFORM  Q(u_j - v_k) for a non-periodic attribute.
%
%   BLOCKSIZE selects the quotient: 0 for absolute (raw coordinates),
%   the tuple length for relative (the whole tuple's all-ones removed),
%   or the co-transposition unit size for a nested attribute (each
%   block's own all-ones removed, the form being the sum over blocks).
%
%   Both operands are shifted by one of the attribute's own values
%   first. The Gram identity cancels two large numbers when the
%   coordinates sit far from the origin, which costs significant digits
%   --- measured against the difference form, the log-kernel departed by
%   4e-3 at magnitude 1e6 and 5e-1 at 1e7. Everything here depends on
%   the operands only through their differences, so a shift shared by
%   both is exact; taking it from the data rather than from a mean makes
%   the subtraction itself exact as well (Sterbenz), where a mean would
%   inject a rounding error at just those magnitudes.
%
%   Twin of Python _tensor.cosine._gram_quadratic_form.
    r = size(U, 1);
    if isempty(U) || isempty(V)
        Q = zeros(size(U, 2), size(V, 2));
        return;
    end
    origin = U(1);
    U = U - origin;
    V = V - origin;

    if blockSize <= 0
        rowSets = {1:r};
        centreEach = false;
    elseif blockSize >= r
        rowSets = {1:r};
        centreEach = true;
    else
        nBlocks = floor(r / blockSize);
        rowSets = cell(1, nBlocks);
        for b = 1:nBlocks
            rowSets{b} = ((b - 1) * blockSize + 1):(b * blockSize);
        end
        centreEach = true;
    end

    Q = zeros(size(U, 2), size(V, 2));
    for b = 1:numel(rowSets)
        Ub = U(rowSets{b}, :);
        Vb = V(rowSets{b}, :);
        if centreEach
            Ub = Ub - mean(Ub, 1);
            Vb = Vb - mean(Vb, 1);
        end
        Q = Q + (sum(Ub .^ 2, 1).' + sum(Vb .^ 2, 1) - 2 * (Ub.' * Vb));
    end
    Q = max(Q, 0);
end


function key = localSelfIpKey(route, tsResolved, extra)
%LOCALSELFIPKEY  Memo key for a self inner product.
%
%   Delegates to INTERNAL.SELFIPKEY, which documents the key's contents.
%   The scheme lives in a package function because
%   INTERNAL.NESTEDCONTRACT memoises into the same caches and must spell
%   its keys the same way; a local function cannot be shared across
%   files.
    key = internal.selfIpKey(route, tsResolved, extra);
end


function [hit, val] = localSelfIpGet(cache, key)
%LOCALSELFIPGET  Look up a memoised self inner product.
    idx = find(strcmp(cache.keys, key), 1);
    hit = ~isempty(idx);
    if hit
        val = cache.vals(idx);
    else
        val = [];
    end
end


function cache = localSelfIpSet(cache, key, val)
%LOCALSELFIPSET  Store a self inner product under its key.
    idx = find(strcmp(cache.keys, key), 1);
    if isempty(idx)
        cache.keys{end + 1} = key;
        cache.vals(end + 1) = val;
    else
        cache.vals(idx) = val;
    end
end


function hit = localSelfIpMemoised(cache)
%LOCALSELFIPMEMOISED  True when any inner-product route has memoised this
%   density's self inner product.
%
%   Delegates to INTERNAL.SELFIPMEMOISED, which documents why the flag is
%   shared by the routes a selector compares rather than read off each
%   route's own memo. Twin of the Python cosine._self_ip_memoised.
    hit = internal.selfIpMemoised(cache);
end


function cache = localSelfIpPurgeRoute(cache, route)
%LOCALSELFIPPURGEROUTE  Drop every entry of the given route (used when
%   the post-hoc guard rejects a Möbius run, so a broken run never
%   seeds the memo).
    keep = ~strncmp(cache.keys, [route '|'], numel(route) + 1);
    cache.keys = cache.keys(keep);
    cache.vals = cache.vals(keep);
end

% =========================================================================
%  All-r = 1 direct inner product and batched broadcast.
% =========================================================================

function ipval = localIpR1Direct(U_cell, wU, nJ, V_cell, wV, nK, ...
    A, sigmaG, isRelG, isPerG, periodG, wrapG, truncResolved)
%LOCALIPR1DIRECT  Direct MA inner product for the all-r = 1 shape.
%
%   Computes the same quantity as the generic maLogKernel path ---
%   per-attribute terms accumulated in the same order, with the same
%   truncation threshold (resolve through the accuracy floor,
%   -0.5 k^2 in log space, tightened by -log(nTerms) for the summed
%   count) --- with a single accumulator and a plain exponential.
%   Chunking along the comb side follows the generic path's memory
%   heuristic. Relative attributes at r_a = 1 have a vanishing
%   quadratic form and contribute nothing.

    tsR = internal.accuracyFloor('resolve', truncResolved);
    threshold = -0.5 * tsR^2;
    nTerms = double(nJ) * double(nK);
    if nTerms > 1
        threshold = threshold - log(nTerms);
    end

    bytesPerCol = 3 * double(nJ) * 8;
    memLimit = internal.kernelChunkBytesResolved();
    chunkSize = max(1, floor(memLimit / bytesPerCol));

    wVcol = wV(:);
    acc = zeros(nJ, 1);
    for c = 1:chunkSize:nK
        cEnd = min(c + chunkSize - 1, nK);
        idx  = c:cEnd;
        L = zeros(nJ, numel(idx));
        for a = 1:A
            if isRelG(a)
                continue;
            end
            d = reshape(U_cell{a}, nJ, 1) ...
              - reshape(V_cell{a}(1, idx), 1, numel(idx));
            if isPerG(a) && strcmp(char(wrapG{a}), 'full-image')
                theta = internal.wrappedGaussian1d( ...
                    d, sigmaG(a), periodG(a), tsR, 4);
                L = L + log(theta);
                continue;
            end
            if isPerG(a)
                P_g = periodG(a);
                d = d - P_g .* floor(d / P_g + 0.5);
            end
            L = L - d.^2 / (4 * sigmaG(a)^2);
        end
        below = L < threshold;
        E = exp(L);
        E(below) = 0;
        acc = acc + E * wVcol(idx);
    end
    ipval = wU(:).' * acc;
end


function [ok, sCell] = localR1BroadcastFast(sharedDens, entryCell, ...
    sharedIsX, normalize, cacheShared, truncationSigmas)
%LOCALR1BROADCASTFAST  Batched broadcast for the all-r = 1 shape.
%
%   Applies when one operand is shared across every pair and all
%   densities are flat MaetDensity structs of identical geometry with
%   every r_a = 1 (no nesting, no kernel covariance; r = 1
%   symmetrisation is vacuous). The cross inner products of the whole
%   sweep are then one kernel pass over the concatenated query
%   columns, with the truncation threshold applied per column segment
%   using that pair's own nTerms, so the set of zeroed entries matches
%   the per-pair path exactly. Self terms use the same memo keys the
%   per-pair route uses (seeded from each struct's selfIP field where
%   present); as X under 'oneSidedDenom' a self term is not consumed
%   and not computed. The only floating-point difference from the
%   per-pair path is the contraction association ((wU * E) . wV here
%   versus wU * (E * wV) per pair), within the toolbox-wide <= 1e-12
%   parity discipline.
%
%   OK = false whenever any structural condition fails; the caller's
%   ordinary per-pair loop then runs and raises the errors a genuine
%   mismatch deserves.

    if nargin < 6
        truncationSigmas = [];
    end
    ok = false;
    sCell = {};

    n = numel(entryCell);
    if n == 0
        return;
    end
    densAll = [{sharedDens}, entryCell(:).'];
    for i = 1:numel(densAll)
        d = densAll{i};
        if ~isstruct(d) || ~isfield(d, 'tag') ...
                || ~strcmp(d.tag, 'MaetDensity')
            return;
        end
        if isfield(d, 'kernelCov') && ~isempty(d.kernelCov)
            return;
        end
        if isfield(d, 'nested') && iscell(d.nested) ...
                && any(~cellfun(@isempty, d.nested))
            return;
        end
    end

    sharedP = internal.prunedExpTens(sharedDens);
    A = sharedP.nAttrs;
    if A < 1 || ~all(sharedP.r == 1)
        return;
    end
    entriesP = cell(1, n);
    for i = 1:n
        entriesP{i} = internal.prunedExpTens(entryCell{i});
    end

    if isfield(sharedP, 'wrap') && ~isempty(sharedP.wrap)
        wrapG = sharedP.wrap;
    else
        wrapG = repmat({'full-image'}, 1, A);
    end
    perMask = logical(sharedP.isPer);
    for i = 1:n
        d = entriesP{i};
        if d.nAttrs ~= A ...
                || ~isequal(d.r, sharedP.r) ...
                || ~isequal(d.sigma, sharedP.sigma) ...
                || ~isequal(logical(d.isRel), logical(sharedP.isRel)) ...
                || ~isequal(logical(d.isPer), logical(sharedP.isPer))
            return;
        end
        if any(d.period(perMask) ~= sharedP.period(perMask))
            return;
        end
        if isfield(d, 'wrap') && ~isempty(d.wrap)
            wrapD = d.wrap;
        else
            wrapD = repmat({'full-image'}, 1, A);
        end
        for a = 1:A
            if ~strcmp(char(wrapD{a}), char(wrapG{a}))
                return;
            end
        end
    end

    % Every structural condition holds: from here the fast path is
    % committed and computes the values.
    ok = true;
    sCell = cell(1, n);

    sigmaG  = sharedP.sigma;
    isRelG  = logical(sharedP.isRel);
    isPerG  = logical(sharedP.isPer);
    periodG = sharedP.period;
    if isempty(truncationSigmas)
        truncResolved = mptDefaults('truncationSigmas');
    else
        truncResolved = truncationSigmas;
    end
    tsR = internal.accuracyFloor('resolve', truncResolved);
    bulgerKey = localSelfIpKey('bulger', tsR, '');
    needSharedSelf = ~sharedIsX || strcmp(normalize, 'cosine');
    needEntrySelf  = sharedIsX || strcmp(normalize, 'cosine');

    if sharedP.N == 0
        for i = 1:n
            sCell{i} = 0.0;
        end
        return;
    end

    sharedP = internal.ensureExpTensExpensive(sharedP);
    nJ = sharedP.nJ;
    U_cell = sharedP.U_perm;
    wU = sharedP.wJ;

    % Shared operand's self term, through the same memo key the
    % per-pair route uses.
    ipShared = [];
    [hitS, valS] = localSelfIpGet(cacheShared, bulgerKey);
    if hitS
        ipShared = valS;
    elseif needSharedSelf
        ipShared = localIpR1Direct(U_cell, wU, nJ, ...
            sharedP.V_comb, sharedP.wv_comb, sharedP.nK, ...
            A, sigmaG, isRelG, isPerG, periodG, wrapG, truncResolved);
    end

    % Concatenate the live entries' comb-side columns; record segment
    % boundaries and per-column thresholds (that pair's own nTerms).
    segLen = zeros(1, n);
    for i = 1:n
        if entriesP{i}.N ~= 0
            entriesP{i} = internal.ensureExpTensExpensive(entriesP{i});
            segLen(i) = entriesP{i}.nK;
        end
    end
    T = sum(segLen);
    starts = [0, cumsum(segLen)];
    ipXY = zeros(1, n);
    if T > 0
        Vc = cell(1, A);
        for a = 1:A
            cols = cell(1, n);
            for i = 1:n
                if segLen(i) > 0
                    cols{i} = entriesP{i}.V_comb{a};
                end
            end
            Vc{a} = [cols{:}];
        end
        thrCol = zeros(1, T);
        for i = 1:n
            if segLen(i) == 0
                continue;
            end
            nTerms = double(nJ) * double(entriesP{i}.nK);
            t = -0.5 * tsR^2;
            if nTerms > 1
                t = t - log(nTerms);
            end
            thrCol(starts(i) + 1 : starts(i + 1)) = t;
        end

        % One kernel pass over the concatenated columns. Chunk width is
        % the smaller of the memory-limit heuristic and a
        % cache-resident cap: a single (nJ, T) block at large nJ leaves
        % cache and drops kernel throughput; a few MB per block keeps
        % locality while amortising the per-chunk cost.
        bytesPerCol = 3 * double(nJ) * 8;
        memLimit = internal.kernelChunkBytesResolved();
        cacheCap = max(1, floor(4e6 / max(double(nJ) * 8, 1)));
        chunkSize = max(1, min(floor(memLimit / bytesPerCol), cacheCap));
        rowSums = zeros(1, T);          % (wU * E) per column
        for c = 1:chunkSize:T
            cEnd = min(c + chunkSize - 1, T);
            idx = c:cEnd;
            L = zeros(nJ, numel(idx));
            for a = 1:A
                if isRelG(a)
                    continue;
                end
                d = reshape(U_cell{a}, nJ, 1) ...
                  - reshape(Vc{a}(1, idx), 1, numel(idx));
                if isPerG(a) && strcmp(char(wrapG{a}), 'full-image')
                    theta = internal.wrappedGaussian1d( ...
                        d, sigmaG(a), periodG(a), tsR, 4);
                    L = L + log(theta);
                    continue;
                end
                if isPerG(a)
                    P_g = periodG(a);
                    d = d - P_g .* floor(d / P_g + 0.5);
                end
                L = L - d.^2 / (4 * sigmaG(a)^2);
            end
            below = L < thrCol(idx);
            E = exp(L);
            E(below) = 0;
            rowSums(idx) = wU(:).' * E;
        end
        for i = 1:n
            if segLen(i) > 0
                seg = starts(i) + 1 : starts(i + 1);
                ipXY(i) = rowSums(seg) * entriesP{i}.wv_comb(:);
            end
        end
    end

    for i = 1:n
        d = entriesP{i};
        if d.N == 0
            sCell{i} = 0.0;
            continue;
        end
        ipSelf = [];
        cacheE = localSelfIpFromStruct(entryCell{i});
        [hitE, valE] = localSelfIpGet(cacheE, bulgerKey);
        if hitE
            ipSelf = valE;
        elseif needEntrySelf
            ipSelf = localIpR1Direct(d.U_perm, d.wJ, d.nJ, ...
                d.V_comb, d.wv_comb, d.nK, ...
                A, sigmaG, isRelG, isPerG, periodG, wrapG, truncResolved);
        end
        if sharedIsX
            ipXX = ipShared;
            ipYY = ipSelf;
        else
            ipXX = ipSelf;
            ipYY = ipShared;
        end
        switch normalize
            case 'cosine'
                denom = sqrt(max(ipXX * ipYY, 0));
            case 'oneSidedDenom'
                denom = ipYY;
        end
        if denom == 0
            sCell{i} = NaN;
        else
            sCell{i} = ipXY(i) / denom;
        end
    end
end

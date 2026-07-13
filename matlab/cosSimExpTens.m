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
%   tag (SA or MA). Shape rule: a length-1 input returns a length-1
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
%     p1, p2          — Vectors of length K_1, K_2 (may differ). SA raw
%                       form (single multiset, single-attribute).
%     P1, P2          — nRows-by-K matrices, both dimensions > 1.
%                       BATCHED-RAW form (rows are independent SA-style
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
%   Inputs (SA raw calling convention):
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
%              are independent SA-style multisets, paired between P1
%              and P2). At least one of P1, P2 must have both
%              dimensions > 1; the other may be a length-K vector that
%              is broadcast against the matrix's rows.
%     W1, W2 — Weights paired with P1, P2 (same shape, or [] for
%              uniform). Broadcast in lockstep with their P operand.
%     sigma, r, isRel, isPer, period — As in the SA raw convention
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
%     'method'  — 'auto' (default), 'bulger', 'mobius', 'direct', or
%                 'contract' (force the nested tree-contraction).
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
% args. 'verbose' applies to all dispatch arms; 'method' and
% 'cancellationThreshold' apply to SA and MA struct/raw-args paths
% (Möbius dispatch) and are forwarded to the per-pair inner calls of
% the batched-raw path; 'spectrum', 'precision', and 'dedup' are
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
normalize = 'cosine';           % 'cosine' | 'oneSidedDenom'
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
                if ~ismember(method, ...
                        {'auto', 'bulger', 'mobius', 'contract'})
                    error('cosSimExpTens:badMethod', ...
                          ['''method'' must be ''auto'', ''bulger'', ' ...
                           '''mobius'', or ''contract''; got ''%s''.'], ...
                          method);
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

% Optional shared [sym] geometry flag. The raw forms (SA and MA) carry
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
%         * (ExpTensDensity, ExpTensDensity) -> SA dens (falls through
%           to shared SA compatibility validation + SA dispatch below).
%         * (MaetDensity,   MaetDensity)   -> MA dens (early return).
%         * tag mismatch                   -> error.
%     - either operand iscell                -> LIST (early return).
%     - otherwise                            -> usage error.
%
%   nArgs == 9:  single-attribute raw form (vectors), with optional
%                batched-raw lift.
%     - either operand iscell                -> "use 10 args" error.
%     - both numeric:
%         * any operand a 2-D matrix         -> BATCHED-RAW (early return).
%         * both vectors                     -> SA raw (falls through).
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
    '  SA struct:    cosSimExpTens(dens_x, dens_y [, ''verbose'', tf])\n' ...
    '  SA raw args:  cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period [, ''verbose'', tf])\n' ...
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
        switch [a.tag '|' b.tag]
            case 'ExpTensDensity|ExpTensDensity'
                % SA dens: validate compatibility below, then fall through.
                dens_x = internal.prunedExpTens(a);
                dens_y = internal.prunedExpTens(b);
            case 'MaetDensity|MaetDensity'
                s = localCosSimMA(a, b, method, normalize, ...
                                  cancellationThreshold, verbose, ...
                                  truncationSigmas);
                return;
            otherwise
                error('cosSimExpTens:tagMismatch', ...
                    ['Both density structs must carry matching tags ' ...
                     '(ExpTensDensity vs ExpTensDensity, or MaetDensity vs ' ...
                     'MaetDensity). Got %s and %s.'], a.tag, b.tag);
        end
    elseif iscell(a) || iscell(b)
        % LIST: cell-of-struct on either side (scalar struct may be
        % broadcast against the cell). Inner-element validation occurs
        % within localCosSimDensityList.
        s = localCosSimDensityList(a, b, normalize, verbose);
        return;
    else
        error('cosSimExpTens:badPairTypes', USAGE_MSG);
    end

elseif nArgs == 9
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
                              cancellationThreshold, verbose, ...
                              truncationSigmas);
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
        for m = 1:M
            dens_m = buildExpTens(listPAttr{m}, listW, sigmaVec, rVec, ...
                isRelVec, isPerVec, periodVec, symArgs{:}, 'verbose', false);
            if scalarFirst
                s{m} = localCosSimMA(dens_scalar, dens_m, method, ...
                                     normalize, cancellationThreshold, false, ...
                                     truncationSigmas);
            else
                s{m} = localCosSimMA(dens_m, dens_scalar, method, ...
                                     normalize, cancellationThreshold, false, ...
                                     truncationSigmas);
            end
        end
        return;
    end
    if ~isnumeric(a) || ~isnumeric(c)
        error('cosSimExpTens:badPairTypes', USAGE_MSG);
    end
    if willBatch
        % --- BATCHED-RAW (with optional broadcast) ---
        if internal.isKernelCov(varargin{5})
            error('mpt:aniso:batchedUnsupported', ...
                ['Batched-raw (2-D) input is not supported with a ' ...
                 'matrix-valued kernel covariance; carry the tuples ' ...
                 'as events of an ordered multi-attribute form, or ' ...
                 'build per-row density objects.']);
        end
        P1 = varargin{1};
        W1 = varargin{2};
        P2 = varargin{3};
        W2 = varargin{4};

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
            varargin{5}, varargin{6}, varargin{7}, varargin{8}, varargin{9}, ...
            isSymRaw, method, cancellationThreshold, normalize, verbose, ...
            spectrumGiven, spectrumOpt, ...
            precisionGiven, precisionOpt, ...
            dedupGiven, dedupOpt);
        return;
    end
    % --- SA raw: numeric vectors. Builds skinny; Bulger branch ensures
    %     heavy fields on demand inside localCosSimSA. Falls through to
    %     SA compatibility validation + SA dispatch below.
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
                          symArgs{:}, 'verbose', verbose);
    dens_y = buildExpTens(p2, w2, sigma_arg, r_arg, isRel_arg, isPer_arg, J_arg, ...
                          symArgs{:}, 'verbose', verbose);

else
    error('cosSimExpTens:wrongArgCount', USAGE_MSG);
end

% --- SA compatibility validation (shared by SA dens-struct and SA raw) ---
% For SA raw the two densities are built from identical scalar
% parameters, so these checks are trivially satisfied. They are run
% unconditionally so the same code path serves both entry forms.
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
if ~internal.kernelCovsCompatible(dens_x, dens_y)
    error('mpt:aniso:covMismatch', ...
        ['The two densities were built with different kernel ' ...
         'covariances (or one with a matrix-valued sigma and one ' ...
         'without); inner products require a shared kernel per ' ...
         'attribute.']);
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

% Ordered (isSym = false) densities are not symmetrised, so the orbit
% (Möbius) inner product --- which reconstructs the full S_r orbit from
% p/w/r --- does not represent them. The pairwise/centres path reads the
% actual stored centres and is correct for either reading, so force it
% whenever either operand is ordered at r > 1 (r = 1 is vacuous).
xOrdered = isfield(dens_x, 'isSym') && ~all(logical(dens_x.isSym(:)));
yOrdered = isfield(dens_y, 'isSym') && ~all(logical(dens_y.isSym(:)));
if (xOrdered || yOrdered) && dens_x.r > 1
    chosen = 'bulger';
    routingReason = 'ordered density (sym=0) requires centres path';
end

% Dispatch messages bypass per-call verbose; they're gated by the
% toolbox-wide showHints flag and throttled to once per top-level user
% call per unique (funcName, chosen, reason) triple (via
% +internal/dispatchScope).
internal.maybeShowDispatchMsg('cosSimExpTens', chosen, ...
    routingReason, estSec, probed);

ip_xy = NaN; ip_xx = NaN; ip_yy = NaN;  %#ok<NASGU>  initialised below
ranOrbit = false;

if strcmp(chosen, 'mobius')
    [ip_xy, ip_xx, ip_yy, worstRatio] = localCosSimSAOrbit(dens_x, dens_y, ...
                                                            truncationSigmas);

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

% Final cosine / one-sided-denominator normalisation.
switch normalize
    case 'cosine'
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
    %  (e.g. cosSimExpTens in per-pair tight loops like windowedTensorSimilarity)
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

        if canUseHelper && ~useDefaultKwargs
            ipval = ipViaHelper(U, wU, V, wV);
            return;
        end

        % Default-mode (or rel+per) path: inline / chunked.
        bytesNeeded = (r + 2) * double(nJ) * double(nK) * 8;

        memLimit = internal.kernelChunkBytesResolved();

        if bytesNeeded <= memLimit
            ipval = ipFull(U, wU, nJ, V, wV, nK, truncResolved);
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

                Ec = reshape( ...
                    internal.truncKernelExp(Qc(:), sigma, truncResolved), ...
                    nJ, nKc);
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
    function ipval = ipFull(U, wU, nJ, V, wV, nK, truncResolved)
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

        E = reshape( ...
            internal.truncKernelExp(Qvec(:), sigma, truncResolved), ...
            nJ, nK);
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
%     6. Periodic-relative beyond sigma/period > 0.03: the all-image (Möbius)
%        form is the faster SA path, so it is taken; because it differs from
%        the canonical single-wrap (Bulger) measure above this sigma/period,
%        warn and point to method='bulger' for the single-wrap measure.

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
    if isRel && isPer && sigmaOverP > 0.03 && verbose   % _ORBIT_SIGMA_OVER_P_THRESHOLD
        internal.warnRelPerAllImage(sigmaOverP);
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
%  Extrapolation. Both paths compute three inner products (cross term
%  plus both self-norms), so the cost models count all three: pairwise
%  IP cost scales as P_x*P_y + P_x^2 + P_y^2 with P =
%  falling_factorial(K, r) (ordered r-tuple enumeration on each side);
%  orbit IP cost scales as B_r * (N_xy*K_x*K_y + N_xx*K_x^2 +
%  N_yy*K_y^2), where the N factors are the relative-mode
%  translation-grid sizes (1 in absolute mode). The probe uses
%  (min(K_x, 12), min(K_y, 12)) events — per side, so an asymmetric
%  workload is probed with the same asymmetry — and extrapolates by the
%  ratio of the corresponding op counts. The pairwise estimate removes
%  its fixed per-call cost with a two-point fit before scaling, and the
%  probe decision requires the orbit estimate to beat the pairwise
%  estimate by PROBE_IP_MOBIUS_DECISION_MARGIN.
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
    % Fixed-overhead term for the Möbius inner-product cost estimate, in units
    % of K^2 (added to the grid-scaled kernel-op count inside the orbit-cost
    % expression). The Möbius method carries a per-call setup cost (orbit-table
    % lookup, contraction planning, partition iteration) that scales with the
    % partition count B_r but is independent of K; the bare operation count
    % omits it and so under-estimates Möbius at small K, making the analytical
    % pre-screen route to Möbius well before it is actually faster than
    % Bulger's method. Adding B_r*ORBIT_IP_FIXED_OVERHEAD to the orbit cost
    % shifts the analytical equal-cost point to (just below) the empirically
    % measured Bulger/Möbius crossover per r, so the pre-screen never claims
    % 'mobius' prematurely; the probe still has the final word in the
    % near-crossover region. The value is calibrated for the
    % three-inner-product cost model (cross term plus both self-norms, so
    % 3*K^2 kernel-op units at symmetric K in absolute mode): it is 3x the
    % per-IP setup constant calibrated against the measured absolute-mode
    % crossovers, which keeps the symmetric-K absolute-mode equal-cost point
    % at those measured values.
    ORBIT_IP_FIXED_OVERHEAD = 24000.0;
    % Dominance margin for the *Möbius* side of the analytical pre-screen.
    % Larger than PRESCREEN_IP_DOMINANCE so that near-crossover cases (where the
    % analytical model is least reliable) defer to the timing probe rather than
    % committing to Möbius on an under-estimate. The Bulger side keeps the
    % tighter PRESCREEN_IP_DOMINANCE because over-predicting Bulger is cheap
    % (its cost is genuinely low in that regime) whereas prematurely choosing
    % Möbius pays its fixed overhead needlessly.
    PRESCREEN_IP_MOBIUS_DOMINANCE = 10.0;
    % Margin the orbit probe estimate must beat the pairwise probe estimate
    % by before the probe routes to the Möbius method. The orbit probe's
    % working set is cache-resident while the full-size rel-mode path is
    % memory-bound, so the probe systematically under-measures the
    % full-scale per-op cost; the pairwise estimate carries no matching
    % bias (its fixed per-call cost is removed by the two-point fit
    % below). Near-tie estimates therefore route to the pairwise path,
    % the cheap-to-mispick side, mirroring the asymmetric pre-screen
    % margins.
    PROBE_IP_MOBIUS_DECISION_MARGIN = 1.4;
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
    % Relative-periodic measure note: 'mobius' is the all-image
    % (transposition-integral) form, 'bulger' the single-wrap (minimum-image)
    % form; they diverge above sigma/P = 0.03. The dispatch takes the faster
    % path (pre-screen / probe below); when that path is the all-image Möbius
    % method and sigma/P is above the threshold it warns at the return point
    % and points to method='bulger' for the canonical single-wrap measure.
    relPerAbove = isRel && isPer && sigmaOverP > 0.03;   % _ORBIT_SIGMA_OVER_P_THRESHOLD

    % ---- Relative-periodic measure preference (takes precedence over cost) --
    % Above the sigma/P threshold the Möbius method computes the all-image
    % transposition average while Bulger's method computes the single-wrap
    % (minimum-image) form -- these are *different measures*, not two routes to
    % the same answer. The toolbox's default measure there is the all-image
    % form, so we must choose Möbius on measure grounds regardless of the cost
    % comparison below (which assumes both methods compute the same object, as
    % they do in absolute mode and in relative mode below the threshold). Emit
    % the warning pointing to method='bulger' for the single-wrap measure.
    if relPerAbove
        if verbose
            internal.warnRelPerAllImage(sigmaOverP);
        end
        chosen = 'mobius';
        probed = false;
        estSec = 0;
        routingReason = 'rel-per all-image measure';
        return;
    end

    % ---- Analytical cost models ----
    % Both paths compute three inner products: the cross term <x, y> and
    % the two self-norms <x, x> and <y, y>. The self-norm terms must be
    % counted: the pairwise path's <y, y> costs FF(K_y, r)^2 tuple pairs,
    % which dominates the cross term's FF(K_x, r)*FF(K_y, r) whenever the
    % operand sizes are asymmetric (a small reference against a large
    % candidate set makes <y, y> the whole cost, not a correction).
    P_x = localFallingFactorial(K_x, r);
    P_y = localFallingFactorial(K_y, r);
    pairwiseFull = P_x * P_y + P_x * P_x + P_y * P_y;
    B_r          = BELL_NUMBERS.(sprintf('r%d', r));
    % Orbit cost = B_r * (grid-scaled kernel-op count + fixed overhead).
    % In relative mode each orbit inner product runs the contraction at
    % every node of a translation grid, so the kernel-op count carries the
    % per-inner-product grid size as a multiplicative factor (3332 nodes
    % for sigma = 3, period = 1200 — three orders of magnitude, not a
    % correction). In absolute mode the factors are 1. The fixed-overhead
    % term (see ORBIT_IP_FIXED_OVERHEAD) captures the K-independent Möbius
    % setup cost that the bare operation count omits; without it the
    % pre-screen routes to Möbius well before the measured Bulger/Möbius
    % crossover.
    [N_xy, N_xx, N_yy] = localOrbitIPGridFactors( ...
        dens_x.p, dens_y.p, sigma, isRel, isPer, period);
    orbitVar  = N_xy * K_x * K_y + N_xx * K_x * K_x + N_yy * K_y * K_y;
    orbitFull = B_r * (orbitVar + ORBIT_IP_FIXED_OVERHEAD);

    % ---- Analytical pre-screen ----
    % The Möbius side uses a larger dominance margin than the Bulger side. Even
    % with the fixed-overhead correction the analytical orbit cost slightly
    % under-predicts the measured crossover at higher r, so firing 'mobius' on a
    % bare 3x margin can still route one K-step early. Requiring a larger margin
    % keeps near-crossover cases in the probe's hands (the probe times both
    % paths and is portable across machines), while still short-circuiting the
    % clear-win region.
    if orbitFull * PRESCREEN_IP_MOBIUS_DOMINANCE < pairwiseFull
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
    % Per-side probe sizes, so an asymmetric workload (a small reference
    % against a large candidate set) is probed with the same asymmetry:
    % probing both sides at the smaller K misrepresents the per-op cost
    % of the larger side's self-norm, which dominates the pairwise path
    % at scale. K_probe - r >= 2 is guaranteed per side by the precision
    % rule above.
    K_probe_x = min(K_x, 12);
    K_probe_y = min(K_y, 12);

    tPairwise = localProbeIPPath(dens_x, dens_y, K_probe_x, K_probe_y, ...
                                  'bulger', truncationSigmas, kernelPrecision);
    tOrbit    = localProbeIPPath(dens_x, dens_y, K_probe_x, K_probe_y, ...
                                  'mobius', truncationSigmas, kernelPrecision);

    % ---- Extrapolate to full workload ----
    % Each factor is the ratio of the full-workload op count to the op
    % count of what the probe actually measured. The pairwise probe times
    % one cross inner product at (K_probe_x, K_probe_y); the orbit probe
    % times all three orbit inner products at those sizes. For the orbit
    % path the grid factors at probe scale come from the subset pitch
    % arrays: in periodic-relative mode they equal the full-scale factors
    % (the grid depends only on sigma and period, so they cancel in the
    % ratio); in non-periodic relative mode the subset spans set smaller
    % grids and the ratio carries the difference; in absolute mode all
    % factors are 1.
    P_px = localFallingFactorial(K_probe_x, r);
    P_py = localFallingFactorial(K_probe_y, r);
    pairwiseProbe = P_px * P_py;
    if pairwiseProbe > 0
        pairwiseFactor = pairwiseFull / pairwiseProbe;
    else
        pairwiseFactor = 1;
    end

    % Two-point fixed-cost removal for the pairwise estimate. A probe at
    % the target sizes measures mostly fixed per-call cost (its op count
    % is small relative to the per-call setup), so scaling the raw
    % timing by the op-count ratio inflates the estimate by that fixed
    % share times the ratio -- a factor of 2-3 at large extrapolation
    % ratios, all of it biasing the decision toward the Möbius method. A
    % second probe at a smaller subset separates the two components: fit
    % t = a + b*ops through the two points, carry the fixed part
    % unscaled, and scale only the variable part b. The probe times one
    % cross inner product, so its fitted fixed cost is per inner
    % product; the full pairwise path computes three, hence the 3*a
    % term. Skipped when the extrapolation ratio is small (raw scaling
    % is then accurate) or when a meaningfully smaller second point is
    % unavailable.
    twoPointDone = false;
    if pairwiseFactor > 2
        K2_x = max(r + 2, floor(K_probe_x / 2));
        K2_y = max(r + 2, floor(K_probe_y / 2));
        pairwiseProbe2 = localFallingFactorial(K2_x, r) ...
                       * localFallingFactorial(K2_y, r);
        if pairwiseProbe2 < 0.7 * pairwiseProbe
            tPairwise2 = localProbeIPPath(dens_x, dens_y, K2_x, K2_y, ...
                'bulger', truncationSigmas, kernelPrecision);
            b = max((tPairwise - tPairwise2) ...
                    / (pairwiseProbe - pairwiseProbe2), 0);
            a = max(tPairwise - b * pairwiseProbe, 0);
            tPairwiseEst = 3 * a + b * pairwiseFull;
            twoPointDone = true;
        end
    end
    if ~twoPointDone
        tPairwiseEst = tPairwise * pairwiseFactor;
    end

    [Np_xy, Np_xx, Np_yy] = localOrbitIPGridFactors( ...
        dens_x.p(1:K_probe_x), dens_y.p(1:K_probe_y), ...
        sigma, isRel, isPer, period);
    orbitProbe = Np_xy * K_probe_x * K_probe_y ...
               + Np_xx * K_probe_x * K_probe_x ...
               + Np_yy * K_probe_y * K_probe_y;
    if orbitProbe > 0
        orbitFactor = orbitVar / orbitProbe;
    else
        orbitFactor = 1;
    end

    tOrbitEst = tOrbit * orbitFactor;

    if tPairwiseEst <= tOrbitEst * PROBE_IP_MOBIUS_DECISION_MARGIN
        chosen = 'bulger';
        estSec = tPairwiseEst;
    else
        % Note: rel-per-above-threshold is handled by the measure-preference
        % guard above (it returns before reaching the probe), so the probe only
        % runs where Bulger and Möbius compute the same object; no measure
        % warning here.
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


function [N_xy, N_xx, N_yy] = localOrbitIPGridFactors(p_x, p_y, sigma, ...
                                                       isRel, isPer, period)
%LOCALORBITIPGRIDFACTORS  Cost-model grid weights of the three orbit IPs.
%
%   Returns (N_xy, N_xx, N_yy), the grid weights of the cross term and
%   the two self-norms. The relative-mode orbit inner product
%   marginalises a translation u over a grid and runs the orbit
%   contraction at every grid point, so its kernel-op count carries the
%   grid size as a multiplicative factor. The three factors mirror the
%   grid-sizing rules of mobius.orbitInnerRelSA: in periodic mode the
%   grid covers one period with internal.autoNtauDefault(period, sigma)
%   nodes (identical for all three inner products); in non-periodic mode
%   the line grid spans the two operands' spreads plus the 16-sigma
%   truncation margin at 10 samples per sigma, so each inner product has
%   its own size. The absolute-mode orbit inner product is grid-free, so
%   all three factors are 1.
%
%   In relative mode each grid size is scaled by
%   ORBIT_GRID_OP_UNIT_COST, the per-op cost of a translation-grid
%   orbit kernel op relative to a pairwise kernel op, so the orbit and
%   pairwise cost models price their kernel ops in a shared unit;
%   without it the rel-mode orbit cost is under-priced by that ratio
%   and the modelled equal-cost point sits below the measured one.
%
%   The constant is per-implementation: this is the MATLAB value,
%   calibrated from bench_ip_dispatch.m (tests/) measurements on the
%   EDO-approximation workload (K_x = 5, sigma = 6, period = 1200)
%   with the slabbed translation grid in mobius.orbitInnerRelSA, which
%   keeps the contraction memory-resident at every K: orbit
%   contraction 24-35 ns per kernel op against pairwise 2.2-4.5 ns,
%   median ratio 14. Rerun the bench to recalibrate on new hardware.
%   (The Python sibling constant in _tensor/dispatch.py is calibrated
%   the same way on the Python implementation.)
%
%   Both the full-size and probe-size cost expressions call this
%   helper, so the scaling cancels in the probe's extrapolation ratio:
%   it moves the analytical pre-screen boundaries only. The
%   absolute-mode orbit cost calibration (ORBIT_IP_FIXED_OVERHEAD
%   against measured absolute-mode crossovers) predates no such factor
%   and is left untouched.

    ORBIT_GRID_OP_UNIT_COST = 14;

    if ~isRel
        N_xy = 1; N_xx = 1; N_yy = 1;
        return;
    end
    if isPer
        n = internal.autoNtauDefault(period, sigma) * ORBIT_GRID_OP_UNIT_COST;
        N_xy = n; N_xx = n; N_yy = n;
        return;
    end
    samplesPerSigma = 10;
    spread_x = max(p_x) - min(p_x);
    spread_y = max(p_y) - min(p_y);
    N_xy = max(64, ceil(max(spread_x + spread_y + 16 * sigma, 1.0) ...
                        / sigma * samplesPerSigma)) * ORBIT_GRID_OP_UNIT_COST;
    N_xx = max(64, ceil(max(2 * spread_x + 16 * sigma, 1.0) ...
                        / sigma * samplesPerSigma)) * ORBIT_GRID_OP_UNIT_COST;
    N_yy = max(64, ceil(max(2 * spread_y + 16 * sigma, 1.0) ...
                        / sigma * samplesPerSigma)) * ORBIT_GRID_OP_UNIT_COST;
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


function t = localProbeIPPath(dens_x, dens_y, K_probe_x, K_probe_y, ...
                               path, truncationSigmas, kernelPrecision)
%LOCALPROBEIPPATH  Time one cosSimExpTens IP path on a subset.
%
%   The probe sizes are per side so an asymmetric workload (a small
%   reference against a large candidate set) is probed with the same
%   asymmetry.
%
%   Runs a warmup pass (discarded) to stabilise CPU caches and one-shot
%   table loads -- without it, the path that ran most recently on the
%   full workload comes into the probe with hot caches and gets
%   unfairly favoured -- then repeats the timed work until at least
%   PROBE_MIN_SAMPLE_SEC has elapsed (capped at PROBE_MAX_REPS
%   repetitions) and returns the mean per-run time. A single timed pass
%   is not enough: on fast hardware a probe subset's real work can be
%   microseconds inside ~1 ms of per-call overhead and timer noise, and
%   the two-point pairwise fit divides a difference of two such
%   timings -- sub-millisecond noise there is amplified by the op-count
%   extrapolation ratio into estimates wrong by orders of magnitude.
%   Repetition until the sample is above noise makes the fitted slope
%   meaningful; for probes whose single run already exceeds the floor,
%   the loop exits after one repetition and costs nothing extra.

    subX = buildExpTens(dens_x.p(1:K_probe_x), dens_x.w(1:K_probe_x), ...
        dens_x.sigma, dens_x.r, dens_x.isRel, dens_x.isPer, ...
        dens_x.period, 'verbose', false);
    subY = buildExpTens(dens_y.p(1:K_probe_y), dens_y.w(1:K_probe_y), ...
        dens_y.sigma, dens_y.r, dens_y.isRel, dens_y.isPer, ...
        dens_y.period, 'verbose', false);

    PROBE_MIN_SAMPLE_SEC = 0.008;
    PROBE_MAX_REPS = 64;

    if strcmp(path, 'mobius')
        % Warmup pass (discarded).
        [~, ~, ~, ~] = localCosSimSAOrbit(subX, subY, truncationSigmas);
        % Timed passes: repeat until the sample is above timer noise.
        reps = 0;
        tStart = tic;
        while true
            [~, ~, ~, ~] = localCosSimSAOrbit(subX, subY, truncationSigmas);
            reps = reps + 1;
            elapsed = toc(tStart);
            if elapsed >= PROBE_MIN_SAMPLE_SEC || reps >= PROBE_MAX_REPS
                break;
            end
        end
        t = elapsed / reps;
    else
        subX = internal.ensureExpTensExpensive(subX);
        subY = internal.ensureExpTensExpensive(subY);
        % Warmup pass (discarded).
        localProbePairwiseIP(subX, subY, truncationSigmas, kernelPrecision);
        % Timed passes: repeat until the sample is above timer noise.
        reps = 0;
        tStart = tic;
        while true
            localProbePairwiseIP(subX, subY, truncationSigmas, kernelPrecision);
            reps = reps + 1;
            elapsed = toc(tStart);
            if elapsed >= PROBE_MIN_SAMPLE_SEC || reps >= PROBE_MAX_REPS
                break;
            end
        end
        t = elapsed / reps;
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
    if isempty(truncationSigmas)
        truncResolved = mptDefaults('truncationSigmas');
    else
        truncResolved = truncationSigmas;
    end
    E = internal.truncKernelExp(Q, sigma, truncResolved);
    if ~isempty(kernelPrecision) && strcmp(kernelPrecision, 'single')
        E = single(E);
    end
    ip_xy = wU(:).' * (E * wV(:));
end


function [ip_xy, ip_xx, ip_yy, worstRatio] = localCosSimSAOrbit(dens_x, ...
                                                                 dens_y, ...
                                                                 truncationSigmas)
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

    % Resolve truncationSigmas to a concrete value at call time so the
    % per-call override is honoured (an empty/unset trunc defers to
    % mptDefaults inside the orbit helpers).
    if isempty(truncationSigmas)
        truncResolved = mptDefaults('truncationSigmas');
    else
        truncResolved = truncationSigmas;
    end

    if isRel
        [ip_xy, r_xy] = mobius.orbitInnerRelSA(p_x, w_x, p_y, w_y, sigma, r, isPer, period, ...
            'truncationSigmas', truncResolved);
        [ip_xx, r_xx] = mobius.orbitInnerRelSA(p_x, w_x, p_x, w_x, sigma, r, isPer, period, ...
            'truncationSigmas', truncResolved);
        [ip_yy, r_yy] = mobius.orbitInnerRelSA(p_y, w_y, p_y, w_y, sigma, r, isPer, period, ...
            'truncationSigmas', truncResolved);
    else
        [ip_xy, r_xy] = mobius.orbitInnerAbsSA(p_x, w_x, p_y, w_y, sigma, r, isPer, period, ...
            'truncationSigmas', truncResolved);
        [ip_xx, r_xx] = mobius.orbitInnerAbsSA(p_x, w_x, p_x, w_x, sigma, r, isPer, period, ...
            'truncationSigmas', truncResolved);
        [ip_yy, r_yy] = mobius.orbitInnerAbsSA(p_y, w_y, p_y, w_y, sigma, r, isPer, period, ...
            'truncationSigmas', truncResolved);
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

function s = localCosSimMA(dens_x, dens_y, method, normalize, ...
                            cancellationThreshold, verbose, truncationSigmas)
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
%   The trailing ``normalize`` argument selects the denominator
%   applied to the cross inner product: ``'cosine'`` (strict shape-only)
%   divides by the geometric mean of the operand self inner products;
%   ``'oneSidedDenom'`` divides by ``ip_yy`` alone.
%
%   Both densities must share the full parameter structure: number of
%   attributes, group assignment, per-attribute r, and per-group sigma,
%   isRel, isPer, period. Weights and event/slot counts may differ.

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
    % hence the similarity -- is zero. A windowed carrier whose window caught
    % nothing prunes to zero events here; without this guard it reaches the
    % nested contraction's value-range scan, which has no identity over an
    % empty attribute column. (The raw single-attribute path is unaffected: it
    % is reached only without specs, and an empty windowed carrier always
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

    % --- Method dispatch (mirrors Python _select_ma_inner_product_method) ---
    kVec = zeros(1, A);
    for a = 1:A
        kVec(a) = size(dens_x.pAttr{a}, 1);
    end
    anyPer = false; anyRelNonper = false; anyRelPer = false; sigmaOverPMax = 0;
    for a = 1:A
        if isPerG(a); anyPer = true; end
        if isRelG(a)
            if isPerG(a)
                anyRelPer = true;
                if periodG(a) > 0
                    sigmaOverPMax = max(sigmaOverPMax, sigmaG(a) / periodG(a));
                end
            else
                anyRelNonper = true;
            end
        end
    end
    chosen = internal.selectMaInnerProductMethod( ...
        rVec, kVec, A, dens_x.N, dens_y.N, anyPer, anyRelNonper, anyRelPer, ...
        sigmaOverPMax, method, verbose);

    % Ordered (isSym = false) attributes are not symmetrised, so the
    % orbit (Möbius) per-attribute inner product does not represent
    % them. Force the pairwise/centres path whenever any attribute is
    % ordered at r_a > 1 (r_a = 1 is vacuous). The centres path reads the
    % actual stored per-attribute centres and is correct either way.
    if isfield(dens_x, 'isSym') || isfield(dens_y, 'isSym')
        rRow = rVec(:).';
        sxOrd = false; syOrd = false;
        if isfield(dens_x, 'isSym')
            sxOrd = any(~logical(dens_x.isSym(:).') & (rRow > 1));
        end
        if isfield(dens_y, 'isSym')
            syOrd = any(~logical(dens_y.isSym(:).') & (rRow > 1));
        end
        if sxOrd || syOrd
            chosen = 'bulger';
        end
    end

    % Nested attributes are not handled by the flat orbit/Möbius entry
    % point: that path would have to flatten the levels into one slot set,
    % but the inner unit's metric is block-diagonal (slots couple only
    % within an aligned inner unit), which the flat re-enumeration cannot
    % represent. Route instead to the hierarchical contraction, which
    % contracts the tag tree level by level and itself selects the orbit
    % (Möbius) reduction or permutation/combination enumeration per level;
    % it is not enumeration-only.
    nestedAny = (isfield(dens_x, 'nested') && iscell(dens_x.nested) ...
                 && any(~cellfun(@isempty, dens_x.nested))) ...
             || (isfield(dens_y, 'nested') && iscell(dens_y.nested) ...
                 && any(~cellfun(@isempty, dens_y.nested)));
    if strcmp(method, 'contract') && ~nestedAny
        error('cosSimExpTens:contractUnavailable', ...
            ['method=''contract'' applies to a nested attribute only; ' ...
             'use ''auto'' or ''bulger'' for non-nested densities.']);
    end
    contractTriple = [];
    if nestedAny
        if any(strcmp(method, {'auto', 'contract'}))
            contractTriple = internal.nestedContract( ...
                dens_x, dens_y, normalize, truncationSigmas, ...
                strcmp(method, 'contract'));
        end
        chosen = 'bulger';
    end

    ip_xy = NaN; ip_xx = NaN; ip_yy = NaN;  %#ok<NASGU>  initialised below
    ranOrbit = false;
    if ~isempty(contractTriple)
        ip_xy = contractTriple.xy;
        ip_xx = contractTriple.xx;
        ip_yy = contractTriple.yy;
        ranOrbit = true;   % skip both the orbit and the pairwise enumeration
    end

    if strcmp(chosen, 'mobius')
        [ip_xy, ip_xx, ip_yy] = localCosSimMAOrbit(dens_x, dens_y, ...
                                                    truncationSigmas);

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

    % Final cosine / one-sided-denominator normalisation.
    switch normalize
        case 'cosine'
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
                Ec = internal.truncLogKernelExp(logK, truncResolved);
                acc = acc + Ec * wV(idx).';
            end
            ipval = wU(:).' * acc;
        end
    end

    function ipval = ipFullMA(U_cell, wU, nJ, V_cell, wV, nK, truncResolved)
        logK = maLogKernel(U_cell, V_cell, nJ, nK);
        E = internal.truncLogKernelExp(logK, truncResolved);  % nJ x nK
        ipval = wU(:).' * (E * wV(:));
    end

    function logK = maLogKernel(U_cell, V_cell, nJ, nK)
        % Accumulate sum_a -Q_a / (4 sigma^2) over attributes.
        logK = zeros(nJ, nK);
        for a = 1:A
            r_a = rVec(a);
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
        % Per-attribute quadratic form. Matches the SA computeQ logic:
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



function [ip_xy, ip_xx, ip_yy] = localCosSimMAOrbit(dens_x, dens_y, ...
                                                     truncationSigmas)
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

    % Resolve truncationSigmas to a concrete value at call time so the
    % per-call override path is honoured. [] (no override) defers to
    % mptDefaults inside maPerAttrInnerMatrix.
    if isempty(truncationSigmas)
        truncResolved = mptDefaults('truncationSigmas');
    else
        truncResolved = truncationSigmas;
    end

    for a = 1:A
        r_a     = dens_x.r(a);
        sigma_g = dens_x.sigma(a);
        isRel_g = dens_x.isRel(a);
        isPer_g = dens_x.isPer(a);
        period_g = dens_x.period(a);

        Px = dens_x.pAttr{a};   Wx = dens_x.w{a};
        Py = dens_y.pAttr{a};   Wy = dens_y.w{a};

        I_xy = mobius.maPerAttrInnerMatrix(Px, Wx, Py, Wy, ...
            sigma_g, r_a, isRel_g, isPer_g, period_g, ...
            'truncationSigmas', truncResolved);
        I_xx = mobius.maPerAttrInnerMatrix(Px, Wx, Px, Wx, ...
            sigma_g, r_a, isRel_g, isPer_g, period_g, ...
            'truncationSigmas', truncResolved);
        I_yy = mobius.maPerAttrInnerMatrix(Py, Wy, Py, Wy, ...
            sigma_g, r_a, isRel_g, isPer_g, period_g, ...
            'truncationSigmas', truncResolved);

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

function sCell = localCosSimDensityList(a, b, normalize, verbose)
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
%   at the top of cosSimExpTens (use windowedTensorSimilarity instead).
%
%   The ``normalize`` argument is forwarded to each per-pair
%   cosSimExpTens call so every list entry uses the same denominator.

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
            sCell{i} = cosSimExpTens(a{i}, b{i}, ...
                                     'normalize', normalize, ...
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
    for i = 1:n
        if ~isstruct(cellArg{i})
            error('cosSimExpTens:listNonStruct', ...
                ['cosSimExpTens (list mode): cell entries must be density ' ...
                 'structs from buildExpTens; entry %d is not a struct.'], i);
        end
        if scalarLeft
            sCell{i} = cosSimExpTens(scalarArg, cellArg{i}, ...
                                     'normalize', normalize, ...
                                     'verbose', verbose);
        else
            sCell{i} = cosSimExpTens(cellArg{i}, scalarArg, ...
                                     'normalize', normalize, ...
                                     'verbose', verbose);
        end
    end
end


function s = localCosSimBatchedRaw(P1, W1, P2, W2, sigma, r, isRel, isPer, period, ...
    isSym, method, cancellationThreshold, normalize, verbose, ...
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
                      'cancellationThreshold', cancellationThreshold, ...
                      'normalize', normalize, 'verbose', false);

        % Timed calibration over the sample.
        tCalStart = tic;
        for cs = 1:numel(sampleIdx)
            dA_s = densA{uniquePairs(sampleIdx(cs), 1)};
            dB_s = densB{uniquePairs(sampleIdx(cs), 2)};
            cosSimExpTens(dA_s, dB_s, 'method', method, ...
                          'cancellationThreshold', cancellationThreshold, ...
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
    uniqueS = NaN(nUniquePairs, 1);
    for up = 1:nUniquePairs
        dA = densA{uniquePairs(up, 1)};
        dB = densB{uniquePairs(up, 2)};
        uniqueS(up) = cosSimExpTens(dA, dB, ...
                                    'method', method, ...
                                    'cancellationThreshold', cancellationThreshold, ...
                                    'normalize', normalize, ...
                                    'verbose', false);

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

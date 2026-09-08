function dens = buildExpTens(varargin)
%BUILDEXPTENS Precompute an r-ad expectation tensor density object.
%
%   Input forms, in the order to reach for them: a single multiset;
%   a pre-MAET, the canonical entry for everything else; then the raw
%   positional multi-attribute form.
%
%   SINGLE-ATTRIBUTE (legacy):
%     dens = buildExpTens(p, w, sigma, r, isRel, isPer, period)
%     dens = buildExpTens(..., 'verbose', false)
%
%   PRE-MAET (the canonical multi-attribute entry):
%     dens = buildExpTens(pm)
%     dens = buildExpTens(pm, 'sigma', sigmaVec, ...)
%   A pre-MAET (preMaet) stands in place of pAttr and wAttr, bringing
%   its specs with it. Any of the six per-attribute parameters --
%   'sigma', 'isPer', 'period', 'r', 'rel', 'sym' -- may be given
%   alongside, and a supplied value wins over the specs for every
%   attribute, so a sweep over any of them is one call per value and
%   leaves the pre-MAET untouched.
%
%   MULTI-ATTRIBUTE, positional (MAET):
%     dens = buildExpTens(pAttr, wAttr, sigmaVec, rVec, ...
%                         isRelVec, isPerVec, periodVec)
%     dens = buildExpTens(..., 'verbose', false)
%
%   PRE-MAET:
%     dens = buildExpTens(pm)
%     dens = buildExpTens(pm, 'sigma', sigmaVec, ...)
%   A pre-MAET (preMaet) stands in place of pAttr and wAttr, bringing
%   its specs with it. Any of the six per-attribute parameters --
%   'sigma', 'isPer', 'period', 'r', 'rel', 'sym' -- may be given
%   alongside, and a supplied value wins over the specs for every
%   attribute, so a sweep over any of them is one call per value and
%   leaves the pre-MAET untouched.
%
%   The function dispatches on the type of the first argument:
%     - numeric vector  -> single-multiset (single-attribute) path,
%                          canonicalised to the A = N = 1 corner of the
%                          multi-attribute build; returns a struct with
%                          tag = 'MaetDensity'
%     - cell array      -> multi-attribute path, returns struct with
%                          tag = 'MaetDensity'
%     - pre-MAET struct -> multi-attribute path, its parts and specs
%                          read from the struct
%
%   There is one density type. The single-multiset vector calling
%   convention is input canonicalisation: the collection becomes a
%   (K, 1) attribute and the scalar parameters become length-1 vectors.
%   The fast single-multiset kernels operate on this corner via
%   internal.singleMultisetView.
%
%   Inputs (single-multiset form, the A = N = 1 corner):
%     p         - Pitch or position values (vector of length N).
%     w         - Weights (vector of length N, empty, or scalar — see
%                 the toolbox's standard broadcast convention in
%                 User Guide §5).
%     sigma     - Standard deviation of the Gaussian kernel.
%     r         - Tuple size (positive integer; r >= 2 if isRel = true).
%     isRel     - If true, use transposition-invariant (relative)
%                 quadratic form (effective dim = r - 1).
%     isPer     - If true, wrap differences to periodic interval
%                 [-period/2, period/2).
%     period    - Period for periodic wrapping (e.g., 1200 for one
%                 octave in cents, or the cycle length for rhythmic
%                 analyses).
%
%   Inputs (multi-attribute path):
%     pAttr     - 1 x A cell array of K_a x N matrices (attribute positions).
%                 Shapes are honoured literally: a [K x 1] column is
%                 K atoms of one event, a [1 x N] row is one atom of N
%                 events, and a [K x N] matrix is K atoms of N events.
%                 No flattening is applied.
%     wAttr     - Weights. One of:
%                   []       -> all ones
%                   scalar   -> uniform value, broadcast to all attributes
%                   1 x A cell of per-attribute inputs
%                 Each per-attribute input is [], scalar, 1 x N row,
%                 K_a x 1 column, or K_a x N matrix; broadcasts to K_a x N.
%     sigmaVec  - 1 x A vector of per-attribute Gaussian widths
%     rVec      - 1 x A vector of per-attribute tuple sizes
%     isRelVec  - 1 x A logical vector of per-attribute isRel flags
%     isPerVec  - 1 x A logical vector of per-attribute periodic flags
%     periodVec - 1 x A vector of per-attribute periods (0 when not periodic)
%
%   Every attribute is self-contained, carrying its own geometry, so all
%   geometry vectors are per-attribute (length A); shared geometry is
%   expressed by repeating a value across the attributes that share it.
%
%   Output (multi-attribute path):
%     dens - struct with fields:
%       .tag           = 'MaetDensity'
%       .nAttrs        = A
%       .N             = number of events
%       .r             = 1 x A vector, per-attribute tuple size
%       .K             = 1 x A vector, max K_a (for reference)
%       .pAttr         = 1 x A cell, input value matrices
%       .w             = 1 x A cell, post-broadcast K_a x N weight matrices
%       .sigma         = 1 x G
%       .isRel         = 1 x G logical
%       .isPer         = 1 x G logical
%       .period        = 1 x G
%       .dim           = scalar total, sum_a (r_a - isRel_{g(a)})
%       .dimPerAttr    = 1 x A vector, r_a - isRel_{g(a)}
%       .nJ, .nK       = scalars, total perm-side / comb-side tuple counts
%       .Centres       = 1 x A cell, each (r_a - isRel_{g(a)}) x nJ
%       .U_perm        = 1 x A cell, each r_a x nJ (perm side)
%       .V_comb        = 1 x A cell, each r_a x nK (comb side)
%       .wJ            = 1 x nJ, per-tuple weight products (perm side)
%       .wv_comb       = 1 x nK, per-tuple weight products (comb side)
%       .eventOfJ      = 1 x nJ, event index for each perm-side tuple
%       .eventOfK      = 1 x nK, event index for each comb-side tuple
%
%   Column j in Centres{a}, U_perm{a}, wJ, and eventOfJ all refer to the
%   same global perm-side tuple. Similarly for column k across V_comb{a},
%   wv_comb, and eventOfK.
%
%   Optional name-value pairs (all calling conventions):
%     'verbose' — Logical (default: true). If false, suppresses console
%                 output (time estimates, progress messages).
%     'lazy'    — Logical (default: true). When true, the expensive
%                 density fields (U_perm, wJ, V_comb, wv_comb) are
%                 deferred until a consumer requests them. The Möbius
%                 method's centres-array footprint at high r would
%                 dominate memory if eagerly built; deferring lets
%                 calls that route via Möbius skip the centres array
%                 entirely. Pass 'lazy', false to materialise the
%                 expensive fields up front --- required by external
%                 code that reads U_perm / wJ directly without going
%                 through evalExpTens / cosSimExpTens / entropyExpTens.
%
%   See also preMaet, evalExpTens, cosSimExpTens, entropyExpTens.

    % ------------------------------------------------------------------
    % Parse optional name-value pairs and split positional args
    % ------------------------------------------------------------------
    varargin = internal.expandPreMaetPair(varargin);
    [posArgs, verbose, lazy, nested, specs, sigmaKw, isPerKw, periodKw, ...
     wrapKw, rKw, relKw, symKw] = localExtractKwargs(varargin);

    if isempty(posArgs)
        error('buildExpTens:missingInputs', ...
              'At least the pitch/attribute input is required.');
    end

    % --- Canonical specs form (level-structured geometry lives in specs;
    %     scalar sigma/isPer/period are supplied as name-value kwargs) ---
    if ~isempty(specs)
        if ~iscell(posArgs{1})
            error('buildExpTens:specsMultiOnly', ...
                  ['specs is only valid for multi-attribute calls (first ' ...
                   'argument a cell array of attribute matrices).']);
        end
        if numel(posArgs) ~= 2
            error('buildExpTens:specsPositional', ...
                  ['With specs, pass only (pAttr, wAttr) positionally; supply ' ...
                   'sigma, isPer, period as name-value kwargs (level-' ...
                   'structured r/rel/sym live in specs).']);
        end
        if ~isempty(nested)
            error('buildExpTens:specsAndNested', ...
                  'Pass nesting via specs, not nested.');
        end
        A = numel(posArgs{1});
        specs = internal.overrideSpecs(specs, rKw, relKw, symKw, A);
        % Resolve the kernel geometry first: a kernel covariance may live
        % in the spec as readily as in the keyword, and the flat-geometry
        % rewrite below is driven by whether one is present, so it cannot
        % be decided from the keyword alone.
        [~, ~, ~, ~, names, specKernel] = internal.normaliseSpecs(specs, A);
        % An explicit keyword wins outright and silently: sweeping sigma
        % over a grid while the specs hold a baseline is the ordinary
        % idiom, so a disagreement is intent, not error. What is refused
        % is a value missing from both places.
        sigmaKw = internal.resolveKernelParam(sigmaKw, specKernel.sigma, ...
                                     'sigma', names, A, false);
        isPerKw = internal.resolveKernelParam(isPerKw, specKernel.isPer, ...
                                     'isPer', names, A, false);
        periodKw = internal.resolveKernelParam(periodKw, specKernel.period, ...
                                      'period', names, A, true);
        hasKc = iscell(sigmaKw) && any(cellfun(@internal.isKernelCov, sigmaKw));
        if hasKc
            % Matrix-sigma attributes require flat geometry; degenerate
            % nested specs (e.g. from bindEvents on flat single-value
            % events) are order-isomorphic to flat ordered tuples and
            % are flattened here; non-degenerate nesting errors.
            specs = internal.resolveSpecsForKernelCov(specs, sigmaKw);
        end
        [rVec, isRelVec, isSymVec, nestedList, names] = ...
            internal.normaliseSpecs(specs, A);
        if hasKc
            [pW, sigmaNum, covList, cholList] = ...
                internal.resolveAnisoSigma(posArgs{1}, sigmaKw, rVec, ...
                    isRelVec, isPerKw, isSymVec, nestedList);
            synthArgs = {pW, posArgs{2}, sigmaNum, rVec, isRelVec, ...
                         isPerKw, periodKw, isSymVec};
            dens = localBuildMA(synthArgs, verbose, lazy, nestedList, ...
                                names, wrapKw);
            dens.kernelCov = covList;
            dens.kernelChol = cholList;
            return
        end
        synthArgs = {posArgs{1}, posArgs{2}, sigmaKw, rVec, isRelVec, ...
                     isPerKw, periodKw, isSymVec};
        if iscell(sigmaKw)
            % All-scalar cell sigma: accept, coerce to numeric
            % (mirrors the positional MA path; a cell reaching this
            % branch contains no matrix entries).
            synthArgs{3} = cellfun(@double, sigmaKw);
        end
        dens = localBuildMA(synthArgs, verbose, lazy, nestedList, names, ...
                            wrapKw);
        return
    end
    if ~isempty(sigmaKw) || ~isempty(isPerKw) || ~isempty(periodKw)
        error('buildExpTens:scalarKwargWithoutSpecs', ...
              ['sigma/isPer/period kwargs are only for the specs form; ' ...
               'the positional form takes them in order.']);
    end

    first = posArgs{1};

    if iscell(first)
        % Multi-attribute path. A sigma supplied as a cell array mixing
        % scalars and matrices carries per-attribute matrix-valued
        % kernel covariances: resolve (validate, whiten, sigma -> 1)
        % before the ordinary build, then attach the covariance
        % metadata to the returned density.
        if numel(posArgs) >= 3 && iscell(posArgs{3})
            sigmaArg = posArgs{3};
            if any(cellfun(@internal.isKernelCov, sigmaArg))
                if numel(posArgs) < 7
                    error('buildExpTens:maArgCount', ...
                          ['Multi-attribute call expects 7 or 8 ' ...
                           'positional arguments.']);
                end
                isSymArg = [];
                if numel(posArgs) >= 8, isSymArg = posArgs{8}; end
                [pW, sigmaNum, covList, cholList] = ...
                    internal.resolveAnisoSigma(posArgs{1}, sigmaArg, ...
                        posArgs{4}, posArgs{5}, posArgs{6}, isSymArg, ...
                        nested);
                posArgs{1} = pW;
                posArgs{3} = sigmaNum;
                dens = localBuildMA(posArgs, verbose, lazy, nested, ...
                                    {}, wrapKw);
                dens.kernelCov = covList;
                dens.kernelChol = cholList;
                return
            end
            % All-scalar cell sigma: accept, coerce to numeric.
            posArgs{3} = cellfun(@double, sigmaArg);
        end
        dens = localBuildMA(posArgs, verbose, lazy, nested, {}, wrapKw);
    elseif isnumeric(first)
        % Single-attribute legacy path
        if ~isempty(nested)
            error('buildExpTens:nestedSingleMultisetUnsupported', ...
                  ['nested is only valid for multi-attribute calls ' ...
                   '(first argument a cell array of attribute matrices).']);
        end
        if numel(posArgs) >= 3 && internal.isKernelCov(posArgs{3})
            if numel(posArgs) < 7
                error('buildExpTens:singleMultisetArgCount', ...
                      ['Single-multiset call expects 7 or 8 ' ...
                       'positional arguments.']);
            end
            isSymArg = [];
            if numel(posArgs) >= 8, isSymArg = posArgs{8}; end
            [pW, sigmaOne, Sigma, R] = internal.resolveAnisoSigma( ...
                posArgs{1}, posArgs{3}, posArgs{4}, posArgs{5}, ...
                posArgs{6}, isSymArg, []);
            posArgs{1} = pW;
            posArgs{3} = sigmaOne;
            dens = localBuildSingleMultiset(posArgs, verbose, lazy, ...
                                            wrapKw);
            % Store per-attribute (1-cell) to match the MaetDensity
            % convention; internal.singleMultisetView unwraps for the
            % flat single-multiset consumers.
            dens.kernelCov = {Sigma};
            dens.kernelChol = {R};
            return
        end
        dens = localBuildSingleMultiset(posArgs, verbose, lazy, wrapKw);
    else
        error('buildExpTens:badFirstArg', ...
              ['First argument must be a numeric vector (single-attribute) ' ...
               'or a cell array of attribute matrices (multi-attribute).']);
    end
end


% ======================================================================
%  Helpers: kwarg parsing
% ======================================================================

function [posArgs, verbose, lazy, nested, specs, sigmaKw, isPerKw, ...
          periodKw, wrapKw, rKw, relKw, symKw] = localExtractKwargs(args)
    verbose = true;
    lazy = true;  % default to skinny dens; eager via 'lazy', false
    nested = [];  % per-attribute nesting spec (cell), [] = all flat
    specs = [];   % canonical per-attribute level-geometry spec (cell)
    sigmaKw = []; isPerKw = []; periodKw = [];  % scalar geometry (specs form)
    wrapKw = [];  % abs-per / rel-per full-image vs single-image opt-in
    rKw = []; relKw = []; symKw = [];  % specs-form geometry overrides
    posArgs = args;
    i = 1;
    while i <= numel(posArgs)
        if (ischar(posArgs{i}) || isstring(posArgs{i})) ...
                && any(strcmpi(posArgs{i}, {'verbose', 'lazy', 'nested', ...
                                            'specs', 'sigma', 'isPer', ...
                                            'period', 'wrap', 'r', ...
                                            'rel', 'sym'}))
            key = lower(char(posArgs{i}));
            if i + 1 > numel(posArgs)
                error('buildExpTens:kwargMissingValue', ...
                      'Missing value for ''%s''.', key);
            end
            switch key
                case 'verbose'
                    verbose = logical(posArgs{i + 1});
                case 'lazy'
                    lazy = logical(posArgs{i + 1});
                case 'nested'
                    nested = posArgs{i + 1};
                case 'specs'
                    specs = posArgs{i + 1};
                case 'sigma'
                    sigmaKw = posArgs{i + 1};
                case 'isper'
                    isPerKw = posArgs{i + 1};
                case 'period'
                    periodKw = posArgs{i + 1};
                case 'wrap'
                    wrapKw = posArgs{i + 1};
                case 'r'
                    rKw = posArgs{i + 1};
                case 'rel'
                    relKw = posArgs{i + 1};
                case 'sym'
                    symKw = posArgs{i + 1};
            end
            posArgs(i:i + 1) = [];
        elseif ischar(posArgs{i}) || (isstring(posArgs{i}) && isscalar(posArgs{i}))
            % Every positional input is numeric or a cell, so a leftover
            % name is a misspelled or unsupported option, not a value.
            error('buildExpTens:unknownKwarg', ...
                  ['Unrecognised name-value argument ''%s''. Supported ' ...
                   'names are verbose, lazy, nested, specs, sigma, ' ...
                   'isPer, period, wrap, r, rel, and sym.'], ...
                  char(posArgs{i}));
        else
            i = i + 1;
        end
    end
end


% ======================================================================
%  Single-attribute (legacy) path
% ======================================================================

function dens = localBuildSingleMultiset(posArgs, verbose, lazy, wrap)
    if nargin < 4
        wrap = [];
    end
%LOCALBUILDSINGLEMULTISET  Vector-form build: canonicalise a single
%   weighted multiset to the A = N = 1 corner of the multi-attribute
%   build. There is one density type (MaetDensity); the vector calling
%   convention is pure input canonicalisation --- the collection becomes
%   a (K, 1) attribute matrix and the scalar parameters become length-1
%   vectors. Twin of Python _build_exp_tens_single_multiset.

    if numel(posArgs) == 7
        [p, w, sigma, r, isRel, isPer, period] = posArgs{:};
        isSym = true;
    elseif numel(posArgs) == 8
        [p, w, sigma, r, isRel, isPer, period, isSym] = posArgs{:};
    else
        error('buildExpTens:singleMultisetArgCount', ...
              ['Single-multiset call expects 7 or 8 positional ' ...
               'arguments: p, w, sigma, r, isRel, isPer, period[, isSym].']);
    end
    isSym = logical(isSym);

    p = p(:);
    w = w(:);

    % Degenerate empty collection (e.g. an all-dead pruning, or a window
    % that captures nothing): a valid zero-mass density with no tuples,
    % matching the historical vector-build behaviour. Constructed directly
    % because the general multi-attribute event validation (each event
    % needs at least r valid values) correctly rejects empty events in the
    % multi-event setting. Placed before the r > K validation so the
    % empty case is accepted rather than rejected. Twin of the K == 0
    % branch of Python _build_exp_tens_single_multiset.
    if isempty(p)
        isRel = logical(isRel);
        dim   = r - double(isRel);
        dens = struct();
        dens.tag        = 'MaetDensity';
        dens.nAttrs     = 1;
        dens.N          = 1;
        dens.r          = r;
        dens.K          = 0;
        dens.pAttr      = {zeros(0, 1)};
        dens.w          = {zeros(0, 1)};
        dens.sigma      = sigma;
        dens.isRel      = isRel;
        dens.isPer      = logical(isPer);
        dens.period     = period;
        dens.wrap       = internal.normaliseWrapMa(wrap, 1);
        internal.maybeWarnAbsPerSingleImage(sigma, isRel, isPer, period, ...
                                            dens.wrap);
        dens.isSym      = logical(isSym);
        dens.dim        = dim;
        dens.dimPerAttr = dim;
        dens.nested     = {[]};
        % Empty per-tuple fields (nJ = nK = 0), matching
        % localFillMAExpensive shapes so the density is fully materialised
        % and internal.ensureExpTensExpensive is a no-op on it.
        dens.nJ         = 0;
        dens.nK         = 0;
        dens.Centres    = {zeros(dim, 0)};
        dens.U_perm     = {zeros(r, 0)};
        dens.V_comb     = {zeros(r, 0)};
        dens.wJ         = zeros(1, 0);
        dens.wv_comb    = zeros(1, 0);
        dens.eventOfJ   = zeros(1, 0);
        dens.eventOfK   = zeros(1, 0);
        return;
    end

    if isempty(w)
        w = ones(numel(p), 1);
    end
    if isscalar(w)
        if w == 0
            warning('All weights in w are zero.');
        end
        w = w * ones(numel(p), 1);
    end

    % Historical single-multiset validation, enforced before
    % canonicalisation so callers keep the established messages.
    if rem(r, 1) || r < 1
        error('''r'' must be a positive integer.');
    elseif r > numel(p)
        error('''r'' must not exceed the number of values.');
    elseif numel(p) ~= numel(w)
        error('w must have the same number of entries as p.');
    end

    % r = 1 with isRel = true is a degenerate case: the relative
    % density is constant on a 0-dimensional space. The closed-form
    % total mass is sum(w), and entropyExpTens returns 0 by convention
    % for Rényi-2 in this regime, so we relax to a warning that
    % parallels the MA path's degenerate notice. Downstream consumers
    % that genuinely cannot handle dim = 0 (e.g. evalExpTens with a
    % query in a 0-D space) raise their own clearer errors.
    if isRel && r < 2
        warning('buildExpTens:isRelDegenerate', ...
                ['isRel = true with r = 1 produces a degenerate ' ...
                 '(constant) density. For cross-event translation ' ...
                 'invariance, use differenceEvents as a preprocessing ' ...
                 'step.']);
    end

    % For r = 1, the density depends on the source multiset only through
    % its measure on the pitch line: elements with equal pitch/position
    % contribute additively to the same Gaussian kernel, so they can be
    % collapsed to a single element whose weight is the sum of the
    % originals. This is mathematically exact at r = 1 and reduces
    % downstream work proportionally to the number of repeated pitches in
    % the input. For r >= 2, multiplicity in the source multiset matters
    % for the within-tuple structure, so collapsing would alter the density
    % and is therefore not applied.
    if r == 1 && ~isempty(p)
        [pUnique, ~, inverse] = unique(p);
        if numel(pUnique) < numel(p)
            wSummed = accumarray(inverse, w, [numel(pUnique), 1]);
            p = pUnique;
            w = wSummed;
        end
    end

    % Canonicalise to the A = N = 1 multi-attribute build: the collection
    % is a single flat attribute (K atoms, one event), scalar parameters
    % become length-1 vectors. Every consumer reads the resulting
    % MaetDensity either natively or through internal.singleMultisetView.
    wrapCell = internal.normaliseWrapMa(wrap, 1);
    maArgs = {{p}, {w}, sigma, r, isRel, isPer, period, isSym};
    dens = localBuildMA(maArgs, verbose, lazy, {[]}, {}, wrapCell);
end


% ======================================================================
%  Multi-attribute (MAET) path
% ======================================================================

function dens = localBuildMA(posArgs, verbose, lazy, nested, names, wrap)
    if nargin < 5
        names = {};
    end
    if nargin < 6
        wrap = [];
    end

    if numel(posArgs) == 7
        [pAttr, wIn, sigmaVec, rVec, isRelVec, isPerVec, periodVec] = posArgs{:};
        isSymVec = [];
    elseif numel(posArgs) == 8
        [pAttr, wIn, sigmaVec, rVec, isRelVec, isPerVec, periodVec, isSymVec] ...
            = posArgs{:};
    else
        error('buildExpTens:maArgCount', ...
              ['Multi-attribute call expects 7 or 8 positional arguments: ' ...
               'pAttr, wAttr, sigmaVec, rVec, isRelVec, isPerVec, ' ...
               'periodVec[, isSymVec].']);
    end

    % --- Input normalisation ---

    if ~iscell(pAttr) || isempty(pAttr)
        error('buildExpTens:badPAttr', ...
              'pAttr must be a non-empty cell array of attribute matrices.');
    end
    A = numel(pAttr);

    % Coerce each attribute input to its 2-D K_a x N shape. MATLAB
    % treats everything as at least 2-D, so a user-supplied column
    % vector [K x 1] is read as K atoms / 1 event, a row vector [1 x N]
    % as 1 atom / N events, and a matrix [K x N] as K atoms / N events,
    % with no ambiguity. (A bare scalar is 1 x 1 and fills the K=N=1
    % case.) No flattening is applied — flattening a 2-D input would
    % silently reinterpret columns as rows.
    for a = 1:A
        M = pAttr{a};
        if ~isnumeric(M)
            error('buildExpTens:badAttrType', ...
                  'Attribute %d input must be numeric.', a);
        end
        if ndims(M) > 2
            error('buildExpTens:badAttrDims', ...
                  'Attribute %d input must be at most 2-D; got ndims=%d.', ...
                  a, ndims(M));
        end
        pAttr{a} = double(M);
    end

    Ns = cellfun(@(M) size(M, 2), pAttr);
    if any(Ns ~= Ns(1))
        error('buildExpTens:eventCountMismatch', ...
              ['All attribute matrices must share the same number of ' ...
               'columns (events). Got: %s.'], mat2str(Ns));
    end
    N = Ns(1);

    Ka = cellfun(@(M) size(M, 1), pAttr);

    % r per attribute
    rVec = double(rVec(:).');
    if numel(rVec) ~= A
        error('buildExpTens:rLength', ...
              'rVec must have length equal to the number of attributes.');
    end

    % --- Nested attributes (representation B) ---------------------------
    % A nested attribute carries its level breakdown in nested{a} (a
    % struct with fields: tags (per-value source-event tag), r and sym
    % (per-level vectors, innermost-outward), and optional rel (the
    % co-transposition-unit selector: a per-level vector or
    % 'innermost'/'outermost')) and a flat K_total-value column;
    % rVec(a) is (re)derived to the total tuple dim D_a = prod(r) so
    % dim/allocation stay scalar. nested{a} = [] for an ordinary flat
    % attribute (unchanged path). Two-level only for now.
    if isempty(nested)
        nested = cell(1, A);
    else
        if ~iscell(nested) || numel(nested) ~= A
            error('buildExpTens:nestedLength', ...
                  'nested must be a 1 x %d cell array (one entry per attribute).', A);
        end
        nested = nested(:).';
    end
    % A spec that already carries the internal 'proj' field was produced by
    % a previous build (a rebuild via ensureExpTensExpensive forwards it),
    % not typed by a user; the user-isRel guard below is skipped for those
    % so the derived isRel value round-trips cleanly.
    nestedWasNorm = false(1, A);
    for a = 1:A
        spec = nested{a};
        if isempty(spec)
            continue
        end
        nestedWasNorm(a) = isstruct(spec) && isfield(spec, 'proj');
        if ~isstruct(spec)
            error('buildExpTens:nestedSpec', 'nested{%d} must be a struct.', a);
        end
        % Structural fields (no default): 'r' (per-level tuple size) and
        % 'tags' (value-to-level map). Everything else is optional and
        % defaults here, so a hand-edited spec can carry only the fields
        % being changed (unknown fields such as name/names/proj ride
        % through untouched via the struct copy).
        if ~isfield(spec, 'r')
            error('buildExpTens:nestedR', ...
                  ['nested{%d}: spec must have an ''r'' field (the per-level ' ...
                   'tuple-size vector); it is structural and has no default.'], a);
        end
        if ~isfield(spec, 'tags')
            error('buildExpTens:nestedTags', ...
                  ['nested{%d}: spec must have a ''tags'' field (the value-to-' ...
                   'level map); it is structural and has no default.'], a);
        end
        rLevels   = double(spec.r(:).');
        L = numel(rLevels);
        if isfield(spec, 'sym') && ~isempty(spec.sym)
            symLevels = logical(spec.sym(:).');
        else
            % Optional: default every level symmetric (flat sym=True default).
            symLevels = true(1, L);
        end
        if L < 2
            error('buildExpTens:nestedDepth', ...
                  ['nested{%d}: a nested spec needs L >= 2 levels; got ' ...
                   'L = %d. A single-level attribute is flat (no spec).'], a, L);
        end
        if numel(symLevels) ~= L
            error('buildExpTens:nestedSymLen', ...
                  'nested{%d}: sym must have length %d (one per level).', a, L);
        end
        if any(rLevels < 1) || any(rem(rLevels, 1) ~= 0)
            error('buildExpTens:nestedR', ...
                  'nested{%d}: all per-level r must be positive integers.', a);
        end
        % tags: a K_total x (L-1) integer matrix, one column per grouping
        % level innermost-outward (column 1 the finest grouping above the
        % leaf values, column L-1 the outermost). A vector is the single-
        % column (L = 2) case and is stored as a 1 x K_total row.
        rawTags = spec.tags;
        if isvector(rawTags)
            if L ~= 2
                error('buildExpTens:nestedTags', ...
                      ['nested{%d}: a tags vector is only valid for L = 2 ' ...
                       '(one grouping column); for L = %d supply a ' ...
                       '(K_total, L-1) = (%d, %d) tag matrix.'], ...
                      a, L, Ka(a), L - 1);
            end
            if numel(rawTags) ~= Ka(a)
                error('buildExpTens:nestedTags', ...
                      ['nested{%d}: tags length %d must equal K_total = %d ' ...
                       '(value count).'], a, numel(rawTags), Ka(a));
            end
            tags = double(rawTags(:).');           % 1 x K_total (L = 2)
        else
            if size(rawTags, 1) ~= Ka(a) || size(rawTags, 2) ~= L - 1
                error('buildExpTens:nestedTags', ...
                      ['nested{%d}: tags matrix is %d x %d but must be ' ...
                       '(K_total, L-1) = (%d, %d).'], ...
                      a, size(rawTags, 1), size(rawTags, 2), Ka(a), L - 1);
            end
            tags = double(rawTags);                % K_total x (L-1)
        end
        relRaw = [];
        if isfield(spec, 'rel')
            relRaw = spec.rel;
        end
        [relUnit, proj] = localCanonicaliseNestedRel(relRaw, L, a);
        spec.r       = rLevels;
        spec.sym     = symLevels;
        spec.tags    = tags;
        spec.relUnit = relUnit;
        spec.proj    = proj;
        nested{a}    = spec;
        rVec(a)      = prod(rLevels);   % total tuple dim D_a
    end

    if any(rVec < 1) || any(rem(rVec, 1) ~= 0)
        error('buildExpTens:rNotInt', 'All r_a must be positive integers.');
    end

    % Per-attribute parameters (every attribute is self-contained)
    sigmaVec  = double(sigmaVec(:).');
    isRelVec  = logical(isRelVec(:).');
    isPerVec  = logical(isPerVec(:).');
    periodVec = double(periodVec(:).');
    if numel(sigmaVec)  ~= A, error('buildExpTens:sigmaLength',  'sigmaVec must have length %d (nAttrs).',  A); end
    if numel(isRelVec)  ~= A, error('buildExpTens:isRelLength',  'isRelVec must have length %d (nAttrs).',  A); end
    if numel(isPerVec)  ~= A, error('buildExpTens:isPerLength',  'isPerVec must have length %d (nAttrs).',  A); end
    if numel(periodVec) ~= A, error('buildExpTens:periodLength', 'periodVec must have length %d (nAttrs).', A); end

    % isSym per attribute. Default (empty) is symmetric for every
    % attribute (legacy reading). Must otherwise have length A, matching
    % the other per-attribute parameter vectors.
    if isempty(isSymVec)
        isSymVec = true(1, A);
    else
        isSymVec = logical(isSymVec(:).');
        if numel(isSymVec) ~= A
            error('buildExpTens:isSymLength', ...
                  'isSymVec must have length %d (nAttrs).', A);
        end
    end

    % --- Collapse a vacuous inner nesting level to flat ----------------
    % A nested attribute whose inner level reads one value from each of
    % K_a singleton groups, with the outer level reading every group
    % (r = [1, K_a]), is mathematically a flat r = K_a attribute: the
    % inner level is the identity and the outer level forms the one full
    % K_a-tuple per event. Carried as a nested spec it adds a vacuous axis
    % to the density and routes the cosine through the general nested
    % contraction rather than the direct flat path. Dropping it gives the
    % same density and the same inner products (to floating-point floor)
    % and lets both the build and -- the larger cost -- the cosine take
    % the flat route. Periodicity rides through unchanged. Whole-tuple
    % co-transposition maps onto the flat isRel ('outer' -> relative,
    % 'absolute' -> absolute); an 'inner'/'intermediate' projection
    % reduces within sub-tuples and never collapses to flat.
    %
    % Restricted to the full-read case r_out == K_a. There each event
    % contributes exactly one tuple, so the flat path can never enumerate
    % a combinatorial set of sub-tuples: a partial read of singleton
    % groups (r_out < K_a) -- including every ragged attribute, whose
    % variable-length groups are padded to K_a and read with r_out < K_a
    % -- stays nested so the orbit contraction carries it. An attribute
    % whose isRelVec entry is already set is left nested so the check
    % below can reject setting [rel] outside the spec.
    for a = 1:A
        spec = nested{a};
        if isempty(spec)
            continue
        end
        if isRelVec(a)                      % reject below, do not mask
            continue
        end
        rLevels = spec.r(:).';
        if numel(rLevels) ~= 2 || rLevels(1) ~= 1
            continue
        end
        if rLevels(2) ~= Ka(a)              % not a full read of all values
            continue
        end
        if ~any(strcmp(spec.proj, {'absolute', 'outer'}))
            continue
        end
        tg = spec.tags;
        if ~isvector(tg) || numel(unique(tg(:))) ~= Ka(a)   % not all singletons
            continue
        end
        symLevels    = spec.sym(:).';
        nested{a}    = [];
        rVec(a)      = rLevels(2);
        isSymVec(a)  = symLevels(2);
        isRelVec(a)  = strcmp(spec.proj, 'outer');
    end

    % isRel + r_a = 1 degenerate warning (per attribute); nested mapping
    for a = 1:A
        if ~isempty(nested{a})
            if isRelVec(a) && ~nestedWasNorm(a)
                error('buildExpTens:nestedUserIsRel', ...
                      ['nested attribute %d: set the [rel] co-transposition ' ...
                       'unit via the nested spec''s rel field, not the ' ...
                       'isRelVec entry (leave it false for nested attributes).'], a);
            end
            % The outer / whole-tuple co-transposition unit is exactly the
            % flat isRel reduction on the whole D_a-tuple, so map it onto
            % the internal isRel machinery; absolute leaves it off.
            isRelVec(a) = strcmp(nested{a}.proj, 'outer');
            continue
        end
        if isRelVec(a) && rVec(a) < 2
            warning('buildExpTens:isRelDegenerate', ...
                    ['isRel = true combined with r_a = 1 for ' ...
                     'attribute %d produces a degenerate (constant) density. ' ...
                     'For cross-event translation invariance, use ' ...
                     'differenceEvents as a preprocessing step.'], a);
        end
    end

    % Weights: normalise to 1 x A cell of K_a x N matrices
    wCell = localNormaliseWeights(wIn, A, Ka, N);

    % --- Single-multiset collapse (MAET-base optimisation) ------------
    % A single flat (non-nested) attribute read at r = 1 is one pooled
    % multiset: a tuple is a lone atom, so which event an atom came from
    % is irrelevant and cross-event tuples never arise. Collapse the
    % events into one here, at the base, so every downstream consumer only
    % ever meets the canonical A = N = 1 form (no N > 1 single-multiset
    % case to special-case anywhere else). Equal values merge in the
    % per-event r = 1 path (localFillMAExpensive) exactly as for a
    % directly-built single multiset. Mirrors the Python collapse in
    % _build_exp_tens_ma.
    if A == 1 && rVec(1) == 1 && N > 1 && isempty(nested{1})
        P     = pAttr{1};
        W     = wCell{1};
        keep  = ~isnan(P);
        pVals = P(keep);
        wVals = W(keep);
        pAttr = {pVals(:)};
        wCell = {wVals(:)};
        N     = 1;
        Ka    = numel(pVals);
    end

    % --- Per-attribute dim profile (cheap; doesn't need tuple enumeration) ---

    dimPerAttr = zeros(1, A);
    for a = 1:A
        r_a = rVec(a);
        if ~isempty(nested{a}) && (strcmp(nested{a}.proj, 'inner') ...
                || strcmp(nested{a}.proj, 'intermediate'))
            % Co-transposition at unit u: D_a leaves split into
            % G_u = prod(r(u+1:end)) contiguous blocks of size
            % s_u = prod(r(1:u)); each block loses its own all-ones, so
            % dim = D_a - G_u = G_u * (s_u - 1).
            u   = nested{a}.relUnit;            % 1-based level
            s_u = prod(nested{a}.r(1:u));
            G_u = r_a / s_u;
            dimPerAttr(a) = r_a - G_u;
        elseif isRelVec(a)
            if r_a >= 2
                dimPerAttr(a) = r_a - 1;
            else
                dimPerAttr(a) = 0;  % degenerate (warned above)
            end
        else
            dimPerAttr(a) = r_a;
        end
    end
    dim = sum(dimPerAttr);

    % --- Eager input validation: each event must have enough non-NaN
    % values in every attribute. We check here (cheap) so that bad inputs
    % fail at buildExpTens time even when lazy=true. The full per-event
    % enumeration in localFillMAExpensive recomputes the valid index
    % vectors anyway, so this is just a guard.
    for n = 1:N
        for a = 1:A
            valCol = pAttr{a}(:, n);
            spec   = nested{a};
            if ~isempty(spec)
                rLv = spec.r(:).';
                tg  = spec.tags;
                if isvector(tg)
                    tg = tg(:);                      % K_total x 1 (L = 2)
                end
                validIdx = find(~isnan(valCol(:))).';   % 1 x Kv value indices
                if ~localNestedFeasible(validIdx, tg, rLv, numel(rLv))
                    error('buildExpTens:nestedInfeasible', ...
                          ['Event %d, nested attribute %d: the non-NaN ' ...
                           'values do not admit a full nested r-tuple for ' ...
                           'r = [%s] (too few groups or values at some ' ...
                           'nesting level).'], n, a, num2str(rLv));
                end
                continue
            end
            valid = ~isnan(valCol);
            K_na = sum(valid);
            r_a = rVec(a);
            if K_na < r_a
                error('buildExpTens:insufficientValues', ...
                      ['Event %d, attribute %d has %d non-NaN value(s) ' ...
                       'but r_a = %d.'], n, a, K_na, r_a);
            end
        end
    end

    % --- Pack skinny struct ---

    dens = struct();
    dens.tag          = 'MaetDensity';
    dens.nAttrs       = A;
    dens.N            = N;
    dens.r            = rVec;
    dens.K            = Ka;
    dens.pAttr        = pAttr;
    dens.w            = wCell;
    dens.sigma        = sigmaVec;
    dens.isRel        = isRelVec;
    dens.isPer        = isPerVec;
    dens.period       = periodVec;
    % Per-attribute wrap choice. 'full-image' (default) sums the kernel
    % over all periodic images (torus measure); 'single-image' evaluates
    % the nearest image only. Ignored for non-periodic attributes.
    dens.wrap = internal.normaliseWrapMa(wrap, A);
    internal.maybeWarnAbsPerSingleImage(sigmaVec, isRelVec, isPerVec, ...
                                        periodVec, dens.wrap);
    dens.isSym        = isSymVec;
    dens.dim          = dim;
    dens.dimPerAttr   = dimPerAttr;
    dens.nested       = nested;
    % Optional per-attribute user-defined names ([] where unnamed).
    % Attribute-indexed and prune-invariant; per-level names for a nested
    % attribute live inside nested{a}.
    if isempty(names)
        dens.names = cell(1, A);
    else
        dens.names = names;
    end

    if lazy
        if verbose
            fprintf(['buildExpTens (MAET): skinny density (%d attributes, ' ...
                     '%d events); per-tuple fields populated ' ...
                     'lazily on first consumer use.\n'], A, N);
        end
        return
    end

    % --- Populate expensive fields (eager mode) ---
    dens = localFillMAExpensive(dens, verbose);
end


function dens = localFillMAExpensive(dens, verbose)
%LOCALFILLMAEXPENSIVE  Populate per-tuple MA fields on a skinny dens.

    A           = dens.nAttrs;
    N           = dens.N;
    rVec        = dens.r;
    isRelVec    = dens.isRel;
    isSymVec    = dens.isSym;
    pAttr       = dens.pAttr;
    wCell       = dens.w;
    if isfield(dens, 'nested')
        nested = dens.nested;
    else
        nested = cell(1, A);
    end

    % --- All-r = 1, K = 1-per-event fast fill --------------------------
    % One value per event per attribute at r_a = 1 means every event's
    % tuple set is the event's own value: the perm and comb sides are
    % the input value matrices themselves, the per-tuple weight is the
    % product of the event's per-attribute weights, and the tuple index
    % is the event index. The general per-(n, a) enumeration below
    % computes exactly this through N x A rounds of small-array
    % machinery, which dominates the build cost of point-set-shaped
    % densities (one note per event); assembling the fields directly
    % produces identical arrays. NaN values (a dead event) fall through
    % to the general path, which raises the K < r error the fast path
    % must not silently bypass. Mirrors the Python _ma_build_perm_arrays
    % fast path.
    allR1K1 = all(rVec == 1) && all(cellfun(@isempty, nested));
    if allR1K1
        for a = 1:A
            if size(pAttr{a}, 1) ~= 1 || any(isnan(pAttr{a}(1, :)))
                allR1K1 = false;
                break;
            end
        end
    end
    if allR1K1
        Uc = cell(1, A);
        Vc = cell(1, A);
        Cc = cell(1, A);
        wJ = ones(1, N);
        for a = 1:A
            Uc{a} = double(pAttr{a});
            Vc{a} = Uc{a};
            wJ = wJ .* double(wCell{a}(1, :));
            if isRelVec(a)
                Cc{a} = zeros(0, N);
            else
                Cc{a} = Uc{a};
            end
        end
        dens.nJ       = N;
        dens.nK       = N;
        dens.Centres  = Cc;
        dens.U_perm   = Uc;
        dens.V_comb   = Vc;
        dens.wJ       = wJ;
        dens.wv_comb  = wJ;
        dens.eventOfJ = 1:N;
        dens.eventOfK = 1:N;
        return;
    end

    % --- Single flat attribute (A = 1) fast fill -----------------------
    % One attribute means nothing to Cartesian-product across attributes,
    % so the general per-(n,a) machinery below is overhead. Enumerate the
    % attribute directly (shared localEnumFlatAttr, so the tuples are
    % identical) and, for N > 1, concatenate the events. When the non-NaN
    % value pattern is the same every event and r >= 2 (the build has
    % already reduced any r = 1, N > 1 case to N = 1), the tuple-index
    % structure is event-invariant: compute it once and reuse it,
    % recomputing only the per-event positions and weights. Mirrors the
    % Python _ma_build_perm_arrays A = 1 fast path.
    if A == 1 && isempty(nested{1})
        r_a = rVec(1);
        P   = pAttr{1};
        W   = wCell{1};
        if N == 1
            valCol = P(:, 1);
            valid  = find(~isnan(valCol));
            if numel(valid) < r_a
                error('buildExpTens:insufficientValues', ...
                      ['Event %d, attribute %d has %d non-NaN value(s) ' ...
                       'but r_a = %d.'], 1, 1, numel(valid), r_a);
            end
            [permMat, combMat, wJ, wvComb] = ...
                localEnumFlatAttr(valCol, valid, r_a, isSymVec(1), W(:, 1));
            nJ = size(permMat, 2);
            nK = size(combMat, 2);
            U = reshape(valCol(permMat), r_a, nJ);
            V = reshape(valCol(combMat), r_a, nK);
            eventOfJ = ones(1, nJ);
            eventOfK = ones(1, nK);
        else
            valid0 = find(~isnan(P(:, 1)));
            reuse = r_a >= 2 && numel(valid0) >= r_a;
            if reuse
                for n = 2:N
                    if ~isequal(find(~isnan(P(:, n))), valid0)
                        reuse = false;
                        break;
                    end
                end
            end
            if reuse
                [permMat, combMat] = ...
                    localEnumFlatAttr(P(:, 1), valid0, r_a, isSymVec(1), W(:, 1));
                nje = size(permMat, 2);
                nke = size(combMat, 2);
                nJ = nje * N;
                nK = nke * N;
                U = zeros(r_a, nJ);
                V = zeros(r_a, nK);
                wJ = zeros(1, nJ);
                wvComb = zeros(1, nK);
                eventOfJ = zeros(1, nJ);
                eventOfK = zeros(1, nK);
                for n = 1:N
                    val = P(:, n);
                    wc  = W(:, n);
                    jj = (n - 1) * nje + 1 : n * nje;
                    kk = (n - 1) * nke + 1 : n * nke;
                    U(:, jj) = reshape(val(permMat), r_a, nje);
                    V(:, kk) = reshape(val(combMat), r_a, nke);
                    wJ(jj)     = prod(reshape(wc(permMat), r_a, []), 1);
                    wvComb(kk) = prod(reshape(wc(combMat), r_a, []), 1);
                    eventOfJ(jj) = n;
                    eventOfK(kk) = n;
                end
            else
                Ub = cell(1, N); Vb = cell(1, N);
                wJb = cell(1, N); wvb = cell(1, N);
                eojb = cell(1, N); eokb = cell(1, N);
                for n = 1:N
                    val    = P(:, n);
                    validn = find(~isnan(val));
                    if numel(validn) < r_a
                        error('buildExpTens:insufficientValues', ...
                              ['Event %d, attribute %d has %d non-NaN ' ...
                               'value(s) but r_a = %d.'], n, 1, ...
                              numel(validn), r_a);
                    end
                    [pm, cm, pw, cw] = ...
                        localEnumFlatAttr(val, validn, r_a, isSymVec(1), W(:, n));
                    Ub{n}   = reshape(val(pm), r_a, size(pm, 2));
                    Vb{n}   = reshape(val(cm), r_a, size(cm, 2));
                    wJb{n}  = pw;
                    wvb{n}  = cw;
                    eojb{n} = repmat(n, 1, size(pm, 2));
                    eokb{n} = repmat(n, 1, size(cm, 2));
                end
                U = [Ub{:}];
                V = [Vb{:}];
                wJ = [wJb{:}];
                wvComb = [wvb{:}];
                eventOfJ = [eojb{:}];
                eventOfK = [eokb{:}];
                nJ = size(U, 2);
                nK = size(V, 2);
            end
        end
        if isRelVec(1)
            if r_a >= 2
                C = U(2:r_a, :) - U(1, :);
            else
                C = zeros(0, nJ);
            end
        else
            C = U;
        end
        dens.nJ       = nJ;
        dens.nK       = nK;
        dens.Centres  = {C};
        dens.U_perm   = {U};
        dens.V_comb   = {V};
        dens.wJ       = wJ;
        dens.wv_comb  = wvComb;
        dens.eventOfJ = eventOfJ;
        dens.eventOfK = eventOfK;
        return;
    end

    % --- Per-event, per-attribute r-ad enumeration ---

    permIdx = cell(N, A);     % value indices, perm side: r_a x P_{n,a}
    combIdx = cell(N, A);     % value indices, comb side: r_a x C_{n,a}
    permW   = cell(N, A);     % per-tuple value weight products, perm side
    combW   = cell(N, A);     % per-tuple value weight products, comb side

    for n = 1:N
        for a = 1:A
            valCol  = pAttr{a}(:, n);
            valid   = find(~isnan(valCol));
            K_na    = numel(valid);

            % --- Nested attribute (representation B): tag-scoped two-level
            % enumeration. rVec(a) holds the total tuple dim D_a =
            % rInner * rOuter; the level breakdown and per-value source-
            % event tags live in nested{a}. Flat attributes (empty
            % nested{a}) take the original single-level path below.
            spec = nested{a};
            if ~isempty(spec)
                tg = spec.tags;
                if isvector(tg)
                    tg = tg(:);                       % K_total x 1
                end
                tagsValid = tg(valid, :);             % Kv x (L-1)
                [permMat, combMat] = localNestedEnumIndices( ...
                    valid(:).', tagsValid, spec.r(:).', spec.sym(:).');
                permIdx{n, a} = permMat;
                combIdx{n, a} = combMat;
                wCol = wCell{a}(:, n);
                D_a = prod(spec.r);
                permW{n, a} = prod(reshape(wCol(permMat), D_a, []), 1);
                combW{n, a} = prod(reshape(wCol(combMat), D_a, []), 1);
                continue
            end

            r_a     = rVec(a);
            if K_na < r_a
                error('buildExpTens:insufficientValues', ...
                      ['Event %d, attribute %d has %d non-NaN value(s) ' ...
                       'but r_a = %d.'], n, a, K_na, r_a);
            end

            [permIdx{n, a}, combIdx{n, a}, permW{n, a}, combW{n, a}] = ...
                localEnumFlatAttr(valCol, valid, r_a, isSymVec(a), ...
                                  wCell{a}(:, n));
        end
    end

    % --- Cartesian product within each event, concatenate across events ---

    nJ_n = zeros(1, N);
    nK_n = zeros(1, N);
    for n = 1:N
        szP = cellfun(@(M) size(M, 2), permIdx(n, :));
        szC = cellfun(@(M) size(M, 2), combIdx(n, :));
        nJ_n(n) = prod(szP);
        nK_n(n) = prod(szC);
    end
    nJ = sum(nJ_n);
    nK = sum(nK_n);

    if verbose
        fprintf(['buildExpTens (MAET): %d attributes, %d events. ' ...
                 'Total tuples: nJ = %d (perm), nK = %d (comb).\n'], ...
                A, N, nJ, nK);
    end

    U_perm = cell(1, A);
    V_comb = cell(1, A);
    for a = 1:A
        U_perm{a} = zeros(rVec(a), nJ);
        V_comb{a} = zeros(rVec(a), nK);
    end
    wJ        = ones(1, nJ);
    wv_comb   = ones(1, nK);
    eventOfJ  = zeros(1, nJ);
    eventOfK  = zeros(1, nK);

    offJ = 0;
    offK = 0;
    for n = 1:N
        nJh = nJ_n(n);
        nKh = nK_n(n);

        szP = cellfun(@(M) size(M, 2), permIdx(n, :));
        szC = cellfun(@(M) size(M, 2), combIdx(n, :));

        idxPerm = localCartesianIndices(szP);
        idxComb = localCartesianIndices(szC);

        wJh = ones(1, nJh);
        wKh = ones(1, nKh);

        for a = 1:A
            r_a = rVec(a);
            valCol = pAttr{a}(:, n);

            valPerm = permIdx{n, a}(:, idxPerm{a});
            U_perm{a}(:, offJ + 1 : offJ + nJh) = reshape(valCol(valPerm), r_a, nJh);
            wJh = wJh .* permW{n, a}(idxPerm{a});

            valComb = combIdx{n, a}(:, idxComb{a});
            V_comb{a}(:, offK + 1 : offK + nKh) = reshape(valCol(valComb), r_a, nKh);
            wKh = wKh .* combW{n, a}(idxComb{a});
        end

        wJ(offJ + 1 : offJ + nJh)        = wJh;
        wv_comb(offK + 1 : offK + nKh)   = wKh;
        eventOfJ(offJ + 1 : offJ + nJh)  = n;
        eventOfK(offK + 1 : offK + nKh)  = n;

        offJ = offJ + nJh;
        offK = offK + nKh;
    end

    % --- Centres (per-attribute isRel reduction) ---

    Centres = cell(1, A);
    for a = 1:A
        r_a = rVec(a);
        spec = nested{a};
        if ~isempty(spec) && (strcmp(spec.proj, 'inner') ...
                || strcmp(spec.proj, 'intermediate'))
            % Co-transposition at unit u: the leaves split into G_u
            % contiguous blocks of size s_u = prod(r(1:u)) (the depth-first
            % enumeration lays each level-u sub-tuple out contiguously).
            % Reduce each block by its own first value (per-block interval
            % space), then stack. At u = 1 this is the per-event inner
            % reduction; for an intermediate u a per-intermediate-group one.
            u   = spec.relUnit;
            s_u = prod(spec.r(1:u));
            G_u = r_a / s_u;
            if s_u >= 2
                blocks = cell(1, G_u);
                for b = 1:G_u
                    base = (b - 1) * s_u;
                    blocks{b} = U_perm{a}(base + 2:base + s_u, :) ...
                              - U_perm{a}(base + 1, :);
                end
                Centres{a} = vertcat(blocks{:});
            else
                Centres{a} = zeros(0, nJ);
            end
        elseif isRelVec(a)
            if r_a >= 2
                Centres{a} = U_perm{a}(2:r_a, :) - U_perm{a}(1, :);
            else
                Centres{a} = zeros(0, nJ);
            end
        else
            Centres{a} = U_perm{a};
        end
    end

    % --- Append heavy fields ---

    dens.nJ       = nJ;
    dens.nK       = nK;
    dens.Centres  = Centres;
    dens.U_perm   = U_perm;
    dens.V_comb   = V_comb;
    dens.wJ       = wJ;
    dens.wv_comb  = wv_comb;
    dens.eventOfJ = eventOfJ;
    dens.eventOfK = eventOfK;
end


function [permMat, combMat, permW, combW] = ...
        localEnumFlatAttr(valCol, valid, r_a, isSym, wColOrig)
%LOCALENUMFLATATTR  Delegates to the shared internal.enumFlatAttr so the
%   build's per-(n, a) fill loop and evalExpTens's factored centres path
%   enumerate identical tuples from one source. See internal.enumFlatAttr.
    [permMat, combMat, permW, combW] = ...
        internal.enumFlatAttr(valCol, valid, r_a, isSym, wColOrig);
end


% ======================================================================
%  Helpers: weight normalisation, Cartesian index
% ======================================================================

function wCell = localNormaliseWeights(wIn, A, Ka, N)
    % Top-level normalisation: [] / scalar / cell of per-attribute inputs.
    if isempty(wIn) && ~iscell(wIn)
        wIn = 1;  % unified downstream: scalar one, then broadcast
    end

    if isnumeric(wIn) && isscalar(wIn)
        if wIn == 0
            warning('buildExpTens:zeroWeights', 'All weights are zero.');
        end
        wCell = cell(1, A);
        for a = 1:A
            wCell{a} = wIn * ones(Ka(a), N);
        end
        return;
    end

    if iscell(wIn)
        if numel(wIn) ~= A
            error('buildExpTens:weightCellLength', ...
                  'Weight cell array must have length equal to the number of attributes (%d).', A);
        end
        wCell = cell(1, A);
        for a = 1:A
            wCell{a} = localBroadcastWeight(wIn{a}, Ka(a), N, a);
        end
        return;
    end

    error('buildExpTens:badWeightsType', ...
          ['Top-level weight argument must be [], a scalar, or a cell ' ...
           'array of per-attribute inputs.']);
end


function Wab = localBroadcastWeight(w, Ka, N, attrIdx)
    % Broadcast a per-attribute weight input to a full Ka x N matrix.
    if isempty(w)
        Wab = ones(Ka, N);
        return;
    end
    if ~isnumeric(w)
        error('buildExpTens:badPerAttrWeightType', ...
              'Attribute %d weight input must be numeric.', attrIdx);
    end

    w = double(w);

    if isscalar(w)
        Wab = w * ones(Ka, N);
        return;
    end

    sz = size(w);
    % Row vector 1 x N -> broadcast across values
    if sz(1) == 1 && sz(2) == N
        Wab = repmat(w, Ka, 1);
        return;
    end
    % Column vector Ka x 1 -> broadcast across events
    if sz(1) == Ka && sz(2) == 1
        Wab = repmat(w, 1, N);
        return;
    end
    % Full matrix Ka x N -> as-is
    if sz(1) == Ka && sz(2) == N
        Wab = w;
        return;
    end

    error('buildExpTens:badPerAttrWeightShape', ...
          ['Attribute %d weight input has shape [%d %d]; expected [], ' ...
           'scalar, [1 %d], [%d 1], or [%d %d].'], ...
          attrIdx, sz(1), sz(2), N, Ka, Ka, N);
end


function idxCell = localCartesianIndices(sizes)
    %LOCALCARTESIANINDICES  Column-major Cartesian-product indices.
    %   For a Cartesian product of axes with sizes sizes(1), ..., sizes(A),
    %   returns a 1 x A cell, each cell a 1 x prod(sizes) row vector giving
    %   the index along the corresponding axis. First axis varies fastest
    %   (MATLAB column-major ndgrid convention).
    A = numel(sizes);
    idxCell = cell(1, A);
    for a = 1:A
        repInner = prod(sizes(1:a - 1));   % consecutive repeats of each atom
        repOuter = prod(sizes(a + 1:end));  % tiles of the full cycle
        row = 1:sizes(a);
        if repInner > 1
            row = kron(row, ones(1, repInner));
        end
        if repOuter > 1
            row = repmat(row, 1, repOuter);
        end
        idxCell{a} = row;
    end
end

function [permIdx, combIdx] = localNestedEnumIndices( ...
        validValues, tagsValid, rLevels, symLevels)
    %LOCALNESTEDENUMINDICES  Delegates to the shared
    %   internal.nestedEnumIndices so the build's nested fill loop and
    %   evalExpTens's factored centres path enumerate identical nested
    %   tuples from one source. See internal.nestedEnumIndices.
    [permIdx, combIdx] = internal.nestedEnumIndices( ...
        validValues, tagsValid, rLevels, symLevels);
end


function tf = localNestedFeasible(vals, tagsMat, rLevels, level)
    %LOCALNESTEDFEASIBLE  Whether `vals` admit a full level-`level` nested
    %   r-tuple. Recurses outermost-inward through tagsMat (K_total x (L-1),
    %   indexed by absolute value index), mirroring localNestedEnumIndices:
    %   enough distinct groups at each grouping level (each recursively
    %   feasible) and enough leaf values in the finest groups.
    vals = vals(:).';
    if level == 1
        tf = numel(vals) >= rLevels(1);
        return
    end
    col = level - 1;
    gids = tagsMat(vals, col).';
    ug = unique(gids);
    need = rLevels(level);
    feasible = 0;
    for gi = 1:numel(ug)
        sub = vals(gids == ug(gi));
        if localNestedFeasible(sub, tagsMat, rLevels, level - 1)
            feasible = feasible + 1;
            if feasible >= need
                tf = true;
                return
            end
        end
    end
    tf = feasible >= need;
end


function [relUnit, proj] = localCanonicaliseNestedRel(rel, L, a)
    %LOCALCANONICALISENESTEDREL  Resolve a nested attribute's [rel] selector.
    %   Returns relUnit (NaN = absolute, or a 1-based level index,
    %   innermost-outward, of the finest selected co-transposition unit)
    %   and proj ('absolute' | 'inner' | 'outer' | 'intermediate').
    %   [rel] carries subsumption: a finer (lower-index) unit subsumes
    %   every coarser one, so the finest 1 wins (extra 1s warn). Strings
    %   'innermost' -> level 1 and 'outermost' -> level L are depth-proof.
    %   A bare scalar/bool is rejected for a nested attribute (L > 1);
    %   [] (absent) means absolute.
    if isempty(rel)
        relUnit = NaN; proj = 'absolute'; return
    end
    if ischar(rel) || isstring(rel)
        key = lower(char(rel));
        switch key
            case 'innermost'
                unit = 1;
            case 'outermost'
                unit = L;
            otherwise
                error('buildExpTens:nestedRelString', ...
                      ['nested attribute %d: [rel] string must be ' ...
                       '''innermost'' or ''outermost''.'], a);
        end
    elseif isscalar(rel)
        error('buildExpTens:nestedRelScalar', ...
              ['nested attribute %d: [rel] must be a length-%d per-level ' ...
               'vector or ''innermost''/''outermost''; a scalar/bool is not ' ...
               'allowed for a nested attribute (it is ambiguous about which ' ...
               'co-transposition unit is meant).'], a, L);
    else
        v = logical(rel(:).');
        if numel(v) ~= L
            error('buildExpTens:nestedRelLength', ...
                  ['nested attribute %d: [rel] vector must have length %d ' ...
                   '(one per nesting level).'], a, L);
        end
        onesIdx = find(v);
        if isempty(onesIdx)
            relUnit = NaN; proj = 'absolute'; return
        end
        if numel(onesIdx) > 1
            warning('buildExpTens:nestedRelSubsumption', ...
                    ['nested attribute %d: multiple [rel] levels set; a finer ' ...
                     'co-transposition unit subsumes every coarser one, so the ' ...
                     'innermost (level %d) is used and the rest are redundant.'], ...
                    a, onesIdx(1));
        end
        unit = onesIdx(1);
    end
    relUnit = unit;
    if unit == 1
        proj = 'inner';
    elseif unit == L
        proj = 'outer';
    else
        proj = 'intermediate';
    end
end

function dens = buildExpTens(varargin)
%BUILDEXPTENS Precompute an r-ad expectation tensor density object.
%
%   SINGLE-ATTRIBUTE (legacy):
%     dens = buildExpTens(p, w, sigma, r, isRel, isPer, period)
%     dens = buildExpTens(..., 'verbose', false)
%
%   MULTI-ATTRIBUTE (MAET):
%     dens = buildExpTens(pAttr, w, sigmaVec, rVec, ...
%                         isRelVec, isPerVec, periodVec)
%     dens = buildExpTens(..., 'verbose', false)
%
%   The function dispatches on the type of the first argument:
%     - numeric vector  -> single-attribute path, returns struct with
%                          tag = 'ExpTensDensity'
%     - cell array      -> multi-attribute path, returns struct with
%                          tag = 'MaetDensity'
%
%   Inputs (single-attribute path):
%     p         - Pitch or position values (vector of length N).
%     w         - Weights (vector of length N, empty, or scalar — see
%                 the toolbox's standard broadcast convention in
%                 User Guide §4).
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
%     pAttr     - 1 x A cell array of K_a x N matrices (attribute values).
%                 Shapes are honoured literally: a [K x 1] column is
%                 K slots of one event, a [1 x N] row is one slot of N
%                 events, and a [K x N] matrix is K slots of N events.
%                 No flattening is applied.
%     w         - Weights. One of:
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
%   See also evalExpTens, cosSimExpTens.

    % ------------------------------------------------------------------
    % Parse optional name-value pairs and split positional args
    % ------------------------------------------------------------------
    [posArgs, verbose, lazy, nested] = localExtractKwargs(varargin);

    if isempty(posArgs)
        error('buildExpTens:missingInputs', ...
              'At least the pitch/attribute input is required.');
    end

    first = posArgs{1};

    if iscell(first)
        % Multi-attribute path
        dens = localBuildMA(posArgs, verbose, lazy, nested);
    elseif isnumeric(first)
        % Single-attribute legacy path
        if ~isempty(nested)
            error('buildExpTens:nestedSAUnsupported', ...
                  ['nested is only valid for multi-attribute calls ' ...
                   '(first argument a cell array of attribute matrices).']);
        end
        dens = localBuildSA(posArgs, verbose, lazy);
    else
        error('buildExpTens:badFirstArg', ...
              ['First argument must be a numeric vector (single-attribute) ' ...
               'or a cell array of attribute matrices (multi-attribute).']);
    end
end


% ======================================================================
%  Helpers: kwarg parsing
% ======================================================================

function [posArgs, verbose, lazy, nested] = localExtractKwargs(args)
    verbose = true;
    lazy = true;  % default to skinny dens; eager via 'lazy', false
    nested = [];  % per-attribute nesting spec (cell), [] = all flat
    posArgs = args;
    i = 1;
    while i <= numel(posArgs)
        if (ischar(posArgs{i}) || isstring(posArgs{i})) ...
                && any(strcmpi(posArgs{i}, {'verbose', 'lazy', 'nested'}))
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
            end
            posArgs(i:i + 1) = [];
        else
            i = i + 1;
        end
    end
end


% ======================================================================
%  Single-attribute (legacy) path
% ======================================================================

function dens = localBuildSA(posArgs, verbose, lazy)

    if numel(posArgs) == 7
        [p, w, sigma, r, isRel, isPer, period] = posArgs{:};
        isSym = true;
    elseif numel(posArgs) == 8
        [p, w, sigma, r, isRel, isPer, period, isSym] = posArgs{:};
    else
        error('buildExpTens:saArgCount', ...
              ['Single-attribute call expects 7 or 8 positional ' ...
               'arguments: p, w, sigma, r, isRel, isPer, period[, isSym].']);
    end
    isSym = logical(isSym);

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

    dim = r - isRel;

    % --- Pack skinny struct (cheap fields only) ---
    dens = struct();
    dens.tag    = 'ExpTensDensity';
    dens.p      = p;
    dens.w      = w;
    dens.sigma  = sigma;
    dens.r      = r;
    dens.isRel  = isRel;
    dens.isPer  = isPer;
    dens.period = period;
    dens.isSym  = isSym;
    dens.dim    = dim;

    if lazy
        if verbose
            fprintf(['buildExpTens: skinny density (%d values, r = %d); ' ...
                     'per-tuple fields populated lazily on first consumer use.\n'], ...
                    numel(p), r);
        end
        return
    end

    % --- Populate expensive fields (eager mode) ---
    dens = localFillSAExpensive(dens, verbose);
end


function dens = localFillSAExpensive(dens, verbose)
%LOCALFILLSAEXPENSIVE  Populate per-tuple SA fields on a skinny dens.

    p     = dens.p;
    w     = dens.w;
    r     = dens.r;
    isRel = dens.isRel;
    isSym = dens.isSym;

    n      = numel(p);
    nCombs = nchoosek(n, r);

    % isSym = true (default): symmetrise each combination over its full
    % S_r orbit (the perm side has r! copies). isSym = false: keep each
    % combination in listed order, so the perm side equals the comb side
    % (the de-reflected, ordered density). r = 1 has no order to
    % symmetrise, so perms(1) gives the single identity either way.
    if isSym
        allPerms = perms(1:r)';
    else
        allPerms = (1:r)';   % identity only
    end
    nPerms = size(allPerms, 2);
    nJ     = nPerms * nCombs;
    nK     = nCombs;

    if verbose
        fprintf('buildExpTens: building %d ordered %d-tuples from %d values.\n', ...
            nJ, r, n);
    end

    nck      = nchoosek(1:numel(p), r)';

    Ju     = zeros(r, nJ);
    offset = 0;
    for i = 1:nPerms
        Ju(:, offset + 1 : offset + nCombs) = nck(allPerms(:, i), :);
        offset = offset + nCombs;
    end

    U_perm = reshape(p(Ju), r, nJ);
    w_perm = reshape(prod(reshape(w(Ju), r, nJ), 1), 1, nJ);

    Kv      = nck;
    V_comb  = reshape(p(Kv), r, nK);
    wv_comb = reshape(prod(reshape(w(Kv), r, nK), 1), 1, nK);

    if isRel
        Centres = U_perm(2:r, :) - U_perm(1, :);
    else
        Centres = U_perm;
    end

    dens.Centres = Centres;
    dens.wJ      = w_perm;
    dens.nJ      = nJ;

    dens.U_perm  = U_perm;
    dens.w_perm  = w_perm;
    dens.nJ_perm = nJ;
    dens.V_comb  = V_comb;
    dens.wv_comb = wv_comb;
    dens.nK      = nK;
end


% ======================================================================
%  Multi-attribute (MAET) path
% ======================================================================

function dens = localBuildMA(posArgs, verbose, lazy, nested)

    if numel(posArgs) == 7
        [pAttr, wIn, sigmaVec, rVec, isRelVec, isPerVec, periodVec] = posArgs{:};
        isSymVec = [];
    elseif numel(posArgs) == 8
        [pAttr, wIn, sigmaVec, rVec, isRelVec, isPerVec, periodVec, isSymVec] ...
            = posArgs{:};
    else
        error('buildExpTens:maArgCount', ...
              ['Multi-attribute call expects 7 or 8 positional arguments: ' ...
               'pAttr, w, sigmaVec, rVec, isRelVec, isPerVec, ' ...
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
    % vector [K x 1] is read as K slots / 1 event, a row vector [1 x N]
    % as 1 slot / N events, and a matrix [K x N] as K slots / N events,
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
    % struct with fields: tags (per-slot source-event tag), r and sym
    % (per-level vectors, innermost-outward), and optional rel (the
    % co-transposition-unit selector: a per-level vector or
    % 'innermost'/'outermost')) and a flat K_total-slot value column;
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
        if ~isstruct(spec) || ~all(isfield(spec, {'r', 'sym', 'tags'}))
            error('buildExpTens:nestedSpec', ...
                  ['nested{%d} must be a struct with fields r, sym, tags ' ...
                   '(and optional rel).'], a);
        end
        rLevels   = double(spec.r(:).');
        symLevels = logical(spec.sym(:).');
        L = numel(rLevels);
        if L ~= 2
            error('buildExpTens:nestedDepth', ...
                  ['nested{%d}: only two-level nesting (L = 2) is supported ' ...
                   'for now; got L = %d.'], a, L);
        end
        if numel(symLevels) ~= L
            error('buildExpTens:nestedSymLen', ...
                  'nested{%d}: sym must have length %d (one per level).', a, L);
        end
        if any(rLevels < 1) || any(rem(rLevels, 1) ~= 0)
            error('buildExpTens:nestedR', ...
                  'nested{%d}: all per-level r must be positive integers.', a);
        end
        tags = double(spec.tags(:).');
        if numel(tags) ~= Ka(a)
            error('buildExpTens:nestedTags', ...
                  ['nested{%d}: tags length %d must equal K_total = %d ' ...
                   '(slot count).'], a, numel(tags), Ka(a));
        end
        relRaw = [];
        if isfield(spec, 'rel')
            relRaw = spec.rel;
        end
        [relUnit, proj] = localCanonicaliseNestedRel(relRaw, L, a);
        if strcmp(proj, 'intermediate')
            error('buildExpTens:nestedRelIntermediate', ...
                  ['nested{%d}: an intermediate [rel] co-transposition unit ' ...
                   'needs L > 2 nesting, which is a later step. Use the ' ...
                   'innermost or outermost unit, or absolute.'], a);
        end
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

    % --- Per-attribute dim profile (cheap; doesn't need tuple enumeration) ---

    dimPerAttr = zeros(1, A);
    for a = 1:A
        r_a = rVec(a);
        if ~isempty(nested{a}) && strcmp(nested{a}.proj, 'inner')
            % Inner unit: r_outer blocks each reduced to (r_inner - 1).
            dimPerAttr(a) = nested{a}.r(2) * (nested{a}.r(1) - 1);
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
    % slots in every attribute. We check here (cheap) so that bad inputs
    % fail at buildExpTens time even when lazy=true. The full per-event
    % enumeration in localFillMAExpensive recomputes the valid index
    % vectors anyway, so this is just a guard.
    for n = 1:N
        for a = 1:A
            valCol = pAttr{a}(:, n);
            spec   = nested{a};
            if ~isempty(spec)
                tags     = spec.tags(:).';          % 1 x K_total row
                validRow = (~isnan(valCol(:))).';   % 1 x K_total row
                ri   = spec.r(1);                   % innermost
                ro   = spec.r(2);                   % outermost
                present = unique(tags(validRow));
                present = present(:).';             % force row for the loop
                good = 0;
                for t = present
                    if sum(validRow & (tags == t)) >= ri
                        good = good + 1;
                    end
                end
                if good < ro
                    error('buildExpTens:nestedInsufficientTags', ...
                          ['Event %d, nested attribute %d: only %d ' ...
                           'source-event(s) have >= rInner = %d non-NaN ' ...
                           'slot(s), but rOuter = %d.'], n, a, good, ri, ro);
                end
                continue
            end
            valid = ~isnan(valCol);
            K_na = sum(valid);
            r_a = rVec(a);
            if K_na < r_a
                error('buildExpTens:insufficientSlots', ...
                      ['Event %d, attribute %d has %d non-NaN slot(s) ' ...
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
    dens.isSym        = isSymVec;
    dens.dim          = dim;
    dens.dimPerAttr   = dimPerAttr;
    dens.nested       = nested;

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

    % --- Per-event, per-attribute r-ad enumeration ---

    permIdx = cell(N, A);     % slot indices, perm side: r_a x P_{n,a}
    combIdx = cell(N, A);     % slot indices, comb side: r_a x C_{n,a}
    permW   = cell(N, A);     % per-tuple slot weight products, perm side
    combW   = cell(N, A);     % per-tuple slot weight products, comb side

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
                tagsValid = spec.tags(valid);
                [permMat, combMat] = localNestedEnumIndices( ...
                    valid(:).', tagsValid(:).', ...
                    spec.r(1), spec.r(2), spec.sym(1), spec.sym(2));
                permIdx{n, a} = permMat;
                combIdx{n, a} = combMat;
                wCol = wCell{a}(:, n);
                D_a = spec.r(1) * spec.r(2);
                permW{n, a} = prod(reshape(wCol(permMat), D_a, []), 1);
                combW{n, a} = prod(reshape(wCol(combMat), D_a, []), 1);
                continue
            end

            r_a     = rVec(a);
            if K_na < r_a
                error('buildExpTens:insufficientSlots', ...
                      ['Event %d, attribute %d has %d non-NaN slot(s) ' ...
                       'but r_a = %d.'], n, a, K_na, r_a);
            end

            % For attributes with r_a = 1, equal-valued slots within the
            % same event are exchangeable and can be collapsed (see SA
            % path comment for full rationale).
            collapsed = false;
            wColOrig = wCell{a}(:, n);
            if r_a == 1 && K_na > 1
                valsValid = valCol(valid);
                [uniqueVals, firstIdx, inverse] = ...
                    unique(valsValid, 'first');
                if numel(firstIdx) < K_na
                    wColLocal = wColOrig;
                    summed = accumarray(inverse, wColOrig(valid), ...
                                         [numel(uniqueVals), 1]);
                    wColLocal(valid(firstIdx)) = summed;
                    valid = valid(firstIdx);
                    K_na  = numel(valid);
                    collapsed = true;
                end
            end
            if ~collapsed
                wColLocal = wColOrig;
            end

            % Combinations: r_a x C(K_na, r_a)
            if K_na == r_a
                combMat = valid(:);
            else
                combMat = nchoosek(valid, r_a).';
            end

            % Permutations: r_a x (r_a! * C(K_na, r_a)) when symmetric.
            % An ordered attribute (isSym = false) keeps each combination
            % in listed order, so the perm side equals the comb side.
            % r_a = 1 has no order to symmetrise either way.
            if r_a == 1 || ~isSymVec(a)
                permMat = combMat;
            else
                Pm = perms(1:r_a).';
                nC = size(combMat, 2);
                nP = size(Pm, 2);
                permMat = zeros(r_a, nC * nP);
                for pp = 1:nP
                    permMat(:, (pp - 1) * nC + 1 : pp * nC) = combMat(Pm(:, pp), :);
                end
            end

            permIdx{n, a} = permMat;
            combIdx{n, a} = combMat;

            % Slot-weight products (per-tuple).
            wCol = wColLocal;
            if r_a == 1
                permW{n, a} = reshape(wCol(permMat), 1, []);
                combW{n, a} = reshape(wCol(combMat), 1, []);
            else
                permW{n, a} = prod(reshape(wCol(permMat), r_a, []), 1);
                combW{n, a} = prod(reshape(wCol(combMat), r_a, []), 1);
            end
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

            slotPerm = permIdx{n, a}(:, idxPerm{a});
            U_perm{a}(:, offJ + 1 : offJ + nJh) = reshape(valCol(slotPerm), r_a, nJh);
            wJh = wJh .* permW{n, a}(idxPerm{a});

            slotComb = combIdx{n, a}(:, idxComb{a});
            V_comb{a}(:, offK + 1 : offK + nKh) = reshape(valCol(slotComb), r_a, nKh);
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
        if ~isempty(spec) && strcmp(spec.proj, 'inner')
            % Inner unit: reduce each r_outer event-block independently by
            % subtracting its own first slot, then stack (tensor-joined).
            rIn  = spec.r(1);
            rOut = spec.r(2);
            if rIn >= 2
                blocks = cell(1, rOut);
                for b = 1:rOut
                    base = (b - 1) * rIn;
                    blocks{b} = U_perm{a}(base + 2:base + rIn, :) ...
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
    % Row vector 1 x N -> broadcast across slots
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
        repInner = prod(sizes(1:a - 1));   % consecutive repeats of each value
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
        validSlots, tagsValid, rInner, rOuter, symInner, symOuter)
    %LOCALNESTEDENUMINDICES  Tag-scoped nested r-tuple enumeration (rep. B).
    %   Two-level enumeration for one output-event of a nested attribute.
    %   validSlots : 1 x Kv slot indices (into the attribute's K_total
    %                axis) that are non-NaN for this event.
    %   tagsValid  : 1 x Kv source-event-in-window tag per valid slot.
    %   Returns permIdx, combIdx: each D x M slot-index arrays,
    %   D = rInner * rOuter. permIdx is the symmetrised deposit (the
    %   density's kernel centres): inner S_{rInner} orbit per chosen
    %   event when symInner; outer listed order (or full orbit when
    %   symOuter). combIdx is the canonical one-per-combination side
    %   (inner combinations, outer listed) used for inner-product
    %   pairing. Columns are concatenated in (outer-order, inner-order).

    uniq = unique(tagsValid);            % sorted ascending
    L = numel(uniq);
    slotsOf = cell(1, L);
    for ti = 1:L
        slotsOf{ti} = validSlots(tagsValid == uniq(ti));
    end

    % Outer selections of rOuter distinct tags (rows of tag-indices into
    % uniq). symOuter = 0 keeps sequence order (combinations); symOuter =
    % 1 pools as an unordered bag (full orbit).
    if rOuter > L
        outerComb = zeros(0, rOuter);
    elseif rOuter == L
        outerComb = 1:L;
    else
        outerComb = nchoosek(1:L, rOuter);
    end
    if symOuter
        outerPerm = localExpandPerms(outerComb);
    else
        outerPerm = outerComb;
    end

    permIdx = localNestedAssemble(slotsOf, outerPerm, rInner, rOuter, symInner);
    combIdx = localNestedAssemble(slotsOf, outerComb, rInner, rOuter, false);
end


function cols = localNestedAssemble(slotsOf, outerSel, rInner, rOuter, symInner)
    %LOCALNESTEDASSEMBLE  Assemble nested r-tuple slot-index columns.
    D = rInner * rOuter;
    colsList = {};
    for s = 1:size(outerSel, 1)
        tagIdxRow = outerSel(s, :);
        perEvent = cell(1, rOuter);
        ok = true;
        for j = 1:rOuter
            perEvent{j} = localInnerTuples(slotsOf{tagIdxRow(j)}, rInner, symInner);
            if isempty(perEvent{j})
                ok = false;
                break
            end
        end
        if ~ok
            continue
        end
        counts = cellfun(@numel, perEvent);
        total = prod(counts);
        for c = 0:total - 1
            choice = zeros(1, rOuter);
            rem = c;
            for j = rOuter:-1:1          % last event varies fastest
                choice(j) = mod(rem, counts(j)) + 1;
                rem = floor(rem / counts(j));
            end
            seg = zeros(1, D);
            pos = 0;
            for j = 1:rOuter
                seg(pos + 1 : pos + rInner) = perEvent{j}{choice(j)};
                pos = pos + rInner;
            end
            colsList{end + 1} = seg(:);  %#ok<AGROW>
        end
    end
    if isempty(colsList)
        cols = zeros(D, 0);
    else
        cols = [colsList{:}];
    end
end


function tuples = localInnerTuples(sl, rInner, symInner)
    %LOCALINNERTUPLES  Inner r-tuples (slot indices) within one event.
    sl = sl(:).';
    nsl = numel(sl);
    if rInner > nsl
        tuples = {};
        return
    elseif rInner == nsl
        combs = 1:nsl;
    else
        combs = nchoosek(1:nsl, rInner);
    end
    if symInner
        combs = localExpandPerms(combs);
    end
    nT = size(combs, 1);
    tuples = cell(1, nT);
    for i = 1:nT
        tuples{i} = sl(combs(i, :));
    end
end


function out = localExpandPerms(combs)
    %LOCALEXPANDPERMS  Expand each row (a combination) into its full
    %   permutation orbit; stacks rows of width size(combs, 2).
    if isempty(combs)
        out = combs;
        return
    end
    r = size(combs, 2);
    P = perms(1:r);
    nP = size(P, 1);
    nC = size(combs, 1);
    out = zeros(nC * nP, r);
    row = 0;
    for i = 1:nC
        for pp = 1:nP
            row = row + 1;
            out(row, :) = combs(i, P(pp, :));
        end
    end
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

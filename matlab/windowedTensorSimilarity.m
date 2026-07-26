function profile = windowedTensorSimilarity(densContext, densQuery, windowSpec, offsets, varargin)
%WINDOWEDSIMILARITY  Sliding-window similarity profile (cross-correlation).
%
%   profile = windowedTensorSimilarity(densContext, densQuery, windowSpec, offsets)
%   returns a 1 x M profile of windowed similarities. For each offset
%   column, the context density is windowed with windowSpec at the
%   corresponding centre, and its windowed inner product against
%   densQuery (unwindowed) is finalised by a normaliser selected via
%   the 'normalize' name-value option (default 'oneSidedDenom').
%
%   The two operands play asymmetric roles:
%     * densContext is the operand that the window multiplies. As the
%       sweep proceeds, the window shifts to each centre defined by
%       the offsets matrix, selecting different regions of densContext
%       at each step.
%     * densQuery is the unwindowed operand whose self inner product
%       appears in the denominator. The query supplies the
%       "comparison template" against which each windowed context
%       region is scored.
%
%   profile = windowedTensorSimilarity(densContext, {d_q_1, ..., d_q_n}, ...)
%   profile = windowedTensorSimilarity({d_c_1, ..., d_c_m}, densQuery, ...)
%   profile = windowedTensorSimilarity({d_c_1, ...}, {d_q_1, ...}, ...)
%   List mode. Either or both density inputs may be a cell array of
%   MaetDensity structs. Returns a cell array of 1-by-M profiles.
%   Modes (controlled via the 'mode' name-value option):
%     'bulger'    — n_c == n_q required; pair element-by-element.
%                   Returns a 1-by-n cell.
%     'cartesian' — Cross every context with every query.
%                   Returns an n_c-by-n_q cell.
%     'auto'      — pairwise if lengths match, cartesian otherwise
%                   (default).
%   Shape rule: a length-1 list returns a length-1 cell (never
%   collapses to a scalar profile).
%
%   Normalisation: 'oneSidedDenom' versus 'cosine'
%   ----------------------------------------------
%   The numerator at each sweep position is the windowed inner product
%   ip_qc = <h * dens_c, dens_q>. The denominator depends on
%   'normalize':
%
%     'oneSidedDenom' (default) — divide by the unwindowed query self
%         inner product, <dens_q, dens_q>. The result is magnitude-
%         aware: self-similarity at full window coverage equals 1,
%         silent regions of the context score near zero, and a region
%         where the windowed context has more matching mass than the
%         query holds in total may score above 1. This is the
%         intended reading for sliding-motif analysis -- a dense
%         local match should outscore a sparse one.
%
%     'cosine' — divide by sqrt(<h * dens_c, h * dens_c> *
%         <dens_q, dens_q>). The result is the strict shape-only
%         cosine, bounded in [-1, 1] and invariant to a positive
%         scalar on either operand. Closed-form across the (size,
%         mix) family only for pure-Gaussian (mix = 0) and pure-
%         boxcar (mix = 1) windows; intermediate mix raises and
%         directs the user to 'oneSidedDenom'.
%
%   Reference-point semantics
%   -------------------------
%   Offsets are measured from a reference point to the window centre
%   on each windowed attribute. Two options are provided:
%
%     * Default ('reference' not given or empty): the reference on
%       each attribute is the unweighted column mean of the query's
%       tuple centres. A purely geometric property of the tuple
%       centres, independent of the tuple weights.
%
%     * User-supplied ('reference' given as a 1 x A cell array): one
%       vector per query attribute, of length equal to that
%       attribute's dimension. The reference does not depend on the
%       query.
%
%   The peak offset under either option equals P* - ref, where P* is
%   the window centre (in context coordinates) at which the profile
%   peaks. Peak offsets under the default therefore track the
%   quantity P* - mu_q across between-query variation; peak offsets
%   under a fixed reference track P* directly.
%
%   The choice matters most when a pitch attribute has more than one
%   slot per event (chords with exchangeable voices, or partials
%   added by addSpectra), because queries can then vary in slot
%   count, slot values, and slot weights. For slot-weight sweeps the
%   two options coincide. For slot-value sweeps (e.g., stretching
%   partials), the default's peak offset drifts while a fixed
%   reference's stays put. For slot-count sweeps, the default's
%   peak offset is stable only for harmonic queries -- those whose
%   slots lie at (or close to) integer-harmonic values
%   f_e + 1200*log2(n) cents. See User Guide §3.1 "Post-tensor
%   windowing" and the demo_windowingReference demo for analysis
%   and worked examples.
%
%   In both cases, a peak at offset delta means the context has
%   similarity-relevant structure at reference + delta.
%
%   Periodic attributes
%   -------------------
%   For periodic attributes, the window is the wrapped Gaussian (or
%   wrapped rect-conv-Gaussian for mix > 0): the sum of line-case
%   window functions at all periodic images of the centre. The
%   toolbox sums these contributions adaptively, truncating when the
%   latest image-pair's contribution falls below the floating-point
%   threshold (1e-12 for double, 1e-7 for kernelPrecision='single').
%   For multi-D absolute periodic attributes, the image sum factorises
%   per axis (linear in dimension). For multi-D relative periodic
%   attributes, image summation is deferred to a future release and the
%   existing line-case formula is used (matching the pre-2.2
%   behaviour). See User Guide §3.1 "Post-tensor windowing".
%
%   Inputs
%       densContext - MaetDensity to be windowed (positional arg 1).
%       densQuery   - MaetDensity, unwindowed (positional arg 2).
%       windowSpec  - Window spec struct (see windowTensor). Only the
%                     'size' and 'mix' fields are read; any 'centre'
%                     field is ignored (offsets are used instead).
%       offsets     - dim x M matrix of per-sweep offsets in effective
%                     space, using the attribute-concatenated flat
%                     convention of windowTensor. M is the number of
%                     sweep positions. A 1-D vector is accepted when
%                     dim == 1.
%
%   Name-value arguments
%       'reference' - 1 x A cell array, one entry per query attribute,
%                     each a column vector of length equal to that
%                     attribute's dimension. Overrides the default
%                     unweighted-centroid reference. Default: [] (use
%                     unweighted centroid).
%
%                     In list mode, 'reference' may also be a
%                     length-n_q cell-of-cells, with each entry itself
%                     a 1 x A cell of per-attribute vectors specifying
%                     the reference for the corresponding query.
%                     Disambiguation: the input is treated as per-query
%                     iff its length equals n_q AND its first entry is
%                     itself a cell. Otherwise it is broadcast as a
%                     shared reference across all queries.
%       'mode'      - List-mode pairing. 'bulger', 'cartesian',
%                     or 'auto' (default). Ignored when both inputs are
%                     scalar densities.
%       'normalize' / 'normalise' — 'oneSidedDenom' (default) or
%                     'cosine'. Selects the denominator applied to the
%                     windowed inner product (see "Normalisation"
%                     section above). Either spelling of the keyword
%                     is accepted; matching on the value is case-
%                     insensitive.
%       'verbose'   - Default true.
%       'truncationSigmas' - Numeric scalar or []. Override the
%                     toolbox-wide mptDefaults('truncationSigmas')
%                     setting for this call. Passes through to the
%                     kernel evaluator on the centres path; skips
%                     Gaussian contributions whose centre-to-query
%                     distance exceeds k*sigma. [] (default) means
%                     use the global default (factory: Inf).
%       'kernelPrecision' - 'double', 'single', or [] for the global
%                     default. Override the toolbox-wide
%                     kernelPrecision setting for this call. 'single'
%                     casts the kernel matrix to float32 for a ~2x
%                     speedup at ~7 sig fig precision.
%
%   Output
%       profile     - 1 x M vector of windowed similarities.
%
%   See also windowTensor, cosSimExpTens.

    % Top-level call guard: dispatch throttle + kernelChunkBytes pin. See internal.callGuard.
    guard = internal.callGuard(); %#ok<NASGU>

    verbose = true;
    reference = [];
    mode = 'auto';
    normalize = 'oneSidedDenom';
    truncationSigmas = [];
    kernelPrecision = [];
    for i = 1:2:numel(varargin)
        switch lower(varargin{i})
            case 'verbose'
                verbose = logical(varargin{i + 1});
            case 'reference'
                reference = varargin{i + 1};
            case 'mode'
                mode = lower(char(varargin{i + 1}));
                if ~any(strcmp(mode, {'bulger', 'cartesian', 'auto'}))
                    error('windowedTensorSimilarity:badMode', ...
                        ['''mode'' must be ''bulger'', ''cartesian'', or ' ...
                         '''auto''; got ''%s''.'], mode);
                end
            case {'normalize', 'normalise'}
                % Accept both American and British spellings of the
                % keyword; case-insensitive matching on the value.
                val = char(varargin{i + 1});
                if strcmpi(val, 'cosine')
                    normalize = 'cosine';
                elseif strcmpi(val, 'oneSidedDenom')
                    normalize = 'oneSidedDenom';
                else
                    error('windowedTensorSimilarity:badNormalize', ...
                          ['''normalize'' must be ''cosine'' or ' ...
                           '''oneSidedDenom''; got ''%s''.'], val);
                end
            case 'truncationsigmas'
                truncationSigmas = varargin{i + 1};
            case 'kernelprecision'
                kernelPrecision = varargin{i + 1};
            otherwise
                error('windowedTensorSimilarity:badNVpair', ...
                      'Unknown name-value pair: %s.', varargin{i});
        end
    end

    % truncationSigmas and kernelPrecision are accepted for API
    % compatibility but currently unused: the closed-form windowed
    % inner product (internal.windowedInnerProduct) does not yet
    % expose kernel-precision or truncation controls. In list mode
    % they continue to be forwarded to recursive windowedTensorSimilarity
    % calls so the API contract on the recursive form is unchanged.

    % --- LIST mode ------------------------------------------
    % Either or both of densContext, densQuery may be a cell array of
    % MaetDensity structs, in which case the function returns a cell
    % array of profiles. Modes:
    %   'bulger'    — n_c == n_q required; pair element-by-element.
    %                 Returns a 1-by-n cell of 1-by-M profiles.
    %   'cartesian' — Cross every context with every query.
    %                 Returns an n_c-by-n_q cell of 1-by-M profiles.
    %   'auto'      — pairwise if n_c == n_q, otherwise cartesian.
    % Shape rule: a length-1 list returns a length-1 cell (never
    % collapses to a scalar profile).
    isContextList = iscell(densContext);
    isQueryList   = iscell(densQuery);

    % Densities built with a matrix-valued kernel covariance store
    % whitened coordinates, so windows and offsets (specified in
    % original coordinates) would be applied in the wrong frame.
    if localAnyKernelCov(densContext) || localAnyKernelCov(densQuery)
        error('mpt:aniso:windowedTensorSimilarity', ...
            ['windowedTensorSimilarity does not support densities ' ...
             'built with a matrix-valued kernel covariance: their ' ...
             'stored coordinates are whitened, so windows and ' ...
             'offsets (specified in original coordinates) would be ' ...
             'applied in the wrong frame. Use windowedSimilarity, ' ...
             'whose sweep translates and windows the raw events ' ...
             'before each density is built.']);
    end

    if isContextList || isQueryList
        profile = localWindowedSimilarityList( ...
            densContext, densQuery, windowSpec, offsets, ...
            reference, mode, normalize, ...
            truncationSigmas, kernelPrecision, verbose);
        return;
    end

    if ~strcmp(mode, 'auto')
        % 'mode' was explicitly set, but neither operand is a list.
        % Honour the request only by ignoring it (scalar inputs have no
        % combinatoric structure). No need to error; this preserves the
        % scalar single-pair contract.
    end

    if ~isstruct(densQuery) || ~isfield(densQuery, 'tag') || ...
            ~strcmp(densQuery.tag, 'MaetDensity')
        error('windowedTensorSimilarity:badQuery', ...
              'densQuery must be a MaetDensity.');
    end
    if ~isstruct(densContext) || ~isfield(densContext, 'tag') || ...
            ~strcmp(densContext.tag, 'MaetDensity')
        error('windowedTensorSimilarity:badContext', ...
              'densContext must be a MaetDensity.');
    end

    % Ensure both densities have per-tuple fields populated. Cheap
    % no-op if they came from buildExpTens with 'lazy', false.
    densQuery   = internal.ensureExpTensExpensive(densQuery);
    densContext = internal.ensureExpTensExpensive(densContext);

    dim_c = densContext.dim;
    offsets = double(offsets);
    if isvector(offsets) && dim_c == 1
        offsets = offsets(:).';
    end
    if size(offsets, 1) ~= dim_c
        error('windowedTensorSimilarity:offsetsShape', ...
              'offsets must have %d rows (dim of densContext); got %d.', ...
              dim_c, size(offsets, 1));
    end
    M = size(offsets, 2);

    % --- Reference point, per attribute -----------------------------
    A = densQuery.nAttrs;
    dimPerAttr_q = densQuery.dimPerAttr;
    refPerA = cell(1, A);
    if isempty(reference)
        % Default: unweighted column mean of Centres{a}.
        for a = 1:A
            refPerA{a} = mean(densQuery.Centres{a}, 2);
        end
    else
        if ~iscell(reference) || numel(reference) ~= A
            error('windowedTensorSimilarity:badReference', ...
                  'reference must be a 1 x %d cell array.', A);
        end
        for a = 1:A
            r = double(reference{a});
            r = r(:);
            if numel(r) ~= dimPerAttr_q(a)
                error('windowedTensorSimilarity:badReferenceLength', ...
                      'reference{%d} must have length %d; got %d.', ...
                      a, dimPerAttr_q(a), numel(r));
            end
            refPerA{a} = r;
        end
    end

    % --- Strip any user-supplied 'centre' field; offsets replace it --
    baseSpec = windowSpec;
    if isfield(baseSpec, 'centre')
        baseSpec = rmfield(baseSpec, 'centre');
    end

    % --- Dispatch announce ---
    % windowedTensorSimilarity uses a single algorithmic path: the closed-form
    % windowed inner product (no Bulger / Möbius / centres choice to
    % make). The announce reads 'chose direct path' to surface the
    % method to the user; throttled to once per top-level call.
    internal.maybeShowDispatchMsg('windowedTensorSimilarity', 'direct', ...
        'closed-form windowed inner product (single algorithmic path)');

    % --- Pre-compute the unwindowed L2 norm of the query (denominator) ---
    % The unwindowed query self inner product <dens_q, dens_q> appears
    % in the denominator under both 'oneSidedDenom' (where it IS the
    % denominator) and 'cosine' (where it is one factor of the
    % geometric mean). It depends only on densQuery, not on the
    % window offset, so we compute it once and cache it across the
    % sweep, letting every per-offset call to
    % internal.windowedInnerProduct skip the redundant work.
    ipQQcache = internal.windowedInnerProduct(densQuery, [], false);

    % --- Up-front time estimate + adaptive progress stride ---
    % Calibrate empirically (warm-up + timed sample) and extrapolate to
    % the full M-point sweep, matching the pattern used by the other
    % batched helpers (cosSimExpTens batched-raw, entropyExpTens, etc.).
    % Threshold 10 s via internal.printBatchedEstimate; gated on
    % verbose. For M = 1 the calibration is skipped entirely (nothing
    % to estimate or count down).
    progStride = 1;
    showProgress = false;
    if verbose && M >= 2
        nCal = min(5, M);
        sampleIdx = unique(round(linspace(1, M, nCal)));

        % Warm-up: one iteration of the loop body to absorb one-time
        % setup (cache populate, etc.) before the timed sample.
        centre_cell_w = cell(1, A);
        off_ptr = 0;
        for a = 1:A
            da = dimPerAttr_q(a);
            centre_cell_w{a} = refPerA{a} + ...
                offsets(off_ptr + 1 : off_ptr + da, sampleIdx(1));
            off_ptr = off_ptr + da;
        end
        spec_w = baseSpec;
        spec_w.centre = centre_cell_w;
        wmd_w = windowTensor(densContext, spec_w);
        internal.windowedInnerProduct(densQuery, wmd_w, false, ...
            ipQQcache, normalize);

        % Timed calibration sample over the same indices.
        tCalStart = tic;
        for cs = 1:numel(sampleIdx)
            centre_cell_s = cell(1, A);
            off_ptr = 0;
            for a = 1:A
                da = dimPerAttr_q(a);
                centre_cell_s{a} = refPerA{a} + ...
                    offsets(off_ptr + 1 : off_ptr + da, sampleIdx(cs));
                off_ptr = off_ptr + da;
            end
            spec_s = baseSpec;
            spec_s.centre = centre_cell_s;
            wmd_s = windowTensor(densContext, spec_s);
            internal.windowedInnerProduct(densQuery, wmd_s, false, ...
                ipQQcache, normalize);
        end
        tCalTotal  = toc(tCalStart);
        tPerPoint  = tCalTotal / numel(sampleIdx);
        estTotal   = tCalTotal + tPerPoint * M;
        internal.printBatchedEstimate('windowedTensorSimilarity', M, estTotal);
        progStride = internal.progressStride(tPerPoint);
        showProgress = estTotal >= 5;
    end

    profile = zeros(1, M);
    for m = 1:M
        % Per-attribute absolute centre = reference + per-attribute slice
        % of this sweep's offset vector.
        centre_cell = cell(1, A);
        off_ptr = 0;
        for a = 1:A
            da = dimPerAttr_q(a);
            centre_cell{a} = refPerA{a} + offsets(off_ptr + 1 : off_ptr + da, m);
            off_ptr = off_ptr + da;
        end
        spec_m = baseSpec;
        spec_m.centre = centre_cell;
        wmd = windowTensor(densContext, spec_m);
        profile(m) = internal.windowedInnerProduct(densQuery, wmd, false, ...
            ipQQcache, normalize);

        if verbose && showProgress && (mod(m, progStride) == 0 || m == M)
            fprintf('  %d / %d points computed.\n', m, M);
        end
    end

    if verbose
        fprintf('windowedTensorSimilarity: done.\n');
    end
end


function profile = localWindowedSimilarityList( ...
    densContext, densQuery, windowSpec, offsets, ...
    reference, mode, normalize, ...
    truncationSigmas, kernelPrecision, verbose)
%LOCALWINDOWEDSIMILARITYLIST  Polymorphic list dispatch.
%
%   Iterates over context and query lists, calling windowedTensorSimilarity
%   recursively for each pair. The recursive call uses the same
%   positional convention as the public entry: context first, query
%   second.
%
%   ``normalize``, ``truncationSigmas`` and ``kernelPrecision`` are
%   forwarded to each recursive call so per-call kwargs reach the
%   per-offset ``internal.windowedInnerProduct`` consumers without
%   going through ``mptDefaults`` global state.

    % Wrap singletons so the loops below can index uniformly.
    if iscell(densContext)
        C = densContext;
    else
        C = {densContext};
    end
    if iscell(densQuery)
        Q = densQuery;
    else
        Q = {densQuery};
    end
    nC = numel(C);
    nQ = numel(Q);

    % Resolve mode.
    if strcmp(mode, 'auto')
        if nC == nQ
            modeR = 'bulger';
        else
            modeR = 'cartesian';
        end
    else
        modeR = mode;
    end
    if strcmp(modeR, 'bulger') && nC ~= nQ
        error('windowedTensorSimilarity:listLengthMismatch', ...
              ['windowedTensorSimilarity (list mode, pairwise): context and ' ...
               'query must have the same length, got %d and %d.'], nC, nQ);
    end

    % Resolve reference per-query. Three forms:
    %   1) [] (auto-centroid per query)
    %   2) length-A cell (shared across queries; broadcast)
    %   3) length-nQ cell-of-cells (per-query references; each entry
    %      itself a 1-by-A cell of per-attribute vectors)
    perQueryRef = false;
    if iscell(reference) && numel(reference) == nQ && nQ >= 1
        % Disambiguate by looking at the first entry: if it's itself a
        % cell, treat as per-query. Otherwise treat as a shared length-A
        % cell that happens to have nQ entries (rare).
        if ~isempty(reference) && iscell(reference{1})
            perQueryRef = true;
        end
    end

    if strcmp(modeR, 'bulger')
        % Pairwise: paired indices.
        profile = cell(1, nC);
        for k = 1:nC
            if perQueryRef
                refK = reference{k};
            else
                refK = reference;
            end
            profile{k} = windowedTensorSimilarity(C{k}, Q{k}, windowSpec, offsets, ...
                'verbose', verbose, 'reference', refK, ...
                'normalize', normalize, ...
                'truncationSigmas', truncationSigmas, ...
                'kernelPrecision', kernelPrecision);
        end
    else
        % Cartesian: row i = context i, column j = query j.
        profile = cell(nC, nQ);
        for i = 1:nC
            for j = 1:nQ
                if perQueryRef
                    refIJ = reference{j};
                else
                    refIJ = reference;
                end
                profile{i, j} = windowedTensorSimilarity( ...
                    C{i}, Q{j}, windowSpec, offsets, ...
                    'verbose', verbose, 'reference', refIJ, ...
                    'normalize', normalize, ...
                    'truncationSigmas', truncationSigmas, ...
                    'kernelPrecision', kernelPrecision);
            end
        end
    end
end


function tf = localAnyKernelCov(op)
%LOCALANYKERNELCOV  True if any density in a scalar-or-cell operand
%carries a matrix-valued kernel covariance.
    if iscell(op)
        tf = any(cellfun(@internal.densityHasKernelCov, op));
    else
        tf = internal.densityHasKernelCov(op);
    end
end

function profile = windowedSimilarity(densQuery, densContext, windowSpec, offsets, varargin)
%WINDOWEDSIMILARITY  Sliding-window similarity profile (cross-correlation).
%
%   profile = windowedSimilarity(densQuery, densContext, windowSpec, offsets)
%   returns a 1 x M profile of windowed similarities. For each offset
%   column, the context density is windowed with windowSpec at the
%   corresponding centre, and its similarity against densQuery
%   (unwindowed) is computed. The normaliser uses the UNWINDOWED L2
%   norms of both operands (Option Z).
%
%   profile = windowedSimilarity({d_q_1, ..., d_q_n}, densContext, ...)
%   profile = windowedSimilarity(densQuery, {d_c_1, ..., d_c_m}, ...)
%   profile = windowedSimilarity({d_q_1, ...}, {d_c_1, ...}, ...)
%   List mode. Either or both density inputs may be a cell
%   array of MaetDensity structs. Returns a cell array of 1-by-M
%   profiles. Modes (controlled via the 'mode' name-value option):
%     'bulger'  — n_q == n_c required; pair element-by-element.
%                   Returns a 1-by-n cell.
%     'cartesian' — Cross every query with every context.
%                   Returns an n_q-by-n_c cell.
%     'auto'      — pairwise if lengths match, cartesian otherwise
%                   (default).
%   Option II shape rule: a length-1 list returns a length-1 cell.
%
%   Why "similarity" rather than "cosine similarity"
%   ------------------------------------------------
%   The output is a magnitude-aware *windowed similarity*: because the
%   denominator uses unwindowed L2 norms (rather than the windowed
%   norm of the context), the profile is not bounded in [-1, 1] across
%   sweep positions and does not correspond to an inner product on a
%   single Hilbert space. This is the intended behaviour for sliding-
%   motif analysis -- a dense local match should outscore a sparse
%   one. The strict shape-only cosine similarity (with windowed
%   denominator) is a separate notion not currently implemented in the
%   toolbox.
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
%   Periodic groups
%   ---------------
%   The closed-form windowed inner product implemented here is the
%   line-case formula at wrapped differences -- exact for non-periodic
%   groups, and an approximation for periodic groups that retains
%   only the leading periodic image of the window. The exact periodic
%   expression is an absolutely convergent series over kernel pairs
%   and periodic images; efficient evaluation of the full series in
%   the regime lambda*sigma > P/(2*sqrt(3)) (in the Gaussian and
%   mixed-shape cases) is left to future work.
%
%   A warning with identifier
%   windowedSimilarity:periodicWindowApprox is emitted on every call
%   involving a windowed periodic group. Within the recommended
%   bound lambda*sigma <= P/(2*sqrt(3)) the warning takes a brief
%   informational form noting that the line-case approximation is
%   in use; past the bound it switches to a stronger form that
%   reports SD/P and phi (rect half-width) against their respective
%   bounds and describes the qualitative behaviour past the bound
%   (at mix = 1 the rect window is no longer localized on the
%   circle; at mix = 0 the approximation degrades smoothly;
%   intermediate mix falls between). The warning is suppressible
%   via the standard MATLAB warning('off', '<id>') mechanism. See
%   User Guide §3.1 "Post-tensor windowing" for the analysis.
%
%   Inputs
%       densQuery   - MaetDensity (not windowed).
%       densContext - MaetDensity to be windowed.
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

    % Top-level call guard: see internal.dispatchScope.
    guard = internal.dispatchScope(); %#ok<NASGU>

    verbose = true;
    reference = [];
    mode = 'auto';
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
                    error('windowedSimilarity:badMode', ...
                        ['''mode'' must be ''bulger'', ''cartesian'', or ' ...
                         '''auto''; got ''%s''.'], mode);
                end
            case 'truncationsigmas'
                truncationSigmas = varargin{i + 1};
            case 'kernelprecision'
                kernelPrecision = varargin{i + 1};
            otherwise
                error('windowedSimilarity:badNVpair', ...
                      'Unknown name-value pair: %s.', varargin{i});
        end
    end

    % truncationSigmas and kernelPrecision are accepted for API
    % compatibility but currently unused: the closed-form windowed
    % inner product (internal.windowedInnerProduct) does not yet
    % expose kernel-precision or truncation controls. In list mode
    % they continue to be forwarded to recursive windowedSimilarity
    % calls so the API contract on the recursive form is unchanged.

    % --- LIST mode ------------------------------------------
    % Either or both of densQuery, densContext may be a cell array of
    % MaetDensity structs, in which case the function returns a cell
    % array of profiles. Modes:
    %   'bulger'  — n_q == n_c required; pair element-by-element.
    %                 Returns a 1-by-n cell of 1-by-M profiles.
    %   'cartesian' — Cross every query with every context.
    %                 Returns an n_q-by-n_c cell of 1-by-M profiles.
    %   'auto'      — pairwise if n_q == n_c, otherwise cartesian.
    % Option II shape rule: a length-1 list returns a length-1 cell.
    isQueryList   = iscell(densQuery);
    isContextList = iscell(densContext);
    if isQueryList || isContextList
        profile = localWindowedSimilarityList( ...
            densQuery, densContext, windowSpec, offsets, ...
            reference, mode, truncationSigmas, kernelPrecision, verbose);
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
        error('windowedSimilarity:badQuery', ...
              'densQuery must be a MaetDensity.');
    end
    if ~isstruct(densContext) || ~isfield(densContext, 'tag') || ...
            ~strcmp(densContext.tag, 'MaetDensity')
        error('windowedSimilarity:badContext', ...
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
        error('windowedSimilarity:offsetsShape', ...
              'offsets must have %d rows (dim of densContext); got %d.', ...
              dim_c, size(offsets, 1));
    end
    M = size(offsets, 2);

    % --- Periodic-window warning -----------------------------------
    % A windowedSimilarity:periodicWindowApprox warning is emitted
    % per periodic windowed group on every call. The message has
    % two forms:
    %   - Within the recommended bound (lambda*sigma <= P/(2*sqrt(3))):
    %     a brief informational notice that the line-case
    %     approximation is in use, with the current SD/P against the
    %     bound. The approximation is sub-percent across the window
    %     shape family within this bound.
    %   - Past the bound (lambda*sigma > P/(2*sqrt(3))): a stronger
    %     notice reporting SD/P and phi (rect half-width) against
    %     their bounds, and describing the qualitative behaviour by
    %     mix (at mix=1 the rect window is no longer localized on
    %     the circle; at mix=0 the approximation degrades smoothly).
    % See User Guide §3.1 "Post-tensor windowing".
    SD_OVER_P_BOUND = 1 / (2 * sqrt(3));  % ~= 0.2887
    G = densContext.nGroups;
    sizeVec = double(windowSpec.size(:).');
    if isscalar(sizeVec)
        sizeVec = repmat(sizeVec, 1, G);
    end
    mixVec = double(windowSpec.mix(:).');
    if isscalar(mixVec)
        mixVec = repmat(mixVec, 1, G);
    end
    for g = 1:G
        if ~densContext.isPer(g),  continue; end
        lambda = sizeVec(g);
        if ~isfinite(lambda) || lambda <= 0,  continue; end
        P = densContext.period(g);
        if P <= 0,  continue; end
        effSigma = lambda * densContext.sigma(g);
        sdOverP = effSigma / P;
        gammaG = mixVec(g);

        if sdOverP <= SD_OVER_P_BOUND
            % Within-bound: brief informational form.
            warning('windowedSimilarity:periodicWindowApprox', ...
                ['Periodic windowed inner product on group %d ' ...
                 'applies the line-case formula at wrapped ' ...
                 'differences -- an approximation that retains ' ...
                 'only the leading periodic image of the window. ' ...
                 'Within the recommended bound, the approximation ' ...
                 'is sub-percent across the window shape family.\n' ...
                 '  Window SD (lambda*sigma) = %g\n' ...
                 '  Period P                 = %g\n' ...
                 '  SD/P                     = %.4f\n' ...
                 '  Recommended bound (SD/P) = %.4f ' ...
                 '(= 1/(2*sqrt(3)))\n' ...
                 'See User Guide §3.1 "Post-tensor windowing". ' ...
                 'Suppress with warning(''off'', ' ...
                 '''windowedSimilarity:periodicWindowApprox'').'], ...
                g, effSigma, P, sdOverP, SD_OVER_P_BOUND);
        else
            % Past-bound: stronger form, with phi and per-mix
            % behaviour.
            phiG = effSigma * sqrt(3 * max(gammaG, 0));
            warning('windowedSimilarity:periodicWindowApprox', ...
                ['Window SD exceeds the recommended bound for ' ...
                 'periodic group %d; the line-case approximation ' ...
                 'is no longer reliable.\n' ...
                 '  Window SD (lambda*sigma) = %g\n' ...
                 '  Period P                 = %g\n' ...
                 '  SD/P                     = %.4f  (bound: %.4f)\n' ...
                 '  phi (rect half-width)    = %g  ' ...
                 '(bound: %g = P/2)\n' ...
                 '  mix (gamma)              = %g\n' ...
                 'Beyond the bound, behaviour depends on mix:\n' ...
                 '  mix = 1 (pure rect):     window is no longer ' ...
                 'localized on the circle (pointless as a window).\n' ...
                 '  mix = 0 (pure Gaussian): line-case approximation ' ...
                 'degrades smoothly; error grows with SD/P.\n' ...
                 '  intermediate mix:        between these two cases.\n' ...
                 'Reduce size or sigma so that lambda*sigma <= ' ...
                 'P/(2*sqrt(3)). See User Guide §3.1 "Post-tensor ' ...
                 'windowing".'], ...
                g, effSigma, P, sdOverP, SD_OVER_P_BOUND, ...
                phiG, P/2, gammaG);
        end
    end

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
            error('windowedSimilarity:badReference', ...
                  'reference must be a 1 x %d cell array.', A);
        end
        for a = 1:A
            r = double(reference{a});
            r = r(:);
            if numel(r) ~= dimPerAttr_q(a)
                error('windowedSimilarity:badReferenceLength', ...
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
    % windowedSimilarity uses a single algorithmic path: the closed-form
    % windowed inner product (no Bulger / Möbius / centres choice to
    % make). The announce reads 'chose direct path' to surface the
    % method to the user; throttled to once per top-level call.
    internal.maybeShowDispatchMsg('windowedSimilarity', 'direct', ...
        'closed-form windowed inner product (single algorithmic path)', ...
        0, false);

    % --- Pre-compute the unwindowed L2 norms (denominator).
    % Both ip_qq = <Q, Q>_unwindowed and ip_cc = <C, C>_unwindowed
    % depend only on the densities, NOT on the window offset.
    % Computing them ONCE here lets every per-offset call to
    % internal.windowedInnerProduct skip the redundant per-call work
    % (originally ~2/3 of inner-loop time on this path).
    ipQQcache = internal.windowedInnerProduct(densQuery, [], false);
    ipCCcache = internal.windowedInnerProduct(densContext, [], false);

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
            ipQQcache, ipCCcache);

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
                ipQQcache, ipCCcache);
        end
        tCalTotal  = toc(tCalStart);
        tPerPoint  = tCalTotal / numel(sampleIdx);
        estTotal   = tCalTotal + tPerPoint * M;
        internal.printBatchedEstimate('windowedSimilarity', M, estTotal);
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
            ipQQcache, ipCCcache);

        if verbose && showProgress && (mod(m, progStride) == 0 || m == M)
            fprintf('  %d / %d points computed.\n', m, M);
        end
    end

    if verbose
        fprintf('windowedSimilarity: done.\n');
    end
end


function profile = localWindowedSimilarityList( ...
    densQuery, densContext, windowSpec, offsets, ...
    reference, mode, truncationSigmas, kernelPrecision, verbose)
%LOCALWINDOWEDSIMILARITYLIST  Polymorphic list dispatch.
%
%   Iterates over query and context lists, calling windowedSimilarity
%   recursively for each pair. Suppresses the periodic-window warning
%   inside the loop after the first emission to avoid spam.
%
%   ``truncationSigmas`` and ``kernelPrecision`` are forwarded to each
%   recursive ``windowedSimilarity`` call so per-call kwargs reach the
%   per-offset ``internal.windowedInnerProduct`` consumers without going
%   through ``mptDefaults`` global state.

    % Wrap singletons so the loops below can index uniformly.
    if iscell(densQuery)
        Q = densQuery;
    else
        Q = {densQuery};
    end
    if iscell(densContext)
        C = densContext;
    else
        C = {densContext};
    end
    nQ = numel(Q);
    nC = numel(C);

    % Resolve mode.
    if strcmp(mode, 'auto')
        if nQ == nC
            modeR = 'bulger';
        else
            modeR = 'cartesian';
        end
    else
        modeR = mode;
    end
    if strcmp(modeR, 'bulger') && nQ ~= nC
        error('windowedSimilarity:listLengthMismatch', ...
              ['windowedSimilarity (list mode, pairwise): query and context ' ...
               'must have the same length, got %d and %d.'], nQ, nC);
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

    % Suppress repeat periodic-window warnings inside the loop. We
    % allow the first inner call to emit normally; afterwards we toggle
    % the warning off and restore the user's previous state at the end.
    warnId = 'windowedSimilarity:periodicWindowApprox';
    prevState = warning('query', warnId);

    cleanupObj = onCleanup(@() warning(prevState));   %#ok<NASGU>

    if strcmp(modeR, 'bulger')
        profile = cell(1, nQ);
        for k = 1:nQ
            if perQueryRef
                refK = reference{k};
            else
                refK = reference;
            end
            profile{k} = windowedSimilarity(Q{k}, C{k}, windowSpec, offsets, ...
                'verbose', verbose, 'reference', refK, ...
                'truncationSigmas', truncationSigmas, ...
                'kernelPrecision', kernelPrecision);
            if k == 1
                warning('off', warnId);
            end
        end
    else
        % cartesian
        profile = cell(nQ, nC);
        first = true;
        for i = 1:nQ
            for j = 1:nC
                if perQueryRef
                    refIJ = reference{i};
                else
                    refIJ = reference;
                end
                profile{i, j} = windowedSimilarity( ...
                    Q{i}, C{j}, windowSpec, offsets, ...
                    'verbose', verbose, 'reference', refIJ, ...
                    'truncationSigmas', truncationSigmas, ...
                    'kernelPrecision', kernelPrecision);
                if first
                    warning('off', warnId);
                    first = false;
                end
            end
        end
    end
end

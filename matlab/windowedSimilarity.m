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
%   Note on naming
%   --------------
%   This function was named windowedCosSim in earlier drafts. The
%   output is a magnitude-aware *windowed similarity*: because the
%   denominator uses unwindowed L2 norms (rather than the windowed
%   norm of the context), the profile is not bounded in [-1, 1] across
%   sweep positions and does not correspond to an inner product on a
%   single Hilbert space. This is the intended behaviour for sliding-
%   motif analysis -- a dense local match should outscore a sparse one
%   -- but it means "cosine similarity" is not the right name for the
%   object. The strict shape-only cosine similarity (with windowed
%   denominator) is reserved as a separate notion in the manuscript
%   and is not currently implemented in the toolbox. See manuscript
%   §5.4.
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
%       'verbose'   - Default true.
%
%   Output
%       profile     - 1 x M vector of windowed similarities.
%
%   See also windowTensor, cosSimExpTens.

    verbose = true;
    reference = [];
    for i = 1:2:numel(varargin)
        switch lower(varargin{i})
            case 'verbose'
                verbose = logical(varargin{i + 1});
            case 'reference'
                reference = varargin{i + 1};
            otherwise
                error('windowedSimilarity:badNVpair', ...
                      'Unknown name-value pair: %s.', varargin{i});
        end
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
        profile(m) = cosSimExpTens(densQuery, wmd, 'verbose', verbose);
    end
end

function [pAttrOut, wOut, groupsOut] = weightEvents( ...
    pAttr, w, groups, inputAttr, targetAttr, ...
    centre, width, shape, isPer, period, opts)
%WEIGHTEVENTS Apply a per-event weight via an input-to-target window factor.
%
%   [pAttrOut, wOut, groupsOut] = weightEvents(pAttr, w, groups, ...
%       inputAttr, targetAttr, centre, width, shape, isPer, period, ...
%       'deleteInput', tf)
%   is a per-event preprocessing helper for multi-attribute tensor
%   input. It reads the K=1 value at every event from inputAttr,
%   evaluates a window function h centred at centre with standard
%   deviation width and shape parameter shape (= gamma), and writes
%   the resulting (1, N) per-event factor into the weight slot of
%   targetAttr, multiplied into any existing weight already there.
%   targetAttr may differ from inputAttr (the typical case --- e.g.,
%   time-driven windowing of pitch events) or coincide with it (the
%   input attribute weights itself).
%
%   When deleteInput=true and inputAttr differs from targetAttr, the
%   input attribute is removed from the returned pAttrOut / wOut /
%   groupsOut after the factor has been transferred to the target.
%   This is the canonical windowed-entropy / windowed-mass workflow:
%   the input attribute provides the scaffolding for the window and
%   is no longer needed downstream. When deleteInput=false, the input
%   attribute is preserved unchanged in the output. deleteInput=true
%   paired with inputAttr == targetAttr is rejected as incoherent
%   (deleting the input would discard the factor just written to it).
%
%   The window family is the peak-normalised convolution of a
%   rectangle and a Gaussian, with derived sub-parameters
%
%       phi = width * sqrt(3 * gamma),
%       xi  = width * sqrt(1 - gamma),
%
%   chosen so that the total variance equals width^2 across the whole
%   family (rectangle on [-phi, phi] has variance phi^2/3 = width^2 *
%   gamma; Gaussian has variance xi^2 = width^2 * (1 - gamma); the
%   two sum to width^2). The window is peak-normalised so h(0) = 1.
%
%   Limits:
%       gamma = 0: pure Gaussian h(delta) = exp(-delta^2 / (2 width^2)).
%       gamma = 1: pure rectangle h(delta) = 1[|delta| <= width sqrt(3)].
%
%   For a periodic input group (isPer = true), the difference
%   delta = v - centre is wrapped to [-P/2, P/2] before applying h;
%   the stored values in pAttr are not modified.
%
%   The per-event factor is broadcast across the target attribute's
%   K_target slots, so every slot of every event sees the same factor.
%
%   Factor entries whose distance from the centre exceeds the global
%   truncationSigmas cutoff (i.e., |delta| > truncationSigmas * width)
%   are hard-zeroed. The threshold is the same one the IP / evaluation
%   kernels use: at that distance a Gaussian window's value is
%   exp(-truncationSigmas^2 / 2). The default global value is Inf
%   (no truncation); set mptDefaults('truncationSigmas', k) to enable
%   hard truncation at k * width.
%
%   Inputs:
%     pAttr        1 x A cell of (K_a, N) per-attribute value matrices.
%                  K_a >= 1.
%     w            Existing weights. [], scalar, or 1 x A cell of
%                  scalar/(1, N)/(K_a, N) entries. None / [] means no
%                  existing weight (factor goes in directly).
%     groups       Group assignment for the input attributes. [], 1xA
%                  numeric vector of group indices, or cell-array of
%                  attribute-index lists per group (canonical-form
%                  matches buildExpTens).
%     inputAttr    Scalar integer in [1, A]. The attribute whose K = 1
%                  value supplies the window argument. Must have K = 1.
%     targetAttr   Scalar integer in [1, A]. The attribute whose
%                  weight slot receives the factor. May equal
%                  inputAttr.
%     centre       Scalar finite double. Window centre c.
%     width        Scalar positive double. Window standard deviation.
%     shape        Scalar double in [0, 1]. Shape parameter gamma.
%     isPer        Scalar logical. If true, the input attribute is
%                  periodic — delta is wrapped to [-period/2, period/2]
%                  before applying h.
%     period       Scalar positive double (only used when isPer).
%
%   Name-Value options:
%     deleteInput  (1,1) logical, REQUIRED (no default; the choice is
%                  destructive enough to be explicit at every call).
%
%   Outputs:
%     pAttrOut     1 x A_out cell of per-attribute value matrices.
%                  Length A if deleteInput=false, A - 1 otherwise.
%     wOut         1 x A_out cell of weights. The targetAttr slot (in
%                  the output indexing) carries the windowed weights.
%     groupsOut    1 x A_out numeric vector of canonical group indices.
%
%   See also BUILDEXPTENS, DIFFERENCEEVENTS, BINDEVENTS, TRANSLATEATTRIBUTES,
%            MPTDEFAULTS.

    arguments
        pAttr cell
        w
        groups
        inputAttr (1,1) double {mustBeInteger, mustBePositive}
        targetAttr (1,1) double {mustBeInteger, mustBePositive}
        centre (1,1) double
        width (1,1) double
        shape (1,1) double
        isPer (1,1) logical
        period (1,1) double
        opts.deleteInput (1,1) logical
    end

    % --- Normalise pAttr ---
    A = numel(pAttr);
    if A == 0
        error('weightEvents:emptyPAttr', ...
              'pAttr must contain at least one attribute.');
    end
    for a = 1:A
        M = pAttr{a};
        if ~ismatrix(M)
            error('weightEvents:badPAttrNdim', ...
                  'Attribute %d value matrix must be 2-D.', a);
        end
        if size(M, 1) == 0
            error('weightEvents:emptyKa', ...
                  ['Attribute %d has K_a = 0 (empty attribute); ' ...
                   'empty attributes are not permitted.'], a);
        end
        pAttr{a} = double(M);
    end

    % --- Shared N ---
    nEvents = size(pAttr{1}, 2);
    for a = 1:A
        if size(pAttr{a}, 2) ~= nEvents
            error('weightEvents:badNEvents', ...
                  ['All attributes must share the same event count N. ' ...
                   'Attribute 1 has N = %d; attribute %d has N = %d.'], ...
                  nEvents, a, size(pAttr{a}, 2));
        end
    end

    % --- Canonicalise groups ---
    groupOfAttr = localCanonicaliseGroups(groups, A);

    % --- Validate inputAttr ---
    if inputAttr > A
        error('weightEvents:badInputAttr', ...
              'inputAttr must be in 1..%d; got %d.', A, inputAttr);
    end
    if size(pAttr{inputAttr}, 1) ~= 1
        error('weightEvents:inputAttrNotK1', ...
              ['inputAttr %d has K = %d; weightEvents requires the ' ...
               'input attribute to have K = 1 (single value per event).'], ...
              inputAttr, size(pAttr{inputAttr}, 1));
    end

    % --- Validate targetAttr ---
    if targetAttr > A
        error('weightEvents:badTargetAttr', ...
              'targetAttr must be in 1..%d; got %d.', A, targetAttr);
    end

    % --- Validate deleteInput ---
    deleteInput = opts.deleteInput;
    if deleteInput && inputAttr == targetAttr
        error('weightEvents:deleteInputIncoherent', ...
              ['deleteInput=true is incoherent when inputAttr == ' ...
               'targetAttr (=%d): deleting the input would discard ' ...
               'the weight factor just written to it. Set ' ...
               'deleteInput=false, or choose a different targetAttr.'], ...
              inputAttr);
    end

    % --- Validate centre, width, shape ---
    if ~isfinite(centre)
        error('weightEvents:badCentre', ...
              'centre must be finite; got %g.', centre);
    end
    if ~isfinite(width) || width <= 0
        error('weightEvents:badWidth', ...
              'width must be finite and > 0; got %g.', width);
    end
    if shape < 0 || shape > 1
        error('weightEvents:badShape', ...
              ['shape (gamma) must lie in [0, 1]: gamma = 0 is pure ' ...
               'Gaussian, gamma = 1 is pure rectangle, intermediate ' ...
               'values are the fixed-variance convolution family. ' ...
               'Got %g.'], shape);
    end
    if isPer && period <= 0
        error('weightEvents:badPeriod', ...
              'period must be > 0 when isPer is true; got %g.', period);
    end

    % --- Compute factor h(delta) from input attribute values ---
    valRow = pAttr{inputAttr};         % (1, N)
    delta = valRow - centre;
    if isPer
        delta = delta - period * floor(delta / period + 0.5);
    end
    factor = localEvaluateShape(delta, width, shape);   % (1, N)

    % Truncate: zero factor entries whose distance exceeds
    % truncationSigmas * width. Uniform convention with the kernel
    % truncation in the IP / eval paths: at that distance a Gaussian
    % window's value is exp(-truncationSigmas^2 / 2), the same
    % threshold the kernel truncation uses. Reads the global default
    % so changes via mptDefaults('truncationSigmas', ...) propagate
    % without an extra kwarg. Inf disables (default).
    truncSig = mptDefaults('truncationSigmas');
    if isfinite(truncSig)
        factor(abs(delta) > truncSig * width) = 0;
    end

    % --- Normalise w to length-A cell; multiply factor into target slot ---
    wOut = localNormaliseWeightsToCell(w, A);
    wOut{targetAttr} = localMultiplyWeights( ...
        wOut{targetAttr}, factor, size(pAttr{targetAttr}, 1));

    % --- Build output structures, applying deleteInput if requested ---
    if deleteInput
        keep = setdiff(1:A, inputAttr);
        pAttrOut = pAttr(keep);
        wOut = wOut(keep);
        % Compact group numbering: if the input's group becomes empty
        % (input was its sole member), drop that group index and
        % decrement higher labels.
        gInput = groupOfAttr(inputAttr);
        keptGroups = groupOfAttr(keep);
        if sum(groupOfAttr == gInput) == 1
            keptGroups(keptGroups > gInput) = ...
                keptGroups(keptGroups > gInput) - 1;
        end
        groupsOut = keptGroups;
    else
        pAttrOut = pAttr;
        groupsOut = groupOfAttr;
    end
end


% =========================================================================
%  localEvaluateShape -- peak-normalised fixed-variance window family
% =========================================================================

function h = localEvaluateShape(delta, width, gamma)
%LOCALEVALUATESHAPE  Peak-normalised rect * Gaussian convolution
%(Section 5.2.1 of the MAET manuscript), with derived parameters
%   phi = width * sqrt(3 * gamma)   (rectangle half-width)
%   xi  = width * sqrt(1 - gamma)   (Gaussian std)
%so that the total variance is width^2 across the whole family.
    if gamma == 0
        % Pure Gaussian, std = width.
        h = exp(-(delta .^ 2) ./ (2 * width ^ 2));
        return;
    end
    if gamma == 1
        % Pure rectangle, half-width = width * sqrt(3).
        phi = width * sqrt(3);
        h = double(abs(delta) <= phi);
        return;
    end
    phi   = width * sqrt(3 * gamma);
    xi    = width * sqrt(1 - gamma);
    scale = xi * sqrt(2);
    num   = erf((delta + phi) ./ scale) - erf((delta - phi) ./ scale);
    peak  = 2 * erf(phi ./ scale);
    h = num ./ peak;
end


% =========================================================================
%  localNormaliseWeightsToCell
% =========================================================================

function wCell = localNormaliseWeightsToCell(w, A)
%LOCALNORMALISEWEIGHTSTOCELL  Coerce w to a 1 x A cell, preserving entries.
    wCell = cell(1, A);
    if isempty(w) && ~iscell(w)
        for a = 1:A
            wCell{a} = [];
        end
        return;
    end
    if isnumeric(w) && isscalar(w)
        sw = double(w);
        for a = 1:A
            wCell{a} = sw;
        end
        return;
    end
    if iscell(w)
        if numel(w) ~= A
            error('weightEvents:badWeightCellLength', ...
                  'w cell must have length A = %d; got %d.', A, numel(w));
        end
        for a = 1:A
            wCell{a} = w{a};
        end
        return;
    end
    error('weightEvents:badWeightType', ...
          ['w must be [], a scalar, or a 1 x A cell of scalar/' ...
           '(1,N)/(K_a,N) entries.']);
end


% =========================================================================
%  localMultiplyWeights
% =========================================================================

function wNew = localMultiplyWeights(wExisting, factor, K_target)
%LOCALMULTIPLYWEIGHTS  Multiply per-attribute weight by factor ((1, N) row).
%
%   factor is (1, N); the target attribute's existing weight may be
%   [], a scalar, a (1, N) row, or a (K_target, N) matrix. The factor
%   broadcasts across K_target slots (every slot of every event sees
%   the same factor).
    if isempty(wExisting)
        % factor broadcast to (K_target, N).
        wNew = repmat(factor, K_target, 1);
        return;
    end
    if isnumeric(wExisting) && isscalar(wExisting)
        wNew = repmat(double(wExisting) * factor, K_target, 1);
        return;
    end
    arr = double(wExisting);
    if isequal(size(arr), [1, size(factor, 2)])
        % (1, N) row: broadcast to K_target after multiplying.
        wNew = repmat(arr .* factor, K_target, 1);
        return;
    end
    if isequal(size(arr), [K_target, size(factor, 2)])
        % (K_target, N): per-slot weights, broadcast factor across rows.
        wNew = arr .* factor;
        return;
    end
    error('weightEvents:badExistingWeightShape', ...
          ['Existing weight shape [%s] is incompatible with target ' ...
           'shape [%d, %d] (factor is (1, %d)).'], ...
          num2str(size(arr)), K_target, size(factor, 2), size(factor, 2));
end


% =========================================================================
%  localCanonicaliseGroups
% =========================================================================

function groupOfAttr = localCanonicaliseGroups(groupsIn, A)
%LOCALCANONICALISEGROUPS  Canonical 1 x A vector of group indices.
%
%   Accepts [] (each attribute its own group), a 1 x A numeric vector
%   (already canonical, relabelled to contiguous 1..G), or a
%   cell-array of attribute-index lists (one per group).
    if isempty(groupsIn)
        groupOfAttr = 1:A;
        return;
    end
    if isnumeric(groupsIn)
        gv = double(groupsIn(:).');
        if numel(gv) ~= A
            error('weightEvents:badGroupsLength', ...
                  'groups vector must have length A = %d; got %d.', ...
                  A, numel(gv));
        end
        % Relabel contiguous 1..G in order of first appearance.
        [~, ~, ic] = unique(gv, 'stable');
        groupOfAttr = ic(:).';
        return;
    end
    if iscell(groupsIn)
        G_in = numel(groupsIn);
        groupOfAttr = zeros(1, A);
        for g = 1:G_in
            idx = groupsIn{g};
            idx = idx(:).';
            for a = idx
                if a < 1 || a > A
                    error('weightEvents:badGroupIdx', ...
                          ['Group %d references attribute %d, out of ' ...
                           'range [1, %d].'], g, a, A);
                end
                if groupOfAttr(a) ~= 0
                    error('weightEvents:duplicateGroupAttr', ...
                          'Attribute %d is listed in more than one group.', a);
                end
                groupOfAttr(a) = g;
            end
        end
        if any(groupOfAttr == 0)
            missing = find(groupOfAttr == 0, 1);
            error('weightEvents:missingGroupAttr', ...
                  'Attribute %d is not assigned to any group.', missing);
        end
        return;
    end
    error('weightEvents:badGroupsType', ...
          'groups must be [], a numeric vector, or a cell array.');
end

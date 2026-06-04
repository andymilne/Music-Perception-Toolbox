function [pAttrOut, wOut, specsOut] = weightEvents( ...
    pAttr, w, inputAttr, targetAttr, centre, shape, nvArgs)
%WEIGHTEVENTS Apply a per-event weight via an input-to-target window factor.
%
%   [pAttrOut, wOut, specsOut] = weightEvents(pAttr, w, ...
%       inputAttr, targetAttr, centre, shape, ...
%       'sd', s,     'deleteInput', tf)
%   [pAttrOut, wOut, specsOut] = weightEvents(pAttr, w, ...
%       inputAttr, targetAttr, centre, shape, ...
%       'width', L,  'deleteInput', tf)
%   is a per-event preprocessing helper for multi-attribute tensor
%   input. It reads the K=1 value at every event from inputAttr,
%   evaluates a window function h centred at centre with shape
%   parameter shape (= gamma), and writes the resulting (1, N)
%   per-event factor into the weight slot of targetAttr, multiplied
%   into any existing weight already there. targetAttr may differ
%   from inputAttr (the typical case --- e.g., time-driven windowing
%   of pitch events) or coincide with it (the input attribute weights
%   itself).
%
%   The window size is specified through exactly one of two
%   Name-Value arguments, 'sd' or 'width'. Both name the same
%   underlying scale on different terms:
%
%     'sd' is the standard deviation of the window. sd = 1.0 gives a
%       Gaussian of standard deviation 1 at shape = 0 and a rectangle
%       whose standard deviation is 1 (i.e., full support 2*sqrt(3))
%       at shape = 1.
%     'width' is the full support of the rectangle at shape = 1.
%       width = 1.0 gives a rectangle on [-1/2, +1/2] at shape = 1
%       and a Gaussian of standard deviation 1/(2*sqrt(3)) at
%       shape = 0. The conversion is sd = width / (2 * sqrt(3)).
%
%   The two conventions exist because each is the natural way to
%   specify the kind of kernel a particular analysis is built around:
%   Gaussian users typically think in standard deviations, rectangle
%   users typically think in full supports. Across the full shape
%   family the SD is held constant regardless of which parameter the
%   caller supplied (variance-normalised behaviour), so the only
%   effect of the parameter choice is the numerical value the user
%   types.
%
%   When deleteInput=true and inputAttr differs from targetAttr, the
%   input attribute is removed from the returned pAttrOut / wOut /
%   specsOut after the factor has been transferred to the target.
%   This is the canonical windowed-entropy / windowed-mass workflow:
%   the input attribute provides the scaffolding for the window and
%   is no longer needed downstream. When deleteInput=false, the input
%   attribute is preserved unchanged in the output. deleteInput=true
%   paired with inputAttr == targetAttr is rejected as incoherent
%   (deleting the input would discard the factor just written to it).
%
%   The window family is the peak-normalised convolution of a
%   rectangle and a Gaussian. Internally, in terms of the standard
%   deviation s (= 'sd' directly, or 'width' / (2 * sqrt(3))):
%
%       phi = s * sqrt(3 * gamma),
%       xi  = s * sqrt(1 - gamma),
%
%   parameterised so the total variance equals s^2 across the whole
%   family. The window is peak-normalised so h(0) = 1.
%
%   Limits:
%       gamma = 0: pure Gaussian h(delta) = exp(-delta^2 / (2 s^2)).
%       gamma = 1: pure rectangle h(delta) = 1[|delta| <= s*sqrt(3)],
%                  i.e., total support 2*s*sqrt(3) (= 'width' when
%                  the caller supplied 'width').
%
%   For a periodic input attribute (isPer = true), the difference
%   delta = v - centre is wrapped to [-P/2, P/2] before applying h;
%   the stored values in pAttr are not modified.
%
%   The per-event factor is broadcast across the target attribute's
%   K_target slots, so every slot of every event sees the same factor.
%
%   Factor entries whose distance from the centre exceeds the global
%   truncationSigmas cutoff (i.e., |delta| > truncationSigmas * s,
%   where s is the kernel's standard deviation, equal to 'sd' or
%   'width' / (2 * sqrt(3))) are hard-zeroed. The threshold is the
%   same one the IP / evaluation kernels use: at that distance a
%   Gaussian window's value is exp(-truncationSigmas^2 / 2). The
%   default global value is Inf (no truncation); set
%   mptDefaults('truncationSigmas', k) to enable hard truncation at
%   k * s.
%
%   Inputs:
%     pAttr        1 x A cell of (K_a, N) per-attribute value matrices.
%                  K_a >= 1.
%     w            Existing weights. [], scalar, or 1 x A cell of
%                  scalar/(1, N)/(K_a, N) entries. None / [] means no
%                  existing weight (factor goes in directly).
%     inputAttr    Scalar integer in [1, A]. The attribute whose K = 1
%                  value supplies the window argument. Must have K = 1.
%     targetAttr   Scalar integer in [1, A]. The attribute whose
%                  weight slot receives the factor. May equal
%                  inputAttr.
%     centre       Scalar finite double. Window centre c.
%     shape        Scalar double in [0, 1]. Shape parameter gamma.
%
%   Name-Value options:
%     specs        Carrier specs: [] (synthesise flat via flatSpecs) or
%                  a 1 x A cell, one spec per attribute. Threaded
%                  through unchanged except that deleteInput=true drops
%                  the input attribute's entry. Not otherwise consulted;
%                  the window is computed from the input attribute's
%                  values, centre, shape, sd/width, and (for a periodic
%                  input) isPer/period.
%     sd           Scalar positive double. Window standard deviation.
%                  Exactly one of 'sd' or 'width' must be supplied.
%     width        Scalar positive double. Full support of the
%                  rectangle at shape = 1; internally translated to
%                  sd = width / (2 * sqrt(3)). Exactly one of 'sd' or
%                  'width' must be supplied.
%     isPer        (1,1) logical, default false. If true, the input
%                  attribute is periodic — delta is wrapped to
%                  [-period/2, period/2] before applying h.
%     period       (1,1) double, default 0. Only used when isPer=true
%                  (must then be > 0).
%     deleteInput  (1,1) logical, REQUIRED (no default; the choice is
%                  destructive enough to be explicit at every call).
%
%   Outputs:
%     pAttrOut     1 x A_out cell of per-attribute value matrices.
%                  Length A if deleteInput=false, A - 1 otherwise.
%     wOut         1 x A_out cell of weights. The targetAttr slot (in
%                  the output indexing) carries the windowed weights.
%     specsOut     1 x A_out cell of carrier specs for the output
%                  attribute list (the input attribute's spec removed
%                  when deleteInput=true).
%
%   See also BUILDEXPTENS, DIFFERENCEEVENTS, BINDEVENTS, TRANSLATEATTRIBUTES,
%            MPTDEFAULTS.

    arguments
        pAttr cell
        w
        inputAttr (1,1) double {mustBeInteger, mustBePositive}
        targetAttr (1,1) double {mustBeInteger, mustBePositive}
        centre (1,1) double
        shape (1,1) double
        nvArgs.specs = []
        nvArgs.sd (1,1) double = NaN
        nvArgs.width (1,1) double = NaN
        nvArgs.isPer (1,1) logical = false
        nvArgs.period (1,1) double = 0
        nvArgs.deleteInput (1,1) logical
    end

    % isPer/period/deleteInput as locals (the rest of the body reads them
    % by these names). deleteInput has no default: omitting it errors when
    % the field is accessed, keeping the destructive choice explicit.
    isPer       = nvArgs.isPer;
    period      = nvArgs.period;
    deleteInput = nvArgs.deleteInput;

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

    % --- Carrier specs: synthesise flat if absent, else validate length ---
    if isempty(nvArgs.specs)
        specsIn = flatSpecs(pAttr);
    else
        specsIn = nvArgs.specs;
        if ~iscell(specsIn) || numel(specsIn) ~= A
            error('weightEvents:badSpecsLength', ...
                  'specs must be a length-A (%d) cell, one per attribute.', A);
        end
    end

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
    if deleteInput && inputAttr == targetAttr
        error('weightEvents:deleteInputIncoherent', ...
              ['deleteInput=true is incoherent when inputAttr == ' ...
               'targetAttr (=%d): deleting the input would discard ' ...
               'the weight factor just written to it. Set ' ...
               'deleteInput=false, or choose a different targetAttr.'], ...
              inputAttr);
    end

    % --- Validate sd/width XOR, centre, shape, period ---
    % Exactly one of nvArgs.sd or nvArgs.width must be supplied
    % (both default to NaN, so use isnan as the "absent" sentinel).
    sdSpec    = ~isnan(nvArgs.sd);
    widthSpec = ~isnan(nvArgs.width);
    if sdSpec == widthSpec
        error('weightEvents:sdWidthXor', ...
              ['weightEvents requires exactly one of ''sd'' or ' ...
               '''width'' (Name-Value). ''sd'' is the window standard ' ...
               'deviation; ''width'' is the full support of the ' ...
               'rectangle at shape=1, equivalent to sd * 2 * sqrt(3). ' ...
               'Got sd=%g, width=%g.'], nvArgs.sd, nvArgs.width);
    end
    if sdSpec
        sd = nvArgs.sd;
        if ~isfinite(sd) || sd <= 0
            error('weightEvents:badSd', ...
                  'sd must be finite and > 0; got %g.', sd);
        end
    else
        if ~isfinite(nvArgs.width) || nvArgs.width <= 0
            error('weightEvents:badWidth', ...
                  'width must be finite and > 0; got %g.', nvArgs.width);
        end
        sd = nvArgs.width / (2 * sqrt(3));
    end
    if ~isfinite(centre)
        error('weightEvents:badCentre', ...
              'centre must be finite; got %g.', centre);
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
    factor = localEvaluateShape(delta, sd, shape);   % (1, N)

    % Truncate: zero factor entries whose distance exceeds
    % truncationSigmas * sd. Uniform convention with the kernel
    % truncation in the IP / eval paths: at that distance a Gaussian
    % window's value is exp(-truncationSigmas^2 / 2), the same
    % threshold the kernel truncation uses. Reads the global default
    % so changes via mptDefaults('truncationSigmas', ...) propagate
    % without an extra kwarg. Inf disables (default).
    truncSig = mptDefaults('truncationSigmas');
    if isfinite(truncSig)
        factor(abs(delta) > truncSig * sd) = 0;
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
        specsOut = specsIn(keep);
    else
        pAttrOut = pAttr;
        specsOut = specsIn;
    end
end


% =========================================================================
%  localEvaluateShape -- peak-normalised fixed-variance window family
% =========================================================================

function h = localEvaluateShape(delta, sd, gamma)
%LOCALEVALUATESHAPE  Peak-normalised rect * Gaussian convolution
%(Section 5.2.1 of the MAET manuscript), with derived parameters
%   phi = sd * sqrt(3 * gamma)   (rectangle half-width)
%   xi  = sd * sqrt(1 - gamma)   (Gaussian std)
%so that the total variance is sd^2 across the whole family.
    if gamma == 0
        % Pure Gaussian, std = sd.
        h = exp(-(delta .^ 2) ./ (2 * sd ^ 2));
        return;
    end
    if gamma == 1
        % Pure rectangle, half-width = sd * sqrt(3).
        phi = sd * sqrt(3);
        h = double(abs(delta) <= phi);
        return;
    end
    phi   = sd * sqrt(3 * gamma);
    xi    = sd * sqrt(1 - gamma);
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

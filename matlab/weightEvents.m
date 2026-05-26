function wOut = weightEvents(pAttr, w, groups, inputAttrs, centre, width, shape, isPer, periods)
%WEIGHTEVENTS Apply a per-event weight via a multi-attribute window.
%
%   wOut = weightEvents(pAttr, w, groups, inputAttrs, centre, width, shape, isPer, periods)
%   is a per-event preprocessing helper for multi-attribute tensor
%   input. It computes a per-event weight contribution from a window
%   in input-attribute space and multiplies it into the existing
%   weight w, returning the new weight wOut. The values and group
%   structure pass through unchanged; only the per-attribute weight
%   slots are updated.
%
%   The window is specified per input attribute by three numeric
%   parameters: centre c_i, the window's standard deviation width_i
%   (in absolute units of the attribute), and a shape parameter
%   shape_i = gamma_i in [0, 1] interpolating between pure Gaussian
%   and pure rectangle. The factor written into the i-th input
%   attribute's weight slot is the peak-normalised convolution
%   rect_{phi_i} * G_{xi_i} with derived sub-parameters
%
%       phi_i = width_i * sqrt(3 * gamma_i),
%       xi_i  = width_i * sqrt(1 - gamma_i),
%
%   chosen so that the rectangular contribution to variance
%   (phi_i^2 / 3 = width_i^2 * gamma_i) and the Gaussian contribution
%   (xi_i^2 = width_i^2 * (1 - gamma_i)) sum to width_i^2 for every
%   gamma_i in [0, 1]. The window's standard deviation is therefore
%   width_i throughout the family, regardless of gamma_i; the window
%   is peak-normalised so h_i(0) = 1.
%
%   Limits (computed directly for numerical cleanliness):
%       gamma_i = 0: pure Gaussian h_i(delta) = exp(-delta^2 / (2 * width_i^2)).
%       gamma_i = 1: pure rectangle h_i(delta) = 1[|delta| <= width_i * sqrt(3)].
%
%   Note that at gamma_i = 1 the rectangle's half-width is
%   width_i * sqrt(3); the parameter width always means the standard
%   deviation, not the half-extent.
%
%   Design P. Each input attribute contributes a separate factor h_i
%   to its own per-attribute weight slot. The kernel product over
%   attributes in buildExpTens then recovers the joint window
%   naturally as W_n = prod_i h_i(p_{a_i, n}). This factoring keeps
%   each input attribute's weight independent and makes weightEvents
%   commute cleanly with translateAttributes (under a centre-shift
%   rule) and pass through differenceEvents / bindEvents.
%
%   For multi-slot attributes (K_a > 1), the window factor is
%   evaluated per slot value: h_i is applied independently to each
%   of the K_a entries in every event column, yielding a K_a x N
%   factor matrix that multiplies the input attribute's weight slot
%   under buildExpTens' broadcast convention.
%
%   The pre-MAET values themselves are not modified by weightEvents.
%   For periodic groups (isPer_i = true), only the difference
%   delta = v - centre_i used inside h_i is wrapped to [-P/2, P/2];
%   the stored values v stay raw. (The kernel in buildExpTens handles
%   the value-axis periodicity downstream via the group [per] flag.)
%
%   Inputs
%       pAttr      - 1 x A cell array of K_a x N per-attribute value
%                    matrices. K_a >= 1; K_a = 0 is rejected.
%       w          - Existing weights to multiply into. [], scalar, or
%                    1 x A cell of per-attribute weight inputs (each
%                    [], scalar, 1 x N row, K_a x 1 column, or K_a x N
%                    matrix). Same convention as buildExpTens.
%       groups     - Group assignment. [] (each attribute its own group),
%                    a 1 x A index vector, or a 1 x G cell of attribute-
%                    index lists. Validated for shape only; not used
%                    by the window math (the relevant [per] / period
%                    info is supplied per input attribute via isPer
%                    and periods).
%       inputAttrs - 1 x M vector of attribute indices (1-based, in
%                    1..A) selecting the attributes the window spans.
%                    Repeats are not allowed.
%       centre     - 1 x M vector of window centres.
%       width      - 1 x M vector of window standard deviations (> 0),
%                    in absolute units of the attribute. The window's
%                    total variance is width(i)^2 for every value of
%                    shape(i).
%       shape      - 1 x M vector of shape parameters gamma in [0, 1].
%                    shape = 0 is pure Gaussian; shape = 1 is pure
%                    rectangle (with half-width width * sqrt(3));
%                    intermediate values are the fixed-variance
%                    convolution family.
%       isPer      - 1 x M logical vector. When isPer(i) is true, the
%                    difference delta = v - centre(i) is wrapped to
%                    [-periods(i)/2, periods(i)/2] before applying the
%                    shape function. Must mirror the [per] flag of
%                    inputAttrs(i)'s group in the downstream
%                    buildExpTens call.
%       periods    - 1 x M vector of periods. Used only when isPer(i)
%                    is true; required to be > 0 in that case.
%
%   Outputs
%       wOut       - 1 x A cell of per-attribute weights, ready to
%                    feed into buildExpTens (possibly after further
%                    pre-MAET chaining). Each input attribute's slot
%                    holds the existing input weight broadcast times
%                    h_i as a K_a x N matrix; non-input attributes
%                    pass through unchanged.
%
%   See also BUILDEXPTENS, DIFFERENCEEVENTS, BINDEVENTS, TRANSLATEATTRIBUTES.

% --- Normalise pAttr ---
if ~iscell(pAttr)
    error('weightEvents:badPAttrType', ...
          'pAttr must be a cell array of per-attribute matrices.');
end
A = numel(pAttr);
if A < 1
    error('weightEvents:noAttrs', ...
          'pAttr must contain at least one attribute.');
end
for a = 1:A
    M = pAttr{a};
    if ~isnumeric(M) || ndims(M) > 2
        error('weightEvents:badAttrShape', ...
              'Attribute %d input must be a numeric 2-D matrix.', a);
    end
    if size(M, 1) == 0
        error('weightEvents:emptyAttribute', ...
              ['Attribute %d has K_a = 0 (empty attribute); empty ' ...
               'attributes are not permitted.'], a);
    end
end

% --- Shared N ---
nEvents = size(pAttr{1}, 2);
for a = 2:A
    if size(pAttr{a}, 2) ~= nEvents
        error('weightEvents:eventCountMismatch', ...
              ['All attributes must share the same event count N. ' ...
               'Attribute 1 has N=%d; attribute %d has N=%d.'], ...
              nEvents, a, size(pAttr{a}, 2));
    end
end

% --- Validate groups (shape only — group internals not used here) ---
[~, ~, ~] = localCanonicaliseGroups(groups, A);

% --- Validate inputAttrs ---
if ~isnumeric(inputAttrs)
    error('weightEvents:badInputAttrsType', ...
          'inputAttrs must be a numeric vector.');
end
inputAttrs = double(inputAttrs(:).');
nInputs = numel(inputAttrs);
if nInputs == 0
    % No window: return existing weights unchanged (canonicalised to cell).
    wOut = localNormaliseWeightsToCell(w, A);
    return;
end
if any(inputAttrs < 1) || any(inputAttrs > A) || ...
        any(inputAttrs ~= round(inputAttrs))
    error('weightEvents:badInputAttrs', ...
          'inputAttrs entries must be integers in 1..A = 1..%d.', A);
end
if numel(unique(inputAttrs)) ~= nInputs
    error('weightEvents:badInputAttrs', ...
          'inputAttrs must not contain repeated indices.');
end

% --- Validate centre, width, shape (all numeric, length M) ---
centre = double(centre(:).');
width  = double(width(:).');
shape  = double(shape(:).');
if numel(centre) ~= nInputs
    error('weightEvents:badCentreLength', ...
          'centre must have length M = %d (one per input attribute); got %d.', ...
          nInputs, numel(centre));
end
if numel(width) ~= nInputs
    error('weightEvents:badWidthLength', ...
          'width must have length M = %d; got %d.', nInputs, numel(width));
end
if numel(shape) ~= nInputs
    error('weightEvents:badShapeLength', ...
          'shape must have length M = %d (gamma per input attribute); got %d.', ...
          nInputs, numel(shape));
end
if any(~isfinite(centre)) || any(~isfinite(width)) || any(~isfinite(shape))
    error('weightEvents:nonFiniteParam', ...
          'centre, width, and shape entries must be finite.');
end
if any(width <= 0)
    error('weightEvents:badWidth', ...
          'width entries must be > 0.');
end
if any(shape < 0) || any(shape > 1)
    error('weightEvents:badShape', ...
          ['shape entries (gamma) must lie in [0, 1]: gamma = 0 is ' ...
           'pure Gaussian, gamma = 1 is pure rectangle, intermediate ' ...
           'values are the fixed-variance convolution family.']);
end

% --- Validate isPer, periods ---
isPer = logical(isPer(:).');
if numel(isPer) ~= nInputs
    error('weightEvents:badIsPerLength', ...
          'isPer must have length M = %d; got %d.', nInputs, numel(isPer));
end
periods = double(periods(:).');
if numel(periods) ~= nInputs
    error('weightEvents:badPeriodsLength', ...
          'periods must have length M = %d; got %d.', nInputs, numel(periods));
end
if any(isPer & (periods <= 0))
    error('weightEvents:badPeriods', ...
          ['periods entries must be > 0 wherever isPer is true ' ...
           '(non-periodic entries are unused and may be 0).']);
end

% --- Compute factor h_i for each input attribute ---
factors = cell(1, nInputs);
for i = 1:nInputs
    a = inputAttrs(i);
    valMat = double(pAttr{a});   % K_a x N
    delta = valMat - centre(i);
    if isPer(i)
        P = periods(i);
        delta = delta - P .* floor(delta ./ P + 0.5);
    end
    factors{i} = localEvaluateShape(delta, width(i), shape(i));
end

% --- Normalise w to 1xA cell and multiply factors in ---
wOut = localNormaliseWeightsToCell(w, A);
for i = 1:nInputs
    a = inputAttrs(i);
    wOut{a} = localMultiplyWeights(wOut{a}, factors{i}, a);
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
            error('weightEvents:badWeightsLength', ...
                  'Weight cell must have length A = %d; got %d.', ...
                  A, numel(w));
        end
        for a = 1:A
            wCell{a} = w{a};
        end
        return;
    end
    error('weightEvents:badWeightsType', ...
          ['w must be [], a scalar, or a 1 x A cell of per-' ...
           'attribute weight inputs.']);
end


% =========================================================================
%  localMultiplyWeights
% =========================================================================

function out = localMultiplyWeights(wExisting, factor, attrIdx)
%LOCALMULTIPLYWEIGHTS  Multiply existing weight by factor (K_a x N).
%Broadcasting follows the toolbox convention (scalar / 1 x N row /
%K_a x 1 column / K_a x N matrix all multiply naturally into the
%K_a x N factor via MATLAB's implicit expansion).
    if isempty(wExisting)
        out = factor;
        return;
    end
    if isnumeric(wExisting) && isscalar(wExisting)
        out = double(wExisting) .* factor;
        return;
    end
    if ~isnumeric(wExisting)
        error('weightEvents:badWeightType', ...
              'Attribute %d weight must be numeric.', attrIdx);
    end
    out = double(wExisting) .* factor;
end


% =========================================================================
%  localCanonicaliseGroups (validation only)
% =========================================================================

function [groupOfAttr, attrsOfGroup, G] = localCanonicaliseGroups(groups, A)
    if isempty(groups)
        groupOfAttr = 1:A;
    elseif iscell(groups)
        G = numel(groups);
        groupOfAttr = zeros(1, A);
        for g = 1:G
            idx = groups{g};
            if any(idx < 1) || any(idx > A) || any(groupOfAttr(idx) ~= 0)
                error('weightEvents:badGroups', ...
                      'Invalid cell-form groups specification.');
            end
            groupOfAttr(idx) = g;
        end
        if any(groupOfAttr == 0)
            error('weightEvents:badGroups', ...
                  'Every attribute must appear in exactly one group.');
        end
    elseif isnumeric(groups) && numel(groups) == A
        groupOfAttr = double(groups(:).');
        if any(groupOfAttr < 1) || any(groupOfAttr ~= round(groupOfAttr))
            error('weightEvents:badGroups', ...
                  'Numeric groups must be positive integers.');
        end
    else
        error('weightEvents:badGroupsShape', ...
              ['groups must be [], a length-A numeric vector, or a ' ...
               'cell of index lists.']);
    end
    G = max(groupOfAttr);
    attrsOfGroup = cell(1, G);
    for g = 1:G
        attrsOfGroup{g} = find(groupOfAttr == g);
    end
end

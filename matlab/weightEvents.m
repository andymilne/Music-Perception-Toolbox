function pm = weightEvents(varargin)
%WEIGHTEVENTS Apply a per-event weight via an input-to-target window factor.
%
%   PM = weightEvents(PM0, inputAttr, targetAttr, centre, shape, ...
%       'sd', s,     'dropInputAttr', tf)
%   PM = weightEvents(pAttr, wAttr, inputAttr, targetAttr, centre, shape, ...
%       'width', L,  'dropInputAttr', tf)
%   is a per-event preprocessing helper for multi-attribute tensor input.
%   It reads one value per event from inputAttr (where an event holds
%   several, the one 'locate' picks), evaluates the profile shape centred
%   at centre, and writes the resulting (1, N)
%   per-event factor into the weight entry of targetAttr, multiplied into
%   any existing weight already there. targetAttr may differ from inputAttr
%   (the typical case --- e.g., time-driven windowing of pitch events) or
%   coincide with it (the input attribute weights itself).
%
%   The window size is specified through exactly one of two Name-Value
%   arguments, 'sd' or 'width'. Both name the same underlying scale on
%   different terms:
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
%   The two conventions exist because each is the natural way to specify
%   the kind of kernel a particular analysis is built around: Gaussian
%   users typically think in standard deviations, rectangle users typically
%   think in full supports. Across the full shape family the SD is held
%   constant regardless of which parameter the caller supplied
%   (variance-normalised behaviour), so the only effect of the parameter
%   choice is the numerical value the user types.
%
%   When dropInputAttr=true and inputAttr differs from targetAttr, the
%   input attribute is removed from the returned pAttrOut / wOut / specsOut
%   after the factor has been transferred to the target. This is the
%   canonical windowed-entropy / windowed-mass workflow: the input
%   attribute provides the scaffolding for the window and is no longer
%   needed downstream. When dropInputAttr=false, the input attribute is
%   preserved unchanged in the output. dropInputAttr=true paired with
%   inputAttr == targetAttr is rejected as incoherent (deleting the input
%   would discard the factor just written to it).
%
%   The window family is the peak-normalised convolution of a rectangle and
%   a Gaussian. Internally, in terms of the standard deviation s (= 'sd'
%   directly, or 'width' / (2 * sqrt(3))):
%
%       phi = s * sqrt(3 * gamma),
%       xi  = s * sqrt(1 - gamma),
%
%   parameterised so the total variance equals s^2 across the whole
%   family. The window is peak-normalised so h(0) = 1.
%
%   Limits:
%       gamma = 0: pure Gaussian h(delta) = exp(-delta^2 / (2 s^2)).
%       gamma = 1: pure rectangle h(delta) = 1[-s*sqrt(3) <= delta <
%                  s*sqrt(3)], i.e., total support 2*s*sqrt(3) (=
%                  'width' when the caller supplied 'width'); half-open,
%                  so rectangles a width apart share no event, unless
%                  'edges' is 'closed'.
%
%   For a periodic input attribute (isPer = true), the difference delta = v
%   - centre is wrapped to [-P/2, P/2] before applying h; the stored values
%   in pAttr are not modified.
%
%   The per-event factor is broadcast across the target attribute's
%   K_target values, so every value of every event sees the same factor.
%
%   Factor entries whose distance from the centre exceeds the global
%   truncationSigmas cutoff (i.e., |delta| > truncationSigmas * s, where s
%   is the kernel's standard deviation, equal to 'sd' or 'width' / (2 *
%   sqrt(3))) are hard-zeroed. The threshold is the same one the IP /
%   evaluation kernels use: at that distance a Gaussian window's value is
%   exp(-truncationSigmas^2 / 2). The default global value is 6; set
%   mptDefaults('truncationSigmas', Inf) for no truncation, or another k to
%   change the hard-truncation distance at k * s.
%
%   Inputs:
%     pm           Pre-MAET, in place of pAttr and wAttr. The
%                  pre-MAET may be passed whole or in its parts; the two
%                  forms are the same call.
%     pAttr        1 x A cell of (K_a, N) per-attribute value matrices, or
%                  of attributes given per event (see packPreMaet).
%                  K_a >= 1.
%     wAttr        Existing weights. [], scalar, or 1 x A cell of
%                  scalar/(1, N)/(K_a, N) entries. None / [] means no
%                  existing weight (factor goes in directly).
%     inputAttr    Scalar integer in [1, A]. The attribute whose values
%                  supply the window argument. Where an event holds
%                  several values there (the onsets of a bound
%                  super-event, say), 'locate' picks the one the profile
%                  is evaluated at.
%     targetAttr   Scalar integer in [1, A]. The attribute whose
%                  weight entry receives the factor. May equal
%                  inputAttr.
%     centre       Scalar finite double. Window centre c.
%     shape        The profile. One of three kinds:
%                    - a scalar double in [0, 1]: the shape parameter
%                      gamma of the rectangle-Gaussian family, taking
%                      exactly one of 'sd' or 'width';
%                    - 'exponential', 'exponentialBefore', or
%                      'exponentialAfter': exponential decay away from
%                      the centre, in both directions or in one, scaled
%                      by 'sd' or 'decayRate';
%                    - 'exponentialFromStart', 'exponentialFromEnd',
%                      'uShape', or 'uAsym': serial-position profiles,
%                      anchored at the first and last events' values
%                      rather than at a centre, which must be NaN. The
%                      first two decay away from one anchor; 'uShape'
%                      mixes both at one rate and 'uAsym' at two.
%                      Applied to an event-number attribute, dropped
%                      afterwards with dropInputAttr, they are profiles
%                      over position; applied to a time attribute, over
%                      elapsed time;
%                    - a function handle f(delta) returning one
%                      non-negative factor per event, where delta is
%                      the centred input values. It carries its own
%                      scale, so neither 'sd' nor 'width' is accepted,
%                      and the kernel truncation is not applied to it.
%
%   Name-Value options:
%     specs        Attribute specifications: [] (synthesise flat via flatSpecs) or
%                  a 1 x A cell, one spec per attribute. Threaded
%                  through unchanged except that dropInputAttr=true drops
%                  the input attribute's entry. Not otherwise consulted;
%                  the window is computed from the input attribute's
%                  values, centre, shape, sd/width, and (for a periodic
%                  input) isPer/period.
%     sd           Scalar positive double. Profile standard deviation.
%                  For a numeric shape, exactly one of 'sd' or 'width'
%                  must be supplied; a named profile takes either 'sd'
%                  or 'decayRate', not both.
%     decayRate    Positive decay rate for the named exponential
%                  profiles, the reciprocal of the decay constant in
%                  the input attribute's units. Used as the default for
%                  'uAsym' when 'decayRateStart' or 'decayRateEnd' are
%                  not supplied. Default 1.
%     decayRateStart
%                  Decay rate for the primacy component of 'uAsym'.
%                  Falls back to 'decayRate' when []. Default [].
%     decayRateEnd Decay rate for the recency component of 'uAsym'.
%                  Falls back to 'decayRate' when []. Default [].
%     alpha        Mixing parameter in [0, 1] for 'uShape' and 'uAsym'.
%                  alpha = 1 is pure primacy, alpha = 0 pure recency,
%                  alpha = 0.5 a balanced mix. Default 0.5.
%     width        Scalar positive double. Full support of the
%                  rectangle at shape = 1; internally translated to
%                  sd = width / (2 * sqrt(3)). Exactly one of 'sd' or
%                  'width' must be supplied.
%     isPer        (1,1) logical, default false. If true, the input
%                  attribute is periodic — delta is wrapped to
%                  [-period/2, period/2] before applying h.
%     period       (1,1) double, default 0. Only used when isPer=true
%                  (must then be > 0).
%     locate       The single value that stands for an event holding
%                  several values on inputAttr: 'centroid' (their mean,
%                  the default), 'start' (the first), 'end' (the last),
%                  'mid' (the midpoint of the first and last), or a
%                  function handle taking the K x N value matrix and
%                  returning N values. No effect where each event holds
%                  one value.
%     edges        For a rectangle (shape = 1): 'halfOpen' (the default)
%                  keeps the lower edge and not the upper, so rectangles a
%                  width apart share no event; 'closed' keeps both.
%                  Refused for other profiles.
%     dropInputAttr  (1,1) logical, REQUIRED (no default; the choice is
%                  destructive enough to be explicit at every call).
%
%   Output:
%     pm           Pre-MAET. Its pAttr is a 1 x A_out cell of
%                  per-attribute value matrices, of length A if
%                  dropInputAttr=false and A - 1 otherwise; its wAttr a
%                  1 x A_out cell of weights whose targetAttr entry (in
%                  the output indexing) carries the windowed weights.
%                  Its specs is a 1 x A_out cell of attribute
%                  specifications for the output attribute list (the input
%                  attribute's spec removed when dropInputAttr=true).
%
%   See also PACKPREMAET, BUILDMAET, DIFFERENCEEVENTS, BINDEVENTS,
%            TRANSLATEATTRIBUTES, MPTDEFAULTS.

    [pAttr, wAttr, specsPm, rest] = internal.preMaetArgs(varargin);
    [pAttrOut, wOut, specsOut] = localWeightEvents( ...
        pAttr, wAttr, specsPm, rest{:});
    pm = packPreMaet(pAttrOut, wOut, specsOut);
end


function [pAttrOut, wOut, specsOut] = localWeightEvents( ...
    pAttr, w, specsPm, inputAttr, targetAttr, centre, shape, nvArgs)
    arguments
        pAttr cell
        w
        specsPm
        inputAttr (1,1) double {mustBeInteger, mustBePositive}
        targetAttr (1,1) double {mustBeInteger, mustBePositive}
        centre (1,1) double
        shape
        nvArgs.specs = []
        nvArgs.sd (1,1) double = NaN
        nvArgs.width (1,1) double = NaN
        nvArgs.decayRate (1,1) double = NaN
        nvArgs.decayRateStart = []
        nvArgs.decayRateEnd = []
        nvArgs.alpha (1,1) double {mustBeInRange(nvArgs.alpha, 0, 1)} = 0.5
        nvArgs.isPer (1,1) logical = false
        nvArgs.period (1,1) double = 0
        nvArgs.locate = 'centroid'
        nvArgs.edges = ''
        nvArgs.dropInputAttr (1,1) logical
    end

    if isempty(nvArgs.specs)
        nvArgs.specs = specsPm;
    end

    % isPer/period/dropInputAttr as locals (the rest of the body reads them
    % by these names). dropInputAttr has no default: omitting it errors when
    % the field is accessed, keeping the destructive choice explicit.
    isPer       = nvArgs.isPer;
    period      = nvArgs.period;
    dropInputAttr = nvArgs.dropInputAttr;

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

    % --- Attribute specifications: synthesise flat if absent, else validate length ---
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

    % --- Validate targetAttr ---
    if targetAttr > A
        error('weightEvents:badTargetAttr', ...
              'targetAttr must be in 1..%d; got %d.', A, targetAttr);
    end

    % --- Validate dropInputAttr ---
    if dropInputAttr && inputAttr == targetAttr
        error('weightEvents:dropInputAttrIncoherent', ...
              ['dropInputAttr=true is incoherent when inputAttr == ' ...
               'targetAttr (=%d): deleting the input would discard ' ...
               'the weight factor just written to it. Set ' ...
               'dropInputAttr=false, or choose a different targetAttr.'], ...
              inputAttr);
    end

    % --- Validate the profile, its scale, centre, and period ---
    prof = internal.resolveProfile(shape, nvArgs.sd, nvArgs.width, ...
        nvArgs.decayRate, nvArgs.decayRateStart, nvArgs.decayRateEnd, ...
        nvArgs.alpha, 'weightEvents');
    closed = internal.resolveEdges(nvArgs.edges, prof, 'weightEvents');
    if strcmp(prof.kind, 'anchored')
        if ~isnan(centre)
            error('weightEvents:centreWithAnchoredProfile', ...
                  ['Profile ''%s'' anchors itself at the first and ' ...
                   'last events'' values, so centre does not apply; ' ...
                   'pass NaN.'], char(shape));
        end
    elseif ~isfinite(centre)
        error('weightEvents:badCentre', ...
              'centre must be finite; got %g.', centre);
    end
    if isPer && period <= 0
        error('weightEvents:badPeriod', ...
              'period must be > 0 when isPer is true; got %g.', period);
    end

    % --- Compute factor h(delta) from the input attribute's values ---
    % Each event is represented by one value: its only value, or, where it
    % holds several (a bound super-event's onsets), the one locate picks.
    valRow = internal.locateRow(pAttr{inputAttr}, nvArgs.locate);   % (1, N)
    factor = internal.weightFactor(valRow, centre, prof, isPer, period, ...
        closed);

    % --- Normalise w to length-A cell; multiply factor into target entry ---
    wOut = internal.normaliseWeightsToCell(w, A);
    wOut{targetAttr} = internal.multiplyWeights( ...
        wOut{targetAttr}, factor, size(pAttr{targetAttr}, 1));

    % --- Build output structures, applying dropInputAttr if requested ---
    if dropInputAttr
        keep = setdiff(1:A, inputAttr);
        pAttrOut = pAttr(keep);
        wOut = wOut(keep);
        specsOut = specsIn(keep);
    else
        pAttrOut = pAttr;
        specsOut = specsIn;
    end
end

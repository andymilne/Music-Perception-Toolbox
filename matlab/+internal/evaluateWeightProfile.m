function h = evaluateWeightProfile(vals, delta, sd, shape, opts)
%EVALUATEWEIGHTPROFILE  Per-event weight profile.
%
%   H = INTERNAL.EVALUATEWEIGHTPROFILE(VALS, DELTA, SD, SHAPE, OPTS)
%   returns a 1 x N row of non-negative factors. VALS is the input
%   attribute's 1 x N row and DELTA the same row centred (and wrapped,
%   where the attribute is periodic). SHAPE selects one of four
%   families:
%
%     numeric gamma in [0, 1]
%         The fixed-variance rectangle-Gaussian convolution family of
%         INTERNAL.EVALUATESHAPE, centred at the caller's centre, with
%         standard deviation SD.
%
%     'exponential', 'exponentialBefore', 'exponentialAfter'
%         Exponential decay away from the centre, in both directions or
%         in one (zero on the other side). OPTS.tau is the decay's
%         constant in the attribute's own units.
%
%     'exponentialFromStart', 'exponentialFromEnd', 'uShape', 'uAsym'
%         Serial-position profiles, anchored at the first and last
%         events' values rather than at a centre. The first two decay
%         away from one anchor; 'uShape' mixes both at one rate and
%         'uAsym' at two, OPTS.alpha weighting the primacy component
%         against the recency one.
%
%     function handle
%         Any user profile f(DELTA) returning non-negative factors the
%         size of DELTA.
%
%   See also INTERNAL.EVALUATESHAPE, WEIGHTEVENTS.

    if isa(shape, 'function_handle')
        h = shape(delta);
        if ~isnumeric(h) || ~isreal(h)
            error('weightEvents:profileNotNumeric', ...
                  'The profile function must return real numeric values.');
        end
        h = double(h);
        if ~isequal(size(h), size(delta))
            error('weightEvents:profileSize', ...
                  ['The profile function must return one factor per ' ...
                   'event: expected %s, got %s.'], ...
                  mat2str(size(delta)), mat2str(size(h)));
        end
        if any(~isfinite(h)) || any(h < 0)
            error('weightEvents:profileNotNonNegative', ...
                  ['The profile function must return finite, ' ...
                   'non-negative factors.']);
        end
        return;
    end

    if ~(ischar(shape) || isstring(shape))
        h = internal.evaluateShape(delta, sd, shape);
        return;
    end

    name = char(shape);
    switch name
        case 'exponential'
            h = exp(-abs(delta) / opts.tau);
        case 'exponentialBefore'
            h = zeros(size(delta));
            m = delta <= 0;
            h(m) = exp(delta(m) / opts.tau);
        case 'exponentialAfter'
            h = zeros(size(delta));
            m = delta >= 0;
            h(m) = exp(-delta(m) / opts.tau);
        case 'exponentialFromStart'
            h = exp(-abs(vals - vals(1)) / opts.tauStart);
        case 'exponentialFromEnd'
            h = exp(-abs(vals(end) - vals) / opts.tauEnd);
        case {'uShape', 'uAsym'}
            hStart = exp(-abs(vals - vals(1)) / opts.tauStart);
            hEnd   = exp(-abs(vals(end) - vals) / opts.tauEnd);
            h = opts.alpha * hStart + (1 - opts.alpha) * hEnd;
        otherwise
            error('weightEvents:badShapeName', ...
                  ['Unknown profile ''%s''. The named profiles are ' ...
                   '''exponential'', ''exponentialBefore'', ' ...
                   '''exponentialAfter'', ''exponentialFromStart'', ' ...
                   '''exponentialFromEnd'', ''uShape'', and ''uAsym''; ' ...
                   'a numeric shape in [0, 1] selects the ' ...
                   'rectangle-Gaussian family, and a function handle ' ...
                   'supplies any other profile.'], name);
    end
end

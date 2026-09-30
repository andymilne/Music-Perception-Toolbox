function prof = resolveProfile(shape, sd, width, decayRate, decayRateStart, ...
        decayRateEnd, alpha, prefix)
%RESOLVEPROFILE  Validate a weighting profile and its scale, as weightEvents
%   and the swept functions take them, and return it as a struct with
%   fields shape, sd, opts, and kind. Twin of the Python _resolve_profile.
%
%   kind is 'family' (the rectangle-Gaussian family, shape its gamma),
%   'named' (an exponential aligned at a reference value), 'anchored' (a
%   serial-position profile, anchored at the first and last events'
%   values), or 'handle' (a function of the displacement). sd is the
%   profile's scale in the input attribute's units (the decay constant for
%   an exponential; NaN for a handle).
%
%   SD, WIDTH, and DECAYRATE are NaN when absent; DECAYRATESTART and
%   DECAYRATEEND are [] when absent. Errors carry the identifier prefix
%   PREFIX ('weightEvents', say).
    if nargin < 8 || isempty(prefix), prefix = 'weightEvents'; end
    if nargin < 7 || isempty(alpha), alpha = 0.5; end
    if nargin < 6, decayRateEnd = []; end
    if nargin < 5, decayRateStart = []; end
    sdSpec    = ~isnan(sd);
    widthSpec = ~isnan(width);
    rateSpec  = ~isnan(decayRate);
    opts = struct('tau', NaN, 'tauStart', NaN, 'tauEnd', NaN, 'alpha', alpha);
    ANCHORED = {'exponentialFromStart', 'exponentialFromEnd', ...
                'uShape', 'uAsym'};
    NAMED = [{'exponential', 'exponentialBefore', 'exponentialAfter'}, ...
             ANCHORED];
    if (ischar(shape) || isstring(shape)) && ...
            any(strcmpi(char(shape), {'rect', 'rectangular', 'box', ...
                                      'gaussian', 'gauss', 'normal'}))
        shape = internal.resolveShape(shape);
    end
    if isa(shape, 'function_handle')
        if sdSpec || widthSpec || rateSpec
            error([prefix ':scaleWithProfile'], ...
                  ['A profile function carries its own scale, so ' ...
                   '''sd'', ''width'', and ''decayRate'' are not ' ...
                   'accepted with one.']);
        end
        prof = struct('shape', {shape}, 'sd', NaN, 'opts', opts, ...
                      'kind', 'handle');
        return
    end
    if ischar(shape) || isstring(shape)
        shape = char(shape);
        if ~ismember(shape, NAMED)
            error([prefix ':badShapeName'], ...
                  ['Unknown profile ''%s''. The named profiles are ' ...
                   '''exponential'', ''exponentialBefore'', ' ...
                   '''exponentialAfter'', ''exponentialFromStart'', ' ...
                   '''exponentialFromEnd'', ''uShape'', and ''uAsym''; ' ...
                   '''rect'', ''gaussian'', or a numeric shape in [0, 1] ' ...
                   'selects the rectangle-Gaussian family, and a function ' ...
                   'handle supplies any other profile.'], shape);
        end
        if widthSpec
            error([prefix ':widthWithNamedProfile'], ...
                  ['''width'' describes the support of the rectangle ' ...
                   'and does not apply to profile ''%s''. Give ''sd'' ' ...
                   '(the profile standard deviation) or ''decayRate'' ' ...
                   '(its reciprocal).'], shape);
        end
        if sdSpec && rateSpec
            error([prefix ':sdRateXor'], ...
                  ['''sd'' and ''decayRate'' are two spellings of one ' ...
                   'scale; give one, not both.']);
        end
        % The decay constant tau, in the input attribute's own units. sd is
        % its standard deviation and decayRate its reciprocal; neither
        % given, the rate is 1.
        if sdSpec
            if ~isfinite(sd) || sd <= 0
                error([prefix ':badSd'], ...
                      'sd must be finite and > 0; got %g.', sd);
            end
            tau = sd;
            if strcmp(shape, 'exponential')
                % Two-sided: the Laplace standard deviation is tau*sqrt(2),
                % so an sd holds the variance across the family.
                tau = sd / sqrt(2);
            end
        else
            rate = 1;
            if rateSpec, rate = decayRate; end
            if ~isfinite(rate) || rate <= 0
                error([prefix ':badDecayRate'], ...
                      'decayRate must be finite and > 0; got %g.', rate);
            end
            tau = 1 / rate;
            sd = tau;
        end
        opts.tau = tau; opts.tauStart = tau; opts.tauEnd = tau;
        if strcmp(shape, 'uAsym')
            if ~isempty(decayRateStart)
                opts.tauStart = localTauFromRate(decayRateStart, ...
                    'decayRateStart', prefix);
            end
            if ~isempty(decayRateEnd)
                opts.tauEnd = localTauFromRate(decayRateEnd, ...
                    'decayRateEnd', prefix);
            end
        elseif ~isempty(decayRateStart) || ~isempty(decayRateEnd)
            error([prefix ':asymRatesWithSymmetricProfile'], ...
                  ['''decayRateStart'' and ''decayRateEnd'' apply to ' ...
                   '''uAsym'' alone; profile ''%s'' has one rate.'], shape);
        end
        if ismember(shape, ANCHORED), kind = 'anchored'; else, kind = 'named'; end
        prof = struct('shape', shape, 'sd', sd, 'opts', opts, 'kind', kind);
        return
    end
    if ~isnumeric(shape) || ~isscalar(shape)
        error([prefix ':badShape'], ...
              ['shape must be a scalar in [0, 1], ''rect'', ''gaussian'', ' ...
               'a named profile, or a function handle.']);
    end
    if rateSpec || ~isempty(decayRateStart) || ~isempty(decayRateEnd)
        error([prefix ':rateWithNumericShape'], ...
              ['''decayRate'' and its asymmetric companions apply to the ' ...
               'named exponential profiles; the rectangle-Gaussian family ' ...
               'is scaled by ''sd'' or ''width''.']);
    end
    % Exactly one of sd or width must be supplied (both NaN when absent).
    if sdSpec == widthSpec
        error([prefix ':sdWidthXor'], ...
              ['The rectangle-Gaussian family takes exactly one of ''sd'' ' ...
               'or ''width''. ''sd'' is the window standard deviation; ' ...
               '''width'' is the full support of the rectangle at ' ...
               'shape=1, equivalent to sd * 2 * sqrt(3). Got sd=%g, ' ...
               'width=%g.'], sd, width);
    end
    if sdSpec
        if ~isfinite(sd) || sd <= 0
            error([prefix ':badSd'], 'sd must be finite and > 0; got %g.', sd);
        end
    else
        if ~isfinite(width) || width <= 0
            error([prefix ':badWidth'], ...
                  'width must be finite and > 0; got %g.', width);
        end
        sd = width / (2 * sqrt(3));
    end
    if shape < 0 || shape > 1
        error([prefix ':badShape'], ...
              ['shape (gamma) must lie in [0, 1]: gamma = 0 is pure ' ...
               'Gaussian, gamma = 1 is pure rectangle, intermediate values ' ...
               'are the fixed-variance convolution family. Got %g.'], shape);
    end
    prof = struct('shape', double(shape), 'sd', sd, 'opts', opts, ...
                  'kind', 'family');
end


function tau = localTauFromRate(rate, name, prefix)
%LOCALTAUFROMRATE  Decay constant from a positive rate.
    if ~isnumeric(rate) || ~isscalar(rate) || ~isfinite(rate) || rate <= 0
        error([prefix ':badDecayRate'], '%s must be a finite scalar > 0.', name);
    end
    tau = 1 / rate;
end

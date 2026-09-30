function factor = weightFactor(valRow, reference, prof, isPer, period, closed)
%WEIGHTFACTOR  The per-event factor h(value - reference), 1 x N, of a
%   resolved profile (internal.resolveProfile): the one implementation of
%   event weighting, shared by weightEvents and the swept functions.
%   Twin of the Python _weight_factor.
%
%   On a periodic attribute (ISPER) the displacement is wrapped to
%   [-PERIOD/2, PERIOD/2). A rectangle is half-open (its lower edge
%   included, its upper not) unless CLOSED. Entries farther than
%   truncationSigmas times the profile's scale are hard-zeroed, as the
%   kernel truncation does; a function handle or an anchored profile is not
%   truncated.
    if nargin < 4 || isempty(isPer), isPer = false; end
    if nargin < 5 || isempty(period), period = 0; end
    if nargin < 6 || isempty(closed), closed = false; end
    valRow = double(valRow);
    if strcmp(prof.kind, 'anchored')
        delta = valRow;            % unused by the anchored profiles
    else
        delta = valRow - reference;
    end
    if isPer
        delta = delta - period * floor(delta / period + 0.5);
    end
    factor = internal.evaluateWeightProfile(valRow, delta, prof.sd, ...
        prof.shape, prof.opts);
    if closed
        % Both edges included: |delta| <= phi, with the same tolerance as
        % the half-open test of internal.evaluateShape.
        phi = prof.sd * sqrt(3);
        scale = max(abs(phi), 1);
        fin = abs(delta(isfinite(delta)));
        if ~isempty(fin), scale = max(scale, max(fin)); end
        factor(abs(delta) <= phi + 1e-9 * scale) = 1;
    end
    % Truncate: zero factor entries whose distance exceeds truncationSigmas
    % * sd, the threshold the kernel truncation uses. Per the truncation
    % contract, Inf resolves to the finite accuracy-floor width, so
    % truncation always applies --- never a "disabled" state.
    if any(strcmp(prof.kind, {'family', 'named'}))
        truncSig = internal.accuracyFloor('resolve', ...
            mptDefaults('truncationSigmas'));
        factor(abs(delta) > truncSig * prof.sd) = 0;
    end
end

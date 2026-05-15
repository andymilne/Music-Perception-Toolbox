function maybeShowDispatchMsg(varargin)
%INTERNAL.MAYBESHOWDISPATCHMSG  Print a dispatch message at most once per session.
%
%   internal.maybeShowDispatchMsg(funcName, chosen, routingReason, ...
%                                 estSec, isProbed)
%       Print a dispatch-decision message for the (funcName, chosen,
%       routingReason) triple, but only if that exact triple has not
%       been printed before in this MATLAB session.
%
%       When isProbed is true, the message includes the empirical
%       extrapolated estimate from the probe:
%           "<funcName>: chose '<chosen>' path (estimated X s);
%            Ctrl+C to cancel."
%
%       When isProbed is false, the message reports the routing
%       reason (e.g., a hard rule or analytical pre-screen):
%           "<funcName>: chose '<chosen>' path (<routingReason>)."
%
%   internal.maybeShowDispatchMsg('reset')
%       Clear the seen-set so that all dispatch messages will fire
%       again on their next call. Called by mptDefaults('reset').
%
%   The seen-set persists within the MATLAB session and is also
%   cleared by `clear all` or `clear internal.maybeShowDispatchMsg`.
%
%   Gating (v2.2.x): dispatch messages are NOT gated by per-call
%   verbose. They are gated by the toolbox-wide showHints flag
%   (mptDefaults('showHints')), matching the kernel-evaluation
%   hint's gating model. Rationale: internal toolbox callers (e.g.,
%   the batched-raw path inside cosSimExpTens, entropyExpTens's
%   evaluation step) routinely pass verbose=false to inner calls to
%   prevent flooding. With the once-per-session throttle in place,
%   flooding is no longer a concern, and users benefit from seeing
%   the routing decision even when internal callers pass verbose=false.
%   To fully silence dispatch messages: mptDefaults('showHints', false).
%
%   Design rationale: dispatch decisions are interesting the first
%   time they happen but redundant when the same call is repeated in
%   a loop (e.g., demo_edoApprox computes SPCS 101 times, all hitting
%   the same hard-rule branch). The once-per-session throttle gives
%   the user one informative message per unique decision and stays
%   quiet thereafter. Parallels the existing INTERNAL.MAYBESHOWKERNELEVALHINT
%   throttle, and mptDefaults('reset') clears both.
%
%   See also: MPTDEFAULTS, INTERNAL.MAYBESHOWKERNELEVALHINT.

    persistent seen
    if isempty(seen)
        seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    end

    % 'reset' form: clear all seen entries.
    if nargin == 1 && (ischar(varargin{1}) || isstring(varargin{1})) ...
            && strcmpi(varargin{1}, 'reset')
        seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
        return;
    end

    if nargin ~= 5
        error('internal:maybeShowDispatchMsg:badArgs', ...
            ['Call form is internal.maybeShowDispatchMsg(funcName, ' ...
             'chosen, routingReason, estSec, isProbed) or ' ...
             'internal.maybeShowDispatchMsg(''reset'').']);
    end

    % Master switch: showHints false → silent for all dispatch messages.
    S = mptDefaults();
    if isfield(S, 'showHints') && ~S.showHints
        return;
    end

    funcName      = char(varargin{1});
    chosen        = char(varargin{2});
    routingReason = char(varargin{3});
    estSec        = double(varargin{4});
    isProbed      = logical(varargin{5});

    key = sprintf('%s|%s|%s', funcName, chosen, routingReason);
    if isKey(seen, key)
        return;
    end
    seen(key) = true;

    if isProbed
        fprintf('%s: chose ''%s'' path (estimated %s); Ctrl+C to cancel.\n', ...
                funcName, chosen, localFormatTime(estSec));
    else
        fprintf('%s: chose ''%s'' path (%s).\n', ...
                funcName, chosen, routingReason);
    end
end


function s = localFormatTime(t)
%LOCALFORMATTIME  Short human-readable duration string.
    if t < 1
        s = sprintf('%.0f ms', t * 1000);
    elseif t < 60
        s = sprintf('%.1f s', t);
    elseif t < 3600
        s = sprintf('%.1f min', t / 60);
    else
        s = sprintf('%.1f hr', t / 3600);
    end
end

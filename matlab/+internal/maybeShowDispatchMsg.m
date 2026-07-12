function maybeShowDispatchMsg(varargin)
%INTERNAL.MAYBESHOWDISPATCHMSG  Print a dispatch message at most once per top-level call.
%
%   internal.maybeShowDispatchMsg(funcName, chosen, routingReason, ...
%                                 estSec, isProbed)
%       Print a dispatch-decision message for the (funcName, chosen)
%       pair, but only if that exact pair has not already been printed
%       in the current top-level user call.
%
%       When isProbed is true, the message includes the empirical
%       extrapolated estimate from the probe:
%           "<funcName>: chose '<chosen>' path (estimated X s);
%            Ctrl+C to cancel."
%
%       When isProbed is false, the message is just the path:
%           "<funcName>: chose '<chosen>' path."
%       The routingReason argument is still accepted (callers continue
%       to pass it, since it remains useful for downstream debugging
%       and for the probed-form estimate string), but is not part of
%       the throttle key — within a single top-level call, two
%       different routing reasons that lead to the same chosen path
%       collapse to one announce. The visible distinction that matters
%       is which path ran, not why; throttling on what the user sees
%       avoids apparent duplicates.
%
%   internal.maybeShowDispatchMsg('reset')
%       Clear the seen-set so that all dispatch messages will fire
%       again on their next call. Called automatically by
%       INTERNAL.DISPATCHSCOPE on every top-level user call, and by
%       mptDefaults('reset').
%
%   The seen-set persists within a top-level call (so a LIST-mode
%   loop or batched-raw iteration produces one announce per unique
%   dispatch, not one per item), and is reset on the next top-level
%   call so that the user sees the announce again. The reset is
%   driven by INTERNAL.DISPATCHSCOPE's depth counter.
%
%   Gating: dispatch messages are NOT gated by per-call verbose.
%   They are gated by the toolbox-wide showHints flag
%   (mptDefaults('showHints')). To fully silence dispatch messages:
%   mptDefaults('showHints', false).
%
%   See also: MPTDEFAULTS, INTERNAL.DISPATCHSCOPE.

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

    key = sprintf('%s|%s', funcName, chosen);
    if isKey(seen, key)
        return;
    end
    seen(key) = true;

    if isProbed
        fprintf('%s: chose ''%s'' path (estimated %s); Ctrl+C to cancel.\n', ...
                funcName, chosen, localFormatTime(estSec));
    else
        fprintf('%s: chose ''%s'' path.\n', funcName, chosen);
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

function maybeShowDispatchMsg(varargin)
%INTERNAL.MAYBESHOWDISPATCHMSG  Print a dispatch decision at most once per top-level call.
%
%   internal.maybeShowDispatchMsg(funcName, chosen, routingReason)
%       Print a dispatch-decision message for the (funcName, chosen)
%       pair, but only if that exact pair has not already been printed
%       in the current top-level user call:
%           "<funcName>: chose '<chosen>' path."
%
%       The routingReason argument is still accepted (callers continue
%       to pass it, since it remains useful for downstream debugging),
%       but is not part of the throttle key --- within a single
%       top-level call, two different routing reasons that lead to the
%       same chosen path collapse to one announce. The visible
%       distinction that matters is which path ran, not why; throttling
%       on what the user sees avoids apparent duplicates.
%
%       This function announces the routing DECISION only. Time
%       estimates ("estimated X s; Ctrl+C to cancel") are a separate
%       concern emitted by estimateCompTime under the per-call verbose
%       flag, so the two never double-report a single dispatch. See the
%       verbose-vs-showHints split below.
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
%   Gating: dispatch DECISIONS are gated by the toolbox-wide showHints
%   flag (mptDefaults('showHints')), NOT by per-call verbose. Time
%   ESTIMATES are gated by per-call verbose (via estimateCompTime).
%   To fully silence dispatch decisions: mptDefaults('showHints',
%   false).
%
%   See also: MPTDEFAULTS, INTERNAL.DISPATCHSCOPE, ESTIMATECOMPTIME.

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

    if nargin ~= 3
        error('internal:maybeShowDispatchMsg:badArgs', ...
            ['Call form is internal.maybeShowDispatchMsg(funcName, ' ...
             'chosen, routingReason) or ' ...
             'internal.maybeShowDispatchMsg(''reset'').']);
    end

    % Master switch: showHints false → silent for all dispatch messages.
    S = mptDefaults();
    if isfield(S, 'showHints') && ~S.showHints
        return;
    end

    funcName      = char(varargin{1});
    chosen        = char(varargin{2});
    routingReason = char(varargin{3}); %#ok<NASGU>  kept for debugging

    key = sprintf('%s|%s', funcName, chosen);
    if isKey(seen, key)
        return;
    end
    seen(key) = true;

    fprintf('%s: chose ''%s'' path.\n', funcName, chosen);
end

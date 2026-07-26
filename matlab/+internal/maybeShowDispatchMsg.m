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
    % dictionary (R2022b+) — MathWorks-recommended replacement for
    % containers.Map. The seen-set is reset on every top-level user
    % entry via internal.dispatchScope, so both construction cost
    % (reallocation on reset) and per-lookup cost matter; dictionary
    % is materially faster than containers.Map on both.
    if isempty(seen)
        seen = dictionary();
    end

    % 'reset' form: clear all seen entries. A fresh dictionary is
    % cheap to construct, unlike containers.Map (Java-backed) where
    % the equivalent reset dominated per-top-level-call overhead in
    % the previous implementation.
    if nargin == 1 && (ischar(varargin{1}) || isstring(varargin{1})) ...
            && strcmpi(varargin{1}, 'reset')
        seen = dictionary();
        return;
    end

    if nargin ~= 3
        error('internal:maybeShowDispatchMsg:badArgs', ...
            ['Call form is internal.maybeShowDispatchMsg(funcName, ' ...
             'chosen, routingReason) or ' ...
             'internal.maybeShowDispatchMsg(''reset'').']);
    end

    funcName      = char(varargin{1});
    chosen        = char(varargin{2});
    % routingReason retained by callers as documentation but no longer
    % consumed here (the throttle key intentionally covers what the user
    % sees, not why the router picked it).

    % Cheap key: direct char concatenation. sprintf('%s|%s', ...) does
    % the same thing but goes through fprintf-family format parsing,
    % which is measurably slower on tight-loop dispatch bookkeeping.
    key = [funcName '|' chosen];

    % Early-exit on the repeat-dispatch case: if we've already
    % announced this (funcName, chosen) pair in this top-level call,
    % nothing else here matters. Checking the throttle key first lets
    % us skip the mptDefaults query on the fast path (repeat calls
    % within a single top-level user call, which is the common case
    % inside batched-raw loops and nested dispatch chains). An
    % unconfigured dictionary (before any insert this session) throws
    % on isKey; guard with numEntries.
    if numEntries(seen) > 0 && isKey(seen, key)
        return;
    end

    % Master switch: showHints false → silent for all dispatch messages.
    S = mptDefaults();
    if isfield(S, 'showHints') && ~S.showHints
        return;
    end

    seen(key) = true;

    fprintf('%s: chose ''%s'' path.\n', funcName, chosen);
end

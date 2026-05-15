function out = dispatchScope(action)
%INTERNAL.DISPATCHSCOPE  Reset throttle on top-level entry; track call depth.
%
%   guard = internal.dispatchScope() should be the first executable
%   line of every user-facing toolbox function that may emit (or whose
%   inner calls may emit) dispatch messages. The returned onCleanup
%   object decrements the depth counter when the calling function
%   exits — including on error or Ctrl+C.
%
%   Mechanics:
%     - On entry from outside the toolbox (depth was 0), the dispatch
%       throttle is reset, so the next top-level user call gets a
%       fresh set of announces.
%     - On nested entry (depth > 0), the throttle is not touched, so
%       inner sub-calls inherit their caller's throttle state. This
%       keeps repeated identical sub-call announces suppressed inside
%       a single top-level call (e.g. a LIST-mode loop produces one
%       message per unique dispatch, not one per iteration).
%     - On exit, the depth counter is decremented via an onCleanup
%       guard returned to the caller.
%
%   Internal-action forms (not for direct user calls):
%     internal.dispatchScope('exit')   - decrement depth (called by guard)
%     internal.dispatchScope('reset')  - force depth back to 0
%     internal.dispatchScope('query')  - return current depth
%
%   The depth counter is persistent for the duration of the MATLAB
%   session; `clear all` or `clear internal.dispatchScope` resets it.
%
%   See also: INTERNAL.MAYBESHOWDISPATCHMSG, MPTDEFAULTS.

    persistent depth
    if isempty(depth)
        depth = 0;
    end

    if nargin == 0
        % Entry form: caller wants an onCleanup guard.
        if depth == 0
            internal.maybeShowDispatchMsg('reset');
        end
        depth = depth + 1;
        out = onCleanup(@() internal.dispatchScope('exit'));
        return;
    end

    switch lower(char(action))
        case 'exit'
            depth = depth - 1;
            if depth < 0
                depth = 0;  % defensive
            end
            out = [];
        case 'reset'
            depth = 0;
            out = [];
        case 'query'
            out = depth;
        otherwise
            error('internal:dispatchScope:badArg', ...
                'Unknown action ''%s''.', char(action));
    end
end

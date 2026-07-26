function guard = callGuard()
%INTERNAL.CALLGUARD  Single-onCleanup combined entry guard for user-facing
%   toolbox functions.
%
%   guard = INTERNAL.CALLGUARD() should be the first executable line of
%   every user-facing toolbox function that needs both dispatch-scope
%   depth tracking and kernel-chunk-bytes pinning. It replaces the
%   previous pattern
%
%       guard    = internal.dispatchScope();
%       chunkPin = internal.kernelChunkBytesResolved('pinForCall');
%
%   which allocated two onCleanup objects per top-level entry (each
%   allocation is ~10-15 microseconds --- meaningful on sub-millisecond
%   workloads). This form allocates one onCleanup that carries both
%   cleanups.
%
%   Behaviour matches the pair it replaces:
%     - On outermost entry (depth was 0), the dispatch throttle is
%       reset, kernel-chunk-bytes is pinned (if it needed OS
%       resolution), and on exit both are torn down.
%     - On nested entry (depth > 0), depth is still incremented for
%       correct throttle/depth tracking on exit, but pinning is
%       skipped (the outer call's pin is still active).
%
%   See also: INTERNAL.DISPATCHSCOPE, INTERNAL.KERNELCHUNKBYTESRESOLVED,
%   INTERNAL.MAYBESHOWDISPATCHMSG.

    % Dispatch scope: increment depth, reset throttle if outermost.
    % Returns whether this is the outermost call (unused directly here
    % but exercises the same code path as the standalone dispatchScope
    % entry).
    isOutermost = internal.dispatchScope('enter'); %#ok<NASGU>

    % Chunk-bytes pin: only actually pins if outermost AND the default
    % is a string needing OS resolution. Returns a bool so the cleanup
    % can conditionally clear.
    didPin = internal.kernelChunkBytesResolved('pinNoCleanup');

    % Single onCleanup carries both teardown actions.
    guard = onCleanup(@() localGuardCleanup(didPin));
end


function localGuardCleanup(didPin)
    internal.dispatchScope('exit');
    if didPin
        internal.kernelChunkBytesResolved('clearPin');
    end
end

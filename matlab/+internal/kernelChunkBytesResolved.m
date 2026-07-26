function varargout = kernelChunkBytesResolved(command)
%KERNELCHUNKBYTESRESOLVED  Resolve the kernelChunkBytes default to bytes.
%
%   bytes = internal.kernelChunkBytesResolved() returns the per-chunk
%   byte budget that kernel-matrix consumers should respect. Reads the
%   'kernelChunkBytes' toolbox default; positive numeric values are
%   returned as-is. The factory value 'auto' resolves to half of
%   currently available physical memory (see internal.availableMemory).
%
%   cleanupObj = internal.kernelChunkBytesResolved('pinForCall') pins
%   the resolved 'auto' value for the lifetime of the returned
%   onCleanup. While pinned, internal.kernelChunkBytesResolved() returns
%   the same cached value rather than re-querying the OS. Top-level
%   entry points (cosSimExpTens, evalExpTens) use this so that
%   recursive inner calls — e.g. batched-raw cosSimExpTens making
%   thousands of unique-pair calls — share one resolution. Nested
%   pinForCall returns a no-op cleanup; only the outermost pin is
%   responsible for clearing.
%
%   When no per-call pin is active, resolutions are cached for up to
%   10 s as a fallback for back-to-back small calls in tight script
%   loops. The cache is keyed on the raw default value, so changing
%   it via mptDefaults flushes naturally.
%
%   internal.kernelChunkBytesResolved('flushCache') clears all
%   caches (used by mptDefaults('reset')).
%
%   The peak transient allocation in a single kernel-matrix chunk
%   runs roughly (2*dim + 2) * nJ * nQ * bytesPerScalar — the
%   broadcast difference tensor, its square, and the summed /
%   exponentiated intermediate are briefly co-resident in MATLAB.
%   The factor of two between the 'auto' budget (half of available
%   memory) and the actual peak gives a safety margin against
%   per-query allocation under-estimates and concurrent memory
%   pressure from other processes.
%
%   See also internal.availableMemory, mptDefaults.

    persistent activePin activePinRaw
    persistent ttlCache ttlRaw ttlTimer

    if nargin == 0
        command = 'get';
    end

    switch command
        case 'get'
            raw = mptDefaults('kernelChunkBytes');
            if isnumeric(raw)
                varargout{1} = double(raw);
                return;
            end

            % Per-call pin: set by pinForCall, valid for the lifetime
            % of the outermost calling top-level function.
            if ~isempty(activePin) && isequal(activePinRaw, raw)
                varargout{1} = activePin;
                return;
            end

            % TTL fallback: cache 'auto' resolutions for 10 s to avoid
            % per-call OS queries when callers make many short
            % independent calls in a tight loop.
            if ~isempty(ttlCache) && isequal(ttlRaw, raw) ...
                    && toc(ttlTimer) <= 10.0
                varargout{1} = ttlCache;
                return;
            end

            % Resolve fresh and update TTL cache.
            bytes = localResolveFresh(raw);
            ttlCache = bytes;
            ttlRaw = raw;
            ttlTimer = tic;
            varargout{1} = bytes;

        case 'pinForCall'
            if ~isempty(activePin)
                % Already pinned by an outer call; nested no-op cleanup.
                varargout{1} = onCleanup(@() []);
                return;
            end
            raw = mptDefaults('kernelChunkBytes');
            if isnumeric(raw)
                % Numeric value: no OS query, no need to pin.
                varargout{1} = onCleanup(@() []);
                return;
            end
            % Use TTL cache if valid; otherwise resolve fresh and update
            % TTL. This is what makes sibling calls (e.g. a
            % templateHarmonicity loop calling evalExpTens repeatedly)
            % share one OS query rather than spawning vm_stat per call.
            if ~isempty(ttlCache) && isequal(ttlRaw, raw) ...
                    && toc(ttlTimer) <= 10.0
                activePin = ttlCache;
            else
                activePin = localResolveFresh(raw);
                ttlCache = activePin;
                ttlRaw = raw;
                ttlTimer = tic;
            end
            activePinRaw = raw;
            varargout{1} = onCleanup(...
                @() internal.kernelChunkBytesResolved('clearPin'));

        case 'pinNoCleanup'
            % Bookkeeping-only pin: pins if outermost and if
            % kernelChunkBytes is a string (needing OS resolution),
            % and returns whether a pin was set so the caller can
            % conditionally clearPin later. No onCleanup allocated --
            % callers are responsible for calling clearPin themselves
            % (typically via a combined internal.callGuard onCleanup
            % that also handles other cleanups in one allocation).
            if ~isempty(activePin)
                varargout{1} = false;   % already pinned by outer
                return;
            end
            raw = mptDefaults('kernelChunkBytes');
            if isnumeric(raw)
                varargout{1} = false;   % numeric: no need to pin
                return;
            end
            if ~isempty(ttlCache) && isequal(ttlRaw, raw) ...
                    && toc(ttlTimer) <= 10.0
                activePin = ttlCache;
            else
                activePin = localResolveFresh(raw);
                ttlCache = activePin;
                ttlRaw = raw;
                ttlTimer = tic;
            end
            activePinRaw = raw;
            varargout{1} = true;

        case 'clearPin'
            activePin = [];
            activePinRaw = [];

        case 'flushCache'
            activePin = [];
            activePinRaw = [];
            ttlCache = [];
            ttlRaw = [];
            ttlTimer = [];

        otherwise
            error('mpt:kernelChunkBytesResolved:badCommand', ...
                'Unknown command ''%s''.', command);
    end
end


function bytes = localResolveFresh(raw)
    v = lower(char(raw));
    if strcmp(v, 'auto')
        bytes = max(1, internal.availableMemory() / 2);
        return;
    end
    error('mpt:kernelChunkBytesResolved:badDefault', ...
        'kernelChunkBytes default ''%s'' is not a recognised value.', v);
end

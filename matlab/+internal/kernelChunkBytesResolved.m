function bytes = kernelChunkBytesResolved()
%KERNELCHUNKBYTESRESOLVED  Resolve the kernelChunkBytes default to bytes.
%   bytes = internal.kernelChunkBytesResolved() returns the per-chunk
%   byte budget that kernel-matrix consumers should respect.
%
%   Reads the 'kernelChunkBytes' toolbox default. The factory value
%   'auto' resolves to half of currently available physical memory
%   (see internal.availableMemory); any positive integer is returned
%   as-is. Resolution happens at call time, so the budget tracks
%   memory pressure across a session.
%
%   The peak transient allocation in a single kernel-matrix chunk
%   runs roughly (2*dim + 2) * nJ * nQ * bytesPerScalar — the
%   broadcast difference tensor, its square, and the
%   summed/exponentiated intermediate are briefly co-resident in
%   MATLAB. The factor of two between the 'auto' budget (half of
%   available memory) and the actual peak gives a safety margin
%   against per-query allocation under-estimates and concurrent
%   memory pressure from other processes.
%
%   See also internal.availableMemory, mptDefaults.

    val = mptDefaults('kernelChunkBytes');
    if ischar(val) || isstring(val)
        v = lower(char(val));
        if strcmp(v, 'auto')
            bytes = max(1, internal.availableMemory() / 2);
            return;
        end
        error('mpt:kernelChunkBytesResolved:badDefault', ...
            'kernelChunkBytes default %s is not a recognised value.', v);
    end
    bytes = double(val);
end

function bytes = availableMemory()
%AVAILABLEMEMORY  Return currently available physical memory in bytes.
%   bytes = internal.availableMemory() queries the operating system for
%   currently available physical memory, returning the result as a
%   double-precision byte count.
%
%   Platform-specific paths:
%
%     Windows  — uses MATLAB's built-in memory() and returns
%                sys.PhysicalMemory.Available.
%
%     Linux    — reads /proc/meminfo and returns MemAvailable * 1024.
%                MemAvailable is the kernel's own estimate of memory
%                available for new allocations without swapping (added
%                to the kernel in 2014; present on essentially all
%                current distributions).
%
%     macOS    — runs `vm_stat` and approximates available memory as
%                (free + inactive + speculative) * pageSize. This is
%                the same approximation Activity Monitor and standard
%                third-party memory tools use; macOS does not expose
%                a single MemAvailable-equivalent counter.
%
%   If all platform-specific queries fail, a 4 GiB fallback is
%   returned so callers can rely on a usable value.
%
%   See also memory, internal.kernelChunkBytesResolved.

    FALLBACK = 4 * 1024^3;  % 4 GiB
    bytes = NaN;

    if ispc
        try
            [~, sys] = memory();
            bytes = double(sys.PhysicalMemory.Available);
        catch
            bytes = NaN;
        end
    elseif isunix && ~ismac
        bytes = localLinuxAvailable();
    elseif ismac
        bytes = localMacAvailable();
    end

    if ~isfinite(bytes) || bytes <= 0
        bytes = FALLBACK;
    end
end


function bytes = localLinuxAvailable()
    bytes = NaN;
    try
        fid = fopen('/proc/meminfo', 'r');
        if fid < 0
            return;
        end
        text = fread(fid, '*char')';
        fclose(fid);
    catch
        return;
    end
    tok = regexp(text, 'MemAvailable:\s+(\d+)\s+kB', 'tokens', 'once');
    if isempty(tok)
        return;
    end
    bytes = str2double(tok{1}) * 1024;
end


function bytes = localMacAvailable()
    bytes = NaN;
    try
        [status, out] = system('vm_stat');
        if status ~= 0
            return;
        end
    catch
        return;
    end
    pgTok = regexp(out, 'page size of (\d+) bytes', 'tokens', 'once');
    if isempty(pgTok)
        return;
    end
    pageSize = str2double(pgTok{1});

    freeTok = regexp(out, 'Pages free:\s+(\d+)',         'tokens', 'once');
    inacTok = regexp(out, 'Pages inactive:\s+(\d+)',     'tokens', 'once');
    specTok = regexp(out, 'Pages speculative:\s+(\d+)',  'tokens', 'once');

    if isempty(freeTok) || isempty(inacTok) || isempty(specTok)
        return;
    end

    pagesAvail = str2double(freeTok{1}) + str2double(inacTok{1}) + ...
                 str2double(specTok{1});
    bytes = pagesAvail * pageSize;
end

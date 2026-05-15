function stride = progressStride(tPerRow, targetSec)
%PROGRESSSTRIDE  Compute progress-print stride for batched loops.
%
%   stride = internal.progressStride(tPerRow) returns the smallest
%   "nice" stride from the set {1, 2, 5, 10, 20, 50, 100, 200, 500,
%   1000, 2000, 5000, 10000} such that each print interval takes at
%   least 5 s (the default target) at the given per-row cost tPerRow
%   (seconds per row, from an empirical calibration). If tPerRow is
%   so small that even a stride of 10000 rows finishes in less than
%   the target, the function returns 10000 (the largest available
%   option); progress prints will then be more frequent than the
%   target, which is the best achievable without wider stride
%   options.
%
%   stride = internal.progressStride(tPerRow, targetSec) uses a
%   custom target interval in seconds (default 5).
%
%   Used by the batched helpers (cosSimExpTens, templateHarmonicity,
%   spectralEntropy, virtualPitches, entropyExpTens) to set the
%   cadence of their "X / Y rows computed" progress prints. Callers
%   also gate the prints on estTotal >= targetSec (i.e. only show
%   progress at all when the loop is expected to take long enough to
%   warrant it); the stride determines cadence within that, and
%   adapts down to 1 when individual rows are themselves slow enough
%   that per-row prints don't exceed the target interval.
%
%   Inputs
%   ------
%   tPerRow  : numeric scalar, per-row cost in seconds. Non-positive
%              or non-finite values fall back to a stride of 1.
%   targetSec: numeric scalar, target print interval in seconds.
%              Defaults to 5.
%
%   Output
%   ------
%   stride : positive integer from the option set.

    if nargin < 2 || isempty(targetSec)
        targetSec = 5;
    end
    options = [1 2 5 10 20 50 100 200 500 1000 2000 5000 10000];
    if ~isfinite(tPerRow) || tPerRow <= 0
        stride = options(1);
        return;
    end
    desired = targetSec / tPerRow;
    idx = find(options >= desired, 1, 'first');
    if isempty(idx)
        stride = options(end);
    else
        stride = options(idx);
    end
end

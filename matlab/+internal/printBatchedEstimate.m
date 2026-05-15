function printBatchedEstimate(label, nRows, estTotal, verbose, minPrintSec)
%PRINTBATCHEDESTIMATE Print a batched-mode upfront time estimate, gated.
%
%   printBatchedEstimate(label, nRows, estTotal):
%   printBatchedEstimate(label, nRows, estTotal, verbose):
%   printBatchedEstimate(label, nRows, estTotal, verbose, minPrintSec):
%
%   Used by the batched dispatch helpers in templateHarmonicity,
%   virtualPitches, spectralEntropy, entropyExpTens, tensorHarmonicity,
%   and similar functions, which compute their estimates empirically
%   (warm-up plus a sample of K rows) rather than via estimateCompTime.
%   The threshold and formatting match the scalar-mode print path
%   (estimateCompTime) so behaviour is consistent across dispatch modes.
%
%   Inputs:
%     label        — Function name tag used in the printed line (e.g.
%                    'spectralEntropy').
%     nRows        — Total number of rows in the batched call (M).
%     estTotal     — Empirical estimate in seconds (calibration time
%                    plus per-row time times M).
%     verbose      — Optional logical (default true). If false,
%                    suppresses output.
%     minPrintSec  — Optional minimum threshold (default 10), matching
%                    estimateCompTime.
%
%   See also estimateCompTime.

if nargin < 4
    verbose = true;
end
if nargin < 5
    minPrintSec = 10;
end

if ~verbose || estTotal < minPrintSec
    return;
end

if estTotal < 1
    estStr = sprintf('%.0f ms', estTotal * 1000);
elseif estTotal < 60
    estStr = sprintf('%.1f s', estTotal);
elseif estTotal < 3600
    estStr = sprintf('%.1f min', estTotal / 60);
else
    estStr = sprintf('%.1f hr', estTotal / 3600);
end

fprintf('%s (batched, %d rows): estimated time ~%s (Ctrl+C to cancel).\n', ...
    label, nRows, estStr);

end

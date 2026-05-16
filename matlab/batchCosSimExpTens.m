function s = batchCosSimExpTens(pMatA, pMatB, sigma, r, isRel, isPer, period, varargin)
%BATCHCOSSIMEXPTENS Batch cosine similarity of expectation tensors.
%
%   DEPRECATED as of v2.1. Use cosSimExpTens(P1, W1, P2, W2, sigma, r,
%   isRel, isPer, period) instead.
%
%   This file is now a thin shim. The batched-raw implementation lives
%   in cosSimExpTens.m (in the localCosSimBatchedRaw local function);
%   calls to batchCosSimExpTens are translated to the new positional
%   form and forwarded to cosSimExpTens. Existing calling
%   conventions continue to work and produce identical results, but
%   emit a batchCosSimExpTens:deprecated warning on each invocation.
%
%   s = batchCosSimExpTens(pMatA, pMatB, sigma, r, isRel, isPer, period):
%   Computes the cosine similarity between the r-ad expectation tensors of
%   paired weighted multisets. Each row of pMatA and pMatB defines one
%   weighted multiset; the function returns one similarity value per row.
%
%   Optional name-value pairs:
%     'weightsA', wA   — nRows x nA matrix of weights for multiset A.
%     'weightsB', wB   — nRows x nB matrix of weights for multiset B.
%     'spectrum', sp   — Cell array forwarded to addSpectra.
%     'verbose', tf    — Logical (default: true). If false, suppresses
%                        console output. Note: dispatch messages from
%                        the method dispatcher are gated by
%                        mptDefaults('showHints'), not by this flag.
%     'precision', n   — Round inputs to n decimal places.
%
%   See also: cosSimExpTens.

% Top-level call guard: see internal.dispatchScope.
guard = internal.dispatchScope(); %#ok<NASGU>

% --- Parse kwargs ---
weightsA = [];
weightsB = [];
specArgs = {};
verbose  = true;
nDec     = [];

i = 1;
while i <= numel(varargin)
    if ischar(varargin{i}) || isstring(varargin{i})
        switch lower(varargin{i})
            case 'weightsa'
                weightsA = varargin{i + 1};
                i = i + 2;
            case 'weightsb'
                weightsB = varargin{i + 1};
                i = i + 2;
            case 'spectrum'
                specArgs = varargin{i + 1};
                if ~iscell(specArgs)
                    error('''spectrum'' value must be a cell array of addSpectra arguments.');
                end
                i = i + 2;
            case 'verbose'
                verbose = logical(varargin{i + 1});
                i = i + 2;
            case 'precision'
                nDec = varargin{i + 1};
                i = i + 2;
            case '__internalcall'
                % Legacy flag from when cosSimExpTens delegated to this
                % function (the direction is now reversed). Accept and
                % ignore for backward compatibility with any external
                % code that may pass it.
                i = i + 2;
            otherwise
                error('Unknown option ''%s''.', varargin{i});
        end
    else
        error('Expected a name-value pair; got a %s.', class(varargin{i}));
    end
end

% --- Deprecation notice (always emitted; deprecation contract is that
%     this shim warns per invocation until removal). ---
warning('batchCosSimExpTens:deprecated', ...
    ['batchCosSimExpTens is deprecated as of v2.1 and will be removed in a future release. ' ...
     'Use cosSimExpTens(P1, W1, P2, W2, sigma, r, isRel, isPer, period) for the same ' ...
     'paired-rows batched cosine similarity (rows of P1/P2 = paired multisets; ' ...
     'pass [] for W1/W2 to use uniform weights, or matrices matching P1/P2 for explicit ' ...
     'weights).']);

% --- Forward to cosSimExpTens via the canonical positional API ---
%     cosSimExpTens(P1, W1, P2, W2, sigma, r, isRel, isPer, period, NV...)
forwardArgs = {pMatA, weightsA, pMatB, weightsB, sigma, r, isRel, isPer, period};
if ~isempty(specArgs)
    forwardArgs = [forwardArgs, {'spectrum', specArgs}]; %#ok<AGROW>
end
if ~isempty(nDec)
    forwardArgs = [forwardArgs, {'precision', nDec}]; %#ok<AGROW>
end
forwardArgs = [forwardArgs, {'verbose', verbose}]; %#ok<AGROW>

s = cosSimExpTens(forwardArgs{:});
end

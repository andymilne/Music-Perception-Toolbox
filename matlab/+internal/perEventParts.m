function [pAttr, wAttr] = perEventParts(pAttr, wAttr)
%PEREVENTPARTS  Convert attributes given per event to NaN-padded matrices.
%
%   An attribute's values may be given per event, as a 1 x N cell whose
%   n-th entry holds the values of event n: a scalar, a vector, or [] for
%   no value, so {[60 64 67], 62, 64, 65} is a three-note chord followed
%   by three single notes. It stands for the K x N matrix with one column
%   per event, padded with NaN where an event has fewer than K values,
%   and is converted to that matrix here.
%
%   Per-event weights are likewise a 1 x N cell, each entry a scalar
%   (weighting every value of its event) or a vector with one weight per
%   value; the slots holding no value take weight 0. Numeric weights pass
%   through unchanged and apply to the padded matrix as usual.
%
%   Inputs
%       pAttr - 1 x A cell of attributes, each a numeric matrix or a
%               per-event cell. Anything else passes through.
%       wAttr - [], a scalar, or a 1 x A cell of per-attribute weights.
%
%   Outputs
%       pAttr, wAttr - with every per-event attribute and its per-event
%                      weights converted.

if ~iscell(pAttr)
    return;
end
wIsCell = iscell(wAttr) && numel(wAttr) == numel(pAttr);
for a = 1:numel(pAttr)
    if ~iscell(pAttr{a})
        if wIsCell && iscell(wAttr{a})
            wAttr{a} = localWeights(wAttr{a}, double(pAttr{a}), a);
        end
        continue;
    end
    pAttr{a} = localValues(pAttr{a}, a);
    if wIsCell && iscell(wAttr{a})
        wAttr{a} = localWeights(wAttr{a}, pAttr{a}, a);
    end
end
end


function M = localValues(c, a)
N = numel(c);
cols = cell(1, N);
for n = 1:N
    v = c{n};
    if isempty(v)
        cols{n} = zeros(0, 1);
        continue;
    end
    if ~(isnumeric(v) || islogical(v)) || ~isvector(v)
        error('mpt:perEvent:badValues', ...
              ['Attribute %d, event %d: each entry of a per-event cell ' ...
               'must be a scalar, a vector of values, or [].'], a, n);
    end
    cols{n} = double(v(:));
end
K = max([cellfun(@numel, cols), 1]);
M = nan(K, N);
for n = 1:N
    M(1:numel(cols{n}), n) = cols{n};
end
end


function W = localWeights(c, V, a)
[K, N] = size(V);
if numel(c) ~= N
    error('mpt:perEvent:badWeightCount', ...
          ['Attribute %d: per-event weights must have one entry per ' ...
           'event (N = %d); got %d.'], a, N, numel(c));
end
W = zeros(K, N);
for n = 1:N
    rows = find(~isnan(V(:, n)));
    v = c{n};
    if isempty(v)
        v = 0;
    end
    if ~isnumeric(v) || ~isvector(v)
        error('mpt:perEvent:badWeights', ...
              ['Attribute %d, event %d: the weights of an event must be ' ...
               'a scalar or a vector.'], a, n);
    end
    v = double(v(:));
    if isscalar(v)
        W(rows, n) = v;
    elseif numel(v) == numel(rows)
        W(rows, n) = v;
    else
        error('mpt:perEvent:badWeights', ...
              ['Attribute %d, event %d: %d weights given for %d values; ' ...
               'give one weight, or one per value.'], a, n, numel(v), ...
              numel(rows));
    end
end
end

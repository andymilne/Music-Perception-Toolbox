function [pBound, wBound] = bindEvents(p, w, n, nvArgs)
%BINDEVENTS Bind n consecutive events into n-attribute super-events.
%
%   [pBound, wBound] = bindEvents(p, w, n) is a cross-event
%   preprocessing helper for multi-attribute tensor input. It takes a
%   single-attribute event sequence and slides a window of width n
%   across it, emitting each window as an n-attribute super-event
%   whose j-th attribute holds the value at lag j-1 (j = 1, ..., n).
%   The output is a 1 x n cell of 1 x N' matrices, suitable for
%   direct use as the pAttr argument of buildExpTens with all n
%   attributes assigned to a single group.
%
%   This complements differenceEvents: differencing aggregates across
%   event boundaries (collapsing k+1 consecutive events into a single
%   value), while binding aggregates within a window (gathering n
%   consecutive values into a single super-event with n separate slot
%   attributes). The two operations compose naturally: differencing
%   then binding gives n-tuples of consecutive step sizes, recovering
%   the n-tuple entropy of Milne & Dean (2016) as a special case
%   (sigma -> 0, integer-step grid, uniform weights, periodic domain)
%   while extending it to non-zero sigma, continuous-valued steps,
%   non-periodic domains, and per-event weights propagated through
%   both stages. The bound MAET is itself a density that can be fed
%   into the rest of the toolbox: pairs of bound MAETs can be
%   compared via cosSimExpTens, queried via windowed similarity, and
%   so on.
%
%   The lag slots are emitted as separate K_a = 1 attributes (rather
%   than packed into a single K_a = n attribute) because lag identity
%   is not exchangeable: slot j and slot j+1 carry distinct
%   positional meanings. The MAET framework's within-attribute
%   exchangeability would collapse ordered tuples to unordered ones,
%   so positional structure must be carried by the attribute axis.
%
%   N' = N - n + 1 (default) or N (when 'circular' is true).
%
%   Inputs
%       p  - Event values: a 1 x N row vector or a 1-cell {1 x N}
%            (the 1-cell form is accepted for symmetry with the
%            output of differenceEvents). K_a = 1 only.
%       w  - Weights. [], scalar, 1 x N row, or 1-cell of any of
%            those (matching the p-input form).
%       n  - Window size (positive integer).
%
%   Name-Value Arguments
%       'circular' - Logical (default: false). When true, the window
%                    wraps around the end of the sequence; N' = N.
%                    When false, N' = N - n + 1.
%
%   Outputs
%       pBound - 1 x n cell of 1 x N' matrices. pBound{j} contains
%                the value at lag j-1 for each window.
%       wBound - Weight of each bound super-event = product of the n
%                input weights it spans (rolling product of width n).
%                Shape mirrors the input:
%                  - []           stays []
%                  - scalar c     becomes c^n
%                  - 1 x N row    becomes 1 x N' row
%                  - 1-cell input returns a 1-cell wrapper.
%
%   Examples
%       % 2-tuple entropy of step sizes (diatonic scale, sigma = 0)
%       p = [0 2 4 5 7 9 11];
%       d = differenceEvents({p}, [], [], 1, 12);   % 1 x 7 (circular)
%       [pB, wB] = bindEvents(d{1}, [], 2, 'circular', true);
%       T = buildExpTens(pB, wB, 1e-6, [1 1], 1, false, true, 12);
%       H = entropyExpTens(T);
%
%       % Compare 2-tuple distributions of two scales via cosine
%       % similarity (smoothed)
%       p1 = [0 2 4 5 7 9 11]; p2 = [0 1 3 5 6 8 10];
%       d1 = differenceEvents({p1}, [], [], 1, 12);
%       d2 = differenceEvents({p2}, [], [], 1, 12);
%       [pB1, wB1] = bindEvents(d1{1}, [], 2, 'circular', true);
%       [pB2, wB2] = bindEvents(d2{1}, [], 2, 'circular', true);
%       T1 = buildExpTens(pB1, wB1, 1, [1 1], 1, false, true, 12);
%       T2 = buildExpTens(pB2, wB2, 1, [1 1], 1, false, true, 12);
%       s = cosSimExpTens(T1, T2);
%
%   See also DIFFERENCEEVENTS, BUILDEXPTENS, ENTROPYEXPTENS,
%   COSSIMEXPTENS, NTUPLEENTROPY.

    arguments
        p
        w = []
        n (1, 1) {mustBePositive, mustBeInteger} = 2
        nvArgs.circular (1, 1) logical = false
    end

    % --- Unwrap 1-cell inputs (symmetry with differenceEvents output) ---
    inputWasCellP = iscell(p);
    if inputWasCellP
        if numel(p) ~= 1
            error('bindEvents:multiAttribute', ...
                  ['p as a cell must contain exactly one attribute; ' ...
                   'got %d. To bind multiple attributes, call ' ...
                   'bindEvents on each separately.'], numel(p));
        end
        p = p{1};
    end

    inputWasCellW = iscell(w);
    if inputWasCellW
        if numel(w) ~= 1
            error('bindEvents:multiAttributeWeight', ...
                  ['w as a cell must contain exactly one entry; got %d.'], ...
                  numel(w));
        end
        w = w{1};
    end

    % --- Validate p shape (K_a = 1 row vector) ---
    if ~isnumeric(p)
        error('bindEvents:badPType', 'p must be numeric.');
    end
    p = double(p);
    if ~isvector(p) && ndims(p) <= 2 && size(p, 1) ~= 1
        error('bindEvents:multiSlotAttribute', ...
              ['p has shape [%s]; bindEvents requires K_a = 1 ' ...
               '(a 1 x N row vector). Within-event slot ' ...
               'exchangeability does not license the cross-event ' ...
               'slot alignment that sliding-window binding imposes ' ...
               '(same constraint as differenceEvents).'], ...
              num2str(size(p)));
    end
    p = p(:).';   % canonicalize to 1 x N
    N = numel(p);

    if nvArgs.circular
        if n > N
            error('bindEvents:windowTooLarge', ...
                  ['Circular window size n = %d exceeds event count ' ...
                   'N = %d.'], n, N);
        end
        nPrime = N;
    else
        nPrime = N - n + 1;
        if nPrime < 1
            error('bindEvents:windowTooLarge', ...
                  ['Window size n = %d exceeds event count N = %d ' ...
                   '(non-circular mode).'], n, N);
        end
    end

    % --- Build the n lag matrices ---
    % pBound{j}(1, i) = p( idx(i, j) )
    if nvArgs.circular
        idxMat = mod((0:nPrime-1).' + (0:n-1), N) + 1;   % nPrime x n
    else
        idxMat = (0:nPrime-1).' + (1:n);                  % nPrime x n
    end
    pBound = cell(1, n);
    for j = 1:n
        pBound{j} = reshape(p(idxMat(:, j)), 1, nPrime);
    end

    % --- Transform weights ---
    if isempty(w) && ~iscell(w)
        wBound = [];
    elseif isnumeric(w) && isscalar(w)
        wBound = double(w) ^ n;
    elseif isnumeric(w) && isvector(w) && numel(w) == N
        wRow = double(w(:).');
        if nvArgs.circular
            % Rolling product over n consecutive entries with wrap
            wBoundRow = zeros(1, nPrime);
            for i = 1:nPrime
                wBoundRow(i) = prod(wRow(idxMat(i, :)));
            end
        else
            wBoundRow = zeros(1, nPrime);
            for i = 1:nPrime
                wBoundRow(i) = prod(wRow(i:i + n - 1));
            end
        end
        wBound = wBoundRow;
    else
        error('bindEvents:badWeightShape', ...
              ['w must be [], a scalar, or a 1 x N row vector ' ...
               '(got shape [%s]).'], num2str(size(w)));
    end

    % --- Re-wrap weight in a 1-cell if input was 1-cell ---
    if inputWasCellW
        wBound = {wBound};
    end
end

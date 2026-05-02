function [pBound, wBound] = bindEvents(p, w, n, nvArgs)
%BINDEVENTS Bind n consecutive events into n-attribute super-events.
%
%   [pBound, wBound] = bindEvents(p, w, n) is a cross-event
%   preprocessing helper for multi-attribute tensor input. It takes a
%   single-attribute event sequence and slides a window of width n
%   across it, emitting each window as an n-attribute super-event
%   whose j-th attribute holds the value at lag j-1 (j = 1, ..., n).
%   The output is a 1 x n cell of K_a x N' matrices, suitable for
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
%   The lag slots are emitted as separate K_a-valued attributes (one
%   attribute per lag) rather than packed into a single attribute,
%   because lag identity is not exchangeable: the value at lag j and
%   the value at lag j+1 carry distinct positional meanings within
%   the bound super-event. By contrast, K_a > 1 inputs (multi-value
%   attributes whose slots are deliberately exchangeable) are
%   permitted: each output attribute then carries the K_a slot values
%   of one underlying event, and the within-attribute exchangeability
%   is preserved per output attribute. Cross-event slot alignment is
%   never imposed, because the cross-event structure is between
%   output attributes, not within.
%
%   N' = N - n + 1 (default) or N (when 'circular' is true).
%
%   Inputs
%       p  - Event values: a K_a x N matrix, a 1 x N row, a length-N
%            vector, or a 1-cell {K_a x N} (the 1-cell form is
%            accepted for symmetry with the output of
%            differenceEvents). K_a >= 1.
%       w  - Weights. [], scalar, 1 x N row, K_a x 1 column, K_a x N
%            matrix, or a 1-cell of any of those (matching the
%            p-input form).
%       n  - Window size (positive integer).
%
%   Name-Value Arguments
%       'circular' - Logical (default: false). When true, the window
%                    wraps around the end of the sequence; N' = N.
%                    When false, N' = N - n + 1.
%
%                    The 'circular' flag describes the *event
%                    sequence* (whether the last event connects back
%                    to the first), and is independent of the
%                    *positional periodicity* set in buildExpTens via
%                    its isPer / period arguments. Both combinations
%                    are meaningful: a non-circular sequence on a
%                    periodic domain (a non-cyclic motif living in
%                    pitch-class space), and a circular sequence on a
%                    linear domain (a cyclic rhythm represented in
%                    linear time, e.g., for windowed analysis). The
%                    two flags are orthogonal.
%
%   Outputs
%       pBound - 1 x n cell of K_a x N' matrices. pBound{j} contains
%                the value(s) at lag j-1 for each window. For K_a = 1
%                input each cell is 1 x N'.
%       wBound - Per-attribute weight propagation, in the form that
%                buildExpTens accepts directly:
%                  - []           stays []
%                  - scalar c     stays c (broadcast in buildExpTens)
%                  - 1 x N row    becomes 1 x n cell of 1 x N' rows
%                  - K_a x 1 col  becomes 1 x n cell of K_a x 1 cols
%                  - K_a x N      becomes 1 x n cell of K_a x N' mats
%                The end-to-end numerics are equivalent to a rolling
%                product of slot weights: each output attribute
%                inherits the slot weights of the underlying event at
%                its lag, and buildExpTens multiplies across attributes
%                during tuple enumeration.
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

    % --- Validate p shape (allow any K_a >= 1) ---
    if ~isnumeric(p)
        error('bindEvents:badPType', 'p must be numeric.');
    end
    p = double(p);
    if ndims(p) > 2
        error('bindEvents:badPDims', ...
              'p must be at most 2-D; got ndims = %d.', ndims(p));
    end
    if isvector(p)
        % Canonicalise length-N vector to 1 x N (K_a = 1).
        p = reshape(p, 1, []);
    end
    Ka = size(p, 1);
    N  = size(p, 2);

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

    % --- Build the n lag matrices, preserving K_a ---
    if nvArgs.circular
        idxMat = mod((0:nPrime-1).' + (0:n-1), N) + 1;   % nPrime x n
    else
        idxMat = (0:nPrime-1).' + (1:n);                  % nPrime x n
    end
    pBound = cell(1, n);
    for j = 1:n
        pBound{j} = p(:, idxMat(:, j));      % K_a x nPrime
    end

    % --- Weights: per-attribute propagation ---
    %
    % Each output attribute inherits the slot weights of the underlying
    % event at its lag. buildExpTens then multiplies across attributes
    % during tuple enumeration, so the end-to-end weight of a bound
    % super-event equals the product of the n constituent events'
    % weights (the same numerical contribution as the prior eager
    % rolling product, for K_a = 1; the natural generalisation for
    % K_a > 1).

    if isempty(w)
        wBound = [];
    elseif isnumeric(w) && isscalar(w)
        % Scalar weight broadcasts in buildExpTens; pass through as scalar.
        % Note: a scalar c here means each event has weight c, which makes
        % each bound super-event's effective weight c^n via the multi-
        % attribute weight machinery. Numerically equivalent to the
        % prior c^n eager return, after buildExpTens' broadcast.
        wBound = double(w);
    elseif isnumeric(w)
        if ndims(w) > 2
            error('bindEvents:badWDims', ...
                  'w must be at most 2-D; got ndims = %d.', ndims(w));
        end
        wMat = double(w);

        % Coerce 1-D vector / 1 x N row / K_a x 1 col / K_a x N matrix
        % into a 2-D matrix consistent with the p-shape.
        if isvector(wMat)
            if numel(wMat) == N
                wMat = reshape(wMat, 1, []);     % 1 x N row
            elseif numel(wMat) == Ka && Ka ~= N
                wMat = reshape(wMat, [], 1);     % K_a x 1 col
            elseif Ka == N
                % Ambiguous: the input length matches both N and K_a.
                % Honour the literal shape supplied (vectors are coerced
                % above to a row by default, so this branch is the
                % literal-shape pass-through).
                wMat = reshape(wMat, 1, []);
            else
                error('bindEvents:badWeightShape', ...
                      ['w as a vector must have length N = %d or ' ...
                       'K_a = %d (got length %d).'], N, Ka, numel(wMat));
            end
        end

        if size(wMat, 1) == 1 && size(wMat, 2) == N
            % 1 x N row -> propagate per attribute as 1 x N' rows
            wBound = cell(1, n);
            for j = 1:n
                wBound{j} = wMat(1, idxMat(:, j));   % 1 x nPrime
            end
        elseif size(wMat, 1) == Ka && size(wMat, 2) == 1
            % K_a x 1 column -> per-attribute K_a x 1 (broadcast in builder)
            wBound = cell(1, n);
            for j = 1:n
                wBound{j} = wMat;                    % K_a x 1
            end
        elseif size(wMat, 1) == Ka && size(wMat, 2) == N
            % K_a x N matrix -> per-attribute K_a x N' matrix
            wBound = cell(1, n);
            for j = 1:n
                wBound{j} = wMat(:, idxMat(:, j));   % K_a x nPrime
            end
        else
            error('bindEvents:badWeightShape', ...
                  ['w must be [], a scalar, a 1 x N row, a K_a x 1 ' ...
                   'column, or a K_a x N matrix (got shape [%s] with ' ...
                   'K_a = %d, N = %d).'], num2str(size(w)), Ka, N);
        end
    else
        error('bindEvents:badWeightType', ...
              'w must be [] or numeric.');
    end

    % --- Re-wrap weight in a 1-cell if input was 1-cell ---
    if inputWasCellW
        wBound = {wBound};
    end
end

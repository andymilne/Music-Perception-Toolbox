function v = gaussianKernelSum(C, wJ, X, sigma, opts)
%INTERNAL.GAUSSIANKERNELSUM  Gaussian-kernel sum over centres against queries.
%
%   v = INTERNAL.GAUSSIANKERNELSUM(C, wJ, X, sigma, opts) computes
%
%      v(q) = sum_j wJ(j) * exp(-Q(c_j - x_q) / (2 * sigma^2))
%
%   where Q is the quadratic form determined by opts.isRel and opts.r:
%
%      abs mode (opts.isRel=false):   Q(D) = sum(D.^2)
%      rel mode (opts.isRel=true):    Q(D) = sum(D.^2) - sum(D)^2 / r
%
%   This helper is the single centres-path numerical kernel used by
%   evalExpTens, cosSimExpTens, entropyExpTens, and (eventually) the
%   batched user-facing wrappers. Consumers route through here so the
%   truncation and kernelPrecision options are applied uniformly across the
%   toolbox.
%
%   opts.truncationSigmas (Inf default): if finite, centres beyond a
%   Q-ball of squared radius (truncationSigmas * sigma)^2 are skipped
%   via a grid-bucket spatial index. Discarded centres' kernel value
%   is bounded by exp(-truncationSigmas^2 / 2) (e.g. ~1.5e-8 at the
%   recommended truncationSigmas=6). At Inf the computation is exact.
%
%   opts.kernelPrecision ('double' default): 'single' casts the hot-loop
%   intermediate arrays to single precision (~2x speedup on most
%   platforms). Output is always cast back to double. Relative
%   accuracy degrades to ~1e-7.
%
%   Periodic mode (opts.isPer = true) currently falls through to the
%   exact path regardless of opts.truncationSigmas; periodic-mode
%   truncation is a follow-up.
%
%   Inputs:
%     C       (dim, nJ) double - centres
%     wJ      (nJ, 1) double - weights
%     X       (dim, nQ) double - queries
%     sigma   (1, 1) positive double
%     opts    struct with fields:
%       isRel             logical (default false)
%       r                 integer >= 2 (required if isRel)
%       isPer             logical (default false)
%       period            scalar (required if isPer; > 0)
%       truncationSigmas  positive scalar or Inf (default from mptDefaults)
%       kernelPrecision         'double' (default from mptDefaults) or 'single'
%
%   Output:
%     v       (1, nQ) double row vector
%
%   See also: MPTDEFAULTS, EVALEXPTENS, COSSIMEXPTENS.

    arguments
        C double
        wJ (:,1) double
        X double
        sigma (1,1) double {mustBePositive}
        opts.isRel (1,1) logical = false
        opts.r (1,1) {mustBeInteger} = 0
        opts.isPer (1,1) logical = false
        opts.period (1,1) double = 0.0
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
        opts.kernelPrecision (1,:) char = mptDefaults('kernelPrecision')
    end

    if ~ismember(opts.kernelPrecision, {'double', 'single'})
        error('internal:gaussianKernelSum:badPrecision', ...
            '''kernelPrecision'' must be ''double'' or ''single''.');
    end
    if opts.truncationSigmas <= 0
        error('internal:gaussianKernelSum:badTruncation', ...
            '''truncationSigmas'' must be positive (use Inf to disable).');
    end

    dim = size(C, 1);
    nJ  = size(C, 2);
    nQ  = size(X, 2);
    if size(X, 1) ~= dim
        error('internal:gaussianKernelSum:dimMismatch', ...
            'C and X must have the same number of rows (got %d vs %d).', ...
            dim, size(X, 1));
    end
    if numel(wJ) ~= nJ
        error('internal:gaussianKernelSum:weightShape', ...
            'wJ must have length nJ = %d.', nJ);
    end
    if opts.isRel && opts.r < 2
        error('internal:gaussianKernelSum:relRequiresR', ...
            'rel mode requires opts.r >= 2 (got %d).', opts.r);
    end
    if opts.isPer && opts.period <= 0
        error('internal:gaussianKernelSum:perRequiresPeriod', ...
            'periodic mode requires opts.period > 0.');
    end

    % Decide path. Truncation is currently exact-only on periodic mode.
    useTruncation = isfinite(opts.truncationSigmas) ...
                 && opts.truncationSigmas > 0 ...
                 && ~opts.isPer ...
                 && nJ > 0 && nQ > 0;

    if strcmp(opts.kernelPrecision, 'single')
        C_w     = single(C);
        wJ_w    = single(wJ);
        X_w     = single(X);
        sigma_w = single(sigma);
        period_w = single(opts.period);
    else
        C_w     = C;
        wJ_w    = wJ;
        X_w     = X;
        sigma_w = sigma;
        period_w = opts.period;
    end
    inv2s2 = 1 / (2 * sigma_w^2);

    if useTruncation
        % 1-D abs case: vectorised path via sorted-centres +
        % searchsorted, much faster than the general per-query loop.
        if size(C_w, 1) == 1 && ~opts.isRel
            v_w = localTruncatedKernelSum1D(C_w, wJ_w, X_w, sigma_w, ...
                opts.truncationSigmas, inv2s2);
        else
            v_w = localTruncatedKernelSum(C_w, wJ_w, X_w, sigma_w, ...
                opts.isRel, opts.r, opts.truncationSigmas, inv2s2);
        end
    else
        v_w = localExactKernelSum(C_w, wJ_w, X_w, ...
            opts.isRel, opts.r, opts.isPer, period_w, inv2s2, sigma_w);
    end

    if strcmp(opts.kernelPrecision, 'single')
        v = double(v_w);
    else
        v = v_w;
    end
end


% =========================================================================
%  Exact path — the v2.0 centres-array body, chunked for memory bounds
% =========================================================================

function v = localExactKernelSum(C, wJ, X, isRel, r, isPer, period, inv2s2, sigma)
    dim = size(C, 1);
    nJ  = size(C, 2);
    nQ  = size(X, 2);

    if nJ == 0 || nQ == 0
        v = zeros(1, nQ, 'like', C);
        return;
    end

    bytesPerScalar = 4 * isa(C, 'single') + 8 * isa(C, 'double');
    bytesNeeded = (dim + 1) * double(nJ) * double(nQ) * bytesPerScalar;
    try
        memInfo  = memory;
        memLimit = memInfo.MaxPossibleArrayBytes * 0.5;
    catch
        memLimit = 4e9;
    end

    v = zeros(1, nQ, 'like', C);
    if bytesNeeded <= memLimit
        v = evalChunk(C, wJ, X, nQ, dim, nJ, isRel, r, isPer, period, ...
            inv2s2, sigma);
    else
        chunkSize = max(1, floor(memLimit / ...
            ((dim + 1) * double(nJ) * bytesPerScalar)));
        for c0 = 1:chunkSize:nQ
            c1 = min(c0 + chunkSize - 1, nQ);
            idx = c0:c1;
            v(idx) = evalChunk(C, wJ, X(:, idx), numel(idx), ...
                dim, nJ, isRel, r, isPer, period, inv2s2, sigma);
        end
    end
end

function v = evalChunk(C, wJ, Xq, nQc, dim, nJ, isRel, r, isPer, period, inv2s2, sigma) %#ok<INUSL>
    D = reshape(C, dim, nJ, 1) - reshape(Xq, dim, 1, nQc);
    if isPer
        D = mod(D + period / 2, period) - period / 2;
    end
    if isRel
        Qvec = sum(D.^2, 1) - sum(D, 1).^2 / r;
    else
        Qvec = sum(D.^2, 1);
    end
    % Use direct Q / (2*sigma^2) division (not the precomputed inv2s2
    % shortcut) so the default-path output is FP-bit-identical to the
    % v2.0/v2.1 evalFull implementation.
    E = reshape(exp(-Qvec(:) / (2 * sigma^2)), nJ, nQc);
    v = wJ(:)' * E;
end


% =========================================================================
%  Truncated path — grid-bucket spatial index
% =========================================================================

function v = localTruncatedKernelSum(C, wJ, X, sigma, isRel, r, kSigma, inv2s2)
    dim = size(C, 1);
    nJ  = size(C, 2);
    nQ  = size(X, 2);

    if nJ == 0
        v = zeros(1, nQ, 'like', C);
        return;
    end

    % Coordinate transform so the Q-ball is a Euclidean sphere in
    % transformed space. For abs mode this is the identity; for rel
    % mode Q = D' M D with M = I - 11'/r, so we diagonalise M and
    % rescale.
    if isRel
        e = ones(dim, 1, 'like', C);
        M = eye(dim, 'like', C) - (1/r) * (e * e');
        [U, Lambda] = eig(double(M));
        lams = diag(Lambda);
        if any(lams < 0)
            % Numerical noise on PSD form; clip.
            lams = max(lams, 0);
        end
        sqrtL = cast(sqrt(lams), 'like', C);
        T = cast(U, 'like', C) * diag(sqrtL);
        T = T';            % so that T * D gives transformed coords
    else
        T = eye(dim, 'like', C);
    end

    Ct = T * C;            % (dim, nJ) transformed centres
    Xt = T * X;            % (dim, nQ) transformed queries

    threshold2 = (kSigma * sigma)^2;
    bucketSize = kSigma * sigma;

    % Compute bucket grid spanning Ct's bounding box.
    cMin = min(Ct, [], 2);
    cMax = max(Ct, [], 2);
    nBuckets = max(1, double(ceil(double(cMax - cMin) / double(bucketSize)) + 1));

    % Centre bucket coordinates.
    buckIdxC = floor(double(Ct - cMin) / double(bucketSize)) + 1;
    buckIdxC = max(1, min(nBuckets, buckIdxC));

    % Group centres by linear bucket index.
    linIdxC = localSubToInd(nBuckets, buckIdxC);
    [sortedLin, perm] = sort(linIdxC);
    % Run-length boundaries:
    if isempty(sortedLin)
        boundaries = [];
    else
        diffs = [true; diff(sortedLin) ~= 0];
        boundaries = find(diffs);
    end
    runStarts = boundaries;
    runEnds   = [boundaries(2:end) - 1; numel(sortedLin)];
    runLinIdx = sortedLin(boundaries);

    % Map linear bucket index -> run index (0 = empty).
    linIdxMax = prod(nBuckets);
    bucketMap = zeros(linIdxMax, 1);
    bucketMap(runLinIdx) = 1:numel(runLinIdx);

    % Neighbour offsets: 3^dim combinations.
    offsetGrid = localNeighbourOffsets(dim);
    nOff = size(offsetGrid, 2);

    % Query bucket coords (clamped).
    buckIdxX = floor(double(Xt - cMin) / double(bucketSize)) + 1;
    buckIdxX = max(1, min(nBuckets, buckIdxX));

    v = zeros(1, nQ, 'like', C);

    for q = 1:nQ
        qBucket = buckIdxX(:, q);

        % Collect candidate centres from neighbouring buckets.
        candidates = zeros(0, 1);
        for off = 1:nOff
            nbBucket = qBucket + offsetGrid(:, off);
            if any(nbBucket < 1) || any(nbBucket > nBuckets)
                continue;
            end
            nbLin = localSubToInd(nBuckets, nbBucket);
            runIdx = bucketMap(nbLin);
            if runIdx > 0
                idxs = perm(runStarts(runIdx):runEnds(runIdx));
                candidates = [candidates; idxs(:)]; %#ok<AGROW>
            end
        end

        if isempty(candidates)
            continue;
        end

        % Compute Q for candidates; keep those inside the Q-ball.
        Dq = C(:, candidates) - X(:, q);
        if isRel
            Q = sum(Dq .* Dq, 1) - sum(Dq, 1).^2 / r;
        else
            Q = sum(Dq .* Dq, 1);
        end
        keep = Q(:) <= threshold2;
        if any(keep)
            survivors = candidates(keep);
            Qkept = Q(keep);
            kernelVals = exp(-Qkept(:) * inv2s2);
            v(q) = sum(wJ(survivors) .* kernelVals);
        end
    end
end


function linIdx = localSubToInd(siz, subs)
%LOCALSUBTOIND  Vectorised sub2ind for (nDim, n) subscript columns.
    [nDim, ~] = size(subs);
    if nDim == 1
        linIdx = subs(:);
        return;
    end
    linIdx = double(subs(1, :)');
    strideMult = 1;
    for d = 2:nDim
        strideMult = strideMult * siz(d - 1);
        linIdx = linIdx + (double(subs(d, :)') - 1) * strideMult;
    end
end


function offsets = localNeighbourOffsets(dim)
%LOCALNEIGHBOUROFFSETS  3^dim integer offsets, columns are dim-vectors.
    nOff = 3^dim;
    offsets = zeros(dim, nOff);
    for i = 1:nOff
        idx = i - 1;
        for d = 1:dim
            offsets(d, i) = mod(idx, 3) - 1;
            idx = floor(idx / 3);
        end
    end
end


% =========================================================================
%  Vectorised 1-D abs-mode truncated path
%
%  For dim=1 absolute-mode workloads, sort the centres along their
%  single axis and use binary search to find each query's active
%  window [x - kσ, x + kσ]. All queries then process a fixed-width
%  slice of centres (the max window in the batch), masked beyond
%  their per-query window. Avoids the per-query MATLAB for-loop in
%  localTruncatedKernelSum; ~3-30x faster at typical orbit-path
%  N (50-300 partials).
%
%  Used in particular by mobius.evalOrbitAbs for the per-block 1-D
%  kernel sum that arises after factoring the block's m-dimensional
%  quadratic form Q_B = var(x_B) + m*(mean(x_B) - p)^2.
% =========================================================================

function v = localTruncatedKernelSum1D(C, wJ, X, sigma, kSigma, inv2s2)
%LOCALTRUNCATEDKERNELSUM1D  1-D abs vectorised truncated kernel sum.
%
%   v(q) = sum_{j: |C(j) - X(q)| <= kSigma*sigma} wJ(j) ...
%             * exp(-0.5 * ((X(q) - C(j))/sigma)^2)
%
%   Inputs:
%     C       (1, nJ) double - single-axis centres
%     wJ      (nJ, 1) double - centre weights
%     X       (1, nQ) double - query coordinates along the single axis
%     sigma   (1, 1) double, positive
%     kSigma  (1, 1) double, positive (truncation radius in sigmas)
%     inv2s2  (1, 1) double = 1/(2*sigma^2)
%
%   Output:
%     v       (1, nQ) row vector matching the dtype family of C.
%
%   Implementation note (subtle MATLAB indexing rule):
%     cSorted is a row vector (1, nJ). After sorting, we explicitly
%     reshape it to a column. This is *required* for correctness, not
%     defensive: when maxWin == 1, the sparse-path index matrix
%     idxClipped collapses to a (nQ, 1) column vector. Then
%     cSorted(idxClipped) follows MATLAB's "vector source, vector
%     index" rule and returns a result matching the *source*'s
%     orientation. If cSorted were left as a row, pSlices would come
%     out as (1, nQ) and the subsequent `xAxis(:) - pSlices` would
%     outer-broadcast (column - row) to (nQ, nQ), which OOMs at
%     large nQ. Forcing cSorted to a column makes the sparse path
%     return (nQ, 1) at maxWin == 1, matching xAxis(:). For
%     maxWin >= 2, idxClipped is a (nQ, maxWin) non-vector matrix,
%     and the "matrix index" rule returns (nQ, maxWin) regardless
%     of cSorted's orientation — so the fix is a no-op there.

    nJ = size(C, 2);
    nQ = size(X, 2);
    if nJ == 0 || nQ == 0
        v = zeros(1, nQ, 'like', C);
        return;
    end

    threshold = double(kSigma) * double(sigma);
    cAxis = double(C(1, :));               % (1, nJ)
    xAxis = double(X(1, :));               % (1, nQ)

    [cSorted, order] = sort(cAxis);
    cSorted = cSorted(:);                   % force COLUMN — see header note
    wSorted = wJ(order);
    wSorted = wSorted(:);                   % match cSorted's orientation

    lo = xAxis - threshold;
    hi = xAxis + threshold;
    iLow0 = sum(cSorted < lo, 1);          % (1, nQ), 0-indexed
    iHigh0 = sum(cSorted <= hi, 1);        % (1, nQ), 0-indexed (one-past-last)
    winSize = iHigh0 - iLow0;
    maxWin = max(winSize);

    if maxWin == 0
        v = zeros(1, nQ, 'like', C);
        return;
    end
    if maxWin >= nJ
        % Window covers everything; dense compute.
        diffs = xAxis(:) - cSorted.';      % (nQ, nJ)
        kernel = exp(-(diffs .^ 2) * inv2s2);
        v = (kernel * wSorted).';
        v = cast(v, 'like', C);
        return;
    end

    % Build (nQ, maxWin) index matrix into the sorted arrays.
    offsets = 0:(maxWin - 1);
    idx = iLow0(:) + offsets + 1;          % 1-indexed for MATLAB
    mask = idx <= iHigh0(:);
    idxClipped = min(idx, nJ);

    % cSorted (and wSorted) are columns (forced above), so the
    % "vector source + vector index" case (maxWin == 1) returns a
    % column matching idxClipped, and the "vector source + matrix
    % index" case (maxWin >= 2) returns a matrix matching idxClipped.
    pSlices = cSorted(idxClipped);          % (nQ, maxWin) for all maxWin
    wSlices = wSorted(idxClipped);
    diffs = xAxis(:) - pSlices;             % (nQ, maxWin)
    kernel = exp(-(diffs .^ 2) * inv2s2);
    kernel(~mask) = 0;

    v = sum(kernel .* wSlices, 2).';
    v = cast(v, 'like', C);
end

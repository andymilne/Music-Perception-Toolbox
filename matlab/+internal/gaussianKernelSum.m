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
%   Periodic mode (opts.isPer = true), 1-D abs: truncates on the circle
%   when the window is narrower than half the circumference
%   (2*truncationSigmas*sigma < period); otherwise, and for the rel or
%   multi-axis periodic cases, the exact wrapped path is taken.
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
        opts.wrap (1,:) char {mustBeMember(opts.wrap, ...
            {'full-image', 'single-image'})} = 'full-image'
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

    % Resolve the truncation knob through the accuracy floor: Inf (the
    % "exact" sentinel) becomes the finite width at which the kernel
    % falls below the 1e-12 parity floor, uniform with Python.
    opts.truncationSigmas = internal.accuracyFloor('resolve', ...
                                                   opts.truncationSigmas);

    % Decide path. Non-periodic and periodic 1-D abs modes both truncate;
    % see the dispatch below.
    useTruncation = isfinite(opts.truncationSigmas) ...
                 && opts.truncationSigmas > 0 ...
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

    if useTruncation && ~opts.isPer
        % 1-D abs case: vectorised path via sorted-centres +
        % searchsorted, much faster than the general per-query loop.
        if size(C_w, 1) == 1 && ~opts.isRel
            v_w = localTruncatedKernelSum1D(C_w, wJ_w, X_w, sigma_w, ...
                opts.truncationSigmas, inv2s2);
        else
            v_w = localTruncatedKernelSum(C_w, wJ_w, X_w, sigma_w, ...
                opts.isRel, opts.r, opts.truncationSigmas, inv2s2);
        end
    elseif useTruncation && opts.isPer && size(C_w, 1) == 1 && ~opts.isRel
        % Circular 1-D truncation, valid only when the window is narrower
        % than the circle; otherwise there are no savings (and the
        % replication trick would double count), so fall through to the
        % exact periodic path.
        radius = double(opts.truncationSigmas) * double(sigma_w);
        if 2.0 * radius < double(period_w)
            v_w = localTruncatedKernelSum1DCircular(C_w, wJ_w, X_w, ...
                sigma_w, period_w, opts.truncationSigmas, inv2s2);
        else
            v_w = localExactKernelSum(C_w, wJ_w, X_w, ...
                opts.isRel, opts.r, opts.isPer, period_w, inv2s2, sigma_w, ...
                opts.wrap, opts.truncationSigmas);
        end
    else
        v_w = localExactKernelSum(C_w, wJ_w, X_w, ...
            opts.isRel, opts.r, opts.isPer, period_w, inv2s2, sigma_w, ...
            opts.wrap, opts.truncationSigmas);
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

function v = localExactKernelSum(C, wJ, X, isRel, r, isPer, period, ...
                                  inv2s2, sigma, wrap, truncationSigmas)
    dim = size(C, 1);
    nJ  = size(C, 2);
    nQ  = size(X, 2);

    if nJ == 0 || nQ == 0
        v = zeros(1, nQ, 'like', C);
        return;
    end

    % Peak per-chunk transient ~ (2*dim + 2) * nJ * nQ * bytesPerScalar:
    % MATLAB briefly holds the broadcast difference tensor, its square,
    % and the summed-then-exponentiated intermediate co-resident.
    bytesPerScalar = 4 * isa(C, 'single') + 8 * isa(C, 'double');
    bytesNeeded = (2 * dim + 2) * double(nJ) * double(nQ) * bytesPerScalar;
    memLimit = internal.kernelChunkBytesResolved();

    v = zeros(1, nQ, 'like', C);
    if bytesNeeded <= memLimit
        v = evalChunk(C, wJ, X, nQ, dim, nJ, isRel, r, isPer, period, ...
            inv2s2, sigma, wrap, truncationSigmas);
    else
        chunkSize = max(1, floor(memLimit / ...
            ((2 * dim + 2) * double(nJ) * bytesPerScalar)));
        for c0 = 1:chunkSize:nQ
            c1 = min(c0 + chunkSize - 1, nQ);
            idx = c0:c1;
            v(idx) = evalChunk(C, wJ, X(:, idx), numel(idx), ...
                dim, nJ, isRel, r, isPer, period, inv2s2, sigma, ...
                wrap, truncationSigmas);
        end
    end
end

function v = evalChunk(C, wJ, Xq, nQc, dim, nJ, isRel, r, isPer, period, ...
                        inv2s2, sigma, wrap, truncationSigmas) %#ok<INUSL>
    D = reshape(C, dim, nJ, 1) - reshape(Xq, dim, 1, nQc);
    % Abs-per: full-image via the shared wrapped-Gaussian helper
    % (image-sum or Fourier by cost; density-kernel convention with
    % exponent_denominator = 2). Single-image opt-in reduces to the
    % nearest image and evaluates that Gaussian only, matching pre-v3
    % behaviour.
    if isPer && ~isRel
        if strcmp(wrap, 'full-image')
            % Per-position theta then product across tuple positions. wrappedGaussian1d
            % handles nearest-image reduction internally and picks the
            % cheaper of image-sum and Fourier for the summation.
            theta = internal.wrappedGaussian1d(D, sigma, period, ...
                                                truncationSigmas, 2);
            % theta has shape (dim, nJ, nQc); product over dim = axis 1.
            E = reshape(prod(theta, 1), nJ, nQc);
            v = wJ(:)' * E;
            return
        end
        % Single-image opt-in: reduce to nearest image and fall through
        % to the sum-of-squares path below.
        D = D - period .* floor(D / period + 0.5);
    end
    if isRel
        if isPer
            % Pairwise-wrap form on the reduced centres
            % representation (position 0 = 0 implicit). Position-0 pairs
            % vectorised in a single pass; within-reduced pairs
            % looped.
            slot0_wrapped = D - period .* floor(D / period + 0.5);
            Qvec = sum(slot0_wrapped .^ 2, 1);
            for i = 1:dim
                for j = i+1:dim
                    delta = D(i, :, :) - D(j, :, :);
                    delta = delta - period .* floor(delta / period + 0.5);
                    Qvec = Qvec + delta.^2;
                end
            end
            Qvec = Qvec / r;
        else
            Qvec = sum(D.^2, 1) - sum(D, 1).^2 / r;
        end
    else
        Qvec = sum(D.^2, 1);
    end
    % Use direct Q / (2*sigma^2) division (not the precomputed inv2s2
    % shortcut) so the default-path output is FP-bit-identical to the
    % v2.0/v2.1 evalFull implementation (in all modes except
    % periodic+relative, where v2.X uses the pairwise-wrap form).
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

    if nJ == 0 || nQ == 0
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
    [sortedLin, permIdx] = sort(linIdxC);
    if isempty(sortedLin)
        v = zeros(1, nQ, 'like', C);
        return;
    end
    diffs = [true; diff(sortedLin) ~= 0];
    boundaries = find(diffs);
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

    % ----- Vectorised per-chunk processing -----
    % For each query chunk:
    %   1. Expand each query's bucket to its 3^dim neighbour buckets.
    %   2. Mask in-bounds neighbours; look up non-empty runs in bucketMap.
    %   3. Flatten each (query, run) pair into per-centre pairs via a
    %      cumsum-based ragged expansion (no AGROW, no inner loop).
    %   4. Compute Q for all (query, centre) pairs in one vector op,
    %      apply the Q-ball threshold, and accumulate kernel sums back
    %      onto queries via accumarray.
    % Chunking bounds peak (dim, totalPairs) workspace memory; the
    % budget matches the exact path's heuristic so memory behaviour is
    % consistent across paths.
    bytesPerScalar = 4 * isa(C, 'single') + 8 * isa(C, 'double');
    memLimit = internal.kernelChunkBytesResolved();
    chunkSize = max(1, floor(memLimit / ...
        ((2 * dim + 2) * double(nJ) * bytesPerScalar)));

    v = zeros(1, nQ, 'like', C);
    for c0 = 1:chunkSize:nQ
        c1 = min(c0 + chunkSize - 1, nQ);
        qIdx = c0:c1;                          % row 1 × nQc
        nQc  = numel(qIdx);

        % 1. Neighbour bucket coords for every query in the chunk.
        %    Shape (dim, nOff, nQc), flattened to (dim, nOff*nQc).
        buckQc = buckIdxX(:, qIdx);                                 % dim × nQc
        nbAll  = reshape(buckQc, dim, 1, nQc) + ...
                 reshape(offsetGrid, dim, nOff, 1);
        nbAll  = reshape(nbAll, dim, nOff * nQc);

        % Local query index (within chunk) for each neighbour column.
        queryOfNb = reshape(repmat(1:nQc, nOff, 1), [], 1);          % column

        % 2. In-bounds mask, then bucket-run lookup.
        inBounds = all(nbAll >= 1 & nbAll <= nBuckets, 1);
        if ~any(inBounds)
            continue;
        end
        nbAll     = nbAll(:, inBounds);
        queryOfNb = queryOfNb(inBounds);

        nbLin   = localSubToInd(nBuckets, nbAll);
        runIdx  = bucketMap(nbLin);
        hasRun  = runIdx > 0;
        if ~any(hasRun)
            continue;
        end
        runIdx     = runIdx(hasRun);
        queryValid = queryOfNb(hasRun);

        % 3. Ragged-expand each (query, run) pair into per-centre pairs.
        runLens    = runEnds(runIdx) - runStarts(runIdx) + 1;
        totalPairs = sum(runLens);
        if totalPairs == 0
            continue;
        end

        queryAll = repelem(queryValid, runLens);                     % column

        % Build positional indices into `permIdx` via cumsum:
        % runMembership(k) is the (queryValid, runIdx) pair-index for
        % the k-th output element; localOffsets(k) is its offset within
        % that run.
        endPos        = cumsum(runLens);
        startPos      = endPos - runLens + 1;
        startMark     = zeros(totalPairs, 1);
        startMark(startPos) = 1;
        runMembership = cumsum(startMark);
        localOffsets  = (1:totalPairs)' - startPos(runMembership) + 1;
        permPos       = runStarts(runIdx(runMembership)) + localOffsets - 1;
        centreAll     = permIdx(permPos);                            % column

        % 4. Distance, threshold, accumulate.
        Dq = C(:, centreAll) - X(:, qIdx(queryAll));
        if isRel
            Q = sum(Dq .* Dq, 1) - sum(Dq, 1).^2 / r;
        else
            Q = sum(Dq .* Dq, 1);
        end
        keep = Q(:) <= threshold2;
        if ~any(keep)
            continue;
        end
        centreKept = centreAll(keep);
        queryKept  = queryAll(keep);
        Qkept      = Q(keep);
        % Force columns: for tiny dims (dim = 1, nQ = 1) the index vectors
        % above can pick up a row orientation, which would broadcast the
        % product into a matrix and break accumarray. All three have
        % nnz(keep) elements, so column-forcing is exact.
        wKept      = wJ(centreKept(:));
        kernelVals = wKept(:) .* exp(-Qkept(:) * inv2s2);

        vChunk = accumarray(queryKept(:), kernelVals, [nQc, 1]);
        v(qIdx) = v(qIdx) + cast(vChunk, 'like', C).';
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
%     The sparse path indexes the sorted centres with a conceptually
%     (nQ, maxWin) index matrix idxClipped. When either nQ == 1 or
%     maxWin == 1 that matrix collapses to a vector, and
%     cSorted(idxClipped) then follows MATLAB's "vector source, vector
%     index" rule, returning a result in the source's orientation rather
%     than (nQ, maxWin). Left unguarded this mis-shapes the output: the
%     single-column corner would outer-broadcast in xAxis(:) - pSlices,
%     and the single-query corner returns maxWin values for one query.
%     The sparse path therefore reshapes pSlices and wSlices to
%     (nQ, maxWin) explicitly, which is exact for every nQ and maxWin
%     (reshape preserves the column-major element order, so entry (i, j)
%     stays cSorted(idxClipped(i, j))) and a no-op once both dimensions
%     exceed one. cSorted and wSorted are still kept as columns for the
%     dense maxWin >= nJ branch below.

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

    % idxClipped is conceptually (nQ, maxWin). MATLAB's indexing returns
    % a shape that depends on whether idxClipped is a vector: when
    % nQ == 1 (single query) or maxWin == 1, it collapses to a vector and
    % cSorted(idxClipped) follows the "vector source + vector index" rule,
    % returning the source's (column) orientation rather than
    % (nQ, maxWin). The explicit reshape restores the intended layout in
    % every case --- a no-op when both nQ >= 2 and maxWin >= 2, and the
    % correctness fix for the single-query and single-column corners.
    % (The Python twin needs no analogue: NumPy advanced indexing already
    % takes the index array's shape.)
    pSlices = reshape(cSorted(idxClipped), nQ, maxWin);
    wSlices = reshape(wSorted(idxClipped), nQ, maxWin);
    diffs = xAxis(:) - pSlices;             % (nQ, maxWin)
    kernel = exp(-(diffs .^ 2) * inv2s2);
    kernel(~mask) = 0;

    v = sum(kernel .* wSlices, 2).';
    v = cast(v, 'like', C);
end


function v = localTruncatedKernelSum1DCircular(C, wJ, X, sigma, period, ...
        kSigma, inv2s2)
%LOCALTRUNCATEDKERNELSUM1DCIRCULAR  Circular twin of localTruncatedKernelSum1D.
%
%   v(q) = sum_i wJ(i) * exp(-wrap(X(q)-C(i))^2/(2*sigma^2)) on the circle
%   of circumference PERIOD, including only centres within kSigma*sigma
%   (wrapped) of each query. Requires the window narrower than the circle
%   (2*radius < period); the caller guards this. Centres are replicated at
%   c-P, c, c+P so a wrapped window maps to a contiguous range of the
%   sorted array; since the window is narrower than P at most one copy of
%   any centre falls inside, so there is no double counting, and distances
%   are then plain (the nearest copy realises the wrapped distance).
%   Mirrors the Python _truncated_kernel_sum_1d_circular. The reshape of
%   pSlices/wSlices guards the single-query and single-column corners
%   exactly as in localTruncatedKernelSum1D (see its header note).

    nJ = size(C, 2);
    nQ = size(X, 2);
    if nJ == 0 || nQ == 0
        v = zeros(1, nQ, 'like', C);
        return;
    end

    threshold = double(kSigma) * double(sigma);
    P = double(period);
    cAxis = mod(double(C(1, :)), P);        % (1, nJ)
    xAxis = mod(double(X(1, :)), P);        % (1, nQ)

    % Triple the centres across one period on each side.
    c3 = [cAxis - P, cAxis, cAxis + P];     % (1, 3*nJ)
    w3 = [wJ(:); wJ(:); wJ(:)];             % (3*nJ, 1)
    [cSorted, order] = sort(c3);            % stable, ascending
    cSorted = cSorted(:);                   % force COLUMN
    wSorted = w3(order);
    wSorted = wSorted(:);                   % match cSorted's orientation

    lo = xAxis - threshold;
    hi = xAxis + threshold;
    iLow0 = sum(cSorted < lo, 1);           % (1, nQ), 0-indexed
    iHigh0 = sum(cSorted <= hi, 1);         % (1, nQ), 0-indexed (one-past-last)
    winSize = iHigh0 - iLow0;
    maxWin = max(winSize);

    if maxWin == 0
        v = zeros(1, nQ, 'like', C);
        return;
    end

    nTot = numel(cSorted);                  % 3*nJ
    offsets = 0:(maxWin - 1);
    idx = iLow0(:) + offsets + 1;           % 1-indexed for MATLAB
    mask = idx <= iHigh0(:);
    idxClipped = min(idx, nTot);

    pSlices = reshape(cSorted(idxClipped), nQ, maxWin);
    wSlices = reshape(wSorted(idxClipped), nQ, maxWin);
    diffs = xAxis(:) - pSlices;             % (nQ, maxWin)
    kernel = exp(-(diffs .^ 2) * inv2s2);
    kernel(~mask) = 0;

    v = sum(kernel .* wSlices, 2).';
    v = cast(v, 'like', C);
end

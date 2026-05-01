function dens = buildExpTens(varargin)
%BUILDEXPTENS Precompute an r-ad expectation tensor density object.
%
%   SINGLE-ATTRIBUTE (legacy):
%     dens = buildExpTens(p, w, sigma, r, isRel, isPer, period)
%     dens = buildExpTens(..., 'verbose', false)
%
%   MULTI-ATTRIBUTE (MAET):
%     dens = buildExpTens(pAttr, w, sigmaVec, rVec, groups, ...
%                         isRelVec, isPerVec, periodVec)
%     dens = buildExpTens(..., 'verbose', false)
%
%   The function dispatches on the type of the first argument:
%     - numeric vector  -> single-attribute path, returns struct with
%                          tag = 'ExpTensDensity' (v2.0.0 behaviour, unchanged)
%     - cell array      -> multi-attribute path, returns struct with
%                          tag = 'MaetDensity'
%
%   Inputs (single-attribute path):
%     p         - Pitch or position values (vector of length N).
%     w         - Weights (vector of length N, empty, or scalar — see
%                 the toolbox's standard broadcast convention in
%                 User Guide §4).
%     sigma     - Standard deviation of the Gaussian kernel.
%     r         - Tuple size (positive integer; r >= 2 if isRel = true).
%     isRel     - If true, use transposition-invariant (relative)
%                 quadratic form (effective dim = r - 1).
%     isPer     - If true, wrap differences to periodic interval
%                 [-period/2, period/2).
%     period    - Period for periodic wrapping (e.g., 1200 for one
%                 octave in cents, or the cycle length for rhythmic
%                 analyses).
%
%   Inputs (multi-attribute path):
%     pAttr     - 1 x A cell array of K_a x N matrices (attribute values).
%                 Shapes are honoured literally: a [K x 1] column is
%                 K slots of one event, a [1 x N] row is one slot of N
%                 events, and a [K x N] matrix is K slots of N events.
%                 No flattening is applied.
%     w         - Weights. One of:
%                   []       -> all ones
%                   scalar   -> uniform value, broadcast to all attributes
%                   1 x A cell of per-attribute inputs
%                 Each per-attribute input is [], scalar, 1 x N row,
%                 K_a x 1 column, or K_a x N matrix; broadcasts to K_a x N.
%     sigmaVec  - 1 x G vector of per-group Gaussian widths
%     rVec      - 1 x A vector of per-attribute tuple sizes
%     groups    - Group assignment. One of:
%                   []                      -> each attribute its own group
%                   1 x A vector of indices -> explicit indices
%                   1 x G cell of attr lists -> explicit partition
%     isRelVec  - 1 x G logical vector of per-group isRel flags
%     isPerVec  - 1 x G logical vector of per-group periodic flags
%     periodVec - 1 x G vector of per-group periods (0 when not periodic)
%
%   Output (multi-attribute path):
%     dens - struct with fields:
%       .tag           = 'MaetDensity'
%       .nAttrs        = A
%       .nGroups       = G
%       .N             = number of events
%       .groupOfAttr   = 1 x A vector, group index per attribute
%       .attrsOfGroup  = 1 x G cell, attribute indices per group
%       .r             = 1 x A vector, per-attribute tuple size
%       .K             = 1 x A vector, max K_a (for reference)
%       .pAttr         = 1 x A cell, input value matrices
%       .w             = 1 x A cell, post-broadcast K_a x N weight matrices
%       .sigma         = 1 x G
%       .isRel         = 1 x G logical
%       .isPer         = 1 x G logical
%       .period        = 1 x G
%       .dim           = scalar total, sum_a (r_a - isRel_{g(a)})
%       .dimPerAttr    = 1 x A vector, r_a - isRel_{g(a)}
%       .nJ, .nK       = scalars, total perm-side / comb-side tuple counts
%       .Centres       = 1 x A cell, each (r_a - isRel_{g(a)}) x nJ
%       .U_perm        = 1 x A cell, each r_a x nJ (perm side)
%       .V_comb        = 1 x A cell, each r_a x nK (comb side)
%       .wJ            = 1 x nJ, per-tuple weight products (perm side)
%       .wv_comb       = 1 x nK, per-tuple weight products (comb side)
%       .eventOfJ      = 1 x nJ, event index for each perm-side tuple
%       .eventOfK      = 1 x nK, event index for each comb-side tuple
%
%   Column j in Centres{a}, U_perm{a}, wJ, and eventOfJ all refer to the
%   same global perm-side tuple. Similarly for column k across V_comb{a},
%   wv_comb, and eventOfK.
%
%   See also evalExpTens, cosSimExpTens.

    % ------------------------------------------------------------------
    % Parse optional 'verbose' name-value pair and split positional args
    % ------------------------------------------------------------------
    [posArgs, verbose] = localExtractVerbose(varargin);

    if isempty(posArgs)
        error('buildExpTens:missingInputs', ...
              'At least the pitch/attribute input is required.');
    end

    first = posArgs{1};

    if iscell(first)
        % Multi-attribute path
        dens = localBuildMA(posArgs, verbose);
    elseif isnumeric(first)
        % Single-attribute legacy path
        dens = localBuildSA(posArgs, verbose);
    else
        error('buildExpTens:badFirstArg', ...
              ['First argument must be a numeric vector (single-attribute) ' ...
               'or a cell array of attribute matrices (multi-attribute).']);
    end
end


% ======================================================================
%  Helpers: verbose parsing
% ======================================================================

function [posArgs, verbose] = localExtractVerbose(args)
    verbose = true;
    posArgs = args;
    for i = 1:numel(args)
        if (ischar(args{i}) || isstring(args{i})) && strcmpi(args{i}, 'verbose')
            if i + 1 <= numel(args)
                verbose = logical(args{i + 1});
            end
            posArgs = args(1:i - 1);
            return;
        end
    end
end


% ======================================================================
%  Single-attribute (legacy v2.0.0) path
% ======================================================================

function dens = localBuildSA(posArgs, verbose)

    if numel(posArgs) ~= 7
        error('buildExpTens:saArgCount', ...
              ['Single-attribute call expects 7 positional arguments: ' ...
               'p, w, sigma, r, isRel, isPer, period.']);
    end
    [p, w, sigma, r, isRel, isPer, period] = posArgs{:};

    p = p(:);
    w = w(:);

    if isempty(w)
        w = ones(numel(p), 1);
    end
    if isscalar(w)
        if w == 0
            warning('All weights in w are zero.');
        end
        w = w * ones(numel(p), 1);
    end

    if rem(r, 1) || r < 1
        error('''r'' must be a positive integer.');
    elseif r > numel(p)
        error('''r'' must not exceed the number of values.');
    elseif numel(p) ~= numel(w)
        error('w must have the same number of entries as p.');
    end

    if isRel && r < 2
        error('For relative densities, ''r'' must be at least 2.');
    end

    dim = r - isRel;

    n      = numel(p);
    nPerms = factorial(r);
    nCombs = nchoosek(n, r);
    nJ     = nPerms * nCombs;
    nK     = nCombs;

    if verbose
        fprintf('buildExpTens: building %d ordered %d-tuples from %d values.\n', ...
            nJ, r, n);
    end

    allPerms = perms(1:r)';
    nck      = nchoosek(1:numel(p), r)';

    Ju     = zeros(r, nJ);
    offset = 0;
    for i = 1:nPerms
        Ju(:, offset + 1 : offset + nCombs) = nck(allPerms(:, i), :);
        offset = offset + nCombs;
    end

    U_perm = reshape(p(Ju), r, nJ);
    w_perm = reshape(prod(reshape(w(Ju), r, nJ), 1), 1, nJ);

    Kv      = nck;
    V_comb  = reshape(p(Kv), r, nK);
    wv_comb = reshape(prod(reshape(w(Kv), r, nK), 1), 1, nK);

    if isRel
        Centres = U_perm(2:r, :) - U_perm(1, :);
    else
        Centres = U_perm;
    end

    dens.tag     = 'ExpTensDensity';
    dens.p       = p;
    dens.w       = w;
    dens.sigma   = sigma;
    dens.r       = r;
    dens.isRel   = isRel;
    dens.isPer   = isPer;
    dens.period  = period;
    dens.dim     = dim;

    dens.Centres = Centres;
    dens.wJ      = w_perm;
    dens.nJ      = nJ;

    dens.U_perm  = U_perm;
    dens.w_perm  = w_perm;
    dens.nJ_perm = nJ;
    dens.V_comb  = V_comb;
    dens.wv_comb = wv_comb;
    dens.nK      = nK;
end


% ======================================================================
%  Multi-attribute (MAET) path
% ======================================================================

function dens = localBuildMA(posArgs, verbose)

    if numel(posArgs) ~= 8
        error('buildExpTens:maArgCount', ...
              ['Multi-attribute call expects 8 positional arguments: ' ...
               'pAttr, w, sigmaVec, rVec, groups, isRelVec, isPerVec, periodVec.']);
    end
    [pAttr, wIn, sigmaVec, rVec, groupsIn, isRelVec, isPerVec, periodVec] = posArgs{:};

    % --- Input normalisation ---

    if ~iscell(pAttr) || isempty(pAttr)
        error('buildExpTens:badPAttr', ...
              'pAttr must be a non-empty cell array of attribute matrices.');
    end
    A = numel(pAttr);

    % Coerce each attribute input to its 2-D K_a x N shape. MATLAB
    % treats everything as at least 2-D, so a user-supplied column
    % vector [K x 1] is read as K slots / 1 event, a row vector [1 x N]
    % as 1 slot / N events, and a matrix [K x N] as K slots / N events,
    % with no ambiguity. (A bare scalar is 1 x 1 and fills the K=N=1
    % case.) No flattening is applied — flattening a 2-D input would
    % silently reinterpret columns as rows.
    for a = 1:A
        M = pAttr{a};
        if ~isnumeric(M)
            error('buildExpTens:badAttrType', ...
                  'Attribute %d input must be numeric.', a);
        end
        if ndims(M) > 2
            error('buildExpTens:badAttrDims', ...
                  'Attribute %d input must be at most 2-D; got ndims=%d.', ...
                  a, ndims(M));
        end
        pAttr{a} = double(M);
    end

    Ns = cellfun(@(M) size(M, 2), pAttr);
    if any(Ns ~= Ns(1))
        error('buildExpTens:eventCountMismatch', ...
              ['All attribute matrices must share the same number of ' ...
               'columns (events). Got: %s.'], mat2str(Ns));
    end
    N = Ns(1);

    Ka = cellfun(@(M) size(M, 1), pAttr);

    % r per attribute
    rVec = double(rVec(:).');
    if numel(rVec) ~= A
        error('buildExpTens:rLength', ...
              'rVec must have length equal to the number of attributes.');
    end
    if any(rVec < 1) || any(rem(rVec, 1) ~= 0)
        error('buildExpTens:rNotInt', 'All r_a must be positive integers.');
    end

    % Groups
    [groupOfAttr, attrsOfGroup, G] = localCanonicalizeGroups(groupsIn, A);

    % Per-group parameters
    sigmaVec  = double(sigmaVec(:).');
    isRelVec  = logical(isRelVec(:).');
    isPerVec  = logical(isPerVec(:).');
    periodVec = double(periodVec(:).');
    if numel(sigmaVec)  ~= G, error('buildExpTens:sigmaLength',  'sigmaVec must have length %d (nGroups).',  G); end
    if numel(isRelVec)  ~= G, error('buildExpTens:isRelLength',  'isRelVec must have length %d (nGroups).',  G); end
    if numel(isPerVec)  ~= G, error('buildExpTens:isPerLength',  'isPerVec must have length %d (nGroups).',  G); end
    if numel(periodVec) ~= G, error('buildExpTens:periodLength', 'periodVec must have length %d (nGroups).', G); end

    % isRel + r_a = 1 degenerate warning (per attribute)
    for a = 1:A
        g = groupOfAttr(a);
        if isRelVec(g) && rVec(a) < 2
            warning('buildExpTens:isRelDegenerate', ...
                    ['isRel = true on group %d combined with r_a = 1 for ' ...
                     'attribute %d produces a degenerate (constant) density. ' ...
                     'For cross-event translation invariance, use ' ...
                     'differenceEvents as a preprocessing step.'], g, a);
        end
    end

    % Weights: normalise to 1 x A cell of K_a x N matrices
    wCell = localNormaliseWeights(wIn, A, Ka, N);

    % --- Per-event, per-attribute r-ad enumeration ---

    permIdx = cell(N, A);     % slot indices, perm side: r_a x P_{n,a}
    combIdx = cell(N, A);     % slot indices, comb side: r_a x C_{n,a}
    permW   = cell(N, A);     % per-tuple slot weight products, perm side
    combW   = cell(N, A);     % per-tuple slot weight products, comb side

    for n = 1:N
        for a = 1:A
            valCol  = pAttr{a}(:, n);
            valid   = find(~isnan(valCol));
            K_na    = numel(valid);
            r_a     = rVec(a);
            if K_na < r_a
                error('buildExpTens:insufficientSlots', ...
                      ['Event %d, attribute %d has %d non-NaN slot(s) ' ...
                       'but r_a = %d.'], n, a, K_na, r_a);
            end

            % Combinations: r_a x C(K_na, r_a)
            if K_na == r_a
                combMat = valid(:);
            else
                combMat = nchoosek(valid, r_a).';
            end

            % Permutations: r_a x (r_a! * C(K_na, r_a))
            if r_a == 1
                permMat = combMat;
            else
                Pm = perms(1:r_a).';
                nC = size(combMat, 2);
                nP = size(Pm, 2);
                permMat = zeros(r_a, nC * nP);
                for pp = 1:nP
                    permMat(:, (pp - 1) * nC + 1 : pp * nC) = combMat(Pm(:, pp), :);
                end
            end

            permIdx{n, a} = permMat;
            combIdx{n, a} = combMat;

            % Slot-weight products (per-tuple). wCell{a}(slot, n).
            wCol = wCell{a}(:, n);
            if r_a == 1
                % permMat and combMat are 1 x P; indexing a column
                % vector with a row-shaped index preserves the row
                % shape, giving a 1 x P weight vector directly.
                permW{n, a} = reshape(wCol(permMat), 1, []);
                combW{n, a} = reshape(wCol(combMat), 1, []);
            else
                permW{n, a} = prod(reshape(wCol(permMat), r_a, []), 1);
                combW{n, a} = prod(reshape(wCol(combMat), r_a, []), 1);
            end
        end
    end

    % --- Cartesian product within each event, concatenate across events ---

    nJ_n = zeros(1, N);
    nK_n = zeros(1, N);
    for n = 1:N
        szP = cellfun(@(M) size(M, 2), permIdx(n, :));
        szC = cellfun(@(M) size(M, 2), combIdx(n, :));
        nJ_n(n) = prod(szP);
        nK_n(n) = prod(szC);
    end
    nJ = sum(nJ_n);
    nK = sum(nK_n);

    if verbose
        fprintf(['buildExpTens (MAET): %d attributes, %d groups, %d events. ' ...
                 'Total tuples: nJ = %d (perm), nK = %d (comb).\n'], ...
                A, G, N, nJ, nK);
    end

    U_perm = cell(1, A);
    V_comb = cell(1, A);
    for a = 1:A
        U_perm{a} = zeros(rVec(a), nJ);
        V_comb{a} = zeros(rVec(a), nK);
    end
    wJ        = ones(1, nJ);
    wv_comb   = ones(1, nK);
    eventOfJ  = zeros(1, nJ);
    eventOfK  = zeros(1, nK);

    offJ = 0;
    offK = 0;
    for n = 1:N
        nJh = nJ_n(n);
        nKh = nK_n(n);

        szP = cellfun(@(M) size(M, 2), permIdx(n, :));
        szC = cellfun(@(M) size(M, 2), combIdx(n, :));

        idxPerm = localCartesianIndices(szP);   % 1 x A cell of 1 x nJh
        idxComb = localCartesianIndices(szC);   % 1 x A cell of 1 x nKh

        wJh = ones(1, nJh);
        wKh = ones(1, nKh);

        for a = 1:A
            r_a = rVec(a);
            valCol = pAttr{a}(:, n);

            % Perm side
            slotPerm = permIdx{n, a}(:, idxPerm{a});           % r_a x nJh
            U_perm{a}(:, offJ + 1 : offJ + nJh) = reshape(valCol(slotPerm), r_a, nJh);
            wJh = wJh .* permW{n, a}(idxPerm{a});

            % Comb side
            slotComb = combIdx{n, a}(:, idxComb{a});           % r_a x nKh
            V_comb{a}(:, offK + 1 : offK + nKh) = reshape(valCol(slotComb), r_a, nKh);
            wKh = wKh .* combW{n, a}(idxComb{a});
        end

        wJ(offJ + 1 : offJ + nJh)        = wJh;
        wv_comb(offK + 1 : offK + nKh)   = wKh;
        eventOfJ(offJ + 1 : offJ + nJh)  = n;
        eventOfK(offK + 1 : offK + nKh)  = n;

        offJ = offJ + nJh;
        offK = offK + nKh;
    end

    % --- Centres (per-attribute isRel reduction) ---

    Centres    = cell(1, A);
    dimPerAttr = zeros(1, A);
    for a = 1:A
        g = groupOfAttr(a);
        r_a = rVec(a);
        if isRelVec(g)
            if r_a >= 2
                Centres{a} = U_perm{a}(2:r_a, :) - U_perm{a}(1, :);
                dimPerAttr(a) = r_a - 1;
            else
                % Degenerate case already warned above; emit an empty
                % (0 x nJ) matrix so downstream code can detect it.
                Centres{a} = zeros(0, nJ);
                dimPerAttr(a) = 0;
            end
        else
            Centres{a} = U_perm{a};
            dimPerAttr(a) = r_a;
        end
    end
    dim = sum(dimPerAttr);

    % --- Pack struct ---

    dens = struct();
    dens.tag          = 'MaetDensity';
    dens.nAttrs       = A;
    dens.nGroups      = G;
    dens.N            = N;
    dens.groupOfAttr  = groupOfAttr;
    dens.attrsOfGroup = attrsOfGroup;
    dens.r            = rVec;
    dens.K            = Ka;
    dens.pAttr        = pAttr;
    dens.w            = wCell;
    dens.sigma        = sigmaVec;
    dens.isRel        = isRelVec;
    dens.isPer        = isPerVec;
    dens.period       = periodVec;
    dens.dim          = dim;
    dens.dimPerAttr   = dimPerAttr;
    dens.nJ           = nJ;
    dens.nK           = nK;
    dens.Centres      = Centres;
    dens.U_perm       = U_perm;
    dens.V_comb       = V_comb;
    dens.wJ           = wJ;
    dens.wv_comb      = wv_comb;
    dens.eventOfJ     = eventOfJ;
    dens.eventOfK     = eventOfK;
end


% ======================================================================
%  Helpers: group canonicalisation, weight normalisation, Cartesian index
% ======================================================================

function [groupOfAttr, attrsOfGroup, G] = localCanonicalizeGroups(groupsIn, A)
    if isempty(groupsIn)
        groupOfAttr = 1:A;
    elseif iscell(groupsIn)
        % Cell array of attribute-index lists, one per group
        G_in = numel(groupsIn);
        groupOfAttr = zeros(1, A);
        for g = 1:G_in
            idx = groupsIn{g};
            idx = idx(:).';
            for a = idx
                if a < 1 || a > A
                    error('buildExpTens:badGroupIdx', ...
                          'Group %d references attribute %d, out of range [1, %d].', g, a, A);
                end
                if groupOfAttr(a) ~= 0
                    error('buildExpTens:duplicateGroupAttr', ...
                          'Attribute %d is listed in more than one group.', a);
                end
                groupOfAttr(a) = g;
            end
        end
        if any(groupOfAttr == 0)
            missing = find(groupOfAttr == 0);
            error('buildExpTens:unassignedAttrs', ...
                  'Attributes [%s] are not assigned to any group.', num2str(missing));
        end
    elseif isnumeric(groupsIn)
        groupOfAttr = double(groupsIn(:).');
        if numel(groupOfAttr) ~= A
            error('buildExpTens:groupsLength', ...
                  'Group-index vector must have length equal to the number of attributes (%d).', A);
        end
        if any(rem(groupOfAttr, 1) ~= 0) || any(groupOfAttr < 1)
            error('buildExpTens:badGroupIndices', ...
                  'Group indices must be positive integers.');
        end
        % Check that indices are contiguous 1:G (no gaps)
        uniq = unique(groupOfAttr);
        if ~isequal(uniq, 1:numel(uniq))
            error('buildExpTens:nonContiguousGroups', ...
                  ['Group indices must be contiguous integers 1:G with ' ...
                   'no gaps. Got unique values: %s.'], mat2str(uniq));
        end
    else
        error('buildExpTens:badGroupsType', ...
              'groups must be [], a numeric vector of indices, or a cell array of attribute lists.');
    end

    G = max(groupOfAttr);
    attrsOfGroup = cell(1, G);
    for a = 1:A
        g = groupOfAttr(a);
        attrsOfGroup{g} = [attrsOfGroup{g}, a];
    end
end


function wCell = localNormaliseWeights(wIn, A, Ka, N)
    % Top-level normalisation: [] / scalar / cell of per-attribute inputs.
    if isempty(wIn) && ~iscell(wIn)
        wIn = 1;  % unified downstream: scalar one, then broadcast
    end

    if isnumeric(wIn) && isscalar(wIn)
        if wIn == 0
            warning('buildExpTens:zeroWeights', 'All weights are zero.');
        end
        wCell = cell(1, A);
        for a = 1:A
            wCell{a} = wIn * ones(Ka(a), N);
        end
        return;
    end

    if iscell(wIn)
        if numel(wIn) ~= A
            error('buildExpTens:weightCellLength', ...
                  'Weight cell array must have length equal to the number of attributes (%d).', A);
        end
        wCell = cell(1, A);
        for a = 1:A
            wCell{a} = localBroadcastWeight(wIn{a}, Ka(a), N, a);
        end
        return;
    end

    error('buildExpTens:badWeightsType', ...
          ['Top-level weight argument must be [], a scalar, or a cell ' ...
           'array of per-attribute inputs.']);
end


function Wab = localBroadcastWeight(w, Ka, N, attrIdx)
    % Broadcast a per-attribute weight input to a full Ka x N matrix.
    if isempty(w)
        Wab = ones(Ka, N);
        return;
    end
    if ~isnumeric(w)
        error('buildExpTens:badPerAttrWeightType', ...
              'Attribute %d weight input must be numeric.', attrIdx);
    end

    w = double(w);

    if isscalar(w)
        Wab = w * ones(Ka, N);
        return;
    end

    sz = size(w);
    % Row vector 1 x N -> broadcast across slots
    if sz(1) == 1 && sz(2) == N
        Wab = repmat(w, Ka, 1);
        return;
    end
    % Column vector Ka x 1 -> broadcast across events
    if sz(1) == Ka && sz(2) == 1
        Wab = repmat(w, 1, N);
        return;
    end
    % Full matrix Ka x N -> as-is
    if sz(1) == Ka && sz(2) == N
        Wab = w;
        return;
    end

    error('buildExpTens:badPerAttrWeightShape', ...
          ['Attribute %d weight input has shape [%d %d]; expected [], ' ...
           'scalar, [1 %d], [%d 1], or [%d %d].'], ...
          attrIdx, sz(1), sz(2), N, Ka, Ka, N);
end


function idxCell = localCartesianIndices(sizes)
    %LOCALCARTESIANINDICES  Column-major Cartesian-product indices.
    %   For a Cartesian product of axes with sizes sizes(1), ..., sizes(A),
    %   returns a 1 x A cell, each cell a 1 x prod(sizes) row vector giving
    %   the index along the corresponding axis. First axis varies fastest
    %   (MATLAB column-major ndgrid convention).
    A = numel(sizes);
    idxCell = cell(1, A);
    for a = 1:A
        repInner = prod(sizes(1:a - 1));   % consecutive repeats of each value
        repOuter = prod(sizes(a + 1:end));  % tiles of the full cycle
        row = 1:sizes(a);
        if repInner > 1
            row = kron(row, ones(1, repInner));
        end
        if repOuter > 1
            row = repmat(row, 1, repOuter);
        end
        idxCell{a} = row;
    end
end

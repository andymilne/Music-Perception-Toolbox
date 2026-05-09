function H = entropyExpTens(varargin)
%ENTROPYEXPTENS Shannon entropy of an expectation tensor density.
%
%   H = ENTROPYEXPTENS(p, w, sigma, r, isRel, isPer, period) returns the
%   Shannon entropy of the single-attribute expectation tensor defined
%   by the weighted multiset (p, w), where p represents pitches or
%   positions. The tensor is built, evaluated on a regular grid, and
%   the Shannon entropy of the resulting probability mass function
%   returned. By default the result is normalized to [0, 1] by
%   dividing by log_base(N), where N is the total number of grid
%   points.
%
%   H = ENTROPYEXPTENS(pAttr, w, sigmaVec, rVec, groups, isRelVec, ...
%                      isPerVec, periodVec) returns the Shannon
%   entropy of the multi-attribute expectation tensor (MAET) defined
%   by the per-attribute matrices pAttr and the per-group parameters.
%   The density is evaluated on the Cartesian product of one 1-D grid
%   per effective dimension (one per non-isRel tuple slot for each
%   attribute), using each attribute's group-domain.
%
%   H = ENTROPYEXPTENS(T) returns the Shannon entropy of a pre-built
%   density struct T (as returned by buildExpTens). Dispatches on the
%   tag field: 'ExpTensDensity' -> SA path, 'MaetDensity' -> MA path.
%   When a struct is passed, no further positional arguments are
%   required.
%
%   HCell = ENTROPYEXPTENS({T_1, ..., T_n}) returns a 1-by-n cell of
%   entropy values, one per density struct. List mode (v2.1+); Option
%   II shape rule applies (length-1 input returns length-1 cell). No
%   further positional arguments may be passed.
%
%   H = ENTROPYEXPTENS(P, W, sigma, r, isRel, isPer, period) where P
%   is an nRows-by-K matrix (rows = multisets) returns an nRows-by-1
%   vector of entropy values (batched-raw mode, v2.1+). Detection is
%   by P having both dimensions > 1; rows with fewer than r valid
%   pitches return NaN.
%
%   H = ENTROPYEXPTENS(..., Name, Value) specifies additional options
%   using one or more name-value arguments.
%
%   The differential entropy of a Gaussian mixture has no closed-form
%   analytic solution. This function therefore discretizes the tensor
%   over a fine grid and computes the Shannon entropy of the resulting
%   probability mass function. Accuracy depends on the ratio of sigma
%   to the grid spacing.
%
%   For periodic groups (isPer = true), the domain is [0, period).
%   For non-periodic groups, the user must specify bounds via xMin
%   and xMax. These should be wide enough to capture the full support
%   of the distribution (e.g., at least 3*sigma beyond the outermost
%   values).
%
%   The convention 0 * log(0) = 0 is applied.
%
%   Inputs (SA path)
%       p       - Pitch or position values (vector); or a struct as
%                 returned by buildExpTens (in which case the
%                 subsequent positional arguments are not required).
%       w       - Weights (vector, same length as p).
%       sigma   - Gaussian bandwidth.
%       r       - Tuple size (positive integer; r >= 2 if isRel == true).
%       isRel   - Logical: true for relative (transposition-invariant).
%       isPer   - Logical: true for periodic domain.
%       period  - Period of the domain.
%
%   Inputs (MA path)
%       pAttr     - 1 x A cell array of K_a x N matrices.
%       w         - Weights. []/scalar/1 x A cell; see buildExpTens.
%       sigmaVec  - 1 x G per-group Gaussian widths.
%       rVec      - 1 x A per-attribute tuple sizes.
%       groups    - Group assignment ([], index vector, or cell of
%                   attribute-index lists); see buildExpTens.
%       isRelVec  - 1 x G per-group relative flags.
%       isPerVec  - 1 x G per-group periodic flags.
%       periodVec - 1 x G per-group periods.
%
%   Name-Value Arguments
%       'spectrum'      - (SA only.) Cell array of arguments passed to
%                         addSpectra. If provided, partials are added
%                         to the multiset before building the tensor.
%                         For MA, apply addSpectra to the pitch
%                         attribute before calling.
%       'normalize'     - Logical (default: true). Divide by log_base(N)
%                         to give a value in [0, 1].
%       'base'          - Logarithm base (default: 2). When normalize
%                         is true, the base cancels and has no effect.
%       'nPointsPerDim' - Grid resolution per effective dimension
%                         (default: 1200).
%       'xMin'          - SA: scalar. MA: scalar (broadcast to all
%                         non-periodic groups) or length-G vector (one
%                         entry per group; periodic-group entries are
%                         ignored). Default: NaN.
%       'xMax'          - As xMin. Default: NaN.
%       'gridLimit'     - Hard ceiling on total grid size
%                         (nPointsPerDim ^ dim) before allocation.
%                         Applies to MA always, and to SA whenever the
%                         density's effective dimension dim > 1 (e.g.
%                         r = 2 with isRel = false). Default: 1e8.
%                         Errors with a suggested reduction if exceeded.
%
%   Examples
%       H = entropyExpTens(0:11, ones(1,12), 100, 1, false, true, 12);
%
%       T = buildExpTens([0 4 7], ones(1,3), 10, 1, false, true, 12);
%       H = entropyExpTens(T);
%
%       % MA: pitch + time
%       pitch = [0 12; 4 15; 7 19];  time = [0 1];
%       H = entropyExpTens({pitch, time}, [], ...
%                          [20, 0.1], [2, 1], [], ...
%                          [true, false], [true, false], [1200, 0], ...
%                          'xMin', -0.5, 'xMax', 1.5, ...
%                          'nPointsPerDim', 80);
%
%   See also BUILDEXPTENS, EVALEXPTENS, COSSIMEXPTENS.

nvDefaults = struct( ...
    'spectrum',      {{}}, ...
    'normalize',     true, ...
    'base',          2, ...
    'nPointsPerDim', 1200, ...
    'xMin',          NaN, ...
    'xMax',          NaN, ...
    'gridLimit',     1e8, ...
    'verbose',       true);

[posArgs, nvArgs] = localParseNVPairs(varargin, nvDefaults);
nPos = numel(posArgs);

if nPos < 1
    error('entropyExpTens:noArgs', ...
          'At least one positional argument is required.');
end

firstArg = posArgs{1};

% --- Dispatch ---

% 0. LIST mode (v2.1+): first arg is a cell of density structs.
%    Returns a 1-by-n cell of per-density entropy values (Option II
%    shape rule). Does NOT match the MA-raw cell-of-arrays form below
%    (disambiguated by element type: structs vs numeric arrays).
if iscell(firstArg) && ~isempty(firstArg) && isstruct(firstArg{1})
    if nPos > 1
        error('entropyExpTens:listExtraArgs', ...
              ['When a cell of density structs is passed, no further ' ...
               'positional arguments may be provided.']);
    end
    H = localEntropyDensityList(firstArg, nvArgs);
    return;
end

% 0b. BATCHED-RAW mode (v2.1+): first arg is a 2-D numeric matrix
%     (rows = multisets) and total positional count is 7.
%     Returns an nRows-by-1 vector of entropy values.
if isnumeric(firstArg) && size(firstArg, 1) > 1 && size(firstArg, 2) > 1 ...
        && nPos == 7
    H = localEntropyBatchedRaw(posArgs, nvArgs);
    return;
end

% 1. Precomputed struct (tag-based).
if isstruct(firstArg) && isfield(firstArg, 'tag')
    if nPos > 1
        error('entropyExpTens:extraArgs', ...
              ['When a precomputed density struct is passed, no ' ...
               'further positional arguments may be provided.']);
    end
    switch firstArg.tag
        case 'ExpTensDensity'
            H = localEntropySA(firstArg, nvArgs);
            return;
        case 'MaetDensity'
            H = localEntropyMA(firstArg, nvArgs);
            return;
        case 'WindowedMaetDensity'
            H = localEntropyMA(firstArg, nvArgs);
            return;
        otherwise
            error('entropyExpTens:unknownTag', ...
                  'Unknown density struct tag: %s.', firstArg.tag);
    end
end

% 2. MA raw args (first arg is a cell).
if iscell(firstArg)
    if nPos ~= 8
        error('entropyExpTens:wrongArgCountMA', ...
              ['Multi-attribute raw call expects 8 positional arguments ' ...
               '(pAttr, w, sigmaVec, rVec, groups, isRelVec, isPerVec, ' ...
               'periodVec); got %d.'], nPos);
    end
    pAttr     = posArgs{1};
    w         = posArgs{2};
    sigmaVec  = posArgs{3};
    rVec      = posArgs{4};
    groups    = posArgs{5};
    isRelVec  = posArgs{6};
    isPerVec  = posArgs{7};
    periodVec = posArgs{8};
    dens = buildExpTens(pAttr, w, sigmaVec, rVec, groups, ...
                        isRelVec, isPerVec, periodVec, 'verbose', false);
    H = localEntropyMA(dens, nvArgs);
    return;
end

% 3. SA raw args.
if nPos ~= 7
    error('entropyExpTens:wrongArgCountSA', ...
          ['Single-attribute raw call expects 7 positional arguments ' ...
           '(p, w, sigma, r, isRel, isPer, period); got %d.'], nPos);
end
p      = posArgs{1};
w      = posArgs{2};
sigma  = posArgs{3};
r      = posArgs{4};
isRel  = posArgs{5};
isPer  = posArgs{6};
period = posArgs{7};

% Apply spectral enrichment if requested.
if ~isempty(nvArgs.spectrum)
    if ~iscell(nvArgs.spectrum)
        error('entropyExpTens:badSpectrum', ...
              '''spectrum'' value must be a cell array of addSpectra arguments.');
    end
    [p, w] = addSpectra(p, w, nvArgs.spectrum{:});
end

T = buildExpTens(p, w, sigma, r, isRel, isPer, period, 'verbose', false);
H = localEntropySA(T, nvArgs);

end


% =========================================================================
%  localEntropySA — single-attribute Shannon entropy (v2.0.0 body)
% =========================================================================

function H = localEntropySA(T, nvArgs)

    isPer  = T.isPer;
    period = T.period;
    dim    = T.dim;

    % Validate bounds for non-periodic case.
    if ~isPer
        if isnan(nvArgs.xMin) || isnan(nvArgs.xMax)
            error('entropyExpTens:missingBounds', ...
                  'xMin and xMax must be specified when isPer = false.');
        end
        if nvArgs.xMin >= nvArgs.xMax
            error('entropyExpTens:invalidBounds', ...
                  'xMin must be less than xMax.');
        end
    end

    % Construct per-dimension query points.
    if isPer
        x1 = linspace(0, period, nvArgs.nPointsPerDim + 1);
        x1 = x1(1:end-1);
    else
        x1 = linspace(nvArgs.xMin, nvArgs.xMax, nvArgs.nPointsPerDim);
    end

    % Build query matrix. For dim = 1, X is a 1 x nQ row vector. For
    % dim > 1, take the Cartesian product of dim copies of x1, giving a
    % dim x (nPointsPerDim^dim) matrix where each column is one point.
    if dim == 1
        X = x1;
    else
        % Hard ceiling on grid size before allocation.
        gridLimit = nvArgs.gridLimit;
        gridSize = nvArgs.nPointsPerDim ^ dim;
        if gridSize > gridLimit
            error('entropyExpTens:gridLimitExceeded', ...
                  ['SA Cartesian grid (%g points = nPointsPerDim^dim = %d^%d) ' ...
                   'exceeds gridLimit (%g). Reduce nPointsPerDim or raise ' ...
                   '''gridLimit''.'], gridSize, nvArgs.nPointsPerDim, dim, gridLimit);
        end
        % Cartesian product via ndgrid. We build dim grid arrays then
        % reshape each to a row, stacking into a dim x nQ matrix.
        gridArgs = repmat({x1}, 1, dim);
        gridCells = cell(1, dim);
        [gridCells{:}] = ndgrid(gridArgs{:});
        X = zeros(dim, gridSize);
        for d = 1:dim
            X(d, :) = gridCells{d}(:).';
        end
    end

    % Evaluate tensor.
    t = evalExpTens(T, X, 'verbose', false);

    % Normalize to pmf.
    q = t(:) / sum(t(:));
    N = numel(q);
    q(q == 0) = [];

    H = -sum(q .* (log(q) / log(nvArgs.base)));
    if nvArgs.normalize
        H = H / (log(N) / log(nvArgs.base));
    end
end


% =========================================================================
%  localEntropyMA — multi-attribute Shannon entropy
% =========================================================================

function H = localEntropyMA(dens, nvArgs)
%LOCALENTROPYMA  Shannon entropy of a MaetDensity or WindowedMaetDensity.
%
%   Builds a Cartesian-product grid with one 1-D linspace per effective
%   dimension of the density's domain (one per non-isRel tuple slot for
%   each attribute, each on its group's domain), evaluates the density
%   at every grid point via evalExpTens, normalises to a pmf, and
%   returns Shannon entropy.
%
%   Accepts either a MaetDensity or a WindowedMaetDensity. Structural
%   fields are read from the underlying density; evaluation itself
%   calls evalExpTens on the input object, so window application (if
%   present) is handled automatically.

    % Structural fields come from the underlying MaetDensity.
    if isfield(dens, 'tag') && strcmp(dens.tag, 'WindowedMaetDensity')
        base_dens = dens.dens;
    else
        base_dens = dens;
    end
    A         = base_dens.nAttrs;
    G         = base_dens.nGroups;
    groupOf   = base_dens.groupOfAttr;
    dimPer    = base_dens.dimPerAttr;
    dim       = base_dens.dim;
    isPerG    = logical(base_dens.isPer);
    periodG   = base_dens.period;

    if dim == 0
        % Degenerate: all attributes isRel with r = 1. Density is
        % constant and entropy is 0.
        H = 0;
        return;
    end

    % --- Resolve xMin/xMax to per-group vectors ---
    xMinG = localBroadcastBounds(nvArgs.xMin, G, 'xMin');
    xMaxG = localBroadcastBounds(nvArgs.xMax, G, 'xMax');

    % --- Check non-periodic groups have valid bounds ---
    needsBounds = find(~isPerG);
    for idx = 1:numel(needsBounds)
        g = needsBounds(idx);
        if isnan(xMinG(g)) || isnan(xMaxG(g))
            error('entropyExpTens:missingBounds', ...
                  'xMin and xMax must be specified for non-periodic group %d.', g);
        end
        if xMinG(g) >= xMaxG(g)
            error('entropyExpTens:invalidBounds', ...
                  'xMin must be less than xMax (group %d).', g);
        end
    end

    % --- Grid-size guard ---
    totalPoints = double(nvArgs.nPointsPerDim) ^ double(dim);
    if totalPoints > nvArgs.gridLimit
        suggested = floor(nvArgs.gridLimit ^ (1 / double(dim)));
        error('entropyExpTens:gridLimitExceeded', ...
              ['Grid size %d^%d = %.3g exceeds gridLimit = %.3g. ' ...
               'Reduce nPointsPerDim to %d or lower, or raise ' ...
               'gridLimit.'], ...
              nvArgs.nPointsPerDim, dim, totalPoints, ...
              nvArgs.gridLimit, suggested);
    end

    % --- Build one 1-D axis per effective dimension ---
    % Each effective dimension belongs to an attribute, which belongs
    % to a group. Each 1-D axis uses that group's domain.
    axes1D = cell(1, dim);
    k = 0;
    for a = 1:A
        da = dimPer(a);
        g  = groupOf(a);
        if isPerG(g)
            P = periodG(g);
            ax = linspace(0, P, nvArgs.nPointsPerDim + 1);
            ax = ax(1:end-1);
        else
            ax = linspace(xMinG(g), xMaxG(g), nvArgs.nPointsPerDim);
        end
        for j = 1:da
            k = k + 1;
            axes1D{k} = ax;
        end
    end

    % --- Cartesian product as (dim x totalPoints) query matrix ---
    % Use ndgrid so the first axis varies fastest (column-major).
    meshCells = cell(1, dim);
    [meshCells{:}] = ndgrid(axes1D{:});
    X = zeros(dim, round(totalPoints));
    for d = 1:dim
        Md = meshCells{d};
        X(d, :) = Md(:).';
    end

    % --- Evaluate density ---
    t = evalExpTens(dens, X, 'verbose', false);

    % --- Shannon entropy ---
    totalMass = sum(t(:));
    if totalMass == 0
        H = 0;
        return;
    end
    q = t(:) / totalMass;
    N = numel(q);
    q(q == 0) = [];

    H = -sum(q .* (log(q) / log(nvArgs.base)));
    if nvArgs.normalize
        H = H / (log(N) / log(nvArgs.base));
    end
end


% =========================================================================
%  Helpers
% =========================================================================

function out = localBroadcastBounds(v, G, name)
% Coerce xMin or xMax input to a length-G vector.
%   - scalar    -> broadcast to all groups
%   - length-G  -> pass through
    v = double(v);
    if isscalar(v)
        out = repmat(v, 1, G);
        return;
    end
    if isvector(v) && numel(v) == G
        out = v(:).';
        return;
    end
    error('entropyExpTens:badBoundsShape', ...
          '%s must be a scalar or a length-%d vector (one per group); got size [%s].', ...
          name, G, num2str(size(v)));
end


function [posArgs, nvArgs] = localParseNVPairs(args, defaults)
% Split varargin into positional and name-value portions, using the
% fields of `defaults` as the set of recognised NV keys.
    nvNames = fieldnames(defaults);
    nvArgs = defaults;
    posArgs = {};
    i = 1;
    N = numel(args);
    while i <= N
        if (ischar(args{i}) || (isstring(args{i}) && isscalar(args{i}))) ...
                && i + 1 <= N && any(strcmpi(char(args{i}), nvNames))
            key = char(args{i});
            keyCan = nvNames{find(strcmpi(key, nvNames), 1)};
            nvArgs.(keyCan) = args{i + 1};
            i = i + 2;
        else
            posArgs{end + 1} = args{i}; %#ok<AGROW>
            i = i + 1;
        end
    end
end


% =====================================================================
%  v2.1 unified dispatch helpers: density-list and batched-raw modes.
% =====================================================================

function HCell = localEntropyDensityList(densCell, nvArgs)
%LOCALENTROPYDENSITYLIST Per-density entropy for a list of densities.
%
%   Returns a 1-by-n cell of entropy values. Each element is computed
%   by recursive call to entropyExpTens. Option II shape rule: a
%   length-1 input returns a length-1 cell.

    n = numel(densCell);
    HCell = cell(1, n);

    % Re-pack the name-value defaults so we can pass them through.
    nvPairs = localPackNVPairs(nvArgs);

    for i = 1:n
        if ~isstruct(densCell{i})
            error('MPT:EntropyList:NonStruct', ...
                ['entropyExpTens (list mode): cell entries must be density ' ...
                 'structs from buildExpTens; entry %d is not a struct.'], i);
        end
        HCell{i} = entropyExpTens(densCell{i}, nvPairs{:});
    end
end


function H = localEntropyBatchedRaw(posArgs, nvArgs)
%LOCALENTROPYBATCHEDRAW Per-row entropy from a 2-D pitch matrix.
%
%   posArgs follows the SA-raw convention: {P, W, sigma, r, isRel,
%   isPer, period} with P an nRows-by-K matrix. Returns an nRows-by-1
%   vector of entropy values; rows with fewer than r valid pitches
%   are NaN.

    P      = posArgs{1};
    W      = posArgs{2};
    sigma  = posArgs{3};
    r      = posArgs{4};
    isRel  = posArgs{5};
    isPer  = posArgs{6};
    period = posArgs{7};

    nRows = size(P, 1);
    H = nan(nRows, 1);

    haveRowWeights = ~isempty(W) && isequal(size(W), size(P));
    if ~isempty(W) && ~haveRowWeights
        if isvector(W) && numel(W) == size(P, 2)
            W_broadcast = W(:).';
        else
            error('MPT:EntropyBatched:WeightShape', ...
                ['entropyExpTens (batched mode): W must be empty, a matrix the ' ...
                 'same size as P, or a vector matching the number of pitch columns.']);
        end
    end

    % Force inner scalar calls silent; one batched estimate at top.
    nvArgsInner = nvArgs;
    nvArgsInner.verbose = false;
    nvPairs = localPackNVPairs(nvArgsInner);

    % Up-front time estimate (printed once). Empirical calibration via
    % a uniformly-sampled subset of K rows, with one warm-up call to
    % absorb first-call overhead.
    if isfield(nvArgs, 'verbose') && nvArgs.verbose && nRows > 1
        nCal = min(10, nRows);
        sampleIdx = unique(round(linspace(1, nRows, nCal)));

        % Warm-up
        warmupDone = false;
        for s = 1:numel(sampleIdx)
            sIdx = sampleIdx(s);
            pRowS = P(sIdx, :);
            validS = ~isnan(pRowS);
            pValidS = pRowS(validS);
            if numel(pValidS) < r
                continue;
            end
            if haveRowWeights
                wValidS = W(sIdx, validS);
            elseif ~isempty(W)
                wValidS = W_broadcast(validS);
            else
                wValidS = [];
            end
            entropyExpTens(pValidS, wValidS, sigma, r, isRel, isPer, period, nvPairs{:});
            warmupDone = true;
            break;
        end

        if warmupDone
            tCalStart = tic;
            nValidCal = 0;
            for s = 1:numel(sampleIdx)
                sIdx = sampleIdx(s);
                pRowS = P(sIdx, :);
                validS = ~isnan(pRowS);
                pValidS = pRowS(validS);
                if numel(pValidS) < r
                    continue;
                end
                if haveRowWeights
                    wValidS = W(sIdx, validS);
                elseif ~isempty(W)
                    wValidS = W_broadcast(validS);
                else
                    wValidS = [];
                end
                entropyExpTens(pValidS, wValidS, sigma, r, isRel, isPer, period, nvPairs{:});
                nValidCal = nValidCal + 1;
            end
            if nValidCal > 0
                tCalTotal = toc(tCalStart);
                tPerRow   = tCalTotal / nValidCal;
                estTotal  = tCalTotal + tPerRow * nRows;
                printBatchedEstimate('entropyExpTens', nRows, estTotal);
            end
        end
    end

    for k = 1:nRows
        pRow = P(k, :);
        validMask = ~isnan(pRow);
        pK = pRow(validMask);
        if haveRowWeights
            wK = W(k, validMask);
        elseif ~isempty(W)
            wK = W_broadcast(validMask);
        else
            wK = [];
        end
        if numel(pK) < r
            H(k) = NaN;
            continue;
        end
        H(k) = entropyExpTens(pK, wK, sigma, r, isRel, isPer, period, ...
            nvPairs{:});
    end
end


function nvPairs = localPackNVPairs(nvArgs)
%LOCALPACKNVPAIRS Re-pack a name-value struct into a flat name-value cell.
%
%   Used by the LIST and BATCHED-RAW dispatch helpers to forward the
%   name-value arguments to recursive entropyExpTens calls.

    nvPairs = {};
    fns = fieldnames(nvArgs);
    for i = 1:numel(fns)
        nvPairs = [nvPairs, {fns{i}, nvArgs.(fns{i})}]; %#ok<AGROW>
    end
end

function H = entropyExpTens(varargin)
%ENTROPYEXPTENS Entropy of an expectation tensor density.
%
%   H = ENTROPYEXPTENS(...) returns the entropy of a single- or
%   multi-attribute expectation tensor density. Two variants are
%   supported via the 'method' name-value argument:
%
%     'shannon' (default) --- discretized Shannon entropy of the
%       density on a Cartesian-product grid (one 1-D linspace per
%       effective dimension, on each group's domain). The differential
%       entropy of a Gaussian mixture has no closed-form analytic
%       solution, so the density is evaluated on the grid, normalized
%       to a probability mass function, and -sum(q*log_b(q)) returned.
%       With 'normalize', true (default), the result is divided by
%       log_b(N), where N is the total number of grid points; this
%       gives a value in [0, 1] independent of grid resolution.
%       Accuracy depends on the ratio of sigma to the grid spacing.
%       The convention 0 * log(0) = 0 is applied.
%
%     'renyi2' --- analytical Rényi-2 (collision) entropy
%       H_2 = -log_b(<T,T> / Z^2), computed in closed form via the
%       orbit-Möbius inner product (<T,T>) and the closed-form total
%       mass (Z). Grid-free; works at arbitrary tensor order r where
%       the Shannon-path Cartesian grid would exhaust memory.
%       Currently restricted to single-density input (scalar density
%       struct or raw scalar SA/MA); list, batched, and windowed
%       input forms are not yet implemented and produce informative
%       errors. 'normalize', true is not supported with 'renyi2' ---
%       the continuous Rényi-2 entropy ranges over (-Inf, log_b V]
%       rather than Shannon's [0, log_b N], so a uniform normaliser
%       does not yield a [0, 1] value; pass 'normalize', false to use
%       this method.
%
%   Both methods accept the input forms below. Forms marked
%   "Shannon-only" raise an informative error under method='renyi2'.
%
%   Input forms (both methods):
%
%     H = ENTROPYEXPTENS(p, w, sigma, r, isRel, isPer, period)
%       Single-attribute raw form. Builds the density from the
%       weighted multiset (p, w), where p represents pitches or
%       positions.
%
%     H = ENTROPYEXPTENS(pAttr, w, sigmaVec, rVec, groups, ...
%                        isRelVec, isPerVec, periodVec)
%       Multi-attribute raw form. pAttr is a cell of per-attribute
%       matrices; per-group parameters as in buildExpTens.
%
%     H = ENTROPYEXPTENS(T)
%       Pre-built density form. T is a struct as returned by
%       buildExpTens. Dispatches on its tag: 'ExpTensDensity' -> SA,
%       'MaetDensity' -> MA, 'WindowedMaetDensity' -> MA (Shannon
%       only). When a struct is passed, no further positional
%       arguments are required.
%
%   Input forms (Shannon-only):
%
%     HCell = ENTROPYEXPTENS({T_1, ..., T_n})
%       List form. Cell of density structs; returns a 1-by-n cell of
%       per-density entropy values. Option II shape rule applies
%       (length-1 input returns length-1 cell).
%
%     H = ENTROPYEXPTENS(P, W, sigma, r, isRel, isPer, period)
%       Batched-raw form. P is an nRows-by-K matrix (rows = multisets);
%       returns an nRows-by-1 column vector. Detection is by P having
%       both dimensions > 1; rows with fewer than r valid pitches
%       return NaN.
%
%   H = ENTROPYEXPTENS(..., Name, Value) specifies additional options
%   using one or more name-value arguments.
%
%   For periodic groups (isPer = true), the Shannon grid spans
%   [0, period). For non-periodic groups, bounds must be specified
%   via xMin and xMax, wide enough to capture the full support of the
%   distribution (e.g., at least 3*sigma beyond the outermost values).
%   Rényi-2 is grid-free and ignores xMin, xMax, and nPointsPerDim.
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
%       'method'        - 'shannon' (default) or 'renyi2'. See above.
%       'spectrum'      - (SA only.) Cell array of arguments passed to
%                         addSpectra. If provided, partials are added
%                         to the multiset before building the tensor.
%                         For MA, apply addSpectra to the pitch
%                         attribute before calling.
%       'normalize'     - Logical (default: true). Shannon only:
%                         divide by log_b(N) to give a value in
%                         [0, 1]. method='renyi2' with normalize=true
%                         errors.
%       'base'          - Logarithm base (default: 2). For Shannon
%                         with normalize=true, the base cancels and
%                         has no effect on the result.
%       'nPointsPerDim' - Shannon only: grid resolution per effective
%                         dimension (default: 1200).
%       'xMin'          - Shannon, non-periodic only. SA: scalar.
%                         MA: scalar (broadcast to all non-periodic
%                         groups) or length-G vector (one entry per
%                         group; periodic-group entries are ignored).
%                         Default: NaN.
%       'xMax'          - As xMin. Default: NaN.
%       'gridLimit'     - Shannon only: hard ceiling on total grid
%                         size (nPointsPerDim ^ dim) before allocation.
%                         Applies to MA always, and to SA whenever the
%                         density's effective dimension dim > 1 (e.g.
%                         r = 2 with isRel = false). Default: 1e8.
%                         Errors with a suggested reduction if exceeded.
%       'truncationSigmas' - Numeric scalar or []. Override the
%                         toolbox-wide mptDefaults('truncationSigmas')
%                         setting for this call. Passes through to the
%                         kernel evaluator on the centres path (Shannon
%                         only); skips Gaussian contributions whose
%                         centre-to-query distance exceeds k*sigma
%                         (kernel floor exp(-k^2/2)). [] (default)
%                         means use the global default (factory: Inf).
%       'kernelPrecision' - 'double' (default via mptDefaults), 'single',
%                         or [] for the global default. Override the
%                         toolbox-wide kernelPrecision setting for this
%                         call. Passes through to the kernel evaluator
%                         on the centres path (Shannon only); 'single'
%                         casts the kernel matrix to float32 for a ~2x
%                         speedup at ~7 sig fig precision.
%       'verbose'       - Logical (default: true). If false, suppresses
%                         console output (time estimates, progress
%                         messages).
%
%   Examples
%       % Shannon entropy of a 12-EDO chromatic scale (periodic, SA)
%       H = entropyExpTens(0:11, ones(1,12), 100, 1, false, true, 12);
%
%       % Same chord via pre-built density (Shannon)
%       T = buildExpTens([0 4 7], ones(1,3), 10, 1, false, true, 12);
%       H = entropyExpTens(T);
%
%       % Rényi-2 of the same chord --- closed-form, no grid
%       H = entropyExpTens(T, 'method', 'renyi2', 'normalize', false);
%
%       % MA: pitch + time, Shannon
%       pitch = [0 12; 4 15; 7 19];  time = [0 1];
%       H = entropyExpTens({pitch, time}, [], ...
%                          [20, 0.1], [2, 1], [], ...
%                          [true, false], [true, false], [1200, 0], ...
%                          'xMin', -0.5, 'xMax', 1.5, ...
%                          'nPointsPerDim', 80);
%
%   See also BUILDEXPTENS, EVALEXPTENS, COSSIMEXPTENS.

% Top-level call guard: see internal.dispatchScope.
guard = internal.dispatchScope(); %#ok<NASGU>

nvDefaults = struct( ...
    'spectrum',          {{}}, ...
    'method',            'shannon', ...
    'normalize',         true, ...
    'base',              2, ...
    'nPointsPerDim',     1200, ...
    'xMin',              NaN, ...
    'xMax',              NaN, ...
    'gridLimit',         1e8, ...
    'truncationSigmas',  [], ...
    'kernelPrecision',   [], ...
    'verbose',           true);

[posArgs, nvArgs] = localParseNVPairs(varargin, nvDefaults);
nPos = numel(posArgs);

if nPos < 1
    error('entropyExpTens:noArgs', ...
          'At least one positional argument is required.');
end

% Validate the method kwarg and reject the unimplementable combination
% renyi2 + normalize=true (the analytical Rényi-2 form has no natural
% [0, 1] reference).
if ~ismember(nvArgs.method, {'shannon', 'renyi2'})
    error('entropyExpTens:badMethod', ...
          '''method'' must be ''shannon'' or ''renyi2''; got ''%s''.', ...
          nvArgs.method);
end
if strcmp(nvArgs.method, 'renyi2') && nvArgs.normalize
    error('entropyExpTens:renyi2NormalizeNotSupported', ...
          ['method=''renyi2'' with normalize=true is not implemented. ' ...
           'The continuous Rényi-2 entropy ranges over (-Inf, log_b V] ' ...
           'rather than Shannon''s [0, log_b N], so a uniform ' ...
           'normaliser does not yield a [0, 1] value. Pass ' ...
           'normalize=false to use this method.']);
end

% Dispatch on method. Both methods do parallel per-input-form
% resolution; see localEntropyShannonDispatch and
% localEntropyRenyi2Dispatch for the per-form branching. Shannon
% supports the full input surface (single density, list of densities,
% raw scalar SA/MA, raw batched SA, windowed MA). Renyi-2 is
% restricted to single-density input — list, batched, and windowed
% forms are not yet implemented and produce informative errors.
if strcmp(nvArgs.method, 'shannon')
    H = localEntropyShannonDispatch(posArgs, nvArgs);
else
    H = localEntropyRenyi2Dispatch(posArgs, nvArgs);
end

end


% =========================================================================
%  localEntropyShannonDispatch — input-form resolution for Shannon entropy
% =========================================================================

function H = localEntropyShannonDispatch(posArgs, nvArgs)
%LOCALENTROPYSHANNONDISPATCH  Resolve input form and route to SA / MA helper.
%
%   Shannon entropy of the density evaluated on a Cartesian-product
%   grid; supports the full input surface (precomputed density struct,
%   list of densities, MA raw args, SA raw args, SA batched 2-D
%   matrix).

    nPos = numel(posArgs);
    firstArg = posArgs{1};

    % 0. LIST mode: first arg is a cell of density structs.
    %    Returns a 1-by-n cell of per-density entropy values (Option II
    %    shape rule). Does NOT match the MA-raw cell-of-arrays form
    %    below (disambiguated by element type: structs vs numeric arrays).
    if iscell(firstArg) && ~isempty(firstArg) && isstruct(firstArg{1})
        if nPos > 1
            error('entropyExpTens:listExtraArgs', ...
                  ['When a cell of density structs is passed, no further ' ...
                   'positional arguments may be provided.']);
        end
        H = localEntropyDensityList(firstArg, nvArgs);
        return;
    end

    % 0b. BATCHED-RAW mode: first arg is a 2-D numeric matrix
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
%  localEntropySA — single-attribute Shannon entropy
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

    % Evaluate tensor. Forward truncation/precision kwargs when set.
    evalKw = {'verbose', false};
    if isfield(nvArgs, 'truncationSigmas') && ~isempty(nvArgs.truncationSigmas)
        evalKw = [evalKw, {'truncationSigmas', nvArgs.truncationSigmas}];
    end
    if isfield(nvArgs, 'kernelPrecision') && ~isempty(nvArgs.kernelPrecision)
        evalKw = [evalKw, {'kernelPrecision', nvArgs.kernelPrecision}];
    end
    t = evalExpTens(T, X, evalKw{:});

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

    % --- Evaluate density. Forward truncation/precision kwargs. ---
    evalKw = {'verbose', false};
    if isfield(nvArgs, 'truncationSigmas') && ~isempty(nvArgs.truncationSigmas)
        evalKw = [evalKw, {'truncationSigmas', nvArgs.truncationSigmas}];
    end
    if isfield(nvArgs, 'kernelPrecision') && ~isempty(nvArgs.kernelPrecision)
        evalKw = [evalKw, {'kernelPrecision', nvArgs.kernelPrecision}];
    end
    t = evalExpTens(dens, X, evalKw{:});

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
%  Unified dispatch helpers: density-list and batched-raw modes.
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
            error('entropyExpTens:listNonStruct', ...
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
            error('entropyExpTens:batchedWeightShape', ...
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
    % Adaptive progress-print state. Defaults: silent.
    progStride = 1;
    showProgress = false;
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
                internal.printBatchedEstimate('entropyExpTens', nRows, estTotal);
                progStride = internal.progressStride(tPerRow);
                showProgress = estTotal >= 5;
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

        if isfield(nvArgs, 'verbose') && nvArgs.verbose ...
                && showProgress ...
                && (mod(k, progStride) == 0 || k == nRows)
            fprintf('  %d / %d rows computed.\n', k, nRows);
        end
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


% =========================================================================
%  Rényi-2 (collision) entropy via orbit-Möbius IP
% =========================================================================

function H = localEntropyRenyi2Dispatch(posArgs, nvArgs)
%LOCALENTROPYRENYI2DISPATCH  Resolve input form and route to SA / MA helper.
%
%   Analytical Rényi-2 (collision) entropy via the orbit-Möbius
%   inner-product machinery. Restricted to single-density input
%   (scalar density struct, raw scalar SA, or raw scalar MA). List
%   and batched input forms raise NotImplementedError-style errors.
%   Windowed MA is also not yet supported.

    nPos = numel(posArgs);
    firstArg = posArgs{1};

    % --- Reject unsupported input forms early ---
    if iscell(firstArg) && ~isempty(firstArg) && isstruct(firstArg{1})
        error('entropyExpTens:renyi2ListNotSupported', ...
            ['method=''renyi2'' does not yet support list input. ' ...
             'Apply it to each density individually.']);
    end
    if isnumeric(firstArg) && size(firstArg, 1) > 1 && size(firstArg, 2) > 1
        error('entropyExpTens:renyi2BatchedNotSupported', ...
            ['method=''renyi2'' does not yet support raw SA batched ' ...
             '(2-D) input. Pass each chord row individually, or ' ...
             'pre-build a density struct.']);
    end

    base = nvArgs.base;

    % --- Resolve input to a density struct ---
    if isstruct(firstArg) && isfield(firstArg, 'tag')
        if nPos > 1
            error('entropyExpTens:extraArgs', ...
                ['When a precomputed density struct is passed, no ' ...
                 'further positional arguments may be provided.']);
        end
        switch firstArg.tag
            case 'ExpTensDensity'
                H = localRenyi2SA(firstArg, base);
                return;
            case 'MaetDensity'
                H = localRenyi2MA(firstArg, base);
                return;
            case 'WindowedMaetDensity'
                error('entropyExpTens:renyi2WindowedNotSupported', ...
                    ['method=''renyi2'' is not yet implemented for ' ...
                     'WindowedMaetDensity. Use method=''shannon'' for ' ...
                     'windowed MA densities, or compute on the ' ...
                     'underlying MaetDensity.']);
            otherwise
                error('entropyExpTens:unknownTag', ...
                    'Unknown density struct tag: %s.', firstArg.tag);
        end
    end

    % --- MA raw args (first arg is a cell of arrays) ---
    if iscell(firstArg)
        if nPos ~= 8
            error('entropyExpTens:wrongArgCountMA', ...
                ['Multi-attribute raw call expects 8 positional ' ...
                 'arguments (pAttr, w, sigmaVec, rVec, groups, ' ...
                 'isRelVec, isPerVec, periodVec); got %d.'], nPos);
        end
        dens = buildExpTens(posArgs{1}, posArgs{2}, posArgs{3}, posArgs{4}, ...
                            posArgs{5}, posArgs{6}, posArgs{7}, posArgs{8}, ...
                            'verbose', false);
        H = localRenyi2MA(dens, base);
        return;
    end

    % --- SA raw args ---
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

    % buildExpTens is cheap in lazy mode; we only read cheap fields.
    dens = buildExpTens(p, w, sigma, r, isRel, isPer, period, 'verbose', false);
    H = localRenyi2SA(dens, base);
end


function H = localRenyi2SA(dens, base)
%LOCALRENYI2SA  Analytical Rényi-2 entropy of a SA expectation tensor.
%
%   Computes H_2 = -log_b(<T,T> / Z^2) where <T,T> is evaluated via
%   the orbit-Möbius inner product machinery (or a direct pairwise
%   formula at r=1 where the orbit table is undefined) and
%   Z = integral T(x) dx via the closed-form total-mass formulae in
%   the +mobius package.

    p = dens.p; w = dens.w;
    sigma = dens.sigma; r = dens.r;
    isRel = dens.isRel; isPer = dens.isPer; period = dens.period;

    % r=1 rel is degenerate: the relative density lives on a 0-D space
    % (one position has no internal relative structure); H_2 is
    % undefined as a continuous quantity. Return 0 by convention,
    % matching the MA path's empty-density short circuit.
    if r == 1 && isRel
        H = 0;
        return;
    end

    if r == 1
        % Direct r=1 abs path: T = sum_i w_i G_sigma(x - p_i), so
        %   <T,T> = sigma*sqrt(pi) * sum_{i,j} w_i w_j exp(-(p_i-p_j)^2/(4 sigma^2))
        % (with wrapped differences in periodic mode).
        internal.maybeShowDispatchMsg('entropyExpTens', 'pairwise', ...
            sprintf('renyi2, r=1 abs (direct pairwise sum)'), 0, false);
        p = p(:); w = w(:);
        diffs = p - p.';
        if isPer
            diffs = diffs - period * floor(diffs / period + 0.5);
        end
        K = exp(-(diffs.^2) / (4 * sigma^2));
        ip_xx = sigma * sqrt(pi) * sum(sum((w * w.') .* K));
        Z = mobius.totalMassAbs(p, w, sigma, r);
    else
        % r >= 2: orbit machinery. Empirical sweeps in the Python audit
        % corpus show the orbit self-IP is robust at every tested
        % musical sigma; the per-orbit-class cancellation ratio in abs
        % mode dips to ~0.13 in the worst tested case, well above the
        % 1e-10 corruption threshold. We rely on a post-hoc finite/
        % positive check rather than a ratio-based fallback. The
        % pairwise fallback explored earlier was abandoned: orbit and
        % pairwise use different normalisation conventions in rel mode,
        % so the fallback gave a different (also wrong) answer rather
        % than recovering the correct value.
        internal.maybeShowDispatchMsg('entropyExpTens', 'mobius', ...
            sprintf('renyi2, r=%d (orbit-Möbius IP)', r), 0, false);
        if isRel
            ip_xx = mobius.orbitInnerRelSA(p, w, p, w, sigma, r, isPer, period);
            Z = mobius.totalMassRel(p, w, sigma, r);
        else
            ip_xx = mobius.orbitInnerAbsSA(p, w, p, w, sigma, r, isPer, period);
            Z = mobius.totalMassAbs(p, w, sigma, r);
        end
    end

    if ~isfinite(ip_xx) || ip_xx <= 0
        error('entropyExpTens:renyi2NonPositiveIP', ...
            ['Computed <T,T>=%g via the orbit-Möbius path is non-positive ' ...
             'or non-finite. The input density may be degenerate (all ' ...
             'weights zero), or the parameters may lie in a regime where ' ...
             'the alternating Möbius sum has lost all significant digits. ' ...
             'Try a less extreme sigma/period ratio, smaller r, or ' ...
             'larger K-r margin.'], ip_xx);
    end
    if ~isfinite(Z) || Z <= 0
        error('entropyExpTens:renyi2NonPositiveZ', ...
            'Computed Z=%g is non-positive or non-finite.', Z);
    end

    H = -log(ip_xx / (Z * Z)) / log(base);
end


function H = localRenyi2MA(dens, base)
%LOCALRENYI2MA  Analytical Rényi-2 entropy of an MA expectation tensor.
%
%   Uses the per-attribute orbit IP factorisation
%       <T,T> = sum_{n,m} prod_a I_a[n,m]
%   with the per-attribute matrix coming from mobius.maPerAttrInnerMatrix
%   (the same machinery cosSimExpTens uses), and
%       Z = sum_n prod_a Z_a^{(n)}
%   where each Z_a^{(n)} is the closed-form SA total mass evaluated on
%   event n's attribute-a slot pitches and weights.
%
%   Windowed densities are not supported on this path.

    if strcmp(dens.tag, 'WindowedMaetDensity')
        error('entropyExpTens:renyi2WindowedNotSupported', ...
            ['method=''renyi2'' is not yet implemented for ' ...
             'WindowedMaetDensity.']);
    end

    A = dens.nAttrs;
    N = dens.N;
    if A == 0 || N == 0
        H = 0;
        return;
    end

    internal.maybeShowDispatchMsg('entropyExpTens', 'mobius', ...
        sprintf('renyi2 MA, A=%d (per-attribute orbit IP)', A), 0, false);

    % --- <T, T> via per-attribute orbit IP ---
    % Per-(n,m) cancellation ratios were shown empirically to fire
    % spuriously for self-IPs in typical musical regimes (off-diagonal
    % entries can be noisy while the diagonal entries — which dominate
    % the sum — are clean). We rely on a post-hoc finite/positive check
    % rather than a ratio fallback.
    P_xx = ones(N, N);
    for a = 1:A
        g = dens.groupOfAttr(a);
        r_a = dens.r(a);
        sigma_g = dens.sigma(g);
        isRel_g = dens.isRel(g);
        isPer_g = dens.isPer(g);
        period_g = dens.period(g);
        Pa = dens.pAttr{a};
        Wa = dens.w{a};
        I_xx = mobius.maPerAttrInnerMatrix(Pa, Wa, Pa, Wa, ...
            sigma_g, r_a, isRel_g, isPer_g, period_g);
        P_xx = P_xx .* I_xx;
    end
    ip_xx = sum(P_xx(:));

    if ~isfinite(ip_xx) || ip_xx <= 0
        error('entropyExpTens:renyi2NonPositiveIP', ...
            ['Computed <T,T>=%g via the orbit-Möbius path is non-positive ' ...
             'or non-finite. The input density may be degenerate, or the ' ...
             'parameters may lie in a regime where the per-attribute ' ...
             'alternating sum has lost all significant digits. Try a ' ...
             'less extreme sigma/period ratio, smaller r, or larger ' ...
             'K-r margin.'], ip_xx);
    end

    % --- Z = sum_n prod_a Z_a^{(n)} ---
    % Each per-event-per-attribute factor is the SA total mass computed
    % on that event's slot vector. NaN slots (ragged events) are dropped
    % before calling totalMass*; the periodic mode handles wrap inside
    % the helper.
    Z_per_event_attr = zeros(N, A);
    for a = 1:A
        g = dens.groupOfAttr(a);
        r_a = dens.r(a);
        sigma_g = dens.sigma(g);
        isRel_g = dens.isRel(g);
        Pa = dens.pAttr{a};   % (K_a, N)
        Wa = dens.w{a};       % (K_a, N)
        for n = 1:N
            pn = Pa(:, n);
            wn = Wa(:, n);
            valid = ~(isnan(pn) | isnan(wn));
            pn = pn(valid);
            wn = wn(valid);
            if isRel_g
                Z_an = mobius.totalMassRel(pn, wn, sigma_g, r_a);
            else
                Z_an = mobius.totalMassAbs(pn, wn, sigma_g, r_a);
            end
            Z_per_event_attr(n, a) = Z_an;
        end
    end
    Z = sum(prod(Z_per_event_attr, 2));

    if ~isfinite(Z) || Z <= 0
        error('entropyExpTens:renyi2NonPositiveZ', ...
            'Computed Z=%g is non-positive or non-finite.', Z);
    end

    H = -log(ip_xx / (Z * Z)) / log(base);
end

function H = entropyExpTens(varargin)
%ENTROPYEXPTENS Entropy of an expectation tensor density.
%
%   H = ENTROPYEXPTENS(...) returns the entropy of a single- or
%   multi-attribute expectation tensor density. Four variants are
%   supported via the 'method' name-value argument:
%
%     'shannon' (default) --- raw discrete Shannon entropy
%       H = -sum_k q_k log_b q_k of the density on an explicit
%       Cartesian-product grid (one 1-D linspace per effective
%       dimension, on each group's domain). Bin-mass integration via
%       per-axis Phi-difference contractions for isRel=false;
%       point-evaluation for isRel=true.
%
%     'normalized' (alias 'normalised') --- the Pielou-style ratio
%       H / log_b(N) in [0, 1]. Reproduces the values reported in
%       Milne et al. (2017) and Smit et al. (2019).
%
%     'differential' --- adaptive nested-grid evaluation of the
%       differential entropy h_hat = H_disc + log_b(Delta-volume).
%       The span auto-derives per attribute from centres +/-
%       truncationSigmas * sigma (non-periodic) or [0, period]
%       (periodic); the grid doubles from a sample-per-sigma initial
%       resolution until successive Richardson-extrapolated estimates
%       fall below tolerance. Grid-independent (no caller choice of
%       grid), and the principled scale-free quantity for comparisons
%       across densities of different cardinality or spread. Currently
%       restricted to single-density input; errors at sigma=0.
%
%     'renyi2' --- analytical Rényi-2 (collision) entropy
%       H_2 = -log_b(<T,T> / Z^2), computed in closed form via the
%       orbit-Möbius inner product (<T,T>) and the closed-form total
%       mass (Z). Grid-free; works at arbitrary tensor order r where
%       the Shannon-path Cartesian grid would exhaust memory.
%       Currently restricted to single-density input. Errors at
%       sigma=0.
%
%   v2.2 breaking change: the legacy 'normalize' boolean kwarg has
%   been removed. Use method='normalized' for the v2.1 default
%   behaviour (H/log_b(N) in [0, 1]) or method='shannon' for raw H.
%   Passing 'normalize' raises a migration-error exception.
%
%   The discrete methods ('shannon', 'normalized') require an
%   explicit 'nPointsPerDim'; the continuous methods ('differential',
%   'renyi2') do not accept it. The previous toolbox-wide default of
%   1200 for nPointsPerDim has been dropped, since the right grid
%   resolution is density- and sigma-dependent.
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
%     H = ENTROPYEXPTENS(pAttr, w, sigmaVec, rVec, ...
%                        isRelVec, isPerVec, periodVec)
%       Multi-attribute raw form. pAttr is a cell of per-attribute
%       matrices; per-attribute parameters as in buildExpTens.
%
%     H = ENTROPYEXPTENS(T)
%       Pre-built density form. T is a struct as returned by
%       buildExpTens. Dispatches on its tag and shape: a single-multiset
%       MaetDensity (A = N = 1) -> single-multiset path, a general
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
%   For periodic attributes (isPer = true), the Shannon grid spans
%   [0, period). For non-periodic attributes, bounds must be specified
%   via xMin and xMax, wide enough to capture the full support of the
%   distribution (e.g., at least 3*sigma beyond the outermost values).
%   Rényi-2 is grid-free and ignores xMin, xMax, and nPointsPerDim.
%
%   Multiset-argument shapes (pick one of three):
%     p      — Vector of length K. single multiset raw form (single multiset,
%              single-attribute).
%     P      — nRows-by-K matrix, both dimensions > 1. BATCHED-RAW
%              form (rows are independent single-attribute-style multisets,
%              processed in lockstep; returns an nRows-by-1 column
%              vector of per-row entropies).
%     pAttr  — 1-by-A cell of K_a-by-N matrices. MA raw form
%              (multi-attribute; per-attribute centre rows).
%   Lowercase p stands for "pitch or position"; uppercase P is the
%   2-D batched lift; pAttr is the multi-attribute generalisation.
%   The same convention is used in evalExpTens and cosSimExpTens.
%
%   Inputs (single multiset raw and BATCHED-RAW paths)
%       p       — Pitch or position values (vector of length K) for
%                 the single multiset raw form. The corresponding BATCHED-RAW form
%                 takes P (nRows-by-K matrix; each row is one
%                 single-attribute-style multiset).
%       w       — Weights. For single multiset raw: vector of length K, or [] for
%                 uniform. For BATCHED-RAW: nRows-by-K matrix, or [].
%                 (In the docstring above, this is denoted W when paired
%                 with P.)
%       sigma   — Gaussian bandwidth.
%       r       — Tuple size (positive integer; r >= 2 if isRel == true).
%       isRel   — Logical: true for relative (transposition-invariant).
%       isPer   — Logical: true for periodic domain.
%       period  — Period of the domain.
%
%   Inputs (MA raw path)
%       pAttr     — 1 x A cell array of K_a x N matrices.
%       w         - Weights. []/scalar/1 x A cell; see buildExpTens.
%       sigmaVec  - 1 x A per-attribute Gaussian widths.
%       rVec      - 1 x A per-attribute tuple sizes.
%       isRelVec  - 1 x A per-attribute relative flags.
%       isPerVec  - 1 x A per-attribute periodic flags.
%       periodVec - 1 x A per-attribute periods.
%
%   Name-Value Arguments
%       'method'        - One of {'shannon' (default), 'normalized',
%                         'differential', 'renyi2'} (or the British
%                         alias 'normalised'). See above.
%       'spectrum'      - (single multiset only.) Cell array of arguments passed to
%                         addSpectra. If provided, partials are added
%                         to the multiset before building the tensor.
%                         For MA, apply addSpectra to the pitch
%                         attribute before calling.
%       'base'          - Logarithm base (default: 2). The base cancels
%                         for method='normalized'.
%       'nPointsPerDim' - Required for method='shannon' and
%                         method='normalized' (no toolbox-wide default
%                         in v2.2); ignored by 'differential' and
%                         'renyi2'. Pass an explicit positive integer.
%       'xMin'          - Discrete methods, non-periodic only.
%                         single multiset: scalar. MA: scalar (broadcast to all
%                         non-periodic attributes) or length-A vector
%                         (one entry per attribute; periodic-attribute
%                         entries are ignored). Default: NaN.
%       'xMax'          - As xMin. Default: NaN.
%       'gridLimit'     - Ceiling on total grid size before allocation.
%                         Applies to MA always, and to single multiset whenever the
%                         density's effective dimension dim > 1 (e.g.
%                         r = 2 with isRel = false). Default: 1e8.
%                         Errors with a suggested reduction if exceeded.
%       'truncationSigmas' - Numeric scalar or []. Override the
%                         toolbox-wide mptDefaults('truncationSigmas')
%                         setting for this call. Passes through to the
%                         kernel evaluator on the centres path
%                         ('shannon', 'normalized', 'differential');
%                         skips Gaussian contributions whose
%                         centre-to-query distance exceeds k*sigma
%                         (kernel floor exp(-k^2/2)). For
%                         method='differential' this also anchors the
%                         convergence tolerance
%                         max(exp(-truncationSigmas^2/2), 1e-12).
%                         [] (default) means use the global default
%                         (factory: Inf).
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
%       % Shannon entropy of a 12-EDO chromatic scale (periodic, single multiset)
%       H = entropyExpTens(0:11, ones(1,12), 100, 1, false, true, 12);
%
%       % Same chord via pre-built density (Shannon)
%       T = buildExpTens([0 4 7], ones(1,3), 10, 1, false, true, 12);
%       H = entropyExpTens(T);
%
%       % Rényi-2 of the same chord --- closed-form, no grid
%       H = entropyExpTens(T, 'method', 'renyi2');
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

% Top-level call guard: dispatch throttle + kernelChunkBytes pin. See internal.callGuard.
guard = internal.callGuard(); %#ok<NASGU>

% Detect the legacy 'normalize' kwarg (removed in v2.2). We scan
% varargin directly *before* invoking localParseNVPairs (which would
% otherwise treat 'normalize' as a positional argument once it's
% gone from nvDefaults). A migration error then points users to
% method='normalized' / method='shannon'.
for kArg = 1:numel(varargin)
    if (ischar(varargin{kArg}) || (isstring(varargin{kArg}) ...
                                   && isscalar(varargin{kArg}))) ...
            && strcmpi(char(varargin{kArg}), 'normalize')
        error('entropyExpTens:normalizeRemoved', ...
              ['entropyExpTens: the ''normalize'' kwarg has been ' ...
               'removed in v2.2. Use method=''normalized'' for ' ...
               'H/log_b(N) in [0, 1] (the v2.1 default behaviour), ' ...
               'or method=''shannon'' for raw H = -sum q log_b q. ' ...
               'method=''differential'' and method=''renyi2'' are ' ...
               'continuous-form entropies and have no [0, 1] reference.']);
    end
end

nvDefaults = struct( ...
    'spectrum',          {{}}, ...
    'method',            'shannon', ...
    'base',              2, ...
    'nPointsPerDim',     [], ...
    'xMin',              NaN, ...
    'xMax',              NaN, ...
    'gridLimit',         1e8, ...
    'truncationSigmas',  [], ...
    'kernelPrecision',   [], ...
    'isSym',             [], ...
    'verbose',           true);

[posArgs, nvArgs] = localParseNVPairs(varargin, nvDefaults);
nPos = numel(posArgs);

if nPos < 1
    error('entropyExpTens:noArgs', ...
          'At least one positional argument is required.');
end

% Optional [sym] geometry flag for the raw forms. Entropy integrates
% over the whole space, so the raw layouts are pure geometry with no
% query: ..., period[, isSym]. A raw call therefore has 7 positional
% args, or 8 with isSym. (Struct and list forms have a single
% positional and never reach 8.) Pop a trailing isSym here, leaving
% posArgs at 7 so the per-method dispatch checks are unchanged, and
% stash it on nvArgs for the build calls. The default (empty =
% symmetric) comes from nvDefaults; an 8th positional overrides it.
if nPos == 8
    nvArgs.isSym = posArgs{8};
    posArgs(8) = [];
    nPos = numel(posArgs);
end

% Canonicalize the method kwarg (accepts British 'normalised') and
% validate against the four supported methods:
%   'differential' - adaptive truncationSigmas-aware nested grid;
%                    returns h_hat. Grid-independent, errors at sigma=0.
%   'shannon'      - raw discrete Shannon entropy H = -sum q log_b q
%                    on an explicit Cartesian-product grid.
%   'normalized'   - the Pielou-style ratio H/log_b(N) in [0, 1].
%   'renyi2'       - analytical Rényi-2 (collision) entropy.
%                    Grid-free, errors at sigma=0.
nvArgs.method = localCanonicalizeMethod(nvArgs.method);

% The internal nvArgs.normalize flag controls whether the discrete
% Shannon helper divides by log_b(N). It is determined here from the
% method (it is no longer user-facing; passing 'normalize' to this
% function triggers the migration error above).
switch nvArgs.method
    case 'normalized'
        nvArgs.normalize = true;
        H = localEntropyShannonDispatch(posArgs, nvArgs);
    case 'shannon'
        nvArgs.normalize = false;
        H = localEntropyShannonDispatch(posArgs, nvArgs);
    case 'differential'
        nvArgs.normalize = false;
        H = localEntropyDifferentialDispatch(posArgs, nvArgs);
    case 'renyi2'
        nvArgs.normalize = false;
        H = localEntropyRenyi2Dispatch(posArgs, nvArgs);
    otherwise
        error('entropyExpTens:internalCanonicalisation', ...
              'Internal error: canonicalised method %s not handled.', ...
              nvArgs.method);
end

end


% =========================================================================
%  localEntropyShannonDispatch — input-form resolution for Shannon entropy
% =========================================================================

function H = localEntropyShannonDispatch(posArgs, nvArgs)
%LOCALENTROPYSHANNONDISPATCH  Resolve input form and route to single multiset / MA helper.
%
%   Shannon entropy of the density evaluated on a Cartesian-product
%   grid; supports the full input surface (precomputed density struct,
%   list of densities, MA raw args, single multiset raw args, single-attribute batched 2-D
%   matrix).

    nPos = numel(posArgs);
    firstArg = posArgs{1};

    % ==================================================================
    % Canonical dispatch order (mirrors evalExpTens and cosSimExpTens):
    %   1. Struct first operand: switch firstArg.tag.
    %   2. Cell first operand:
    %        - cell-of-struct  -> LIST (cell of density structs)
    %        - cell-of-numeric -> MA raw (cell of attribute matrices)
    %   3. Numeric first operand:
    %        - 2-D with both dims > 1 -> BATCHED-RAW (rows = multisets)
    %        - vector or scalar       -> single multiset raw
    %   4. Otherwise -> usage error.
    % Each detector is positive (no reliance on a preceding check having
    % failed) and self-sufficient: reordering branches does not change
    % correctness.
    % ==================================================================

    % --- 1. Struct first operand: precomputed density ---
    if isstruct(firstArg) && isfield(firstArg, 'tag')
        if nPos > 1
            error('entropyExpTens:extraArgs', ...
                  ['When a precomputed density struct is passed, no ' ...
                   'further positional arguments may be provided.']);
        end
        switch firstArg.tag
            case 'MaetDensity'
                localRequireExplicitGrid(nvArgs.nPointsPerDim);
                if internal.isSingleMultiset(firstArg)
                    H = localEntropySingleMultiset(firstArg, nvArgs);
                else
                    H = localEntropyMA(firstArg, nvArgs);
                end
                return;
            case 'WindowedMaetDensity'
                localRequireExplicitGrid(nvArgs.nPointsPerDim);
                H = localEntropyMA(firstArg, nvArgs);
                return;
            otherwise
                error('entropyExpTens:unknownTag', ...
                      'Unknown density struct tag: %s.', firstArg.tag);
        end
    end

    % --- 2. Cell first operand: LIST or MA raw, disambiguated by ---
    % --- the inner element type. ---
    if iscell(firstArg)
        if isempty(firstArg)
            error('entropyExpTens:emptyCell', ...
                  ['First argument is an empty cell. Expected a cell of ' ...
                   'density structs (LIST mode) or a cell of attribute ' ...
                   'matrices (MA raw mode).']);
        end
        if isstruct(firstArg{1})
            % LIST: cell of density structs.
            if nPos > 1
                error('entropyExpTens:listExtraArgs', ...
                      ['When a cell of density structs is passed, no ' ...
                       'further positional arguments may be provided.']);
            end
            H = localEntropyDensityList(firstArg, nvArgs);
            return;
        end
        if isnumeric(firstArg{1})
            % MA raw: cell of attribute matrices.
            if nPos ~= 7
                error('entropyExpTens:wrongArgCountMA', ...
                      ['Multi-attribute raw call expects 7 or 8 positional ' ...
                       'arguments (pAttr, w, sigmaVec, rVec, ' ...
                       'isRelVec, isPerVec, periodVec[, isSymVec]); got %d.'], ...
                      nPos);
            end
            pAttr     = posArgs{1};
            w         = posArgs{2};
            sigmaVec  = posArgs{3};
            rVec      = posArgs{4};
            isRelVec  = posArgs{5};
            isPerVec  = posArgs{6};
            periodVec = posArgs{7};
            localRequireExplicitGrid(nvArgs.nPointsPerDim);
            symArgs = localSymArgs(nvArgs);
            dens = buildExpTens(pAttr, w, sigmaVec, rVec, ...
                                isRelVec, isPerVec, periodVec, symArgs{:}, ...
                                'verbose', false);
            H = localEntropyMA(dens, nvArgs);
            return;
        end
        error('entropyExpTens:badCellContents', ...
              ['Cell first argument must contain either density structs ' ...
               '(LIST mode) or numeric attribute matrices (MA raw mode); ' ...
               'first cell entry is of class %s.'], class(firstArg{1}));
    end

    % --- 3. Numeric first operand: BATCHED-RAW or single multiset raw, by shape. ---
    if isnumeric(firstArg)
        if size(firstArg, 1) > 1 && size(firstArg, 2) > 1
            % BATCHED-RAW: 2-D matrix with both dims > 1 (rows = multisets).
            if nPos ~= 7
                error('entropyExpTens:wrongArgCountBatched', ...
                      ['Batched-raw call expects 7 or 8 positional arguments ' ...
                       '(P, W, sigma, r, isRel, isPer, period[, isSym]); ' ...
                       'got %d.'], nPos);
            end
            localRequireExplicitGrid(nvArgs.nPointsPerDim);
            H = localEntropyBatchedRaw(posArgs, nvArgs);
            return;
        end
        % single multiset raw: numeric vector or scalar.
        if nPos ~= 7
            error('entropyExpTens:wrongArgCountSingleMultiset', ...
                  ['Single-attribute raw call expects 7 or 8 positional ' ...
                   'arguments (p, w, sigma, r, isRel, isPer, period' ...
                   '[, isSym]); got %d.'], nPos);
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

        localRequireExplicitGrid(nvArgs.nPointsPerDim);
        symArgs = localSymArgs(nvArgs);
        maet = buildExpTens( ...
            p, w, sigma, r, isRel, isPer, period, symArgs{:}, ...
            'verbose', false);
        H = localEntropySingleMultiset(maet, nvArgs);
        return;
    end

    % --- 4. Else: usage error ---
    error('entropyExpTens:badFirstArg', ...
          ['First argument must be a density struct, a cell array (LIST or ' ...
           'MA raw), or a numeric array (single multiset raw or BATCHED-RAW); got class %s.'], ...
          class(firstArg));

end


% =========================================================================
%  localEntropySingleMultiset — single-attribute Shannon entropy
% =========================================================================

function H = localEntropySingleMultiset(maet, nvArgs)
%LOCALENTROPYSINGLEMULTISET  Shannon entropy of the single-multiset (A = N = 1) corner.
%
%   Receives the MaetDensity; the flat cell-mass kernel reads the
%   single-multiset view, while the relative-mode point-evaluation branch
%   delegates to evalExpTens on the density itself.

    T = internal.singleMultisetView(maet);

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

    % Grid-size guard applies to both bin-integration and point-eval
    % paths since both materialise an array of length nPointsPerDim^dim.
    if dim > 1
        gridSize = nvArgs.nPointsPerDim ^ dim;
        if gridSize > nvArgs.gridLimit
            error('entropyExpTens:gridLimitExceeded', ...
                  ['single multiset Cartesian grid (%g points = nPointsPerDim^dim = %d^%d) ' ...
                   'exceeds gridLimit (%g). Reduce nPointsPerDim or raise ' ...
                   '''gridLimit''.'], gridSize, nvArgs.nPointsPerDim, dim, nvArgs.gridLimit);
        end
    end

    % Evaluate density on the grid.
    %
    % For absolute-mode densities (isRel=false) the categorical pmf is
    % the genuine bin masses int_{cell} f dx, obtained analytically
    % via per-axis erf differences. This matches Python's
    % _cell_masses_ma_absolute and gives Python/MATLAB parity on this
    % path. For relative-mode densities (isRel=true) the bin integral
    % is a multivariate-normal box probability (off-diagonal kernel
    % covariance in the effective coordinates); pending the v2.3
    % covariance machinery we fall back to point-evaluation here too.
    if ~logical(T.isRel)
        % Bin-integration cell-mass path. truncationSigmas is
        % plumbed for signature parity with the point-evaluation
        % branches --- localCellMassesSingleMultisetAbsolute's per-axis erf
        % differences are exact and do not truncate --- but is
        % still resolved via the contract helper so a user Inf
        % never propagates to the interior. Empty resolves to the
        % global default; Inf resolves to the accuracy-floor width.
        ts = internal.accuracyFloor('resolve', nvArgs.truncationSigmas);
        Tx = internal.singleMultisetView(internal.ensureExpTensExpensive(maet));
        t = localCellMassesSingleMultisetAbsolute(Tx, x1, ts);
    else
        % Build query matrix. For dim = 1, X is a 1 x nQ row vector. For
        % dim > 1, take the Cartesian product of dim copies of x1, giving a
        % dim x (nPointsPerDim^dim) matrix where each column is one point.
        if dim == 1
            X = x1;
        else
            gridArgs = repmat({x1}, 1, dim);
            gridCells = cell(1, dim);
            [gridCells{:}] = ndgrid(gridArgs{:});
            X = zeros(dim, nvArgs.nPointsPerDim ^ dim);
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
        t = evalExpTens(maet, X, evalKw{:});
    end

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

    % --- Resolve xMin/xMax to per-attribute vectors ---
    xMinG = localBroadcastBounds(nvArgs.xMin, A, 'xMin');
    xMaxG = localBroadcastBounds(nvArgs.xMax, A, 'xMax');

    % --- Check non-periodic attributes have valid bounds ---
    needsBounds = find(~isPerG);
    for idx = 1:numel(needsBounds)
        g = needsBounds(idx);
        if isnan(xMinG(g)) || isnan(xMaxG(g))
            error('entropyExpTens:missingBounds', ...
                  'xMin and xMax must be specified for non-periodic attribute %d.', g);
        end
        if xMinG(g) >= xMaxG(g)
            error('entropyExpTens:invalidBounds', ...
                  'xMin must be less than xMax (attribute %d).', g);
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
    % Each effective dimension belongs to an attribute, which carries
    % its own domain.
    axes1D = cell(1, dim);
    k = 0;
    for a = 1:A
        da = dimPer(a);
        if isPerG(a)
            P = periodG(a);
            ax = linspace(0, P, nvArgs.nPointsPerDim + 1);
            ax = ax(1:end-1);
        else
            ax = linspace(xMinG(a), xMaxG(a), nvArgs.nPointsPerDim);
        end
        for j = 1:da
            k = k + 1;
            axes1D{k} = ax;
        end
    end

    % --- Evaluate density on the grid ---
    %
    % For absolute-mode unwindowed densities (isRel=false everywhere)
    % the categorical pmf is the genuine bin masses (int_{cell} f dx),
    % obtained analytically via per-axis erf differences. This matches
    % Python's _cell_masses_ma_absolute and gives Python/MATLAB parity
    % on this path. For relative-mode densities the bin integral is a
    % multivariate-normal box probability (off-diagonal covariance in
    % the effective coordinates); pending the v2.3 covariance machinery
    % we fall back to point-evaluation, which agrees with bin-
    % integration to ~1e-4 on the fine grids relative-mode use-cases
    % require. Windowed densities also use point-evaluation here --
    % windowed cell-integration is a separate problem.
    isWindowed = isfield(dens, 'tag') && strcmp(dens.tag, 'WindowedMaetDensity');
    isAbs = ~any(logical(base_dens.isRel));
    if ~isWindowed && isAbs
        % Bin-integration cell-mass path (see the single multiset sibling above for
        % the truncationSigmas contract note).
        ts = internal.accuracyFloor('resolve', nvArgs.truncationSigmas);
        densX = ensureExpTensExpensive(base_dens);
        t = localCellMassesMAAbsolute(densX, axes1D, ts);
    else
        % --- Cartesian product as (dim x totalPoints) query matrix ---
        % Use ndgrid so the first axis varies fastest (column-major).
        meshCells = cell(1, dim);
        [meshCells{:}] = ndgrid(axes1D{:});
        X = zeros(dim, round(totalPoints));
        for d = 1:dim
            Md = meshCells{d};
            X(d, :) = Md(:).';
        end
        evalKw = {'verbose', false};
        if isfield(nvArgs, 'truncationSigmas') && ~isempty(nvArgs.truncationSigmas)
            evalKw = [evalKw, {'truncationSigmas', nvArgs.truncationSigmas}];
        end
        if isfield(nvArgs, 'kernelPrecision') && ~isempty(nvArgs.kernelPrecision)
            evalKw = [evalKw, {'kernelPrecision', nvArgs.kernelPrecision}];
        end
        t = evalExpTens(dens, X, evalKw{:});
    end

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

function out = localBroadcastBounds(v, A, name)
% Coerce xMin or xMax input to a length-A vector.
%   - scalar    -> broadcast to all attributes
%   - length-A  -> pass through
    v = double(v);
    if isscalar(v)
        out = repmat(v, 1, A);
        return;
    end
    if isvector(v) && numel(v) == A
        out = v(:).';
        return;
    end
    error('entropyExpTens:badBoundsShape', ...
          '%s must be a scalar or a length-%d vector (one per attribute); got size [%s].', ...
          name, A, num2str(size(v)));
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

    % First pass: validate that every entry is a struct. We do this
    % up-front (before any compute) so that the listNonStruct error
    % surfaces deterministically regardless of where the bad entry
    % sits, and ahead of the grid-required check that fires on the
    % recursive entropyExpTens call for the first valid entry.
    for i = 1:n
        if ~isstruct(densCell{i})
            error('entropyExpTens:listNonStruct', ...
                ['entropyExpTens (list mode): cell entries must be density ' ...
                 'structs from buildExpTens; entry %d is not a struct.'], i);
        end
    end

    % Re-pack the name-value defaults so we can pass them through.
    nvPairs = localPackNVPairs(nvArgs);

    for i = 1:n
        HCell{i} = entropyExpTens(densCell{i}, nvPairs{:});
    end
end


function H = localEntropyBatchedRaw(posArgs, nvArgs)
%LOCALENTROPYBATCHEDRAW Per-row entropy from a 2-D pitch matrix.
%
%   posArgs follows the single multiset-raw convention: {P, W, sigma, r, isRel,
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

    % The per-row dedup keys rows by a multiset canonical form, which
    % collapses rows that share a multiset but differ in order. That is
    % correct only for the symmetric reading: under isSym = false the
    % order is significant, so the dedup would silently merge distinct
    % ordered densities (and hence entropies). Reject rather than return
    % a wrong answer (parity with the Python batched path). Order-aware
    % batched dedup is a tracked follow-up; compute ordered densities one
    % row at a time.
    if isfield(nvArgs, 'isSym') && ~isempty(nvArgs.isSym) ...
            && ~all(logical(nvArgs.isSym(:))) && r > 1
        error('entropyExpTens:batchedOrderedUnsupported', ...
              ['entropyExpTens batched (2-D) input does not yet support ' ...
               'isSym = false (ordered) densities at r > 1: the batched ' ...
               'dedup canonicalises each row''s multiset and would merge ' ...
               'order-distinct rows. Compute ordered densities one row ' ...
               'at a time (vector input).']);
    end

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
%
%   The 'normalize' field is omitted: it is an internal-only flag set
%   from the user-facing 'method' value (true for 'normalized', false
%   for the other three methods), not a name-value pair the user is
%   allowed to supply. Recursive entropyExpTens calls would otherwise
%   see 'normalize' in varargin and trip the v2.2 migration error.
%
%   The 'isSym' field is likewise omitted: it is an internal-only
%   carrier for the optional trailing positional flag, popped from the
%   raw-form positional args. It is not a name-value pair, so forwarding
%   it would be mis-parsed as a positional argument by the recursive
%   call. The LIST path's densities already carry their own isSym, and
%   the BATCHED path builds symmetric per-row densities by default
%   (ordered batched input is rejected before any recursion).

    nvPairs = {};
    fns = fieldnames(nvArgs);
    for i = 1:numel(fns)
        if strcmp(fns{i}, 'normalize') || strcmp(fns{i}, 'isSym')
            continue;
        end
        nvPairs = [nvPairs, {fns{i}, nvArgs.(fns{i})}]; %#ok<AGROW>
    end
end


% =========================================================================
%  Rényi-2 (collision) entropy via orbit-Möbius IP
% =========================================================================

% =========================================================================
%  Bin-integration core (cell masses via per-axis Phi-differences)
% =========================================================================
%
% The categorical-path discretization for 'shannon' and 'normalized':
% the discrete pmf entry at grid cell j is the actual probability mass
% inside that cell, int_{cell_j} f dx, not the density sample f(x_j) * Delta.
% For a Gaussian-mixture density with diagonal kernel covariance in the
% effective grid coordinates --- which holds for isRel=false (every group
% absolute) --- the cell mass factorizes into a product of per-axis erf
% differences, summed over tuples. Mirrors the Python implementation in
% python/mpt/entropy.py for bit-for-bit parity.
%
% n_tuple_entropy reaches this path by differencing events externally
% (differenceEvents + bindEvents) and then building an absolute
% (isRel=false) MAET, so its sigma is the effective sigma_eff already.
% Relative-mode direct calls (isRel=true) fall back to point-evaluation,
% which agrees with bin-integration to ~1e-4 on the fine grids relative-
% mode use-cases require; the full multivariate-normal box treatment is
% a v2.3 item.


function Mat = localPhiDiffAxis(centres, edgesLo, edgesHi, sigma)
%LOCALPHIDIFFAXIS  Non-periodic per-axis erf-difference cell mass.
%
%   Returns an (nJ x nCells) array with entry [t, j] equal to
%   Phi((edgesHi(j) - centres(t))/sigma) - Phi((edgesLo(j) -
%   centres(t))/sigma), the 1-D Gaussian probability mass in cell j
%   for the tuple-slot at centres(t).

    centres = centres(:);   % (nJ x 1)
    edgesLo = edgesLo(:).'; % (1 x nCells)
    edgesHi = edgesHi(:).';
    inv = 1.0 / (sigma * sqrt(2));
    zHi = (edgesHi - centres) * inv;   % (nJ x nCells), broadcast
    zLo = (edgesLo - centres) * inv;
    Mat = 0.5 * (erf(zHi) - erf(zLo));
end


function Mat = localPhiDiffAxisPeriodic(centres, edgesLo, edgesHi, sigma, period, truncationSigmas) %#ok<INUSD>
%LOCALPHIDIFFAXISPERIODIC  Periodic per-axis erf-difference cell mass (minimum-image).
%
%   The kernel is the minimum-image Gaussian on the circle of
%   circumference period: each edge offset is wrapped componentwise to
%   [-period/2, period/2), the same wrap of the difference used by
%   evalExpTens and the cosine inner product. A bin straddling a centre's
%   antipode --- where the wrap flips the edge order --- receives the full
%   wrap-around mass erf(period / (2 sqrt(2) sigma)). The masses are
%   renormalized to sum to one by the entropy cores (which divide by their
%   total), absorbing the sub-unit mass of the truncated circle.
%   truncationSigmas is accepted for call-signature parity with the
%   non-periodic path and is unused.

    centres = centres(:);
    edgesLo = edgesLo(:).';
    edgesHi = edgesHi(:).';
    inv = 1.0 / (sigma * sqrt(2));
    a = edgesLo - centres;
    a = a - period * round(a / period);
    b = edgesHi - centres;
    b = b - period * round(b / period);
    Mat = 0.5 * (erf(b * inv) - erf(a * inv));
    Mat = Mat + (a > b) .* erf((0.5 * period) * inv);
end


function [lo, hi] = localAxisEdges(ax, isPer, period)
%LOCALAXISEDGES  Cell edges for a 1-D axis.
%
%   For a periodic group, ax is linspace(0, P, n+1) without its last
%   point, and each cell is symmetric of width P/n around its grid
%   point. For non-periodic, the interior cells are bounded by mid-
%   points between adjacent grid points; the boundary cells extend by
%   half-step on each side. Returns (lo, hi) arrays both shaped like
%   ax.

    ax = double(ax(:).');
    n = numel(ax);
    if n < 2
        error('localAxisEdges:tooFewPoints', ...
              'Each axis needs >= 2 points (got %d).', n);
    end
    if isPer
        step = double(period) / double(n);
        lo = ax - step / 2.0;
        hi = ax + step / 2.0;
        return;
    end
    mids = 0.5 * (ax(1:end-1) + ax(2:end));
    step0 = double(ax(2) - ax(1));
    stepN = double(ax(end) - ax(end-1));
    lo = [ax(1) - step0/2.0, mids];
    hi = [mids, ax(end) + stepN/2.0];
end


function out = localContractTupleAxes(wJ, Mats)
%LOCALCONTRACTTUPLEAXES  Einsum-equivalent reduction sum_t w(t) * prod_d Mats{d}(t, n_d).
%
%   Mats is a 1-by-D cell of (nJ x n_d) per-axis matrices. wJ is
%   (nJ x 1). Returns a flat column vector of length prod_d n_d.
%
%   Implementation notes for memory and performance.
%
%   For D=1 and D=2 the contraction reduces to a matrix-vector and a
%   matrix-matrix product respectively, with no t-dependent intermediate.
%
%   For D>=3 the contraction necessarily holds a t-indexed tensor
%   until the final sum over t (since every Mats{d} carries the t
%   index). Materialising the full (nJ x prod n_d) intermediate would
%   blow up for densities with large nJ at moderate dim (e.g. nJ=6840
%   at dim=3 N=100 -> ~51 GB). We therefore process t in chunks, with
%   the chunk size chosen so the per-chunk working set stays bounded.
%   Python's numpy.einsum handles the same case via C-level iterator
%   accumulation without an explicit intermediate; the t-chunking
%   here is the MATLAB-side equivalent of that bounded-memory
%   strategy.
%
%   Layout: the flat output uses MATLAB column-major ordering over
%   (n_1, n_2, ..., n_D) i.e. n_1 varies fastest. This differs from
%   Python's numpy C-order, but the entropy consumer normalises and
%   sums over cells which is invariant to ordering, so cross-language
%   parity at the entropy value level holds. For symmetric output
%   tensors (e.g. absolute-mode densities under the standard
%   permutation-symmetric build) the orderings happen to coincide
%   element-wise as well.

    D = numel(Mats);
    nJ = numel(wJ);
    if D == 0
        out = sum(wJ);
        return;
    end

    nDims = zeros(1, D);
    for d = 1:D
        nDims(d) = size(Mats{d}, 2);
    end

    % Fast paths for D <= 2: standard BLAS-friendly matrix products,
    % no t-dependent intermediate.
    if D == 1
        out = (wJ(:).' * Mats{1}).';
        return;
    end
    if D == 2
        % cells(a, b) = sum_t wJ(t) * M1(t,a) * M2(t,b)
        %             = M1.' * (wJ .* M2)
        cells2 = Mats{1}.' * (wJ(:) .* Mats{2});
        out = cells2(:);
        return;
    end

    % D >= 3: t-chunked contraction.
    prodN = prod(nDims);
    MAX_BYTES = 2^28;   % 256 MB per-chunk working set
    B = max(1, floor(MAX_BYTES / (max(prodN, 1) * 8)));
    B = min(B, nJ);

    out = zeros(prodN, 1);
    wJcol = wJ(:);
    for tb = 1:B:nJ
        te = min(tb + B - 1, nJ);
        idx = tb:te;
        Bt = numel(idx);

        % Build chunk's (Bt x prodN) accumulator.
        acc = wJcol(idx) .* Mats{1}(idx, :);    % (Bt x n_1)
        for d = 2:D
            nD_ = nDims(d);
            nAcc = size(acc, 2);
            acc = reshape(acc, [Bt, nAcc, 1]) ...
                .* reshape(Mats{d}(idx, :), [Bt, 1, nD_]);
            acc = reshape(acc, [Bt, nAcc * nD_]);
        end

        % Sum over the chunk's tuple axis and accumulate.
        out = out + sum(acc, 1).';
    end
end


function blk = localDiffCellBlock()
%LOCALDIFFCELLBLOCK  Peak working-set ceiling, in matrix elements, for
%   the per-axis erf-difference cell-mass evaluation. Mirrors Python's
%   _DIFF_CELL_BLOCK. A dim==1 differential grid can be refined to a very
%   fine cell count by the adaptive evaluator; evaluating the leading
%   axis in cell blocks of this many elements bounds the peak without
%   changing the result.
    blk = 8e6;
end


function M = localAxisMat(spec, sel, truncationSigmas)
%LOCALAXISMAT  Per-axis (nJ x nCells) erf-difference cell-mass matrix for
%   one axis spec (fields: cents, lo, hi, sigma, isPer, per). sel selects
%   a subset of cells; pass [] for all cells.
    if isempty(sel)
        lo = spec.lo;
        hi = spec.hi;
    else
        lo = spec.lo(sel);
        hi = spec.hi(sel);
    end
    if spec.isPer
        M = localPhiDiffAxisPeriodic(spec.cents, lo, hi, spec.sigma, ...
                                     spec.per, truncationSigmas);
    else
        M = localPhiDiffAxis(spec.cents, lo, hi, spec.sigma);
    end
end


function cells = localContractCellAxes(wJ, axisSpecs, truncationSigmas)
%LOCALCONTRACTCELLAXES  Contract per-axis erf-difference cell masses into
%   a flat grid, streaming the leading axis in cell blocks. Mirrors
%   Python's _contract_cell_axes.
%
%   axisSpecs is a 1-by-D cell of axis-spec structs (see localAxisMat);
%   wJ holds the per-tuple weights. Returns a flat column vector of cell
%   masses, each cell's contribution summed in full over tuples.
%
%   The leading axis is streamed in cell blocks for D <= 2. At D == 1 the
%   adaptive differential evaluator can refine the single axis to a very
%   fine grid, so its (nJ x nCells) matrix must not be built whole; at
%   D == 2 the leading axis is the larger grid, the single trailing
%   matrix staying whole (bounded by grid_limit). For D >= 3 every axis
%   grid is bounded by grid_limit^(1/D), so the whole-matrix t-chunked
%   contraction (localContractTupleAxes) is already memory-safe and is
%   used directly. Blocking the leading axis leaves each cell's full sum
%   over tuples intact, so the result matches the whole-matrix path.

    D = numel(axisSpecs);
    nTup = numel(wJ);
    wJcol = wJ(:);

    if D <= 2
        spec0 = axisSpecs{1};
        n0 = numel(spec0.lo);
        if D == 1
            tail = [];
            rest = 1;
            cellsMat = zeros(n0, 1);
        else
            tail = localAxisMat(axisSpecs{2}, [], truncationSigmas);  % (nJ x n_1)
            rest = size(tail, 2);
            cellsMat = zeros(n0, rest);
        end
        block = max(1, floor(localDiffCellBlock() / max([1, nTup, rest])));
        for s = 1:block:n0
            e = min(s + block - 1, n0);
            head = localAxisMat(spec0, s:e, truncationSigmas);   % (nJ x (e-s+1))
            if isempty(tail)
                cellsMat(s:e) = (wJcol.' * head).';
            else
                cellsMat(s:e, :) = head.' * (wJcol .* tail);
            end
        end
        cells = cellsMat(:);
        return;
    end

    % D >= 3: whole per-axis matrices, t-chunked contraction (its grids
    % are bounded by grid_limit^(1/D)).
    Mats = cell(1, D);
    for d = 1:D
        Mats{d} = localAxisMat(axisSpecs{d}, [], truncationSigmas);
    end
    cells = localContractTupleAxes(wJcol, Mats);
end


function cells = localCellMassesSingleMultisetAbsolute(T, ax, truncationSigmas)
%LOCALCELLMASSESSINGLEMULTISETABSOLUTE  Cell masses for the single-multiset corner.
%
%   Returns a flat (prod_d n_cells x 1) column vector of integrated
%   cell masses int_{cell} f dx via per-axis erf differences. Restricted
%   to isRel=false; the caller is responsible for routing isRel=true
%   elsewhere. Mirrors Python's _cell_masses_ma_absolute exactly so
%   numerical outputs match across languages.

    if logical(T.isRel)
        error('entropyExpTens:cellMassesSingleMultisetNotAbsolute', ...
            'localCellMassesSingleMultisetAbsolute: isRel=true is not supported by this path.');
    end
    dim = double(T.dim);
    sig = double(T.sigma);
    isPer = logical(T.isPer);
    per = double(T.period);
    if ~isPer
        per = 0.0;
    end
    C = double(T.Centres);         % (dim x nJ) when isRel=false
    wJ = double(T.wJ(:));

    % Auto-prune zero-weight tuples (see localCellMassesMAAbsolute).
    mask = wJ > 0;
    if ~all(mask)
        wJ = wJ(mask);
        C = C(:, mask);
    end

    if dim == 0
        cells = sum(wJ);
        return;
    end

    % All effective axes share the same 1-D grid edges (the single multiset grid is a
    % single axis repeated across the dim effective dimensions) but a
    % different centres row. Stream the leading axis in cell blocks for
    % dim <= 2 via the shared contraction.
    [lo, hi] = localAxisEdges(ax, isPer, per);
    axisSpecs = cell(1, dim);
    for d = 1:dim
        axisSpecs{d} = struct('cents', C(d, :), 'lo', lo, 'hi', hi, ...
            'sigma', sig, 'isPer', isPer, 'per', per);
    end
    cells = localContractCellAxes(wJ, axisSpecs, truncationSigmas);
end


function cells = localCellMassesMAAbsolute(dens, axes, truncationSigmas)
%LOCALCELLMASSESMAABSOLUTE  Cell masses for a MaetDensity (MA path).
%
%   Returns a flat column vector of integrated cell masses on the
%   Cartesian-product grid built from axes (a 1-by-D cell of 1-D
%   linspaces). Restricted to absolute-mode densities (every group
%   isRel=false). Mirrors Python's _cell_masses_ma_absolute.

    if any(logical(dens.isRel))
        error('entropyExpTens:cellMassesMANotAbsolute', ...
            ['localCellMassesMAAbsolute: relative-mode densities are ' ...
             'not supported by this path. Route isRel=true via point-' ...
             'evaluation.']);
    end

    A = double(dens.nAttrs);
    dimPer = double(dens.dimPerAttr);
    sigmaG = double(dens.sigma);
    isPerG = logical(dens.isPer);
    periodG = double(dens.period);
    Centres = dens.Centres;        % 1-by-A cell; each (dim_per(a) x nJ)
    wJ = double(dens.wJ(:));

    % Auto-prune zero-weight tuples. Mirrors the eval-path prune:
    % zero-weight tuples contribute exactly zero to the tensor
    % contraction, so dropping them is mathematically exact and avoids
    % building (nJ x n_cells) erf-difference matrices over tuples that
    % weightEvents has truncated to zero.
    mask = wJ > 0;
    if ~all(mask)
        wJ = wJ(mask);
        for a = 1:A
            Centres{a} = Centres{a}(:, mask);
        end
    end

    % Collect per-effective-axis specs; the shared contraction streams
    % the leading axis in cell blocks for D <= 2.
    axisSpecs = {};
    axisD = 0;
    for a = 1:A
        da = double(dimPer(a));
        sig = double(sigmaG(a));
        isPerA = isPerG(a);
        if isPerA
            perA = double(periodG(a));
        else
            perA = 0.0;
        end
        Ca = double(Centres{a});  % (da x nJ)
        for sub = 1:da
            axisD = axisD + 1;
            ax = axes{axisD};
            [lo, hi] = localAxisEdges(ax, isPerA, perA);
            axisSpecs{axisD} = struct('cents', Ca(sub, :), 'lo', lo, ...
                'hi', hi, 'sigma', sig, 'isPer', isPerA, 'per', perA); %#ok<AGROW>
        end
    end

    D = axisD;
    if D == 0
        cells = sum(wJ);
        return;
    end
    cells = localContractCellAxes(wJ, axisSpecs, truncationSigmas);
end


function c = localSymArgs(nvArgs)
%LOCALSYMARGS  Cell of the optional isSym positional for buildExpTens.
%
%   Returns {} when no [sym] flag was supplied (symmetric default) or
%   {isSym} otherwise, for splatting into a buildExpTens call as the
%   trailing positional after periodVec.
    if ~isfield(nvArgs, 'isSym') || isempty(nvArgs.isSym)
        c = {};
    else
        c = {nvArgs.isSym};
    end
end


function methodCanon = localCanonicalizeMethod(methodRaw)
%LOCALCANONICALIZEMETHOD  Validate and canonicalize the 'method' kwarg.
%
%   Accepts 'normalised' as an alias for 'normalized'. Errors for
%   unrecognised names.

    if ~(ischar(methodRaw) || isstring(methodRaw))
        error('entropyExpTens:badMethodType', ...
              '''method'' must be a string; got %s.', class(methodRaw));
    end
    m = lower(strtrim(char(methodRaw)));
    if strcmp(m, 'normalised')
        m = 'normalized';
    end
    valid = {'differential', 'shannon', 'normalized', 'renyi2'};
    if ~any(strcmp(m, valid))
        error('entropyExpTens:badMethod', ...
              ['''method'' must be one of ' ...
               '{''differential'', ''shannon'', ''normalized'', ' ...
               '''renyi2''} (or the British alias ''normalised''); ' ...
               'got ''%s''.'], methodRaw);
    end
    methodCanon = m;
end


function localRaiseIfAnySigmaZero(dens, methodName)
%LOCALRAISEIFANYSIGMAZERO  Reject sigma=0 for continuous methods.
%
%   The continuous-form entropies ('differential', 'renyi2') diverge
%   at sigma=0. Reads sigma from any density form (MaetDensity,
%   WindowedMaetDensity) or the single-multiset view.

    if isfield(dens, 'tag') && strcmp(dens.tag, 'WindowedMaetDensity')
        sigma = dens.dens.sigma;
    else
        sigma = dens.sigma;
    end
    if ~isempty(sigma) && any(double(sigma(:)) <= 0)
        error('entropyExpTens:sigmaZeroNotSupported', ...
              ['method=''%s'' requires sigma > 0 for every group ' ...
               '(the continuous form diverges at sigma=0). For ' ...
               'categorical sigma=0 entropy, use method=''shannon'' ' ...
               'or method=''normalized'' on a category grid.'], ...
              methodName);
    end
end


function localRequireExplicitGrid(nPointsPerDim)
%LOCALREQUIREEXPLICITGRID  Require explicit grid for discrete methods.
%
%   The previous toolbox-wide default of 1200 has been dropped, since
%   the right grid resolution is density- and sigma-dependent. Called
%   from each compute-path branch of the Shannon dispatch (after
%   input-form validation, so the more-specific input-form errors
%   surface first when both apply).

    if isempty(nPointsPerDim)
        error('entropyExpTens:gridRequired', ...
              ['Discrete entropy (method=''shannon'' or ' ...
               '''normalized'') requires an explicit ' ...
               '''nPointsPerDim'' (the previous toolbox-wide default ' ...
               'of 1200 has been dropped, since the right grid ' ...
               'resolution is density- and sigma-dependent). For a ' ...
               'grid-free continuous quantity, use ' ...
               'method=''differential'' (adaptive) or ' ...
               'method=''renyi2'' (analytical).']);
    end
end


% =========================================================================
%  Adaptive differential entropy (method='differential')
% =========================================================================
%
% h_hat = H_disc + log_b(cell_volume), converged on a per-axis nested-
% grid refinement to a truncation-sigma-anchored tolerance. The span
% auto-derives per attribute from `centres +/- truncation_sigmas * sigma`
% (non-periodic) or `[0, period]` (periodic). The initial resolution
% is ~2 samples per sigma per axis; N doubles each iteration until
% successive Richardson-extrapolated estimates fall below tolerance,
% the differences stop decreasing (numerical-floor guard), or
% gridLimit is hit. Mirrors the Python implementation in
% python/mpt/entropy.py.


function H = localEntropyDifferentialDispatch(posArgs, nvArgs)
%LOCALENTROPYDIFFERENTIALDISPATCH  Adaptive differential entropy dispatch.
%
%   Single-density input only (scalar density struct, raw scalar single multiset,
%   or raw scalar MA). List and batched input forms raise informative
%   errors. WindowedMaetDensity is not yet supported.

    nPos = numel(posArgs);
    firstArg = posArgs{1};

    % --- Reject unsupported input forms early ---
    if iscell(firstArg) && ~isempty(firstArg) && isstruct(firstArg{1})
        error('entropyExpTens:differentialListNotSupported', ...
            ['method=''differential'' does not yet support list ' ...
             'input. Apply it to each density individually.']);
    end
    if isnumeric(firstArg) && size(firstArg, 1) > 1 && size(firstArg, 2) > 1
        error('entropyExpTens:differentialBatchedNotSupported', ...
            ['method=''differential'' does not yet support raw single multiset ' ...
             'batched (2-D) input. Pass each chord row individually, ' ...
             'or pre-build a density struct.']);
    end

    % --- Resolve input to a density struct ---
    if isstruct(firstArg) && isfield(firstArg, 'tag')
        if nPos > 1
            error('entropyExpTens:extraArgs', ...
                ['When a precomputed density struct is passed, no ' ...
                 'further positional arguments may be provided.']);
        end
        switch firstArg.tag
            case 'MaetDensity'
                dens = firstArg;
                singleMultisetInput = internal.isSingleMultiset(firstArg);
            case 'WindowedMaetDensity'
                error('entropyExpTens:differentialWindowedNotSupported', ...
                    ['method=''differential'' with ' ...
                     'WindowedMaetDensity is not yet implemented.']);
            otherwise
                error('entropyExpTens:unknownTag', ...
                    'Unknown density struct tag: %s.', firstArg.tag);
        end
    elseif iscell(firstArg)
        % MA raw args.
        if nPos ~= 7
            error('entropyExpTens:wrongArgCountMA', ...
                ['Multi-attribute raw call expects 7 or 8 positional ' ...
                 'arguments (pAttr, w, sigmaVec, rVec, ' ...
                 'isRelVec, isPerVec, periodVec[, isSymVec]); got %d.'], nPos);
        end
        symArgs = localSymArgs(nvArgs);
        dens = buildExpTens(posArgs{1}, posArgs{2}, posArgs{3}, posArgs{4}, ...
                            posArgs{5}, posArgs{6}, posArgs{7}, symArgs{:}, ...
                            'verbose', false);
        singleMultisetInput = false;
    else
        % single multiset raw args.
        if nPos ~= 7
            error('entropyExpTens:wrongArgCountSingleMultiset', ...
                ['Single-attribute raw call expects 7 or 8 positional ' ...
                 'arguments (p, w, sigma, r, isRel, isPer, period' ...
                 '[, isSym]); got %d.'], nPos);
        end
        p      = posArgs{1};
        w      = posArgs{2};
        sigma  = posArgs{3};
        r      = posArgs{4};
        isRel  = posArgs{5};
        isPer  = posArgs{6};
        period = posArgs{7};
        if ~isempty(nvArgs.spectrum)
            if internal.isKernelCov(sigma)
                error('mpt:aniso:spectrumUnsupported', ...
                    ['''spectrum'' is not supported with a matrix-valued ' ...
                     'kernel covariance (spectral augmentation changes ' ...
                     'the multiset size, breaking r == K).']);
            end
            if ~iscell(nvArgs.spectrum)
                error('entropyExpTens:badSpectrum', ...
                    '''spectrum'' value must be a cell array.');
            end
            [p, w] = addSpectra(p, w, nvArgs.spectrum{:});
        end
        symArgs = localSymArgs(nvArgs);
        dens = buildExpTens( ...
            p, w, sigma, r, isRel, isPer, period, symArgs{:}, ...
            'verbose', false);
        singleMultisetInput = true;
    end

    % --- sigma > 0 guard ---
    localRaiseIfAnySigmaZero(dens, 'differential');

    % --- Resolve truncationSigmas via the contract helper ---
    % Empty resolves to the global default; Inf resolves to the finite
    % accuracy-floor width (~7.43 sigma, the 1e-12 floor). The
    % differential span and tolerance anchoring downstream then have a
    % well-defined finite radius without any local isfinite guard.
    ts = internal.accuracyFloor('resolve', nvArgs.truncationSigmas);

    H = localDifferentialAdaptive(dens, singleMultisetInput, nvArgs.base, ts, ...
                                  nvArgs.gridLimit, nvArgs.verbose);
    H = localAnisoEntropyCorrection(H, dens, nvArgs.base);
end


function H = localDifferentialAdaptive(dens, singleMultisetInput, base, ts, gridLimit, verbose)
%LOCALDIFFERENTIALADAPTIVE  Nested-grid h_hat with Richardson extrapolation.

    % ts is resolved to a finite width at the dispatcher entry
    % (localEntropyDifferentialDispatch), so it is always a finite
    % positive scalar here and drives the span, the convergence
    % tolerance, and the downstream kernel truncation from a single
    % source. Inf never reaches this function.
    tol = max(exp(-0.5 * ts * ts), 1e-12);
    maxIter = 10;

    if singleMultisetInput
        [xMin, xMax, n0, dim, perAxisW, perAxisPer] = localDiffSpansSingleMultiset(dens, ts);
    else
        [xMinG, xMaxG, n0, dim, perAxisW, perAxisPer] = localDiffSpansMA(dens, ts);
    end

    N = max(n0, 4);
    % Feasibility ceiling, sized from available memory rather than a fixed
    % constant. During evaluation the grid holds two persistent arrays --
    % the coordinate mesh (dim, N^dim) and the density values (N^dim, 1),
    % together (dim + 1)*N^dim*8 bytes -- while a kernel-evaluation chunk
    % (itself budgeted at kernelChunkBytes) runs on top, and building the
    % mesh transiently doubles the coordinate array. Bound the persistent
    % grid footprint at half the kernel-chunk budget (a quarter of
    % available memory under the factory 'auto' setting) so the footprint,
    % a concurrent chunk, and the build-time transient all fit with
    % headroom. Sizing from kernelChunkBytes tracks the machine and honours
    % a pinned value. Mirrors the Python implementation.
    memCapPoints = max( ...
        floor(internal.kernelChunkBytesResolved() / (16 * (dim + 1))), ...
        4 ^ max(dim, 1));
    effGridLimit = min(gridLimit, memCapPoints);
    hHistory = [];
    rHistory = [];
    logB = log(base);
    H = NaN;

    for it = 1:maxIter
        if dim > 0
            total = double(N) ^ double(dim);
            if total > effGridLimit
                % Reaching here means the accuracy has not yet been
                % certified (that path returns above) and the next grid
                % would exceed the feasible budget. Refuse and direct the
                % user to the real choices, rather than degrading silently
                % or exhausting memory.
                if memCapPoints <= gridLimit
                    choices = ['use a lower truncationSigmas (coarser ' ...
                        'accuracy needs a smaller grid); or use ' ...
                        'method=''renyi2'' (closed form, no grid)'];
                else
                    choices = ['raise gridLimit; use a lower ' ...
                        'truncationSigmas (coarser accuracy needs a ' ...
                        'smaller grid); or use method=''renyi2'' ' ...
                        '(closed form, no grid)'];
                end
                error('entropyExpTens:differentialGridLimit', ...
                    ['method=''differential'' cannot certify the ' ...
                     'requested accuracy (truncationSigmas=%.3g) in ' ...
                     'dim=%d: convergence needs more than %.0f grid ' ...
                     'points. %s.'], ts, dim, effGridLimit, choices);
            end
        end

        % Compute H_disc on this grid via the existing Shannon path.
        nvSub = struct( ...
            'spectrum',          {{}}, ...
            'method',            'shannon', ...
            'normalize',         false, ...
            'base',              base, ...
            'nPointsPerDim',     N, ...
            'xMin',              NaN, ...
            'xMax',              NaN, ...
            'gridLimit',         gridLimit, ...
            'truncationSigmas',  ts, ...
            'kernelPrecision',   [], ...
            'verbose',           false);
        if singleMultisetInput
            nvSub.xMin = xMin;
            nvSub.xMax = xMax;
            HDisc = localEntropySingleMultiset(dens, nvSub);
        else
            nvSub.xMin = xMinG;
            nvSub.xMax = xMaxG;
            HDisc = localEntropyMA(dens, nvSub);
        end

        % log_b(Delta_d) summed across axes.
        % Periodic axes: Delta = W/N (linspace [0,P) at step P/N).
        % Non-periodic: Delta = W/(N-1) (linspace endpoint-inclusive,
        % N-1 intervals).
        logCellVol = 0;
        for d = 1:dim
            if perAxisPer(d)
                delta = perAxisW(d) / N;
            else
                delta = perAxisW(d) / (N - 1);
            end
            logCellVol = logCellVol + log(delta) / logB;
        end
        hHat = double(HDisc) + logCellVol;
        hHistory(end+1) = hHat; %#ok<AGROW>

        if numel(hHistory) >= 2
            hPrev = hHistory(end-1);
            hCurr = hHistory(end);
            % Direct h_hat convergence (1-D regime: fast).
            if abs(hCurr - hPrev) < tol
                H = hCurr;
                return;
            end
            % Richardson extrapolation: O(Delta^2) -> O(Delta^4).
            R = hCurr + (hCurr - hPrev) / 3.0;
            rHistory(end+1) = R; %#ok<AGROW>
            if numel(rHistory) >= 2
                dR = abs(rHistory(end) - rHistory(end-1));
                if dR < tol
                    H = rHistory(end);
                    return;
                end
                if numel(rHistory) >= 3
                    dRPrev = abs(rHistory(end-1) - rHistory(end-2));
                    if dRPrev > 0 && dR >= 0.95 * dRPrev
                        % Floor reached.
                        H = rHistory(end);
                        return;
                    end
                end
            end
        end
        N = N * 2;
    end

    if ~isempty(rHistory)
        H = rHistory(end);
    elseif ~isempty(hHistory)
        H = hHistory(end);
    end
end


function [xMin, xMax, n0, dim, perAxisW, perAxisPer] = localDiffSpansSingleMultiset(maet, ts)
%LOCALDIFFSPANSSINGLEMULTISET  Auto-spans for the single-multiset (A = N = 1) corner.

    T = internal.singleMultisetView(maet);

    sig = double(T.sigma);
    isPer = logical(T.isPer);
    per = double(T.period);
    dim = double(T.dim);

    if isPer
        xMin = NaN;
        xMax = NaN;
        W = per;
    else
        % Zero-weight events contribute nothing to any live tuple, so
        % they must not enlarge the span (see localDiffSpansMA for the
        % auto/manual-prune invariance this preserves). Mask the event
        % pitches by their weights where a weight vector is available.
        c = double(T.p(:));
        if isfield(T, 'w') && ~isempty(T.w)
            wv = double(T.w(:));
            if numel(wv) == numel(c)
                c = c(wv > 0);
            end
        end
        if isempty(c)
            cMin = 0;
            cMax = 0;
        else
            cMin = min(c);
            cMax = max(c);
        end
        xMin = cMin - ts * sig;
        xMax = cMax + ts * sig;
        W = xMax - xMin;
    end
    perAxisW = repmat(W, 1, dim);
    perAxisPer = repmat(isPer, 1, dim);
    n0 = max(4, ceil(2.0 * W / sig));
end


function [xMinG, xMaxG, n0, dim, perAxisW, perAxisPer] = localDiffSpansMA(dens, ts)
%LOCALDIFFSPANSMA  Auto-spans for a MaetDensity.

    A = double(dens.nAttrs);
    dimPer = double(dens.dimPerAttr);
    sigmaG = double(dens.sigma);
    isPerG = logical(dens.isPer);
    periodG = double(dens.period);
    pAttr = dens.pAttr;
    wCell = dens.w;   % 1-by-A cell of K_a x N per-attribute weight matrices

    xMinG = nan(1, A);
    xMaxG = nan(1, A);
    n0PerAttr = zeros(1, A);

    for a = 1:A
        sig = sigmaG(a);
        if isPerG(a)
            Wg = periodG(a);
        else
            Pa = double(pAttr{a});
            % Zero-weight slots contribute nothing to any live tuple, so
            % they must not enlarge the span --- otherwise the auto-prune
            % inside localCellMassesMAAbsolute (which drops zero-weight
            % perm-side tuples from the cell integrand) and manual
            % upstream pruning would discretise differently at a narrower
            % ts. The per-slot mask w{a} > 0 is the pAttr-side mirror of
            % that perm-side wJ > 0 prune, so the auto/manual invariance
            % holds at every truncation width.
            if a <= numel(wCell) && ~isempty(wCell{a})
                Wa = double(wCell{a});
                live = Wa > 0;
                cFlat = Pa(live);
            else
                cFlat = Pa(:);
            end
            if isempty(cFlat)
                cMin = 0;
                cMax = 0;
            else
                cMin = min(cFlat);
                cMax = max(cFlat);
            end
            xMinG(a) = cMin - ts * sig;
            xMaxG(a) = cMax + ts * sig;
            Wg = xMaxG(a) - xMinG(a);
        end
        n0PerAttr(a) = max(4, ceil(2.0 * Wg / sig));
    end

    perAxisW = [];
    perAxisPer = [];
    for a = 1:A
        if isPerG(a)
            Wa = periodG(a);
        else
            Wa = xMaxG(a) - xMinG(a);
        end
        for j = 1:dimPer(a)
            perAxisW(end+1) = Wa; %#ok<AGROW>
            perAxisPer(end+1) = isPerG(a); %#ok<AGROW>
        end
    end
    dim = sum(dimPer);
    if isempty(n0PerAttr)
        n0 = 4;
    else
        n0 = max(n0PerAttr);
    end
end


% =========================================================================
%  localEntropyRenyi2Dispatch — input-form resolution for Rényi-2
% =========================================================================

function H = localEntropyRenyi2Dispatch(posArgs, nvArgs)
%LOCALENTROPYRENYI2DISPATCH  Resolve input form and route to single multiset / MA helper.
%
%   Analytical Rényi-2 (collision) entropy via the orbit-Möbius
%   inner-product machinery. Restricted to single-density input
%   (scalar density struct, raw scalar single multiset, or raw scalar MA). List
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
            ['method=''renyi2'' does not yet support raw single-attribute batched ' ...
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
            case 'MaetDensity'
                localRaiseIfAnySigmaZero(firstArg, 'renyi2');
                if internal.isSingleMultiset(firstArg)
                    H = localRenyi2SingleMultiset(firstArg, base);
                    H = localAnisoEntropyCorrection(H, firstArg, base);
                else
                    H = localRenyi2MA(firstArg, base);
                    H = localAnisoEntropyCorrection(H, firstArg, base);
                end
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
        if nPos ~= 7
            error('entropyExpTens:wrongArgCountMA', ...
                ['Multi-attribute raw call expects 7 or 8 positional ' ...
                 'arguments (pAttr, w, sigmaVec, rVec, ' ...
                 'isRelVec, isPerVec, periodVec[, isSymVec]); got %d.'], nPos);
        end
        symArgs = localSymArgs(nvArgs);
        dens = buildExpTens(posArgs{1}, posArgs{2}, posArgs{3}, posArgs{4}, ...
                            posArgs{5}, posArgs{6}, posArgs{7}, symArgs{:}, ...
                            'verbose', false);
        localRaiseIfAnySigmaZero(dens, 'renyi2');
        H = localRenyi2MA(dens, base);
        H = localAnisoEntropyCorrection(H, dens, base);
        return;
    end

    % --- single multiset raw args ---
    if nPos ~= 7
        error('entropyExpTens:wrongArgCountSingleMultiset', ...
            ['Single-attribute raw call expects 7 or 8 positional arguments ' ...
             '(p, w, sigma, r, isRel, isPer, period[, isSym]); got %d.'], nPos);
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
        if internal.isKernelCov(sigma)
            error('mpt:aniso:spectrumUnsupported', ...
                  ['''spectrum'' is not supported with a matrix-valued ' ...
                   'kernel covariance (spectral augmentation changes ' ...
                   'the multiset size, breaking r == K).']);
        end
        if ~iscell(nvArgs.spectrum)
            error('entropyExpTens:badSpectrum', ...
                  '''spectrum'' value must be a cell array of addSpectra arguments.');
        end
        [p, w] = addSpectra(p, w, nvArgs.spectrum{:});
    end

    % buildExpTens is cheap in lazy mode; we only read cheap fields.
    symArgs = localSymArgs(nvArgs);
    maet = buildExpTens( ...
        p, w, sigma, r, isRel, isPer, period, symArgs{:}, ...
        'verbose', false);
    localRaiseIfAnySigmaZero(maet, 'renyi2');
    H = localRenyi2SingleMultiset(maet, base);
    H = localAnisoEntropyCorrection(H, maet, base);
end


function H = localRenyi2Finalise(ip_xx, Z, base)
%LOCALRENYI2FINALISE  Return H_2 = -log_b(ip_xx / Z^2), or NaN for a
%degenerate (zero-mass / non-finite) density.
%
%   A zero-mass density (every weight zero, or a window with no event in
%   support) has <T,T> = 0 and Z = 0 exactly, so its collision entropy is
%   undefined. Returning NaN rather than erroring is friendlier for
%   sweep-style callers: a windowed sweep already wants NaN at
%   out-of-support centres, and the caller need not wrap each evaluation in
%   try/catch. (Finite-precision catastrophic cancellation in a self-inner-
%   product is not produced analytically and has not been observed; were a
%   genuine cancellation regime ever to surface it would also land here as
%   NaN rather than a wrong number.)
    if ~isfinite(ip_xx) || ip_xx <= 0 || ~isfinite(Z) || Z <= 0
        H = NaN;
        return;
    end
    H = -log(ip_xx / (Z * Z)) / log(base);
end


function H = localAnisoEntropyCorrection(H, dens, base)
%LOCALANISOENTROPYCORRECTION  Change-of-variables constant for whitened
%densities.
%
%   A density built with a matrix-valued kernel covariance stores its
%   values in whitened coordinates, so its continuous entropies are
%   those of the whitened density; the entropy in the original
%   coordinates adds (1/2) log det(Sigma), the change-of-variables
%   constant of the linear whitening map. No-op for ordinary densities
%   and NaN-transparent.

    if internal.densityHasKernelCov(dens)
        H = H + 0.5 * internal.densityLogdetSum(dens) / log(base);
    end
end


function H = localRenyi2SingleMultiset(maet, base)
%LOCALRENYI2SINGLEMULTISET  Analytical Rényi-2 entropy of the single-multiset corner.
%
%   Computes H_2 = -log_b(<T,T> / Z^2) where <T,T> is evaluated via
%   the orbit-Möbius inner product machinery (or a direct pairwise
%   formula at r=1 where the orbit table is undefined) and
%   Z = integral T(x) dx via the closed-form total-mass formulae in
%   the +mobius package.

    dens = internal.singleMultisetView(internal.prunedExpTens(maet));
    p = dens.p; w = dens.w;
    sigma = dens.sigma; r = dens.r;
    isRel = dens.isRel; isPer = dens.isPer; period = dens.period;

    % An ordered (isSym = false) density at r > 1 has no orbit, so the
    % Möbius collision inner product (which presumes symmetrisation) does
    % not apply. Compute it numerically via the direct double sum of
    % Gaussian overlaps over the C(K, r) ordered tuples, reusing the
    % shared per-attribute machinery on a single-attribute, single-event
    % density. r = 1 is exempt ([sym] vacuous; ordered and symmetric
    % coincide) and falls through to the closed-form path below.
    if isfield(dens, 'isSym') && ~all(logical(dens.isSym)) && r > 1
        da = buildExpTens({p(:)}, {w(:)}, sigma, r, isRel, isPer, period, ...
                          false, 'lazy', false, 'verbose', false);
        [I_a, Z_a] = localRenyi2PerAttrNumerical(da, 1);
        H = localRenyi2Finalise(I_a(1, 1), Z_a(1), base);
        return;
    end

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
        %   <T,T> = sigma*sqrt(pi) * sum_{i,j} w_i w_j K(p_i - p_j)
        % where K is the 1-D overlap kernel: exp(-d^2/(4 sigma^2)) in
        % single-image mode, or its wrapped-Gaussian counterpart
        % theta(d) in full-image mode. The (sigma sqrt(pi)) prefactor
        % is the 1-D Gaussian overlap normaliser and is the same in
        % both measures (the 1-D wrapped Gaussian integrates to
        % sigma sqrt(pi) over the circle, matching the line integral
        % of the single Gaussian).
        internal.maybeShowDispatchMsg('entropyExpTens', 'pairwise', ...
            sprintf('renyi2, r=1 abs (direct pairwise sum)'));
        p = p(:); w = w(:);
        diffs = p - p.';
        wrap = 'full-image';
        if isfield(dens, 'wrap') && ~isempty(dens.wrap)
            if iscell(dens.wrap)
                wrap = char(dens.wrap{1});
            else
                wrap = char(dens.wrap);
            end
        end
        if isPer && strcmp(wrap, 'full-image')
            ts = internal.accuracyFloor('resolve', []);
            K = internal.wrappedGaussian1d(diffs, sigma, period, ts, 4);
        else
            if isPer
                diffs = diffs - period * floor(diffs / period + 0.5);
            end
            K = exp(-(diffs.^2) / (4 * sigma^2));
        end
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
            sprintf('renyi2, r=%d (orbit-Möbius IP)', r));
        if isRel
            ip_xx = mobius.orbitInnerRelSingleMultiset(p, w, p, w, sigma, r, isPer, period);
            Z = mobius.totalMassRel(p, w, sigma, r);
        else
            % dens here may be a single-multiset view (wrap as bare
            % char) or an MA density (wrap as cell); handle both.
            wrapA = 'full-image';
            if isfield(dens, 'wrap') && ~isempty(dens.wrap)
                if iscell(dens.wrap)
                    wrapA = char(dens.wrap{1});
                else
                    wrapA = char(dens.wrap);
                end
            end
            ip_xx = mobius.orbitInnerAbsSingleMultiset(p, w, p, w, sigma, r, ...
                isPer, period, 'wrap', wrapA);
            Z = mobius.totalMassAbs(p, w, sigma, r);
        end
    end

    H = localRenyi2Finalise(ip_xx, Z, base);
end


function H = localRenyi2MA(dens, base)
%LOCALRENYI2MA  Analytical Rényi-2 entropy of an MA expectation tensor.
%
%   Uses the per-attribute orbit IP factorisation
%       <T,T> = sum_{n,m} prod_a I_a[n,m]
%   with the per-attribute matrix coming from mobius.maPerAttrInnerMatrix
%   (the same machinery cosSimExpTens uses), and
%       Z = sum_n prod_a Z_a^{(n)}
%   where each Z_a^{(n)} is the closed-form single multiset total mass evaluated on
%   event n's attribute-a slot pitches and weights.
%
%   Windowed densities are not supported on this path.

    if strcmp(dens.tag, 'WindowedMaetDensity')
        error('entropyExpTens:renyi2WindowedNotSupported', ...
            ['method=''renyi2'' is not yet implemented for ' ...
             'WindowedMaetDensity.']);
    end

    dens = internal.prunedExpTens(dens);
    A = dens.nAttrs;
    N = dens.N;
    if A == 0
        H = 0;
        return;
    end
    if N == 0
        % Every event pruned away: a zero-mass density (e.g. a windowed
        % sweep centre with no event in support). Collision entropy is
        % undefined; return NaN rather than 0, matching the single multiset path and
        % the value a windowed sweep wants at out-of-support centres.
        H = NaN;
        return;
    end

    % Per-attribute inner matrices compose as
    % <T,T> = sum_{n,m} prod_a I_a[n,m] and Z = sum_n prod_a Z_a^(n).
    % Symmetric flat attributes take the Möbius per-attribute matrix and
    % closed-form total mass (the fast path; orbit-collapse assumes
    % symmetrisation). Nested attributes, and ordered (isSym = false) flat
    % attributes at r > 1, take the numerical inner matrix
    % (localRenyi2PerAttrNumerical): an ordered attribute has no orbit, so
    % its tuples are summed directly. r = 1 flat attributes are symmetric-
    % equivalent ([sym] vacuous) and stay on the Möbius path.
    nested = {};
    if isfield(dens, 'nested'); nested = dens.nested; end
    isNested = false(1, A);
    for a = 1:A
        if numel(nested) >= a && ~isempty(nested{a}) && isstruct(nested{a}) ...
                && isfield(nested{a}, 'tags')
            isNested(a) = true;
        end
    end
    if isfield(dens, 'isSym')
        isSymVec = logical(dens.isSym(:).');
    else
        isSymVec = true(1, A);
    end
    rVec = dens.r(:).';

    internal.maybeShowDispatchMsg('entropyExpTens', 'mobius', ...
        sprintf('renyi2 MA, A=%d (per-attribute orbit IP)', A));

    % --- <T, T> and Z, per attribute ---
    % Per-(n,m) cancellation ratios were shown empirically to fire
    % spuriously for self-IPs in typical musical regimes; we rely on a
    % post-hoc finite/positive check rather than a ratio fallback.
    P_xx = ones(N, N);
    Z_per_event_attr = zeros(N, A);
    for a = 1:A
        orderedFlat = ~isNested(a) && ~isSymVec(a) && (rVec(a) > 1);
        if isNested(a) || orderedFlat
            [I_xx, Z_a] = localRenyi2PerAttrNumerical(dens, a);
        else
            r_a = dens.r(a);
            sigma_g = dens.sigma(a);
            isRel_g = dens.isRel(a);
            isPer_g = dens.isPer(a);
            period_g = dens.period(a);
            Pa = dens.pAttr{a};
            Wa = dens.w{a};
            % Per-attribute wrap opt-in (default full-image).
            wrapA = 'full-image';
            if isfield(dens, 'wrap') && ~isempty(dens.wrap) ...
                    && a <= numel(dens.wrap)
                wrapA = char(dens.wrap{a});
            end
            I_xx = mobius.maPerAttrInnerMatrix(Pa, Wa, Pa, Wa, ...
                sigma_g, r_a, isRel_g, isPer_g, period_g, 'wrap', wrapA);
            Z_a = zeros(N, 1);
            for n = 1:N
                pn = Pa(:, n);
                wn = Wa(:, n);
                valid = ~(isnan(pn) | isnan(wn));
                pn = pn(valid);
                wn = wn(valid);
                if isRel_g
                    Z_a(n) = mobius.totalMassRel(pn, wn, sigma_g, r_a);
                else
                    Z_a(n) = mobius.totalMassAbs(pn, wn, sigma_g, r_a);
                end
            end
        end
        P_xx = P_xx .* I_xx;
        Z_per_event_attr(:, a) = Z_a;
    end
    ip_xx = sum(P_xx(:));

    Z = sum(prod(Z_per_event_attr, 2));

    H = localRenyi2Finalise(ip_xx, Z, base);
end


function [I_a, Z_a] = localRenyi2PerAttrNumerical(dens, a)
%LOCALRENYI2PERATTRNUMERICAL  Per-attribute (event, event) inner matrix and
%per-event total mass computed numerically via explicit tuple enumeration
%and the (block-diagonal) co-transposition metric. Handles both *nested*
%attributes and *ordered* (isSym = false) flat attributes at r > 1.
%
%   Returns I_a (N x N), where I_a(n,m) = integral k_a^n(x) k_a^m(x), and
%   Z_a (N x 1), where Z_a(n) = integral k_a^n. These compose with the
%   flat-symmetric Möbius matrices in the MA Rényi-2 factorisation. The
%   flat Möbius matrix presumes a single symmetric tuple and re-derives the
%   full S_{r} orbit; that orbit is wrong for an ordered attribute (no
%   symmetrisation) and, for a nested attribute, both wrong and infeasible.
%   The numerical reading builds the attribute's density, whose tuples and
%   metric are correct in either case. For two kernels of common metric M
%   and width sigma the Gaussian overlap is
%   (pi sigma^2)^{d/2}/sqrt(det M) * exp(-Q_M(c_t - c_s)/(4 sigma^2)) and
%   the single-kernel mass is (2 pi sigma^2)^{d/2}/sqrt(det M).
    sig  = dens.sigma(a);
    isper = dens.isPer(a);
    per  = dens.period(a);
    isNestedA = isfield(dens, 'nested') && numel(dens.nested) >= a ...
        && ~isempty(dens.nested{a}) && isstruct(dens.nested{a}) ...
        && isfield(dens.nested{a}, 'tags');
    if isNestedA
        % Nested attribute: rebuild from its resolved spec.
        spec = dens.nested{a};
        da = buildExpTens({dens.pAttr{a}}, {dens.w{a}}, 'specs', {spec}, ...
                          'sigma', sig, 'isPer', isper, 'period', per, ...
                          'lazy', false, 'verbose', false);
    else
        % Flat ordered attribute: rebuild from its flat parameters with
        % isSym = false, so the materialised tuples are the C(K, r_a)
        % ordered sub-tuples (one kernel each, no orbit).
        spec = [];
        r_a0   = dens.r(a);
        isRel0 = dens.isRel(a);
        da = buildExpTens({dens.pAttr{a}}, {dens.w{a}}, sig, r_a0, ...
                          isRel0, isper, per, false, ...
                          'lazy', false, 'verbose', false);
    end
    C   = da.Centres{1};         % (d_a x nJ) reduced centres
    wj  = da.wJ(:);              % (nJ x 1)
    eoj = da.eventOfJ(:);        % (nJ x 1) 1-based event index
    d_a = size(C, 1);
    nj  = numel(wj);
    N   = dens.N;

    blockSize = 0;
    if ~isempty(spec) && isfield(spec, 'proj') ...
            && (strcmp(spec.proj, 'inner') || strcmp(spec.proj, 'intermediate'))
        u = spec.relUnit;
        blockSize = prod(spec.r(1:u));
    end
    isRel = da.isRel(1);
    r_a   = da.r(1);
    detM = internal.quadraticFormDet(r_a, blockSize, isRel);
    vol  = internal.gaussianMassConst(sig, d_a, detM);          % single-kernel mass
    pref = internal.gaussianMassConst(sig, d_a, detM, true);    % overlap prefactor

    I_a = zeros(N, N);
    Z_a = zeros(N, 1);
    if nj > 0
        % Abs-per full-image path: compute the pairwise overlap matrix
        % O directly from per-slot theta products, bypassing the Q ->
        % exp(-Q/(4 sigma^2)) formulation which is single-image. This
        % applies only to flat abs-per (blockSize < 2 and not rel);
        % other configurations use the block-diagonal quadratic form
        % below (either always full-image via pairwise wrap for rel,
        % or nested/block-metric that keeps its own semantics).
        wrapA = 'full-image';
        if isfield(dens, 'wrap') && ~isempty(dens.wrap) ...
                && a <= numel(dens.wrap)
            wrapA = char(dens.wrap{a});
        end
        useAbsPerFullImage = isper && ~isRel && blockSize < 2 ...
            && strcmp(wrapA, 'full-image');
        if useAbsPerFullImage
            D = reshape(C, d_a, nj, 1) - reshape(C, d_a, 1, nj);
            ts = internal.accuracyFloor('resolve', []);
            theta = internal.wrappedGaussian1d(D, sig, per, ts, 4);
            O = pref .* reshape(prod(theta, 1), nj, nj);
        else
            Q = localBlockMetricQ(C, blockSize, isRel, r_a, isper, per);  % nJ x nJ
            O = pref .* exp(-Q ./ (4 * sig^2));
        end
        WO = (wj * wj.') .* O;
        G = zeros(N, nj);
        G(sub2ind([N, nj], eoj.', 1:nj)) = 1;
        I_a = G * WO * G.';
        Z_a = vol .* (G * wj);
    end
end


function Q = localBlockMetricQ(C, blockSize, isRel, r_a, isPer, per)
%LOCALBLOCKMETRICQ  Pairwise block-diagonal co-transposition quadratic
%form on reduced centres. Mirrors the reduced-convention block metric used
%in evalExpTens (qInnerBlocksReducedLocal) and the whole-tuple _compute_Q,
%but operates on the (nJ x nJ) pairwise difference tensor.
    d_a = size(C, 1);
    nj  = size(C, 2);
    % D(k,i,j) = C(k,i) - C(k,j).
    D = reshape(C, d_a, nj, 1) - reshape(C, d_a, 1, nj);
    Q = zeros(nj, nj);
    if blockSize >= 2
        blk = blockSize - 1;          % reduced rows per block
        nBlocks = d_a / blk;
        for b = 1:nBlocks
            rows = (b - 1) * blk + (1:blk);
            Db = D(rows, :, :);
            if isPer
                slot0 = Db - per .* floor(Db ./ per + 0.5);
                Qb = reshape(sum(slot0 .^ 2, 1), nj, nj);
                for i = 1:blk
                    for j = i + 1:blk
                        delta = reshape(Db(i, :, :) - Db(j, :, :), nj, nj);
                        delta = delta - per .* floor(delta ./ per + 0.5);
                        Qb = Qb + delta .^ 2;
                    end
                end
                Qb = Qb / blockSize;
            else
                Qb = reshape(sum(Db .^ 2, 1), nj, nj) ...
                   - reshape(sum(Db, 1) .^ 2, nj, nj) / blockSize;
            end
            Q = Q + Qb;
        end
    elseif isRel && r_a >= 2
        % Whole-tuple reduced relative quotient (outer unit).
        if isPer
            slot0 = D - per .* floor(D ./ per + 0.5);
            Q = reshape(sum(slot0 .^ 2, 1), nj, nj);
            for i = 1:d_a
                for j = i + 1:d_a
                    delta = reshape(D(i, :, :) - D(j, :, :), nj, nj);
                    delta = delta - per .* floor(delta ./ per + 0.5);
                    Q = Q + delta .^ 2;
                end
            end
            Q = Q / r_a;
        else
            Q = reshape(sum(D .^ 2, 1), nj, nj) ...
              - reshape(sum(D, 1) .^ 2, nj, nj) / r_a;
        end
    else
        % Absolute.
        if isPer
            D = D - per .* floor(D ./ per + 0.5);
        end
        Q = reshape(sum(D .^ 2, 1), nj, nj);
    end
end

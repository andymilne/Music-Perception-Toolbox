function s = cosSimExpTens(varargin)
%COSSIMEXPTENS Cosine similarity of two r-ad expectation tensor densities.
%
%   s = cosSimExpTens(dens_x, dens_y):
%   s = cosSimExpTens(dens_x, dens_y, 'verbose', false):
%   Cosine similarity using precomputed density structs from buildExpTens.
%   This avoids recomputing tuple indices and weight products on each call,
%   and is the preferred calling convention when comparing a fixed reference
%   against many other sets.
%
%   s = cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period):
%   s = cosSimExpTens(..., 'verbose', false):
%   Cosine similarity from raw arguments (builds density structs
%   internally via buildExpTens).
%
%   Computes the cosine similarity between the r-ad expectation tensor
%   densities of two weighted multisets (p represents pitches or
%   positions), with Gaussian perception error of standard deviation
%   sigma. The cosine similarity is computed analytically — no grid
%   evaluation is required.
%
%   Four variants are available, depending on the flags isPer and isRel:
%   the inner product assumes periodic equivalence with period set by
%   'period' if isPer == true, and assumes transpositional equivalence
%   (relative rather than absolute pitches or positions) if isRel == true.
%   See buildExpTens for further information about these parameters.
%
%   Inputs (struct calling convention):
%     dens_x — Precomputed density struct from buildExpTens.
%     dens_y — Precomputed density struct from buildExpTens.
%              Both structs must share the same r, sigma, isRel, isPer,
%              and (if periodic) period.
%
%   Inputs (raw calling convention):
%     p1     — Pitch or position values for the first multiset (vector
%              of length n_1).
%     w1     — Weights for the first multiset (vector of length n_1, or
%              empty/scalar for all ones).
%     p2     — Pitch or position values for the second multiset (vector
%              of length n_2).
%     w2     — Weights for the second multiset (vector of length n_2, or
%              empty/scalar for all ones).
%     sigma  — Standard deviation of the Gaussian kernel.
%     r      — Tuple size (positive integer; r >= 2 if isRel == true).
%     isRel  — If true, use transposition-invariant (relative)
%              quadratic form.
%     isPer  — If true, wrap differences to periodic interval [-J/2, J/2).
%     period — Period J for periodic wrapping.
%
%   Optional name-value pair (all calling conventions):
%     'verbose' — Logical (default: true). If false, suppresses console
%                 output (time estimates, progress messages).
%
%   Output:
%     s      — Cosine similarity (scalar in [0, 1] for non-negative
%              weights). Returns NaN if r exceeds the number of elements
%              in either multiset.
%
%   Originally by David Bulger, Macquarie University, Australia (2016).
%   Adapted for the Music Perception Toolbox v2 by Andrew J. Milne
%   (The MARCS Institute, Western Sydney University): preallocated
%   permutation indices, vectorized inner product computation, simplified
%   quadratic form, precomputed index/pitch/weight data shared across the
%   three inner product calls, automatic chunking for large arrays, and
%   optional precomputed density structs via buildExpTens.
%
%   See also buildExpTens, evalExpTens, batchCosSimExpTens.

% === Parse arguments ===

% Extract optional 'verbose' name-value pair first
verbose = true;  % default
verboseIdx = [];
for i = 1:numel(varargin)
    if (ischar(varargin{i}) || isstring(varargin{i})) && strcmpi(varargin{i}, 'verbose')
        if i + 1 <= numel(varargin)
            verbose = logical(varargin{i + 1});
        end
        verboseIdx = [i, i + 1]; %#ok<AGROW>
        break;
    end
end
if ~isempty(verboseIdx)
    varargin(verboseIdx) = [];
end

nArgs = numel(varargin);

% --- Windowed path: at least one operand is a WindowedMaetDensity ---
if nArgs == 2 && isstruct(varargin{1}) && isstruct(varargin{2}) ...
        && isfield(varargin{1}, 'tag') && isfield(varargin{2}, 'tag') ...
        && (strcmp(varargin{1}.tag, 'WindowedMaetDensity') ...
         || strcmp(varargin{2}.tag, 'WindowedMaetDensity'))
    s = localCosSimWindowed(varargin{1}, varargin{2}, verbose);
    return;
end

% --- MA path: two MaetDensity structs ---
if nArgs == 2 && isstruct(varargin{1}) && isstruct(varargin{2}) ...
        && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'MaetDensity') ...
        && isfield(varargin{2}, 'tag') ...
        && strcmp(varargin{2}.tag, 'MaetDensity')
    s = localCosSimMA(varargin{1}, varargin{2}, verbose);
    return;
end

% --- MA path: raw args (10 positional, first is cell) ---
if nArgs == 10 && iscell(varargin{1})
    if ~iscell(varargin{3})
        error(['cosSimExpTens: p1 is a cell (multi-attribute) but p2 is ' ...
            'not. Both must be the same kind: either both cells (MA) ' ...
            'or both numeric vectors (SA).']);
    end
    pAttr1    = varargin{1};
    w1        = varargin{2};
    pAttr2    = varargin{3};
    w2        = varargin{4};
    sigmaVec  = varargin{5};
    rVec      = varargin{6};
    groups    = varargin{7};
    isRelVec  = varargin{8};
    isPerVec  = varargin{9};
    periodVec = varargin{10};
    dens_x = buildExpTens(pAttr1, w1, sigmaVec, rVec, groups, ...
        isRelVec, isPerVec, periodVec, 'verbose', verbose);
    dens_y = buildExpTens(pAttr2, w2, sigmaVec, rVec, groups, ...
        isRelVec, isPerVec, periodVec, 'verbose', verbose);
    s = localCosSimMA(dens_x, dens_y, verbose);
    return;
end

if nArgs == 2 && isstruct(varargin{1}) && isstruct(varargin{2}) ...
        && isfield(varargin{1}, 'tag') ...
        && strcmp(varargin{1}.tag, 'ExpTensDensity') ...
        && isfield(varargin{2}, 'tag') ...
        && strcmp(varargin{2}.tag, 'ExpTensDensity')
    % --- Precomputed structs ---
    dens_x = varargin{1};
    dens_y = varargin{2};

    % Validate that both structs share compatible parameters
    if dens_x.r ~= dens_y.r
        error('Both density structs must have the same r.');
    end
    if dens_x.isRel ~= dens_y.isRel
        error('Both density structs must have the same isRel.');
    end
    if dens_x.isPer ~= dens_y.isPer
        error('Both density structs must have the same isPer.');
    end
    if dens_x.isPer && dens_x.period ~= dens_y.period
        error('Both density structs must have the same period.');
    end
    if dens_x.sigma ~= dens_y.sigma
        error('Both density structs must have the same sigma.');
    end

    r     = dens_x.r;
    sigma = dens_x.sigma;
    isRel = dens_x.isRel;
    isPer = dens_x.isPer;
    J     = dens_x.period;

    Ux_perm  = dens_x.U_perm;
    wx_perm  = dens_x.w_perm;
    nJx      = dens_x.nJ_perm;
    Vx_comb  = dens_x.V_comb;
    wvx_comb = dens_x.wv_comb;
    nKx      = dens_x.nK;

    Uy_perm  = dens_y.U_perm;
    wy_perm  = dens_y.w_perm;
    nJy      = dens_y.nJ_perm;
    Vy_comb  = dens_y.V_comb;
    wvy_comb = dens_y.wv_comb;
    nKy      = dens_y.nK;

elseif nArgs == 9
    % --- Raw arguments (SA) ---
    if iscell(varargin{1}) || iscell(varargin{3})
        error(['cosSimExpTens: cell-form p1/p2 (multi-attribute) requires ' ...
            '10 positional arguments: pAttr1, w1, pAttr2, w2, sigmaVec, ' ...
            'rVec, groups, isRelVec, isPerVec, periodVec.']);
    end
    p1     = varargin{1};
    w1     = varargin{2};
    p2     = varargin{3};
    w2     = varargin{4};
    sigma  = varargin{5};
    r      = varargin{6};
    isRel  = varargin{7};
    isPer  = varargin{8};
    J      = varargin{9};

    % Build density structs on the fly
    dens_x = buildExpTens(p1, w1, sigma, r, isRel, isPer, J, 'verbose', verbose);
    dens_y = buildExpTens(p2, w2, sigma, r, isRel, isPer, J, 'verbose', verbose);

    Ux_perm  = dens_x.U_perm;
    wx_perm  = dens_x.w_perm;
    nJx      = dens_x.nJ_perm;
    Vx_comb  = dens_x.V_comb;
    wvx_comb = dens_x.wv_comb;
    nKx      = dens_x.nK;

    Uy_perm  = dens_y.U_perm;
    wy_perm  = dens_y.w_perm;
    nJy      = dens_y.nJ_perm;
    Vy_comb  = dens_y.V_comb;
    wvy_comb = dens_y.wv_comb;
    nKy      = dens_y.nK;

else
    error(['Usage:\n' ...
        '  SA struct:   cosSimExpTens(dens_x, dens_y [, ''verbose'', tf])\n' ...
        '  SA raw args: cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period [, ''verbose'', tf])\n' ...
        '  MA struct:   cosSimExpTens(densMA_x, densMA_y [, ''verbose'', tf])\n' ...
        '  MA raw args: cosSimExpTens(pAttr1, w1, pAttr2, w2, sigmaVec, rVec, groups, isRelVec, isPerVec, periodVec [, ''verbose'', tf]).']);
end

% === Early return for degenerate case ===
if r > min(numel(dens_x.p), numel(dens_y.p))
    s = NaN;
    return;
end

% === Cosine similarity from three inner products ===

% Estimated computation time for all three inner products:
% Total pairs = nJx*nKy + nJx*nKx + nJy*nKy
totalPairs = double(nJx)*double(nKy) + double(nJx)*double(nKx) ...
           + double(nJy)*double(nKy);
estimateCompTime(totalPairs, r, 'cosSimExpTens', verbose);

ip_xy = ipCore(Ux_perm, wx_perm, nJx, Vy_comb, wvy_comb, nKy);
ip_xx = ipCore(Ux_perm, wx_perm, nJx, Vx_comb, wvx_comb, nKx);
ip_yy = ipCore(Uy_perm, wy_perm, nJy, Vy_comb, wvy_comb, nKy);

s = ip_xy / sqrt(ip_xx * ip_yy);


% =====================================================================
%  NESTED HELPER FUNCTIONS
%  (r, sigma, J, isPer, isRel are in scope from the parent workspace.)
% =====================================================================

    % -----------------------------------------------------------------
    %  ipCore
    %  Core inner product between one perm-side (U, wU) and one
    %  comb-side (V, wV). Dispatches to the fully vectorized fast path
    %  if the intermediate 3D array fits in memory; otherwise processes
    %  in chunks along the comb-side dimension.
    % -----------------------------------------------------------------
    function ipval = ipCore(U, wU, nJ, V, wV, nK)
        bytesNeeded = (r + 2) * double(nJ) * double(nK) * 8;

        try
            memInfo  = memory;
            memLimit = memInfo.MaxPossibleArrayBytes * 0.5;
        catch
            memLimit = 4e9;
        end

        if bytesNeeded <= memLimit
            ipval = ipFull(U, wU, nJ, V, wV, nK);
        else
            chunkSize = max(1, ...
                floor(memLimit / ((r + 2) * double(nJ) * 8)));

            acc = zeros(nJ, 1);
            for c = 1:chunkSize:nK
                cEnd = min(c + chunkSize - 1, nK);
                idx  = c:cEnd;
                nKc  = numel(idx);

                Dc = reshape(U, r, nJ, 1) ...
                   - reshape(V(:, idx), r, 1, nKc);

                if isPer
                    Dc = mod(Dc + J/2, J) - J/2;
                end

                Qc = computeQ(Dc);

                Ec = reshape(exp(-Qc(:) / (4 * sigma^2)), nJ, nKc);
                acc = acc + Ec * wV(idx)';
            end

            ipval = wU(:)' * acc;
        end
    end

    % -----------------------------------------------------------------
    %  ipFull
    %  Fully vectorized inner product (no chunking).
    %
    %  The computation is:
    %    ip = sum_{j,k} prod(w_j) * prod(w_k) * exp(-Q_jk / (4*s^2))
    %
    %  The quadratic form Q depends on the mode:
    %    Absolute: Q(d) = sum(d.^2)
    %    Relative (non-periodic): Q(d) = sum(d.^2) - sum(d)^2 / r
    %    Relative + periodic: Q(d) = sum_{i<j} wrap(d_i - d_j)^2 / r
    %
    %  The relative quadratic form (induced by the Riemannian metric on
    %  the quotient space R^r / R*1) projects out the mean, yielding
    %  transpositional equivalence. In the periodic case, the pairwise
    %  differences between wrapped components are themselves wrapped to
    %  [-J/2, J/2), which restores exact transposition invariance on the
    %  circle.
    %
    %  The two relative formulas are algebraically identical in the
    %  non-periodic case: sum_{i<j} (d_i - d_j)^2 == r*(sum(d^2) -
    %  sum(d)^2/r).
    %
    %  The weighted sum is computed as wU' * (E * wV), avoiding the full
    %  outer-product weight matrix.
    % -----------------------------------------------------------------
    function ipval = ipFull(U, wU, nJ, V, wV, nK)
        D = reshape(U, r, nJ, 1) - reshape(V, r, 1, nK);

        if isPer
            D = mod(D + J/2, J) - J/2;
        end

        Qvec = computeQ(D);

        E = reshape(exp(-Qvec(:) / (4 * sigma^2)), nJ, nK);
        ipval = wU(:)' * (E * wV(:));
    end

    % -----------------------------------------------------------------
    %  computeQ
    %  Compute the quadratic form Q from the (already-wrapped)
    %  difference array D (r x nJ x nK or r x nJ x nKc).
    %
    %  When isPer && isRel, pairwise differences between components of
    %  D are wrapped to [-J/2, J/2) before squaring. This restores
    %  exact transposition invariance on the circle, which is otherwise
    %  broken by component-wise wrapping. The two formulas are
    %  algebraically identical in the non-periodic case:
    %     sum_{i<j} (d_i - d_j)^2 == r * (sum(d^2) - sum(d)^2/r).
    % -----------------------------------------------------------------
    function Qvec = computeQ(D)
        if isRel
            if isPer
                Qvec = zeros(1, size(D, 2), size(D, 3));
                for i = 1:r
                    for j = i+1:r
                        delta = D(i,:,:) - D(j,:,:);
                        delta = mod(delta + J/2, J) - J/2;
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
    end

end

% =========================================================================
%  localCosSimMA — multi-attribute (MAET) cosine similarity
% =========================================================================


function s = localCosSimMA(dens_x, dens_y, verbose)
%LOCALCOSSIMMA  Cosine similarity between two MaetDensities.
%
%   The inner product factors as an elementwise product of per-attribute
%   kernels (Section 2.7 of the MAET specification); no numerical
%   integration is required.
%
%   Both densities must share the full parameter structure: number of
%   attributes, group assignment, per-attribute r, and per-group sigma,
%   isRel, isPer, period. Weights and event/slot counts may differ.

    % --- Structural compatibility ---
    if dens_x.nAttrs ~= dens_y.nAttrs
        error('cosSimExpTens:nAttrsMismatch', ...
            'Both MaetDensities must have the same nAttrs.');
    end
    if ~isequal(dens_x.groupOfAttr, dens_y.groupOfAttr)
        error('cosSimExpTens:groupsMismatch', ...
            'Both MaetDensities must have the same groupOfAttr.');
    end
    if ~isequal(dens_x.r, dens_y.r)
        error('cosSimExpTens:rMismatch', ...
            'Both MaetDensities must have the same r (per attribute).');
    end
    if ~isequal(dens_x.sigma, dens_y.sigma)
        error('cosSimExpTens:sigmaMismatch', ...
            'Both MaetDensities must have the same sigma (per group).');
    end
    if ~isequal(logical(dens_x.isRel), logical(dens_y.isRel))
        error('cosSimExpTens:isRelMismatch', ...
            'Both MaetDensities must have the same isRel (per group).');
    end
    if ~isequal(logical(dens_x.isPer), logical(dens_y.isPer))
        error('cosSimExpTens:isPerMismatch', ...
            'Both MaetDensities must have the same isPer (per group).');
    end
    perMask = logical(dens_x.isPer);
    if any(dens_x.period(perMask) ~= dens_y.period(perMask))
        error('cosSimExpTens:periodMismatch', ...
            'Both MaetDensities must have the same period for periodic groups.');
    end

    % --- Unpack shared parameters (scope for nested helpers) ---
    A        = dens_x.nAttrs;
    groupOf  = dens_x.groupOfAttr;
    rVec     = dens_x.r;
    sigmaG   = dens_x.sigma;
    isRelG   = logical(dens_x.isRel);
    isPerG   = logical(dens_x.isPer);
    periodG  = dens_x.period;

    Ux_perm  = dens_x.U_perm;
    wx_perm  = dens_x.wJ;
    nJx      = dens_x.nJ;
    Vx_comb  = dens_x.V_comb;
    wvx_comb = dens_x.wv_comb;
    nKx      = dens_x.nK;

    Uy_perm  = dens_y.U_perm;
    wy_perm  = dens_y.wJ;
    nJy      = dens_y.nJ;
    Vy_comb  = dens_y.V_comb;
    wvy_comb = dens_y.wv_comb;
    nKy      = dens_y.nK;

    % --- Three inner products ---
    totalPairs = double(nJx)*double(nKy) + double(nJx)*double(nKx) ...
               + double(nJy)*double(nKy);
    maxR = max(rVec);
    estimateCompTime(totalPairs, maxR, 'cosSimExpTens (MAET)', verbose);

    ip_xy = ipCoreMA(Ux_perm, wx_perm, nJx, Vy_comb, wvy_comb, nKy);
    ip_xx = ipCoreMA(Ux_perm, wx_perm, nJx, Vx_comb, wvx_comb, nKx);
    ip_yy = ipCoreMA(Uy_perm, wy_perm, nJy, Vy_comb, wvy_comb, nKy);

    denom = sqrt(ip_xx * ip_yy);
    if denom == 0
        s = NaN;
    else
        s = ip_xy / denom;
    end

    % =====================================================================
    %  Nested helpers (rVec, sigmaG, isRelG, isPerG, periodG, groupOf, A
    %  are in scope from the parent).
    % =====================================================================

    function ipval = ipCoreMA(U_cell, wU, nJ, V_cell, wV, nK)
        % Memory-aware chunking along the comb-side (nK) dimension.
        maxRa = double(max(rVec));
        bytesNeeded = (maxRa + 2) * double(nJ) * double(nK) * 8;

        try
            memInfo  = memory;
            memLimit = memInfo.MaxPossibleArrayBytes * 0.5;
        catch
            memLimit = 4e9;
        end

        if bytesNeeded <= memLimit
            ipval = ipFullMA(U_cell, wU, nJ, V_cell, wV, nK);
        else
            chunkSize = max(1, floor(memLimit / ((maxRa + 2) * double(nJ) * 8)));
            acc = zeros(nJ, 1);
            for c = 1:chunkSize:nK
                cEnd = min(c + chunkSize - 1, nK);
                idx  = c:cEnd;
                nKc  = numel(idx);

                V_chunk = cell(1, A);
                for a = 1:A
                    V_chunk{a} = V_cell{a}(:, idx);
                end

                logK = maLogKernel(U_cell, V_chunk, nJ, nKc);
                Ec = exp(logK);
                acc = acc + Ec * wV(idx).';
            end
            ipval = wU(:).' * acc;
        end
    end

    function ipval = ipFullMA(U_cell, wU, nJ, V_cell, wV, nK)
        logK = maLogKernel(U_cell, V_cell, nJ, nK);
        E = exp(logK);                    % nJ x nK
        ipval = wU(:).' * (E * wV(:));
    end

    function logK = maLogKernel(U_cell, V_cell, nJ, nK)
        % Accumulate sum_a -Q_a / (4 sigma_g^2) over attributes.
        logK = zeros(nJ, nK);
        for a = 1:A
            g = groupOf(a);
            r_a = rVec(a);
            D = reshape(U_cell{a}, r_a, nJ, 1) ...
              - reshape(V_cell{a}, r_a, 1, nK);

            if isPerG(g)
                P_g = periodG(g);
                D = mod(D + P_g/2, P_g) - P_g/2;
            end

            Qa = computeQaMA(D, g, r_a);
            logK = logK - reshape(Qa, nJ, nK) / (4 * sigmaG(g)^2);
        end
    end

    function Qa = computeQaMA(D, g, r_a)
        % Per-attribute quadratic form. Matches the SA computeQ logic:
        %   - is_rel && is_per: pairwise-differences formula (wraps
        %     each pairwise delta to [-P/2, P/2), restores exact
        %     transposition invariance on the circle).
        %   - is_rel && ~is_per: sum(d.^2) - sum(d)^2 / r_a.
        %   - ~is_rel:           sum(d.^2).
        if isRelG(g)
            if isPerG(g)
                sz = size(D);
                if numel(sz) < 3, sz = [sz, 1]; end
                Qa = zeros(1, sz(2), sz(3));
                P_g = periodG(g);
                for i = 1:r_a
                    for j = i+1:r_a
                        delta = D(i, :, :) - D(j, :, :);
                        delta = mod(delta + P_g/2, P_g) - P_g/2;
                        Qa = Qa + delta.^2;
                    end
                end
                Qa = Qa / r_a;
            else
                Qa = sum(D.^2, 1) - sum(D, 1).^2 / r_a;
            end
        else
            Qa = sum(D.^2, 1);
        end
    end

end


% =========================================================================
%  localCosSimWindowed — cosine similarity with one operand windowed
% =========================================================================

function s = localCosSimWindowed(a, b, verbose)
%LOCALCOSSIMWINDOWED  Cosine similarity when at least one operand is a
%WindowedMaetDensity.
%
%   Option Z normalisation: numerator is the windowed inner product
%   <W_a f_a, W_b f_b>, denominator is the product of UNWINDOWED L2
%   norms. This gives a magnitude-aware profile where silent regions of
%   the context produce zeros and matching content near the window
%   centre produces values proportional to how much content is there.
%
%   One-sided windowing only: exactly one of a, b must be a
%   WindowedMaetDensity. Two-sided windowing is not supported in v2.1.0.

    a_win = strcmp(a.tag, 'WindowedMaetDensity');
    b_win = strcmp(b.tag, 'WindowedMaetDensity');
    if a_win && b_win
        error('cosSimExpTens:twoSidedWindowing', ...
              ['Two-sided windowing (both operands windowed) is not ' ...
               'supported in v2.1.0. Use windowedSimilarity for profile sweeps.']);
    end

    % Canonicalise: put the windowed operand on the 'c' (context) side.
    if a_win
        dens_q = b;
        wmd    = a;
    else
        dens_q = a;
        wmd    = b;
    end
    dens_c = wmd.dens;

    if ~(strcmp(dens_q.tag, 'MaetDensity'))
        error('cosSimExpTens:windowedBadOperand', ...
              'Windowed density can only be compared with a MaetDensity.');
    end

    % Structural compatibility checks.
    localCheckMACompat(dens_q, dens_c);

    % --- Unwindowed norms (denominator) ---
    ip_qq = localCosSimNumeratorMA(dens_q, dens_q, [], verbose);
    ip_cc = localCosSimNumeratorMA(dens_c, dens_c, [], verbose);

    % --- Windowed numerator ---
    ip_qc = localCosSimNumeratorMA(dens_q, dens_c, wmd, verbose);

    denom = sqrt(ip_qq * ip_cc);
    if denom == 0
        s = NaN;
    else
        s = ip_qc / denom;
    end
end


function localCheckMACompat(dx, dy)
    if dx.nAttrs ~= dy.nAttrs
        error('cosSimExpTens:nAttrsMismatch', ...
              'Both densities must have the same nAttrs.');
    end
    if ~isequal(dx.groupOfAttr, dy.groupOfAttr)
        error('cosSimExpTens:groupsMismatch', ...
              'Both densities must have the same groupOfAttr.');
    end
    if ~isequal(dx.r, dy.r)
        error('cosSimExpTens:rMismatch', ...
              'Both densities must have the same r.');
    end
    if ~isequal(dx.sigma, dy.sigma)
        error('cosSimExpTens:sigmaMismatch', ...
              'Both densities must have the same sigma.');
    end
    if ~isequal(logical(dx.isRel), logical(dy.isRel))
        error('cosSimExpTens:isRelMismatch', ...
              'Both densities must have the same isRel.');
    end
    if ~isequal(logical(dx.isPer), logical(dy.isPer))
        error('cosSimExpTens:isPerMismatch', ...
              'Both densities must have the same isPer.');
    end
    perMask = logical(dx.isPer);
    if any(dx.period(perMask) ~= dy.period(perMask))
        error('cosSimExpTens:periodMismatch', ...
              'Both densities must agree on periods of periodic groups.');
    end
end


function ip = localCosSimNumeratorMA(dx, dy, wmd, ~)
%LOCALCOSSIMNUMERATORMA  Compute sum_{j,k} w_j^x w_k^y * prod_g (factor),
%where the factor is the unwindowed U_g by default, or U_g * F_g for
%windowed groups when wmd is non-empty.
%
%   When ``wmd`` is non-empty, the inner product is interpreted as a
%   CROSS-CORRELATION between the (unwindowed) query ``dx`` and the
%   windowed context ``dy``: at the window centre ``c_g`` in each
%   windowed group, the query is translated so that its effective-space
%   mean ``mu_q_g`` moves onto ``c_g``. A peak at ``c_g`` thus means
%   the query pattern is present in the context near ``c_g``.
%
%   Mathematically this is implemented by the coordinate substitution
%       cx_g  ->  cx_g  - mu_q_g
%       cy_g  ->  cy_g  - c_g
%       c_g   ->  0
%   inside the windowed-factor integrand (so the closed-form helper
%   ``localWindowedContribution`` is reused verbatim), and by adding a
%   per-attribute tuple-space shift
%       delta_a = { (c_g - mu_q_g)|_a            if group g is absolute
%                 { [0, (c_g - mu_q_g)|_a_eff]   if group g is relative
%                                                 (slot-0-anchored lift)
%   to D = U - V before computing Q_a. Groups that are not windowed
%   receive no shift.
%
%   For the unwindowed call (``wmd`` empty) the block is skipped and
%   ``cos_sim_exp_tens`` semantics are preserved exactly.

    A         = dx.nAttrs;
    groupOf   = dx.groupOfAttr;
    rVec      = dx.r;
    sigmaG    = dx.sigma;
    isRelG    = logical(dx.isRel);
    isPerG    = logical(dx.isPer);
    periodG   = dx.period;

    n_jx = dx.nJ;
    n_ky = dy.nK;
    nGroups = dx.nGroups;
    attrsOfGroup = dx.attrsOfGroup;
    dimPerAttr = dx.dimPerAttr;

    % --- Pre-compute cross-correlation shifts (windowed path only) ---
    % shiftPerAttr{a} : (r_a x 1) shift added to D for attribute a.
    %                   Empty if attribute a is in a non-windowed group.
    % muQperG{g}      : (d_g x 1) effective-space query mean, windowed g.
    shiftPerAttr = cell(1, A);
    muQperG = cell(1, nGroups);
    if ~isempty(wmd)
        for g = 1:nGroups
            if ~localIsWindowedGroupG(wmd.size(g), wmd.mix(g))
                continue;
            end
            attrs_g = attrsOfGroup{g};
            % Query effective-space mean in group g: average over perm
            % rows, concatenated across attributes.
            mu_parts = cell(1, numel(attrs_g));
            centre_parts = cell(1, numel(attrs_g));
            for ia = 1:numel(attrs_g)
                a = attrs_g(ia);
                mu_parts{ia} = mean(dx.Centres{a}, 2);   % (d_a x 1)
                centre_parts{ia} = wmd.centre{a}(:);     % (d_a x 1)
            end
            mu_q_g = vertcat(mu_parts{:});       % (d_g x 1)
            centre_g = vertcat(centre_parts{:}); % (d_g x 1)
            delta_g = centre_g - mu_q_g;         % (d_g x 1)
            muQperG{g} = mu_q_g;

            % Lift delta_g into per-attribute r_a-slot shifts.
            offset = 0;
            g_is_rel = isRelG(g);
            for ia = 1:numel(attrs_g)
                a = attrs_g(ia);
                r_a = rVec(a);
                d_a = dimPerAttr(a);
                delta_a_eff = delta_g(offset + 1 : offset + d_a);  % (d_a x 1)
                offset = offset + d_a;
                if g_is_rel
                    if r_a == 1
                        shift_a = zeros(1, 1);
                    else
                        % Slot-0 anchored lift: first slot = 0, remaining
                        % r_a - 1 slots = effective shift.
                        shift_a = [0; delta_a_eff(:)];
                    end
                else
                    % Absolute: effective dim == r_a, direct mapping.
                    shift_a = delta_a_eff(:);
                end
                if numel(shift_a) ~= r_a
                    error('localCosSimNumeratorMA:shiftShape', ...
                          'Internal: shift for attribute %d has size %d, expected %d.', ...
                          a, numel(shift_a), r_a);
                end
                shiftPerAttr{a} = shift_a;
            end
        end
    end

    % --- Base unwindowed log-kernel: sum_g -Q_g / (4 sigma_g^2) ---
    log_kernel = zeros(n_jx, n_ky);
    for a = 1:A
        g = groupOf(a);
        r_a = rVec(a);
        Ua = dx.U_perm{a};
        Va = dy.V_comb{a};
        D = reshape(Ua, r_a, n_jx, 1) - reshape(Va, r_a, 1, n_ky);

        % Apply per-attribute cross-correlation shift before wrap / Q_a.
        if ~isempty(shiftPerAttr{a})
            D = D + shiftPerAttr{a};   % broadcasts over (nJ, nK)
        end

        if isPerG(g)
            P_g = periodG(g);
            D = mod(D + P_g/2, P_g) - P_g/2;
        end

        Qa = localComputeQ(D, g, r_a, isRelG, isPerG, periodG);
        log_kernel = log_kernel - reshape(Qa, n_jx, n_ky) / (4 * sigmaG(g)^2);
    end

    % --- Add windowed-group contributions (log F_g per pair) ---
    if ~isempty(wmd)
        effX = localEffectiveCentresPerm(dx);    % perm-side eff centres
        effY = localEffectiveCentresComb(dy);    % comb-side eff centres

        for g = 1:nGroups
            if ~localIsWindowedGroupG(wmd.size(g), wmd.mix(g))
                continue;
            end
            attrs_g = attrsOfGroup{g};
            % Stack per-attribute effective centres into a (d_g x nJ/nK) matrix.
            cx_parts = cell(1, numel(attrs_g));
            cy_parts = cell(1, numel(attrs_g));
            centre_parts = cell(1, numel(attrs_g));
            for ia = 1:numel(attrs_g)
                a = attrs_g(ia);
                cx_parts{ia} = effX{a};
                cy_parts{ia} = effY{a};
                centre_parts{ia} = wmd.centre{a}(:);
            end
            cx_g = vertcat(cx_parts{:});   % (d_g, nJ)
            cy_g = vertcat(cy_parts{:});   % (d_g, nK)
            centre_g = vertcat(centre_parts{:});  % (d_g, 1)
            mu_q_g = muQperG{g};                  % (d_g, 1)

            % Cross-correlation coordinate substitution: translate query
            % centres to origin via mu_q_g, translate context centres to
            % origin via centre_g, then apply the window at 0.
            cx_sub = cx_g - mu_q_g;
            cy_sub = cy_g - centre_g;
            centre_sub = zeros(size(centre_g));
            d_g = size(cx_sub, 1);

            s_g = wmd.size(g);
            mix_g = wmd.mix(g);
            sigma_g = sigmaG(g);
            is_rel = isRelG(g);
            r_a = rVec(attrs_g(1));

            log_F = localWindowedContribution( ...
                cx_sub, cy_sub, centre_sub, ...
                s_g, mix_g, sigma_g, is_rel, r_a, d_g);
            log_kernel = log_kernel + log_F;
        end
    end

    E = exp(log_kernel);
    w_u = dx.wJ;
    w_v = dy.wv_comb;
    ip = w_u(:).' * (E * w_v(:));
end


function Qa = localComputeQ(D, g, r_a, isRelG, isPerG, periodG)
%LOCALCOMPUTEQ  Per-attribute quadratic form for MA cos-sim (mirrors
%the logic of computeQaMA in localCosSimMA).
    if isRelG(g)
        if isPerG(g)
            sz = size(D);
            if numel(sz) < 3, sz = [sz, 1]; end
            Qa = zeros(1, sz(2), sz(3));
            P_g = periodG(g);
            for i = 1:r_a
                for j = i+1:r_a
                    delta = D(i, :, :) - D(j, :, :);
                    delta = mod(delta + P_g/2, P_g) - P_g/2;
                    Qa = Qa + delta.^2;
                end
            end
            Qa = Qa / r_a;
        else
            Qa = sum(D.^2, 1) - sum(D, 1).^2 / r_a;
        end
    else
        Qa = sum(D.^2, 1);
    end
end


function eff = localEffectiveCentresPerm(dens)
%LOCALEFFECTIVECENTRESPERM  Return per-attribute effective-space centres
%on the perm side. For MaetDensity these are stored in dens.Centres.
    eff = dens.Centres;
end


function eff = localEffectiveCentresComb(dens)
%LOCALEFFECTIVECENTRESCOMB  Reconstruct per-attribute effective-space
%centres on the comb side from V_comb, using the same reduction as
%build_exp_tens (drop first slot; v[i] = u[i+1] - u[1]).
    A = dens.nAttrs;
    groupOf = dens.groupOfAttr;
    isRelG = logical(dens.isRel);
    rVec = dens.r;
    nK = dens.nK;
    eff = cell(1, A);
    for a = 1:A
        g = groupOf(a);
        r_a = rVec(a);
        V = dens.V_comb{a};
        if ~isRelG(g)
            eff{a} = V;
        elseif r_a >= 2
            eff{a} = V(2:end, :) - V(1, :);
        else
            eff{a} = zeros(0, nK);
        end
    end
end


function tf = localIsWindowedGroupG(size_g, mix_g)
    tf = isfinite(size_g) && size_g > 0;
end


function log_F = localWindowedContribution(cx_g, cy_g, centre_g, ...
        s_g, mix_g, sigma_g, is_rel, r_a, d_g)
%LOCALWINDOWEDCONTRIBUTION  Closed-form log(F_g) per (j, k) pair.
%
%   Dispatches on group geometry:
%     - 1-D groups (any type) and multi-D absolute groups: per-axis
%       factorisable form, full (size, mix) family.
%     - Multi-D relative groups: Gaussian window only (mix = 0). Raises
%       on mix > 0.

    a_rect = s_g * sigma_g * sqrt(3 * mix_g);
    b_conv = s_g * sigma_g * sqrt(1 - mix_g);

    is_1d = (d_g == 1);
    is_multi_abs = (d_g >= 2) && (~is_rel);
    is_multi_rel = (d_g >= 2) && is_rel;

    rho = mix_g;
    if is_multi_rel && rho > 0
        error('cosSimExpTens:unsupportedWindow', ...
              ['Multi-D relative groups (d_g = %d, r_a = %d) do not ' ...
               'support rectangular or raised-rectangular windows ' ...
               '(mix = %g) in v2.1.0. Use mix = 0 (pure Gaussian ' ...
               'window), or wait for a future release with Gaussian-' ...
               'mixture-window approximation.'], d_g, r_a, rho);
    end

    if is_1d || is_multi_abs
        log_F = localWindowedFactorisable(cx_g, cy_g, centre_g, ...
            a_rect, b_conv, sigma_g, is_rel, r_a, d_g);
    else
        % Multi-D relative, mix = 0.
        log_F = localWindowedGaussianMultiRel(cx_g, cy_g, centre_g, ...
            b_conv, sigma_g, r_a, d_g);
    end
end


function log_F = localWindowedFactorisable(cx_g, cy_g, centre_g, ...
        a_rect, b_conv, sigma_g, is_rel, r_a, d_g)
%LOCALWINDOWEDFACTORISABLE  Full (size, mix) family; per-axis product of
%1-D closed-form factors.

    % Effective variance of the (j, k) product Gaussian per axis.
    if is_rel
        sigma_pair_sq = r_a * sigma_g^2 / 2;   % 1-D relative
    else
        sigma_pair_sq = sigma_g^2 / 2;          % absolute
    end
    sigma_t_sq = sigma_pair_sq + b_conv^2;
    sigma_t    = sqrt(sigma_t_sq);

    % Per-pair midpoint minus window centre, per axis.
    nJ = size(cx_g, 2);
    nK = size(cy_g, 2);
    m        = 0.5 * (reshape(cx_g, d_g, nJ, 1) + reshape(cy_g, d_g, 1, nK));
    mu_shift = m - reshape(centre_g, d_g, 1, 1);

    if a_rect == 0 && b_conv > 0
        % Pure Gaussian window (via L'Hopital of the general formula).
        per_axis = (b_conv / sigma_t) * ...
                   exp(-mu_shift.^2 / (2 * sigma_t_sq));
    elseif b_conv == 0 && a_rect > 0
        % Pure rectangular.
        denom = sigma_t * sqrt(2);
        arg_plus  = (mu_shift + a_rect) / denom;
        arg_minus = (mu_shift - a_rect) / denom;
        per_axis = 0.5 * (erf(arg_plus) - erf(arg_minus));
    else
        % General case.
        denom = sigma_t * sqrt(2);
        arg_plus  = (mu_shift + a_rect) / denom;
        arg_minus = (mu_shift - a_rect) / denom;
        numer = erf(arg_plus) - erf(arg_minus);
        norm_denom = 2 * erf(a_rect / (b_conv * sqrt(2)));
        per_axis = numer / norm_denom;
    end

    per_axis = max(per_axis, 1e-300);
    log_F = squeeze(sum(log(per_axis), 1));     % (nJ, nK)
    % Ensure shape nJ x nK even if either dim is 1.
    if isvector(log_F)
        if size(log_F, 1) ~= nJ
            log_F = log_F.';
        end
        log_F = reshape(log_F, nJ, nK);
    end
end


function log_F = localWindowedGaussianMultiRel(cx_g, cy_g, centre_g, ...
        b_conv, sigma_g, r_a, d_g)
%LOCALWINDOWEDGAUSSIANMULTIREL  Gaussian window on a multi-D relative
%group.
%
%   In the "drop first slot, v[i] = u[i+1] - u[1]" reduction used by
%   buildExpTens, the quadratic form on reduced coords is M_rel =
%   I - (1/r) * 1 1^T, so the product Gaussian has covariance
%       Sigma_pair = sigma_g^2 * M_rel^{-1} = sigma_g^2 * (I + 1 1^T).
%   Adding an isotropic Gaussian window (variance b^2) gives
%   Sigma_K = Sigma_pair + b^2 I.

    A_g_inv = eye(d_g) + ones(d_g);   % = M_rel^{-1}
    Sigma_pair = sigma_g^2 * A_g_inv;
    T = b_conv^2 * eye(d_g);
    Sigma_K = Sigma_pair + T;
    K_precision = inv(Sigma_K);

    det_pair = det(Sigma_pair);
    det_K = det(Sigma_K);
    log_prefactor = 0.5 * (log(det_pair) - log(det_K));

    nJ = size(cx_g, 2);
    nK = size(cy_g, 2);
    m        = 0.5 * (reshape(cx_g, d_g, nJ, 1) + reshape(cy_g, d_g, 1, nK));
    mu_shift = m - reshape(centre_g, d_g, 1, 1);

    % Quadratic form (mu_shift)^T * K_precision * (mu_shift) over d_g.
    % Result shape: (nJ, nK).
    % First: temp = K_precision * mu_shift (along dim 1).
    % Then:  K_quad = sum(mu_shift .* temp, 1)
    temp = zeros(size(mu_shift));
    for i = 1:d_g
        row = zeros(1, nJ, nK);
        for j = 1:d_g
            row = row + K_precision(i, j) * mu_shift(j, :, :);
        end
        temp(i, :, :) = row;
    end
    K_quad = squeeze(sum(mu_shift .* temp, 1));
    if isvector(K_quad)
        K_quad = reshape(K_quad, nJ, nK);
    end

    log_F = log_prefactor - 0.5 * K_quad;
end

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
%              empty/scalar for uniform).
%     p2     — Pitch or position values for the second multiset (vector
%              of length n_2).
%     w2     — Weights for the second multiset (vector of length n_2, or
%              empty/scalar for uniform).
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
    % --- Raw arguments ---
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
    error(['Usage: cosSimExpTens(dens_x, dens_y [, ''verbose'', tf]) or ' ...
        'cosSimExpTens(p1, w1, p2, w2, sigma, r, isRel, isPer, period ' ...
        '[, ''verbose'', tf]).']);
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
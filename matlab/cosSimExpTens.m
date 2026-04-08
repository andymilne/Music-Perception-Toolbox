function s = cosSimExpTens(varargin)
%COSSIMEXPTENS Cosine similarity of two r-ad expectation tensors.
%
%   s = cosSimExpTens(dens_x, dens_y):
%   s = cosSimExpTens(dens_x, dens_y, 'verbose', false):
%   Cosine similarity using precomputed density structs from buildExpTens.
%   This avoids recomputing tuple indices and weight products on each call,
%   and is the preferred calling convention when comparing a fixed reference
%   against many other sets.
%
%   s = cosSimExpTens(x_p, x_w, y_p, y_w, sigma, r, isRel, isPer, period):
%   s = cosSimExpTens(x_p, x_w, y_p, y_w, sigma, r, isRel, isPer, period, 'verbose', false):
%   Cosine similarity from raw arguments (builds tuples internally).
%
%   Computes an inner product between the r-ad expectation densities given
%   by two weighted pitch multisets, and with normal perception error with
%   standard deviation sigma. The tensors can be smoothed or unsmoothed,
%   periodic or nonperiodic, absolute or relative.
%
%   Note that this function is usually faster than expectationTensor followed
%   by cosSim or spCosSim in cases where r > 2 and I < 10.
%
%   Four different inner products can be computed, depending on the flags:
%   the inner product assumes periodic equivalence with period set by
%   'period' if isPer == 1, and assumes transpositional equivalence
%   (relative rather than absolute pitches) if isRel == true.
%
%   See the expectationTensor function for further information about these
%   parameters.
%
%   Optional name-value pair (all calling conventions):
%     'verbose' — Logical (default: true). If false, suppresses console
%                 output (time estimates, progress messages).
%
%   Originally by David Bulger, Macquarie University, Australia (2016).
%   (Andrew J. Milne, The MARCS Institute, Western Sydney University made a
%   few trivial changes to comments and variable names for consistency with
%   other routines in the Music Perception Toolbox.)
%
%   Optimized version: preallocated permutation indices, vectorized inner
%   product computation, simplified quadratic form (no explicit quadratic
%   form matrix), precomputed index/pitch/weight data shared across the
%   three inner product calls, automatic chunking for large arrays, and
%   optional precomputed density structs via buildExpTens.
%
%   See also buildExpTens, evalExpTens.

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
    x_p    = varargin{1};
    x_w    = varargin{2};
    y_p    = varargin{3};
    y_w    = varargin{4};
    sigma  = varargin{5};
    r      = varargin{6};
    isRel  = varargin{7};
    isPer  = varargin{8};
    J      = varargin{9};

    % Build density structs on the fly
    dens_x = buildExpTens(x_p, x_w, sigma, r, isRel, isPer, J, 'verbose', verbose);
    dens_y = buildExpTens(y_p, y_w, sigma, r, isRel, isPer, J, 'verbose', verbose);

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
        'cosSimExpTens(x_p, x_w, y_p, y_w, sigma, r, isRel, isPer, period ' ...
        '[, ''verbose'', tf]).']);
end

% === Early return for degenerate case ===
if r > min(numel(dens_x.p), numel(dens_y.p))
    s = NaN;
    return;
end

% === Cosine similarity from three inner products ===
% cos_sim = ip(x,y) / sqrt(ip(x,x) * ip(y,y))

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

                if isRel
                    Qc = sum(Dc.^2, 1) - sum(Dc, 1).^2 / r;
                else
                    Qc = sum(Dc.^2, 1);
                end

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
    %    Relative: Q(d) = sum(d.^2) - sum(d)^2 / r
    %      where d = u_j - v_k and the quadratic form induced by the
    %      Riemannian metric on the quotient space R^r / R*1 projects out
    %      the mean, yielding transpositional equivalence.
    %
    %  The weighted sum is computed as wU' * (E * wV), avoiding the full
    %  outer-product weight matrix.
    % -----------------------------------------------------------------
    function ipval = ipFull(U, wU, nJ, V, wV, nK)
        D = reshape(U, r, nJ, 1) - reshape(V, r, 1, nK);

        if isPer
            D = mod(D + J/2, J) - J/2;
        end

        if isRel
            Qvec = sum(D.^2, 1) - sum(D, 1).^2 / r;
        else
            Qvec = sum(D.^2, 1);
        end

        E = reshape(exp(-Qvec(:) / (4 * sigma^2)), nJ, nK);
        ipval = wU(:)' * (E * wV(:));
    end

end
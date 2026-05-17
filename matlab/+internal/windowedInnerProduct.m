% =========================================================================
%  windowedInnerProduct — closed-form windowed inner product (internal)
% =========================================================================

function s = windowedInnerProduct(a, b, verbose)
%WINDOWEDINNERPRODUCT  Closed-form windowed inner product (internal helper).
%
%   s = internal.windowedInnerProduct(densQ, wmd, verbose)
%
%   Single-scalar windowed similarity between a MaetDensity densQ and
%   a WindowedMaetDensity wmd (in either argument order). This is the
%   inner step used by windowedSimilarity at each offset of its sweep,
%   and the only place the closed-form windowed inner product is
%   implemented.
%
%   Magnitude-aware normalisation: numerator is the windowed inner
%   product <f_q, W f_c>, denominator is the product of the
%   UNWINDOWED L2 norms of the two operands. The result is therefore a
%   magnitude-aware *windowed similarity*, not a strict cosine
%   similarity: silent regions of the context produce values near
%   zero, and matching content near the window centre produces values
%   proportional to how much matching mass is there. See User Guide
%   §3.1 "Magnitude-aware normalisation, not strict cosine
%   similarity".
%
%   One-sided windowing only: exactly one of a, b must be a
%   WindowedMaetDensity. Two-sided windowing is not supported.
%
%   Direct user access to this function is intentionally not provided
%   (the entry is the +internal/ package, not a top-level name). User
%   code computes windowed similarities through windowedSimilarity,
%   which handles both the scalar (single-offset) and sweep
%   (multi-offset) cases uniformly.
%
%   See also windowedSimilarity, windowTensor.

    a_win = strcmp(a.tag, 'WindowedMaetDensity');
    b_win = strcmp(b.tag, 'WindowedMaetDensity');
    if a_win && b_win
        error('internal:windowedInnerProduct:twoSidedWindowing', ...
              ['Two-sided windowing (both operands windowed) is not ' ...
               'supported. Use windowedSimilarity for profile sweeps.']);
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
        error('internal:windowedInnerProduct:badOperand', ...
              'Windowed density can only be compared with a MaetDensity.');
    end

    % Ensure both operands have per-tuple fields populated (cheap if
    % they came from buildExpTens with 'lazy', false; otherwise this
    % is the one-line lazy expansion).
    dens_q = internal.ensureExpTensExpensive(dens_q);
    dens_c = internal.ensureExpTensExpensive(dens_c);

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


function ip = localCosSimNumeratorMA(dx, dy, wmd, verbose)
%LOCALCOSSIMNUMERATORMA  Public-internal wrapper around
%LOCALCOSSIMNUMERATORMACORE that handles within-attribute centre
%symmetrisation in the windowed case.
%
%   The MAET density is symmetric under permutations of components
%   within each attribute's effective coordinates (JMM windowing-
%   theorem remark on within-attribute symmetry of the windowed
%   integral). The integral therefore depends on the within-attribute
%   centre components only through their multiset. The toolbox uses a
%   perm-comb summation for efficiency, which only matches the
%   framework-correct (perm-perm) integral when within-attribute
%   centre components are uniform. For non-uniform within-attribute
%   centres, this wrapper detects the situation and averages the
%   inner product over within-attribute permutations of the centre,
%   one Cartesian product across all non-uniform attributes. Result
%   matches the perm-perm form. Uniform-centre inputs (the common
%   case) bypass this and take the existing fast path.
%
%   See also localCosSimNumeratorMACore.
    if isempty(wmd)
        ip = localCosSimNumeratorMACore(dx, dy, wmd, verbose);
        return;
    end

    % Detect non-uniform within-attribute centres on windowed groups.
    nuAttrs = [];
    nuPerms = {};
    for a = 1:dx.nAttrs
        d_a = dx.dimPerAttr(a);
        if d_a < 2
            continue;
        end
        g = dx.groupOfAttr(a);
        if ~localIsWindowedGroupG(wmd.size(g), wmd.mix(g))
            continue;
        end
        c_a = wmd.centre{a}(:);
        if numel(c_a) ~= d_a
            continue;
        end
        % Use np.allclose-equivalent: rtol = 1e-5, atol = 1e-8.
        ref = c_a(1);
        if any(abs(c_a - ref) > 1e-8 + 1e-5 * abs(ref))
            nuAttrs(end+1) = a;             %#ok<AGROW>
            nuPerms{end+1} = perms(1:d_a);  %#ok<AGROW>  d_a! x d_a
        end
    end

    if isempty(nuAttrs)
        ip = localCosSimNumeratorMACore(dx, dy, wmd, verbose);
        return;
    end

    % Cartesian product over non-uniform attributes' permutations.
    nNu = numel(nuAttrs);
    sizes = cellfun(@(M) size(M, 1), nuPerms);
    counter = ones(1, nNu);
    total_ip = 0;
    n_combos = 0;
    while true
        centre_perm = wmd.centre;
        for kk = 1:nNu
            a = nuAttrs(kk);
            perm_idx = nuPerms{kk}(counter(kk), :).';
            c_a = wmd.centre{a}(:);
            centre_perm{a} = c_a(perm_idx);
        end
        wmd_perm = wmd;
        wmd_perm.centre = centre_perm;
        ip_combo = localCosSimNumeratorMACore(dx, dy, wmd_perm, verbose);
        total_ip = total_ip + ip_combo;
        n_combos = n_combos + 1;

        % Advance the multi-index counter (last index varies fastest).
        kk = nNu;
        while kk >= 1
            counter(kk) = counter(kk) + 1;
            if counter(kk) > sizes(kk)
                counter(kk) = 1;
                kk = kk - 1;
            else
                break;
            end
        end
        if kk == 0
            break;
        end
    end

    ip = total_ip / n_combos;
end


function ip = localCosSimNumeratorMACore(dx, dy, wmd, ~)
%LOCALCOSSIMNUMERATORMACORE  Compute sum_{j,k} w_j^x w_k^y * prod_g (factor),
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
%
%   This is the inner core of localCosSimNumeratorMA. Within-attribute
%   centre symmetrisation is handled by the wrapper; this function
%   computes the IP for a single concrete wmd.

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
            D = D - P_g .* floor(D / P_g + 0.5);
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
                    delta = delta - P_g .* floor(delta / P_g + 0.5);
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
               '(mix = %g). Use mix = 0 (pure Gaussian ' ...
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

function s = sweepCosSimExpTens(densX, densY, offsets, nvArgs)
%SWEEPCOSSIMEXPTENS Similarity against uniform translates of a density.
%
%   s = sweepCosSimExpTens(densX, densY, offsets)
%   computes, for each sweep index m, the similarity of densX against
%   densY with every value of attribute a shifted by offsets(a, m). The
%   whole sweep costs one pass over the tuple pairs plus M evaluations
%   of a mixture in the offset, rather than M inner products.
%
%   The identity. Starting from the multi-attribute inner product
%
%       <T_X, T_Y> = sum_{j,k} w_j w_k prod_a
%                    exp(-Q_a(c_{a,j} - c_{a,k}) / (4 sigma_a^2)),
%
%   translate every value of attribute a on the Y side by a common mu_a.
%   On an absolute attribute Q_a is a sum of squared components, and
%   writing d = c_{a,j} - c_{a,k} with mean dbar,
%
%       Q_a(d - mu_a * 1) = r_a (mu_a - dbar)^2 + sum_i (d_i - dbar)^2.
%
%   The second term does not depend on mu_a. Each (tuple pair, attribute)
%   therefore contributes a fixed SHAPE weight and a PLACEMENT Gaussian in
%   the offset, centred at the mean difference dbar with effective width
%   sigma_a / sqrt(r_a). Placement and shape separate exactly.
%
%   Note the 4 sigma_a^2 denominator: the inner product of two width-sigma
%   kernels is a width-sqrt(2)-sigma kernel. A 2 sigma_a^2 denominator
%   agrees wherever the tuples match exactly (the numerator vanishes
%   there) and is wrong everywhere else.
%
%   Exactness. Two properties make the reduction exact rather than
%   approximate. The shape term is the RELATIVE-mode quadratic form, so
%   it is computed as a Gram matrix of within-tuple-centred coordinates
%   and no r_a x nJ x nK difference array is ever formed. And the
%   truncation rule is the one the per-offset path applies, a floor on
%   the accumulated log-kernel: because the placement term is
%   non-positive, a component whose shape weight alone falls below the
%   floor cannot rise above it at any offset, so dropping it up front
%   changes nothing.
%
%   Modes. An attribute is handled by the split when it is flat or
%   nested, non-periodic, and isotropic. Absolute contributes both a
%   shape and a placement term; relative contributes the shape term
%   alone, since the relative quadratic form IS the shape term --- a
%   uniform translation cancels in every within-tuple difference, which
%   is the same statement as having no placement term, and is why a
%   relative attribute cannot be swept. A nested attribute at an inner
%   or intermediate co-transposition unit is that same case read per
%   block: its form is the sum over blocks of each block's relative
%   form, so it contributes one shape term per block, no placement term,
%   and cannot be swept either. A periodic attribute that is never
%   translated is supported too: it is not split, but contributes an
%   offset-independent factor through the same wrapped kernel the
%   per-offset path applies.
%
%   Relative-periodic. Such an attribute is never swept (a uniform
%   translation is a no-op there) but may contribute its fixed factor,
%   computed through the pairwise wrapped-difference form. That form is
%   one of two measures which coincide only below a sigma/P limit, so
%   the limit the inner-product dispatcher already calibrates governs
%   acceptance here: below it the two agree inside the floor
%   truncationSigmas implies. Above it, wrap = 'single-image' names the
%   measure this form computes and is honoured, while the default
%   full-image reading is refused and left to the per-offset path.
%
%   Inputs
%       densX   - Context density struct from buildExpTens (MA form).
%       densY   - Query density struct; translated by each offset.
%       offsets - A x M matrix of per-attribute translations, one column
%                 per sweep index (a row vector is accepted when A = 1).
%                 A row of zeros leaves that attribute untranslated.
%
%   Name-value pairs
%       'method'            - 'auto' (default), 'mixture', or 'orbit'.
%                             Which decomposition carries the sweep.
%                             'mixture' is the placement/shape split
%                             described above: one pass over the tuple
%                             pairs, then a mixture evaluation per
%                             offset. 'orbit' evaluates the Mobius/orbit
%                             inner product at the shifted values, with
%                             the (event pair, offset) index riding the
%                             orbit routine's batch axis; its cost scales
%                             with the orbit count rather than with the
%                             tuple-pair count [C(K, r) r!]^2, which the
%                             mixture must both enumerate and store.
%                             'auto' compares the two costs and picks.
%                             The orbit route also covers a swept
%                             PERIODIC attribute, which the mixture
%                             refuses: it never forms the split, so the
%                             wrapped kernel absorbs the periodicity.
%       'normalize'         - 'cosine' (default) or 'oneSidedDenom'. Both
%                             self inner products are invariant under a
%                             uniform translation of their own values, so
%                             each is computed once for the whole sweep.
%       'truncationSigmas'  - As in cosSimExpTens ([] takes the default).
%       'verbose'           - Default true.
%
%   Outputs
%       s - 1 x M similarities, indexed as the columns of offsets.
%
%   Errors when the sweep cannot be reduced: a swept relative attribute
%   or a swept nested inner/intermediate one (a uniform translation
%   cancels within every block, so there is nothing to sweep), a swept
%   periodic attribute (the wrapped kernel admits no such split, and the
%   reduction is untested on the torus), a relative-and-periodic
%   attribute above the sigma/P limit whose wrap does not name this
%   measure, or an anisotropic kernel covariance. Translate the query
%   with translateAttributes and compare offset by offset in those
%   cases.
%
%   See also COSSIMEXPTENS, TRANSLATEATTRIBUTES, BUILDEXPTENS.

arguments
    densX struct
    densY struct
    offsets double
    nvArgs.method (1, :) char {mustBeMember(nvArgs.method, ...
        {'auto', 'mixture', 'orbit'})} = 'auto'
    nvArgs.normalize (1, :) char = 'cosine'
    nvArgs.truncationSigmas = []
    nvArgs.verbose (1, 1) logical = true
end

if ~ismember(nvArgs.normalize, {'cosine', 'oneSidedDenom'})
    error('sweepCosSimExpTens:normalize', ...
          ['normalize must be ''cosine'' or ''oneSidedDenom''; got ' ...
           '''%s''.'], nvArgs.normalize);
end

A = double(densX.nAttrs);
off = offsets;
if isrow(off) && A == 1
    off = reshape(off, 1, []);
end
if size(off, 1) ~= A
    error('sweepCosSimExpTens:offsetsShape', ...
          ['offsets must be an A x M matrix with A = %d; got %d rows.'], ...
          A, size(off, 1));
end
if any(~isfinite(off(:)))
    error('sweepCosSimExpTens:offsetsFinite', ...
          'offsets must be finite.');
end
M = size(off, 2);

densX = internal.prunedExpTens(densX);
densY = internal.prunedExpTens(densY);

if isempty(nvArgs.truncationSigmas)
    tsResolved = mptDefaults('truncationSigmas');
else
    tsResolved = nvArgs.truncationSigmas;
end
tsResolved = internal.accuracyFloor('resolve', tsResolved);

% --- Route selection ----------------------------------------------------
orbitOk = localOrbitSupported(densX, densY, off, A, nvArgs.truncationSigmas);
mixtureOk = true;
mixtureErr = [];
try
    localCheckEligible(densX, densY, off, A, nvArgs.truncationSigmas);
catch mixtureErr
    mixtureOk = false;
end

switch nvArgs.method
    case 'auto'
        chosen = localChooseRoute(densX, densY, off, A, mixtureOk, orbitOk);
    otherwise
        chosen = nvArgs.method;
end
if strcmp(chosen, 'mixture') && ~mixtureOk
    rethrow(mixtureErr);
end
if strcmp(chosen, 'orbit') && ~orbitOk
    error('sweepCosSimExpTens:orbitUnsupported', ...
          ['The orbit route does not support a swept relative, a ' ...
           'nested, or an anisotropic attribute in a sweep, nor a ' ...
           'relative-periodic attribute above the sigma/P limit whose ' ...
           'wrap names the single-image measure.']);
end

if strcmp(chosen, 'orbit')
    % The orbit route reads only the cheap per-event fields, so the
    % expensive per-tuple arrays are never materialised for it.
    ipXYorb = localOrbitSweep(densX, densY, off, A, tsResolved);
    ipYYorb = localOrbitSelfIp(densY, A, tsResolved);
    switch nvArgs.normalize
        case 'cosine'
            ipXXorb = localOrbitSelfIp(densX, A, tsResolved);
            denomOrb = sqrt(max(ipXXorb * ipYYorb, 0));
        case 'oneSidedDenom'
            denomOrb = ipYYorb;
    end
    if denomOrb == 0
        s = NaN(1, M);
    else
        s = ipXYorb(:).' / denomOrb;
    end
    return;
end

% The mixture reads the per-tuple fields directly, so a skinny density
% from buildExpTens's default lazy build must be materialised first.
densX = internal.ensureExpTensExpensive(densX);
densY = internal.ensureExpTensExpensive(densY);

% --- Build the mixture once ---------------------------------------------
[centres, logW, amp, threshold, sweptIdx] = ...
    localBuildMixture(densX, densY, off, A, tsResolved);

scales = zeros(numel(sweptIdx), 1);
for i = 1:numel(sweptIdx)
    a = sweptIdx(i);
    rA = size(densX.U_perm{a}, 1);
    scales(i) = rA / (4 * densX.sigma(a)^2);
end

% --- Evaluate at every offset -------------------------------------------
ipXY = localEvaluateMixture(centres, logW, amp, scales, threshold, ...
                            off(sweptIdx, :), M);

% --- Denominators: one per sweep, not one per offset --------------------
ipYY = localSelfIp(densY, A, tsResolved);
switch nvArgs.normalize
    case 'cosine'
        ipXX = localSelfIp(densX, A, tsResolved);
        denom = sqrt(max(ipXX * ipYY, 0));
    case 'oneSidedDenom'
        denom = ipYY;
end

if denom == 0
    s = NaN(1, M);
else
    s = ipXY(:).' / denom;
end

end


% =========================================================================
%  Helpers
% =========================================================================

function localCheckEligible(densX, densY, off, A, tsRaw)
%LOCALCHECKELIGIBLE  Refuse any shape the reduction does not cover.
    swept = any(off ~= 0, 2);
    innerR = zeros(1, A);
    for a = 1:A
        innerR(a) = localInnerBlock(densX, a);
    end
    for a = 1:A
        if densX.isRel(a) && densX.isPer(a)
            % A relative-periodic attribute contributes an
            % offset-independent factor, computed here through the
            % pairwise wrapped-difference form. That form is one of two
            % measures which coincide only below a sigma/P limit, so the
            % same limit the inner-product dispatcher uses governs
            % acceptance: below it the two agree inside the floor
            % truncationSigmas implies, and no choice is being made
            % silently. Above it an explicit wrap = 'single-image' names
            % the measure this form computes and is honoured; the default
            % full-image reading is refused, and the per-offset path
            % decides it by 'method'.
            P_a = densX.period(a);
            if isfinite(P_a) && P_a > 0
                sop = densX.sigma(a) / P_a;
                limit = internal.relPerSigmaOverPThreshold(tsRaw);
                if sop > limit
                    wrapA = 'full-image';
                    if isfield(densX, 'wrap') && ~isempty(densX.wrap) ...
                            && numel(densX.wrap) >= a
                        wrapA = char(densX.wrap{a});
                    end
                    if ~strcmp(wrapA, 'single-image')
                        error('sweepCosSimExpTens:relativePeriodic', ...
                              ['Attribute %d is relative and periodic ' ...
                               'at sigma/P = %.3f, above the limit of ' ...
                               '%.3f for this truncationSigmas; the ' ...
                               'single-wrap and transposition-average ' ...
                               'kernels differ there, so pass wrap = ' ...
                               '''single-image'' to name the former, or ' ...
                               'compare offset by offset, where ' ...
                               '''method'' selects it.'], a, sop, limit);
                    end
                end
            end
        end
        if swept(a) && densX.isPer(a)
            error('sweepCosSimExpTens:periodicAttribute', ...
                  ['Attribute %d is periodic and swept; the wrapped ' ...
                   'kernel does not admit the placement/shape split, ' ...
                   'and the reduction is untested on the torus. ' ...
                   'Translate the query with translateAttributes and ' ...
                   'compare offset by offset instead.'], a);
        end
        if innerR(a) > 0 && swept(a)
            error('sweepCosSimExpTens:sweptNested', ...
                  ['Attribute %d is nested at an inner or intermediate ' ...
                   'co-transposition unit and swept; each block removes ' ...
                   'its own all-ones, so a uniform translation cancels ' ...
                   'within every block and there is nothing to sweep ' ...
                   '(the attribute is supported when it is not ' ...
                   'translated).'], a);
        end
        if swept(a) && densX.isRel(a)
            error('sweepCosSimExpTens:sweptRelative', ...
                  ['Attribute %d is relative and swept; a uniform ' ...
                   'translation cancels in every within-tuple ' ...
                   'difference, so there is nothing to sweep.'], a);
        end
    end
    for d = {densX, densY}
        dd = d{1};
        if internal.densityHasKernelCov(dd)
            error('sweepCosSimExpTens:anisotropicKernel', ...
                  ['An operand carries an anisotropic kernel ' ...
                   'covariance; the split assumes an isotropic ' ...
                   'kernel per attribute.']);
        end
    end
end


function chosen = localChooseRoute(densX, densY, off, A, mixtureOk, orbitOk)
%LOCALCHOOSEROUTE  Pick between the mixture and the orbit route.
%
%   The two scale differently on the same problem. The mixture pays one
%   pass over the tuple pairs --- nJ * nK, which grows as
%   [C(K, r) r!]^2 --- and must also STORE the survivors, so it is the
%   memory-bound route at high tuple order. The orbit route pays per
%   offset instead, but its unit of work is an orbit contraction over
%   the Kx * Ky value kernel, with no tuple enumeration anywhere.
%
%   The two constants below convert between those units and set a floor
%   below which the mixture always wins. They were calibrated on the
%   Python twin's timings and are machine-specific in the same way as
%   mptDefaults('orbitCostIntercept'); tools/ recalibration on this
%   machine is worthwhile before relying on near-crossover routing.
%   Away from the crossover the two routes differ by 10x or more, where
%   a misplaced constant changes nothing.
%
%   Twin of Python _tensor.sweep._choose_sweep_route.
    ORBIT_WORK_RATIO = 64;
    ORBIT_MIN_PAIRS = 1e6;

    if ~orbitOk
        chosen = 'mixture'; return;
    end
    if ~mixtureOk
        chosen = 'orbit'; return;
    end

    % nJ / nK are lazy fields, and forcing them here would build the very
    % arrays the orbit route exists to avoid. Predict the tuple counts
    % from the geometry instead, exactly as the pairwise cost model does:
    % nJ = N * prod_a r_a! * C(K_a, r_a).
    % X enters on the perm side and Y on the comb side, so the two
    % counts differ by prod_a r_a!: nJ = N_x * prod_a r_a! C(K_a, r_a)
    % and nK = N_y * prod_a C(K_a, r_a). Verified against the built
    % densities' own nJ / nK.
    tuplesX = 1; tuplesY = 1;
    for a = 1:A
        r_a = double(densX.r(a));
        cX = nchoosek(double(size(densX.pAttr{a}, 1)), r_a);
        cY = nchoosek(double(size(densY.pAttr{a}, 1)), r_a);
        tuplesX = tuplesX * factorial(r_a) * cX;
        tuplesY = tuplesY * cY;
    end
    nPairs = double(densX.N) * tuplesX * double(densY.N) * tuplesY;
    M = size(off, 2);
    orbitWork = 0;
    for a = 1:A
        if ~any(off(a, :) ~= 0)
            continue;
        end
        r_a = double(densX.r(a));
        if r_a < 2
            % r = 1 has no orbit decomposition to reduce: the
            % per-attribute matrix is a plain kernel sum, and the
            % mixture handles that shape at least as cheaply.
            chosen = 'mixture'; return;
        end
        nOrb = double(numel(mobius.getOrbitTable(r_a)));
        kX = double(size(densX.pAttr{a}, 1));
        kY = double(size(densY.pAttr{a}, 1));
        orbitWork = orbitWork + nOrb * kX * kY * r_a;
    end
    if orbitWork <= 0
        chosen = 'mixture'; return;
    end
    orbitTotal = M * double(densX.N) * double(densY.N) * orbitWork;

    % Memory decides before speed does: the mixture must hold its
    % surviving components, and at high tuple order that array is what
    % fails first, whatever the timings say.
    nSwept = 0;
    for a = 1:A
        if any(off(a, :) ~= 0), nSwept = nSwept + 1; end
    end
    mixtureBytes = nPairs * (nSwept + 2) * 8;
    if mixtureBytes > internal.kernelChunkBytesResolved()
        chosen = 'orbit'; return;
    end

    if nPairs < ORBIT_MIN_PAIRS
        chosen = 'mixture'; return;
    end
    if orbitTotal < ORBIT_WORK_RATIO * nPairs
        chosen = 'orbit';
    else
        chosen = 'mixture';
    end
end


function ok = localOrbitSupported(densX, densY, off, A, tsRaw)
%LOCALORBITSUPPORTED  Whether the orbit route can carry this sweep.
%
%   The route evaluates the Mobius/orbit inner product at the shifted
%   values, so it never forms the placement/shape split and is
%   indifferent to periodicity, which the wrapped kernel absorbs. It
%   computes both self inner products through the same per-attribute
%   routine as the numerator, so the two carry one convention and a
%   relative attribute is admissible here --- untranslated, since a
%   uniform shift cancels in every within-tuple difference either way.
%
%   On a relative-and-periodic attribute the route computes the
%   transposition-average (all-image) kernel. Below the dispatcher's
%   sigma/P limit that agrees with the single-wrap form inside the
%   accuracy floor. Above it the two differ and the attribute's wrap
%   decides: 'full-image' (the default) is the measure this route
%   computes and is accepted, while 'single-image' names the other one
%   and is declined --- the same resolution the per-offset dispatcher
%   reaches for the same inputs.
%
%   Twin of Python _tensor.sweep.orbit_sweep_supported.
    ok = true;
    swept = any(off ~= 0, 2);
    % The orbit decomposition sums over unordered value subsets with
    % multiplicity, and mobius.maPerAttrInnerMatrix takes no symmetry
    % flag: it computes the symmetrised inner product and nothing else.
    % On an ordered attribute that is a different quantity, not an
    % approximation of the right one --- measured departures up to 0.22
    % --- so the route declines rather than silently symmetrising.
    if isfield(densX, 'isSym') && ~isempty(densX.isSym) ...
            && ~all(logical(densX.isSym))
        ok = false; return;
    end
    for a = 1:A
        if localInnerBlock(densX, a) > 0
            ok = false; return;                % nested: no orbit form here
        end
        if densX.isRel(a) && swept(a)
            ok = false; return;                % nothing to sweep
        end
        if densX.isRel(a) && densX.isPer(a)
            P_a = densX.period(a);
            if isfinite(P_a) && P_a > 0
                sop = densX.sigma(a) / P_a;
                if sop > internal.relPerSigmaOverPThreshold(tsRaw)
                    wrapA = 'full-image';
                    if isfield(densX, 'wrap') && ~isempty(densX.wrap) ...
                            && numel(densX.wrap) >= a
                        wrapA = char(densX.wrap{a});
                    end
                    if ~strcmp(wrapA, 'full-image')
                        ok = false; return;
                    end
                end
            end
        end
    end
    if internal.densityHasKernelCov(densX) ...
            || internal.densityHasKernelCov(densY)
        ok = false;
    end
end


function out = localOrbitAttrMatrixSweep(Px, Wx, Py, Wy, sigma, r, mus, ...
                                         isPer, period, wrapA, tsResolved)
%LOCALORBITATTRMATRIXSWEEP  Per-attribute IP matrices at every offset.
%
%   Returns (N_x, N_y, M): entry (n_x, n_y, m) is the per-attribute
%   inner product between event n_x of X and event n_y of Y translated
%   by mus(m). This is the sweep form of mobius.maPerAttrInnerMatrix's
%   absolute branch: the (event pair, offset) index rides the orbit
%   routine's batch axis, so the tuple enumeration never appears and the
%   cost scales with the orbit count rather than with [C(K, r) r!]^2.
%
%   Twin of Python _tensor.sweep._orbit_attr_matrix_sweep.
    nanX = isnan(Px) | isnan(Wx);
    if any(nanX(:)), Px(nanX) = 0; Wx(nanX) = 0; end
    nanY = isnan(Py) | isnan(Wy);
    if any(nanY(:)), Py(nanY) = 0; Wy(nanY) = 0; end

    [Kx, Nx] = size(Px);
    [Ky, Ny] = size(Py);
    M = numel(mus);
    prefactor = (sigma * sqrt(pi))^r;
    useOrbit = r >= 2;      % r = 1 has no orbit decomposition
    out = zeros(Nx, Ny, M);

    fullImage = isPer && strcmp(wrapA, 'full-image');

    % Chunk the offsets: the transient kernel block is
    % (Kx, Nx, Ky, Ny, chunk), with a few live copies.
    perOffset = 4 * Kx * Nx * Ky * Ny * 8;
    memLimit = internal.kernelChunkBytesResolved();
    chunk = max(1, floor(memLimit / max(perOffset, 1)));

    for m0 = 1:chunk:M
        m1 = min(m0 + chunk - 1, M);
        mIdx = m0:m1;
        mc = numel(mIdx);

        % (Kx, Nx, Ky, Ny, mc)
        diffs = reshape(Px, Kx, Nx, 1, 1, 1) ...
              - reshape(Py, 1, 1, Ky, Ny, 1) ...
              - reshape(mus(mIdx), 1, 1, 1, 1, mc);
        if fullImage
            K_tens = internal.wrappedGaussian1d(diffs, sigma, period, ...
                                                 tsResolved, 4);
        else
            if isPer
                diffs = diffs - period * floor(diffs / period + 0.5);
            end
            K_tens = internal.truncKernelExp(diffs .^ 2, sigma, tsResolved);
        end

        % -> (Nx, Ny, mc, Kx, Ky), then flatten the batch axis.
        K_perm  = permute(K_tens, [2, 4, 5, 1, 3]);
        K_pairs = reshape(K_perm, Nx * Ny * mc, Kx, Ky);

        Wx_t = Wx.';                                    % (Nx, Kx)
        Wx_pairs = reshape(repmat(reshape(Wx_t, Nx, 1, 1, Kx), ...
                                  1, Ny, mc, 1), Nx * Ny * mc, Kx);
        Wy_t = Wy.';                                    % (Ny, Ky)
        Wy_pairs = reshape(repmat(reshape(Wy_t, 1, Ny, 1, Ky), ...
                                  Nx, 1, mc, 1), Nx * Ny * mc, Ky);

        if useOrbit
            flat = mobius.innerProductOrbitPwBatched( ...
                K_pairs, Wx_pairs, Wy_pairs, r, 'prefactor', prefactor);
        else
            % r = 1: each event contributes a single kernel, so the
            % per-attribute matrix is the weighted kernel sum directly.
            flat = prefactor * sum(sum( ...
                K_pairs .* reshape(Wx_pairs, [], Kx, 1) ...
                        .* reshape(Wy_pairs, [], 1, Ky), 3), 2);
        end
        out(:, :, mIdx) = reshape(flat, Nx, Ny, mc);
    end
end


function ipXY = localOrbitSweep(densX, densY, off, A, tsResolved)
%LOCALORBITSWEEP  Sweep numerator via the orbit decomposition.
%
%   The multi-attribute inner product is a sum over event pairs of a
%   product over attributes, so each attribute contributes an
%   (N_x, N_y, M) block and the blocks multiply. An attribute that is
%   never translated contributes the same block at every offset, so it
%   is computed once and broadcast.
    M = size(off, 2);
    P = ones(densX.N, densY.N, M);
    for a = 1:A
        sigma = densX.sigma(a);
        r_a = densX.r(a);
        isPer = densX.isPer(a);
        period = densX.period(a);
        if ~isfinite(period), period = 0; end
        wrapA = 'full-image';
        if isfield(densX, 'wrap') && ~isempty(densX.wrap) ...
                && numel(densX.wrap) >= a
            wrapA = char(densX.wrap{a});
        end
        if ~any(off(a, :) ~= 0)
            block = mobius.maPerAttrInnerMatrix( ...
                densX.pAttr{a}, densX.w{a}, densY.pAttr{a}, densY.w{a}, ...
                sigma, r_a, densX.isRel(a), isPer, period, ...
                'truncationSigmas', tsResolved, 'wrap', wrapA);
            P = P .* block;
        else
            P = P .* localOrbitAttrMatrixSweep( ...
                densX.pAttr{a}, densX.w{a}, densY.pAttr{a}, densY.w{a}, ...
                sigma, r_a, off(a, :), isPer, period, wrapA, tsResolved);
        end
    end
    ipXY = reshape(sum(sum(P, 1), 2), 1, M);
end


function val = localOrbitSelfIp(dens, A, tsResolved)
%LOCALORBITSELFIP  <T, T> through the same per-attribute routine as the
%   numerator.
%
%   The orbit path's own similarity triple chooses per attribute between
%   the closed-form centres route and the grid contraction, and the two
%   differ by a constant per-attribute prefactor that cancels only
%   within one route's own triple. Taking the self terms from the
%   routine the numerator uses keeps numerator and denominator on one
%   convention, and is what lets a relative attribute ride this route.
    P = ones(dens.N, dens.N);
    for a = 1:A
        period = dens.period(a);
        if ~isfinite(period), period = 0; end
        wrapA = 'full-image';
        if isfield(dens, 'wrap') && ~isempty(dens.wrap) ...
                && numel(dens.wrap) >= a
            wrapA = char(dens.wrap{a});
        end
        P = P .* mobius.maPerAttrInnerMatrix( ...
            dens.pAttr{a}, dens.w{a}, dens.pAttr{a}, dens.w{a}, ...
            dens.sigma(a), dens.r(a), dens.isRel(a), dens.isPer(a), ...
            period, 'truncationSigmas', tsResolved, 'wrap', wrapA);
    end
    val = sum(P(:));
end


function Qa = localRelPerQ(D, r, period)
%LOCALRELPERQ  Relative-periodic quadratic form from differences D.
%
%   Q = sum_{i<j} wrap(d_i - d_j)^2 / r, over all r*(r-1)/2 pairs of the
%   full r-tuple. Each pairwise delta is wrapped into [-P/2, P/2);
%   wrapping pairwise rather than component-wise is what preserves exact
%   transposition invariance on the circle. D is r x nJ x nK and Qa is
%   returned as 1 x nJ x nK, matching computeQaMA's convention.
%
%   Twin of the rel-and-per branch of computeQaMA in cosSimExpTens and
%   of _compute_Q in the Python dispatch module.
    sz = size(D);
    if numel(sz) < 3
        sz = [sz, 1];
    end
    Qa = zeros(1, sz(2), sz(3));
    for i = 1:r
        for j = i+1:r
            delta = D(i, :, :) - D(j, :, :);
            delta = delta - period .* floor(delta / period + 0.5);
            Qa = Qa + delta .^ 2;
        end
    end
    Qa = Qa / r;
end


function blockSize = localInnerBlock(dens, a)
%LOCALINNERBLOCK  Co-transposition unit size for attribute A, else 0.
%
%   Non-zero only for a nested attribute resolved to an inner or
%   intermediate unit, where the quadratic form is block-diagonal: each
%   block removes its own all-ones. Twin of the per-attribute entry of
%   the Python dispatch helper _inner_r_vec.
    blockSize = 0;
    if ~isfield(dens, 'nested') || isempty(dens.nested) ...
            || numel(dens.nested) < a
        return;
    end
    sp = dens.nested{a};
    if isstruct(sp) && isfield(sp, 'proj') && isfield(sp, 'relUnit') ...
            && ismember(char(sp.proj), {'inner', 'intermediate'})
        % relUnit is a 1-BASED level index on the MATLAB side, so the
        % block size is prod(r(1:u)) --- matching cosSimExpTens,
        % evalExpTens, and internal.nestedContract. The Python twin
        % stores the same quantity 0-based and writes prod(r[:u + 1]);
        % the two agree on the block size, not on the index.
        rLevels = sp.r(:);
        u = double(sp.relUnit);
        blockSize = prod(rLevels(1:u));
    end
end


function blocks = localShapeBlocks(U, blockSize)
%LOCALSHAPEBLOCKS  Within-block centred residuals of an r x n array.
%
%   BLOCKSIZE is the row count of one co-transposition unit: 0 (or the
%   whole tuple) for a flat attribute, or the nested inner unit's size.
%   The quadratic form removes each block's own all-ones, so the shape
%   term is the sum over blocks of the squared distance between
%   block-centred residuals --- one Gram matrix per block. Twin of the
%   Python _tensor.sweep._shape_blocks.
    r = size(U, 1);
    if blockSize <= 0 || blockSize >= r
        blocks = {U - mean(U, 1)};
        return;
    end
    nBlocks = floor(r / blockSize);
    blocks = cell(1, nBlocks);
    for b = 1:nBlocks
        rows = ((b - 1) * blockSize + 1):(b * blockSize);
        Ub = U(rows, :);
        blocks{b} = Ub - mean(Ub, 1);
    end
end


function [centres, logW, amp, threshold, sweptIdx] = ...
        localBuildMixture(densX, densY, off, A, tsResolved)
%LOCALBUILDMIXTURE  One pass over tuple pairs, producing the mixture.
%
%   Returns the surviving components: CENTRES is P x S (one row per tuple
%   pair, one column per swept attribute), LOGW the fixed log-kernel, AMP
%   the product of the two tuple weights, and THRESHOLD the log-kernel
%   floor. A splittable attribute that is never translated has a constant
%   placement term, so its whole contribution folds into LOGW and it
%   costs the mixture no axis at all.
    nJ = double(densX.nJ);
    nK = double(densY.nK);
    swept = any(off ~= 0, 2);

    % Block size of one co-transposition unit per attribute: 0 for a
    % flat attribute, or the nested inner/intermediate unit's size.
    blockOf = zeros(1, A);
    for a = 1:A
        blockOf(a) = localInnerBlock(densX, a);
    end
    % An attribute carries a placement term only where its quadratic
    % form keeps the tuple's own mean: absolute, and not block-quotiented.
    hasPlacement = false(1, A);
    for a = 1:A
        hasPlacement(a) = ~densX.isRel(a) && ~densX.isPer(a) ...
                          && blockOf(a) == 0;
    end
    sweptIdx = [];
    for a = 1:A
        if swept(a) && hasPlacement(a)
            sweptIdx(end + 1) = a; %#ok<AGROW>
        end
    end
    S = numel(sweptIdx);
    sweptPos = zeros(1, A);
    sweptPos(sweptIdx) = 1:S;

    threshold = -0.5 * tsResolved^2;
    nTerms = nJ * nK;
    if nTerms > 1
        threshold = threshold - log(nTerms);
    end

    % Per attribute: tuple means (which set the placement centres) and
    % within-tuple-centred residuals (which set the shape weights).
    meanU = cell(1, A);  meanV = cell(1, A);
    cenU  = cell(1, A);  cenV  = cell(1, A);
    for a = 1:A
        if densX.isPer(a)
            continue;      % handled by the wrapped kernel, not the split
        end
        % Everything below depends on the two operands only through
        % their differences, so a shift shared by both is exact --- and
        % it keeps the shape term well conditioned, that term being a
        % Gram form whose cancellation costs significant digits when the
        % coordinates sit far from the origin. The shift is one of the
        % attribute's own values rather than their mean, because a data
        % value is exactly representable and subtracting it from a
        % nearby value is itself exact (Sterbenz).
        U = densX.U_perm{a};
        V = densY.V_comb{a};
        if ~isempty(U)
            origin = U(1);
            U = U - origin;
            V = V - origin;
        end
        cenU{a} = localShapeBlocks(U, blockOf(a));
        cenV{a} = localShapeBlocks(V, blockOf(a));
        if hasPlacement(a)
            meanU{a} = mean(U, 1);
            meanV{a} = mean(V, 1);
        end
    end

    wJ = densX.wJ(:);
    wK = densY.wv_comb(:);

    % Chunk along the comb side so the transient nJ x nKc blocks stay
    % within the kernel memory budget.
    perCol = max(1, 2 * max(S, 1) + 2) * nJ * 8;
    memLimit = internal.kernelChunkBytesResolved();
    chunk = max(1, floor(memLimit / perCol));

    centres = zeros(0, S);
    logW    = zeros(0, 1);
    amp     = zeros(0, 1);

    for c = 1:chunk:nK
        cEnd = min(c + chunk - 1, nK);
        idx  = c:cEnd;
        nKc  = numel(idx);

        logFixed = zeros(nJ, nKc);
        cenBlk   = zeros(nJ, nKc, max(S, 1));

        for a = 1:A
            if densX.isPer(a)
                % Not splittable, and (by the eligibility check) never
                % swept: an offset-independent factor, computed by the
                % same kernel the per-offset path applies. Which kernel
                % that is depends on the mode. Relative-and-periodic
                % takes the pairwise-wrap form below. Absolute-periodic
                % factors across coordinates as prod_k theta(d_k), so
                % its full-image log-kernel is sum_k log theta(d_k),
                % while the single-image opt-in takes the nearest-image
                % reduction.
                D = reshape(densX.U_perm{a}, size(densX.U_perm{a}, 1), nJ, 1) ...
                  - reshape(densY.V_comb{a}(:, idx), ...
                            size(densY.V_comb{a}, 1), 1, nKc);
                if densX.isRel(a)
                    % Relative and periodic: the pairwise-wrap form,
                    % Q = sum_{i<j} wrap(d_i - d_j)^2 / r. Wrapping each
                    % pairwise delta rather than each component is what
                    % preserves exact transposition invariance on the
                    % circle, so this cannot be read as a component-wise
                    % wrap of the absolute form. Twin of computeQaMA's
                    % rel-and-per branch in cosSimExpTens.
                    Qa = localRelPerQ(D, size(D, 1), densX.period(a));
                    logFixed = logFixed ...
                        - reshape(Qa, nJ, nKc) / (4 * densX.sigma(a)^2);
                    continue;
                end
                wrapA = 'full-image';
                if isfield(densX, 'wrap') && ~isempty(densX.wrap) ...
                        && numel(densX.wrap) >= a
                    wrapA = char(densX.wrap{a});
                end
                if strcmp(wrapA, 'full-image')
                    theta = internal.wrappedGaussian1d( ...
                        D, densX.sigma(a), densX.period(a), tsResolved, 4);
                    logFixed = logFixed + reshape(sum(log(theta), 1), nJ, nKc);
                else
                    Pa = densX.period(a);
                    D = D - Pa .* floor(D / Pa + 0.5);
                    Qa = sum(D .^ 2, 1);
                    logFixed = logFixed ...
                        - reshape(Qa, nJ, nKc) / (4 * densX.sigma(a)^2);
                end
                continue;
            end
            inv = 1 / (4 * densX.sigma(a)^2);
            % Shape term: one Gram matrix per co-transposition block.
            spread = zeros(nJ, nKc);
            for b = 1:numel(cenU{a})
                Ub = cenU{a}{b};
                Vb = cenV{a}{b}(:, idx);
                spread = spread + (sum(Ub .^ 2, 1).' + sum(Vb .^ 2, 1) ...
                                   - 2 * (Ub.' * Vb));
            end
            spread = max(spread, 0);
            logFixed = logFixed - spread * inv;

            if ~hasPlacement(a)
                % No placement term: for a relative attribute the form
                % *is* the shape term, and for a block-quotiented nested
                % one each block removes its own all-ones.
                continue;
            end
            % Explicit orientation: the outer difference of the two
            % mean vectors must be nJ x nKc. Vector-by-vector indexing
            % takes the array's orientation rather than the index's, so
            % the shapes are stated rather than inherited.
            dbar = reshape(meanU{a}, nJ, 1) - reshape(meanV{a}(idx), 1, nKc);
            if sweptPos(a) > 0
                cenBlk(:, :, sweptPos(a)) = dbar;
            else
                rA = size(densX.U_perm{a}, 1);
                logFixed = logFixed - (rA * dbar .^ 2) * inv;
            end
        end

        % A component whose fixed log-kernel already falls below the floor
        % can never rise above it: the placement term is non-positive at
        % every offset. Dropping it here is exact.
        keep = logFixed >= threshold;
        if ~any(keep(:))
            continue;
        end
        ampBlk = wJ * reshape(wK(idx), 1, nKc);

        nKeep = sum(keep(:));
        cenKeep = zeros(nKeep, S);
        for i = 1:S
            slab = cenBlk(:, :, i);
            cenKeep(:, i) = slab(keep);
        end
        centres = [centres; cenKeep];       %#ok<AGROW>
        logW    = [logW;    logFixed(keep)]; %#ok<AGROW>
        amp     = [amp;     ampBlk(keep)];   %#ok<AGROW>
    end
end


function out = localEvaluateMixture(centres, logW, amp, scales, ...
                                    threshold, offs, M)
%LOCALEVALUATEMIXTURE  Evaluate the mixture at every offset.
%
%   Components outside the truncation radius are culled by a sorted index
%   on one axis, then tested exactly on the surviving slice, so the
%   result matches a dense evaluation term for term.
    out = zeros(1, M);
    P = size(centres, 1);
    S = size(centres, 2);
    if P == 0 || M == 0
        return;
    end
    if S == 0
        % Nothing is translated: the mixture takes the same value at
        % every sweep index.
        out(:) = exp(logW).' * amp;
        return;
    end

    % Every surviving component has logW >= threshold, so its placement
    % budget is at most -threshold. That bounds the cull radius on each
    % axis uniformly.
    budget = max(logW) - threshold;
    if ~isfinite(budget) || budget < 0
        return;
    end
    radii = sqrt(budget ./ max(scales, realmin));

    % Index on the axis whose radius excludes most.
    spans = reshape(max(centres, [], 1) - min(centres, [], 1), 1, S);
    radiiRow = reshape(radii, 1, S);
    ratio = inf(1, S);
    pos = radiiRow > 0;
    ratio(pos) = spans(pos) ./ radiiRow(pos);
    [~, axisSel] = max(ratio);

    [key, order] = sort(centres(:, axisSel));
    key = reshape(key, [], 1);
    cenS  = centres(order, :);
    logWS = logW(order);
    ampS  = amp(order);

    lo = localLowerBound(key, offs(axisSel, :) - radii(axisSel));
    hi = localUpperBound(key, offs(axisSel, :) + radii(axisSel));

    % Two regimes. The culled loop visits one offset at a time and pays a
    % fixed per-offset cost, which is the right trade when each offset
    % sees a small slice of a large mixture. When the mixture is small or
    % the cull excludes little, that per-offset cost dominates the
    % arithmetic it saves, and evaluating every component against a block
    % of offsets at once is faster. The threshold compares the two
    % directly rather than guessing: dense work is P per offset, culled
    % work is the mean slice plus the per-offset overhead expressed in
    % component-equivalents. The constant is calibrated against both
    % regimes: sparse mixtures (point-set-seeded sweeps, where each offset
    % sees a slice of order ten components out of thousands) and dense ones
    % (a single wide-kernel attribute, where nearly every component is
    % live). Culling's advantage where it wins reaches an order of
    % magnitude, dense's is under a factor of two, so the constant is set
    % to favour culling when the two are close.
    offsetOverheadInComponents = 512;
    meanSlice = mean(max(hi - lo + 1, 0));
    if P <= meanSlice + offsetOverheadInComponents
        out = localEvaluateDense(cenS, logWS, ampS, scales, threshold, ...
                                 offs, M);
        return;
    end

    for m = 1:M
        i0 = lo(m);
        i1 = hi(m);
        if i1 < i0
            continue;
        end
        d = cenS(i0:i1, :) - offs(:, m).';
        L = logWS(i0:i1) - (d .^ 2) * scales;
        live = L >= threshold;
        if ~any(live)
            continue;
        end
        Lm = L(live);
        Am = ampS(i0:i1);
        out(m) = exp(Lm).' * Am(live);
    end
end


function out = localEvaluateDense(centres, logW, amp, scales, ...
                                  threshold, offs, M)
%LOCALEVALUATEDENSE  Evaluate every component against a block of offsets.
%
%   Computes exactly what the culled loop computes --- same threshold,
%   same terms --- without the per-offset dispatch. Blocked over the
%   offsets so the transient (block, P) array honours the kernel memory
%   budget.
    P = size(centres, 1);
    S = size(centres, 2);
    out = zeros(1, M);
    memLimit = internal.kernelChunkBytesResolved();
    block = max(1, floor(memLimit / max(P * 8 * 4, 1)));
    for m0 = 1:block:M
        m1 = min(m0 + block - 1, M);
        nb = m1 - m0 + 1;
        % (nb, P) accumulated over the swept axes.
        L = repmat(reshape(logW, 1, P), nb, 1);
        for i = 1:S
            d = reshape(centres(:, i), 1, P) - reshape(offs(i, m0:m1), nb, 1);
            L = L - scales(i) * (d .^ 2);
        end
        below = L < threshold;
        L = exp(L);
        L(below) = 0;
        out(m0:m1) = (L * amp).';
    end
end


function idx = localLowerBound(key, q)
%LOCALLOWERBOUND  First index whose sorted key is >= q (numel+1 if none).
%
%   Vectorised binary search: ~log2(P) iterations over all M queries at
%   once, rather than a linear scan per query.
%   KEY is a sorted column; Q may be a row or a column. Indexing a
%   vector by a vector gives the result the orientation of the ARRAY,
%   not the index, so key(mid(active)) comes back as a column whatever
%   the shape of Q. Comparing that against the row q(active) would
%   broadcast to a square matrix rather than compare elementwise, so
%   the probe is deposited into a size(Q) buffer first: assignment by
%   logical index matches on element count, not orientation.
    n = numel(key);
    lo = ones(size(q));
    hi = repmat(n + 1, size(q));
    probe = zeros(size(q));
    while any(lo(:) < hi(:))
        mid = floor((lo + hi) / 2);
        active = lo < hi;
        probe(active) = key(mid(active));
        take = false(size(q));
        take(active) = probe(active) < q(active);
        lo(take) = mid(take) + 1;
        hi(active & ~take) = mid(active & ~take);
    end
    idx = lo;
end


function idx = localUpperBound(key, q)
%LOCALUPPERBOUND  Last index whose sorted key is <= q (0 if none).
%   See localLowerBound for why the probe is buffered rather than
%   compared directly against q.
    n = numel(key);
    lo = ones(size(q));
    hi = repmat(n + 1, size(q));
    probe = zeros(size(q));
    while any(lo(:) < hi(:))
        mid = floor((lo + hi) / 2);
        active = lo < hi;
        probe(active) = key(mid(active));
        take = false(size(q));
        take(active) = probe(active) <= q(active);
        lo(take) = mid(take) + 1;
        hi(active & ~take) = mid(active & ~take);
    end
    idx = lo - 1;
end


function val = localSelfIp(dens, A, tsResolved)
%LOCALSELFIP  <T, T> for one density, through this file's own mixture.
%
%   The self inner product is invariant under a uniform translation of
%   the density's own values, so the whole sweep shares one denominator.
%   It is computed here by the same routine that produces the numerator,
%   evaluated at zero offset. Routing it through cosSimExpTens instead
%   would risk picking up the orbit route's per-attribute prefactor
%   convention, which cancels only within that route's own triple and
%   would not match this file's numerator.
    zeroOff = zeros(A, 1);
    [centres, logW, amp, threshold, sweptIdx] = ...
        localBuildMixture(dens, dens, zeroOff, A, tsResolved);
    val = localEvaluateMixture(centres, logW, amp, zeros(0, 1), ...
                               threshold, zeroOff(sweptIdx, :), 1);
end

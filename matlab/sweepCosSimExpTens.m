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
%   Modes. An attribute is handled by the split when it is flat,
%   non-periodic, and isotropic. A periodic attribute that is never
%   translated is supported too: it is not split, but contributes an
%   offset-independent factor through the same wrapped kernel the
%   per-offset path applies. Absolute contributes both terms;
%   relative contributes the shape term alone, since the relative
%   quadratic form IS the shape term --- a uniform translation cancels in
%   every within-tuple difference, which is the same statement as having
%   no placement term, and is why a relative attribute cannot be swept.
%
%   Inputs
%       densX   - Context density struct from buildExpTens (MA form).
%       densY   - Query density struct; translated by each offset.
%       offsets - A x M matrix of per-attribute translations, one column
%                 per sweep index (a row vector is accepted when A = 1).
%                 A row of zeros leaves that attribute untranslated.
%
%   Name-value pairs
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
%   (a no-op by construction), a swept periodic attribute (the wrapped
%   kernel admits no such split, and the reduction is untested on the
%   torus), a relative-and-periodic attribute whether swept or not (the
%   single-wrap and transposition-average kernels differ there), a
%   nested attribute (its quadratic form is a block-diagonal quotient),
%   or an anisotropic kernel covariance. Translate the query with
%   translateAttributes and compare offset by offset in those cases.
%
%   See also COSSIMEXPTENS, TRANSLATEATTRIBUTES, BUILDEXPTENS.

arguments
    densX struct
    densY struct
    offsets double
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
% The mixture reads the per-tuple fields directly, so a skinny density
% from buildExpTens's default lazy build must be materialised first.
densX = internal.ensureExpTensExpensive(densX);
densY = internal.ensureExpTensExpensive(densY);

localCheckEligible(densX, densY, off, A);

if isempty(nvArgs.truncationSigmas)
    tsResolved = mptDefaults('truncationSigmas');
else
    tsResolved = nvArgs.truncationSigmas;
end
tsResolved = internal.accuracyFloor('resolve', tsResolved);

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

function localCheckEligible(densX, densY, off, A)
%LOCALCHECKELIGIBLE  Refuse any shape the reduction does not cover.
    swept = any(off ~= 0, 2);
    innerR = zeros(1, A);
    if isfield(densX, 'nested') && ~isempty(densX.nested)
        for a = 1:A
            sp = densX.nested{a};
            if isstruct(sp) && isfield(sp, 'proj') && ...
                    ismember(char(sp.proj), {'inner', 'intermediate'})
                rLevels = sp.r(:);
                u = double(sp.rel_unit);
                innerR(a) = prod(rLevels(1:u + 1));
            end
        end
    end
    for a = 1:A
        if densX.isRel(a) && densX.isPer(a)
            % Above sigma/P ~ 0.03 the pairwise and orbit routes compute
            % genuinely different measures on a relative-periodic
            % attribute --- the single-wrap and the transposition-average
            % kernel. The fixed factor here would commit to one silently,
            % so the choice is left with the per-offset path, where
            % 'method' selects it explicitly.
            error('sweepCosSimExpTens:relativePeriodic', ...
                  ['Attribute %d is both relative and periodic; the ' ...
                   'single-wrap and transposition-average kernels differ ' ...
                   'there, and the reduction would fix that choice ' ...
                   'silently.'], a);
        end
        if swept(a) && densX.isPer(a)
            error('sweepCosSimExpTens:periodicAttribute', ...
                  ['Attribute %d is periodic and swept; the wrapped ' ...
                   'kernel does not admit the placement/shape split, ' ...
                   'and the reduction is untested on the torus. ' ...
                   'Translate the query with translateAttributes and ' ...
                   'compare offset by offset instead.'], a);
        end
        if innerR(a) > 0
            error('sweepCosSimExpTens:nestedAttribute', ...
                  ['Attribute %d is nested; its quadratic form is a ' ...
                   'block-diagonal quotient, not a sum of squared ' ...
                   'components.'], a);
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

    sweptIdx = [];
    for a = 1:A
        if swept(a) && ~densX.isRel(a) && ~densX.isPer(a)
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
        U = densX.U_perm{a};
        V = densY.V_comb{a};
        meanU{a} = mean(U, 1);
        meanV{a} = mean(V, 1);
        cenU{a}  = U - meanU{a};
        cenV{a}  = V - meanV{a};
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
                % same wrapped kernel the per-offset path applies. The
                % abs-per full-image r-tuple kernel factors across
                % coordinates as prod_k theta(d_k), so the log-kernel is
                % sum_k log theta(d_k); the single-image opt-in takes the
                % nearest-image reduction instead.
                D = reshape(densX.U_perm{a}, size(densX.U_perm{a}, 1), nJ, 1) ...
                  - reshape(densY.V_comb{a}(:, idx), ...
                            size(densY.V_comb{a}, 1), 1, nKc);
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
            % Shape term: ||cu_j - cv_k||^2 as a Gram matrix.
            U = cenU{a};
            V = cenV{a}(:, idx);
            spread = sum(U .^ 2, 1).' + sum(V .^ 2, 1) - 2 * (U.' * V);
            spread = max(spread, 0);
            logFixed = logFixed - spread * inv;

            if densX.isRel(a)
                % No placement term: the relative form is the shape term.
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

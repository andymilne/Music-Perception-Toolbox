function [in, orderedAny, nestedAny] = flatSelectorInputs( ...
        densX, densY, normalize, truncationSigmas, cacheX, cacheY)
%INTERNAL.FLATSELECTORINPUTS  The flat selector's inputs for a cosine.
%
%   [IN, ORDEREDANY, NESTEDANY] = INTERNAL.FLATSELECTORINPUTS(DENSX,
%   DENSY, NORMALIZE, TRUNCATIONSIGMAS, CACHEX, CACHEY) builds, for two
%   pruned MaetDensity structs, every argument that
%   INTERNAL.SELECTMAINNERPRODUCTMETHOD takes apart from the user
%   method, as a struct IN with fields
%
%     rVec, kVec, A, Nx, Ny, anyPer, anyRelNonper, anyRelPer,
%     sigmaOverPMax, relVec, nuVec, kVecY, wrapVec, truncationSigmas,
%     skipXX, skipYY, symVec, guardForcedBulger, perVec
%
%   together with ORDEREDANY (an attribute ordered at r > 1 on either
%   side, which overrides the selector's choice with Bulger's method)
%   and NESTEDANY (either density nested). CACHEX / CACHEY are the
%   operands' self-IP memo caches; omitted, they are read from the
%   structs' 'selfIP' fields (INTERNAL.SELFIPFROMSTRUCT).
%
%   One function builds these for the real call (cosSimExpTens's
%   localCosSimMA) and for EXPLAINDISPATCH, so the report cannot drift
%   from the route the call takes: the report used to omit the wrap
%   vector, the grid node counts and the memo flags, and so named the
%   wrong route wherever those decided. Twin of the Python
%   cosine._flat_selector_inputs.
%
%   The wrap declaration is checked here (mpt:wrapMismatch): the wrap
%   declares the measure, and a cosine between two measures is not a
%   cosine, so the two densities must agree on every periodic
%   attribute.
%
%   See also INTERNAL.SELECTMAINNERPRODUCTMETHOD, EXPLAINDISPATCH.

    if nargin < 5 || isempty(cacheX); cacheX = internal.selfIpFromStruct(densX); end
    if nargin < 6 || isempty(cacheY); cacheY = internal.selfIpFromStruct(densY); end

    A        = densX.nAttrs;
    rVec     = densX.r;
    sigmaG   = densX.sigma;
    isRelG   = logical(densX.isRel);
    isPerG   = logical(densX.isPer);
    periodG  = densX.period;

    % Per-attribute wrap opt-in (v3+). The density's wrap cell selects the
    % abs-per measure: 'full-image' (default) uses the torus (all-image)
    % 1-D wrapped Gaussian per coordinate; 'single-image' uses the
    % nearest-image reduction. Non-periodic attributes ignore this axis.
    % A disagreement is an error rather than a silent reading of densX's
    % declaration, which made the result depend on operand order. Twin
    % of the Python cosine._declared_wrap.
    wrapVec = repmat({'full-image'}, 1, A);
    for a = 1:A
        wx = localWrapOf(densX, a);
        wy = localWrapOf(densY, a);
        if ~strcmp(wx, wy) && isPerG(a)
            error('mpt:wrapMismatch', ...
                ['wrap mismatch on attribute %d: dens_x declares ''%s'' ' ...
                 'and dens_y declares ''%s''. The wrap declares the ' ...
                 'measure, so the two densities must declare the same ' ...
                 'wrap on every periodic attribute.'], a, wx, wy);
        end
        wrapVec{a} = wx;
    end

    % One value-count vector per density: the two need not carry the same
    % number of values in an attribute, and a chord against a scale, or a
    % reference tuning against an equal division, is the ordinary case.
    kVec = zeros(1, A);
    kVecY = zeros(1, A);
    for a = 1:A
        kVec(a)  = size(densX.pAttr{a}, 1);
        kVecY(a) = size(densY.pAttr{a}, 1);
    end
    anyPer = false; anyRelNonper = false; anyRelPer = false; sigmaOverPMax = 0;
    for a = 1:A
        if isPerG(a); anyPer = true; end
        if isRelG(a)
            if isPerG(a)
                anyRelPer = true;
                if periodG(a) > 0
                    sigmaOverPMax = max(sigmaOverPMax, sigmaG(a) / periodG(a));
                end
            else
                anyRelNonper = true;
            end
        end
    end

    % Per-attribute vectors for the Möbius-side cost model: which
    % attributes are relative, and each one's translation-grid node
    % estimate (matching the grid rules of the batched rel helper).
    % The per-call truncation width sizes the grids the routes are
    % priced on, as it sizes the kernels they run: pricing at the global
    % default while truncating at the per-call width would race the
    % routes on a grid neither of them uses.
    tsSel = internal.accuracyFloor('resolve', truncationSigmas);
    relVec = false(1, A);
    nuVec = 2000 * ones(1, max(A, 1));
    nuVec = nuVec(1:A);
    for a = 1:A
        relVec(a) = isRelG(a);
        if isRelG(a) && rVec(a) >= 2
            if isPerG(a)
                nuVec(a) = internal.autoNtauDefault(periodG(a), ...
                                                    sigmaG(a), tsSel);
            else
                PxA = densX.pAttr{a};
                PyA = densY.pAttr{a};
                marginA = internal.relWindowMargin(tsSel);
                span = (max(PxA(:), [], 'omitnan') ...
                        - min(PxA(:), [], 'omitnan')) ...
                     + (max(PyA(:), [], 'omitnan') ...
                        - min(PyA(:), [], 'omitnan')) ...
                     + 2 * marginA * sigmaG(a);
                nuVec(a) = max(64, ceil(max(span, 1.0) / sigmaG(a) * 10));
            end
        end
    end

    % A self inner product costs nothing at call time when it is
    % memoised, or (for <X,X>) when the requested normalisation does
    % not consume it; tell the selector so its pricing reflects the
    % work this call will actually perform. The flags are *shared* by
    % the two routes' prices --- see INTERNAL.SELFIPMEMOISED for why a
    % per-route flag makes the comparison unfair, and INTERNAL.SELFIPKEY
    % for why the memoised values themselves stay per route.
    needXX = strcmp(normalize, 'cosine');
    skipXX = ~needXX || internal.selfIpMemoised(cacheX);
    skipYY = internal.selfIpMemoised(cacheY);

    % Nested densities route through the hierarchical contraction, not
    % the flat Bulger pairwise path, so the flat forced-Bulger
    % feasibility guard must not fire for them; the selector receives
    % that as its guard flag. The per-attribute sym flags let the guard
    % count an ordered attribute's C(K_a, r_a) tuples rather than the
    % unordered K_a!/(K_a - r_a)!.
    nestedAny = localAnyNested(densX) || localAnyNested(densY);
    if isfield(densX, 'isSym')
        symVec = logical(densX.isSym(:).');
    else
        symVec = true(1, A);
    end

    % Ordered (isSym = false) attributes at r_a > 1 on either side.
    rRow = double(rVec(:).');
    orderedAny = any(~symVec & (rRow > 1));
    if isfield(densY, 'isSym')
        orderedAny = orderedAny ...
            || any(~logical(densY.isSym(:).') & (rRow > 1));
    end

    in = struct( ...
        'rVec', rVec, 'kVec', kVec, 'A', A, ...
        'Nx', densX.N, 'Ny', densY.N, ...
        'anyPer', anyPer, 'anyRelNonper', anyRelNonper, ...
        'anyRelPer', anyRelPer, 'sigmaOverPMax', sigmaOverPMax, ...
        'relVec', relVec, 'nuVec', nuVec, 'kVecY', kVecY, ...
        'wrapVec', {wrapVec}, 'truncationSigmas', truncationSigmas, ...
        'skipXX', skipXX, 'skipYY', skipYY, 'symVec', symVec, ...
        'guardForcedBulger', ~nestedAny, 'perVec', isPerG);
end


function w = localWrapOf(d, a)
%LOCALWRAPOF  The wrap a density declares on attribute a ('full-image'
%   when undeclared).
    w = 'full-image';
    if isfield(d, 'wrap') && ~isempty(d.wrap) ...
            && a <= numel(d.wrap) && ~isempty(d.wrap{a})
        w = char(d.wrap{a});
    end
end


function tf = localAnyNested(d)
    tf = isfield(d, 'nested') && iscell(d.nested) ...
        && any(~cellfun(@isempty, d.nested));
end

function [centresMs, mobiusMs] = nestedEvalCostsMs(dens, nQ)
%NESTEDEVALCOSTSMS  Cost row for the nested attributes of a density.
%
%   [centresMs, mobiusMs] = internal.nestedEvalCostsMs(dens, nQ) prices,
%   in milliseconds on the calibration machine, the tag-tree centres
%   enumeration and the per-level Möbius evaluator
%   (mobius.evalNestedAttrOrbit) over the nested attributes of dens for
%   nQ query points. Flat attributes contribute nothing here (they are
%   priced by internal.maEvalCostsMs); a density without nested
%   attributes returns zeros. Twin of the Python
%   mpt._tensor.dispatch._nested_eval_costs_ms.
%
%   Two laws per nested attribute:
%
%       centresMs = C0 + C1 * T + nQ * C2 * T^gamma * d^delta
%       mobiusMs  = M0 + N * (M1 + nQ * M2 * (K * s)^alpha * nU)
%
%   with T = mPerm * N the tuple-centre count, d the reduced dimension,
%   s the leaf slots, K the values per event, and nU the translation-grid
%   node count (1 when absolute). Fitted September 2026 by
%   python/tools/fit_nested_eval_cost.py on the 3096-cell grid of
%   tools/benchNestedEval.m (two-level shapes of 2--4 groups of 3--5
%   values, r up to 2x4, every [sym] pattern, absolute and both units,
%   periodic and not, N = 4 and 32, 1 to 200 queries). The Möbius
%   per-event per-query cost is a near power law in K * s (log residual
%   0.18) and does not see the tuple count at all. Routing regret against
%   the measured oracle on random half-splits by shape: 1.036, against
%   1.65 for the previous 'auto', which kept the centres route
%   throughout. Python carries its own constants (same form).

    MOBIUS_SETUP_MS      = 0.170484;
    MOBIUS_PER_EVENT_MS  = 0.16837;
    MOBIUS_PER_OP_MS     = 8.72168e-06;
    MOBIUS_OP_EXP        = 1.1434;
    CENTRES_SETUP_MS     = 0.435035;
    CENTRES_PER_TUPLE_MS = 0.000302415;
    CENTRES_PER_QUERY_MS = 7.65419e-05;
    CENTRES_TUPLE_EXP    = 0.7632;
    CENTRES_DIM_EXP      = 0.3134;

    centresMs = 0;
    mobiusMs = 0;
    if ~isfield(dens, 'nested') || isempty(dens.nested)
        return;
    end
    A = double(dens.nAttrs);
    sigmaG = double(dens.sigma(:).');
    isPerG = logical(dens.isPer(:).');
    periodG = double(dens.period(:).');
    nQeff = max(double(nQ), 1);
    for a = 1:A
        if a > numel(dens.nested) || isempty(dens.nested{a})
            continue;
        end
        spec = dens.nested{a};
        pA = double(dens.pAttr{a});
        N = size(pA, 2);
        rLevels = double(spec.r(:).');
        s = prod(rLevels);
        tg = double(spec.tags);
        if isvector(tg), tg = tg(:); end
        T = 0;
        Ksum = 0;
        for n = 1:N
            live = ~isnan(pA(:, n));
            T = T + internal.nestedTupleCount(tg(live, :), rLevels, spec.sym);
            Ksum = Ksum + sum(live);
        end
        K = Ksum / max(N, 1);
        relUnit = [];
        if isfield(spec, 'relUnit') && ~isempty(spec.relUnit) ...
                && ~any(isnan(spec.relUnit)) && spec.relUnit > 0
            relUnit = double(spec.relUnit);   % 1-based level
        end
        if isempty(relUnit)
            dA = s;
            nU = 1;
        else
            sUnit = prod(rLevels(1:relUnit));
            dA = s - s / sUnit;
            spp = internal.resolveSamplesPerSigma([], max(2, sUnit), []);
            if isPerG(a) && periodG(a) > 0
                nU = max(64, ceil(spp * periodG(a) / sigmaG(a)));
            else
                v = pA(~isnan(pA));
                if isempty(v), span = 0; else, span = max(v) - min(v); end
                span = span + 16 * sigmaG(a);
                nU = max(64, ceil(spp * max(span, 1) / sigmaG(a)));
            end
        end
        centresMs = centresMs + CENTRES_SETUP_MS ...
            + CENTRES_PER_TUPLE_MS * T ...
            + nQeff * CENTRES_PER_QUERY_MS * T^CENTRES_TUPLE_EXP ...
              * max(dA, 1)^CENTRES_DIM_EXP;
        mobiusMs = mobiusMs + MOBIUS_SETUP_MS ...
            + N * (MOBIUS_PER_EVENT_MS ...
                   + nQeff * MOBIUS_PER_OP_MS * (K * s)^MOBIUS_OP_EXP * nU);
    end
end

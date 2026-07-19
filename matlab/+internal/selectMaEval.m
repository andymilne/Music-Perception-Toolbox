function [chosen, routingReason] = selectMaEval(dens, nQ, verbose)
%SELECTMAEVAL  Cost-model path selection for multi-attribute evalExpTens.
%
%   [CHOSEN, ROUTINGREASON] = INTERNAL.SELECTMAEVAL(DENS, NQ) chooses
%   between the joint-centres path (which materialises the joint tuple
%   set) and the factored Möbius evaluator MOBIUS.EVALMAORBIT, returning
%   CHOSEN in {'centres', 'mobius'} and a short ROUTINGREASON. NQ is the
%   query count; it scales both paths' per-query work.
%
%   No probe. Because the MAET density factorises across attributes
%   (Milne 2026, Eq. maet-density), both paths' costs are closed-form from
%   the shape (r_a, K_a, N), the geometry, and NQ, and the crossover is
%   sharp, so a pure cost model suffices. User overrides are honoured by
%   the caller before this is reached (evalExpTens passes 'auto' here).
%
%   Hard rules (in order):
%     - nested attribute            -> centres (flat Möbius not applicable)
%     - all r <= 1                  -> centres (Möbius degenerate)
%     - precision floor / feasibility bound on any attribute forces the
%       single-image centres route, GUARDED: if its joint tuple set is
%       infeasible to materialise it raises mpt:dispatch:singleImageInfeasible
%       (no cheaper all-image substitute exists there).
%     - relative-periodic above sigma/P threshold -> the all-image Möbius
%       form is the preferred (cheaper, memory-safe) default measure;
%       warns and points to method='centres' for the single-image measure.
%     - otherwise the cost model: two closed-form per-call time estimates
%       (centres materialisation and kernel work, linear in the joint
%       tuple count; factored Möbius setup and per-query distinct-block
%       work), with a safety factor favouring Möbius at near-ties.
%
%   Twin of python _select_ma_eval.
%
%   See also MOBIUS.EVALMAORBIT, INTERNAL.SELECTMAINNERPRODUCTMETHOD.

    if nargin < 2 || isempty(nQ), nQ = 200; end
    if nargin < 3, verbose = true; end

    % --- Hard-rule constants (mirror Python dispatch.py) ---
    ORBIT_R_MAX_FEASIBLE = 10;
    ORBIT_SIGMA_OVER_P_THRESHOLD = 0.03;

    % --- Calibrated cost-model constants, in milliseconds ---
    % Fitted to the selection-quality grid (single-attribute, r = 2..4,
    % K = 6..48, all four mode combinations, nQ = 1 and 200, sigma = 15
    % over spans of 1200--3600 cents; July 2026 MATLAB harness
    % bench_ma_eval_calibration.m). Absolute values are machine-specific;
    % selection depends only on their ratios. Python carries its own
    % constants (same functional form, per-language calibration): where
    % Python's centres per-query grows with the joint count (its kernel
    % does not cull), MATLAB's non-periodic centres kernel bucket-culls,
    % so its per-query work is nearly flat and the joint-count growth
    % shows up in the per-call materialisation term instead --- the same
    % "cost grows with the joint set" captured in a different term.
    %
    % Centres: a per-call materialisation term and a per-query kernel
    % term, both linear in the joint tuple count.
    MA_COST_CENTRES_SETUP_MS          = 0.15;
    MA_COST_CENTRES_CALL_PER_JOINT_MS = 8e-5;
    MA_COST_CENTRES_QUERY_PER_JOINT_MS = 1.5e-8;

    % Möbius: a per-call setup scaling with the partition count B_r, plus
    % per-query work linear in the distinct-block op count (2^r - 1) r K
    % (each distinct block's factor is computed once and reused across
    % partitions). Relative attributes multiply the per-query work by the
    % u-grid node count; each node costs the cheaper of the direct
    % strategy (op-count linear) and, non-periodically, the factored
    % strategy (K-free after tabulation).
    MA_COST_MOBIUS_SETUP_MS           = 0.10;
    MA_COST_MOBIUS_SETUP_PER_BELL_MS  = 0.01;
    MA_COST_MOBIUS_QUERY_PER_OP_MS    = 1.3e-6;
    % Relative-mode u-grid node costs. Periodic direct nodes and the two
    % non-periodic strategies are calibrated separately.
    MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS      = 4e-6;
    MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS  = 9e-7;
    MA_COST_MOBIUS_REL_NODE_FACTORED_PER_BELL_MS  = 2.1e-5;
    % u-grid tabulation setup (once per call): building the interpolation
    % table costs K source evaluations over the N_u grid nodes. MATLAB's
    % lean per-query readback leaves this setup as the dominant Möbius cost
    % at small nQ, so it is modelled explicitly; Python's larger per-query
    % node cost absorbs it, so its twin constant is ~0.
    MA_COST_MOBIUS_REL_TABULATION_PER_NODE_MS     = 3.7e-5;
    % u-grid nodes per sigma for the relative-mode node-count estimate,
    % mirroring the evaluators' samples_per_sigma default.
    MA_COST_REL_SAMPLES_PER_SIGMA     = 10.0;
    % Safety factor favouring Möbius at near-ties: Möbius is chosen
    % whenever its estimate is below the centres estimate times this
    % factor. The asymmetry is deliberate -- Möbius is failure-safe (flat,
    % bounded cost) while centres materialises the joint tuple set and can
    % exhaust memory -- so a near-tie breaks toward Möbius.
    MA_MOBIUS_SAFETY                  = 1.5;

    BELL = [1 2 5 15 52 203 877 4140 21147 115975];  % B_1..B_10

    A       = double(dens.nAttrs);
    rVec    = double(dens.r(:).');
    kVec    = double(dens.K(:).');
    isRel   = logical(dens.isRel(:).');
    isPer   = logical(dens.isPer(:).');
    sigmaG  = double(dens.sigma(:).');
    periodG = double(dens.period(:).');

    % ---- Hard rule: nested attributes -> centres. ----
    if isfield(dens, 'nested') && ~isempty(dens.nested)
        for a = 1:A
            if ~isempty(dens.nested{a})
                chosen = 'centres';
                routingReason = 'nested attribute (flat Möbius not applicable)';
                return;
            end
        end
    end

    % ---- Hard rule: all r <= 1 -> centres (Möbius degenerate). ----
    if all(rVec <= 1)
        chosen = 'centres';
        routingReason = 'all r <= 1';
        return;
    end

    % ---- Hard rules per attribute: precision floor / feasibility force
    % the single-image centres route (Möbius genuinely unavailable). ----
    forceCentresReason = '';
    for a = 1:A
        r_a = rVec(a); K_a = kVec(a);
        if r_a < 2
            continue;
        end
        if ~internal.orbitSafeForPrecision(r_a, K_a)
            forceCentresReason = sprintf( ...
                'attr %d: K - r = %d below precision floor', a, K_a - r_a);
            break;
        end
        if r_a > ORBIT_R_MAX_FEASIBLE
            forceCentresReason = sprintf( ...
                'attr %d: r = %d exceeds orbit feasibility bound', a, r_a);
            break;
        end
    end

    if ~isempty(forceCentresReason)
        % Centres is the only route; guard against OOM (no cheaper
        % all-image fallback here).
        jointWs = internal.estimateMaJointWorkingSetBytes(rVec, kVec, isRel);
        if jointWs > internal.dispatchMemBudget()
            error('mpt:dispatch:singleImageInfeasible', ...
                ['evalExpTens requires the single-image centres route ' ...
                 '(%s, so the Möbius method is not available), but its ' ...
                 'joint tuple set would need ~%.1f GB. Reduce the tuple ' ...
                 'order r or the collection size K.'], ...
                forceCentresReason, jointWs / 1024^3);
        end
        chosen = 'centres';
        routingReason = forceCentresReason;
        return;
    end

    % ---- Relative-periodic measure preference (precedes the cost
    % model): above sigma/P the all-image Möbius form is the preferred,
    % memory-safe default; single-image via method='centres'. ----
    for a = 1:A
        if isRel(a) && isPer(a) && periodG(a) > 0 ...
                && sigmaG(a) / periodG(a) > ORBIT_SIGMA_OVER_P_THRESHOLD
            if verbose
                internal.warnRelPerAllImage(sigmaG(a) / periodG(a));
            end
            chosen = 'mobius';
            routingReason = 'rel-per all-image measure';
            return;
        end
    end

    % ---- Cost model: two closed-form per-call time estimates (ms), each
    % a per-call setup term plus per-query work scaled by nQ. Centres:
    % materialisation and kernel work linear in the joint tuple count
    % (product across attributes). Möbius: setup scaling with the
    % partition count, plus per-query distinct-block work summed across
    % attributes; relative attributes multiply their per-query work by a
    % u-grid node count estimated from the geometry (period over sigma
    % when periodic; source spread over sigma when not -- the query
    % spread is unknown at selection time, so the source spread stands in
    % for the alignment window). ----
    nQeff = max(double(nQ), 1);
    jointTuples = 1.0;
    for a = 1:A
        r_a = rVec(a); K_a = kVec(a);
        if r_a < 1
            continue;
        end
        jointTuples = jointTuples * factorial(r_a) * localComb(K_a, r_a);
    end
    centresMs = MA_COST_CENTRES_SETUP_MS ...
        + MA_COST_CENTRES_CALL_PER_JOINT_MS * jointTuples ...
        + MA_COST_CENTRES_QUERY_PER_JOINT_MS * jointTuples * nQeff;

    mobiusMs = MA_COST_MOBIUS_SETUP_MS;
    for a = 1:A
        r_a = rVec(a); K_a = kVec(a);
        if r_a < 2
            continue;  % r_a <= 1: a plain kernel sum either way
        end
        if r_a <= numel(BELL)
            B_r = BELL(r_a);
        else
            B_r = Inf;
        end
        ops = (2^r_a - 1) * r_a * K_a;
        mobiusMs = mobiusMs + MA_COST_MOBIUS_SETUP_PER_BELL_MS * B_r;
        perQueryMs = MA_COST_MOBIUS_QUERY_PER_OP_MS * ops;
        if isRel(a)
            sps = MA_COST_REL_SAMPLES_PER_SIGMA;
            if isPer(a) && periodG(a) > 0
                N_u = max(64, ceil(sps * periodG(a) / sigmaG(a)));
                nodeMs = MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS * ops;
            else
                spread = 0.0;
                if a <= numel(dens.pAttr) && ~isempty(dens.pAttr{a})
                    arr = double(dens.pAttr{a}(:));
                    if ~isempty(arr)
                        spread = max(arr) - min(arr);
                    end
                end
                window = 2.0 * spread + 16.0 * sigmaG(a);
                N_u = max(64, ceil(sps * window / sigmaG(a)));
                nodeMs = min( ...
                    MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS * ops, ...
                    MA_COST_MOBIUS_REL_NODE_FACTORED_PER_BELL_MS * B_r);
            end
            % Tabulation setup is paid once per call, not per query.
            mobiusMs = mobiusMs ...
                + MA_COST_MOBIUS_REL_TABULATION_PER_NODE_MS * K_a * N_u;
            perQueryMs = N_u * nodeMs;
        end
        mobiusMs = mobiusMs + perQueryMs * nQeff;
    end

    if mobiusMs < centresMs * MA_MOBIUS_SAFETY
        chosen = 'mobius';
        routingReason = 'cost model (factored Möbius cheaper)';
    else
        chosen = 'centres';
        routingReason = 'cost model (joint centres cheaper)';
    end
end

function c = localComb(nn, kk)
%LOCALCOMB  Binomial coefficient C(nn, kk), integer-valued.
    if kk < 0 || kk > nn
        c = 0; return;
    end
    c = 1;
    for ii = 0:kk - 1
        c = c * (nn - ii) / (ii + 1);
    end
    c = round(c);
end

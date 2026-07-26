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
    % Bucket-grid culling geometry factor for the non-periodic culled
    % kernels (dimensionless; shared with the Python twin).
    MA_COST_CENTRES_CULL_C            = 15.0;
    % Per-attribute per-query overhead of the factored centres route
    % (bucket lookup and gather). Seeded from the Python fit; re-derive
    % with bench_ma_eval_calibration on this side if picks look off.
    MA_COST_CENTRES_FACTORED_QUERY_BASE_MS = 1.5e-3;

    % Möbius: a per-call setup scaling with the partition count B_r, plus
    % per-query work linear in the distinct-block op count (2^r - 1) r K
    % (each distinct block's factor is computed once and reused across
    % partitions). Relative attributes multiply the per-query work by the
    % u-grid node count; each node costs the cheaper of the direct
    % strategy (a fixed per-node floor plus op-count-linear work) and,
    % non-periodically, the factored strategy (K-free after tabulation).
    MA_COST_MOBIUS_SETUP_MS           = 0.10;
    MA_COST_MOBIUS_SETUP_PER_BELL_MS  = 0.01;
    MA_COST_MOBIUS_QUERY_PER_OP_MS    = 1.3e-6;
    % Relative-mode u-grid node costs. Periodic direct nodes and the two
    % non-periodic strategies are calibrated separately. The non-periodic
    % direct node carries a fixed per-node floor (the alignment shift and
    % read-back each node pays) alongside its op-count slope. Here the
    % factored strategy is the cheaper branch for every practical shape
    % (r >= 2, K >= 2), so this floor does not affect the current
    % selection; it is carried for parity with the per-query node model and
    % to keep the direct estimate floored should a factored recalibration
    % make it the binding branch. Its magnitude tracks the factored
    % constant (the analytic floor:factored ratio of the calibrated model)
    % and would be re-derived by bench_ma_eval_calibration if it ever binds.
    MA_COST_MOBIUS_REL_NODE_DIRECT_BASE_MS        = 1.2e-5;
    MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS      = 4e-6;
    MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS  = 9e-7;
    MA_COST_MOBIUS_REL_NODE_FACTORED_PER_BELL_MS  = 2.1e-5;
    % u-grid tabulation setup (once per call): building the interpolation
    % table costs K source evaluations over the N_u grid nodes. MATLAB's
    % lean per-query readback leaves this setup as the dominant Möbius cost
    % at small nQ, so it is modelled explicitly; Python's larger per-query
    % node cost absorbs it, so its twin constant is ~0.
    MA_COST_MOBIUS_REL_TABULATION_PER_NODE_MS     = 3.7e-5;
    % u-grid nodes per sigma for the relative-mode node-count estimate
    % are derived per attribute via internal.resolveSamplesPerSigma,
    % mirroring the evaluators' accuracy-tied resolution.
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

    % ---- Hard rule: ordered ([sym] = 0) attributes at r > 1 -> centres.
    % The Möbius decomposition sums over set partitions of the slot
    % indices, which counts every ordering of each block and so realises
    % the symmetrised tuple set; on an ordered attribute that is a
    % different density, not a faster route to the same one. r = 1 is
    % exempt ([sym] vacuous at a single slot). Twin of the Python
    % _has_ordered_attr rule in _tensor/dispatch.py. ----
    if internal.hasOrderedAttr(dens)
        chosen = 'centres';
        routingReason = 'ordered ([sym]=0) attribute (no orbit to collapse)';
        return;
    end

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

    % The factored centres route (all r_a >= 2, scalar sigma) never
    % materialises the joint tuple set: cost is the SUM of per-attribute
    % tuple counts through the culled per-attribute kernels, plus a small
    % per-attribute per-query overhead (bucket lookup and gather). The
    % joint-materialisation pricing applies only where that route is
    % unsupported (any r_a < 2, or a matrix kernel covariance), mirroring
    % localMaEvalFactored's support predicate. Non-periodic attributes
    % take the bucket-grid culling discount min(1, c*sigma/spread);
    % periodic ones run dense (the pairwise wrap is not a
    % tail-truncatable ball).
    factoredSupported = (A > 1) && all(rVec >= 2) ...
        && ~internal.densityHasKernelCov(dens);
    if factoredSupported
        centresMs = MA_COST_CENTRES_SETUP_MS;
        for a = 1:A
            r_a = rVec(a); K_a = kVec(a);
            T_a = factorial(r_a) * localComb(K_a, r_a);
            cull_a = 1.0;
            if ~isPer(a) && sigmaG(a) > 0
                spread = 0.0;
                if a <= numel(dens.pAttr) && ~isempty(dens.pAttr{a})
                    arr = double(dens.pAttr{a}(:));
                    if ~isempty(arr)
                        spread = max(arr) - min(arr);
                    end
                end
                if spread > 0
                    cull_a = min(1.0, ...
                        MA_COST_CENTRES_CULL_C * sigmaG(a) / spread);
                end
            end
            centresMs = centresMs ...
                + MA_COST_CENTRES_CALL_PER_JOINT_MS * T_a ...
                + nQeff * (MA_COST_CENTRES_FACTORED_QUERY_BASE_MS ...
                           + MA_COST_CENTRES_QUERY_PER_JOINT_MS ...
                             * T_a * cull_a);
        end
    else
        centresMs = MA_COST_CENTRES_SETUP_MS ...
            + MA_COST_CENTRES_CALL_PER_JOINT_MS * jointTuples ...
            + MA_COST_CENTRES_QUERY_PER_JOINT_MS * jointTuples * nQeff;
    end

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
            % The spectral (Fourier) strategy engages inside the mobius
            % relative evaluator for r_a in 2..4 above its query
            % thresholds (see the gate in mobius.evalOrbitRel); where it
            % would engage, price the per-query cost with its measured,
            % K-free slope. The slope scales with the mode count, i.e.
            % with window/sigma (session-calibrated on the Python side;
            % re-derive here via bench_ma_eval_calibration if picks
            % look off).
            FOUR_PER_MODE = [1.05e-4, 2.2e-3, 6.7e-3];   % r = 2, 3, 4
            % Periodic-only K term (per r) added to the K-free slope: the
            % per-event spectrum build A_m(eta) = sum_i w^m exp(-i eta p_i)
            % carries K, which the fixed-period window does not absorb. In
            % periodic mode the spectral branch fires for r = 2 and 3 (the
            % r = 4 mode grid exceeds the memory guard and falls to the
            % node path, so its K growth is priced there). Fitted from the
            % periodic engaging cells of bench_ma_eval_calibration.
            FOUR_PERIODIC_K_MS = [1.8e-5, 3.8e-5, 0.0]; % r = 2, 3, 4
            FOUR_MIN_Q = [16, 32, 64];
            FOUR_MIN_K = [2, 8, 16];
            spreadF = 0.0;
            if a <= numel(dens.pAttr) && ~isempty(dens.pAttr{a})
                arrF = double(dens.pAttr{a}(:));
                if ~isempty(arrF)
                    spreadF = max(arrF) - min(arrF);
                end
            end
            if isPer(a) && periodG(a) > 0
                windowF = periodG(a);
            else
                windowF = 2.0 * spreadF + 16.0 * sigmaG(a);
            end
            % Mirror the spectral branch's own MAX_POINTS decline (see
            % _SPECTRAL_IP_MAX_POINTS in the Python cosine module and
            % spectralRelInnerMatrix on this side): the mode grid is
            % (r_a - 1)-dimensional, so at small sigma/P and r_a = 4 it
            % can pass the query/K thresholds yet still stand down,
            % falling through to the u-grid node path. Price the path
            % that actually runs.
            MODE_SIGMAS = 8.6;   MAX_POINTS = 4e6;
            if isPer(a) && periodG(a) > 0
                Lspec = periodG(a);
            else
                Lspec = 2.0 * spreadF + 2.0 * (MODE_SIGMAS + 2.0) * sigmaG(a);
            end
            Mspec = ceil(MODE_SIGMAS / sqrt(2) ...
                         * Lspec / (2 * pi * sigmaG(a))) + 2;
            spectralFits = (2 * Mspec + 1)^(r_a - 1) <= MAX_POINTS;
            if r_a >= 2 && r_a <= 4 ...
                    && nQ >= FOUR_MIN_Q(r_a - 1) ...
                    && K_a >= FOUR_MIN_K(r_a - 1) ...
                    && spectralFits
                fourMs = FOUR_PER_MODE(r_a - 1) ...
                    * (windowF / max(sigmaG(a), 1e-12)) * nQeff;
                if isPer(a) && FOUR_PERIODIC_K_MS(r_a - 1) > 0
                    fourMs = fourMs + FOUR_PERIODIC_K_MS(r_a - 1) * K_a ...
                        * (windowF / max(sigmaG(a), 1e-12)) * nQeff;
                end
                mobiusMs = mobiusMs + MA_COST_MOBIUS_SETUP_MS + fourMs;
                continue;
            end
            sps = internal.resolveSamplesPerSigma([], r_a, []);
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
                    MA_COST_MOBIUS_REL_NODE_DIRECT_BASE_MS ...
                    + MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS * ops, ...
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

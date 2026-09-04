function [centresMs, mobiusMs] = maEvalCostsMs(dens, nQ)
%MAEVALCOSTSMS  Closed-form eval cost estimates for a flat MA density.
%
%   [CENTRESMS, MOBIUSMS] = INTERNAL.MAEVALCOSTSMS(DENS, NQ) returns the
%   two predicted per-call wall times in milliseconds on the calibration
%   machine: the joint-centres path and the factored Möbius evaluator.
%
%   Depends only on the density shape (r_a, K_a, N), the geometry, and
%   the query count NQ --- no probe, no timing. Shared by the eval path
%   selector INTERNAL.SELECTMAEVAL, which compares the two, and by
%   EXPLAINDISPATCH, which reports both prices even where a hard rule
%   settled the route without consulting them. One implementation
%   guarantees that the report and the dispatch decision price identical
%   work.
%
%   Twin of the Python _ma_eval_costs_ms.
%
%   See also INTERNAL.SELECTMAEVAL, MOBIUS.EVALMAORBIT.

    % --- Calibrated cost-model constants, in milliseconds ---
    % Refit September 2026 by python/tools/fit_ma_eval_cost.py on the
    % 442-cell bench_ma_eval_calibration grid (Sections A--E; the July
    % grid plus the geometry, node-count, spectral and K = 34 sweeps),
    % second run in one MATLAB session on the maintainer's Mac. Fit
    % quality, predicted over measured: centres geometric mean 0.94
    % (spread 1.41, worst 3.0), Moebius 0.90 (spread 1.61, worst 6.3 ---
    % the residual is rel-per r = 4, K >= 24 at high nQ, under-priced
    % 5--6x but routed to Moebius regardless). Routing regret against the
    % measured oracle, geometric mean over the 328 cells with both arms
    % timed: 1.007 (3 cells beyond 1.3x, worst 1.94x) against 1.018 (9,
    % worst 3.9x) for the July values. Previous values kept in the
    % comments beside each constant so the refit reverts in one edit.
    % Absolute values are machine-specific;
    % selection depends only on their ratios. Python carries its own
    % constants (same functional form, per-language calibration). The two
    % fits distribute the joint-count growth differently: MATLAB's puts it
    % chiefly in the per-call materialisation term and leaves the
    % per-query term nearly flat, while Python's carries it in the
    % per-query term. Both express the same "cost grows with the joint
    % set", and either fit reproduces its own language's timings; the
    % split between the two terms is a property of the fits, not of the
    % evaluators. The kernels themselves agree: internal.gaussianKernelSum
    % and the Python _kernel.gaussian_kernel_sum apply grid-bucket
    % truncation under the same conditions, so neither language culls
    % where the other does not.
    %
    % Centres: a per-call materialisation term, linear in the joint tuple
    % count, and a per-query kernel term split three ways by kernel (see
    % localCentresQuerySlope below).
    MA_COST_CENTRES_SETUP_MS = 0.1994;   % was 0.2096
    MA_COST_CENTRES_CALL_PER_JOINT_MS = 6.925e-05;   % was 6.907e-5
    % Per-query cost has a floor that no culling removes (the bucket
    % lookup and gather each query pays) plus a term linear in the joint
    % tuple count, and the two kernels carry different constants: the
    % non-periodic kernel is bucket-culled, the periodic one runs dense.
    % One shared pair cannot express that --- at 255024 joint tuples the
    % measured per-query costs differ by a factor of 300 --- so they are
    % calibrated separately.
    MA_COST_CENTRES_QUERY_BASE_MS = 0.001329;   % was 1.220e-3
    MA_COST_CENTRES_QUERY_PER_JOINT_MS = 1.699e-05;   % was 6.893e-6
    MA_COST_CENTRES_QUERY_BASE_PER_MS = 0.0004084;   % was 7.442e-4
    MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS = 2.319e-06;   % was 3.658e-6
    % The dense periodic per-query term is split again, absolute against
    % relative, and each half carries an exponent on the tuple count.
    % The two periodic kernels are not one kernel: the absolute one
    % measures a wrapped distance per coordinate, the relative one forms
    % wrapped differences first. On the Python side their measured
    % per-tuple costs differ by an order of magnitude and grow
    % differently --- absolute essentially linear in the tuple count,
    % relative going as count^1.3 over counts from 30 to 1e5 --- and
    % sharing one linear term between them under-priced large
    % relative-periodic shapes badly enough to misroute them.
    %
    % The form is carried here so the twins stay structurally identical;
    % the values are MATLAB's own. On the MATLAB grid both periodic
    % exponents fit at 1: the rel-per per-query cost is linear here
    % (K = 12 -> 28 at nQ = 200 multiplies the tuple count by 15 and the
    % time by 9.5), the fitter's profile over 1.0..1.2 moves the centres
    % log-ratio error by under 1 per cent, and 1.0 keeps the K = 40 cell
    % of Section E on centres, which is the measured winner there (6.5 ms
    % against 9.5). What distinguishes the rel-per kernel in MATLAB is
    % its slope, 1.5x the abs-per one, not its exponent. The Section E
    % family never actually misroutes in MATLAB --- centres wins through
    % K = 48, where it ties --- so the audit's K = 34 finding was a
    % Python-only miss.
    MA_COST_CENTRES_QUERY_JOINT_EXP_PER = 1;
    MA_COST_CENTRES_QUERY_PER_JOINT_REL_PER_MS = 3.425e-06;   % was 3.658e-6
    MA_COST_CENTRES_QUERY_JOINT_EXP_REL_PER = 1;
    % Culling geometry factor for the non-periodic kernels. The truncated
    % kernel visits only the centres inside a ball of radius k*sigma, so
    % the surviving fraction is a volume ratio in the attribute's own
    % dimension: (c*sigma/spread) raised to r_a - [rel]_a, capped at 1.
    % Profiled by the September 2026 refit over a log-spaced grid from
    % 4 to 120 and landing on 14.32 --- the same value the Python fit
    % lands on from its own measurements, which is what a geometric
    % factor should do: it describes the truncation ball, not the
    % implementation. (The July 2026 fits agreed with each other in the
    % same way, at 25.6 here and 26.2 in Python; the exponent left free
    % then returned 2.5 and fitted worse than pinning it to the
    % dimension, so the volume reading is the one the measurements
    % prefer.)
    MA_COST_CENTRES_CULL_C = 14.32;   % was 25.6
    % Per-attribute per-query overhead of the factored centres route
    % (bucket lookup and gather). Seeded from the Python fit; re-derive
    % with bench_ma_eval_calibration on this side if picks look off.
    MA_COST_CENTRES_FACTORED_QUERY_BASE_MS = 0.0015;   % was 1.5e-3

    % Refitted on 150 cells of bench_ma_eval_calibration spanning sigma
    % from 3 to 80 cents over spans of 1200 to 9600 cents, absolute and
    % relative, both periodicities, both the spectral and node branches.
    % Predicted over measured went from a geometric mean of 0.73 with a
    % 2.59x spread to 1.00 with a 1.44x spread, worst case 6.4x to 5.0x.
    %
    % The per-query node cost is linear in the quadrature node count, as
    % before. Fitting an exponent on that count returns 1.06 and 0.92 and
    % improves the spread by 0.03, so the linear form stands. An earlier
    % reading of 0.63 came from regressing the measured time on the node
    % count while a constant setup term was still in it: a 5 ms offset
    % turns a true exponent of 1.00 into an apparent 0.66.
    %
    % Möbius: a per-call setup scaling with the partition count B_r, plus
    % per-query work linear in the distinct-block op count (2^r - 1) r K
    % (each distinct block's factor is computed once and reused across
    % partitions). Relative attributes multiply the per-query work by the
    % u-grid node count; each node costs the cheaper of the direct
    % strategy (a fixed per-node floor plus op-count-linear work) and,
    % non-periodically, the factored strategy (K-free after tabulation).
    MA_COST_MOBIUS_SETUP_MS = 0.1321;   % was 0.1200
    MA_COST_MOBIUS_SETUP_PER_BELL_MS = 0.01048;   % was 0.01493
    MA_COST_MOBIUS_QUERY_PER_OP_MS = 1.988e-06;   % was 2.719e-6
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
    % Zero because the fit that produced the constants below omitted
    % this floor, so the per-op term carries whatever it contributed.
    % Carried as a lever rather than removed: if a later calibration
    % finds the node cost floor-bound at low op counts, this is where it
    % goes, and it must then be fitted alongside the per-op term rather
    % than set beside it.
    MA_COST_MOBIUS_REL_NODE_DIRECT_BASE_MS = 0;
    MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_MS = 3.538e-06;   % was 1.237e-6
    MA_COST_MOBIUS_REL_NODE_DIRECT_PER_OP_PER_MS = 1.245e-06;   % was 3.027e-6
    MA_COST_MOBIUS_REL_NODE_FACTORED_PER_BELL_MS = 1.748e-05;   % was 5.936e-5
    % u-grid tabulation setup (once per call): building the interpolation
    % table costs K source evaluations over the N_u grid nodes. MATLAB's
    % lean per-query readback leaves this setup as the dominant Möbius cost
    % at small nQ, so it is modelled explicitly; Python's larger per-query
    % node cost absorbs it, so its twin constant is ~0.
    MA_COST_MOBIUS_REL_TABULATION_PER_NODE_MS = 3.232e-05;   % was 3.516e-5

    BELL = [1 2 5 15 52 203 877 4140 21147 115975];  % B_1..B_10

    A       = double(dens.nAttrs);
    rVec    = double(dens.r(:).');
    kVec    = double(dens.K(:).');
    isRel   = logical(dens.isRel(:).');
    isPer   = logical(dens.isPer(:).');
    sigmaG  = double(dens.sigma(:).');
    periodG = double(dens.period(:).');

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
    % Culled fraction for attribute A: the surviving share of its tuple
    % set per query. Dimensionless, in (0, 1].
    cullOf = @(a) localCentresCull(dens, a, isPer(a), sigmaG(a), ...
                                   rVec(a), isRel(a), ...
                                   MA_COST_CENTRES_CULL_C);

    if factoredSupported
        centresMs = MA_COST_CENTRES_SETUP_MS;
        for a = 1:A
            r_a = rVec(a); K_a = kVec(a);
            T_a = factorial(r_a) * localComb(K_a, r_a);
            if isPer(a)
                qBase = MA_COST_CENTRES_QUERY_BASE_PER_MS;
            else
                qBase = MA_COST_CENTRES_QUERY_BASE_MS;
            end
            [qPer, T_q] = localCentresQuerySlope( ...
                T_a, isPer(a), isRel(a), ...
                MA_COST_CENTRES_QUERY_PER_JOINT_MS, ...
                MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS, ...
                MA_COST_CENTRES_QUERY_JOINT_EXP_PER, ...
                MA_COST_CENTRES_QUERY_PER_JOINT_REL_PER_MS, ...
                MA_COST_CENTRES_QUERY_JOINT_EXP_REL_PER);
            centresMs = centresMs ...
                + MA_COST_CENTRES_CALL_PER_JOINT_MS * T_a ...
                + nQeff * (MA_COST_CENTRES_FACTORED_QUERY_BASE_MS ...
                           + qBase + qPer * T_q * cullOf(a));
        end
    else
        % Joint materialisation. The per-query term takes the geometry of
        % the widest-culling attribute: the joint tuple set is the
        % product across attributes, and a query reaches a joint centre
        % only if it reaches that centre in every attribute, so the
        % joint culled fraction is the product of the per-attribute ones.
        cullJoint = 1.0;
        anyPer = false;
        anyRelPer = false;
        for a = 1:A
            cullJoint = cullJoint * cullOf(a);
            anyPer = anyPer || isPer(a);
            anyRelPer = anyRelPer || (isPer(a) && isRel(a));
        end
        if anyPer
            qBase = MA_COST_CENTRES_QUERY_BASE_PER_MS;
        else
            qBase = MA_COST_CENTRES_QUERY_BASE_MS;
        end
        [qPer, jointQ] = localCentresQuerySlope( ...
            jointTuples, anyPer, anyRelPer, ...
            MA_COST_CENTRES_QUERY_PER_JOINT_MS, ...
            MA_COST_CENTRES_QUERY_PER_JOINT_PER_MS, ...
            MA_COST_CENTRES_QUERY_JOINT_EXP_PER, ...
            MA_COST_CENTRES_QUERY_PER_JOINT_REL_PER_MS, ...
            MA_COST_CENTRES_QUERY_JOINT_EXP_REL_PER);
        centresMs = MA_COST_CENTRES_SETUP_MS ...
            + MA_COST_CENTRES_CALL_PER_JOINT_MS * jointTuples ...
            + nQeff * (qBase + qPer * jointQ * cullJoint);
    end

    mobiusMs = MA_COST_MOBIUS_SETUP_MS;
    for a = 1:A
        r_a = rVec(a);
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
            FOUR_PER_MODE = [1.509e-05, 0.0002111, 0.001757]; % r = 2, 3, 4   % was [2.491e-5, 4.909e-4, 9.671e-3]
            % Periodic-only K term (per r) added to the K-free slope: the
            % per-event spectrum build A_m(eta) = sum_i w^m exp(-i eta p_i)
            % carries K, which the fixed-period window does not absorb. In
            % periodic mode the spectral branch fires for r = 2 and 3 (the
            % r = 4 mode grid exceeds the memory guard and falls to the
            % node path, so its K growth is priced there). Fitted from the
            % periodic engaging cells of bench_ma_eval_calibration.
            FOUR_PERIODIC_K_MS = [3.272e-05, 6.648e-05, 0]; % r = 2, 3, 4   % was [4.836e-5, 1.535e-4, 0.0]
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
end

function [slope, Tq] = localCentresQuerySlope(T, isPerA, isRelA, ...
        qNonPer, qPer, expPer, qRelPer, expRelPer)
%LOCALCENTRESQUERYSLOPE  Per-query centres slope and the count it scales.
%
%   Three kernels, three constants. The non-periodic kernel is
%   bucket-culled and linear in the (culled) tuple count. The two
%   periodic kernels run dense and each carries its own slope and its own
%   exponent on the tuple count: the absolute one measures a wrapped
%   distance per coordinate, the relative one forms wrapped differences
%   first and is the superlinear of the two. The caller applies the
%   culling factor, which is 1 on either periodic kernel.
%
%   Twin of the Python _centres_query_slope.
    if isPerA && isRelA
        slope = qRelPer;  Tq = T ^ expRelPer;
    elseif isPerA
        slope = qPer;     Tq = T ^ expPer;
    else
        slope = qNonPer;  Tq = T;
    end
end


function cullA = localCentresCull(dens, a, isPerA, sigmaA, r_a, isRelA, cullC)
%LOCALCENTRESCULL  Share of an attribute's tuple set a query reaches.
%
%   The truncated non-periodic kernel visits only the centres inside a
%   ball of radius k*sigma about the query, so the surviving share is a
%   volume ratio in the attribute's own dimension, r_a - [rel]_a:
%
%       cull = min(1, (cullC * sigma / spread) ^ dim)
%
%   SPREAD is the attribute's value range, which stands in for the
%   extent the centres occupy. The periodic kernel is not truncated ---
%   it sums over images rather than discarding a tail --- so it takes no
%   discount and CULLA is 1.
    cullA = 1.0;
    if isPerA || sigmaA <= 0
        return;
    end
    spread = 0.0;
    if isfield(dens, 'pAttr') && a <= numel(dens.pAttr) ...
            && ~isempty(dens.pAttr{a})
        arr = double(dens.pAttr{a}(:));
        if ~isempty(arr)
            spread = max(arr) - min(arr);
        end
    end
    if spread <= 0
        return;
    end
    dim = double(r_a) - double(logical(isRelA));
    if dim < 1
        dim = 1;
    end
    cullA = min(1.0, (cullC * double(sigmaA) / spread) ^ dim);
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

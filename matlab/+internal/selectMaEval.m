function [chosen, routingReason] = selectMaEval(dens, verbose)
%SELECTMAEVAL  Cost-model path selection for multi-attribute evalExpTens.
%
%   [CHOSEN, ROUTINGREASON] = INTERNAL.SELECTMAEVAL(DENS) chooses between
%   the joint-centres path (which materialises the joint tuple set) and
%   the factored Möbius evaluator MOBIUS.EVALMAORBIT, returning CHOSEN in
%   {'centres', 'mobius'} and a short ROUTINGREASON.
%
%   No probe. Because the MAET density factorises across attributes
%   (Milne 2026, Eq. maet-density), the joint-centres tuple count and the
%   factored per-attribute cost are both closed-form from the shape and
%   the crossover is sharp, so a pure cost model suffices. The method
%   argument is threaded via DENS or the caller; user overrides are
%   honoured by the caller before this is reached (evalExpTens passes
%   'auto' shapes here).
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
%     - otherwise the cost model: factored orbit cost (sum across
%       attributes) vs joint tuple count (product), with a calibratable
%       dominance margin favouring Möbius (failure-safe).
%
%   Twin of python _select_ma_eval.
%
%   See also MOBIUS.EVALMAORBIT, INTERNAL.SELECTMAINNERPRODUCTMETHOD.

    if nargin < 2, verbose = true; end

    % --- Constants (mirror Python dispatch.py) ---
    ORBIT_R_MAX_FEASIBLE = 10;
    ORBIT_SIGMA_OVER_P_THRESHOLD = 0.03;
    % Calibratable crossover margin favouring the factored Möbius path.
    % CALIBRATED FROM MATLAB TIMINGS (bench_ma_eval_dispatch.m), and
    % deliberately NOT the Python value (2.0): MATLAB's factored path
    % carries a higher fixed per-call overhead (each evalOrbitAbs/Rel
    % call, the u-grid quadrature), so the centres path wins at small
    % shapes where Python's Möbius still won. The measured crossover sits
    % at op-count ratio orbit/joint ~ 0.15-0.20 (Möbius wins below ~0.14,
    % centres wins above ~0.21), so Möbius is chosen only when its cost
    % is well below the joint tuple count. This is the opposite bias to
    % Python and is exactly the per-language calibration the harness
    % exists to establish.
    MA_CENTRES_DOMINANCE = 0.17;
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

    % ---- Cost model: joint tuple count (product) vs factored orbit
    % cost (sum). ----
    jointTuples = 1.0;
    orbitCost = 0.0;
    for a = 1:A
        r_a = rVec(a); K_a = kVec(a);
        if r_a < 1
            continue;
        end
        jointTuples = jointTuples * factorial(r_a) * localComb(K_a, r_a);
        if r_a <= numel(BELL)
            B_r = BELL(r_a);
        else
            B_r = Inf;
        end
        perAttr = B_r * r_a * K_a;
        if isRel(a) && r_a >= 2
            if isPer(a)
                N_u = max(64, ceil(10.0 * periodG(a) / sigmaG(a)));
            else
                N_u = 128;
            end
            perAttr = perAttr * N_u;
        end
        orbitCost = orbitCost + perAttr;
    end

    if orbitCost < jointTuples * MA_CENTRES_DOMINANCE
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

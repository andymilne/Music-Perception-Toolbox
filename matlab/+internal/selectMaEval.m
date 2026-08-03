function [chosen, routingReason, centresMsOut, mobiusMsOut] = ...
        selectMaEval(dens, nQ, truncationSigmas)
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
%     - feasibility bound on any attribute forces the
%       single-image centres route, GUARDED: if its joint tuple set is
%       infeasible to materialise it raises mpt:dispatch:singleImageInfeasible
%       (no cheaper all-image substitute exists there).
%     - relative-periodic above the sigma/P threshold -> the two routes
%       compute different measures, so the wrap axis picks the route:
%       'full-image' (the default) takes the Möbius route for the
%       transposition average, 'single-image' the centres route for the
%       wrapped-difference form. Below the threshold the two agree
%       numerically and the cost model chooses.
%     - otherwise the cost model: two closed-form per-call time estimates
%       (centres materialisation and kernel work, linear in the joint
%       tuple count; factored Möbius setup and per-query distinct-block
%       work), with a safety factor favouring Möbius at near-ties. The
%       functional forms and the calibrated constants live in
%       INTERNAL.MAEVALCOSTSMS.
%
%   [CHOSEN, REASON, CENTRESMS, MOBIUSMS] = ... also returns the two
%   predicted wall times in milliseconds. They are NaN on the hard rules,
%   which return before the cost model is consulted. The rel-per measure
%   rule does not return early: it prices both routes and then overrides
%   the choice, so a report can show what the price would have settled
%   had the measure not settled it first. ROUTINGREASON, not the
%   finiteness of these two, is what distinguishes a priced decision from
%   a structural one.
%
%   Twin of python _select_ma_eval.
%
%   See also INTERNAL.MAEVALCOSTSMS, MOBIUS.EVALMAORBIT,
%   INTERNAL.SELECTMAINNERPRODUCTMETHOD.

    if nargin < 2 || isempty(nQ), nQ = 200; end
    if nargin < 3, truncationSigmas = []; end
    % Guard the removed verbose parameter. This function once took
    % (dens, nQ, verbose, truncationSigmas); verbose was never read, and
    % dispatch decisions are announced by internal.maybeShowDispatchMsg
    % under showHints rather than under a per-call flag. A stale caller
    % passing the old form would otherwise hand a logical to
    % relPerSigmaOverPThreshold, which reads false as 0 and returns the
    % positive-definiteness ceiling in place of the accuracy threshold ---
    % a silently different route. Fail loudly instead.
    if islogical(truncationSigmas)
        error('mpt:selectMaEval:staleCallForm', ...
            ['internal.selectMaEval no longer takes a verbose argument. ' ...
             'Call internal.selectMaEval(dens, nQ, truncationSigmas).']);
    end
    centresMsOut = NaN;   % set below only where the cost model prices
    mobiusMsOut  = NaN;

    % --- Hard-rule constants (mirror Python dispatch.py) ---
    ORBIT_R_MAX_FEASIBLE = 10;
    % Resolved from the accuracy setting rather than fixed: see
    % internal.relPerSigmaOverPThreshold.
    ORBIT_SIGMA_OVER_P_THRESHOLD = ...
        internal.relPerSigmaOverPThreshold(truncationSigmas);

    % Safety factor favouring Möbius at near-ties: Möbius is chosen
    % whenever its estimate is below the centres estimate times this
    % factor. The asymmetry is deliberate -- Möbius is failure-safe (flat,
    % bounded cost) while centres materialises the joint tuple set and can
    % exhaust memory -- so a near-tie breaks toward Möbius.
    MA_MOBIUS_SAFETY                  = 1.5;

    A       = double(dens.nAttrs);
    rVec    = double(dens.r(:).');
    kVec    = double(dens.K(:).');
    isRel   = logical(dens.isRel(:).');
    isPer   = logical(dens.isPer(:).');
    sigmaG  = double(dens.sigma(:).');
    periodG = double(dens.period(:).');

    % ---- Hard rule: ordered ([sym] = 0) attributes at r > 1 -> centres.
    % The Möbius decomposition sums over set partitions of the tuple
    % indices, which counts every ordering of each block and so realises
    % the symmetrised tuple set; on an ordered attribute that is a
    % different density, not a faster route to the same one. r = 1 is
    % exempt ([sym] vacuous at a single value). Twin of the Python
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

    % ---- Hard rule per attribute: feasibility forces the single-image
    % centres route (Möbius beyond its shipped order there). ----
    forceCentresReason = '';
    for a = 1:A
        r_a = rVec(a);
        if r_a < 2
            continue;
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
    % model). Above the sigma/P threshold the two routes compute
    % different measures, so the choice is a choice of measure and the
    % wrap axis makes it: 'full-image' asks for the transposition
    % average, which the Möbius route computes, and 'single-image' for
    % the wrapped-difference form, which the centres route computes. A
    % density carrying no wrap vector takes the 'full-image' default.
    % Below the threshold the two agree numerically and the cost model
    % chooses. Twin of the Python rule in _select_ma_eval; the cosine
    % selector applies the same rule via
    % internal.selectMaInnerProductMethod.
    %
    %  The rule does not return here. Both routes are priced first, so a
    %  report can say what the cost model would have chosen and the
    %  reader can see that the measure, not the price, settled it. The
    %  choice is overridden after pricing, below. ----
    measureForcesMobius = false;
    measureForcesCentres = false;
    hasWrap = isfield(dens, 'wrap') && ~isempty(dens.wrap);
    for a = 1:A
        if isRel(a) && isPer(a) && periodG(a) > 0 ...
                && sigmaG(a) / periodG(a) > ORBIT_SIGMA_OVER_P_THRESHOLD
            wrapA = 'full-image';
            if hasWrap && a <= numel(dens.wrap)
                wrapA = char(dens.wrap{a});
            end
            if strcmp(wrapA, 'single-image')
                measureForcesCentres = true;
            else
                measureForcesMobius = true;
            end
            break;
        end
    end

    % ---- Cost model: the two closed-form per-call time estimates
    % (ms). The functional forms and the calibrated constants live in
    % internal.maEvalCostsMs, which EXPLAINDISPATCH also calls so that
    % the report and the decision price identical work. ----
    [centresMs, mobiusMs] = internal.maEvalCostsMs(dens, nQ);

    centresMsOut = centresMs;
    mobiusMsOut  = mobiusMs;
    if measureForcesCentres
        chosen = 'centres';
        routingReason = 'rel-per single-image measure (wrap opt-in)';
    elseif measureForcesMobius
        chosen = 'mobius';
        routingReason = 'rel-per full-image measure';
    elseif mobiusMs < centresMs * MA_MOBIUS_SAFETY
        chosen = 'mobius';
        routingReason = 'cost model (factored Möbius cheaper)';
    else
        chosen = 'centres';
        routingReason = 'cost model (joint centres cheaper)';
    end
end

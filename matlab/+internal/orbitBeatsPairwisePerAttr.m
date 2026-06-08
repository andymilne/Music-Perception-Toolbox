function tf = orbitBeatsPairwisePerAttr(r, K, isRel, isPer)
%ORBITBEATSPAIRWISEPERATTR  Per-attribute K-vs-r Möbius/Bulger crossover.
%   Mirror of Python dispatch._orbit_beats_pairwise_per_attr. A legacy
%   crossover heuristic (the cost-model dispatcher is preferred for the
%   flat MA path); the per-level nested orbit choice reuses it with K = g.
    ORBIT_R_MAX_SHIPPED = 8;    % match Python _ORBIT_R_MAX_SHIPPED
    if r == 1 || r > ORBIT_R_MAX_SHIPPED
        tf = false;
        return;
    end
    if isRel && isPer
        thr = [Inf, Inf, 10, 8, 7, 6];   % _K_THRESHOLD_REL_PER, index r=2..6
    else
        thr = [Inf, 7, 6, 5, 4, 4];      % _K_THRESHOLD_ABS, index r=2..6
    end
    if r <= 6
        t = thr(r);
    else
        t = 999;                          % r=7,8: dict-miss -> .get(r,999)
    end
    tf = (K >= t);
end

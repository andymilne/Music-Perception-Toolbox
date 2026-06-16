function tf = orbitBeatsPairwisePerAttr(r, K, isRel, isPer)
%ORBITBEATSPAIRWISEPERATTR  Per-level orbit-vs-enumeration decision for one
%   symmetric attribute or nesting level, from the mode-aware K thresholds.
%   Mirror of Python dispatch._orbit_beats_pairwise_per_attr.
%
%   This is the live predicate for the per-level decision in the nested
%   contraction (internal.nestedContract): it runs once per level, so it uses
%   a cheap mode-aware threshold rather than a probe. The flat multi-attribute
%   and single-attribute inner-product paths instead make a whole-call decision
%   through the cost model and probe (internal.selectMaInnerProductMethod and
%   localSelectAndEstimateSAIP in cosSimExpTens), which also weigh N. The two
%   mechanisms are matched to their contexts, not redundant: the per-level
%   predicate cannot afford a probe, and its thresholds encode the
%   relative-periodic u-grid overhead that an op-count comparison would miss.
%   tf = true means orbit (Möbius) is the cheaper route here.
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

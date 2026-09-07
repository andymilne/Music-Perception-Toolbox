function [x, so] = dyadSweep(rInner, useFlag, mus, normalize)
%DYADSWEEP  One-sided similarity of the dyad-skeleton query across the chorale.
%
%   [x, so] = jmm.dyadSweep(rInner, useFlag, mus)
%   [x, so] = jmm.dyadSweep(rInner, useFlag, mus, normalize)
%
%   One-sided similarity of the dyad-skeleton query (jmm.dyadQuery)
%   against the chorale, swept over candidate resolution moments mus
%   (every beat, QN); useFlag adds the pitch-derived root-position flag.
%   Returns the played-through bar coordinate x = jmm.b2bar(mus) and the
%   profile so (NaN where the sweep does not reach; 0 where a window
%   lacks events). normalize is 'oneSidedDenom' (default) or 'cosine'.
%
%   Twin of dyad_sweep in demo_jmm_1_4_cadence_nesting.py.
%
%   See also JMM.PROTOTYPESWEEP, JMM.DYADQUERY, JMM.BUILDPAIR.
    if nargin < 4 || isempty(normalize), normalize = 'oneSidedDenom'; end
    S = jmm.bwvWindowState();
    if useFlag, qFlag = S.rootYes; else, qFlag = []; end
    qd = jmm.dyadQuery(qFlag, rInner);
    so = nan(1, numel(mus));
    wins = {}; idxs = [];
    for k = 1:numel(mus)
        mu = mus(k);
        if mu - 1.0 < S.T0 - 1e-9 || mu + 1.0 > S.T1 + 1e-9
            continue;
        end
        % The two 1-QN halves either side of mu: the approach chord(s) in
        % [mu-1, mu) and the resolution chord(s) in [mu, mu+1).
        c1 = jmm.winEvents(mu - 1.0, mu);
        c2 = jmm.winEvents(mu, mu + 1.0);
        if numel(c1.times) < 2 || numel(c2.times) < 2
            so(k) = 0.0;
            continue;
        end
        % The optional inversion attribute is pitch-derived: a predicate on
        % the sonority sounding at mu (no harmonic labels are consulted).
        flag = [];
        if useFlag
            if jmm.isRootPosition(jmm.sonAt(mu)), flag = S.rootYes;
            else,                                  flag = S.rootNo; end
        end
        wins{end + 1} = jmm.buildPair(c1, c2, flag, rInner); %#ok<AGROW>
        idxs(end + 1) = k; %#ok<AGROW>
    end
    % One batched call: density list vs single query, one-sided
    % (query-normalized) similarity.
    vals = cell2mat(cosSimExpTens(wins, qd, 'normalize', normalize, ...
                                  'verbose', false));
    so(idxs) = vals;
    x = jmm.b2bar(mus);
end

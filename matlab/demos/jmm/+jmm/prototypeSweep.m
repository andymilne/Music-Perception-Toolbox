function profiles = prototypeSweep(rInner, queries, mus, normalize)
%PROTOTYPESWEEP  One-sided similarity profiles of the three-chord queries.
%
%   profiles = jmm.prototypeSweep(rInner, queries, mus)
%   profiles = jmm.prototypeSweep(rInner, queries, mus, normalize)
%
%   One-sided similarity profiles of all three-chord queries at one inner
%   r: a 1 x Q cell, entry q the (1 x numel(mus)) profile of query q.
%   queries is a 1 x Q struct array with fields .chords (1 x 3 cell of
%   MIDI pitch vectors) and .flagged (logical); mus the candidate
%   resolution moments (QN); normalize 'oneSidedDenom' (default) or
%   'cosine'. Positions whose windows lack events score 0. The chorale's
%   aligned span at resolution moment mu is the three beat aggregates
%   [mu-2, mu-1), [mu-1, mu), [mu, mu+1). One batched density-list call
%   per query.
%
%   Twin of prototype_sweep in demo_jmm_1_4_cadence_nesting.py.
%
%   See also JMM.DYADSWEEP, JMM.QUERYDENSITY, JMM.BOUNDDENSITY.
    if nargin < 4 || isempty(normalize), normalize = 'oneSidedDenom'; end
    S = jmm.bwvWindowState();
    ctxPlain = {}; ctxFlag = {}; idxs = [];
    for k = 1:numel(mus)
        mu = mus(k);
        if mu - 2.0 < S.T0 - 1e-9 || mu + 1.0 > S.T1 + 1e-9
            continue;
        end
        wins = {jmm.winEvents(mu - 2.0, mu - 1.0), jmm.winEvents(mu - 1.0, mu), ...
                jmm.winEvents(mu, mu + 1.0)};
        if any(cellfun(@(w) numel(w.times) < 2, wins))
            continue;
        end
        aggs = [jmm.aggregate(wins{1}), jmm.aggregate(wins{2}), ...
                jmm.aggregate(wins{3})];
        % The inversion flag is pitch-derived: a predicate on the sonority
        % at the antepenult beat (no harmonic labels are consulted).
        if jmm.isSixFour(jmm.sonAt(mu - 2.0)), flag = S.rootYes;
        else,                                   flag = S.rootNo; end
        ctxPlain{end + 1} = jmm.boundDensity(aggs, [], rInner); %#ok<AGROW>
        ctxFlag{end + 1} = jmm.boundDensity(aggs, flag, rInner); %#ok<AGROW>
        idxs(end + 1) = k; %#ok<AGROW>
    end
    profiles = cell(1, numel(queries));
    for q = 1:numel(queries)
        qd = jmm.queryDensity(queries(q).chords, queries(q).flagged, rInner);
        if queries(q).flagged, ctx = ctxFlag; else, ctx = ctxPlain; end
        vals = cell2mat(cosSimExpTens(ctx, qd, 'normalize', normalize, ...
                                      'verbose', false));
        prof = zeros(1, numel(mus));
        prof(idxs) = vals;
        profiles{q} = prof;
    end
end

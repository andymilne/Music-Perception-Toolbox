function dens = boundDensity(aggs, flag, rInner)
%BOUNDDENSITY  MAET density of one bound super-event from L beat aggregates.
%
%   dens = jmm.boundDensity(aggs)
%   dens = jmm.boundDensity(aggs, flag)
%   dens = jmm.boundDensity(aggs, flag, rInner)
%
%   The pitch attribute nests the aggregates: inner level the chord
%   multiset ([sym] = 1, r = rInner, default 1), outer level the L
%   ordered aggregates ([sym] = 0, r = L), relative at the outer level
%   alone ([rel] = (0, 1)), periodic at the octave. An optional flag value
%   ([] for none) adds the simplex-coded inversion attribute.
%
%   aggs is a 1 x L struct array with fields .p and .w (jmm.aggregate).
%   Unequal-K aggregates are NaN-padded to a common K (the standard
%   pre-MAET convention for variable per-event cardinality).
%
%   Twin of bwv_window.bound_density in the Python demos.
%
%   See also BINDEVENTS, FLATSPECS, BUILDEXPTENS, JMM.AGGREGATE.
    if nargin < 2, flag = []; end
    if nargin < 3 || isempty(rInner), rInner = 1; end
    S = jmm.bwvWindowState();
    L = numel(aggs);
    kMax = max(arrayfun(@(a) numel(a.p), aggs));
    P = nan(kMax, L);
    W = nan(kMax, L);
    for j = 1:L
        P(1:numel(aggs(j).p), j) = aggs(j).p(:);
        W(1:numel(aggs(j).w), j) = aggs(j).w(:);
    end
    specs = flatSpecs({P}, 'r', rInner, 'rel', false, 'sym', true, ...
                      'name', 'pitch');
    [pb, wb, sb] = bindEvents({P}, {W}, L, 'relOuter', true, 'specs', specs);
    attrs = {pb{1}}; ws = {wb{1}}; sp = {sb{1}};
    sigma = S.sigmaPitch; isPer = true; period = S.period;
    if ~isempty(flag)
        attrs{end + 1} = double(flag);
        ws{end + 1} = 1.0;
        fs = flatSpecs({attrs{end}}, 'r', 1, 'rel', false, 'sym', false, ...
                       'name', 'flag');
        sp{end + 1} = fs{1};
        sigma(end + 1) = S.sigmaFlag;
        isPer(end + 1) = false;
        period(end + 1) = 0.0;
    end
    dens = buildExpTens(attrs, ws, 'specs', sp, 'sigma', sigma, ...
                        'isPer', isPer, 'period', period, 'verbose', false);
end

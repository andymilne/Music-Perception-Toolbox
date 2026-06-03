function dens = ensureExpTensExpensive(dens)
%ENSUREEXPTENSEXPENSIVE  Populate per-tuple fields on a skinny density.
%
%   DENS = ENSUREEXPTENSEXPENSIVE(DENS) returns DENS with the per-tuple
%   "expensive" fields populated. If DENS already has those fields
%   (e.g. it came from buildExpTens with 'lazy', false), the call is a
%   no-op pass-through.
%
%   This is the helper used by consumer entry points (cosSimExpTens,
%   evalExpTens, windowedSimilarity, etc.) on density inputs.
%   buildExpTens defaults to 'lazy', true, returning a skinny density
%   that exposes only cheap fields (p/pAttr, w, sigma, r, isRel,
%   isPer, period, dim, etc.). Orbit-method consumers operate
%   directly on the cheap fields and skip this helper; pairwise/centre
%   consumers prepend a single call to it.
%
%   Cheap fields (always present after buildExpTens):
%     SA tag 'ExpTensDensity':
%       p, w, sigma, r, isRel, isPer, period, dim
%     MA tag 'MaetDensity':
%       nAttrs, N, r, K, pAttr,
%       w, sigma, isRel, isPer, period, dim, dimPerAttr
%
%   Expensive fields (populated by this helper):
%     SA: Centres, wJ, nJ, U_perm, w_perm, nJ_perm, V_comb, wv_comb, nK
%     MA: Centres, U_perm, V_comb, wJ, wv_comb, nJ, nK,
%         eventOfJ, eventOfK
%
%   Idempotence: detection is by the presence of `Centres`. Calling on a
%   fully-populated density returns it unchanged.
%
%   See also buildExpTens, cosSimExpTens, evalExpTens.

    if ~isstruct(dens)
        error('ensureExpTensExpensive:badInput', ...
              'Input must be a density struct from buildExpTens.');
    end
    if ~isfield(dens, 'tag')
        error('ensureExpTensExpensive:missingTag', ...
              'Input is not a density struct (no .tag field).');
    end

    % Already populated: return as-is.
    if isfield(dens, 'Centres') && ~isempty(dens.Centres)
        return
    end

    switch dens.tag
        case 'ExpTensDensity'
            dens = buildExpTens( ...
                dens.p, dens.w, dens.sigma, dens.r, ...
                dens.isRel, dens.isPer, dens.period, ...
                'lazy', false, 'verbose', false);

        case 'MaetDensity'
            dens = buildExpTens( ...
                dens.pAttr, dens.w, dens.sigma, dens.r, ...
                dens.isRel, dens.isPer, dens.period, ...
                'lazy', false, 'verbose', false);

        otherwise
            error('ensureExpTensExpensive:badTag', ...
                  'Unknown density tag: %s', dens.tag);
    end
end

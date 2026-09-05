function [bundle, cache] = nestedCentresMemoised(cache, dens, a)
%INTERNAL.NESTEDCENTRESMEMOISED  Memoised tuple-centres bundle for attribute a.
%
%   [BUNDLE, CACHE] = INTERNAL.NESTEDCENTRESMEMOISED(CACHE, DENS, A)
%   returns MOBIUS.CLOSEDFORMATTRCENTRES(DENS, A), read from the memo
%   CACHE when it already holds attribute A's bundle and built (and
%   stored) otherwise. CACHE is a density's self-IP memo struct (the
%   'keys' / 'vals' struct COSSIMEXPTENS threads through every route and
%   returns in the density's 'selfIP' field); the bundles live in its
%   optional 'nestedCentres' field, a 1 x A cell with [] where no bundle
%   has been built, so they ride the same channel back to the caller as
%   the memoised self inner products do and persist across calls once
%   the caller passes the returned struct back in.
%
%   The rebuild depends only on the density's own contents (pruning
%   drops zero-weight events, a pure function of those contents), and
%   the nested cosine asks for it once per inner product in its
%   (xy, xx, yy) triple --- three times per density per call, plus once
%   more on every later call against the same density. The rebuild
%   materialises the attribute's permutation arrays, which at small r is
%   the dominant cost of the whole nested route. Twin of the Python
%   density attribute _nested_centres_cache read by
%   _mobius_inner._closed_form_attr_centres.
%
%   See also MOBIUS.CLOSEDFORMATTRCENTRES, INTERNAL.SELFIPKEY,
%   INTERNAL.NESTEDCONTRACT.

    if ~isstruct(cache) || ~isfield(cache, 'keys') || ~isfield(cache, 'vals')
        cache = struct('keys', {{}}, 'vals', []);
    end
    if ~isfield(cache, 'nestedCentres') || ~iscell(cache.nestedCentres)
        cache.nestedCentres = {};
    end
    if numel(cache.nestedCentres) >= a && ~isempty(cache.nestedCentres{a})
        bundle = cache.nestedCentres{a};
        return;
    end
    bundle = mobius.closedFormAttrCentres(dens, a);
    cache.nestedCentres{a} = bundle;
end

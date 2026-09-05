function hit = selfIpMemoised(cache)
%INTERNAL.SELFIPMEMOISED  Has any route memoised this density's self IP?
%
%   HIT = INTERNAL.SELFIPMEMOISED(CACHE) is true when CACHE holds an entry
%   written by any of the inner-product routes that memoise a self inner
%   product: Bulger's enumeration, the tuple-centres enumeration, the
%   per-attribute Möbius matrices, and the nested contraction's
%   single-attribute and multi-attribute forms. The sweep path's own
%   'sweep' memo (written by SWEEPCOSSIMEXPTENS's mixture route under
%   INTERNAL.SELFIPKEY('sweep', ...), as the Python sweep writes it) is
%   deliberately not among them: it is produced by a different evaluator
%   and consumed by neither route here, so it would spare neither of
%   them any work.
%
%   The flag is shared by the routes a selector compares, rather than read
%   off each route's own memo, and that is deliberate. The memoised
%   *values* are per route (see INTERNAL.SELFIPKEY), so a route that finds
%   only another route's memo will still recompute its own self matrices
%   on this call. Pricing each route against its own memo nonetheless
%   makes the comparison unfair in a way that compounds: the first call
%   seeds only the winner's memo, so on the second call the winner is
%   priced at one matrix and the loser at three, and the choice locks in
%   even where the loser, once warm, is the cheaper route. Sharing the
%   flag prices the comparison on the routes' per-matrix costs, which is
%   what the selector is meant to decide on.
%
%   The trade is per-call: on the one call where the comparison flips, the
%   newly chosen route does pay for the self matrices the flag priced as
%   free. It memoises them, so the flag is honest from the next call
%   onwards; the mispricing is bounded by a single call per crossover, and
%   it buys amortised correctness over the repeated calls a sweep makes.
%
%   Twin of the Python cosine._self_ip_memoised.
%
%   See also INTERNAL.SELFIPKEY, INTERNAL.SELECTMAINNERPRODUCTMETHOD.

    routes = {'bulger', 'centres', 'mobius', 'contract', 'contract_ma'};
    hit = false;
    if ~isstruct(cache) || ~isfield(cache, 'keys') || isempty(cache.keys)
        return;
    end
    for i = 1:numel(routes)
        r = routes{i};
        if any(strncmp(cache.keys, [r '|'], numel(r) + 1))
            hit = true;
            return;
        end
    end
end

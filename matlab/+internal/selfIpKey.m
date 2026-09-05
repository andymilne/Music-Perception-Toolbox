function key = selfIpKey(route, tsResolved, extra)
%INTERNAL.SELFIPKEY  Memo key for a self inner product.
%
%   KEY = INTERNAL.SELFIPKEY(ROUTE, TSRESOLVED, EXTRA) builds the string
%   key under which <X,X> and <Y,Y> are memoised on a density's selfIP
%   cache.
%
%   The key carries everything the value depends on beyond the density's
%   own contents: the route, the resolved truncation budget, and any
%   route-specific choices (EXTRA --- the Möbius route's per-attribute
%   closed-form-vs-grid selections, or the nested contraction's
%   per-attribute routes and shared tau-grid signature, each of which
%   changes the per-attribute prefactor or the discretisation).
%   Chunking granularity is deliberately not keyed: it perturbs only the
%   floating-point accumulation order, within the toolbox-wide <= 1e-12
%   parity discipline.
%
%   Why the route stays in the key, when the routes' scales are related
%   in closed form. The bare triples differ by a constant that cancels
%   within one route's triple, and the constant is known exactly: per
%   attribute, Bulger's enumeration against the Möbius per-attribute
%   matrix is r_a! (sigma_a sqrt(pi))^r_a in an absolute mode and
%   r_a! (sigma_a sqrt(pi))^(r_a - 1) sqrt(r_a) in relative
%   non-periodic; against the tuple-centres closed form (and the
%   unrestricted centres route) it is r_a!; against a nested attribute's
%   per-level contraction it is 1, and against the nested centres route
%   the wreath-product orbit order of INTERNAL.NESTEDORBITMULT.
%   Converting a memo to a canonical scale is therefore arithmetically
%   possible.
%
%   It is not numerically possible. Each route applies the truncation
%   budget to its own arrays --- a different threshold tightening for a
%   different array size, a different set of kernel entries dropped, and
%   for the Möbius route an alternating sum where the enumeration has a
%   plain one --- so after the exact rescaling the routes hold different
%   numbers, not the same number in different units. Measured on the
%   self inner product at the shipped 6-sigma default: the Möbius route
%   departs from Bulger's by up to 3e-9 relative (absolute non-periodic,
%   r = 2..3, K = 6..9), the centres route by up to 1e-12; at
%   truncationSigmas = 4 those become 9e-5 and 2e-8, and at the accuracy
%   floor (inf) they fall to 1e-13 and 1e-16. A shared memo would put
%   that difference into the returned cosine whenever a route consumed a
%   value another route produced, so two identical calls with the same
%   forced METHOD would return different numbers depending on what ran
%   before them. Route-keyed values keep each route's answer
%   reproducible; the *pricing* is shared instead, via
%   INTERNAL.SELFIPMEMOISED.
%
%   One case is a difference of measure rather than of accuracy, and is
%   not even arithmetically convertible: on a relative-periodic attribute
%   the tau-grid computes the all-image transposition average (C) while
%   the enumeration and the tuple-centres closed form compute the
%   minimum-image reading (A). Below the sigma/period threshold they
%   agree inside the truncation floor but are still not the same number
%   (measured 1.9e-5 relative at sigma/P = 0.058, 4.8e-2 at 0.125), and
%   above it they are different quantities.
%
%   Promoted out of COSSIMEXPTENS's local LOCALSELFIPKEY so that
%   INTERNAL.NESTEDCONTRACT, which memoises into the same caches, spells
%   the key the same way; LOCALSELFIPKEY now delegates here. Twin of the
%   Python cosine._self_ip_cache_key.
%
%   See also COSSIMEXPTENS, INTERNAL.NESTEDCONTRACT,
%   INTERNAL.SELFIPMEMOISED.

    if nargin < 3 || isempty(extra)
        extra = '';
    end
    key = sprintf('%s|%.17g|%s', route, tsResolved, extra);
end

function out = lastNestedRoutes(routes)
%INTERNAL.LASTNESTEDROUTES  Per-attribute nested routes of the most recent
%   nested-contraction call.
%
%   INTERNAL.LASTNESTEDROUTES(ROUTES) records ROUTES (a cell row of
%   route names in attribute order, '-' for an attribute that took
%   neither a nested nor an ordered-flat route).
%   ROUTES = INTERNAL.LASTNESTEDROUTES() reads the record back; it is
%   {} before any nested call in this session.
%
%   Diagnostic only. It feeds the dispatch-decision message's routing
%   reason and gives the tests a way to assert which route ran without
%   timing it. Nothing routes on it, and no result depends on it.
%
%   Mirror of the Python cosine._LAST_NESTED_ROUTES module variable.
%
%   See also INTERNAL.NESTEDCONTRACT, COSSIMEXPTENS.

    persistent stored
    if isempty(stored)
        stored = {};
    end
    if nargin > 0
        if isempty(routes)
            stored = {};
        else
            stored = routes(:).';
        end
    end
    out = stored;
end

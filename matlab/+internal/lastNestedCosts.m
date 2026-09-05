function out = lastNestedCosts(costs)
%INTERNAL.LASTNESTEDCOSTS  Prices behind the most recent nested
%   plan-versus-enumeration decision.
%
%   INTERNAL.LASTNESTEDCOSTS(COSTS) records COSTS, a struct with fields
%   chosen ('contract' or 'bulger'), planMs, enumMs and detail (the
%   per-attribute detail INTERNAL.NESTEDCOST('selectNestedMethod')
%   returns).
%   COSTS = INTERNAL.LASTNESTEDCOSTS() reads the record back; it is []
%   before any nested call in this session, and
%   INTERNAL.LASTNESTEDCOSTS([]) clears it.
%
%   Diagnostic only, exactly as INTERNAL.LASTNESTEDROUTES is: it gives
%   the tests a way to assert which side the cost model took, and how
%   close it was, without timing anything. Nothing routes on it, and no
%   result depends on it. EXPLAINDISPATCH reports the same quantities by
%   re-running the model rather than by reading this.
%
%   Mirror of the Python cosine._LAST_NESTED_COSTS module variable.
%
%   See also INTERNAL.LASTNESTEDROUTES, INTERNAL.NESTEDCOST.

    persistent stored
    if nargin > 0
        if isempty(costs)
            stored = [];
        else
            stored = costs;
        end
    end
    out = stored;
end

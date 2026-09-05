function out = nestedCostOverride(prices)
%INTERNAL.NESTEDCOSTOVERRIDE  TEST-ONLY price stub for the nested cost model.
%
%   INTERNAL.NESTEDCOSTOVERRIDE(PRICES) installs PRICES, a struct with
%   one field per route name ('centres', 'taugrid',
%   'contract_relnonper', 'contract', 'bulger') holding a fixed
%   millisecond price. While one is installed,
%   INTERNAL.NESTEDCOST('routeCostMs', ...) returns that price for a
%   named route instead of evaluating its fitted law; a route the struct
%   does not name is priced by its law as usual.
%   PRICES = INTERNAL.NESTEDCOSTOVERRIDE() reads the record back, and
%   INTERNAL.NESTEDCOSTOVERRIDE([]) clears it.
%
%   TEST-ONLY. Nothing in the toolbox installs an override, and no
%   public entry point exposes one. It exists so that
%   TESTS/TEST_NESTED_COST_MODEL can pin what the cost model *does* with
%   a price without depending on what the shipped constants *say* the
%   price is --- the MATLAB analogue of the Python tests'
%   ``monkeypatch.setattr(_nc, "nested_route_cost_ms", ...)``. A test
%   that installs one must clear it on cleanup (onCleanup), or every
%   later call in the session is priced by the stub.
%
%   No stub can move a route that carries a different measure: the
%   admissible set is settled by the measure rule before any price is
%   read, which is exactly what those tests check.
%
%   See also INTERNAL.NESTEDCOST, TESTS/TEST_NESTED_COST_MODEL.

    persistent stored
    if nargin > 0
        if isempty(prices)
            stored = [];
        else
            stored = prices;
        end
    end
    out = stored;
end

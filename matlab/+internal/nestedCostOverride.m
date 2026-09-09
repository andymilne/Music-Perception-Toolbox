function out = nestedCostOverride(prices)
%INTERNAL.NESTEDCOSTOVERRIDE  TEST-ONLY cost stub for the nested cost model.
%
%   INTERNAL.NESTEDCOSTOVERRIDE(PRICES) installs PRICES, a struct with
%   one field per route name ('centres', 'taugrid',
%   'contract_relnonper', 'contract', 'bulger') holding a fixed
%   millisecond cost. While one is installed,
%   INTERNAL.NESTEDCOST('routeCostMs', ...) returns that cost for a
%   named route instead of evaluating its fitted law; a route the struct
%   does not name is estimated by its law as usual.
%   PRICES = INTERNAL.NESTEDCOSTOVERRIDE() reads the record back, and
%   INTERNAL.NESTEDCOSTOVERRIDE([]) clears it.
%
%   TEST-ONLY. Nothing in the toolbox installs an override, and no
%   public entry point exposes one. It exists so that
%   TESTS/TEST_NESTED_COST_MODEL can pin what the cost model *does* with
%   an estimate without depending on what the shipped constants *say* the
%   cost is --- the MATLAB analogue of the Python tests'
%   ``monkeypatch.setattr(_nc, "nested_route_cost_ms", ...)``. A test
%   that installs one must clear it on cleanup (onCleanup), or every
%   later call in the session is estimated by the stub.
%
%   No stub can move a route that carries a different measure: the
%   admissible set is settled by the measure rule before any estimated cost is
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

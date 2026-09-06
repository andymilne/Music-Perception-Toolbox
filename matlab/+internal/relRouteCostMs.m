function ms = relRouteCostMs(route, r_a, term)
%INTERNAL.RELROUTECOSTMS  Predicted wall time (ms) for one flat route.
%
%   MS = INTERNAL.RELROUTECOSTMS(ROUTE, R_A, TERM) prices one route of
%   the flat multi-attribute inner product from its fitted power law.
%
%   Promoted out of INTERNAL.SELECTMAINNERPRODUCTMETHOD's local function
%   of the same name so that INTERNAL.PREDICTORBITCOSTMS --- itself
%   promoted, because the nested cost model prices a nested density's
%   flat companion attributes with it --- reads one set of constants.
%   The selector now delegates here. Twin of the Python
%   dispatch._rel_route_cost_ms.
%
%   Cost model for the method comparison: one power law per route and
%   tuple order,
%
%       t_ms = exp(a_r) * term ^ b_r
%
%   on the quantity each route works over -- Bulger's method and the
%   tuple-centres route on the tuple-pair entries they materialise, the
%   translation grid on the node count times the larger value count.
%   Every term carries the event-pair count, since both methods price
%   per pair. The Mobius side takes the smaller of its two routes, as
%   the orchestrator does. Each law is fitted against the quantity the
%   caller passes, not an idealisation of it, so the intercepts absorb
%   the constant factors between them; a refit must use the same terms.
%   Orders above 4 reuse the r = 4 row.
%
%   Fitted on 684 cells: r in {2, 3, 4}, value counts 5 to 40, event
%   counts 1 to 64, three kernel widths, both periodicities, three
%   weight profiles, equal and unequal value counts, each route timed in
%   isolation with relAttrRoute pinning it. Cross-validated on the
%   routing decision, eight-fold: 0.93 against 0.62 for the count-based
%   model it replaces.
%
%   Constants are per-language: the two implementations amortise
%   differently. Refit with tools/calibrateRelIpCost.m, and run it with
%   'check', true first -- three earlier fits shipped badly because an
%   axis was missing from the sweep, and the check asserts that each
%   axis varies what it claims to.
%
%   See also INTERNAL.SELECTMAINNERPRODUCTMETHOD, INTERNAL.PREDICTORBITCOSTMS.
    switch route
        case 'bulger'
            % Intercepts at r = 2 and r = 3 re-anchored on the 2052-cell
            % calibration of 6 September 2026 (tools/calibrateRelIpCost.m,
            % seeds 1-3): a per-order multiplicative correction to the
            % Bulger prediction, chosen to minimise routing regret, lowers
            % the held-out regret over random halves from 30 s to 2.3 s
            % (factors 0.459 at r = 2 and 0.522 at r = 3; r = 4 unchanged).
            % A plain log-log refit of all nine laws on the same cells
            % scores seven times the shipped regret, so the exponents stand.
            A = [-7.9936, -8.6669, -7.7822];
            B = [ 0.6970,  0.7493,  0.7500];
        case 'centres'
            A = [-7.7429, -8.6263, -8.8269];
            B = [ 0.6800,  0.7781,  0.8029];
        case 'grid'
            A = [-4.1388, -2.1985, -0.5409];
            B = [ 0.3916,  0.5021,  0.5768];
        otherwise
            error('mpt:badRoute', 'Unknown route ''%s''.', route);
    end
    idx = min(max(r_a, 2), 4) - 1;
    ms = exp(A(idx)) * max(term, 1)^B(idx);
end

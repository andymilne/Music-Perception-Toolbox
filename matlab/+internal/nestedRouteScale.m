function f = nestedRouteScale(dens, a, route)
%NESTEDROUTESCALE  Factor putting nested attribute A's bare matrix on ROUTE
%on the canonical scale.
%
%   f = internal.nestedRouteScale(dens, a, route)
%
%   With s the leaf positions of one tuple (prod of the per-level r), |G|
%   the wreath-product orbit order of INTERNAL.NESTEDORBITMULT and g the
%   block-metric prefactor of INTERNAL.IPCANONICALSCALE:
%     - 'centres': g (every arrangement on both sides, kernel peak 1);
%     - 'bulger' and 'contract': |G| g (one side restricted to one
%       combination per orbit);
%     - 'contract_relnonper': |G| s (sigma sqrt(pi))^(s-2) / 2, the
%       integral over the alignment line of the product of the s leaf
%       kernels, which is 2 sigma sqrt(pi / s) times the centres kernel;
%     - 'taugrid': P times that, the mean over the period of the same
%       integrand.
%
%   The routes therefore differ by exactly known constants, and the nested
%   combiners of INTERNAL.NESTEDCONTRACT divide them out (to the 'centres'
%   scale) before combining, so a ratio may draw its terms from different
%   routes. Twin of the Python cosine._nested_route_scale; pinned against
%   first-principles references by the Python
%   tests/test_nested_route_scale.py.

    spec = dens.nested{a};
    sp = double(dens.sigma(a)) * sqrt(pi);
    rLevels = double(spec.r(:).');
    sTot = prod(rLevels);
    relUnit = [];
    if isfield(spec, 'relUnit') && ~isempty(spec.relUnit) ...
            && ~any(isnan(spec.relUnit)) && spec.relUnit > 0
        relUnit = double(spec.relUnit);
    end
    if isempty(relUnit)
        g = sp^sTot;
    else
        sU = prod(rLevels(1:relUnit));
        g = (sp^(sU - 1) * sqrt(sU))^(sTot / sU);
    end
    G = double(internal.nestedOrbitMult(rLevels, logical(spec.exch(:).')));
    switch route
        case 'centres'
            f = g;
        case 'contract_relnonper'
            % Trapezoid over the alignment line: an integral over the
            % translation of the s leaf kernels, in the ones direction.
            f = G * sTot * sp^(sTot - 2) / 2;
        case 'taugrid'
            % Mean over the period of the same integrand.
            f = double(dens.period(a)) * G * sTot * sp^(sTot - 2) / 2;
        case {'contract', 'bulger'}
            f = G * g;
        otherwise
            error('mpt:internal:unknownNestedRoute', ...
                'Unknown nested route ''%s''.', route);
    end
end

function scale = ipCanonicalScale(dens, chosen, nestedRoutes)
%IPCANONICALSCALE  Factor putting a route's bare inner product on the
%canonical scale.
%
%   scale = internal.ipCanonicalScale(dens, chosen, nestedRoutes)
%
%   The canonical scale is the physical one: <X, Y> = integral T_X T_Y
%   with each event's density the sum, over the attribute's full ordered
%   tuple set (every arrangement a symmetric level admits), of
%   unnormalised Gaussian kernels exp(-Q(x - c) / 2 sigma^2). It is the
%   scale on which the Möbius per-attribute matrix and the closed-form
%   total masses already agree, and so the one the Rényi-2 entropy has
%   always been computed on.
%
%   Every route drops constant per-attribute prefactors because they
%   cancel in a ratio. Per attribute, with g_a = (sigma sqrt(pi))^d for an
%   absolute attribute of tuple dimension d and, for a relative one whose
%   co-transposition blocks have s_u slots, g_a = [(sigma sqrt(pi))^(s_u-1)
%   sqrt(s_u)]^(number of blocks), the factors are:
%     - flat Möbius matrix: 1;
%     - flat centres closed form, and the ordered-flat centres inside the
%       nested plan: g_a;
%     - Bulger's enumeration: r! g_a on a symmetric flat attribute, g_a on
%       an ordered one, |G| g_a on a nested one (|G| the wreath-product
%       orbit order of INTERNAL.NESTEDORBITMULT);
%     - nested contraction of an absolute attribute: |G| g_a; the nested
%       centres route: g_a; the relative non-periodic contraction (a
%       trapezoid over the alignment line of the product of the s leaf
%       kernels): |G| s (sigma sqrt(pi))^(s-2) / 2; the relative-periodic
%       tau grid (the mean over the period of the same integrand):
%       P |G| s (sigma sqrt(pi))^(s-2) / 2.
%
%   CHOSEN is the flat route ('bulger', 'mobius', 'centres') or
%   'contract' for the nested plan, whose per-attribute routes are in
%   NESTEDROUTES ('-' for a flat attribute). Pinned by
%   tests/test_inner_product_scale.m against an enumeration reference on
%   every shape. Twin of the Python cosine._ip_canonical_scale.

    if nargin < 3; nestedRoutes = {}; end
    A = double(dens.nAttrs);
    nested = cell(1, A);
    if isfield(dens, 'nested') && ~isempty(dens.nested)
        for a = 1:min(A, numel(dens.nested)); nested{a} = dens.nested{a}; end
    end
    if isfield(dens, 'isSym') && ~isempty(dens.isSym)
        isSym = logical(dens.isSym(:).');
    else
        isSym = true(1, A);
    end
    scale = 1.0;
    for a = 1:A
        sigma = double(dens.sigma(a));
        isRel = logical(dens.isRel(a));
        sp = sigma * sqrt(pi);
        spec = nested{a};
        if isempty(spec)
            r_a = double(dens.r(a));
            if isRel && r_a < 2
                continue;   % 0-D point mass: no kernel, no prefactor
            end
            if isRel
                g = sp^(r_a - 1) * sqrt(r_a);
            else
                g = sp^r_a;
            end
            ordered = ~isSym(a) && r_a > 1;
            switch chosen
                case 'mobius'
                    f = 1.0;
                case 'centres'
                    f = g;
                case 'bulger'
                    if ordered, f = g; else, f = factorial(r_a) * g; end
                otherwise   % nested plan: flat attributes take the Möbius
                    if ordered, f = g; else, f = 1.0; end   % matrix, ordered ones the centres
            end
            scale = scale * f;
            continue;
        end
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
        G = double(internal.nestedOrbitMult(rLevels, logical(spec.sym(:).')));
        switch chosen
            case 'bulger'
                f = G * g;
            case 'centres'
                f = g;
            otherwise
                route = 'contract';
                if a <= numel(nestedRoutes) && ~isempty(nestedRoutes{a})
                    route = char(nestedRoutes{a});
                end
                switch route
                    case 'centres'
                        f = g;
                    case 'contract_relnonper'
                        f = G * sTot * sp^(sTot - 2) / 2;
                    case 'taugrid'
                        f = double(dens.period(a)) * G * sTot * sp^(sTot - 2) / 2;
                    otherwise
                        f = G * g;
                end
        end
        scale = scale * f;
    end
end

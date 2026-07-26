function X = whitenQuery(dens, X)
%WHITENQUERY  Whiten a query array for a density built with kernel covariance.
%
%   single-attribute density: X is dim x nQ (a vector accepted when
%     dim == 1).
%   MA density: X is D x nQ with per-attribute row blocks in attribute
%   order; only blocks whose attribute carries a covariance are
%   transformed. Returns X unchanged for densities without a kernel
%   covariance.

    if ~internal.densityHasKernelCov(dens)
        return;
    end
    ch = dens.kernelChol;
    X = double(X);
    if ~iscell(ch)
        % Single-attribute density.
        dim = dens.dim;
        if isvector(X) && dim == 1
            X = internal.whitenValues(ch, X);
            return;
        end
        if size(X, 1) ~= dim
            error('mpt:aniso:queryShape', ...
                ['X must have %d rows (each column is a %d-D query ' ...
                 'point).'], dim, dim);
        end
        X = internal.whitenValues(ch, X);
        return;
    end
    % Multi-attribute density: slice rows by per-attribute dims.
    dims = dens.dimPerAttr(:).';
    if size(X, 1) ~= sum(dims)
        error('mpt:aniso:queryShape', ...
            ['X must have %d rows (each column is a joint query ' ...
             'point across the attributes).'], sum(dims));
    end
    row = 0;
    for a = 1:numel(dims)
        if a <= numel(ch) && ~isempty(ch{a})
            X(row + 1:row + dims(a), :) = ...
                internal.whitenValues(ch{a}, X(row + 1:row + dims(a), :));
        end
        row = row + dims(a);
    end
end

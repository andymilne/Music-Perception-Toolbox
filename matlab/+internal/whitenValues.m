function Y = whitenValues(R, V)
%WHITENVALUES  Whiten value columns: solve R * Y = V columnwise.
%
%   Y = internal.whitenValues(R, V) applies the whitening change of
%   coordinates y = R \ v to each column of V (dim x n; a vector of
%   length dim is treated as one column and returned with its input
%   orientation). R is the lower Cholesky factor of the kernel
%   covariance (Sigma = R * R'). NaN or infinite values are rejected:
%   whitening mixes coordinates, so a NaN pad would contaminate the
%   whole column.

    wasVec = isvector(V);
    if wasVec
        origIsRow = isrow(V);
        V = V(:);
    else
        origIsRow = false;
    end
    V = double(V);
    if size(V, 1) ~= size(R, 1)
        error('mpt:aniso:rowMismatch', ...
            ['Values have %d rows but the kernel covariance is ' ...
             '%dx%d.'], size(V, 1), size(R, 1), size(R, 1));
    end
    if ~all(isfinite(V(:)))
        error('mpt:aniso:nanValues', ...
            ['NaN or infinite values are not supported on an ' ...
             'attribute with a matrix-valued kernel covariance ' ...
             '(whitening mixes coordinates within each tuple).']);
    end
    Y = R \ V;
    if wasVec && origIsRow
        Y = Y.';
    end
end

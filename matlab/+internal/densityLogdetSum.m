function ld = densityLogdetSum(dens)
%DENSITYLOGDETSUM  Sum of log det(Sigma_a) over the density's covariances.
%
%   Computed from the stored lower Cholesky factors:
%   log det(Sigma) = 2 * sum(log(diag(R))).

    ld = 0.0;
    if ~isstruct(dens) || ~isfield(dens, 'kernelChol')
        return;
    end
    ch = dens.kernelChol;
    if iscell(ch)
        for a = 1:numel(ch)
            if ~isempty(ch{a})
                ld = ld + 2 * sum(log(diag(ch{a})));
            end
        end
    elseif ~isempty(ch)
        ld = 2 * sum(log(diag(ch)));
    end
end

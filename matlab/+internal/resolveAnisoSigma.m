function [pOut, sigmaOut, covOut, cholOut] = resolveAnisoSigma( ...
    pIn, sigmaIn, rIn, isRelIn, isPerIn, isSymIn, nestedIn)
%RESOLVEANISOSIGMA  Resolve matrix-valued sigma for build-time whitening.
%
%   Two forms, selected by the type of PIN:
%
%   SA (PIN numeric vector, SIGMAIN a matrix): validates the aniso
%   constraints and the covariance, whitens the value vector, and
%   returns (pWhitened, 1.0, Sigma, R).
%
%   MA (PIN a cell of K_a x N matrices, SIGMAIN a cell mixing scalars
%   and matrices): resolves each matrix entry per attribute; returns
%   the whitened pAttr cell, the all-scalar numeric sigma vector, and
%   length-A cell arrays covOut/cholOut with [] for isotropic
%   attributes.
%
%   Scalar geometry inputs are broadcast per attribute where vectors
%   are expected. isSymIn may be [] (the buildExpTens default,
%   symmetric), which the constraint check will reject for a matrix
%   entry with the appropriate message.

    if ~iscell(pIn)
        % ----- SA form -----
        p = double(pIn(:));
        K = numel(p);
        isSym = true;
        if nargin >= 6 && ~isempty(isSymIn), isSym = logical(isSymIn); end
        internal.checkAnisoConstraints(rIn, K, isRelIn, isPerIn, ...
            isSym, false, 'sigma');
        [Sigma, R] = internal.validateKernelCov(sigmaIn, double(rIn), ...
            'sigma');
        pOut = internal.whitenValues(R, p);
        sigmaOut = 1.0;
        covOut = Sigma;
        cholOut = R;
        return;
    end

    % ----- MA form -----
    A = numel(pIn);
    if ~iscell(sigmaIn)
        error('mpt:aniso:sigmaVecForm', ...
            ['To carry a matrix-valued kernel covariance on a ' ...
             'multi-attribute call, pass sigma as a 1 x %d cell ' ...
             'array mixing scalars and r_a x r_a matrices.'], A);
    end
    if numel(sigmaIn) ~= A
        error('mpt:aniso:sigmaVecLength', ...
            'sigma cell array must have length %d (nAttrs).', A);
    end
    pOut = pIn;
    covOut = cell(1, A);
    cholOut = cell(1, A);
    sigmaNum = zeros(1, A);
    expand = @(v, a, dflt) localEntry(v, a, dflt);
    for a = 1:A
        if ~internal.isKernelCov(sigmaIn{a})
            e = sigmaIn{a};
            if ~isnumeric(e) || ~isscalar(e)
                error('mpt:aniso:sigmaCellEntry', ...
                    ['sigma{%d} must be a scalar standard deviation ' ...
                     'or an r_a x r_a covariance matrix.'], a);
            end
            sigmaNum(a) = double(e);
            continue;
        end
        nestedA = false;
        if nargin >= 7 && ~isempty(nestedIn) && iscell(nestedIn) ...
                && a <= numel(nestedIn)
            nestedA = ~isempty(nestedIn{a});
        end
        P = double(pIn{a});
        % Rows are slots, columns are events (buildExpTens's coercion):
        % no reshape — a 1 x N row vector is one slot across N events
        % and correctly fails the r == K check below.
        Ka = size(P, 1);
        rA = expand(rIn, a, []);
        relA = expand(isRelIn, a, false);
        perA = expand(isPerIn, a, false);
        symA = expand(isSymIn, a, true);
        internal.checkAnisoConstraints(rA, Ka, relA, perA, symA, ...
            nestedA, sprintf('sigma{%d}', a));
        [Sigma, R] = internal.validateKernelCov(sigmaIn{a}, ...
            double(rA), sprintf('sigma{%d}', a));
        pOut{a} = internal.whitenValues(R, P);
        sigmaNum(a) = 1.0;
        covOut{a} = Sigma;
        cholOut{a} = R;
    end
    sigmaOut = sigmaNum;
end


function e = localEntry(v, a, dflt)
    if isempty(v)
        e = dflt;
    elseif iscell(v)
        e = v{min(a, numel(v))};
    elseif isscalar(v)
        e = v;
    else
        e = v(min(a, numel(v)));
    end
end

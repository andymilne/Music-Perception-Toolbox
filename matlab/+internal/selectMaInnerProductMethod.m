function chosen = selectMaInnerProductMethod(rVec, kVec, A, Nx, Ny, ...
        anyPer, anyRelNonper, anyRelPer, sigmaOverPMax, userMethod, verbose)
%SELECTMAINNERPRODUCTMETHOD  Pick the MA inner-product method (cost model).
%   Mirror of Python dispatch._select_ma_inner_product_method. Routing
%   rules, in order: (1) userMethod override; (2) r_max <= 1 -> Bulger;
%   (3) r_max > _ORBIT_R_MAX_SHIPPED -> Bulger; (4) K-vs-r precision guard
%   (orbitSafeForPrecision) -> Bulger; (5) rel+per with sigma/P beyond the
%   integration-exact regime -> warn, Bulger; (6) otherwise predict both
%   wall times (ms) and pick the smaller (ties favour Bulger). Constants
%   are the Python-calibrated values, so the route is identical to Python.
    if nargin < 11; verbose = true; end
    if ~strcmp(userMethod, 'auto')
        chosen = userMethod;
        return;
    end
    if A > 0
        r_max = max(rVec);
    else
        r_max = 1;
    end
    if r_max <= 1
        chosen = 'bulger'; return;
    end
    if r_max > 8                        % _ORBIT_R_MAX_SHIPPED
        chosen = 'bulger'; return;
    end
    if A > 0 && ~internal.orbitSafeForPrecision(rVec, kVec)
        chosen = 'bulger'; return;     % K-vs-r precision guard
    end
    if anyRelPer && sigmaOverPMax > 0.03   % _ORBIT_SIGMA_OVER_P_THRESHOLD
        if verbose
            warning('cosSimExpTens:mobiusSigmaOverPFallback', ...
                ['Maximum sigma/period = %.3f across periodic-relative ' ...
                 'attributes exceeds the Möbius-method threshold (0.03); ' ...
                 'falling back to Bulger''s method (the pairwise-wrap ' ...
                 'form). Pass ''method'', ''bulger'' explicitly to ' ...
                 'silence this warning.'], sigmaOverPMax);
        end
        chosen = 'bulger'; return;
    end
    pwSize = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny);
    pwCost = pwSize * pwPerEntryMs(anyPer);
    orbitCost = predictOrbitCostMs(r_max, A, Nx, Ny, kVec, ...
                                   anyRelNonper, anyRelPer);
    if pwCost <= orbitCost
        chosen = 'bulger';
    else
        chosen = 'mobius';
    end
end


function ms = pwPerEntryMs(anyPer)
    if anyPer
        ms = 7.0e-4;   % _PW_PER_ENTRY_MS_PER
    else
        ms = 1.0e-4;   % _PW_PER_ENTRY_MS_NONPER
    end
end


function sz = predictPairwiseKernelSize(rVec, kVec, A, Nx, Ny)
    % n_J . n_K = Nx . Ny . prod_a r_a! . C(K_a, r_a)^2
    if A == 0
        sz = Nx * Ny; return;
    end
    sz = Nx * Ny;
    for a = 1:A
        ra = rVec(a); Ka = kVec(a);
        if Ka < ra
            sz = Inf; return;
        end
        c = combCount(Ka, ra);
        sz = sz * factorial(ra) * c * c;
    end
end


function ms = predictOrbitCostMs(r_max, A, Nx, Ny, kVec, anyRelNonper, anyRelPer)
    if A > 0
        Kmax = max(kVec);
    else
        Kmax = 1;
    end
    % Index r = 2..8 (entry 1 unused). Python-calibrated constants.
    ABS       = [NaN, 3.0, 11.2, 45.0, 150.0, 500.0, 1500.0, 4500.0];
    RELPER    = [NaN, 0.06, 0.40, 1.0, 5.0, 20.0, 60.0, 200.0];
    RELNONPER = [NaN, 0.25, 1.05, 3.30, 12.0, 50.0, 150.0, 500.0];
    if anyRelNonper
        c = RELNONPER(r_max);
        ms = A * (5.0 + Nx * Ny * Kmax * Kmax * c);   % _ORBIT_RELNONPER_BASE_MS
    elseif anyRelPer
        c = RELPER(r_max);
        ms = A * (5.0 + Nx * Ny * Kmax * Kmax * c);   % _ORBIT_RELPER_BASE_MS
    else
        ms = A * ABS(r_max);
    end
end


function c = combCount(nn, kk)
    if kk < 0 || kk > nn
        c = 0; return;
    end
    c = 1;
    for ii = 0:kk - 1
        c = c * (nn - ii) / (ii + 1);
    end
    c = round(c);
end

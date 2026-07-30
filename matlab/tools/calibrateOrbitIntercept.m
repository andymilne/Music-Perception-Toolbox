function [intercept, detail] = calibrateOrbitIntercept(varargin)
%CALIBRATEORBITINTERCEPT  Measure orbitCostIntercept for this machine.
%
%   INTERCEPT = tools.calibrateOrbitIntercept() times the two combine
%   routes on a small set of shapes and returns the value to set as
%   mptDefaults('orbitCostIntercept', INTERCEPT).
%
%   [INTERCEPT, DETAIL] = ... also returns a struct array of the cells
%   used, with the measured and predicted log ratios per cell.
%
%   Why only one number. internal.orbitCostModel predicts
%
%       log(t_orbit / t_enum) = c0 + c1 log|Omega_r| + c2 log K
%                                  + c3 log B + c4 log(Tx Ty) + c5 log r
%
%   where c1..c5 are scaling exponents -- properties of how each route's
%   work grows -- and c0 carries the machine. Recalibrating therefore
%   means measuring one quantity, not repopulating a table. c0 is the mean
%   residual between measured and predicted log ratio, so this is a
%   one-parameter fit with the exponents held fixed.
%
%   How accurate the model is. Over 484 timed cells on the machine it was
%   fitted on, the predicted time ratio sits within a factor of about 2.5
%   of the measured one typically, with occasional cells much further out.
%   That is adequate for the purpose: the two routes differ by 10x to 200x
%   away from the crossover, so a factor of 2.5 changes nothing there, and
%   near the crossover the routes are close enough that choosing the wrong
%   one costs little. It does mean the per-cell residuals below scatter
%   widely -- that is the model, not the machine.
%
%   Caveat: the exponents are more portable than a table but not perfectly
%   portable. The batch exponents are sub-linear, which is cache and
%   vectorisation behaviour rather than arithmetic, so on markedly
%   different hardware they may shift too. A median far from the shipped
%   value, or residuals that trend with r rather than scattering, both
%   point to needing the full sweep rather than one intercept.
%
%   Options (name-value):
%     rVals      tuple sizes to sample          (default 3:6)
%     BVals      batch extents to sample        (default [32 256])
%     budgetSec  skip a cell whose predicted enumeration time exceeds
%                this                           (default 0.5)
%     verbose    print per-cell detail          (default true)
%
%   Timing is delegated to tools/calibrateOrbitCrossover, so there is only
%   one place that times the two routes. A cell is skipped when
%   enumeration is predicted to exceed the budget: at high r and K it runs
%   for minutes, and the Mobius route wins there by default, so timing it
%   buys nothing. Only the absolute mode is used, since the two modes time
%   identically and counting both would double-weight every shape.
%
%   Example:
%       c = tools.calibrateOrbitIntercept();
%       mptDefaults('orbitCostIntercept', c);
%
%   See also internal.orbitCostModel, tools/calibrateOrbitCrossover.
    p = inputParser;
    p.addParameter('rVals', 3:6);
    p.addParameter('BVals', [32 256]);
    p.addParameter('budgetSec', 0.5);
    p.addParameter('verbose', true);
    p.parse(varargin{:});
    opt = p.Results;

    fprintf('\ncalibrateOrbitIntercept  %s\n', ...
            datestr(now, 'yyyy-mm-dd HH:MM:SS')); %#ok<TNOW1,DATST>
    fprintf('  shipped orbitCostIntercept = %.4f\n\n', ...
            mptDefaults('orbitCostIntercept'));

    % Reuse calibrateOrbitCrossover for the timing rather than restating
    % the two routes here. It already times the shipped bodies, checks the
    % routes agree before timing, and applies the repeated-measure policy;
    % a second copy of that would be one more thing to drift.
    margins = 0:4;
    res = calibrateOrbitCrossover('rVals', opt.rVals, 'BVals', opt.BVals, ...
                                  'margins', margins);

    shipped = mptDefaults('orbitCostIntercept');
    detail = struct('r', {}, 'K', {}, 'B', {}, 'tOrbit', {}, ...
                    'tEnum', {}, 'measLog', {}, 'predLog', {}, 'resid', {});
    resids = [];
    if opt.verbose
        fprintf('%3s %3s %6s %11s %11s %10s %10s %9s\n', ...
                'r', 'K', 'B', 'orbit ms', 'enum ms', 'measured', ...
                'predicted', 'residual');
        fprintf('%s\n', repmat('-', 1, 72));
    end
    for i = 1:numel(res)
        c = res(i);
        if ~strcmp(c.mode, 'absolute')
            continue;   % the two modes time identically; avoid double weight
        end
        if predictedEnumSeconds(c.r, c.K, c.B) > opt.budgetSec
            continue;   % enumeration impractical; the Mobius route wins here
        end
        measLog = log(c.timeRatio);
        % c.workRatio is the model's predicted ratio, which already carries
        % the shipped intercept; remove it so the residual is the intercept
        % this machine implies.
        predLog = log(c.workRatio) - shipped;
        resid = measLog - predLog;
        resids(end+1) = resid; %#ok<AGROW>
        detail(end+1) = struct('r', c.r, 'K', c.K, 'B', c.B, ...
            'tOrbit', c.tOrbit, 'tEnum', c.tEnum, 'measLog', measLog, ...
            'predLog', predLog, 'resid', resid); %#ok<AGROW>
        if opt.verbose
            fprintf('%3d %3d %6d %11.3f %11.3f %10.3f %10.3f %9.3f\n', ...
                    c.r, c.K, c.B, c.tOrbit * 1e3, c.tEnum * 1e3, ...
                    measLog, predLog, resid);
        end
    end

    if isempty(resids)
        error('mpt:calibrateOrbitIntercept:noCells', ...
              ['No cell was both measurable and near the crossover. ' ...
               'Raise budgetSec or widen rVals.']);
    end

    intercept = median(resids);
    fprintf('\n  cells used            %d\n', numel(resids));
    fprintf('  intercept (median)    %.4f\n', intercept);
    fprintf('  spread (min to max)   %.4f to %.4f\n', ...
            min(resids), max(resids));
    % The model's own scatter, measured over 484 cells on the machine it
    % was fitted on: standard deviation 0.93 in log, span 6.5. A wide
    % spread here is therefore expected and says nothing; what would
    % indicate a machine-specific problem is the median sitting well away
    % from the shipped value.
    offset = intercept - mptDefaults('orbitCostIntercept');
    if abs(offset) > 1.0
        fprintf(['\n  The median is %.2f in log from the shipped value ' ...
                 '(a factor of %.1f).\n  Adopting it is worthwhile. If ' ...
                 'the residuals also trend with r rather\n  than ' ...
                 'scattering, the exponents need refitting too, not just ' ...
                 'the\n  intercept; run tools/calibrateOrbitCrossover ' ...
                 'for the full sweep.\n'], offset, exp(abs(offset)));
    else
        fprintf(['\n  Within one in log of the shipped value, which is ' ...
                 'inside the\n  model''s own scatter (sd 0.93). No ' ...
                 'change needed.\n']);
    end
    fprintf('\n  To adopt:  mptDefaults(''orbitCostIntercept'', %.4f)\n\n', ...
            intercept);
end




function t = predictedEnumSeconds(r, K, B)
    % Rough enumerated-route time from the model's own enum scaling, used
    % only to skip cells that would take minutes.
    logTxTy = gammaln(K + 1) - gammaln(K - r + 1) ...
            + gammaln(K + 1) - gammaln(r + 1) - gammaln(K - r + 1);
    % Calibrated so that Tx*Ty*B = 1e6 reads as roughly 10 ms.
    t = exp(logTxTy + log(B)) * 1e-8;
end





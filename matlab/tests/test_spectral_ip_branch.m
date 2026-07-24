%% test_spectral_ip_branch.m
%  Tests for the spectral (Fourier) branch of the relative-mode inner
%  product, mobius.spectralRelInnerMatrix. Twin of the Python
%  spectral-branch coverage.
%
%  The branch replaces the translation grid with a mode sum: each
%  event's spectrum is built once and the matrix over event pairs is
%  their Gram matrix, so the per-pair cost carries no K and no grid
%  nodes. Two independent Hermitian halvings apply -- one on the mode
%  grid, one on the per-event phase table.
%
%  Two things are checked. First, value parity with Python against
%  spectral_ip_parity.json, including which shapes the branch DECLINES:
%  both languages must decline on the same shapes, because a
%  disagreement would route one language to the grid and the other to
%  the branch. Second, route parity within MATLAB: the branch and the
%  translation grid must agree, since both compute the full-image
%  measure.
%
%  No local functions: this file is executed as a script from
%  test_mpt.m.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

siJson = fullfile(fileparts(mfilename('fullpath')), 'spectral_ip_parity.json');
if exist(siJson, 'file') ~= 2
    results(end+1, :) = {'spectral IP: parity fixture present', false}; %#ok<*SAGROW>
else
    siFix = jsondecode(fileread(siJson));
    siP   = siFix.period;
    % jsondecode yields a cell array rather than a struct array when the
    % cases are non-uniform (here the position/weight vectors differ in
    % length across K), so accept either shape.
    if iscell(siFix.cases)
        siGet = @(k) siFix.cases{k};
        siN   = numel(siFix.cases);
    else
        siGet = @(k) siFix.cases(k);
        siN   = numel(siFix.cases);
    end
    siNDeclineOK = 0; siNDecline = 0;
    siNValueOK   = 0; siNValue   = 0;
    siWorst = 0;
    for ii = 1:siN
        c = siGet(ii);
        Px = c.px(:); Py = c.py(:);
        Wx = c.wx(:); Wy = c.wy(:);
        got = mobius.spectralRelInnerMatrix(Px, Wx, Py, Wy, c.sigma, ...
                                            c.r, logical(c.isPer), siP);
        if c.declined
            siNDecline = siNDecline + 1;
            if isempty(got)
                siNDeclineOK = siNDeclineOK + 1;
            end
        else
            siNValue = siNValue + 1;
            if ~isempty(got)
                rel = abs(got(1,1) - c.value) / max(abs(c.value), realmin);
                siWorst = max(siWorst, rel);
                if rel <= 1e-12
                    siNValueOK = siNValueOK + 1;
                end
            end
        end
    end
    results(end+1, :) = { ...
        sprintf('spectral IP: decline parity (%d/%d shapes)', ...
                siNDeclineOK, siNDecline), ...
        siNDeclineOK == siNDecline};
    results(end+1, :) = { ...
        sprintf('spectral IP: value parity (%d/%d, worst %.1e)', ...
                siNValueOK, siNValue, siWorst), ...
        siNValueOK == siNValue};
end

% --- route parity within MATLAB: branch vs translation grid ----------
% Both compute the full-image measure, so they must agree. This is the
% check that would catch a gate or transcription divergence.
rng(21);
siRouteWorst = 0;
siRouteN = 0;
for r = 2:4
    for K = [6, 10]
        if K < r, continue; end
        for sop = [0.02, 0.05, 0.20]
            sigma = sop * 1200;
            p = sort(rand(K, 1) * 1200);
            q = sort(rand(K, 1) * 1200);
            w = ones(K, 1);
            internal.spectralIpEnabled(true);
            aOn  = cosSimExpTens(p, w, q, w, sigma, r, true, true, 1200, ...
                                 'method', 'mobius', 'verbose', false);
            internal.spectralIpEnabled(false);
            aOff = cosSimExpTens(p, w, q, w, sigma, r, true, true, 1200, ...
                                 'method', 'mobius', 'verbose', false);
            internal.spectralIpEnabled(true);
            siRouteWorst = max(siRouteWorst, abs(aOn - aOff));
            siRouteN = siRouteN + 1;
        end
    end
end
results(end+1, :) = { ...
    sprintf('spectral IP: branch matches grid route (%d cells, worst %.1e)', ...
            siRouteN, siRouteWorst), ...
    siRouteWorst <= 1e-11};

% --- the enable switch round-trips -----------------------------------
siPrevState = internal.spectralIpEnabled();
internal.spectralIpEnabled(false);
siOffOK = ~internal.spectralIpEnabled();
internal.spectralIpEnabled(true);
siOnOK = internal.spectralIpEnabled();
internal.spectralIpEnabled(siPrevState);
results(end+1, :) = {'spectral IP: enable switch round-trips', ...
                     siOffOK && siOnOK};


if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_spectral_ip_branch: %d passed, %d failed\n', nPass, nFail);
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
end

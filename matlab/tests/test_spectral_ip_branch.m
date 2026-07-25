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
%  Two things are checked. First, VALUE parity with Python against
%  spectral_ip_parity.json: wherever the branch runs, the spectral value
%  must match the Python reference. Decline parity is deliberately NOT
%  checked -- the cost gate is per-language (Python 3160, MATLAB 282),
%  so the two legitimately route different shapes; only the value has to
%  agree, and forceBranch is used to compare it on cells the gates split
%  on. Second, route parity within MATLAB: the branch and the
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
    % Value parity only. The cost gate is per-language (Python 3160,
    % MATLAB 282), so the two decline on different shapes -- that is
    % expected and is NOT checked. What must agree is the spectral VALUE
    % wherever the branch runs, so forceBranch skips MATLAB's cost gate
    % and the value is compared against the Python reference on every
    % cell the Python side produced one for. Cells the Python fixture
    % marks declined are skipped: those are Python's own MAXPOINTS or
    % cost-gate declines and carry no reference value.
    siNValueOK = 0; siNValue = 0; siWorst = 0;
    for ii = 1:siN
        c = siGet(ii);
        if c.declined
            continue;   % no Python reference value for this cell
        end
        Px = c.px(:); Py = c.py(:);
        Wx = c.wx(:); Wy = c.wy(:);
        got = mobius.spectralRelInnerMatrix(Px, Wx, Py, Wy, c.sigma, ...
                                            c.r, logical(c.isPer), siP, true);
        siNValue = siNValue + 1;
        if ~isempty(got)
            rel = abs(got(1,1) - c.value) / max(abs(c.value), realmin);
            siWorst = max(siWorst, rel);
            if rel <= 1e-12
                siNValueOK = siNValueOK + 1;
            end
        end
    end
    results(end+1, :) = { ...
        sprintf('spectral IP: value parity, forced (%d/%d, worst %.1e)', ...
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

% --- guard removal preserved values on the single-multiset path ------
% The per-node cancellation-ratio fallback was removed from
% cosSimExpTens (it fired only where the Moebius result was already
% bit-identical to Bulger's). Where it used to fire -- sigma/period
% <= 0.0125, well below the 0.03 measure-divergence threshold -- the
% two methods share a measure, so removing the fallback must not move
% any value. Checked directly against Bulger.
rng(31);
siGuardWorst = 0;
siGuardN = 0;
for isRelC = [true, false]
    for r = 2:4
        for K = [4, 6, 8, 12]
            if K < r, continue; end
            for sigma = [0.1, 0.5, 2.0, 6.0, 15.0]
                p_ = sort(rand(K, 1) * 1200);
                q_ = sort(rand(K, 1) * 1200);
                w_ = ones(K, 1);
                cm = cosSimExpTens(p_, w_, q_, w_, sigma, r, isRelC, ...
                                   true, 1200, 'method', 'mobius', ...
                                   'verbose', false);
                cb = cosSimExpTens(p_, w_, q_, w_, sigma, r, isRelC, ...
                                   true, 1200, 'method', 'bulger', ...
                                   'verbose', false);
                siGuardWorst = max(siGuardWorst, abs(cm - cb));
                siGuardN = siGuardN + 1;
            end
        end
    end
end
results(end+1, :) = { ...
    sprintf(['spectral IP: mobius matches bulger below the divergence ' ...
             'threshold (%d cells, worst %.1e)'], siGuardN, siGuardWorst), ...
    siGuardWorst <= 1e-9};

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

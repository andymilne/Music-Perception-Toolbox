%% test_nested_abs_per_full_image.m — the nested contraction computes the wrap the density declares
%
%  Mirror of the Python tests/test_nested_abs_per_full_image.py.
%
%  An absolute-periodic attribute defaults to wrap = 'full-image': the
%  per-coordinate kernel is the wrapped Gaussian
%  theta(d) = sum_n exp(-(d + n P)^2 / (4 sigma^2)). The nested contraction
%  (both the batched (N_x, N_y) matrix and the per-event-pair reference
%  nestedIp) used to reduce d to the nearest image and exponentiate, which
%  is the 'single-image' kernel -- a different measure once the accuracy
%  floor asks for more than one image. It agreed with the enumeration
%  (method 'bulger') only below sigma/P ~ 0.05 and departed above (4e-2 in
%  the cosine at sigma/P = 0.2).
%
%  These pin 'contract' to 'bulger' across sigma/P under both wraps, and
%  check the two wraps genuinely differ where the floor asks for images, so
%  the agreement is not both routes taking the same shortcut.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf, so the routes are
%  compared at the accuracy floor rather than at the 6-sigma truncation.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_nafi
    cleanupDefaults_nafi = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

nafi_P = 12.0;
nafi_chord = 3; nafi_nCh = 2; nafi_N = 2;
nafi_tags = repelem(0:nafi_nCh - 1, nafi_chord);
nafi_spec = struct('tags', nafi_tags, 'r', [2 nafi_nCh], ...
                   'sym', [true false], 'rel', [0 0]);

nafi_dens = @(seed, sigma, wrap) nafiBuild(seed, sigma, wrap, nafi_P, ...
                                           nafi_chord * nafi_nCh, nafi_N, ...
                                           nafi_spec);

% --- contract == bulger at every sigma/P, under either wrap ---
for nafi_wrapC = {'full-image', 'single-image'}
    nafi_wrap = nafi_wrapC{1};
    for nafi_sop = [0.01 0.05 0.1 0.2 0.3]
        nafi_sigma = nafi_sop * nafi_P;
        cC = cosSimExpTens(nafi_dens(1, nafi_sigma, nafi_wrap), ...
                           nafi_dens(2, nafi_sigma, nafi_wrap), ...
                           'method', 'contract', 'verbose', false);
        cB = cosSimExpTens(nafi_dens(1, nafi_sigma, nafi_wrap), ...
                           nafi_dens(2, nafi_sigma, nafi_wrap), ...
                           'method', 'bulger', 'verbose', false);
        results{end+1, 1} = sprintf( ...
            'nested abs-per %s: contract==bulger at sigma/P=%.2f', ...
            nafi_wrap, nafi_sop); %#ok<*SAGROW>
        results{end, 2} = abs(cC - cB) <= 1e-11 * max(abs(cB), 1) + 1e-13;
    end
end

% --- the two wraps are different measures at sigma/P = 0.2 ---
nafi_sigma = 0.2 * nafi_P;
cFull = cosSimExpTens(nafi_dens(1, nafi_sigma, 'full-image'), ...
                      nafi_dens(2, nafi_sigma, 'full-image'), ...
                      'method', 'contract', 'verbose', false);
cSingle = cosSimExpTens(nafi_dens(1, nafi_sigma, 'single-image'), ...
                        nafi_dens(2, nafi_sigma, 'single-image'), ...
                        'method', 'contract', 'verbose', false);
results{end+1, 1} = 'nested abs-per: full-image and single-image differ at sigma/P=0.2';
results{end, 2}   = abs(cFull - cSingle) > 1e-3;

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_nested_abs_per_full_image: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_nafi
    if nFail > 0
        error('test_nested_abs_per_full_image:failed', '%d test(s) failed.', nFail);
    end
end


function d = nafiBuild(seed, sigma, wrap, P, nVal, N, spec)
    rng(seed, 'twister');
    p = sort(P * rand(nVal, N), 1);
    d = buildExpTens({p}, {[]}, 'specs', {spec}, 'sigma', sigma, ...
                     'isPer', true, 'period', P, 'wrap', {wrap}, ...
                     'verbose', false);
end

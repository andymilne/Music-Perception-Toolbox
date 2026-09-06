%% test_routing_parity_fixes.m — regressions from the routing-map parity audit
%
%  Mirror of the Python tests/test_routing_parity_fixes.py, plus one
%  MATLAB-side item.
%
%  Python dropped an attribute's declared wrap on two routes (the flat
%  attribute of a nested-MA density, and the flat symmetric attribute of
%  the Renyi-2 loop); MATLAB had always passed it, so the first three
%  blocks pin the shared behaviour in both languages: under either wrap
%  the contraction agrees with the enumeration, and the two wraps differ
%  where the truncation budget admits images beyond the nearest one.
%
%  MATLAB's MA evaluation honoured method = 'mobius' on an ordered
%  ([sym]=0) attribute, silently evaluating the symmetrised density,
%  where its own single-multiset path and the Python twin raise; the last
%  block pins the error.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_rpf
    cleanupDefaults_rpf = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

rpf_P   = 12.0;
rpf_SIG = 0.2 * rpf_P;      % sigma/P = 0.2: L >= 1 at the 6-sigma default

% --- nested-MA flat attribute honours its wrap ---
for rpf_wrapC = {'full-image', 'single-image'}
    rpf_wrap = rpf_wrapC{1};
    x = rpfNestedPlusFlat(1, rpf_wrap, rpf_P, rpf_SIG);
    y = rpfNestedPlusFlat(2, rpf_wrap, rpf_P, rpf_SIG);
    c = cosSimExpTens(x, y, 'method', 'contract', 'verbose', false);
    b = cosSimExpTens(x, y, 'method', 'bulger', 'verbose', false);
    results{end+1, 1} = sprintf( ...
        'parity fixes: nested-MA flat attribute %s: contract == bulger', ...
        rpf_wrap); %#ok<*SAGROW>
    results{end, 2} = abs(c - b) <= 1e-9 * max(abs(b), 1) + 1e-12;
end

cf = cosSimExpTens(rpfNestedPlusFlat(1, 'full-image', rpf_P, rpf_SIG), ...
                   rpfNestedPlusFlat(2, 'full-image', rpf_P, rpf_SIG), ...
                   'method', 'contract', 'verbose', false);
cs = cosSimExpTens(rpfNestedPlusFlat(1, 'single-image', rpf_P, rpf_SIG), ...
                   rpfNestedPlusFlat(2, 'single-image', rpf_P, rpf_SIG), ...
                   'method', 'contract', 'verbose', false);
results{end+1, 1} = 'parity fixes: nested-MA flat attribute: the two wraps differ at sigma/P=0.2';
results{end, 2}   = abs(cf - cs) > 1e-4;

% --- Renyi-2 flat attribute honours its wrap ---
hf = entropyExpTens(rpfFlat('full-image', rpf_P, rpf_SIG), ...
                    'method', 'renyi2', 'verbose', false);
hs = entropyExpTens(rpfFlat('single-image', rpf_P, rpf_SIG), ...
                    'method', 'renyi2', 'verbose', false);
results{end+1, 1} = 'parity fixes: renyi2 flat attribute: the two wraps differ at sigma/P=0.2';
results{end, 2}   = isfinite(hf) && isfinite(hs) && abs(hf - hs) > 1e-4;

% --- MA evalExpTens refuses method='mobius' on an ordered attribute ---
rpf_ok = false;
try
    dOrd = buildExpTens({[0 4 7; 2 5 9].', [1 2 3; 4 5 6].'}, {[], []}, ...
                        [0.5 0.5], [2 2], [false false], [false false], ...
                        [0 0], [false true], 'verbose', false);
    evalExpTens(dOrd, zeros(dOrd.dim, 3), 'method', 'mobius', ...
                'verbose', false);
catch rpf_err
    rpf_ok = strcmp(rpf_err.identifier, 'mpt:evalExpTens:orderedMobius');
end
results{end+1, 1} = 'parity fixes: MA evalExpTens method=mobius on an ordered attribute raises orderedMobius';
results{end, 2}   = rpf_ok;

% --- evalExpTens refuses method='mobius' on a nested density ---
% A forced 'mobius' used to evaluate the flattened multiset silently; it
% is refused, as on an ordered attribute, and 'auto' takes the centres.
rpf_ok = false;
try
    rng(1, 'twister');
    rpf_tags = repelem(0:1, 3);
    rpf_p = sort(rpf_P * rand(6, 2), 1);
    rpf_spec = struct('tags', rpf_tags, 'r', [1 2], 'sym', [true true], ...
                      'rel', [0 1]);
    dNest = buildExpTens({rpf_p}, {[]}, 'specs', {rpf_spec}, 'sigma', 0.5, ...
                         'isPer', true, 'period', rpf_P, 'verbose', false);
    evalExpTens(dNest, zeros(dNest.dim, 3), 'method', 'mobius', ...
                'verbose', false);
catch rpf_err
    rpf_ok = strcmp(rpf_err.identifier, 'mpt:evalExpTens:nestedMobius');
end
results{end+1, 1} = 'parity fixes: evalExpTens method=mobius on a nested density raises nestedMobius';
results{end, 2}   = rpf_ok;
if rpf_ok
    vA = evalExpTens(dNest, zeros(dNest.dim, 3), 'verbose', false);
    vC = evalExpTens(dNest, zeros(dNest.dim, 3), 'method', 'centres', ...
                     'verbose', false);
    results{end+1, 1} = 'parity fixes: evalExpTens auto == centres on a nested density';
    results{end, 2}   = max(abs(vA(:) - vC(:))) <= 1e-12 * max(1, max(abs(vC(:))));
end

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
    fprintf('\n=== test_routing_parity_fixes: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_rpf
    if nFail > 0
        error('test_routing_parity_fixes:failed', '%d test(s) failed.', nFail);
    end
end


function d = rpfNestedPlusFlat(seed, wrap, P, SIG)
    % A nested (absolute-periodic) attribute tensored with a flat abs-per
    % r = 2 attribute carrying the given wrap.
    rng(seed, 'twister');
    tags = repelem(0:1, 3);
    p0 = sort(P * rand(6, 2), 1);
    p1 = sort(P * rand(4, 2), 1);
    sp0 = struct('tags', tags, 'r', [1 2], 'sym', [true true], 'rel', [0 0]);
    sp1 = struct('r', 2, 'sym', true, 'rel', false);
    d = buildExpTens({p0, p1}, {[], []}, 'specs', {sp0, sp1}, ...
                     'sigma', [0.05 * P, SIG], 'isPer', [true true], ...
                     'period', [P P], 'wrap', {'full-image', wrap}, ...
                     'verbose', false);
end


function d = rpfFlat(wrap, P, SIG)
    rng(3, 'twister');
    p = sort(P * rand(5, 2), 1);
    d = buildExpTens({p}, {[]}, SIG, 2, false, true, P, ...
                     'wrap', {wrap}, 'verbose', false);
end

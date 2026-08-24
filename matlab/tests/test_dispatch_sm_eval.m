%% test_dispatch_sm_eval.m — v2.2 single multiset method dispatch in evalExpTens
%
%  Tests for the new method keyword introduced in v2.2 (Commit 6b).
%  Covers:
%    - Orbit and centres paths agree to numerical tolerance on healthy
%      regimes (r in {3, 4, 5}, abs and rel modes, periodic and
%      non-periodic).
%    - Auto dispatch picks the expected path given r, K, mode (rel auto
%      always falls back to centres; Möbius only on opt-in).
%    - Skinny dens flows through the dispatcher without forcing eager
%      build on the Möbius branch.
%    - Bad method values raise informative errors.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

%% ---- Method kwarg validation ----

% Bad method string raises informative error.
ok_badMethod = false;
try
    evalExpTens([0 4 7 11 14 18 21 25], [], 30, 3, false, false, 0, ...
        [0; 4; 7], 'method', 'bogus', 'verbose', false);
catch ME
    ok_badMethod = strcmp(ME.identifier, 'evalExpTens:badMethod');
end
results{end+1,1} = 'dispatch.single multiset eval: bad method string raises evalExpTens:badMethod';
results{end,2}   = ok_badMethod;

%% ---- Orbit and centres agree (auto + explicit), abs nonperiodic ----

rng(31, 'twister');
p = sort(2000 * rand(8, 1));
w = 0.5 + rand(8, 1);
sigma = 30; r = 3;
X = [linspace(0, 2000, 5); linspace(500, 2500, 5); linspace(1000, 3000, 5)];

v_auto    = evalExpTens(p, w, sigma, r, false, false, 0, X, 'verbose', false);
v_centres = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'centres', 'verbose', false);
v_orbit   = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'mobius', 'verbose', false);

results{end+1,1} = 'dispatch.single multiset eval: r=3 abs nonper auto matches centres (within 1e-10)';
results{end,2}   = max(abs(v_auto - v_centres)) < 1e-10;
results{end+1,1} = 'dispatch.single multiset eval: r=3 abs nonper Möbius matches centres (within 1e-8)';
results{end,2}   = max(abs(v_orbit - v_centres)) < 1e-8;

% --- 'direct' is retired: it was a synonym for 'centres' here, and named
% Bulger's method on the inner product, so one word meant two things. ---
ok_directRetired = false;
try
    evalExpTens(p, w, sigma, r, false, false, 0, X, ...
        'method', 'direct', 'verbose', false);
catch ME
    ok_directRetired = strcmp(ME.identifier, 'evalExpTens:badMethod');
end
results{end+1,1} = 'dispatch.single multiset eval: retired ''direct'' raises evalExpTens:badMethod';
results{end,2}   = ok_directRetired;

%% ---- Auto agrees with centres for small r=2 ----
% The cost model routes small-K r=2 (abs) to the Möbius path: its factored
% cost B_r*r*K undercuts the joint tuple count r!*C(K,r). Auto and centres
% therefore agree to the Möbius alternating-sum accuracy, not bit-exactly.
% Twin of the Python dispatcher test (test_eval_dispatcher.py), which
% asserts allclose(auto, centres) at ATOL=1e-11, RTOL=1e-8.

rng(33, 'twister');
p2 = sort(2000 * rand(6, 1));
w2 = ones(6, 1);
X2 = [linspace(100, 1900, 4); linspace(500, 1500, 4)];
v_auto2 = evalExpTens(p2, w2, 30, 2, false, false, 0, X2, 'verbose', false);
v_cen2  = evalExpTens(p2, w2, 30, 2, false, false, 0, X2, ...
    'method', 'centres', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: r=2 small-K auto agrees with centres (rtol 1e-8)';
results{end,2}   = all(abs(v_auto2 - v_cen2) <= 1e-11 + 1e-8 * abs(v_cen2));

%% ---- K-vs-r margin guard ----

% n=4, r=3. A collection barely larger than the tuple size is no longer
% a reason to refuse the Mobius method, so auto now routes on cost. The
% two routes agree to within the accuracy truncationSigmas asks for,
% judged as absolute error on the scale the density lives on.
p_small = [0; 100; 400; 700];
w_small = ones(4, 1);
X_small = [50; 200; 350];
v_auto_sm    = evalExpTens(p_small, w_small, 50, 3, false, false, 0, X_small, ...
    'verbose', false);
v_centres_sm = evalExpTens(p_small, w_small, 50, 3, false, false, 0, X_small, ...
    'method', 'centres', 'verbose', false);
% Budget: the truncation floor is stated per kernel entry, while the
% density sums many entries, so allow a small multiple of floor times
% the value scale.
tolSm = 10 * internal.truncationFloor([]) * max(abs(v_centres_sm(:)));
results{end+1,1} = 'dispatch.single multiset eval: K-r margin <2 auto agrees with centres on the value scale';
results{end,2}   = all(abs(v_auto_sm(:) - v_centres_sm(:)) <= tolSm);

%% ---- Relative mode below the sigma/P threshold: auto and centres agree ----
% Below the threshold the single-image (centres) and all-image (Möbius)
% measures coincide to ~1e-6, so auto dispatches on speed and either route
% is a valid pick; the invariant is agreement with the single-image
% measure to the coincidence bound, not path identity.

rng(37, 'twister');
p_rel = sort(1200 * rand(8, 1));
w_rel = ones(8, 1);
X_rel = [200; 400];   % (r-1)=2 rows for r=3
v_auto_rel    = evalExpTens(p_rel, w_rel, 30, 3, true, true, 1200, X_rel, ...
    'verbose', false);
v_centres_rel = evalExpTens(p_rel, w_rel, 30, 3, true, true, 1200, X_rel, ...
    'method', 'centres', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: rel mode auto agrees with centres (below threshold)';
results{end,2}   = max(abs(v_auto_rel(:) - v_centres_rel(:))) ...
                   < 1e-5 * max(abs(v_centres_rel(:)));

% --- Explicit Möbius for rel mode runs the relative-mode evaluator and agrees with centres ---
v_orbit_rel = evalExpTens(p_rel, w_rel, 30, 3, true, true, 1200, X_rel, ...
    'method', 'mobius', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: rel mode explicit Möbius matches centres (1e-6)';
results{end,2}   = max(abs(v_orbit_rel - v_centres_rel)) < 1e-6;

%% ---- r=4 abs periodic: Möbius and centres agree ----

rng(41, 'twister');
p_per = sort(1200 * rand(8, 1));
w_per = ones(8, 1);
X_per = [linspace(0, 1100, 5); linspace(200, 1100, 5); ...
         linspace(400, 1100, 5); linspace(600, 1100, 5)];
v_orbit_per = evalExpTens(p_per, w_per, 30, 4, false, true, 1200, X_per, ...
    'method', 'mobius', 'verbose', false);
v_cen_per   = evalExpTens(p_per, w_per, 30, 4, false, true, 1200, X_per, ...
    'method', 'centres', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: r=4 abs per Möbius matches centres (1e-8)';
results{end,2}   = max(abs(v_orbit_per - v_cen_per)) < 1e-8;

%% ---- Skinny dens flows transparently ----

dens_skinny = buildExpTens(p, w, sigma, r, false, false, 0, 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: skinny dens has no Centres before dispatch';
results{end,2}   = ~isfield(dens_skinny, 'Centres');
v_skinny_orbit = evalExpTens(dens_skinny, X, ...
    'method', 'mobius', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: Möbius on skinny dens matches raw-args centres';
results{end,2}   = max(abs(v_skinny_orbit - v_centres)) < 1e-8;

%% ---- Normalization consistency (Möbius vs centres) ----

% 'gaussian' normalization: same multiplicative constant applied to both.
v_orbit_g = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'mobius', 'gaussian', 'verbose', false);
v_cen_g   = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'centres', 'gaussian', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: gaussian-normalized Möbius matches centres (1e-8)';
results{end,2}   = max(abs(v_orbit_g - v_cen_g)) < 1e-8;

% 'pdf' normalization: divides by sum(wJ); same factor for both paths.
v_orbit_pdf = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'mobius', 'pdf', 'verbose', false);
v_cen_pdf   = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'centres', 'pdf', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset eval: pdf-normalized orbit matches centres (1e-8)';
results{end,2}   = max(abs(v_orbit_pdf - v_cen_pdf)) < 1e-8;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_dispatch_sm_eval: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end

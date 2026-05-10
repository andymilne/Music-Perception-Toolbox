%% test_dispatch_sa_eval.m — v2.2 SA orbit dispatch in evalExpTens
%
%  Tests for the new method keyword introduced in v2.2 (Commit 6b).
%  Covers:
%    - Orbit and centres paths agree to numerical tolerance on healthy
%      regimes (r in {3, 4, 5}, abs and rel modes, periodic and
%      non-periodic).
%    - Auto dispatch picks the expected path given r, K, mode (rel auto
%      always falls back to centres; orbit only on opt-in).
%    - Skinny dens flows through the dispatcher without forcing eager
%      build on the orbit branch.
%    - Bad method values raise informative errors.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone = true;
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
results{end+1,1} = 'dispatch.SA eval: bad method string raises evalExpTens:badMethod';
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
    'method', 'orbit', 'verbose', false);

results{end+1,1} = 'dispatch.SA eval: r=3 abs nonper auto matches centres (within 1e-10)';
results{end,2}   = max(abs(v_auto - v_centres)) < 1e-10;
results{end+1,1} = 'dispatch.SA eval: r=3 abs nonper orbit matches centres (within 1e-8)';
results{end,2}   = max(abs(v_orbit - v_centres)) < 1e-8;

% --- 'direct' is a synonym for 'centres' ---
v_direct = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'direct', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: ''direct'' alias produces same result as ''centres''';
results{end,2}   = isequal(v_direct, v_centres);

%% ---- Auto routes to centres for small r=2 ----

rng(33, 'twister');
p2 = sort(2000 * rand(6, 1));
w2 = ones(6, 1);
X2 = [linspace(100, 1900, 4); linspace(500, 1500, 4)];
v_auto2 = evalExpTens(p2, w2, 30, 2, false, false, 0, X2, 'verbose', false);
v_cen2  = evalExpTens(p2, w2, 30, 2, false, false, 0, X2, ...
    'method', 'centres', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: r=2 small-K auto agrees with centres (exact)';
results{end,2}   = isequal(v_auto2, v_cen2);

%% ---- K-vs-r margin guard ----

% n=4, r=3 -> margin = 1 < 2, so auto picks centres.
p_small = [0; 100; 400; 700];
w_small = ones(4, 1);
X_small = [50; 200; 350];
v_auto_sm    = evalExpTens(p_small, w_small, 50, 3, false, false, 0, X_small, ...
    'verbose', false);
v_centres_sm = evalExpTens(p_small, w_small, 50, 3, false, false, 0, X_small, ...
    'method', 'centres', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: K-r margin <2 auto agrees with centres (exact)';
results{end,2}   = isequal(v_auto_sm, v_centres_sm);

%% ---- Relative mode: auto always picks centres ----

rng(37, 'twister');
p_rel = sort(1200 * rand(8, 1));
w_rel = ones(8, 1);
X_rel = [200; 400];   % (r-1)=2 rows for r=3
v_auto_rel    = evalExpTens(p_rel, w_rel, 30, 3, true, true, 1200, X_rel, ...
    'verbose', false);
v_centres_rel = evalExpTens(p_rel, w_rel, 30, 3, true, true, 1200, X_rel, ...
    'method', 'centres', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: rel mode auto = centres (exact)';
results{end,2}   = isequal(v_auto_rel, v_centres_rel);

% --- Explicit orbit for rel mode runs orbit-rel and agrees with centres ---
v_orbit_rel = evalExpTens(p_rel, w_rel, 30, 3, true, true, 1200, X_rel, ...
    'method', 'orbit', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: rel mode explicit orbit matches centres (1e-6)';
results{end,2}   = max(abs(v_orbit_rel - v_centres_rel)) < 1e-6;

%% ---- r=4 abs periodic: orbit and centres agree ----

rng(41, 'twister');
p_per = sort(1200 * rand(8, 1));
w_per = ones(8, 1);
X_per = [linspace(0, 1100, 5); linspace(200, 1100, 5); ...
         linspace(400, 1100, 5); linspace(600, 1100, 5)];
v_orbit_per = evalExpTens(p_per, w_per, 30, 4, false, true, 1200, X_per, ...
    'method', 'orbit', 'verbose', false);
v_cen_per   = evalExpTens(p_per, w_per, 30, 4, false, true, 1200, X_per, ...
    'method', 'centres', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: r=4 abs per orbit matches centres (1e-8)';
results{end,2}   = max(abs(v_orbit_per - v_cen_per)) < 1e-8;

%% ---- Skinny dens flows transparently ----

dens_skinny = buildExpTens(p, w, sigma, r, false, false, 0, 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: skinny dens has no Centres before dispatch';
results{end,2}   = ~isfield(dens_skinny, 'Centres');
v_skinny_orbit = evalExpTens(dens_skinny, X, ...
    'method', 'orbit', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: orbit on skinny dens matches raw-args centres';
results{end,2}   = max(abs(v_skinny_orbit - v_centres)) < 1e-8;

%% ---- Normalization consistency (orbit vs centres) ----

% 'gaussian' normalization: same multiplicative constant applied to both.
v_orbit_g = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'orbit', 'gaussian', 'verbose', false);
v_cen_g   = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'centres', 'gaussian', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: gaussian-normalized orbit matches centres (1e-8)';
results{end,2}   = max(abs(v_orbit_g - v_cen_g)) < 1e-8;

% 'pdf' normalization: divides by sum(wJ); same factor for both paths.
v_orbit_pdf = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'orbit', 'pdf', 'verbose', false);
v_cen_pdf   = evalExpTens(p, w, sigma, r, false, false, 0, X, ...
    'method', 'centres', 'pdf', 'verbose', false);
results{end+1,1} = 'dispatch.SA eval: pdf-normalized orbit matches centres (1e-8)';
results{end,2}   = max(abs(v_orbit_pdf - v_cen_pdf)) < 1e-8;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_dispatch_sa_eval: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end

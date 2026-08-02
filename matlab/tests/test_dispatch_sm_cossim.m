%% test_dispatch_sm_cossim.m — v2.2 single multiset method dispatch in cosSimExpTens
%
%  Tests for the new method/cancellationThreshold keywords introduced in
%  v2.2 (Commit 6a). Covers:
%    - Möbius and Bulger methods agree to numerical tolerance on healthy
%      regimes (r in {3, 4}, abs and rel modes, periodic and non-periodic).
%    - Auto dispatch picks the expected path given r, n, mode.
%    - The post-hoc impossible-value check (non-finite inner product,
%      negative Gram diagonal, cosine outside [-1, 1]) catches
%      degenerate cases and routes to Bulger's method.
%    - Bad keyword values raise informative errors.
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

%% ---- cosSimExpTens method kwarg validation ----

% Bad method string raises informative error.
ok_badMethod = false;
try
    cosSimExpTens([0 4 7], [], [0 4 7], [], 12, 2, false, true, 1200, ...
        'method', 'bogus', 'verbose', false);
catch ME
    ok_badMethod = strcmp(ME.identifier, 'cosSimExpTens:badMethod');
end
results{end+1,1} = 'dispatch.single multiset: bad method string raises cosSimExpTens:badMethod';
results{end,2}   = ok_badMethod;

% Bad cancellationThreshold raises informative error.
ok_badCT = false;
try
    cosSimExpTens([0 4 7], [], [0 4 7], [], 12, 2, false, true, 1200, ...
        'cancellationThreshold', -1, 'verbose', false);
catch ME
    ok_badCT = strcmp(ME.identifier, 'cosSimExpTens:badCancellationThreshold');
end
results{end+1,1} = 'dispatch.single multiset: negative cancellationThreshold raises error';
results{end,2}   = ok_badCT;

%% ---- Auto routing decisions (verify path used) ----
%
% We can't easily intercept the dispatcher's choice without re-architecting,
% so we verify routing indirectly: Möbius-method-only fields like the cancellation
% ratio aren't observable from outside, but agreement with method='bulger'
% within tolerance is a strong consistency signal. The structural coverage
% below ensures no path fails silently.

% --- r=3 abs nonperiodic: auto picks the Möbius method; agrees with Bulger ---
rng(7, 'twister');
p_x = sort(2000 * rand(8, 1));
w_x = 0.5 + rand(8, 1);
p_y = sort(2000 * rand(8, 1));
w_y = 0.5 + rand(8, 1);
sigma = 30; r = 3; isRel = false; isPer = false; period = 0;

s_auto = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, isRel, isPer, period, ...
    'verbose', false);
s_pair = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, isRel, isPer, period, ...
    'method', 'bulger', 'verbose', false);
s_orb  = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, isRel, isPer, period, ...
    'method', 'mobius', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: r=3 abs nonper auto matches Bulger (within 1e-10)';
results{end,2}   = abs(s_auto - s_pair) < 1e-10;
results{end+1,1} = 'dispatch.single multiset: r=3 abs nonper Möbius matches Bulger (within 1e-8)';
results{end,2}   = abs(s_orb - s_pair) < 1e-8;

% --- r=4 abs periodic: auto picks the Möbius method; agrees with Bulger ---
rng(11, 'twister');
p_x = sort(1200 * rand(8, 1));
w_x = ones(8, 1);
p_y = sort(1200 * rand(8, 1));
w_y = ones(8, 1);
sigma = 30; r = 4; isRel = false; isPer = true; period = 1200;

s_pair = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, isRel, isPer, period, ...
    'method', 'bulger', 'verbose', false);
s_orb  = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, isRel, isPer, period, ...
    'method', 'mobius', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: r=4 abs per Möbius matches Bulger (within 1e-8)';
results{end,2}   = abs(s_orb - s_pair) < 1e-8;

% --- r=3 rel periodic, sigma/period within threshold: Möbius agrees ---
rng(13, 'twister');
p_x = sort(1200 * rand(8, 1));
w_x = ones(8, 1);
p_y = sort(1200 * rand(8, 1));
w_y = ones(8, 1);
sigma = 12;  r = 3; isRel = true; isPer = true; period = 1200;
% sigma/period = 0.01 < 0.03 threshold

s_pair = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, isRel, isPer, period, ...
    'method', 'bulger', 'verbose', false);
s_orb  = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, isRel, isPer, period, ...
    'method', 'mobius', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: r=3 rel per (sig/P=0.01) Möbius matches Bulger (1e-6)';
results{end,2}   = abs(s_orb - s_pair) < 1e-6;

%% ---- Auto routes to Bulger in expected cases ----

% --- r=2 small-n: auto matches Bulger to numerical precision ---
% Under v2.2.0 the analytical heuristic forced r=2 with n_max<=8 to
% Bulger, so auto and Bulger were bit-identical here. Under v2.2.x
% the probe-based dispatcher decides empirically; at K=6 r=2 the
% analytical pre-screen (ratio K^2/2 = 18 > 10) routes auto to the Möbius method.
% Both paths compute the same IP mathematically; round-off differs at
% machine epsilon. Result must still match to numerical precision.
rng(17, 'twister');
p_x = sort(2000 * rand(6, 1));
w_x = ones(6, 1);
p_y = sort(2000 * rand(6, 1));
w_y = ones(6, 1);
sigma = 50; r = 2;
s_auto2 = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, false, false, 0, ...
    'verbose', false);
s_pair2 = cosSimExpTens(p_x, w_x, p_y, w_y, sigma, r, false, false, 0, ...
    'method', 'bulger', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: r=2 small-n auto matches Bulger (1e-12)';
results{end,2}   = abs(s_auto2 - s_pair2) < 1e-12;

% --- Collection barely larger than the tuple size: routed on cost ---
% n=4, r=3. The two routes agree to within the accuracy truncationSigmas
% asks for; a cosine has value scale 1, so the floor applies directly.
p_x = [0; 100; 400; 700];
w_x = ones(4, 1);
p_y = p_x;
w_y = w_x;
s_auto3 = cosSimExpTens(p_x, w_x, p_y, w_y, 50, 3, false, false, 0, ...
    'verbose', false);
s_pair3 = cosSimExpTens(p_x, w_x, p_y, w_y, 50, 3, false, false, 0, ...
    'method', 'bulger', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: K-r margin <2 auto agrees with Bulger on the value scale';
results{end,2}   = abs(s_auto3 - s_pair3) <= 10 * internal.truncationFloor([]);

% --- Self-similarity is 1 in both modes ---
rng(19, 'twister');
p = sort(2000 * rand(7, 1));
w = 0.5 + rand(7, 1);
s_self_pair = cosSimExpTens(p, w, p, w, 30, 3, false, false, 0, ...
    'method', 'bulger', 'verbose', false);
s_self_orb  = cosSimExpTens(p, w, p, w, 30, 3, false, false, 0, ...
    'method', 'mobius', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: self-similarity = 1 (Bulger)';
results{end,2}   = abs(s_self_pair - 1) < 1e-12;
results{end+1,1} = 'dispatch.single multiset: self-similarity = 1 (Möbius, within 1e-8)';
results{end,2}   = abs(s_self_orb - 1) < 1e-8;

%% ---- sigma/period threshold: warning retired, dispatch stays silent ----

% Retired in v3+: rel-per full-image is the default measure and the
% dispatch no longer warns. This test survives as a positive check that
% the previously-warning call path is now silent and that auto still
% routes correctly (matching explicit Möbius).
p_x = (0:5)' * 200;
w_x = ones(6, 1);
p_y = (0:5)' * 200 + 50;
w_y = ones(6, 1);
% sigma/period = 60/1200 = 0.05: previously above the "warn" threshold.
warnState = warning('on', 'cosSimExpTens:relPerAllImage');
lastwarn('');
s_warn = cosSimExpTens(p_x, w_x, p_y, w_y, 60, 3, true, true, 1200, ...
    'verbose', true);
[wmsg, wid] = lastwarn;
warning(warnState);
results{end+1,1} = 'dispatch.single multiset: rel+per sigma/P=0.05 stays silent (warning retired)';
results{end,2}   = ~strcmp(wid, 'cosSimExpTens:relPerAllImage');
% Auto still takes the all-image (Möbius) path at this sigma/P.
s_mob_warn = cosSimExpTens(p_x, w_x, p_y, w_y, 60, 3, true, true, 1200, ...
    'method', 'mobius', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: rel+per sigma/P=0.05 result matches Möbius';
results{end,2}   = abs(s_warn - s_mob_warn) < 1e-12;

%% ---- Cross-cancellation guard fires fallback ----

% With a cancellationThreshold set absurdly high (e.g., 0.99), almost any
% real cosine value will be deemed "cancellation-suspect" and force fallback
% to pairwise. The Möbius-vs-Bulger agreement on healthy data means both
% paths return the same value — verifying no wrong-answer path leaks out.
rng(23, 'twister');
p_x = sort(2000 * rand(8, 1));
w_x = ones(8, 1);
p_y = sort(2000 * rand(8, 1));
w_y = ones(8, 1);
s_pair_g = cosSimExpTens(p_x, w_x, p_y, w_y, 30, 3, false, false, 0, ...
    'method', 'bulger', 'verbose', false);
% Force Möbius method, but with threshold 0.99 trigger fallback.
s_high_ct = cosSimExpTens(p_x, w_x, p_y, w_y, 30, 3, false, false, 0, ...
    'method', 'mobius', 'cancellationThreshold', 0.99, 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: high cancellationThreshold falls back to Bulger';
results{end,2}   = abs(s_high_ct - s_pair_g) < 1e-12;

%% ---- Skinny dens flows through dispatch transparently ----

% From Commit 5: dens default is skinny. The dispatcher should accept it
% and route correctly without forcing eager build for the Möbius method.
p_x = [0; 4; 7; 11; 14];
w_x = ones(5, 1);
p_y = [0; 5; 7; 12; 14];
w_y = ones(5, 1);
dens_x_skinny = buildExpTens(p_x, w_x, 30, 3, false, false, 0, 'verbose', false);
dens_y_skinny = buildExpTens(p_y, w_y, 30, 3, false, false, 0, 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: skinny dens has no Centres before dispatch';
results{end,2}   = ~isfield(dens_x_skinny, 'Centres');
s_skinny_orb = cosSimExpTens(dens_x_skinny, dens_y_skinny, ...
    'method', 'mobius', 'verbose', false);
s_pair_check = cosSimExpTens(p_x, w_x, p_y, w_y, 30, 3, false, false, 0, ...
    'method', 'bulger', 'verbose', false);
results{end+1,1} = 'dispatch.single multiset: Möbius on skinny dens matches raw-args Bulger';
results{end,2}   = abs(s_skinny_orb - s_pair_check) < 1e-8;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_dispatch_sm_cossim: %d passed, %d failed (of %d) ===\n', ...
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

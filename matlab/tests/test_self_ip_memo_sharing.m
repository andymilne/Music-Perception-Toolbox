%% test_self_ip_memo_sharing.m — what the routes' self IPs may and may not share
%
%  Mirror of the Python tests/test_self_ip_memo_sharing.py.
%
%  The multi-attribute inner product is computed by several routes ---
%  Bulger's joint-tuple enumeration, the unrestricted tuple-centres
%  enumeration, the per-attribute Moebius matrices (themselves either a
%  translation grid or the tuple-centres closed form), and, for a nested
%  attribute, the per-level contraction --- and each memoises <X,X> and
%  <Y,Y> in the density struct's selfIP field, so a repeated call or a
%  sweep pays for the cross term alone.
%
%  Two questions follow, with different answers.
%
%  Are the routes' bare triples on the same scale? Yes, and by an exactly
%  known constant per attribute; the first block pins the one identity
%  MATLAB can read without a raw-triple entry point: the Moebius
%  per-attribute matrix over the tuple-centres closed form is the
%  single-multiset Gaussian prefactor (sigma sqrt(pi))^r in an absolute
%  mode and (sigma sqrt(pi))^(r-1) sqrt(r) in relative non-periodic ---
%  the two routes' constants against Bulger's scale, divided. (The full
%  table, including the r_a! that both carry over Bulger's perm-versus-comb
%  enumeration, is pinned in the Python twin, which can call the flat
%  routes' bare triples directly.)
%
%  May a memo written by one route therefore be read by another, rescaled?
%  No: each route applies the truncation budget to its own arrays, so after
%  the exact rescaling the routes hold different numbers rather than the
%  same number in different units. The keys stay route-specific because of
%  it --- a call's answer must not depend on which route warmed the memo,
%  which the call-order block below pins.
%
%  What IS shared is the pricing: INTERNAL.SELFIPMEMOISED reports whether
%  ANY route has paid for a density's self inner product, and both sides of
%  every route comparison are priced against it. Otherwise the first route
%  to run is priced at one matrix and its rival at three, and that first
%  choice locks in however cheap the rival becomes once warm.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.
%  mptTestIsolateDefaults sets truncationSigmas = Inf (accuracy floor).

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_sms
    cleanupDefaults_sms = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

sms_P = 12.0;

% Deterministic inputs, formula-based so the two languages see the same
% numbers without a seeded generator.
sms_i = (1:6).';
sms_absX = 3.0 + mod(7 * sms_i.^2, 17) / 2.0;      % (6, 1), well spread
sms_absY = 2.5 + mod(11 * sms_i.^2, 19) / 2.0;
sms_perX = mod(1.7 * sms_i.^2 + 0.6, sms_P);
sms_perY = mod(2.3 * sms_i.^2 + 3.1, sms_P);

smsFlat = @(v, sigma, r, isRel, isPer) buildExpTens( ...
    {v}, [], sigma, r, isRel, isPer, sms_P, 'verbose', false);


%% --- 1. The scale identity between the two Moebius-side builders ------

sms_ok = true;
sms_worst = 0.0;
for sms_case = {{false, false}, {false, true}, {true, false}}
    sms_isRel = sms_case{1}{1};
    sms_isPer = sms_case{1}{2};
    for sms_r = [2 3]
        for sms_sigma = [0.3 1.0]
            if sms_isPer
                sms_v = sms_perX;
            else
                sms_v = sms_absX;
            end
            sms_d = smsFlat(sms_v, sms_sigma, sms_r, sms_isRel, sms_isPer);
            sms_grid = sum(sum(mobius.maPerAttrInnerMatrix( ...
                sms_d.pAttr{1}, sms_d.w{1}, sms_d.pAttr{1}, sms_d.w{1}, ...
                sms_sigma, sms_r, sms_isRel, sms_isPer, sms_P)));
            sms_cx = mobius.closedFormAttrCentres(sms_d, 1);
            sms_closed = sum(sum(mobius.closedFormAttrMatrixFrom( ...
                sms_cx, sms_cx, 'full-image')));
            if sms_isRel
                sms_expect = (sms_sigma * sqrt(pi))^(sms_r - 1) * sqrt(sms_r);
            else
                sms_expect = (sms_sigma * sqrt(pi))^sms_r;
            end
            sms_rel = abs(sms_grid / sms_closed / sms_expect - 1);
            sms_worst = max(sms_worst, sms_rel);
            sms_ok = sms_ok && (sms_rel < 1e-10);
        end
    end
end
results{end+1, 1} = sprintf( ...
    ['selfIpShare: the grid and closed-form builders differ by the ' ...
     'Gaussian prefactor alone (worst %.2e)'], sms_worst); %#ok<*SAGROW>
results{end, 2} = sms_ok;

% Relative-periodic is the case no constant relates: the grid computes the
% all-image transposition average (C), the closed form the minimum-image
% reading (A). Below the sigma/period threshold they sit inside the
% truncation floor of each other; the gap is still not zero, and it grows
% with sigma/P, so no memo may cross.
sms_gaps = zeros(1, 2);
sms_sigmas = [0.7 1.5];
for sms_k = 1:2
    sms_sigma = sms_sigmas(sms_k);
    sms_d = smsFlat(sms_perX, sms_sigma, 2, true, true);
    sms_grid = sum(sum(mobius.maPerAttrInnerMatrix( ...
        sms_d.pAttr{1}, sms_d.w{1}, sms_d.pAttr{1}, sms_d.w{1}, ...
        sms_sigma, 2, true, true, sms_P)));
    sms_cx = mobius.closedFormAttrCentres(sms_d, 1);
    sms_closed = sum(sum(mobius.closedFormAttrMatrixFrom( ...
        sms_cx, sms_cx, 'full-image')));
    sms_expect = (sms_sigma * sqrt(pi))^1 * sqrt(2);
    sms_gaps(sms_k) = abs(sms_grid / sms_closed / sms_expect - 1);
end
results{end+1, 1} = ['selfIpShare: the rel-per tau grid is a different ' ...
                     'measure, not a rescaling'];
results{end, 2} = sms_gaps(1) > 1e-6 && sms_gaps(2) > 20 * sms_gaps(1);


%% --- 2. The pricing flag reads any route; the values stay route-keyed --

sms_empty = struct('keys', {{}}, 'vals', []);
results{end+1, 1} = 'selfIpShare: an empty memo is not memoised';
results{end, 2} = ~internal.selfIpMemoised(sms_empty);

sms_cache = struct('keys', {{internal.selfIpKey('mobius', 6.0, '1')}}, ...
                   'vals', 1.0);
results{end+1, 1} = ['selfIpShare: a Moebius memo counts for the ' ...
                     'shared pricing flag'];
results{end, 2} = internal.selfIpMemoised(sms_cache);

% The sweep's own memo is produced by a different evaluator and consumed by
% neither inner-product route, so it spares neither of them any work.
sms_sweepCache = struct('keys', {{'sweep|6|none'}}, 'vals', 1.0);
results{end+1, 1} = ['selfIpShare: a sweep memo does not count as a ' ...
                     'route memo'];
results{end, 2} = ~internal.selfIpMemoised(sms_sweepCache);

% Two routes on one pair leave two entries, not one shared entry: the
% values are the same quantity in different units and are stored that way.
sms_dx = smsFlat(sms_absX, 0.5, 2, false, false);
sms_dy = smsFlat(sms_absY, 0.5, 2, false, false);
[~, sms_dx, sms_dy] = cosSimExpTens(sms_dx, sms_dy, 'method', 'bulger', ...
                                    'verbose', false);
[~, sms_dx, sms_dy] = cosSimExpTens(sms_dx, sms_dy, 'method', 'mobius', ...
                                    'verbose', false);
sms_keys = sms_dx.selfIP.keys;
results{end+1, 1} = ['selfIpShare: two routes leave two memo entries, ' ...
                     'one per route'];
% Both prefixes present is the whole point: a canonical-scale scheme would
% have left one shared entry instead.
results{end, 2} = any(strncmp(sms_keys, 'bulger|', 7)) ...
    && any(strncmp(sms_keys, 'mobius|', 7));


%% --- 3. The lock-in, and the value invariance it must not cost --------

% Two relative-periodic cells, chosen so that the cold cost model prefers a
% different route on each --- a small one, whose tuple-pair count stays
% under the Moebius side's per-attribute setup, and a larger one, where it
% does not. WHICH route each cell draws is a matter of the fitted cost
% constants, and those differ between the two languages, so the cold route
% is read off the call rather than named; what is asserted is only the
% invariance. Two cells are used so that a memo on the cheaper route and a
% memo on the dearer one are both exercised in whichever language runs them.
smsCells = {[5 4], [9 4]};
smsAllRoutes = {'bulger', 'centres', 'mobius'};

smsLockin = @(KN) deal( ...
    buildExpTens({mod(1.9 * (1:KN(1)).' .^ 2 + 0.4 * (1:KN(2)), sms_P)}, ...
                 [], 0.25, 2, true, true, sms_P, 'verbose', false), ...
    buildExpTens({mod(2.6 * (1:KN(1)).' .^ 2 + 0.7 * (1:KN(2)), sms_P)}, ...
                 [], 0.25, 2, true, true, sms_P, 'verbose', false));

% The route that wrote a memo is the key's prefix, up to the first '|'.
smsRoutesOf = @(d) unique(cellfun(@(k) strtok(k, '|'), d.selfIP.keys, ...
                                  'UniformOutput', false));

% Warm each cell on every route the cold call did NOT take, then call auto
% again. Before the pricing flags were shared, the memo made the route that
% wrote it free and left its rival priced at three matrices, so whichever
% route ran first locked itself in and the route the cold model prefers
% could never be reached again on that pair. What shared pricing
% guarantees is that a warm auto call prices both routes against the same
% flags, so its choice cannot depend on WHICH route warmed the pair. The
% warm choice may legitimately differ from the cold one --- a call with
% both self products memoised is a different workload, and the Moebius
% side's setup floor does not scale with the matrix count, so a near-tie
% can fall the other way once warm --- so the reference is auto warmed by
% auto itself, and every rival-warmed call must take that route and return
% the cold value.
sms_definite = true;
sms_noLock = true;
sms_valOk = true;
for sms_ci = 1:numel(smsCells)
    sms_KN = smsCells{sms_ci};
    [sms_lx, sms_ly] = smsLockin(sms_KN);
    [sms_cold, sms_lx, sms_ly] = cosSimExpTens(sms_lx, sms_ly, 'verbose', false);
    sms_coldRoutes = smsRoutesOf(sms_lx);
    sms_definite = sms_definite && (numel(sms_coldRoutes) == 1);
    if numel(sms_coldRoutes) ~= 1
        continue
    end
    sms_coldRoute = sms_coldRoutes{1};
    % Reference warm route: the cold route's memo is present either way,
    % so it is the route that appears in the second call, or the cold
    % route again if none did.
    [~, sms_lx, ~] = cosSimExpTens(sms_lx, sms_ly, 'verbose', false);
    sms_warmRoutes = smsRoutesOf(sms_lx);
    sms_new = sms_warmRoutes(~strcmp(sms_warmRoutes, sms_coldRoute));
    if isempty(sms_new)
        sms_warmRoute = sms_coldRoute;
    else
        sms_warmRoute = sms_new{1};
    end
    for sms_o = smsAllRoutes
        if strcmp(sms_o{1}, sms_coldRoute)
            continue
        end
        [sms_ax, sms_ay] = smsLockin(sms_KN);
        [~, sms_ax, sms_ay] = cosSimExpTens(sms_ax, sms_ay, ...
            'method', sms_o{1}, 'verbose', false);
        sms_forced = smsRoutesOf(sms_ax);
        sms_noLock = sms_noLock && numel(sms_forced) == 1 ...
            && strcmp(sms_forced{1}, sms_o{1});
        [sms_warm, sms_ax, ~] = cosSimExpTens(sms_ax, sms_ay, ...
                                              'verbose', false);
        % Exactly the rival's memo plus the reference route's (one entry
        % when they coincide): auto took the reference route, no other.
        sms_got = smsRoutesOf(sms_ax);
        sms_want = unique({sms_o{1}, sms_warmRoute});
        sms_noLock = sms_noLock && isequal(sort(sms_got(:)), sort(sms_want(:)));
        sms_valOk = sms_valOk && (abs(sms_warm - sms_cold) <= 1e-13);
    end
end

results{end+1, 1} = ['selfIpShare: cold auto takes one definite route ' ...
                     'on each lock-in cell'];
results{end, 2} = sms_definite;

results{end+1, 1} = ['selfIpShare: a memo on another route no longer ' ...
                     'locks the route in'];
results{end, 2} = sms_noLock;

% Whatever ran first, auto returns the number its route returns: the memos
% are route-keyed, so no value computed under one route's truncation
% treatment is ever consumed by another.
results{end+1, 1} = ['selfIpShare: the warmed call returns the cold ' ...
                     'call''s value (1e-13)'];
results{end, 2} = sms_valOk;

% The remaining blocks work on the first cell alone.
[sms_lx, sms_ly] = smsLockin(smsCells{1});
sms_cold = cosSimExpTens(sms_lx, sms_ly, 'verbose', false);

sms_ok = true;
for sms_pre = {'bulger', 'centres', 'mobius'}
    [sms_ax, sms_ay] = smsLockin(smsCells{1});
    [~, sms_ax, sms_ay] = cosSimExpTens(sms_ax, sms_ay, ...
        'method', sms_pre{1}, 'verbose', false);
    sms_got = cosSimExpTens(sms_ax, sms_ay, 'verbose', false);
    sms_ok = sms_ok && (abs(sms_got - sms_cold) <= 1e-13);
end
results{end+1, 1} = ['selfIpShare: call order does not change the ' ...
                     'value (1e-13)'];
results{end, 2} = sms_ok;

% A forced route reads only its own memo, so it is reproducible whatever
% warmed the cache --- bit for bit.
sms_ok = true;
for sms_m = {'bulger', 'centres', 'mobius'}
    [sms_rx, sms_ry] = smsLockin(smsCells{1});
    sms_ref = cosSimExpTens(sms_rx, sms_ry, 'method', sms_m{1}, ...
                            'verbose', false);
    for sms_other = {'bulger', 'centres', 'mobius'}
        [sms_ax, sms_ay] = smsLockin(smsCells{1});
        [~, sms_ax, sms_ay] = cosSimExpTens(sms_ax, sms_ay, ...
            'method', sms_other{1}, 'verbose', false);
        sms_got = cosSimExpTens(sms_ax, sms_ay, 'method', sms_m{1}, ...
                                'verbose', false);
        sms_ok = sms_ok && (sms_got == sms_ref);
    end
end
results{end+1, 1} = ['selfIpShare: a forced route is reproducible ' ...
                     'whatever warmed the memo (0 diff)'];
results{end, 2} = sms_ok;


%% ---- Standalone summary ---------------------------------------------
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
    fprintf(['\n=== test_self_ip_memo_sharing: %d passed, %d failed ' ...
             '(of %d) ===\n\n'], nPass, nFail, nPass + nFail);
    clear cleanupDefaults_sms
    if nFail > 0
        error('test_self_ip_memo_sharing:failed', '%d test(s) failed.', nFail);
    end
end

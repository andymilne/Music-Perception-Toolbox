function results = test_method_parity()
%TEST_METHOD_PARITY  Method-name vocabulary, and parity with Python.
%
%   There are three algorithms in this area: enumerate the tuple centres
%   ('centres'), enumerate them with one side restricted to combinations
%   and multiply by r! ('bulger'), and sum over the partition lattice
%   ('mobius'). Evaluation has two of them, there being no two-sided
%   pairing for Bulger's identity to exploit.
%
%   These tests pin the vocabulary, because it drifted before: 'direct'
%   named Bulger's method on the inner product while promising an
%   unrestricted enumeration, and named the centres route on evaluation,
%   so one word meant two things and neither matched its docstring. The
%   Python twin is python/tests/test_method_parity.py; the two must
%   accept and reject the same strings.

results = cell(0, 2);

rng(20260823, 'twister');
period = 1200; sigma = 30; r = 3; K = 7;
p = sort(rand(K, 1) * period);  w  = 0.4 + 0.6 * rand(K, 1);
q = sort(rand(K, 1) * period);  wq = 0.4 + 0.6 * rand(K, 1);

ipAccepted   = {'auto', 'bulger', 'centres', 'mobius'};
evalAccepted = {'auto', 'centres', 'mobius'};

%% ---- accepted method strings ----
okIp = true;
for i = 1:numel(ipAccepted)
    try
        A = buildExpTens(p, w, sigma, r, false, false, period, 'verbose', false);
        B = buildExpTens(q, wq, sigma, r, false, false, period, 'verbose', false);
        v = cosSimExpTens(A, B, 'method', ipAccepted{i}, 'verbose', false);
        okIp = okIp && isfinite(v);
    catch
        okIp = false;
    end
end
results{end+1, 1} = 'method parity: cosSimExpTens accepts auto/bulger/centres/mobius';
results{end, 2}   = okIp;

okEval = true;
for i = 1:numel(evalAccepted)
    try
        A = buildExpTens(p, w, sigma, r, false, false, period, 'verbose', false);
        v = evalExpTens(A, [0; 100; 250], 'method', evalAccepted{i}, 'verbose', false);
        okEval = okEval && all(isfinite(v(:)));
    catch
        okEval = false;
    end
end
results{end+1, 1} = 'method parity: evalExpTens accepts auto/centres/mobius';
results{end, 2}   = okEval;

%% ---- 'direct' retired on both entry points ----
okRetiredIp = false;
try
    A = buildExpTens(p, w, sigma, r, false, false, period, 'verbose', false);
    B = buildExpTens(q, wq, sigma, r, false, false, period, 'verbose', false);
    cosSimExpTens(A, B, 'method', 'direct', 'verbose', false);
catch ME
    okRetiredIp = strcmp(ME.identifier, 'cosSimExpTens:badMethod');
end
results{end+1, 1} = 'method parity: cosSimExpTens rejects retired ''direct''';
results{end, 2}   = okRetiredIp;

okRetiredEval = false;
try
    A = buildExpTens(p, w, sigma, r, false, false, period, 'verbose', false);
    evalExpTens(A, 0, 'method', 'direct', 'verbose', false);
catch ME
    okRetiredEval = strcmp(ME.identifier, 'evalExpTens:badMethod');
end
results{end+1, 1} = 'method parity: evalExpTens rejects retired ''direct''';
results{end, 2}   = okRetiredEval;

%% ---- the three routes agree, in all four modes ----
% The centres route shares no reduction with the other two, so this is a
% stronger check than Bulger against Moebius alone. Relative-periodic is
% the documented exception: Bulger and centres compute the
% wrapped-difference kernel while Moebius computes the transposition
% average, and the two agree only as sigma/P tends to zero; sigma/P here
% is 0.025, well inside that regime.
modes = [false false; false true; true false; true true];
okAgree = true;  worstDev = 0;
for m = 1:size(modes, 1)
    isRel = modes(m, 1); isPer = modes(m, 2);
    A = buildExpTens(p, w, sigma, r, isRel, isPer, period, 'verbose', false);
    B = buildExpTens(q, wq, sigma, r, isRel, isPer, period, 'verbose', false);
    vals = zeros(1, 3); names = {'bulger', 'centres', 'mobius'};
    for i = 1:3
        vals(i) = cosSimExpTens(A, B, 'method', names{i}, ...
                                'truncationSigmas', Inf, 'verbose', false);
    end
    dev = max(abs(vals - vals(1))) / max([abs(vals(1)), 1e-300]);
    worstDev = max(worstDev, dev);
    okAgree = okAgree && (max(abs(vals - vals(1))) < 1e-12 || dev < 1e-9);
end
results{end+1, 1} = sprintf( ...
    'method parity: bulger/centres/mobius agree in all four modes (worst %.1e)', ...
    worstDev);
results{end, 2}   = okAgree;

%% ---- an explicit method must be the method that runs ----
% Inside the Moebius route a relative attribute's inner matrices may be
% computed either by the Moebius decomposition under a shift quadrature
% or by unrestricted enumeration over materialised tuple centres. The
% second is a different algorithm, and the cost gate used to substitute
% it under 'auto' even when the caller had asked for Moebius by name.
prev = mptDefaults('relAttrRoute');
okForced = true;
for isPerA = [false true]
    for routeC = {'auto', 'mobius'}
        mptDefaults('relAttrRoute', routeC{1});
        tf = mobius.maRelAttrPrefersCentres(p, q, 10, 2, true, isPerA, ...
                                            period, Inf, true);
        okForced = okForced && ~tf;
    end
    mptDefaults('relAttrRoute', 'centres');
    tfC = mobius.maRelAttrPrefersCentres(p, q, 10, 2, true, isPerA, ...
                                         period, Inf, true);
    okForced = okForced && tfC;
end
mptDefaults('relAttrRoute', prev);
results{end+1, 1} = 'method parity: method=''mobius'' pins the Moebius sub-route';
results{end, 2}   = okForced;

%% ---- relAttrRoute vocabulary: 'mobius' accepted, 'grid' aliased ----
prev = mptDefaults('relAttrRoute');
mptDefaults('relAttrRoute', 'mobius');
okName = strcmp(mptDefaults('relAttrRoute'), 'mobius');
mptDefaults('relAttrRoute', 'grid');
okName = okName && strcmp(mptDefaults('relAttrRoute'), 'mobius');
okBad = false;
try
    mptDefaults('relAttrRoute', 'bulger');
catch ME
    okBad = strcmp(ME.identifier, 'mptDefaults:badValue');
end
mptDefaults('relAttrRoute', prev);
results{end+1, 1} = 'method parity: relAttrRoute accepts ''mobius'', aliases ''grid''';
results{end, 2}   = okName && okBad;

%% ---- the sub-route is a cost choice, not a value choice ----
prev = mptDefaults('relAttrRoute');
okVal = true;
for isPerA = [false true]
    A2 = buildExpTens(p, w, 10, 3, true, isPerA, period, 'verbose', false);
    B2 = buildExpTens(q, wq, 10, 3, true, isPerA, period, 'verbose', false);
    mptDefaults('relAttrRoute', 'auto');
    ref = cosSimExpTens(A2, B2, 'method', 'bulger', ...
                        'truncationSigmas', Inf, 'verbose', false);
    for routeC = {'auto', 'centres', 'mobius'}
        mptDefaults('relAttrRoute', routeC{1});
        v = cosSimExpTens(A2, B2, 'method', 'mobius', ...
                          'truncationSigmas', Inf, 'verbose', false);
        okVal = okVal && (abs(v - ref) <= 1e-9 * max(abs(ref), 1));
    end
end
mptDefaults('relAttrRoute', prev);
results{end+1, 1} = 'method parity: relAttrRoute changes cost, not value';
results{end, 2}   = okVal;

%% ---- the spectral-branch lever changes cost, not value ----
% Both the spectral branch and the translation grid compute the
% full-image measure, so bypassing the branch's cost gate is a cost
% decision only. The gate is known to misroute outside the shapes it was
% calibrated on -- it models the grid route as costing K^2 per event
% pair, omitting the node count, which scales with span/sigma -- so
% benchmarks pin the branch rather than measure the routing.
prevSpec = internal.spectralIpForce();
Ks = 40; rs = 3; sig = 10;
rng(3, 'twister');
pS = sort(rand(Ks, 1) * 3 * period);  wS  = 0.4 + 0.6 * rand(Ks, 1);
qS = sort(rand(Ks, 1) * 3 * period);  wqS = 0.4 + 0.6 * rand(Ks, 1);
vals = zeros(1, 2); flags = [false true];
for iF = 1:2
    internal.spectralIpForce(flags(iF));
    AS = buildExpTens(pS, wS, sig, rs, true, false, period, 'verbose', false);
    BS = buildExpTens(qS, wqS, sig, rs, true, false, period, 'verbose', false);
    vals(iF) = cosSimExpTens(AS, BS, 'method', 'mobius', ...
                             'truncationSigmas', Inf, 'verbose', false);
end
internal.spectralIpForce(prevSpec);
results{end+1, 1} = 'method parity: spectral-branch lever changes cost, not value';
results{end, 2}   = abs(vals(1) - vals(2)) <= 1e-12 * max(abs(vals(1)), 1);
end

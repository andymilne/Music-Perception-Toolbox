%% diag_contract_disagreement.m
%
%  Are the contract/bulger disagreements numerical, or a defect?
%
%  bench_nested_dims.m flagged three configurations where the two routes
%  differed by more than 1e-9. That threshold was a blunt instrument: it
%  printed only a verdict, never a magnitude, and 1e-9 is tight for
%  configurations carrying thousands of leaf tuples. This script measures
%  the differences properly and separates the two explanations.
%
%  The decisive comparison is the pair of configurations with an identical
%  leaf-tuple count of 36:
%
%      K = 6, r = 1, L = 2   -- flagged as disagreeing
%      K = 3, r = 2, L = 2   -- agreed
%
%  Equal tuple counts mean equal opportunity for rounding to accumulate, so
%  if only the first disagrees the cause is not accumulation and inner
%  r = 1 is implicated.
%
%  Three tests per configuration:
%
%    1. The difference at several kernel truncation radii. Truncation error
%       falls as exp(-k^2/2), so a difference caused by truncation shrinks
%       sharply from k = 6 to the accuracy floor. One that does not move is
%       not a truncation artefact.
%
%    2. The three inner products separately. The cosine hides where a
%       divergence sits; xy, xx, and yy do not.
%
%    3. Self-similarity. cosSim(X, X) must be exactly 1 on either route.
%       This needs no cross-route comparison, so it tells us whether a
%       route is wrong on its own terms.
%
%  Run from the matlab directory:  clear all; rehash; diag_contract_disagreement
%
%  A diagnostic, not part of the toolbox. Delete when done.

clear functions %#ok<CLFUNC>

SIG    = 0.15;
PERIOD = 12.0;
NSUPER = 4;
TRUNCS = [4, 6, 8, Inf];    % Inf = the accuracy floor

% K, r, L, label
CASES = { ...
    6, 1, 2, 'FLAGGED   K=6 r=1 L=2  (36 tuples)'; ...
    3, 2, 2, 'control   K=3 r=2 L=2  (36 tuples, same count, agreed)'; ...
    4, 1, 2, 'control   K=4 r=1 L=2  (16 tuples, r=1, calibration case)'; ...
    5, 2, 2, 'control   K=5 r=2 L=2  (400 tuples, agreed)'; ...
    7, 2, 2, 'FLAGGED   K=7 r=2 L=2  (1764 tuples)'; ...
    8, 2, 2, 'FLAGGED   K=8 r=2 L=2  (3136 tuples)'; ...
    4, 2, 3, 'control   K=4 r=2 L=3  (1728 tuples, agreed)'};

fprintf('\n');
fprintf('Contract vs bulger: is the disagreement numerical or a defect?\n');
fprintf('sigma = %.3g, period = %.4g, super-events = %d\n\n', SIG, PERIOD, NSUPER);

% =====================================================================
% 1. Difference against truncation radius
% =====================================================================
fprintf('1. Cosine difference against kernel truncation radius\n');
fprintf('   Truncation error falls as exp(-k^2/2). A difference that does not\n');
fprintf('   shrink across these columns is not a truncation artefact.\n\n');

fprintf('   %-52s', 'configuration');
for t = TRUNCS
    if isinf(t); fprintf('%12s', 'floor'); else; fprintf('%12s', sprintf('k=%g', t)); end
end
fprintf('\n');

for c = 1:size(CASES, 1)
    K = CASES{c,1}; r = CASES{c,2}; L = CASES{c,3};
    fprintf('   %-52s', CASES{c,4});
    try
        [X, Y] = local_pair(NSUPER, L, K, r, SIG, PERIOD);
        for t = TRUNCS
            sC = cosSimExpTens(X, Y, 'method', 'contract', ...
                               'truncationSigmas', t, 'verbose', false);
            sB = cosSimExpTens(X, Y, 'method', 'bulger', ...
                               'truncationSigmas', t, 'verbose', false);
            fprintf('%12.2e', abs(sC - sB));
        end
    catch err
        fprintf('  failed: %s', err.message);
    end
    fprintf('\n');
end

% =====================================================================
% 2. Where the divergence sits
% =====================================================================
fprintf('\n2. Localising the divergence, at the accuracy floor\n');
fprintf('   There is no raw inner-product option, but the two normalisations\n');
fprintf('   give one: cosine divides by sqrt(<X,X>*<Y,Y>) and oneSidedDenom by\n');
fprintf('   <Y,Y>, so their quotient is sqrt(<X,X>/<Y,Y>). Comparing that\n');
fprintf('   between routes shows whether the self terms diverge relative to\n');
fprintf('   each other or whether only the cross term <X,Y> is at fault.\n');
fprintf('   The last column checks each route is symmetric in its arguments.\n\n');
fprintf('   %-52s %11s %11s %11s %11s\n', ...
        'configuration', 'd(cosine)', 'd(oneSide)', 'd(selfrat)', 'asym(C/B)');

for c = 1:size(CASES, 1)
    K = CASES{c,1}; r = CASES{c,2}; L = CASES{c,3};
    fprintf('   %-52s', CASES{c,4});
    try
        [X, Y] = local_pair(NSUPER, L, K, r, SIG, PERIOD);
        [cosC, osdC, symC] = local_readings(X, Y, 'contract');
        [cosB, osdB, symB] = local_readings(X, Y, 'bulger');
        ratC = osdC / max(abs(cosC), eps);
        ratB = osdB / max(abs(cosB), eps);
        fprintf('%11.2e%11.2e%11.2e%6.0e/%.0e', ...
                abs(cosC - cosB), abs(osdC - osdB), abs(ratC - ratB), symC, symB);
    catch err
        fprintf('  failed: %s', err.message);
    end
    fprintf('\n');
end

% =====================================================================
% 3. Self-similarity: each route judged on its own terms
% =====================================================================
fprintf('\n3. Self-similarity, cosSim(X, X), which must be exactly 1\n');
fprintf('   No cross-route comparison here, so a departure convicts one route\n');
fprintf('   on its own rather than merely showing the two differ.\n\n');
fprintf('   %-52s %16s %16s\n', 'configuration', '|contract - 1|', '|bulger - 1|');

for c = 1:size(CASES, 1)
    K = CASES{c,1}; r = CASES{c,2}; L = CASES{c,3};
    fprintf('   %-52s', CASES{c,4});
    try
        [X, ~] = local_pair(NSUPER, L, K, r, SIG, PERIOD);
        sC = cosSimExpTens(X, X, 'method', 'contract', ...
                           'truncationSigmas', Inf, 'verbose', false);
        sB = cosSimExpTens(X, X, 'method', 'bulger', ...
                           'truncationSigmas', Inf, 'verbose', false);
        fprintf('%16.2e%16.2e', abs(sC - 1), abs(sB - 1));
    catch err
        fprintf('  failed: %s', err.message);
    end
    fprintf('\n');
end

fprintf('\n');
fprintf('Reading the result. If the flagged rows shrink towards 1e-12 as the\n');
fprintf('truncation widens, and self-similarity holds on both routes, the\n');
fprintf('disagreements are ordinary accumulation and the 1e-9 threshold was\n');
fprintf('simply too tight for these tuple counts. If the K=6 r=1 row stays put\n');
fprintf('while its 36-tuple control stays clean, or if either route misses\n');
fprintf('self-similarity, the cause is structural and inner r=1 is the suspect.\n');
fprintf('\n');


% ---------------------------------------------------------------------
function [cosv, osd, asym] = local_readings(X, Y, method)
    % Both normalisations, plus a check that the route is symmetric in its
    % arguments. All at the accuracy floor so truncation plays no part.
    cosv = cosSimExpTens(X, Y, 'method', method, ...
                         'truncationSigmas', Inf, 'verbose', false);
    osd  = cosSimExpTens(X, Y, 'method', method, 'normalize', 'oneSidedDenom', ...
                         'truncationSigmas', Inf, 'verbose', false);
    rev  = cosSimExpTens(Y, X, 'method', method, ...
                         'truncationSigmas', Inf, 'verbose', false);
    asym = abs(cosv - rev);
end


% ---------------------------------------------------------------------
function [X, Y] = local_pair(nSuper, L, K, rIn, sigma, period)
    % Identical construction to bench_nested_dims.m, so the configurations
    % here are the same ones that were flagged there.
    N  = nSuper + L - 1;
    rs = RandStream('mt19937ar', 'Seed', 11);

    rootsA = mod(cumsum(randi(rs, [-4 4], 1, N)), period);
    rootsB = mod(rootsA + 1 + randi(rs, [0 2], 1, N), period);
    stack  = (0:K-1)' * (period / K);

    A = mod(repmat(rootsA, K, 1) + repmat(stack, 1, N), period);
    B = mod(repmat(rootsB, K, 1) + repmat(stack, 1, N), period);

    spA = flatSpecs({A}, 'r', rIn, 'rel', false, 'sym', true);
    spB = flatSpecs({B}, 'r', rIn, 'rel', false, 'sym', true);
    [pa, ~, spa] = bindEvents({A}, [], L, 'specs', spA, 'relOuter', true);
    [pb, ~, spb] = bindEvents({B}, [], L, 'specs', spB, 'relOuter', true);

    Pa = pa{1}; Pb = pb{1};
    X = buildExpTens({Pa}, {ones(size(Pa))}, 'specs', {spa{1}}, ...
                     'sigma', sigma, 'isPer', true, 'period', period, ...
                     'verbose', false);
    Y = buildExpTens({Pb}, {ones(size(Pb))}, 'specs', {spb{1}}, ...
                     'sigma', sigma, 'isPer', true, 'period', period, ...
                     'verbose', false);
end
